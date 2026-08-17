"""What the whole workload would cost if you actually followed the advice.

`adder savings` prices each lever on its own and then composes them
arithmetically. That answers "which lever is biggest". It does not answer the
question anyone actually has, which is "if I ran my work the way this tool
keeps telling me to, what would the bill be" -- and the two are not the same
number, because the levers interact through a context trajectory that a
per-lever estimate never simulates.

So this replays every recorded session turn by turn under a **regime**: a
concrete, followable operating configuration, priced end to end.

What this does that `simulate` does not
---------------------------------------
`adder simulate` exists to test whether the multiplicative composition
approximation holds, and for that purpose it prices only the context-read side.
It charges **nothing for the subagent**. That is fine for measuring a ratio and
badly wrong for quoting a total: a delegated read still has to be read, once, by
somebody, and that somebody still writes a summary. Here both sides are on the
books, along with the escalation term -- a cheap subagent that fails and gets
redone on Opus is not cheap.

Delegability is measured, not assumed
-------------------------------------
Every prior estimate in this repo used "assume 25% of turns are delegable",
which is a guess with a percent sign. It is also unfollowable: nobody can act on
"delegate a quarter of your turns". The regime here triggers on a quantity the
transcript records exactly -- how many tokens a turn admitted to context -- so
the rule is "delegate any step that would pull more than N tokens in", the
fraction that matches is a measurement, and the rule is one a person or a hook
can actually apply.

What it is still modelling, and therefore still assuming
--------------------------------------------------------
Three things, all of which move the answer and none of which the transcript can
settle:

  * that a delegated read compresses to `summary_ratio` without losing what the
    session needed from it,
  * that work is separable at a turn boundary, and that `handoff_tokens` is
    enough to re-establish the thread across the break,
  * that lower effort produces proportionally less output without failing more
    often, using the priors in `cost.EFFORT_OUTPUT_MULT`.

They are labelled MODELLED wherever they appear, and the residual line shows how
far the replay drifts from the measured total before any regime is applied. If
that residual is large, nothing below it is worth reading.

What changed when restarts stopped being free
---------------------------------------------
A split used to reset the context and charge nothing for it. That is a lever
with no cost term, and an optimiser handed one will push it to the end of its
range for free -- which is exactly what happened: the grid's answer to a 10x
target leaned on restarting every 50 turns, and the price of doing so appeared
nowhere in the total.

Restarting now costs what the transcripts say it costs. Every session records
what its own opening turn was billed, so a restart is charged that, plus the
handoff written in at the cache-write rate. `adder prefix` measures it: on this
machine an opening is 74% cache read, because the expensive part of the floor is
identical across sessions and still resident, so a restart is ~2.9x cheaper than
a rebuild and ~$0.10 rather than free.

Both corrections matter and they point in opposite directions. Charging for
restarts stops the optimiser from taking them for nothing; charging the
*measured* price rather than a rebuild is what makes a short cadence the right
answer instead of an expensive one.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path

from adder.core import settings as _settings
from adder.core.filters import root_of as _root_of
from adder.core.trace import DEFAULT_ROOT, load_sessions
from adder.decide.route.classify import Tier
from adder.measure.window.prefix import DEFAULT_HANDOFF, Opening
from adder.pricing.cost import EFFORT_OUTPUT_MULT, Rates, run_cost
from adder.pricing.registry import context_window, fits, rate

M = 1_000_000.0

# Tokens of task brief a delegated step has to be given. Small, but it is paid
# on every delegation and there are a lot of them.
BRIEF_TOKENS = 400

# Fraction of context growth that is assistant output rather than tool results
# or user input. Measured per-run by `debt.output_share_of_growth`; this is the
# fallback for callers that have no sessions to measure.
DEFAULT_OUTPUT_SHARE = 0.50


@dataclass(frozen=True)
class Regime:
    """A followable operating configuration, not a wish list.

    Every field is something a person or a hook can actually enforce. That
    constraint is why delegation is expressed as a token threshold rather than
    a fraction of turns: you cannot act on "delegate 25% of turns", but you can
    act on "if this would pull in more than 5,000 tokens, send it out".
    """

    label: str = "as run"
    delegate_above: int | None = None     # admit-size trigger, in tokens
    summary_ratio: float = 0.10           # what a delegated step hands back
    split_turns: int | None = None        # start a fresh session every N turns
    handoff_tokens: int = DEFAULT_HANDOFF  # what a restart has to be told. MODELLED.
    effort: str | None = None             # main-session effort, from "high"
    terseness: float = 0.0                # fraction of assistant output not written
    tool_discipline: float = 0.0          # fraction of tool output not admitted
    right_size: bool = True               # pick the subagent tier by expected cost
    # Used only when `right_size` is False. A factory rather than a literal so a
    # non-Claude workload gets its own configured cheap tier.
    sub_model: str = field(default_factory=_settings.sub_model)
    p_fail: float = 0.15                  # chance a delegated step has to be redone
    session_model: str | None = None      # start the session here, not on Opus
    session_rework: float = 0.20          # share of it redone on the original model

    @property
    def effort_mult(self) -> float:
        """Output volume relative to `high`, which is what the transcripts are."""
        if not self.effort:
            return 1.0
        return EFFORT_OUTPUT_MULT[self.effort] / EFFORT_OUTPUT_MULT["high"]


@dataclass
class Result:
    """Where the money went under one regime. Every field is a summed cost."""

    regime: Regime
    main_input: float = 0.0       # carrying the main context: reads, writes, misses
    main_out: float = 0.0         # the session's own output
    sub_run: float = 0.0          # delegated steps, priced on their own model
    sub_escalation: float = 0.0   # the modelled cost of redoing the ones that fail
    session_rework: float = 0.0   # the modelled cost of a cheaper session model failing
    restart: float = 0.0          # re-opening a session: the prefix read back, plus a handoff
    turns: int = 0
    restarts: int = 0
    delegated: int = 0
    delegated_tokens: int = 0
    admitted_tokens: int = 0
    reprised: int = 0             # turns a cheaper session model could have held
    by_tier: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return (self.main_input + self.main_out + self.sub_run
                + self.sub_escalation + self.session_rework + self.restart)

    @property
    def delegated_share(self) -> float:
        """Share of admitted *tokens* that were delegated, not share of turns.

        Turns are the wrong denominator. A session is thousands of small steps
        and a handful of large reads, and the large reads are the entire bill;
        a rule that catches 4% of turns can still catch half the tokens.
        """
        return self.delegated_tokens / self.admitted_tokens if self.admitted_tokens else 0.0


def _admissions(sess) -> tuple[int, int, list[int]]:
    """(starting context, restart floor, tokens admitted per turn).

    A compaction shows up as a context that went down. Clamping admission at
    zero treats that turn as admitting nothing, which understates admission
    rather than inventing a negative one -- but it also means a naive replay
    keeps accumulating through a drop the real session took, and then prices
    every later turn against a context the session never had. The caller has to
    apply the drop too; `replay` does, proportionally.
    """
    if not sess.turns:
        return 0, 0, []
    # Each chain differenced against ITSELF, and the result aligned to
    # `sess.turns` so a caller can index it by turn. A subagent runs in its own
    # window: the step down into one and the climb back out are not admissions
    # to the main context, and the climb back reads as a single enormous one
    # that this replay then prices on every later turn.
    # `carry._context_dynamics`, `agents.missed` and `anomaly.scan` all split
    # the chains for the same reason and say so.
    adm = [0] * len(sess.turns)
    prev: dict[bool, int | None] = {False: None, True: None}
    for i, t in enumerate(sess.turns):
        was = prev[t.sidechain]
        adm[i] = 0 if was is None else max(0, t.context - was)
        prev[t.sidechain] = t.context
    main = [t.context for t in sess.main_turns]
    return main[0], min(main), adm


@dataclass(frozen=True)
class Step:
    """One recorded turn, reduced to the numbers a replay needs.

    Built once and reused across every regime. `replay` is called hundreds of
    times by `solve`, and re-deriving a rate lookup and two cost calculations
    per turn per regime made the search take 37 seconds for an answer that is
    the same arithmetic either way.
    """

    read_cost: float      # of its input bill, the part driven by context carried
    write_cost: float     # of its input bill, the part driven by content admitted
    out_cost: float       # what its output actually cost
    real_ctx: int         # tokens it had to read
    adm: int              # tokens admitted since the previous turn
    inp: float            # $/MTok input, at this turn's model and speed
    out_rate: float       # $/MTok output, likewise
    model: str
    ttl: str = "5m"       # write multiplier this turn's cache writes were billed at
    # The day this turn was billed on. Every counterfactual in `replay` -- the
    # subagent's run, the escalation redo, the cheaper session model -- has to
    # be priced on the same day as the turn it is replacing, or the comparison
    # is between a recorded past and a hypothetical present. `ladder()` names
    # `claude-sonnet-5` as the cheap session model, and that is the one model
    # in the table with an introductory rate about to expire.
    on: date | None = None

    @property
    def in_cost(self) -> float:
        return self.read_cost + self.write_cost


def prepare(sessions, on: date | None = None) -> list[tuple[int, int, list[Step]]]:
    """Flatten sessions into (start context, restart floor, steps) per session."""
    out = []
    for sess in sessions.values():
        start_ctx, floor_ctx, adm = _admissions(sess)
        steps = []
        for i, t in enumerate(sess.turns):
            r = t.rates(on)
            # Split the input bill by what drives each half. Reads and uncached
            # input are proportional to the context being carried; writes are
            # proportional to what the turn admitted. A regime that shrinks the
            # context does not shrink the writes, and scaling both by context --
            # which this used to do -- makes every split-heavy regime look
            # cheaper than it is. Measured here, cache writes run at 2.1x
            # admitted tokens, so they track admission, not context size.
            w_cost = t.cache_write * t.rates(on).cache_write / M
            steps.append(Step(t.input_cost(on) - w_cost, w_cost, t.output_cost(on),
                              t.context, adm[i], r.inp, r.out, t.model, t.ttl,
                              t.pricing_date(on)))
        out.append((start_ctx, floor_ctx, steps))
    return out


def cheapest_tier(read_tokens: int, summary_tokens: int, *, p_fail: float,
                  overhead: float, on: date | None = None) -> Tier:
    """Lowest expected-cost tier that can hold a delegated read.

    The same arithmetic `policy.right_size` uses, minus the classifier: here the
    task is known to be a read of a known size, so feasibility and escalation
    risk are the only things left to decide it.
    """
    need = read_tokens + summary_tokens + BRIEF_TOKENS
    t2 = run_cost(Tier.T2.model, min(need, context_window(Tier.T2.model, need)),
                  summary_tokens, on)
    best, best_cost = Tier.T2, t2 + p_fail * (t2 + overhead)
    for tier in Tier:
        if tier >= Tier.T2 or not fits(tier.model, need):
            continue
        run = run_cost(tier.model, need, summary_tokens, on)
        cost = run + p_fail * (t2 + overhead)
        if cost < best_cost:
            best, best_cost = tier, cost
    return best


def replay(sessions, regime: Regime, *, output_share: float = DEFAULT_OUTPUT_SHARE,
           on: date | None = None) -> Result:
    """Re-price every recorded turn under `regime`.

    `sessions` may be a session dict or the output of `prepare()`; callers that
    replay repeatedly should prepare once and pass that.

    `session_model` is the one lever here that this repo has previously argued
    against, so it is worth being exact about what changed. The standing result
    is that downgrading a **warm** conversation loses money: the prompt cache is
    model-scoped, so the cheaper model re-reads the whole prefix uncached, and
    `adder savings` prices that swap at $22 out of $4,818.

    That argument is about a switch. It says nothing about the model a session
    *starts* on, because a session that begins on Sonnet never had an Opus
    prefix to invalidate -- there is no rebuild, only a cheaper rate applied to
    every turn from the first one. Nothing in the transcripts contradicts this;
    the question had simply never been asked, because the router was built to
    decide turns rather than sessions.

    What it costs instead is capability, and that is not free and not measurable
    from a transcript that only ever ran on one model. So it is priced as
    rework: `session_rework` of the session is redone at the original model's
    rate. That fraction is an assumption, it is the weakest number in this
    module, and the report prints it next to the row rather than burying it.
    """
    prepared = sessions if isinstance(sessions, list) else prepare(sessions, on)
    res = Result(regime=regime)
    cheap = regime.session_model
    keep_out = (1.0 - regime.terseness) * regime.effort_mult
    keep_adm = output_share * keep_out + (1.0 - output_share) * (1.0 - regime.tool_discipline)

    for start_ctx, floor_ctx, steps in prepared:
        ctx, real_prev = float(start_ctx), 0
        # What re-opening this session costs, taken from what opening it cost.
        # `steps[0].in_cost` is the recorded bill for the turn that established
        # this session's prefix -- mostly a cache read of a floor that other
        # sessions had already made resident. Assuming a rebuild instead
        # over-states it ~3x; assuming zero, which is what this replay used to
        # do, under-states it by all of it.
        reopen = steps[0].in_cost if steps else 0.0
        for i, st in enumerate(steps):
            res.turns += 1

            # The session compacted: it dropped a share of its context, so the
            # replay drops the same share. Without this the simulated context
            # runs away from the real one and every later turn is priced against
            # a conversation that never existed.
            if real_prev and st.real_ctx < real_prev:
                ctx *= st.real_ctx / real_prev
            real_prev = st.real_ctx

            if regime.split_turns and i and i % regime.split_turns == 0:
                # A fresh session starts at the opening context, not at the
                # session's smallest context: `floor_ctx` can sit below the
                # opening on a session that compacted, and a restart cannot
                # open smaller than an opening. It also has to be told what the
                # last one knew, which is written in at the cache-write rate.
                handoff = regime.handoff_tokens
                ctx = float(max(start_ctx, floor_ctx) + handoff)
                cost = reopen + handoff * Rates.for_model(
                    st.model, ttl=st.ttl, on=st.on).cache_write / M
                if (cheap and cheap != st.model and st.inp > 0
                        and fits(cheap, int(ctx))):
                    # `st.inp > 0`: the catalog carries free endpoints, and a
                    # recorded turn on one has an input rate of zero. Rescaling
                    # by a ratio of rates divides by it.
                    cost *= rate(cheap, st.on).inp / st.inp
                res.restart += cost
                res.restarts += 1

            raw = st.adm
            res.admitted_tokens += raw

            if regime.delegate_above is not None and raw >= regime.delegate_above:
                summary = max(1, int(raw * regime.summary_ratio))
                overhead = ctx * Rates.for_model(st.model, on=st.on).cache_read / M
                if regime.right_size:
                    tier = cheapest_tier(raw, summary, p_fail=regime.p_fail,
                                         overhead=overhead, on=st.on)
                    sub_model = tier.model
                    res.by_tier[tier.name] = res.by_tier.get(tier.name, 0) + 1
                else:
                    sub_model = regime.sub_model
                redo = run_cost(Tier.T2.model,
                                min(raw + BRIEF_TOKENS,
                                    context_window(Tier.T2.model, raw + BRIEF_TOKENS)),
                                summary, st.on)
                res.sub_run += run_cost(sub_model, raw + BRIEF_TOKENS, summary, st.on)
                res.sub_escalation += regime.p_fail * (redo + overhead)
                res.delegated += 1
                res.delegated_tokens += raw
                kept = float(summary)
                # The delegated step's own output happened elsewhere -- but it
                # still happened, and somebody was billed for it. Charging zero
                # here made delegation a free way to delete the session's
                # output, and an optimiser handed a free lever takes it: solving
                # the threshold down to ~300 tokens delegated 99% of admitted
                # tokens and dropped main-session output to 1% of the total,
                # which is not a regime anyone could run. It is a rate
                # substitution, not a deletion: the same work, produced on the
                # subagent's model.
                out_mult = rate(sub_model, st.on).out / st.out_rate if st.out_rate else 0.0
            else:
                # Terseness and effort both attack assistant output and compose
                # on it; tool discipline attacks the other half. Neither reaches
                # the other's half, which is the whole reason the split is
                # measured rather than assumed.
                kept = raw * keep_adm
                out_mult = keep_out

            # Admit before pricing. `adm[i]` is the growth that happened
            # *between* the previous turn and this one, so it is already in the
            # context this turn has to read; charging it from the next turn
            # onward under-prices every turn by one admission.
            ctx += kept

            # Price the input side off what this turn ACTUALLY cost, scaling
            # each half by the thing that drives it.
            #
            # Rebuilding the input side from first principles -- context at the
            # cache-read rate plus admissions at the write rate -- missed the
            # measured total by 22%, because a real turn's uncached / read /
            # written split is not recoverable from the context size alone. It
            # does not have to be, as long as the recorded number is scaled by
            # the right quantity: reads and uncached input by how much context
            # the regime left this turn carrying, writes by how much of the
            # admission it kept.
            #
            # Scaling both by context was wrong in a way that only showed up at
            # the extremes, which is where the recommendations live. Splitting
            # every 19 turns cuts carried context ~12x; it does not cut what the
            # session writes at all, because the same content is still admitted,
            # just into a shorter-lived context. Charging writes as if they had
            # fallen 12x too made restarts look free a second time over.
            scale = (ctx / st.real_ctx) if st.real_ctx else 1.0
            kept_frac = (kept / raw) if raw else 1.0
            in_cost = st.read_cost * scale + st.write_cost * kept_frac
            out_cost = st.out_cost * out_mult

            # Running the session on a cheaper model is a pure rate substitution
            # on the recorded accounting -- no cache rebuild, because there was
            # never a prefix on the expensive model to lose. It only applies to
            # turns the cheaper model could actually have held.
            if (cheap and cheap != st.model and st.inp > 0 and st.out_rate > 0
                    and fits(cheap, int(max(ctx, st.real_ctx)))):
                # Both rates strictly positive: the substitution below is a
                # ratio, and a turn recorded on a free endpoint (the catalog
                # carries sixteen) has a rate of zero on one or both sides.
                cr = rate(cheap, st.on)
                res.main_input += in_cost * (cr.inp / st.inp)
                res.main_out += out_cost * (cr.out / st.out_rate)
                res.session_rework += regime.session_rework * (in_cost + out_cost)
                res.reprised += 1
            else:
                res.main_input += in_cost
                res.main_out += out_cost
    return res


# The regimes the report walks through, each one adding a single lever to the
# one above it. Cumulative on purpose: the levers are substitutes, so quoting
# them independently and adding them up double-counts the same pool.
def ladder(delegate_above: int = 5_000, split_turns: int = 300,
           effort: str = "medium", session_model: str = "claude-sonnet-5",
           session_rework: float = 0.20,
           handoff_tokens: int = DEFAULT_HANDOFF) -> list[Regime]:
    steps = [Regime()]
    # Placement first, price second, and deliberately in that order: the first
    # row delegates to the model the session was already on, so it measures the
    # value of moving the read out of context with the model held fixed. Only
    # then does right-sizing get to claim what choosing the tier is worth.
    steps.append(replace(steps[-1], label=f"delegate reads over {delegate_above:,} tok",
                         delegate_above=delegate_above, right_size=False,
                         sub_model="claude-opus-5"))
    steps.append(replace(steps[-1], label="+ right-size the subagent",
                         right_size=True))
    steps.append(replace(steps[-1], label=f"+ split sessions at {split_turns} turns",
                         split_turns=split_turns, handoff_tokens=handoff_tokens))
    steps.append(replace(steps[-1], label=f"+ effort high -> {effort}", effort=effort))
    steps.append(replace(steps[-1], label="+ 30% terser, 40% less tool output",
                         terseness=0.30, tool_discipline=0.40))
    steps.append(replace(steps[-1],
                         label=f"+ start sessions on {session_model}",
                         session_model=session_model, session_rework=session_rework))
    return steps


def frontier(delegate_above: int = 1_000, split_turns: int = 15) -> Regime:
    """Every lever pushed to the end of its range, as the family's floor.

    Not a recommendation -- splitting every 15 turns and cutting output by half
    is a different way of working, and the assumptions behind it are exactly the
    ones the module docstring flags as unsettled. It is here to bound the claim:
    if a target does not fit under this, no combination of the levers in this
    file reaches it, and saying so is more useful than an optimiser that keeps
    searching.
    """
    return Regime(
        label="everything, at the end of its range",
        delegate_above=delegate_above, split_turns=split_turns, effort="low",
        terseness=0.50, tool_discipline=0.60, right_size=True,
        session_model="claude-sonnet-5",
    )


def dominant_model(sessions) -> str:
    """The model the spend is actually on. Cost-weighted, not turn-weighted."""
    spend: dict[str, float] = {}
    for s in sessions.values():
        for t in s.turns:
            spend[t.model] = spend.get(t.model, 0.0) + t.cost()
    return max(spend, key=spend.get) if spend else "claude-opus-5"


def recommended_cadence(sessions, *, handoff_tokens: int = DEFAULT_HANDOFF,
                        model: str | None = None,
                        on: date | None = None) -> tuple[int, Opening, str]:
    """How often to restart, solved from the measurement instead of picked by hand.

    The old default was 300 turns, which was a round number chosen to be
    defensible rather than derived. With the restart price measured rather than
    assumed, there is a closed form for it -- `k* = sqrt(2W/(m*r*g))` -- and it
    can be evaluated against this workload's own growth rate, re-read multiplier
    and opening cost. On this machine it answers 19 turns, not 300, and the
    difference is almost entirely the cache: a restart re-reads the shared floor
    instead of rebuilding it.

    Returns the cadence, the opening it was priced from, and a sentence saying
    where the number came from -- including when it came from a prior, because
    a machine with no transcripts gets the pessimistic cold-rebuild answer and
    should be told so.
    """
    from adder.measure.window.carry import Carry
    from adder.measure.window.prefix import cadence, measure, weighted_median_turns

    op = measure(sessions)
    c = Carry.measure(sessions)
    model = model or dominant_model(sessions)
    k, _at_k, _never = cadence(op, model=model, growth=max(1.0, c.growth),
                             read_mult=c.read_mult, handoff_tokens=handoff_tokens,
                             observed_turns=weighted_median_turns(sessions), on=on)
    basis = "measured" if op.measured else "prior (no openings to measure)"
    why = (f"{k:,} turns: k* = sqrt(2W/(m*r*g)) at a ${op.cost(model, handoff_tokens=handoff_tokens, on=on):,.4f} "
           f"restart [{basis}], {c.growth:,.0f} tok/turn of growth and a "
           f"{c.read_mult:.3f}x re-read multiplier")
    return k, op, why


def recommended_threshold(sessions, *, split_turns: int | None = None,
                          sub_model: str | None = None, p_fail: float = 0.15,
                          model: str | None = None, ttl: str = "1h",
                          on: date | None = None) -> tuple[int | None, str]:
    """The admit size above which delegating pays, solved rather than picked.

    5,000 tokens was a round number, and it is the wrong one in a way that only
    shows up once the cadence is solved. Both sides of the delegation decision
    are affine in the read size, so `carry.delegate_threshold` gets the
    break-even in one division -- and it depends on how many turns will re-read
    the content, which is exactly what the restart cadence sets. A 19-turn
    cycle leaves ~9 re-reads to avoid, not ~500.

    That cuts both ways and the direction is not the obvious one. A shorter
    horizon raises the threshold, because there is less carry to save; but even
    with *no* re-reads at all the threshold stays small, because admitting a
    token to an Opus context costs 2.00x its input rate as a cache write while
    reading it once on Haiku costs 1.00x of a rate five times lower. Delegation
    is not only a carry play. Measured here it lands near 800 tokens, and the
    hand-picked 5,000 was leaving most of the lever unused.

    Returns None when delegation never pays, which is a real answer and better
    than a threshold nobody will reach.
    """
    from adder.measure.window.carry import Carry, delegate_threshold
    from adder.measure.window.prefix import measure, weighted_median_turns

    c = Carry.measure(sessions)
    op = measure(sessions)
    model = model or dominant_model(sessions)
    turns = split_turns or weighted_median_turns(sessions) or 1
    # A step lands at a random point in the cycle, so it has half of one left.
    horizon = max(1, turns // 2)
    # If a delegated step fails, the main session re-reads its context to redo
    # the work. Priced off the floor, which is the part a restart cannot avoid.
    overhead = op.floor_tokens * c.read_mult * rate(model, on).inp / M
    x, why = delegate_threshold(main_model=model,
                                sub_model=sub_model or _settings.sub_model(),
                                remaining_turns=horizon, carry=c, p_redo=p_fail,
                                redo_overhead=overhead, ttl=ttl, on=on)
    if x == float("inf"):
        return None, why
    # Rounded, because this is a rule a person or a hook applies, and nobody
    # applies "delegate anything over 799 tokens".
    return max(100, round(x / 100.0) * 100), why


def report(root: Path | str = DEFAULT_ROOT, *, target: float = 10.0,
           delegate_above: int | None = None, split_turns: int | None = None,
           effort: str = "medium", session_model: str = "claude-sonnet-5",
           session_rework: float = 0.20, handoff_tokens: int = DEFAULT_HANDOFF,
           on: date | None = None) -> int:
    from adder.measure.spend.debt import output_share_of_growth

    sessions = load_sessions(root, use_cache=True)
    measured = sum(s.cost_on(on) for s in sessions.values())
    if not measured:
        print(f"\n  No priced turns found under {root}\n")
        return 1

    share = output_share_of_growth(sessions)
    prepared = prepare(sessions, on)
    # The cadence is solved from this workload unless the caller names one.
    cadence_note = ""
    if split_turns is None:
        split_turns, opening, cadence_note = recommended_cadence(
            sessions, handoff_tokens=handoff_tokens, on=on)
    threshold_note = ""
    if delegate_above is None:
        delegate_above, threshold_note = recommended_threshold(
            sessions, split_turns=split_turns, on=on)
    steps = ladder(delegate_above, split_turns, effort, session_model,
                   session_rework, handoff_tokens)
    results = [replay(prepared, r, output_share=share, on=on) for r in steps]
    base = results[0]

    residual = (base.total - measured) / measured if measured else 0.0
    print(f"\n  Measured spend            ${measured:>10,.0f}   "
          f"{base.turns:,} turns, {len(sessions)} sessions")
    print(f"  Replay of the same turns  ${base.total:>10,.0f}   "
          f"residual {residual:+.1%} -- everything below is relative to this")
    if abs(residual) > 0.25:
        print("  ! the replay does not reproduce the measured total closely enough;")
        print("    treat the multiples below as illustrative, not as a quote")
    print(f"  Context growth is ~{share:.0%} assistant output, "
          f"~{1 - share:.0%} tool output and user input.")
    if cadence_note:
        print(f"  Restart cadence, solved rather than assumed: {cadence_note}.")
        print(f"  A restart is charged what an opening actually costs -- "
              f"{opening.warm_share:.0%} of it is a cache read.")
    if threshold_note:
        print(f"  Delegation threshold, likewise: {threshold_note}.")
    print()

    print("  Each row adds ONE lever to the row above it. They are substitutes:")
    print("  quoting them separately and adding up double-counts the same pool.\n")
    print(f"  {'regime':<42}{'total':>11}{'vs baseline':>13}{'tok deleg.':>12}")
    print("  " + "-" * 77)
    for res in results:
        mult = base.total / res.total if res.total else float("inf")
        dele = f"{res.delegated_share:.0%}" if res.regime.delegate_above else "-"
        print(f"  {res.regime.label:<42}${res.total:>10,.0f}{mult:>12.1f}x{dele:>12}")

    best = results[-1]
    if best.delegated_share > 0.75:
        print()
        print(f"  {best.delegated_share:.0%} of admitted tokens are delegated at this "
              f"threshold, which is not a")
        print("  tweak to how you work -- it is the orchestrator pattern: the main "
              "session holds")
        print("  the thread and almost every step that would admit content runs "
              "somewhere else.")
        print(f"  MODELLED: that each delegated step hands back "
              f"{best.regime.summary_ratio:.0%} of what it read without")
        print("  losing what the session needed from it. At this share that "
              "assumption carries")
        print("  most of the multiple, and no transcript can settle it. "
              "`adder ab` is the test.")

    if best.restarts:
        print()
        print(f"  MODELLED, and now the softest input here: the {handoff_tokens:,}-token "
              f"handoff. Restarting")
        print(f"  {best.restarts:,} times only works if {handoff_tokens:,} tokens is "
              f"enough to carry the thread. Nothing in a")
        print("  transcript records what a person needs to resume, so the multiple "
              "is quoted against")
        print("  a sweep of it rather than on its own:")
        for h in (handoff_tokens, 10_000, 20_000, 50_000):
            if h < handoff_tokens:
                continue
            k, _op, _why = recommended_cadence(sessions, handoff_tokens=h, on=on)
            alt = replay(prepared, replace(best.regime, split_turns=k,
                                           handoff_tokens=h),
                         output_share=share, on=on)
            mult = base.total / alt.total if alt.total else float("inf")
            print(f"    {h:>7,}-token handoff  restart every {k:>3,} turns   "
                  f"${alt.total:>7,.0f}{mult:>7.1f}x")

    print()
    print("  Where the remaining money is:")
    for name, amount, note in (
        ("carrying the main context", best.main_input,
         "what is left of the pool after splitting and delegation"),
        ("main-session output", best.main_out, "generation, at the session model's rate"),
        ("delegated steps", best.sub_run, "read once, on the tier that fits"),
        ("redoing the ones that fail", best.sub_escalation,
         f"at p_fail {best.regime.p_fail:.0%}"),
        ("rework on the session model", best.session_rework,
         f"at {best.regime.session_rework:.0%}, an assumption not a measurement"),
        ("restarting sessions", best.restart,
         f"{best.restarts:,} restarts, priced off what openings cost"),
    ):
        pct = 100 * amount / best.total if best.total else 0.0
        print(f"    ${amount:>9,.0f}{pct:>7.1f}%  {name:<32}{note}")
    if best.regime.session_model:
        print(f"\n  {best.reprised:,} of {best.turns:,} turns would have fit in "
              f"{best.regime.session_model}; the rest stayed where they were.")
        print(f"  MODELLED: {best.regime.session_rework:.0%} of the session is "
              f"redone at the original model's rate. Nothing in a transcript that")
        print("  only ever ran on one model can settle that number. Vary it with "
              "--session-rework.")
    if best.by_tier:
        mix = "  ".join(f"{k} {v:,}" for k, v in sorted(best.by_tier.items()))
        print(f"\n  Subagent tier mix, chosen by expected cost: {mix}")

    _target_verdict(prepared, base, best, share, target, session_rework, on)
    return 0


# The search grid, and how disruptive each setting is to live with.
#
# A cheaper regime is not automatically a better one: restarting every 50 turns
# and halving output is a different job, not a discount. So the search does not
# minimise cost -- it finds the *mildest* configuration that reaches the target,
# scored on a scale that is a stated judgement rather than a measurement. Someone
# who disagrees with the ordering can read it here and change it.
GRID: dict[str, list[tuple[object, int]]] = {
    "delegate_above": [(None, 0), (10_000, 1), (5_000, 2), (2_000, 3), (1_000, 4)],
    # 25 and 15 are here because restarting is now priced. While a split was
    # free the grid had no business offering cadences a person would feel; now
    # that each one costs a measured ~$0.10, the optimiser has to earn them.
    "split_turns": [(None, 0), (300, 1), (200, 2), (150, 3), (100, 4), (50, 6),
                    (25, 8), (15, 10)],
    "effort": [(None, 0), ("medium", 1), ("low", 3)],
    "writing": [((0.0, 0.0), 0), ((0.20, 0.30), 1), ((0.30, 0.40), 2), ((0.50, 0.60), 4)],
    "session_model": [(None, 0), ("claude-sonnet-5", 2)],
}


def _candidates() -> list[tuple[int, dict]]:
    """Every grid point, mildest first. Deterministic order, no wall clock."""
    out = []
    for d, sd in GRID["delegate_above"]:
        for sp, ss in GRID["split_turns"]:
            for ef, se in GRID["effort"]:
                for (te, td), sw in GRID["writing"]:
                    for sm, sms in GRID["session_model"]:
                        out.append((sd + ss + se + sw + sms, {
                            "delegate_above": d, "split_turns": sp, "effort": ef,
                            "terseness": te, "tool_discipline": td,
                            "session_model": sm,
                        }))
    out.sort(key=lambda row: row[0])
    return out


def solve(sessions, *, target: float, baseline: float, output_share: float,
          session_rework: float = 0.20, on: date | None = None
          ) -> tuple[Regime | None, Result | None, int]:
    """Mildest regime on the grid that reaches `target`. None if nothing does.

    Exhaustive within a severity level, and ascending across them. The levers
    interact through one shared pool, so a greedy walk down the biggest-lever-
    first path lands on a configuration that is cheap and needlessly harsh; but
    once *some* configuration at severity k works, nothing at severity k+1 can
    be a better answer, so the search stops there.

    That early exit is for clarity more than speed -- most of the grid sits at
    or below the severity that usually wins, so it skips less than a third of
    the work. What made the search usable was `prepare()`: precomputing each
    turn's recorded cost once instead of per regime took it from 37 seconds to
    10 on 20,000 turns.
    """
    if target <= 0:
        raise ValueError(
            f"target must be a positive multiple of today's spend, got {target!r}; "
            "`target=2` asks how to halve the bill")
    need = baseline / target
    best: tuple[Regime, Result, int] | None = None
    current = None
    for severity, params in _candidates():
        if best is not None and severity > current:
            break                      # a harsher level cannot improve on a hit
        current = severity
        r = Regime(label="solved", session_rework=session_rework, **params)
        res = replay(sessions, r, output_share=output_share, on=on)
        if res.total > need:
            continue
        if best is None or res.total < best[1].total:
            best = (r, res, severity)
    return best if best is not None else (None, None, 0)


def _multiple(text: str) -> float:
    """An argparse type for a reduction factor. It is a divisor, so never zero."""
    import argparse

    try:
        x = float(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from e
    if x <= 0:
        raise argparse.ArgumentTypeError(
            f"--target is a multiple of today's spend and must be above zero, "
            f"got {x:g}; `--target 2` asks how to halve the bill")
    return x


def _target_verdict(sessions, base: Result, best: Result, share: float,
                    target: float, session_rework: float, on: date | None) -> None:
    """Say plainly whether the target is reachable, and stop short of pretending."""
    achieved = base.total / best.total if best.total else float("inf")
    need = base.total / target
    print(f"\n  Target {target:.0f}x means getting ${base.total:,.0f} down to "
          f"${need:,.0f}.")
    if achieved >= target:
        print(f"  The regime above reaches {achieved:.1f}x. Target met, on these "
              f"assumptions;")
        print("  run `adder quality` before and after, because none of this is free.")
        return

    print(f"  The regime above reaches {achieved:.1f}x, short by "
          f"${best.total - need:,.0f}.")
    print("  Searching the grid for the mildest configuration that does...",
          flush=True)
    reg, res, severity = solve(sessions, target=target, baseline=base.total,
                               output_share=share, session_rework=session_rework, on=on)
    if reg is None:
        edge = replay(sessions, frontier(), output_share=share, on=on)
        edge_mult = base.total / edge.total if edge.total else float("inf")
        floor = edge.main_out + edge.sub_run
        print(f"  Nothing on the grid reaches it. The hardest setting of every "
              f"lever reaches {edge_mult:.1f}x")
        print(f"  (${edge.total:,.0f}), of which ${floor:,.0f} is work that has to "
              f"happen somewhere at somebody's")
        print("  rate. Getting past that needs a different lever, not a harder "
              "setting on these ones.")
        return

    print(f"  The mildest configuration on the grid that does reach it "
          f"(${res.total:,.0f}, {base.total / res.total:.1f}x):")
    print(f"    {_describe(reg)}")
    print(f"  Severity {severity} on the scale in `plan.GRID` -- lower is less "
          f"change to how you work.")
    print("  Every one of those is an assumption about work you have not done yet. "
          "Run it,")
    print("  then `adder verify --since <date>` to find out which of them held.")


def _describe(r: Regime) -> str:
    bits = []
    if r.session_model:
        bits.append(f"start sessions on {r.session_model}")
    if r.delegate_above:
        bits.append(f"delegate anything over {r.delegate_above:,} tok")
    if r.split_turns:
        bits.append(f"restart every {r.split_turns} turns")
    if r.effort:
        bits.append(f"effort {r.effort}")
    if r.terseness:
        bits.append(f"{r.terseness:.0%} terser")
    if r.tool_discipline:
        bits.append(f"{r.tool_discipline:.0%} less tool output")
    return ", ".join(bits)


def _json_report(a) -> int:
    """The same regimes `report` prints, as data.

    Shares the ladder and the solver rather than reconstructing them: a JSON
    view that recomputes the regime grid by hand is a second answer waiting to
    disagree with the first.
    """
    import json

    from adder.measure.spend.debt import output_share_of_growth

    sessions = load_sessions(a.root, use_cache=True)
    measured = sum(s.cost_on() for s in sessions.values())
    if not measured:
        print(json.dumps({"error": "no priced turns", "root": a.root}))
        return 1
    share = output_share_of_growth(sessions)

    prepared = prepare(sessions)
    split_turns, cadence_note = a.split_turns, ""
    if split_turns is None:
        split_turns, _opening, cadence_note = recommended_cadence(
            sessions, handoff_tokens=a.handoff)
    delegate_above, threshold_note = a.delegate_above, ""
    if delegate_above is None:
        delegate_above, threshold_note = recommended_threshold(
            sessions, split_turns=split_turns)

    steps = ladder(delegate_above, split_turns, a.effort, a.session_model,
                   a.session_rework, a.handoff)
    results = [replay(prepared, r, output_share=share) for r in steps]
    base, result = results[0], results[-1]
    solved, solved_res, severity = solve(
        prepared, target=a.target, baseline=base.total, output_share=share,
        session_rework=a.session_rework)

    def block(r):
        return {
            "total": round(r.total, 4),
            "main_input": round(r.main_input, 4),
            "main_out": round(r.main_out, 4),
            "sub_run": round(r.sub_run, 4),
            "sub_escalation": round(r.sub_escalation, 4),
            "session_rework": round(r.session_rework, 4),
            "restart": round(r.restart, 4),
            "turns": r.turns, "restarts": r.restarts,
            "delegated_turns": r.delegated,
            "delegated_share_of_tokens": round(r.delegated_share, 5),
            "by_tier": r.by_tier,
        }

    print(json.dumps({
        "measured": round(measured, 4),
        "replay_residual": (round((base.total - measured) / measured, 5)
                            if measured else None),
        "output_share_of_growth": round(share, 5),
        "target": a.target,
        "delegate_above": delegate_above,
        "delegate_above_reason": threshold_note,
        "split_turns": split_turns,
        "split_turns_reason": cadence_note,
        "baseline": block(base),
        "ladder": [
            {"label": r.regime.label, **block(r),
             "vs_baseline": round(base.total / r.total, 3) if r.total else None}
            for r in results
        ],
        "regime": {**block(result), "describe": _describe(result.regime)},
        "reduction": round(base.total / result.total, 3) if result.total else None,
        "solved": (None if solved is None else
                   {**block(solved_res), "describe": _describe(solved),
                    "severity": severity}),
        "solved_reduction": (None if solved_res is None or not solved_res.total
                             else round(base.total / solved_res.total, 3)),
    }))
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="adder plan",
        description="price a whole workload under a followable operating regime")
    ap.add_argument("root", nargs="?", default=None,
                    help="transcript directory (default: the `root` setting)")
    ap.add_argument("--target", type=_multiple, default=10.0,
                    help="cost reduction to test for (default: 10x)")
    ap.add_argument("--delegate-above", type=int, default=None, metavar="TOK",
                    help="admit-size that triggers delegation (default: solved "
                         "from the carry model at the chosen cadence)")
    ap.add_argument("--split-turns", type=int, default=None, metavar="N",
                    help="restart every N turns (default: solved from the "
                         "measured restart cost)")
    ap.add_argument("--handoff", type=int, default=DEFAULT_HANDOFF, metavar="TOK",
                    help=f"tokens a restarted session has to be told "
                         f"(default: {DEFAULT_HANDOFF})")
    ap.add_argument("--effort", default="medium", choices=sorted(EFFORT_OUTPUT_MULT))
    ap.add_argument("--session-model", default="claude-sonnet-5",
                    help="model to start sessions on (default: claude-sonnet-5)")
    ap.add_argument("--session-rework", type=float, default=0.20, metavar="F",
                    help="modelled share of the session redone on the original "
                         "model (default: 0.20)")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    a = ap.parse_args(argv)
    # `root_of`: the argument if one was given, else the `root`
    # setting. Resolved here so two commands cannot disagree
    # about which transcript directory `adder config` names.
    a.root = str(_root_of(a))

    if a.json:
        return _json_report(a)

    rc = report(a.root, target=a.target, delegate_above=a.delegate_above,
                split_turns=a.split_turns, effort=a.effort,
                handoff_tokens=a.handoff,
                session_model=a.session_model, session_rework=a.session_rework)
    print()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
