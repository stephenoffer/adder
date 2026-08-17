"""Cache- and context-aware cost model.

The term every other LLM router omits
-------------------------------------
Routers built for stateless APIs price a request as `in*rate_in + out*rate_out`.
In a persistent agent session that is wrong, because the conversation prefix is
re-sent on every turn. A token admitted to the main context is billed once as a
cache write and then again, as a cache read, on **every remaining turn**:

    admitted_token_cost(n, model, remaining_turns)
      = n * rate_in * write_mult                    # once
      + n * rate_in * 0.10 * remaining_turns        # forever after

On measured data (median session: 607 turns, 544K context) this term is ~76% of
total spend. It dominates model selection by roughly an order of magnitude, and
it is why downgrading a *warm* conversation to a cheaper model loses money: the
cheap model must re-read the whole context uncached.

Four failure modes this module gates on, which a price-only model cannot see
-----------------------------------------------------------------------------
* **Feasibility.** Haiku holds 200K; the median context here is 544K. A
  "saving" that names a model the context does not fit in is a 400 error.
* **Cache minimums.** A prefix under the model's minimum silently does not
  cache, so a delegation whose brief is below it pays full input rate twice.
* **TTL.** A 5m cache that expires between turns is re-written at 1.25x rather
  than read at 0.10x -- a 12.5x swing. Idle gaps, not token counts, decide it.
* **Fan-out.** N parallel calls sharing a prefix all miss: a cache entry is only
  readable once the first response starts streaming. Serialising one call first
  turns N cache writes into 1 write + (N-1) reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from adder.pricing.prices import BATCH_MULT
from adder.pricing.registry import (
    UnknownModelError,
    UnpricedModelError,
    cache_min,
    caches,
    context_limit,
    fits,
    resolve,
)

M = 1_000_000.0

# Last-resort subagent model for `placement_cost`, used only when the caller
# names none.
#
# It is a constant here rather than a config lookup because this layer may not
# import the settings module -- pricing sits below `core`, and the layering
# test enforces it. That is the right split: "what does a token cost" is a
# pricing question, "which cheap model do we actually have" is a policy one.
# Every caller above this layer resolves the configured tier and passes it, so
# a Codex or Gemini CLI workload never reaches this fallback; it exists so the
# arithmetic is still well-defined for a caller that has no policy to consult.
DEFAULT_SUB_MODEL = "claude-haiku-4-5"

# Relative output volume by effort level, indexed to `high` = 1.0.
#
# These are PRIORS, not measurements: effort controls thinking depth, and the
# mapping from level to token volume is workload-specific. Anything using them
# is labelled MODELLED, and `adder.measure.session.effort` re-fits them from a transcript when
# there is enough per-effort history to do so.
EFFORT_OUTPUT_MULT = {"low": 0.35, "medium": 0.60, "high": 1.00,
                      "xhigh": 1.50, "max": 2.20}


@dataclass(frozen=True)
class Admitted:
    """The two halves of what a token costs once it is in a context."""

    write: float      # paid once, when it enters
    reads: float      # paid again on every turn that re-reads it

    @property
    def total(self) -> float:
        return self.write + self.reads


@dataclass(frozen=True)
class Rates:
    """USD per million tokens for one model, however they were obtained.

    This module prices Claude from the hand-maintained table, where cache rates
    are fixed multiples of the input rate. `select.py` prices ~500 models from
    the catalog, where providers publish cache rates absolutely and the
    multiples do not hold. Both need the same arithmetic, and for a while both
    carried their own copy of it -- which is exactly how they came to disagree
    about whether the carry term could be corrected at all. One expression,
    two ways of filling in the rates.
    """

    inp: float
    out: float
    cache_read: float
    cache_write: float

    # USD per million tokens per hour of explicit-cache storage. Google is the
    # only major provider that bills it; everyone else leaves it at zero.
    storage: float = 0.0

    @staticmethod
    def for_model(model: str, *, ttl: str | None = None, on: date | None = None,
                  speed: str = "standard") -> Rates:
        """Rates for any model, from any provider, with its own cache economics.

        This is the universal constructor. It replaces four different places
        that each rebuilt Anthropic's 0.10x/1.25x multipliers over whatever
        model id they were handed -- which is correct for Claude and wrong for
        everyone else, in both directions. A provider with no cache gets its
        re-reads priced at full input rate; a provider with automatic caching
        gets no write premium, because it has no write decision.
        """
        spec = resolve(model)
        r = spec.rate(on, speed=speed)
        return Rates(
            inp=r.inp, out=r.out,
            cache_read=spec.cache_read_rate(on, speed=speed),
            cache_write=spec.cache_write_rate(ttl, on, speed=speed),
            storage=spec.cache_storage_abs or 0.0,
        )

    @staticmethod
    def claude(model: str, *, ttl: str = "5m", on: date | None = None,
               speed: str = "standard") -> Rates:
        """Deprecated alias for `for_model`, kept for callers that predate it.

        It is not a separate code path any more. Keeping the name means the
        Claude-only call sites still read honestly about what they are pricing,
        but they no longer get a private copy of the arithmetic.
        """
        return Rates.for_model(model, ttl=ttl, on=on, speed=speed)


def admitted_cost(n_tokens: int, rates: Rates, *, reads: float) -> Admitted:
    """One cache write now, then `reads` cache reads over the rest of the session.

    `reads` is a float, not a turn count, because it is not always the turn
    count: a fitted carry model discounts it by the token's chance of surviving
    each compaction. Callers that have no such model pass `remaining_turns`.
    """
    write = n_tokens * rates.cache_write / M
    read = n_tokens * rates.cache_read * max(0.0, reads) / M
    return Admitted(write=write, reads=read)


def turn_cost(
    model: str,
    *,
    uncached_in: int = 0,
    cache_read: int = 0,
    cache_write: int = 0,
    out: int = 0,
    ttl: str = "5m",
    speed: str = "standard",
    batch: bool = False,
    on: date | None = None,
) -> float:
    """USD for a single turn, given its token accounting.

    Prices through the registry, so a turn recorded against any provider is
    billed with that provider's cache economics rather than Anthropic's. The
    two that differ most: an automatic-caching provider has no write premium,
    and a provider with no cache at all bills `cache_read` tokens at full
    input rate -- which is what actually happens, since there was no cache to
    read them from.
    """
    r = Rates.for_model(model, ttl=ttl, on=on, speed=speed)
    total = (
        uncached_in * r.inp
        + cache_read * r.cache_read
        + cache_write * r.cache_write
        + out * r.out
    ) / M
    bm = batch_mult(model) if batch else 1.0
    return total * bm


def admitted_token_cost(
    n_tokens: int,
    model: str,
    remaining_turns: int,
    *,
    ttl: str = "5m",
    on: date | None = None,
    carry=None,
    context_tokens: int = 0,
) -> float:
    """Full lifetime cost of putting `n_tokens` into a persistent context.

    This is the number that should drive placement decisions, not the one-off
    read cost. 10K tokens in a 1000-turn session costs ~$5.06 on Opus 5.

    Two assumptions live in the default arithmetic, and both are wrong in
    measurable ways: that every future re-read hits a warm cache at 0.10x, and
    that the token is still in the context to be re-read at all. Pass a
    `carry.Carry` fitted to local transcripts and neither is assumed -- the
    multiplier comes from what turns actually paid, and the read count is
    discounted by the token's chance of surviving each compaction. The default
    of `None` reproduces the old, uncorrected number exactly, so nothing that
    does not opt in moves.
    """
    if carry is not None:
        return carry.token_cost(n_tokens, model, remaining_turns, ttl=ttl, on=on,
                                context_tokens=context_tokens)
    return admitted_cost(n_tokens, Rates.for_model(model, ttl=ttl, on=on),
                         reads=max(0, remaining_turns)).total


@dataclass(frozen=True)
class Decision:
    """A gate result that can explain itself. Never returns a bare bool."""

    ok: bool
    saving: float          # USD saved by taking the action (negative = loses money)
    reason: str

    def __bool__(self) -> bool:  # allows `if gate(...)`
        return self.ok


def switch_is_profitable(
    from_model: str,
    to_model: str,
    ctx_tokens: int,
    est_out_tokens: int,
    *,
    switch_in_mult: float = 1.0,
    check_context: bool = True,
    on: date | None = None,
) -> Decision:
    """Is switching model for ONE turn of a warm conversation worth it?

    Staying keeps a warm cache (context read at 0.1x). Switching invalidates it --
    the cache is model-scoped, with no escape hatch -- so the new model re-reads
    the entire context at `switch_in_mult` x its input rate.

    `switch_in_mult=1.0` (uncached) is the optimistic case and yields the
    headline break-even `out > ctx/40` for Opus->Haiku. Real Claude Code caching
    writes at 1.25x, which tightens it to ~ctx/26.7. If the optimistic case
    already fails, the switch is definitely wrong.

    `check_context=True` additionally refuses switches the target model cannot
    physically hold. Set it False to probe the pure break-even arithmetic.
    """
    if resolve(from_model).id == resolve(to_model).id:
        # Not a switch. The arithmetic below charges the target model a full
        # uncached re-read of the context, because that is what invalidating a
        # model-scoped prefix costs -- but staying on the model you are already
        # on invalidates nothing. Left unguarded this reported a $0.45 loss for
        # doing nothing, which is the wrong answer to a question no caller
        # should be punished for asking.
        return Decision(False, 0.0,
                        f"already on {to_model}; there is no switch to price")
    rf, rt = Rates.for_model(from_model, on=on), Rates.for_model(to_model, on=on)
    stay = (ctx_tokens * rf.cache_read + est_out_tokens * rf.out) / M
    switch = (ctx_tokens * rt.inp * switch_in_mult + est_out_tokens * rt.out) / M
    saving = stay - switch
    breakeven = _breakeven_out(ctx_tokens, rf, rt, switch_in_mult)

    if check_context and not fits(to_model, ctx_tokens):
        return Decision(
            False, min(saving, 0.0) if saving < 0 else 0.0,
            f"impossible: {ctx_tokens:,} tok exceeds {to_model}'s "
            f"{_limit_str(to_model)} context limit. Even if it fit, the switch "
            f"needs >{breakeven:,.0f} output tok (est {est_out_tokens:,})",
        )

    if saving > 0:
        return Decision(True, saving, f"saves ${saving:.4f}: output win beats cache loss")
    return Decision(
        False,
        saving,
        f"loses ${-saving:.4f}: re-reading {ctx_tokens:,} tok uncached on "
        f"{to_model} costs more than the output saving "
        f"(needs >{breakeven:,.0f} output tok, est {est_out_tokens:,})",
    )


def _breakeven_out(ctx: int, rf: Rates, rt: Rates, mult: float) -> float:
    """Output tokens at which switching breaks even. inf if it never does.

    Takes `Rates`, not raw price tuples, so the cache read term is whatever the
    *source* model's provider actually charges to keep the conversation warm.
    Under an Anthropic-shaped 0.10x that is the familiar `ctx/40`; on a
    provider with no cache the staying cost is ten times higher and the switch
    breaks even far sooner. Hardcoding 0.10x here made the second case invisible.
    """
    denom = rf.out - rt.out
    if denom <= 0:
        return float("inf")
    return ctx * (rt.inp * mult - rf.cache_read) / denom


def placement_cost(
    *,
    tokens_read: int,
    summary_tokens: int,
    remaining_turns: int,
    main_model: str,
    sub_model: str | None = None,
    brief_tokens: int = 400,
    on: date | None = None,
    p_redo: float = 0.0,
    redo_overhead: float = 0.0,
    carry=None,
    context_tokens: int = 0,
) -> tuple[float, float, Decision]:
    """Inline vs. subagent for a read-heavy task. Returns (inline, sub, decision).

    Inline: everything read lands in the main context and is amortized.
    Subagent: reads happen once in a throwaway context on a cheap model; only
    the summary is amortized. This is the dominant lever in long sessions.

    Refuses to recommend a subagent whose context window cannot hold the read.

    `p_redo` is the term this gate spent a long time without, and its absence
    was not symmetric. Delegation can fail: the summary comes back missing the
    one detail the session needed, and the only repair is to read the thing
    inline after all. That path pays for the subagent run, the turn that noticed,
    *and* the inline read it was supposed to replace -- so a delegation that
    fails is strictly worse than never having delegated. Pricing it at zero made
    delegation look free of downside, which is exactly the error that turns a
    cost tool into a more expensive way to work. `escalation_is_profitable` has
    carried the equivalent term for tiers since the beginning; placement is the
    larger lever and had none.

    The default of 0.0 reproduces the older arithmetic for callers that have no
    measured redo rate to supply.
    """
    if not 0.0 <= p_redo <= 1.0:
        raise ValueError(f"p_redo must be in [0,1], got {p_redo}")
    sub_model = sub_model or DEFAULT_SUB_MODEL
    inline = admitted_token_cost(tokens_read, main_model, remaining_turns, on=on,
                                 carry=carry, context_tokens=context_tokens)

    rs = Rates.for_model(sub_model, on=on)
    # Subagent reads fresh (uncached) in its own short-lived context.
    sub_side = (brief_tokens * rs.inp + tokens_read * rs.inp + summary_tokens * rs.out) / M
    # Only the returned summary is admitted to the main context.
    sub = sub_side + admitted_token_cost(summary_tokens, main_model, remaining_turns,
                                         on=on, carry=carry,
                                         context_tokens=context_tokens)
    # A failed delegation is paid for twice over: the run above, the turn that
    # caught it, and then the inline read anyway.
    sub += p_redo * (inline + redo_overhead)
    saving = inline - sub

    need = brief_tokens + tokens_read + summary_tokens
    if not fits(sub_model, need):
        return inline, sub, Decision(
            False, 0.0,
            f"cannot delegate to {sub_model}: the task needs ~{need:,} tok of "
            f"context, over its {_limit_str(sub_model)} limit. Use a "
            f"larger subagent model or split the read",
        )

    reads = (carry.expected_reads(remaining_turns, context_tokens=context_tokens)
             if carry is not None else float(max(0, remaining_turns)))
    risk = (f", including a {p_redo:.0%} chance of having to read it inline anyway"
            if p_redo else "")
    if saving > 0:
        return inline, sub, Decision(
            True, saving,
            f"delegate: saves ${saving:.4f} "
            f"({tokens_read:,} tok stay out of a context re-read "
            f"{reads:,.0f} more times{risk})",
        )
    return inline, sub, Decision(
        False, saving,
        f"keep inline: delegating costs ${-saving:.4f} more "
        f"(only {reads:,.0f} re-reads left to amortize over{risk})",
    )


def escalation_is_profitable(
    cheap_model: str,
    expensive_model: str,
    *,
    ctx_tokens: int,
    est_out_tokens: int,
    p_fail: float,
    retry_overhead: float = 0.0,
    on: date | None = None,
) -> Decision:
    """Should we ATTEMPT the cheap tier, given it may fail and force a retry?

    A failed cheap attempt is not free: you pay for it AND for the expensive run.
    Try cheap only when  cheap + p_fail*(expensive + overhead) < expensive.
    `p_fail` must come from the outcome log, not a guess.

    The cheap run appears once, not twice. An earlier version of this priced the
    failure branch as `cheap + expensive`, which charged the cheap attempt again
    on top of the attempt you had already paid for. It made every cheap tier
    look worse than it is and it had no reading under which it was true -- the
    task is not run on the cheap model a second time, that is the whole point of
    escalating.

    `retry_overhead` is the part this gate used to omit. The failure does not
    announce itself: a main-session turn has to read the bad result, notice it
    is bad, and dispatch again, and that turn re-reads the whole session context
    at its own model's rate. Leaving it out made every cheap tier look better
    than it was, because the only cost of being wrong was priced inside the
    throwaway subagent context. Callers that know the session's context size
    should pass `policy.routing_overhead(...)`; the default of 0.0 reproduces
    the older, more optimistic arithmetic.
    """
    if not 0.0 <= p_fail <= 1.0:
        raise ValueError(f"p_fail must be in [0,1], got {p_fail}")
    if not fits(cheap_model, ctx_tokens):
        return Decision(
            False, 0.0,
            f"go straight to {expensive_model}: {ctx_tokens:,} tok exceeds "
            f"{cheap_model}'s {_limit_str(cheap_model)} limit",
        )
    cheap = run_cost(cheap_model, ctx_tokens, est_out_tokens, on)
    exp = run_cost(expensive_model, ctx_tokens, est_out_tokens, on)
    expected = cheap + p_fail * (exp + retry_overhead)
    saving = exp - expected
    max_p = max_tolerable_p_fail(cheap_model, expensive_model, ctx_tokens=ctx_tokens,
                                 est_out_tokens=est_out_tokens,
                                 retry_overhead=retry_overhead, on=on)
    if saving > 0:
        return Decision(True, saving,
                        f"try cheap: expected saving ${saving:.4f} "
                        f"(p_fail {p_fail:.0%} < {max_p:.0%})")
    return Decision(
        False, saving,
        f"go straight to {expensive_model}: at p_fail={p_fail:.0%} the retry risk "
        f"costs ${-saving:.4f} more than just doing it once "
        f"(needs p_fail below {max_p:.0%})",
    )


def max_tolerable_p_fail(
    cheap_model: str,
    expensive_model: str,
    *,
    ctx_tokens: int,
    est_out_tokens: int,
    retry_overhead: float = 0.0,
    on: date | None = None,
) -> float:
    """Highest failure rate at which attempting the cheap tier still pays.

    Solving `cheap + p*(exp + overhead) <= exp` for p. This is the
    number that decides how much measured history a tier needs before it earns
    the work: a tier whose observed `p_fail` sits above its own break-even is
    not a cheap tier, it is a slow way to pay for the expensive one.
    """
    cheap = run_cost(cheap_model, ctx_tokens, est_out_tokens, on)
    exp = run_cost(expensive_model, ctx_tokens, est_out_tokens, on)
    denom = exp + retry_overhead
    if denom <= 0:
        return 0.0
    return max(0.0, min(1.0, (exp - cheap) / denom))


def run_cost(model: str, ctx_tokens: int, est_out_tokens: int,
             on: date | None = None) -> float:
    """One cold run of a task: read the context once, write the answer once.

    Uncached on purpose. This prices a *subagent* run, which starts with no
    prefix of its own, so it is the right unit for comparing tiers against each
    other. It is the wrong unit for a turn inside a warm session -- use
    `marginal_turn_cost` for that.
    """
    r = Rates.for_model(model, on=on)
    return (ctx_tokens * r.inp + est_out_tokens * r.out) / M


# --------------------------------------------------------------------------
# Cache TTL, expiry, and fan-out: three levers a price-only model cannot see.
# --------------------------------------------------------------------------

def cache_write_cost(n_tokens: int, model: str, ttl: str | None = "5m",
                     on: date | None = None) -> float:
    return n_tokens * Rates.for_model(model, ttl=ttl, on=on).cache_write / M


def cache_read_cost(n_tokens: int, model: str, on: date | None = None) -> float:
    return n_tokens * Rates.for_model(model, on=on).cache_read / M


def cache_storage_cost(n_tokens: int, model: str, hours: float,
                       on: date | None = None) -> float:
    """USD to *hold* an explicit cache entry for `hours`, independent of reads.

    Zero everywhere except Google, which is the point of having it. On a
    provider that bills storage, an idle session is not free: a 500K-token
    context parked for an hour costs real money while nothing happens. Every
    other cost term in this module is driven by tokens moved; this one is
    driven by elapsed time, and no amount of prompt discipline reduces it.
    """
    r = Rates.for_model(model, on=on)
    return n_tokens * r.storage * max(0.0, hours) / M


def cache_miss_cost(n_tokens: int, model: str, ttl: str = "5m",
                    on: date | None = None) -> float:
    """Extra USD paid when a cache entry expired and had to be rewritten.

    Re-writing costs `write_mult` x input rate; reading would have cost 0.10x.
    On Opus 5 with a 5m TTL that is 1.15x input rate of pure waste -- 11.5x
    what the read would have been.
    """
    return cache_write_cost(n_tokens, model, ttl, on) - cache_read_cost(n_tokens, model, on)


def choose_ttl(
    n_tokens: int,
    model: str,
    *,
    turns: int,
    gap_seconds: float,
    on: date | None = None,
) -> tuple[str, float, str]:
    """Pick the cheaper cache TTL for a session with a known think-time gap.

    5m costs 1.25x to write but expires after 300s of idle; 1h costs 2.00x and
    survives an hour. A session whose turns are more than 5 minutes apart pays
    the 5m write premium on EVERY turn instead of once.

    Returns (ttl, saving_vs_other, reason).
    """
    spec = resolve(model)
    prov = spec.provider
    turns = max(1, turns)

    if not prov.caches:
        return "", 0.0, (
            f"{model} has no prompt cache on {prov.name}: there is no TTL to "
            f"choose, and every turn re-reads the prefix at full input rate")

    ttls = prov.ttl_seconds or {}
    if len(ttls) < 2:
        only = prov.default_ttl
        secs = prov.ttl_for(only)
        expired = secs is not None and gap_seconds > secs
        why = (
            f"{prov.name} caching is {prov.cache_style}: the TTL is not "
            f"selectable" + (
                f", and at {gap_seconds:.0f}s between turns the prefix expires "
                f"between them anyway -- the lever here is turn latency, not TTL"
                if expired else
                f", and at {gap_seconds:.0f}s between turns the prefix stays warm")
        )
        return only, 0.0, why

    costs = {}
    for ttl, secs in ttls.items():
        # A write happens on the first turn and again after every expiry.
        expiries = turns - 1 if gap_seconds > secs else 0
        writes = 1 + expiries
        reads = turns - writes
        w = spec.cache_write_rate(ttl, on)
        rd = spec.cache_read_rate(on)
        costs[ttl] = (writes * n_tokens * w + max(0, reads) * n_tokens * rd) / M

    best = min(costs, key=lambda t: costs[t])
    others = [c for t, c in costs.items() if t != best]
    saving = (min(others) - costs[best]) if others else 0.0
    short = min(ttls, key=lambda t: ttls[t])
    long = max(ttls, key=lambda t: ttls[t])
    wm_short = spec.cache_write_rate(short, on) / max(1e-12, spec.rate(on).inp)
    wm_long = spec.cache_write_rate(long, on) / max(1e-12, spec.rate(on).inp)
    rm = spec.cache_read_rate(on) / max(1e-12, spec.rate(on).inp)

    if gap_seconds > ttls[short] and best == long:
        why = (f"turns are {gap_seconds:.0f}s apart, past the {short} TTL: a "
               f"{short} cache expires every turn and is rewritten at "
               f"{wm_short:.2f}x instead of read at {rm:.2f}x")
    elif best == short and gap_seconds > ttls[short]:
        # The short TTL still wins, but not because the prefix survives -- it
        # does not. There is simply no second turn for the expiry to cost
        # anything on, so the only term left is the cheaper write. Saying
        # "inside the TTL" here was false on its face: the gap is right there
        # in the sentence, larger than the TTL it claims to be inside of.
        why = (f"turns are {gap_seconds:.0f}s apart, past the {short} TTL, but "
               f"{turns} turn{'s' if turns != 1 else ''} never re-reads the "
               f"prefix -- with no cache read to lose, the cheaper "
               f"{wm_short:.2f}x write wins outright")
    elif best == short:
        why = (f"turns are {gap_seconds:.0f}s apart, inside the {short} TTL; the "
               f"{wm_long:.2f}x write premium of {long} never pays back")
    else:
        why = f"{long} keeps the prefix warm across the measured gap"
    return best, saving, why


def fanout_cost(
    n_agents: int,
    shared_prefix_tokens: int,
    model: str,
    *,
    serialise_first: bool = True,
    ttl: str = "5m",
    on: date | None = None,
) -> tuple[float, float, Decision]:
    """Cost of N parallel calls over a shared prefix, staggered vs. all-at-once.

    A cache entry only becomes readable once the first response starts
    streaming. Fire N identical-prefix requests simultaneously and all N pay the
    write premium. Await the first token of one, then fan out the rest, and it
    is 1 write + (N-1) reads.

    Returns (parallel_cost, staggered_cost, decision).
    """
    n = max(1, n_agents)
    w = cache_write_cost(shared_prefix_tokens, model, ttl, on)
    rd = cache_read_cost(shared_prefix_tokens, model, on)
    parallel = n * w
    staggered = w + (n - 1) * rd
    saving = parallel - staggered
    if not caches(model, shared_prefix_tokens):
        spec = resolve(model)
        if not spec.provider.caches:
            return parallel, staggered, Decision(
                False, 0.0,
                f"{model} has no prompt cache on {spec.provider.name}: every "
                f"call re-reads the {shared_prefix_tokens:,}-token prefix at "
                f"full input rate whether they are staggered or not. The lever "
                f"is a smaller shared prefix, not the ordering",
            )
        return parallel, staggered, Decision(
            False, 0.0,
            f"prefix is {shared_prefix_tokens:,} tok, under {model}'s "
            f"{cache_min(model):,}-token cache minimum: nothing caches either way",
        )
    if n < 2 or saving <= 0:
        return parallel, staggered, Decision(
            False, saving, "single call; nothing to stagger")
    if not serialise_first:
        return parallel, staggered, Decision(
            True, saving,
            f"stagger the fan-out: {n} simultaneous calls each rewrite the "
            f"{shared_prefix_tokens:,}-token prefix (${parallel:.4f}); waiting for "
            f"the first to start streaming makes the rest cache reads (${staggered:.4f})",
        )
    return parallel, staggered, Decision(
        True, saving, f"staggered fan-out saves ${saving:.4f} over {n} cold writes")


def effort_saving(
    out_tokens: int,
    model: str,
    *,
    from_effort: str = "high",
    to_effort: str = "medium",
    remaining_turns: int = 0,
    mult: dict[str, float] | None = None,
    on: date | None = None,
) -> tuple[float, Decision]:
    """USD saved by lowering reasoning effort, including downstream re-reads.

    Effort is the only lever that reduces output volume without changing the
    model, so it keeps the cache warm -- unlike a downgrade. And because output
    is re-read on every later turn, the saving carries the same debt multiple as
    terseness does.
    """
    mult = mult or EFFORT_OUTPUT_MULT
    if from_effort not in mult or to_effort not in mult:
        raise ValueError(f"unknown effort level; known: {sorted(mult)}")

    # Feasibility before profitability, the same order every other gate here
    # uses. Haiku 4.5 rejects the `effort` parameter outright, and this happily
    # quoted a saving for lowering it -- an instruction the API returns a 400
    # for is not a lever, at any price. Only checked where the answer is known:
    # `efforts` is authoritative on the first-party table and merely absent on a
    # catalog entry, and refusing on absence would silently switch the lever off
    # for every non-Claude model.
    spec = resolve(model)
    if spec.first_party and not spec.supports_effort(to_effort):
        levels = ", ".join(spec.efforts) or "none"
        return 0.0, Decision(
            False, 0.0,
            f"{model} does not accept effort={to_effort} "
            f"(it takes: {levels}); there is no effort lever to pull here")

    r = Rates.for_model(model, on=on)
    delta = out_tokens * (mult[from_effort] - mult[to_effort])
    gen = delta * r.out / M
    reread = delta * r.cache_read * max(0, remaining_turns) / M
    total = gen + reread
    if delta <= 0:
        return total, Decision(
            False, total, f"{to_effort} is not cheaper than {from_effort}")
    return total, Decision(
        True, total,
        f"effort {from_effort}->{to_effort} cuts ~{delta:,.0f} output tok: "
        f"${gen:.4f} generation + ${reread:.4f} downstream re-reads, "
        f"with no cache invalidation (same model)",
    )


def batch_mult(model: str) -> float:
    """The provider's async-batch multiplier, or 1.0 where there is no batch tier.

    Anthropic, OpenAI, Google and Mistral all publish 50% off; several hosted
    open-weight endpoints publish nothing. Returning 1.0 for those is the point:
    a "batch it" recommendation on a provider with no batch API is not a saving,
    it is a suggestion to use a product that does not exist.
    """
    try:
        m = resolve(model).batch_mult()
    except (UnknownModelError, UnpricedModelError):
        return BATCH_MULT
    return m if m is not None else 1.0


def batch_saving(cost: float, model: str | None = None) -> float:
    """USD saved by moving a workload to the provider's batch tier.

    `model` is optional only because the older callers did not pass one. Given
    a model, the discount is that provider's; without one it falls back to the
    Anthropic 50%, which is what every caller was silently assuming before.
    """
    m = BATCH_MULT if model is None else batch_mult(model)
    return cost * (1.0 - m)


def marginal_turn_cost(context_tokens: int, out_tokens: int, model: str,
                       on: date | None = None) -> float:
    """What one more turn costs at the current context size. The number that
    should be on screen when deciding whether to keep going.

    On a provider with no cache this is the number that stops a session: the
    context term is billed at full input rate, so the marginal turn is roughly
    ten times what an Anthropic-shaped estimate would say.
    """
    r = Rates.for_model(model, on=on)
    return (context_tokens * r.cache_read + out_tokens * r.out) / M


def _limit_str(model: str) -> str:
    """`"200,000-token"` or `"undeclared"`. Never `"None-token"`.

    The catalog does not know every model's context window, and a limit of
    None reaching an f-string used to render as a sentence claiming the model
    has a `None,`-token limit. Unknown is a real answer and is worth saying.
    """
    n = context_limit(model)
    return f"{n:,}-token" if n else "undeclared (catalog has no context window for it)"
