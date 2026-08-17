"""What a token admitted to context actually costs to carry, measured.

`cost.admitted_token_cost` prices a token that enters the context as one cache
write plus `remaining_turns` cache reads at 0.10x. That expression is the
foundation of every placement decision in this repo, and all three of its terms
are wrong in ways that do not cancel.

**The multiplier is not 0.10.** It is 0.10 only when the prefix is warm on the
turn that re-reads it. Real turns miss: the 5m TTL expires while you read the
diff, a tool result lands past the cache-breakpoint lookback, a parallel fan-out
races the first write. A miss re-writes at 1.25x or 2.00x instead of reading at
0.10x, which is a 12.5x to 20x swing on that turn. The honest number is the
*realized* input multiplier, and it is sitting in the transcripts already:
every turn records how its input actually split across uncached, read, and
written. `measured_read_mult` reads it off rather than assuming it. On this
machine it comes out well above 0.10, which means the carry term -- the term
that is already ~76% of spend -- was being **under**-priced, and delegation
under-recommended.

**The token does not survive to the end of the session.** Compaction evicts it.
Pricing `remaining_turns` re-reads of a token that will be summarised away in 80
turns over-states the carry, and over-statement is how a cost tool starts
recommending work that loses money. Residency is a survival problem: the token
is re-read every turn until the next compaction, survives that compaction with
roughly the ratio the compaction kept, and so on. `expected_reads` sums that
geometric-per-epoch series exactly instead of truncating or ignoring it.

**The horizon is a mean, not a median.** Carry cost is *linear* in remaining
turns, so its expectation is set by `E[R]` -- and session length is heavy-tailed
enough here that the median sits far below the mean. Using the median under-
prices exactly the long sessions that hold the spend. `horizon.mean_remaining`
supplies the right one; this module takes whichever it is handed and says which.

The two corrections push in opposite directions, which is the point: they are
not a fudge factor, and neither one is safe to apply without the other.

Two closed forms fall out once the carry number is honest
---------------------------------------------------------
* `optimal_split` -- how many turns to run before starting fresh. Per-turn cost
  in a session growing at `g` tokens/turn is `m*r*F + m*r*g*(k+1)/2 + W/k` for a
  cycle of `k` turns, so `k* = sqrt(2W / (m*r*g))`. It is a square root, which is
  why the answer is a few hundred turns and not "compact constantly".
* `delegate_threshold` -- the read size above which delegating pays. Both sides
  are affine in the read size, so the break-even is one division. That matters
  more than it looks: a threshold is a rule a hook can apply for free, with no
  routing turn to pay for, which is the only form of advice here that is
  guaranteed cheaper than not asking.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import sqrt
from pathlib import Path

from adder.core import settings as _settings
from adder.core.filters import root_of as _root_of
from adder.core.trace import COMPACT_MAX_SURVIVAL as _COMPACT_MAX_SURVIVAL
from adder.core.trace import COMPACT_TRIGGER_FRACTION as _COMPACT_TRIGGER_FRACTION
from adder.core.trace import is_compaction
from adder.pricing.cost import Rates
from adder.pricing.prices import CACHE_READ_MULT
from adder.pricing.registry import provider_for

M = 1_000_000.0

# Fallbacks for a machine with no transcripts to measure. Deliberately close to
# the old assumptions so that installing this module changes nothing until
# there is data; every one of them is replaced by a measurement when there is.
DEFAULT_READ_MULT = CACHE_READ_MULT      # 0.10: the assumption being tested
DEFAULT_GROWTH = 900                     # tokens admitted per turn
DEFAULT_SURVIVAL = 0.30                  # share of context a compaction keeps
DEFAULT_COMPACT_EVERY = 0                # 0 = never observed one; do not model it
MIN_SESSIONS = 3

# What counts as a compaction rather than a wobble.
#
# Any turn whose context is smaller than the previous turn's is a candidate, and
# most candidates are not compactions: measured here, 122 turns out of 20,524
# show a context drop, but only 7 are auto-compactions. The rest are small dips
# from branch resumption and sidechain accounting, clustered between 0.65x and
# 0.98x of the previous context. Treating those as compactions put the fitted
# period at 4 turns -- which would have said a token admitted now is re-read 16
# times in a 348-turn session, understating carry cost by 20x and switching off
# delegation across the board.
#
# Auto-compaction is triggered by the context limit, so the detector keys on
# that: the context has to have been near the model's ceiling AND have lost most
# of itself. The 7 real events all sit at 999.5K-999.9K dropping to 4-6%.
# Re-exported from `core.trace`, which owns the detector: `Session.compactions`
# needs it too and `core` may not import `measure`. The measurement behind the
# two numbers is written up beside them there.
COMPACT_TRIGGER_FRACTION = _COMPACT_TRIGGER_FRACTION
COMPACT_MAX_SURVIVAL = _COMPACT_MAX_SURVIVAL


@dataclass(frozen=True)
class Carry:
    """The measured parameters of carrying context, and where they came from.

    `source` is not decoration. A carry model fitted to 200 sessions and one
    fitted to a hard-coded prior produce the same dataclass, and a caller about
    to multiply a $30 decision by `read_mult` needs to know which it holds.
    """

    read_mult: float = DEFAULT_READ_MULT
    # The multiplier the uncorrected model *would* have assumed, which is a
    # property of the provider rather than a constant. Comparing a measured
    # 0.28x against Anthropic's 0.10x says the old model under-priced carry
    # 2.8x; comparing the same 0.28x against a no-cache provider's 1.00x says
    # the opposite. Reporting one number as if it were the other is how a
    # correction becomes a new error.
    baseline_read_mult: float = DEFAULT_READ_MULT
    growth: float = DEFAULT_GROWTH
    survival: float = DEFAULT_SURVIVAL
    compact_every: int = DEFAULT_COMPACT_EVERY
    context_limit: int = 0               # observed compaction trigger, 0 if none
    sessions: int = 0
    source: str = "prior"

    @property
    def measured(self) -> bool:
        return self.source == "measured"

    @property
    def inflation(self) -> float:
        """How much the honest multiplier exceeds the one the old model assumed."""
        return self.read_mult / max(1e-12, self.baseline_read_mult)

    def expected_reads(self, remaining_turns: int, *, context_tokens: int = 0) -> float:
        """Expected number of future turns that re-read a token admitted now.

        The naive answer is `remaining_turns`. This is that, minus the turns the
        token will not be present for, because compaction removed it.

        Compaction is modelled as periodic in *tokens*, not turns: the context
        fills at `growth` per turn, trips at `context_limit`, and drops to
        `survival` of the trigger. A token survives one compaction with
        probability `survival` -- it is the share of the context that was kept --
        so residency after `j` compactions is `survival^j`, and the expected
        read count is the sum over epochs of the turns in that epoch, discounted
        by how likely the token is to still be there.

        With no observed compaction the sum collapses to `remaining_turns`,
        which is the old behaviour, reached honestly.
        """
        R = max(0, int(remaining_turns))
        if R == 0:
            return 0.0
        first, period = self._epochs(context_tokens)
        if period <= 0:
            return float(R)
        total, t, alive = 0.0, 0, 1.0
        span = first
        # Bounded by R/period + 1 iterations; `survival < 1` makes the tail
        # vanish quickly, but the loop is exact rather than truncated.
        while t < R:
            take = min(span, R - t)
            total += alive * take
            t += take
            alive *= self.survival
            span = period
            if alive < 1e-9:
                break
        return total

    def _epochs(self, context_tokens: int) -> tuple[int, int]:
        """(turns to the first compaction, turns between later ones)."""
        if self.compact_every <= 0 or self.growth <= 0:
            return (0, 0)
        period = self.compact_every
        if self.context_limit and context_tokens:
            headroom = self.context_limit - context_tokens
            first = max(1, int(headroom / self.growth)) if headroom > 0 else 1
            return (min(first, period * 4), period)
        return (period, period)

    def token_cost(
        self,
        n_tokens: int,
        model: str,
        remaining_turns: int,
        *,
        context_tokens: int = 0,
        ttl: str = "5m",
        on: date | None = None,
    ) -> float:
        """Lifetime cost of admitting `n_tokens`, with both corrections applied."""
        r = Rates.for_model(model, ttl=ttl, on=on)
        reads = self.expected_reads(remaining_turns, context_tokens=context_tokens)
        write = n_tokens * r.cache_write / M
        # `read_mult` is measured as a fraction of the *input* rate, so it is
        # applied to the input rate here rather than to the provider's cache
        # read rate -- the measurement already contains whatever that is.
        return write + n_tokens * r.inp * self.read_mult * reads / M

    def describe(self) -> str:
        if not self.measured:
            return ("carry model: prior (no local transcripts); assuming the "
                    f"{self.baseline_read_mult:.2f}x re-read multiplier and no "
                    "compaction")
        comp = (f"compacts every ~{self.compact_every} turns keeping "
                f"{self.survival:.0%}" if self.compact_every else "no compaction observed")
        return (f"carry model from {self.sessions} sessions: re-read multiplier "
                f"{self.read_mult:.3f}x ({self.inflation:.1f}x the assumed "
                f"{self.baseline_read_mult:.2f}x), growth {self.growth:,.0f} "
                f"tok/turn, {comp}")

    @classmethod
    def default(cls) -> Carry:
        return cls()

    @classmethod
    def measure(cls, sessions, *, min_turns: int = 20) -> Carry:
        """Fit the carry parameters to recorded transcripts."""
        rows = [s for s in sessions.values() if len(s.turns) >= min_turns]
        if len(rows) < MIN_SESSIONS:
            return cls.default()

        mult = measured_read_mult(sessions, min_turns=min_turns)
        growth, survival, period, trigger = _context_dynamics(rows)
        return cls(
            read_mult=mult if mult > 0 else DEFAULT_READ_MULT,
            baseline_read_mult=_baseline_read_mult(rows),
            growth=growth or DEFAULT_GROWTH,
            survival=survival,
            compact_every=period,
            context_limit=trigger,
            sessions=len(rows),
            source="measured",
        )


def _baseline_read_mult(rows) -> float:
    """The read multiplier the uncorrected model would have assumed here.

    Token-weighted across the models actually present, because a workload can
    span providers and the comparison is only meaningful against what *this*
    workload's uncorrected estimate would have been. Falls back to Anthropic's
    0.10x when nothing resolves, which is the number the old model used
    unconditionally.
    """
    num = den = 0.0
    for s in rows:
        for t in s.turns:
            ctx = t.context
            if ctx <= 0:
                continue
            prov = provider_for(t.model)
            mult = prov.cache_read_mult if prov.caches else 1.0
            num += (mult if mult is not None else CACHE_READ_MULT) * ctx
            den += ctx
    return (num / den) if den else DEFAULT_READ_MULT


def measured_read_mult(sessions, *, min_turns: int = 20) -> float:
    """The realized per-token input multiplier, weighted by tokens carried.

    For every turn, `(uncached*1.0 + read*0.10 + written*write_mult) / context`
    is what that turn actually paid, per token of context, as a multiple of the
    input rate. The token-weighted mean over all turns is the number
    `admitted_token_cost` should be using in place of the flat 0.10.

    The weighting is by context tokens, not by turn, on purpose: a 900K-token
    turn and a 3K-token turn are not two equal observations of the same
    quantity, and averaging them per-turn lets cheap early turns dilute the
    number that prices expensive late ones.

    First turns are excluded. A session's opening turn writes its whole prefix
    by construction -- there is nothing to have cached -- so including it
    measures the cost of starting a session, not the cost of continuing one,
    and that is a different decision priced elsewhere.
    """
    weighted = carried = 0.0
    for s in sessions.values():
        if len(s.turns) < min_turns:
            continue
        # Each chain skipped past its own first turn, not just the session's.
        # The docstring's reason -- "a session's opening turn writes its whole
        # prefix by construction, there is nothing to have cached" -- is exactly
        # as true of a subagent's opening turn, which sits at some index past
        # zero in the combined list. Worth 0.00% here, where the token-weighted
        # denominator swamps a handful of openings; worth stating anyway,
        # because the reasoning is the same one the exclusion rests on.
        for chain in (s.main_turns, [t for t in s.turns if t.sidechain]):
            for t in chain[1:]:
                ctx = t.context
                if ctx <= 0:
                    continue
            # Multipliers are per-turn because a workload can span providers,
            # and the whole point of this fit is what the turns *actually*
            # paid. Normalising by the turn's own input rate keeps the result a
            # pure multiplier, comparable across models with different prices.
                rt = t.rates()
                inp = rt.inp or 1.0
                paid = (t.uncached_in
                        + t.cache_read * (rt.cache_read / inp)
                        + t.cache_write * (rt.cache_write / inp))
                weighted += paid
                carried += ctx
    return weighted / carried if carried else 0.0


# The private name this was born with. `compact` needs the detector and a
# second implementation of "was that a compaction" is exactly the kind of
# disagreement this repo has paid for before.
_is_compaction = is_compaction


def _context_dynamics(rows) -> tuple[float, float, int, int]:
    """(growth per turn, compaction survival ratio, turns between, trigger size).

    Everything here is read off the compaction events themselves: how big the
    context was when it tripped, what share of it came back, and how many turns
    separated one from the next.
    """
    admissions: list[int] = []
    survivals: list[float] = []
    gaps: list[int] = []
    triggers: list[int] = []
    for s in rows:
        # Main chain only. A subagent runs in its own window, so the step down
        # into one and the step back out are not admissions to this context --
        # the climb back reads as a single enormous one. The median absorbs it
        # on this corpus (955 vs 945 tok/turn, 1%), which is why it went
        # unnoticed; on a delegation-heavy workload it does not, and `growth`
        # sets `k*` as 1/sqrt(g).
        main = s.main_turns
        ctxs = [t.context for t in main]
        models = [t.model for t in main]
        last_compaction = None
        for i in range(1, len(ctxs)):
            d = ctxs[i] - ctxs[i - 1]
            if d >= 0:
                admissions.append(d)
                continue
            if not _is_compaction(ctxs[i - 1], ctxs[i], models[i]):
                continue
            survivals.append(ctxs[i] / ctxs[i - 1])
            triggers.append(ctxs[i - 1])
            if last_compaction is not None:
                gaps.append(i - last_compaction)
            last_compaction = i
    growth = _median(admissions) if admissions else 0.0
    survival = _median(survivals) if survivals else DEFAULT_SURVIVAL
    period = int(_median(gaps)) if gaps else 0
    trigger = int(_median(triggers)) if triggers else 0
    # Most sessions end before they compact twice, so the observed gap is often
    # unavailable while the trigger and survival ratio are not. The period is
    # then implied rather than invented: at `growth` tokens per turn it takes
    # this many turns to refill what the compaction threw away.
    if not period and trigger and growth > 0:
        period = max(1, int(trigger * (1.0 - survival) / growth))
    return growth, survival, period, trigger


def _median(xs) -> float:
    """The shared median. `stats.median` is the one definition in this repo.

    `util.stats` opens by naming this exact pair -- "`carry` and `prefix` each
    carried a private `_median`" -- as the duplication it exists to replace,
    and both copies were still here. They agreed with it by luck rather than by
    construction: nothing stopped one of them drifting, which is how three
    different p90s from the same data happened the first time.
    """
    from adder.util.stats import median

    return median(xs)


# --------------------------------------------------------------------------
# The two closed forms.
# --------------------------------------------------------------------------

def optimal_split(
    *,
    model: str,
    carry: Carry | None = None,
    floor_tokens: int,
    handoff_tokens: int = 2_000,
    observed_turns: int = 0,
    ttl: str = "5m",
    restart_cost: float | None = None,
    growth: float | None = None,
    read_mult: float | None = None,
    on: date | None = None,
) -> tuple[int, float, str]:
    """How many turns to run before starting fresh. Returns (k*, saving/turn, why).

    Within one session the context grows at `g` tokens per turn, so the average
    per-turn input cost over a cycle of `k` turns is

        A(k) = m*r*F  +  m*r*g*(k+1)/2  +  W/k  +  H*m*r

    where `F` is the floor a restart cannot avoid (system prompt, tools,
    CLAUDE.md), `m` the measured re-read multiplier, `W = (F+H)*w*r` the one-off
    write a restart pays, and `H` the working context that has to be
    re-established and then carried. Only the middle two terms depend on `k`:

        dA/dk = m*r*g/2 - W/k^2 = 0   =>   k* = sqrt(2W / (m*r*g))

    **The square root is the result.** It is why the answer is not "compact
    constantly" and not "never": quadrupling the cost of a handoff only doubles
    the optimal session length. It is also the reason this number is robust to
    the softest input it has. `H` is a judgement call -- nothing in a transcript
    records how much context a person needs to resume -- but being wrong about
    it by 10x moves `k*` by about 3x, which is the difference between "split
    around 50 turns" and "split around 150", not the difference between an
    answer and a guess.

    `W` is assumed here unless it is handed over: the whole floor, rewritten at
    the cache-write multiplier. That is a **cold** restart, and `adder prefix`
    measures that restarts are not cold. About 75% of an opening context arrives
    as a cache read at 0.10x, because the expensive part of the floor is
    identical across sessions and still resident. Assuming the rebuild
    over-states `W` several-fold and so over-states `k*` by its square root,
    which is most of the distance between "split in the hundreds of turns" and
    the measured answer. Pass `restart_cost` -- `prefix.Opening.cost` returns
    exactly that number -- to price it off the measurement instead. The cold
    assumption stays the default because with no data it errs in the safe
    direction: it recommends restarting less, not more.

    What this does NOT price is the part that is not tokens: a restart costs
    attention, and a handoff loses things nobody wrote down. So `k*` is the
    token-optimal cycle and a lower bound on the sensible one. It is quoted
    with the sensitivity rather than on its own for exactly that reason.
    """
    c = carry or Carry.default()
    rates = Rates.for_model(model, ttl=ttl, on=on)
    r = rates.inp
    g = max(1.0, growth if growth is not None else c.growth)
    m = max(1e-6, read_mult if read_mult is not None else c.read_mult)
    per_tok = m * r / M                       # $ per token of context per turn
    # The restart is paid at whatever this provider charges to lay a prefix
    # down, expressed as a multiple of input so the algebra below is unchanged.
    w = (rates.cache_write / r) if r else 1.0
    if restart_cost is not None:
        W = max(1e-12, restart_cost)
    else:
        W = (floor_tokens + handoff_tokens) * w * r / M
    k_int = max(1, round(sqrt(2.0 * W / (per_tok * g))))

    def avg(kk: float) -> float:
        """Average per-turn input cost on a `kk`-turn cycle, handoff included."""
        return (per_tok * floor_tokens + per_tok * g * (kk + 1.0) / 2.0
                + W / kk + handoff_tokens * per_tok)

    # Compare against how long sessions actually run, not against a made-up
    # alternative. A session that never splits pays no handoff and no restart,
    # so those terms come off the baseline rather than being assumed away.
    n = observed_turns if observed_turns > 0 else k_int * 4
    never = per_tok * floor_tokens + per_tok * g * (n + 1.0) / 2.0
    saving = never - avg(k_int)
    how = ("re-opening costs" if restart_cost is not None
           else f"restarting rewrites {floor_tokens + handoff_tokens:,} tok for")
    why = (f"context grows {g:,.0f} tok/turn at {m:.3f}x, and {how} ${W:,.3f}; "
           f"k* = sqrt(2W/(m*r*g)) = {k_int:,} turns")
    return k_int, saving, why


def split_sensitivity(
    *,
    model: str,
    carry: Carry | None = None,
    floor_tokens: int,
    handoffs=(2_000, 20_000, 100_000, 400_000),
    observed_turns: int = 0,
    on: date | None = None,
) -> list[tuple[int, int, float]]:
    """(handoff tokens, k*, saving per turn) across plausible handoff sizes.

    The handoff is the one input here that cannot be measured from a transcript,
    so it is swept rather than assumed. If `k*` stays well below how long
    sessions actually run across the whole sweep, the conclusion does not depend
    on the number nobody knows.
    """
    out = []
    for h in handoffs:
        k, saving, _ = optimal_split(model=model, carry=carry,
                                     floor_tokens=floor_tokens, handoff_tokens=h,
                                     observed_turns=observed_turns, on=on)
        out.append((h, k, saving))
    return out


def delegate_threshold(
    *,
    main_model: str,
    sub_model: str,
    remaining_turns: int,
    carry: Carry | None = None,
    summary_ratio: float = 0.10,
    brief_tokens: int = 400,
    p_redo: float = 0.0,
    redo_overhead: float = 0.0,
    context_tokens: int = 0,
    ttl: str = "5m",
    on: date | None = None,
) -> tuple[float, str]:
    """Smallest read, in tokens, for which delegating beats reading inline.

    Both sides are affine in the read size `x`:

        inline(x) = x * r_m * (w + m*E)
        deleg(x)  = (b + x) * r_s + p*x*r_s_out
                    + p*x * r_m * (w + m*E)
                    + p_redo * (x*r_m*(w + m*E) + redo_overhead)

    so the break-even is one division rather than a search. Returns `inf` when
    delegation never pays -- which happens, and saying so is more useful than
    quoting a threshold nobody will reach.

    The reason a threshold is worth having at all, when `policy.decide` can
    answer the same question per task: applying a threshold costs nothing. There
    is no routing turn, so there is no overhead to clear, so the advice is
    cheaper than not taking it by construction. Every other recommendation in
    this repo has to earn back the turn spent producing it.
    """
    c = carry or Carry.default()
    rm = Rates.for_model(main_model, ttl=ttl, on=on)
    rs = Rates.for_model(sub_model, on=on)
    E = c.expected_reads(remaining_turns, context_tokens=context_tokens)
    # $/token admitted: one write at the provider's write rate, then `E` reads
    # at the measured multiple of its input rate.
    admit = (rm.cache_write + c.read_mult * rm.inp * E) / M
    p = max(0.0, min(1.0, summary_ratio))

    # Coefficient of x on each side.
    inline_slope = admit
    deleg_slope = rs.inp / M + p * rs.out / M + p * admit + p_redo * admit
    gain = inline_slope - deleg_slope
    fixed = brief_tokens * rs.inp / M + p_redo * redo_overhead
    if gain <= 0:
        return float("inf"), (
            "delegation never pays at this horizon: the summary plus the "
            "subagent's own read costs at least as much per token as keeping "
            "the read inline")
    x = fixed / gain
    return x, (f"delegate reads over ~{x:,.0f} tok: below that the {brief_tokens:,}-token "
               f"brief and the summary cost more than the {E:,.0f} re-reads they avoid")


def _threshold_json(c: Carry, model: str, remaining: int, p_redo: float,
                    context_tokens: int) -> dict:
    """The delegate-above threshold, as a value and the reason behind it."""
    tokens, why = delegate_threshold(
        main_model=model, sub_model=_settings.sub_model(),
        remaining_turns=remaining, carry=c, p_redo=p_redo,
        context_tokens=context_tokens)
    return {"tokens": None if tokens == float("inf") else round(tokens, 1),
            "reason": why}


def main(argv: list[str] | None = None) -> int:
    import argparse

    from adder.core.trace import load_sessions
    from adder.measure.session.horizon import load as load_horizon

    ap = argparse.ArgumentParser(
        prog="adder carry",
        description="Measure what carrying a token in context actually costs.")
    ap.add_argument("root", nargs="?", default=None,
                    help="transcript directory (default: the `root` setting)")
    ap.add_argument("--model", default=_settings.session_model())
    ap.add_argument("--remaining", type=int, default=None,
                    help="horizon to price against (default: the measured MEAN)")
    ap.add_argument("--context", type=int, default=0,
                    help="current context size, for the compaction headroom")
    ap.add_argument("--p-redo", type=float, default=0.15,
                    help="chance a delegated read has to be redone inline")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    a = ap.parse_args(argv)
    # `root_of`: the argument if one was given, else the `root`
    # setting. Resolved here so two commands cannot disagree
    # about which transcript directory `adder config` names.
    a.root = str(_root_of(a))

    sessions = load_sessions(Path(a.root))
    c = Carry.measure(sessions)
    h = load_horizon(a.root)
    remaining = a.remaining if a.remaining is not None else int(h.mean_remaining(0))
    rates = Rates.for_model(a.model)
    r = rates.inp
    # What this workload's uncorrected estimate would have used. On Claude
    # that is 0.10x; on a provider with no cache it is 1.00x, and quoting
    # 0.10x there would invent a correction that runs the wrong way.
    base = c.baseline_read_mult

    if a.json:
        import json

        naive_10k = (10_000 * rates.cache_write
                     + 10_000 * r * base * remaining) / M
        print(json.dumps({
            "measured": c.measured,
            "read_mult": round(c.read_mult, 5),
            "assumed_read_mult": base,
            "inflation": round(c.inflation, 4),
            "growth_per_turn": round(c.growth, 1),
            "survival": round(c.survival, 4),
            "compact_every": c.compact_every,
            "model": a.model,
            "remaining_turns": remaining,
            "expected_reads": round(c.expected_reads(remaining,
                                                     context_tokens=a.context), 2),
            "cost_10k_tokens": round(c.token_cost(10_000, a.model, remaining,
                                                  context_tokens=a.context), 4),
            "cost_10k_uncorrected": round(naive_10k, 4),
            "delegate_threshold": _threshold_json(c, a.model, remaining,
                                                  a.p_redo, a.context),
        }))
        return 0

    print(f"\n  {c.describe()}\n")

    if h.lengths:
        med, mean = h.remaining(0), h.mean_remaining(0)
        print(f"  Horizon at turn 0: median {med:,} turns, mean {mean:,.0f} "
              f"({mean / med:.2f}x)" if med else "")
        print("  Carry cost is LINEAR in remaining turns, so the mean prices it and")
        print("  the median only describes it. Using the median under-prices "
              "admission.")
    else:
        print(f"  No local sessions; using the flat {remaining}-turn prior.")
    print()

    reads = c.expected_reads(remaining, context_tokens=a.context)
    naive = (10_000 * rates.cache_write + 10_000 * r * base * remaining) / M
    honest = c.token_cost(10_000, a.model, remaining, context_tokens=a.context)
    print(f"  Admitting 10,000 tokens on {a.model} at {remaining:,} remaining turns")
    print(f"    assumed  {base:>5.2f}x x {remaining:>6,} reads   ${naive:>8,.2f}")
    print(f"    measured  {c.read_mult:.3f}x x {reads:>6,.0f} reads   ${honest:>8,.2f}"
          f"   {honest / naive:.2f}x" if naive else "")
    if c.compact_every:
        print(f"    (a token admitted now survives ~{c.compact_every:,} turns before "
              f"compaction keeps {c.survival:.0%} of it)")
    print()

    floor = _floor(sessions)
    obs = _cost_weighted_length(sessions)
    print("  Optimal session length, swept over the handoff cost nobody can measure")
    print(f"  (the sessions holding half the spend here run {obs:,} turns):")
    print(f"    {'handoff tok':>12}{'k*':>8}{'$/turn saved vs running to ' + str(obs):>34}")
    for hand, k, saving in split_sensitivity(model=a.model, carry=c,
                                             floor_tokens=floor, observed_turns=obs):
        print(f"    {hand:>12,}{k:>8,}{saving:>34,.4f}")
    print("    k* = sqrt(2W/(m*r*g)): quadrupling the handoff only doubles k*, so")
    print("    the conclusion survives being wrong about the one soft input.")
    print()

    overhead = (a.context or 500_000) * rates.cache_read / M
    print("  Delegation threshold: the read size above which delegating pays.")
    print(f"    {'brief tok':>10}{'p_redo':>9}{'threshold':>14}")
    for brief in (400, 2_000, 10_000):
        for p in (0.0, a.p_redo):
            x, _ = delegate_threshold(main_model=a.model,
                                      sub_model=_settings.sub_model(),
                                      remaining_turns=remaining, carry=c,
                                      brief_tokens=brief, p_redo=p,
                                      redo_overhead=overhead,
                                      context_tokens=a.context)
            shown = "never pays" if x == float("inf") else f"{x:,.0f} tok"
            print(f"    {brief:>10,}{p:>9.0%}{shown:>14}")
    print("    A threshold is the only advice here that is free to apply: no routing")
    print("    turn to pay for, so it cannot cost more than not asking.")
    print()
    return 0


def _cost_weighted_length(sessions) -> int:
    """Median session length, weighted by what the sessions cost.

    A plain median counts a 5-turn session and an 1,800-turn session as one
    observation each, and short sessions are numerous and nearly free. The
    question here is how long the sessions that hold the *money* run, because
    those are the ones a split cadence would apply to. `validate.sessions_are_long`
    makes the same correction for the same reason.
    """
    rows = sorted((len(s.turns), s.cost) for s in sessions.values() if s.turns)
    total = sum(c for _, c in rows)
    if not rows or total <= 0:
        return 0
    acc = 0.0
    for n, c in rows:
        acc += c
        if acc >= total / 2:
            return n
    return rows[-1][0]


def _floor(sessions) -> int:
    floors = [s.base_context for s in sessions.values() if s.turns]
    return int(_median(floors)) if floors else 20_000


if __name__ == "__main__":
    raise SystemExit(main())
