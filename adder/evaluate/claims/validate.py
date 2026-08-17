"""Re-test the empirical claims this repo's conclusions depend on.

Every design decision here rests on a handful of measured claims. Those claims
came from one machine's transcripts at one point in time, so they are re-checked
against live data rather than baked in as constants.

Three of these were wrong at some point during development. Two survived only
because a check like this caught them:

  * `cache_write` overcounts admitted content ~5x (cache segments get refreshed).
  * An OLS fit of growth-on-output suggested output drove only 37% of context.
    That was an artifact: output and tool results correlate, so OLS pushed shared
    variance into the intercept, and compacted sessions polluted the fit. The
    clean test -- sessions that never compacted -- gives 1.02x, confirming the
    original claim. Method matters more than the number.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from adder.core.filters import root_of as _root_of
from adder.core.trace import DEFAULT_ROOT, load_sessions
from adder.measure.session.horizon import DEFAULT_REMAINING, MIN_SAMPLES, Horizon
from adder.measure.spend.debt import decompose_read_cost

M = 1_000_000.0


@dataclass
class Claim:
    name: str
    ok: bool
    measured: str
    expected: str
    note: str = ""

    def line(self) -> str:
        return (f"  [{'PASS' if self.ok else 'FAIL'}] {self.name:<44}"
                f"{self.measured:>14}  expected {self.expected}")


def output_drives_context(sessions, min_turns: int = 30) -> Claim:
    """Claim: assistant output is essentially all of what fills a context.

    Measured ONLY on sessions that never compacted. A compacted session's net
    growth understates admission (content was removed), which inflates the ratio
    and makes the test meaningless.
    """
    growth = out = n = 0
    for s in sessions.values():
        # Main chain only. The step down into a subagent is a context drop that
        # is not a compaction, so the filter below excluded every session that
        # ever delegated -- silently, and in the direction that biases the
        # sample toward sessions with no delegation at all.
        main = s.main_turns
        if len(main) < min_turns:
            continue
        ctxs = [t.context for t in main]
        if any(ctxs[i] < ctxs[i - 1] for i in range(1, len(ctxs))):
            continue
        growth += ctxs[-1] - ctxs[0]
        out += sum(t.out for t in main[:-1])
        n += 1
    if not growth or n < 5:
        return Claim("output is the largest growth source", False, "insufficient data",
                     "0.35-0.75x", f"only {n} non-compacting sessions")
    r = out / growth
    # Expected range was 0.85-1.15x while the parser multi-counted every turn
    # with more than one content block, which inflated output ~1.78x without
    # touching context deltas (duplicate records carry an identical context).
    # Deduplicated, the ratio is ~0.50x: output is still the single largest
    # source of growth, but roughly half of it is read content, and a terseness
    # claim scaled to 1.0x over-claims about twofold.
    return Claim("output is the largest growth source", 0.35 <= r <= 0.75,
                 f"{r:.2f}x", "0.35-0.75x", f"{n} non-compacting sessions")


def input_side_dominates(sessions) -> Claim:
    inp = outp = 0.0
    for s in sessions.values():
        for t in s.turns:
            inp += t.input_cost()
            outp += t.output_cost()
    tot = inp + outp
    if not tot:
        return Claim("input-side dominates spend", False, "no data", ">=80%")
    share = inp / tot
    return Claim("input-side dominates spend", share >= 0.80,
                 f"{share:.0%}", ">=80%")


def debt_pool_is_addressable(sessions) -> Claim:
    total, base, acc = decompose_read_cost(sessions)
    if not total:
        return Claim("addressable pool >> baseline", False, "no data", ">=3x")
    ratio = acc / base if base else float("inf")
    return Claim("addressable pool >> baseline", ratio >= 3.0,
                 f"{ratio:.1f}x", ">=3x", f"${acc:,.0f} addressable / ${base:,.0f} fixed")


def sessions_are_long(sessions, min_median: int = 200) -> Claim:
    """Claim: the sessions that carry the SPEND are long enough for debt to matter.

    Cost-weighted on purpose. A plain median treats a 5-turn session the same as
    a 1,800-turn one, and short sessions are numerous but nearly free: measured
    here, sessions under 50 turns are ~47% of all sessions and 0.6% of spend.
    Using the plain median made this claim fail while 96% of the money sat in
    sessions well past the threshold -- the statistic was wrong, not the claim.
    """
    from adder.measure.window.prefix import weighted_median_turns

    rows = [(len(s_.main_turns), s_.cost) for s_ in sessions.values() if s_.turns]
    if not rows:
        return Claim("spend sits in long sessions", False, "no data",
                     f">={min_median} turns")
    total = sum(c for _, c in rows)
    if total <= 0:
        return Claim("spend sits in long sessions", False, "no cost",
                     f">={min_median} turns")
    # `prefix.weighted_median_turns`, not a second copy of it. This function had
    # its own transcription of the same statistic, and a claim that validates a
    # number should be computing that number the way the tool does -- a second
    # implementation is where the disagreements in this repo have come from.
    weighted_median = weighted_median_turns(sessions)
    share = sum(c for n, c in rows if n >= min_median) / total
    return Claim("spend sits in long sessions", weighted_median >= min_median,
                 f"{weighted_median:,} turns", f">={min_median} turns",
                 f"{share:.0%} of spend is in sessions >={min_median} turns; "
                 f"debt multiple is 1 + R/50")


def model_routing_is_marginal(sessions) -> Claim:
    """Claim: the thing people ask for is the smallest lever."""
    from adder.evaluate.claims.savings import model_routing

    total = sum(s.cost for s in sessions.values())
    if not total:
        return Claim("model routing is a minor lever", False, "no data", "<5% of spend")
    share = model_routing(sessions).saving / total
    return Claim("model routing is a minor lever", share < 0.05,
                 f"{share:.1%}", "<5% of spend")


def session_model_choice_is_not_marginal(sessions, min_share: float = 0.20) -> Claim:
    """Claim: choosing the model at turn 1 is a different lever from switching at turn 300.

    `model_routing_is_marginal` is about switching a warm conversation, and it
    holds: the prompt cache is model-scoped, so the switch rebuilds the prefix
    and the saving is under 5% of spend. That result was read for years as "the
    model does not matter here", which does not follow. A session that *starts*
    on the cheaper model never had a prefix on the expensive one, so there is no
    rebuild to pay for -- only a cheaper rate, applied from the first turn.

    This checks that the two really are different sizes of lever on this data.
    If it ever fails, the distinction is not worth drawing and `adder plan`
    should stop drawing it.
    """
    from adder.evaluate.replay.plan import Regime, replay

    base = replay(sessions, Regime())
    cheap = replay(sessions, Regime(session_model="claude-sonnet-5", session_rework=0.0))
    if not base.total:
        return Claim("starting cheap beats switching cheap", False, "no data",
                     f">={min_share:.0%} of spend")
    share = (base.total - cheap.total) / base.total
    return Claim("starting cheap beats switching cheap", share >= min_share,
                 f"{share:.1%}", f">={min_share:.0%} of spend",
                 "before any rework allowance; the capability cost is modelled, "
                 "not measured")


def replay_reproduces_measured_spend(sessions, tol: float = 0.05) -> Claim:
    """Claim: the `adder plan` replay, with no regime applied, is the real bill.

    Every multiple that command prints is a ratio against this baseline, so if
    the baseline drifts the multiples are decorative. Two ordering bugs were
    caught by exactly this check: admitting a turn's content after pricing it
    (under-priced every turn by one admission), and letting the simulated
    context keep climbing through a compaction the session actually took
    (over-priced the tail by 37%).
    """
    from adder.evaluate.replay.plan import Regime, replay

    measured = sum(s.cost for s in sessions.values())
    if not measured:
        return Claim("plan replay reproduces measured spend", False, "no data",
                     f"within {tol:.0%}")
    err = (replay(sessions, Regime()).total - measured) / measured
    return Claim("plan replay reproduces measured spend", abs(err) <= tol,
                 f"{err:+.1%}", f"within +/-{tol:.0%}")


def attribution_is_bounded(sessions) -> Claim:
    """The guard that caught three over-claiming bugs."""
    total, base, acc = decompose_read_cost(sessions)
    measured = sum(
        t.cache_read * t.rates().cache_read / M
        for s in sessions.values() for t in s.turns
    )
    ok = (base + acc) <= measured * 1.001 and abs((base + acc) - total) < 0.01
    return Claim("attribution never exceeds measured spend", ok,
                 f"${base + acc:,.0f}", f"<=${measured:,.0f}")


def horizon_is_calibrated(sessions) -> Claim:
    """Claim: enough local history to estimate remaining turns empirically.

    With fewer than MIN_SAMPLES sessions the estimator falls back to a flat prior
    calibrated to a long-session workload. On a short-session workload that prior
    over-estimates remaining turns and makes adder route too eagerly, so it
    is surfaced rather than applied silently.
    """
    h = Horizon.from_sessions(sessions)
    n = len(h.lengths)
    if n < MIN_SAMPLES:
        return Claim("horizon estimated from local data", False, f"{n} sessions",
                     f">={MIN_SAMPLES}",
                     f"falling back to a flat {DEFAULT_REMAINING}-turn prior "
                     "calibrated to long sessions; verify it fits this workload")
    return Claim("horizon estimated from local data", True, f"{n} sessions",
                 f">={MIN_SAMPLES}",
                 f"median remaining at turn 0 = {h.remaining(0):,}")


def countdown_would_underestimate(sessions) -> Claim:
    """Claim: the naive countdown is wrong, and wrong in the expensive direction."""
    h = Horizon.from_sessions(sessions)
    if len(h.lengths) < MIN_SAMPLES:
        return Claim("countdown estimator is unsafe", False, "no data", "n/a")
    worst = 0.0
    for n in (400, 600, 1000):
        cd, emp = h.countdown(n), h.remaining(n)
        if emp > 0:
            worst = max(worst, emp / cd if cd else float("inf"))
    return Claim("countdown estimator is unsafe", worst > 1.5,
                 "inf" if worst == float("inf") else f"{worst:.1f}x",
                 ">1.5x under",
                 "justifies using the survivor function instead of a countdown")


def composition_is_conservative(sessions) -> Claim:
    """Claim: the multiplicative composition model never overstates savings.

    The headline figure combines substitute levers as
    `pool * (1 - prod(1 - f_i))`. Re-simulating each session's real context
    trajectory under the same interventions shows whether that approximation
    over- or under-predicts. Under-prediction is safe; over-prediction inflates
    the headline and must be corrected.
    """
    from adder.evaluate.replay.simulate import Intervention, evaluate

    rows = evaluate(sessions, [
        Intervention(terseness=0.30),
        Intervention(delegation=0.25),
        Intervention(terseness=0.30, delegation=0.25, split_turns=300),
    ])
    worst = 0.0
    for _, sim, pred in rows:
        if sim > 0:
            worst = max(worst, (pred - sim) / sim)
    return Claim("composition model never overstates", worst <= 0.05,
                 f"{worst:+.0%}", "<=+5%",
                 "simulated trajectory vs multiplicative prediction; "
                 "negative means conservative")


def carry_multiplier_is_above_the_assumption(sessions) -> Claim:
    """Claim: the 0.10x re-read multiplier under-prices carrying context.

    `admitted_token_cost` prices every future re-read at the cache-read rate,
    which is only right when the prefix is warm on the turn that reads it. Turns
    miss -- TTL expiry, breakpoint lookback, fan-out races -- and a miss rewrites
    at 1.25x instead of reading at 0.10x. The realized multiplier is recoverable
    from the transcripts directly, so it is measured rather than assumed.

    The upper bound is a sanity check, not a target: a workload measuring above
    0.60x is barely caching at all, and the right response is to fix the caching
    rather than to trust any number this repo prints about placement.
    """
    from adder.measure.window.carry import _baseline_read_mult, measured_read_mult

    m = measured_read_mult(sessions)
    # The claim is "the realised multiplier exceeds the one the uncorrected
    # model assumed", and that assumption is per-provider. Stating it against
    # Anthropic's 0.10x on a workload that ran somewhere without a cache would
    # test a claim nobody made.
    rows = [x for x in sessions.values() if x.turns]
    base = _baseline_read_mult(rows) if rows else 0.10
    label = f"{base:.2f}x assumption"
    band = f"{base:.2f}-{max(0.60, base * 6):.2f}x"
    if not m:
        return Claim(f"carry multiplier exceeds the {label}", False, "no data", band)
    return Claim(f"carry multiplier exceeds the {label}",
                 base <= m <= max(0.60, base * 6), f"{m:.3f}x", band,
                 f"the old model under-prices carry by {m / max(1e-12, base):.2f}x")


def horizon_mean_exceeds_median(sessions) -> Claim:
    """Claim: remaining turns is skewed enough that the mean is the right one.

    Carry cost is linear in remaining turns, so its expectation is set by E[R].
    That only matters if E[R] and median(R) actually differ on this workload; if
    they do not, `mean_remaining` is machinery for nothing and the simpler
    estimator should be used. Measured here the mean runs above the median,
    which is what a heavy right tail does and why the median under-prices.
    """
    h = Horizon.from_sessions(sessions)
    if len(h.lengths) < MIN_SAMPLES:
        return Claim("horizon mean exceeds its median", False, "no data", ">=1.0x")
    med, mean = h.remaining(0), h.mean_remaining(0)
    if med <= 0:
        return Claim("horizon mean exceeds its median", False, "no median", ">=1.0x")
    return Claim("horizon mean exceeds its median", mean >= med,
                 f"{mean / med:.2f}x", ">=1.0x",
                 f"mean {mean:,.0f} vs median {med:,} remaining at turn 0")


def emitted_advice_clears_its_own_overhead(sessions) -> Claim:
    """Claim: every recommendation adder emits is cheaper than not taking it.

    The whole point of the tool. Swept rather than argued: the router is run
    across a grid of context sizes, horizons and read sizes, and any plan that
    is not `inline` has to save more than the routing turn that produced it.
    A single counterexample means adder is, for that case, a more expensive way
    to work than not having it -- which is the one failure this repo cannot
    tolerate, because nobody audits the recommendations that quietly cost money.

    Independent of the transcripts on purpose. It is a property of the decision
    rule, so it should hold on a machine with no history at all.
    """
    from adder.decide.route.policy import decide

    tasks = ("what does prices.py do",
             "make the ingest step tolerate a partial batch",
             "refactor the storage layer across the codebase")
    bad: list[str] = []
    n = 0
    for task in tasks:
        for ctx in (10_000, 100_000, 500_000, 900_000):
            for rem in (0, 3, 40, 200, 800):
                for read in (500, 8_000, 60_000, 400_000):
                    n += 1
                    p = decide(task, context_tokens=ctx, remaining_turns=rem,
                               est_read_tokens=read)
                    if p.action != "inline" and p.saving <= p.overhead:
                        bad.append(f"{task[:20]}/{ctx}/{rem}/{read}")
    return Claim("emitted advice clears its own overhead", not bad,
                 f"{len(bad)}/{n} bad", "0 counterexamples",
                 "" if not bad else f"first: {bad[0]}")


def a_prior_never_buys_a_downgrade(sessions) -> Claim:
    """Claim: the router only routes below the classifier when the log says it may.

    This is the safety property of the whole right-sizing change, and it is the
    one worth sweeping, because the failure is silent. Under a no-evidence prior
    the cheapest rung always has the lowest expected cost -- the arithmetic
    genuinely says Haiku -- so a router that minimises expected cost without a
    permission gate will route real work to the cheapest model it can hold and
    report a saving while doing it. Nothing in the output would look wrong.

    So: whenever the chosen tier is below the tier the classifier asked for, the
    outcome log has to have been informative at that tier. On a machine with an
    empty log that reduces to "never de-escalates", which is the correct
    behaviour on day one and the thing a future refactor is most likely to lose.

    Independent of the transcripts on purpose: it is a property of the decision
    rule, so it must hold on a machine with no history at all.
    """
    from adder.decide.route.classify import classify
    from adder.decide.route.policy import decide
    from adder.decide.track.outcomes import evidence

    tasks = ("what does prices.py do",
             "make the ingest step tolerate a partial batch",
             "refactor the storage layer across the codebase",
             "add a test for the retry helper in adder/decide/route/select.py")
    bad: list[str] = []
    n = descended = 0
    for task in tasks:
        asked = classify(task).tier
        for ctx in (10_000, 100_000, 500_000):
            for rem in (0, 40, 400):
                for read in (500, 20_000, 200_000):
                    n += 1
                    p = decide(task, context_tokens=ctx, remaining_turns=rem,
                               est_read_tokens=read)
                    if p.tier >= asked:
                        continue
                    descended += 1
                    if not evidence(p.tier.name, None).informative:
                        bad.append(f"{task[:20]}/{ctx}/{rem}/{read} -> {p.tier.name}")
    return Claim("a prior never buys a downgrade", not bad,
                 f"{len(bad)}/{n} bad", "0 counterexamples",
                 f"{descended} of {n} cases routed below the classifier"
                 + (f"; first bad: {bad[0]}" if bad else ""))


def the_target_reduction_is_reachable(sessions, target: float = 10.0) -> Claim:
    """Claim: some followable regime gets this workload to `target` x cheaper.

    Checked against the frontier -- every lever at the end of its range -- rather
    than by searching, because the frontier is a bound: if it does not reach the
    target, nothing in `plan.GRID` can, and saying so costs one replay instead
    of four hundred.

    Workload-dependent, and expected to fail on some workloads. A session that
    never gets long has little carry to remove, and the honest answer there is
    that a 10x is not available, not that the tool should look harder. That is
    the same caveat `sessions_are_long` carries.
    """
    from adder.evaluate.replay.plan import Regime, frontier, replay

    base = replay(sessions, Regime())
    if not base.total:
        return Claim(f"a regime exists that reaches {target:.0f}x", False,
                     "no data", f">={target:.0f}x")
    edge = replay(sessions, frontier())
    got = base.total / edge.total if edge.total else float("inf")
    return Claim(f"a regime exists that reaches {target:.0f}x", got >= target,
                 f"{got:.1f}x", f">={target:.0f}x",
                 "frontier regime; it is a bound, not a recommendation")


def the_tool_has_paid_for_itself(sessions) -> Claim:
    """Claim: cumulative guaranteed saving covers cumulative routing overhead.

    `cost_with_adder = baseline - savings + overhead`, so the tool is cheaper
    than not having it exactly when savings cover overhead. The ledger records
    both sides; this asserts the invariant over whatever it holds. An empty
    ledger passes, because a tool that has never been asked anything has never
    charged for an answer.
    """
    from adder.decide.track.ledger import current

    led = current()
    if not led.accepted:
        return Claim("advice has been worth more than the asking", True,
                     "nothing spent", "banked >= spent",
                     "no recommendations recorded; the invariant holds trivially")
    return Claim("advice has been worth more than the asking", led.solvent,
                 f"${led.margin:,.2f}", "banked >= spent", led.describe())


def an_opening_is_mostly_a_cache_read(sessions, min_share: float = 0.40) -> Claim:
    """Claim: restarting a session re-reads the shared prefix, it does not rebuild it.

    This is the claim the restart cadence rests on, and the cadence is now the
    largest single lever `adder plan` recommends. If openings on this machine
    are cold, restarting costs a full prefix write every time, the closed form
    `k* = sqrt(2W/(m*r*g))` returns a much longer cycle, and the regime that
    reaches a 10x target is a different and harsher one.

    Measured on openings that followed a turn within the 5m TTL, which is the
    condition a restart regime creates. Openings after longer gaps also measure
    warm here and are deliberately excluded: that has no explanation a TTL
    supports, and a claim resting on an unexplained observation is not a claim.

    The threshold is 40% rather than the ~74% measured, because what the
    conclusion needs is that a restart is materially cheaper than a rebuild, not
    that it is exactly this cheap.
    """
    from adder.measure.window.prefix import measure

    op = measure(sessions)
    if not op.measured:
        return Claim("a session opening is mostly a cache read", False,
                     "no openings", f">={min_share:.0%} from cache",
                     "no session opened within the TTL of another turn")
    return Claim("a session opening is mostly a cache read",
                 op.warm_share >= min_share, f"{op.warm_share:.0%}",
                 f">={min_share:.0%} from cache",
                 f"{op.openings} openings; a restart is "
                 f"{op.discount('claude-opus-5'):.1f}x cheaper than the rebuild "
                 f"the split lever used to assume")


def restarting_beats_running_long(sessions) -> Claim:
    """Claim: at the measured restart price, the optimal cycle is shorter than
    the sessions this workload actually runs.

    The two halves of the recommendation have to hold together. `k*` being small
    is not interesting if sessions are already that short -- there would be
    nothing to change. What makes the lever real is the gap between the cadence
    the arithmetic wants and the cadence the transcripts show.
    """
    from adder.measure.window.carry import Carry
    from adder.measure.window.prefix import cadence, measure, weighted_median_turns

    op = measure(sessions)
    observed = weighted_median_turns(sessions)
    if not op.measured or not observed:
        return Claim("restarting beats running long", False, "no data", "k* < as-run")
    c = Carry.measure(sessions)
    k, at_k, never = cadence(op, model="claude-opus-5", growth=max(1.0, c.growth),
                             read_mult=c.read_mult, observed_turns=observed)
    ratio = never / at_k if at_k else 0.0
    return Claim("restarting beats running long", k < observed and ratio > 1.0,
                 f"{k:,} vs {observed:,}", "k* < as-run",
                 f"{ratio:.1f}x cheaper per turn on the input side")


_BENCH_MEMO: tuple[object, object] | None = None


def _bench(sessions):
    """`bench.run` for this session map, computed at most once.

    Two claims need the same answer and it is the most expensive computation in
    this file by two orders of magnitude: 250s, of which `corner_sweep` is 206s
    because the worst-case corner is evaluated by replaying the whole workload
    at each vertex. Running it twice made `adder validate` -- the command whose
    entire job is to let someone re-check the foundations -- take ten minutes.

    Keyed on object identity, and the map is held so the identity cannot be
    reused by a later allocation.
    """
    global _BENCH_MEMO
    if _BENCH_MEMO is None or _BENCH_MEMO[0] is not sessions:
        from adder.evaluate.replay.bench import run as bench_run

        _BENCH_MEMO = (sessions, bench_run(sessions))
    return _BENCH_MEMO[1]


def installing_adder_pays_by_itself(sessions, min_mult: float = 1.3) -> Claim:
    """Claim: the mechanisms that act without being obeyed are worth having alone.

    Two things here can save money with nobody following any advice: the
    PreToolUse guard, which prices a read before it lands, and the tier files,
    which decide what a delegated step runs on. Everything else is a report, and
    a report saves nothing until someone acts on it.

    This is deliberately the *small* number. If it ever fails, the honest
    consequence is not to look for a better lever -- it is that the README's
    "install it and change nothing" line has to come down, because the only
    remaining case for the tool would be one that asks the reader to work
    differently first.
    """
    b = _bench(sessions)
    if not b.baseline:
        return Claim("installing it pays before you obey it", False, "no data",
                     f">={min_mult:.1f}x")
    return Claim("installing it pays before you obey it", b.installed >= min_mult,
                 f"{b.installed:.1f}x", f">={min_mult:.1f}x",
                 "guard at its shipped defaults plus the agent tiers; no behaviour change")


def the_benchmark_headline_holds(sessions, min_mult: float = 5.0) -> Claim:
    """Claim: following the threshold and cadence the reports solve reaches 5x.

    This is the number quoted in the README, so it is re-measured here rather
    than remembered. It is quoted at the nominal assumptions and the note
    carries the worst corner of the sweep, because the two are far apart and
    printing only the first is how a benchmark becomes marketing.

    Workload-dependent, and expected to fail on some workloads for the same
    reason `sessions_are_long` is: a workload whose sessions stay short has
    little carry to remove, and 5x is not available there at any threshold.
    """
    b = _bench(sessions)
    if not b.baseline:
        return Claim(f"the advice reaches {min_mult:.0f}x", False, "no data",
                     f">={min_mult:.0f}x")
    return Claim(f"the advice reaches {min_mult:.0f}x", b.followed >= min_mult,
                 f"{b.followed:.1f}x", f">={min_mult:.0f}x",
                 f"nominal; worst corner of the modelled inputs is {b.worst_corner:.1f}x")


def the_guard_never_speaks_at_a_loss(sessions) -> Claim:
    """Claim: every fire the guard emits is worth more than saying it costs.

    The guard injects text into the context, and injected text is carried for
    the rest of the session like any other token. Before this was priced, the
    guard fired 903 times on this machine at a median real result size of 143
    tokens -- it was spending carry to warn about reads that were never going
    to be expensive.

    Swept rather than asserted, because the gate is a comparison between two
    modelled costs and either could move: 240 combinations of size, horizon and
    model, and a single fire with non-positive expected value fails the claim.
    """
    from adder.core.shapes import SizeModel
    from adder.decide.guard import Settings, decide

    cfg = Settings()
    worst = None
    fired = 0
    for p90 in (2_000, 5_000, 20_000, 60_000, 150_000):
        sizes = SizeModel(shapes={"cat": (p90 // 4, p90, 40)},
                          heads={"cat": (p90 // 4, p90, 40)}, built=1.0, calls=40)
        for remaining in (5, 25, 100, 300, 800, 2_000):
            for model in ("claude-opus-5", "claude-sonnet-5",
                          "claude-haiku-4-5", "claude-opus-5[1m]"):
                for taken in (0.25, 0.5):
                    v = decide("Bash", {"command": "cat f.py"}, model=model,
                               remaining_turns=remaining, sizes=sizes,
                               cfg=Settings(advice_taken=taken,
                                            min_cost=cfg.min_cost))
                    if not v.fire:
                        continue
                    fired += 1
                    if worst is None or v.net < worst:
                        worst = v.net
    return Claim(
        "a guard fire always clears the cost of saying it",
        ok=fired > 0 and (worst or 0) > 0,
        measured=f"{worst:+.4f} worst" if worst is not None else "no fires",
        expected="> $0 over 240 cases",
        note=f"{fired} of 240 swept cases fire; the rest are declined for stated reasons",
    )


def the_guard_predicts_sizes_it_used_to_assume(sessions, root=None) -> Claim:
    """Claim: learned result sizes beat the constant they replaced, on holdout.

    The guard assumed 15,000 tokens for any command matching `cat ` or
    `git log`. Held out against local transcripts -- even-numbered calls train,
    odd-numbered calls test -- the learned model's median absolute error is two
    orders of magnitude smaller. This is the claim that would fail first if the
    shape key were made so specific that every shape became a singleton, which
    is exactly what a regex splitter did to it once.
    """
    from adder.core.shapes import SizeModel, iter_results, segments, shape

    OLD_CONSTANT = 15_000
    train_shapes: dict[str, list[int]] = {}
    train_heads: dict[str, list[int]] = {}
    test: list[tuple[str, int]] = []
    for i, (tool, inp, size) in enumerate(iter_results(root or DEFAULT_ROOT)):
        if tool != "Bash":
            continue
        cmd = (inp or {}).get("command") or ""
        if i % 2:
            test.append((cmd, size))
            continue
        train_shapes.setdefault(shape(cmd), []).append(size)
        segs = segments(cmd)
        if segs:
            train_heads.setdefault(segs[-1][0], []).append(size)

    if len(test) < 200:
        return Claim("learned sizes beat the assumed constant", ok=True,
                     measured="n/a", expected="10x lower error",
                     note="too few local Bash calls to hold out; run on a machine "
                          "with transcripts")

    def quantiles(d):
        out = {}
        for k, xs in d.items():
            xs.sort()
            mid = xs[len(xs) // 2]
            p90 = xs[min(len(xs) - 1, int(0.9 * (len(xs) - 1)))]
            out[k] = (mid, p90, len(xs))
        return out

    model = SizeModel(shapes=quantiles(train_shapes), heads=quantiles(train_heads),
                      built=1.0, calls=len(test))
    err_model = sorted(abs(model.predict_command(c).p50 - n) for c, n in test)
    err_const = sorted(abs(OLD_CONSTANT - n) for _, n in test)
    med_model = err_model[len(err_model) // 2]
    med_const = err_const[len(err_const) // 2]
    ratio = med_const / max(med_model, 1)
    return Claim(
        "learned sizes beat the assumed constant",
        ok=ratio >= 10.0,
        measured=f"{ratio:,.0f}x lower",
        expected=">= 10x lower median error",
        note=f"median |error|: {med_model:,} tok learned vs {med_const:,} tok "
             f"assumed, over {len(test):,} held-out calls",
    )


def memory_is_carried_not_written(sessions, min_ratio: float = 5.0) -> Claim:
    """Claim: an instruction-file token costs far more to carry than to install.

    Everything `adder memory` recommends rests on this. If the floor were paid
    once per session, a 4,000-token `CLAUDE.md` would be a rounding error and
    trimming it would be fussiness. It is paid once per *turn*, so the ratio
    between the carry term and the opening term is the whole argument, and it
    is measurable from the session lengths and cache behaviour on record.

    The threshold is 5x rather than the several-hundred-fold this machine
    measures, because the conclusion needs "carry dominates", not a specific
    multiple that moves with the length distribution.
    """
    from adder.measure.window.memory import Pricing

    p = Pricing.measure(sessions)
    if not p.measured:
        return Claim("memory is carried, not written", False, "no sessions",
                     f">={min_ratio:.0f}x carry", "no local sessions to fit")
    open_cost, carry = p.open_cost(1_000), p.carry_cost(1_000)
    ratio = carry / open_cost if open_cost else 0.0
    return Claim("memory is carried, not written", ratio >= min_ratio,
                 f"{ratio:,.0f}x", f">={min_ratio:.0f}x carry",
                 f"1,000 resident tokens cost ${p.session_cost(1_000):.3f} per "
                 f"session over {p.turns:.0f} turns")


def compaction_keeps_less_than_assumed(sessions, assumed: float = 0.35) -> Claim:
    """Claim: the 35% survival this repo prices compaction with is conservative.

    `live.compaction_net` and `compact.breakeven_remaining` both assume a
    compaction keeps 35% of the context, and both would over-state the rebuild
    -- and so under-recommend compacting -- if real compactions kept less. They
    do: the auto-compactions on record keep single-digit percentages. The claim
    passes when the assumption is on the safe side of the measurement, which is
    the direction a cost tool is allowed to be wrong in.
    """
    from adder.measure.window.compact import analyse

    rep = analyse(sessions)
    if not rep.events:
        return Claim("compaction keeps less than assumed", False, "no events",
                     f"<={assumed:.0%} kept", "no compactions on record")
    kept = rep.mean_kept()
    return Claim("compaction keeps less than assumed", kept <= assumed,
                 f"{kept:.0%}", f"<={assumed:.0%} kept",
                 f"{rep.n} compactions; the modelled rebuild is "
                 f"{assumed / max(kept, 1e-9):.0f}x what was really written back")


def a_brief_can_cross_a_restart(sessions, min_tokens: int = 2_000) -> Claim:
    """Claim: restarting does not require throwing the session's state away.

    The restart lever is only followable if something may be carried, and the
    objection to it has always been "I would lose the context". The brief
    budget answers it with a number: at the median context and horizon on
    record, this is how much may cross and still leave the restart ahead. If
    that came out below a couple of thousand tokens the honest advice would be
    "do not restart", and `plan` would have to stop recommending it.
    """
    from adder.decide.handoff import max_handoff
    from adder.measure.session.horizon import load as load_horizon
    from adder.measure.window.carry import Carry
    from adder.measure.window.prefix import Opening, weighted_median_turns
    from adder.util.stats import median

    peaks = [s.peak_context for s in sessions.values() if s.turns]
    if not peaks:
        return Claim("a brief can cross a restart", False, "no sessions",
                     f">={min_tokens:,} tok", "no local sessions")
    context = int(median(peaks))
    turns = weighted_median_turns(sessions)
    c = Carry.measure(sessions)
    op = Opening.default()
    for s in sessions.values():
        if s.turns:
            op = Opening.from_session(s)
            break
    load_horizon()          # keeps the horizon fit warm for the note below
    budget = max_handoff(context=context, remaining=turns, model="claude-opus-5",
                         read_mult=c.read_mult, opening=op)
    return Claim("a brief can cross a restart", budget >= min_tokens,
                 f"{budget:,} tok", f">={min_tokens:,} tok",
                 f"at the median {context:,}-token context with {turns:,} turns "
                 "left, that much may be carried and the restart still pays")


def the_aggregate_beats_the_tail(sessions, root=None) -> Claim:
    """Claim: repeated small commands admit more than the big ones do.

    This is the claim that justifies the guard having an aggregate rule at all.
    If the largest cumulative shapes were also the ones with the largest single
    results, the per-call gate would already catch them and the extra state
    would be waste. They are not: the biggest channel on the author's machine
    is a *bounded* read repeated a few hundred times, every instance of which
    the guard is right to wave through.

    Expressed as a share so it survives a different workload: the shapes whose
    session totals clear the aggregate threshold should hold a materially
    larger share of result tokens than the calls the per-call gate would fire
    on. A machine that only ever makes a few big calls will fail this, and
    should -- there the aggregate rule is not earning its state.
    """
    from collections import defaultdict

    from adder.core.shapes import iter_results, shape
    from adder.decide.guard import AGGREGATE_TOKENS

    per: dict[tuple[str, str], int] = defaultdict(int)
    total = 0
    big_single = 0
    for tool, inp, size in iter_results(root or DEFAULT_ROOT):
        if tool != "Bash":
            continue
        total += size
        # `iter_results` does not carry the session id, so the key is the shape
        # alone; that under-counts nothing and over-counts nothing that matters,
        # because a shape's total is what the rule reads.
        per[("", shape((inp or {}).get("command") or ""))] += size
        if size >= 5_000:
            big_single += size

    if total < 100_000:
        return Claim("repeated small commands out-admit the big ones", ok=True,
                     measured="n/a", expected="aggregate > tail",
                     note="too little local Bash volume to compare")

    crossed = sum(v for v in per.values() if v >= AGGREGATE_TOKENS)
    return Claim(
        "repeated small commands out-admit the big ones",
        ok=crossed > big_single,
        measured=f"{crossed / total:.0%} vs {big_single / total:.0%}",
        expected="cumulative share > single-call share",
        note=f"{sum(1 for v in per.values() if v >= AGGREGATE_TOKENS)} shapes clear "
             f"{AGGREGATE_TOKENS:,} tok cumulatively; calls over 5,000 tok are the "
             f"ones a per-call gate sees",
    )


def the_guard_is_worth_more_than_it_costs(sessions, root=None) -> Claim:
    """Claim: the one mechanism that runs unattended pays for itself here.

    Every other claim in this file is about a report, and a report costs
    nothing until someone reads it. The guard is different: it speaks without
    being asked, and every sentence it injects is admitted to the context and
    carried for the rest of the session. So it is the one component that can
    make this tool a *more expensive* way to work, and the only honest check is
    to replay it over transcripts that have already been paid for.

    An upper bound on the saving and an exact figure on the cost, which is the
    conservative direction: the horizon is the one the guard would have
    projected rather than the turns that really remained, and the saving is
    already discounted by the assumed uptake. If this ever fails, the guard
    should be off by default.
    """
    from adder.decide.guard import replay as guard_replay
    from adder.decide.guard import uptake as guard_uptake

    # Prefer the measured uptake to the assumed one the moment there is enough
    # of it. The claim gets stronger or weaker on evidence rather than staying
    # anchored to a default nobody has checked.
    # The root this run was given, not the default. Every session-based claim
    # here reads the corpus it was pointed at; the transcript-scanning ones
    # silently read `~/.claude/projects`, so `adder validate <a-corpus>` tested
    # half its claims against one workload and half against another.
    u = guard_uptake(root)
    r = guard_replay(root, advice_taken=u.rate if u.measured else None)
    if not r.calls:
        return Claim("the guard pays for the advice it gives", ok=True,
                     measured="n/a", expected="saving > advice cost",
                     note="no local tool calls to replay")
    if not r.fires:
        return Claim("the guard pays for the advice it gives", ok=True,
                     measured="never speaks", expected="saving > advice cost",
                     note=f"{r.calls:,} calls replayed, nothing cleared the gates")
    ratio = r.saving / r.overhead if r.overhead else float("inf")
    return Claim(
        "the guard pays for the advice it gives",
        ok=r.net > 0,
        measured=f"{ratio:,.0f}x",
        expected="saving > advice cost",
        note=f"{r.fires:,} findings in {r.calls:,} calls ({r.fire_rate:.2%}); "
             f"${r.saving:,.0f} modelled against ${r.overhead:,.2f} of injected "
             f"advice at "
             + (f"a measured {u.rate:.0%} uptake" if u.measured
                else "the assumed uptake")
             + "; a bound, not a saving to bank",
    )


CHECKS = (
    output_drives_context,
    input_side_dominates,
    debt_pool_is_addressable,
    sessions_are_long,
    model_routing_is_marginal,
    attribution_is_bounded,
    horizon_is_calibrated,
    countdown_would_underestimate,
    composition_is_conservative,
    session_model_choice_is_not_marginal,
    replay_reproduces_measured_spend,
    carry_multiplier_is_above_the_assumption,
    an_opening_is_mostly_a_cache_read,
    restarting_beats_running_long,
    horizon_mean_exceeds_median,
    emitted_advice_clears_its_own_overhead,
    a_prior_never_buys_a_downgrade,
    the_target_reduction_is_reachable,
    installing_adder_pays_by_itself,
    the_benchmark_headline_holds,
    the_tool_has_paid_for_itself,
    the_guard_never_speaks_at_a_loss,
    the_guard_predicts_sizes_it_used_to_assume,
    memory_is_carried_not_written,
    compaction_keeps_less_than_assumed,
    a_brief_can_cross_a_restart,
    the_aggregate_beats_the_tail,
    the_guard_is_worth_more_than_it_costs,
)


def run(root: Path | str = DEFAULT_ROOT) -> list[Claim]:
    """Every claim, all of them against the same corpus.

    A claim that scans transcripts takes `root`; one that only needs the
    already-loaded sessions does not. Passing it by inspection rather than by
    convention so a new claim cannot quietly read a different directory from
    the rest of the report.
    """
    import inspect

    sessions = load_sessions(root)
    out = []
    for c in CHECKS:
        takes_root = "root" in inspect.signature(c).parameters
        out.append(c(sessions, root) if takes_root else c(sessions))
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="adder validate",
                                 description="Re-test the claims this repo depends on.")
    ap.add_argument("root", nargs="?", default=None,
                    help="transcript directory (default: the `root` setting)")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    a = ap.parse_args(argv)
    # `root_of`: the argument if one was given, else the `root`
    # setting. Resolved here so two commands cannot disagree
    # about which transcript directory `adder config` names.
    a.root = str(_root_of(a))

    claims = run(a.root)
    if a.json:
        import json

        failed = [c for c in claims if not c.ok]
        print(json.dumps({
            "ok": not failed,
            "passed": len(claims) - len(failed),
            "failed": len(failed),
            "claims": [{"name": c.name, "ok": c.ok, "measured": c.measured,
                        "expected": c.expected, "note": c.note} for c in claims],
        }))
        return 1 if failed else 0
    print("\n  Foundational claims, re-measured against local transcripts\n")
    for c in claims:
        print(c.line())
        if c.note:
            print(f"         {c.note}")
    failed = [c for c in claims if not c.ok]
    print()
    if failed:
        print(f"  {len(failed)} claim(s) no longer hold. The savings estimates in this")
        print("  repo are calibrated to this workload and may not apply here.\n")
        return 1
    print("  All claims hold on this data.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
