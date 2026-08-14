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

from .debt import decompose_read_cost
from .horizon import DEFAULT_REMAINING, MIN_SAMPLES, Horizon
from .prices import CACHE_READ_MULT, rate
from .trace import DEFAULT_ROOT, load_sessions

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
        if len(s.turns) < min_turns:
            continue
        ctxs = [t.context for t in s.turns]
        if any(ctxs[i] < ctxs[i - 1] for i in range(1, len(ctxs))):
            continue
        growth += ctxs[-1] - ctxs[0]
        out += sum(t.out for t in s.turns[:-1])
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
    lens = sorted(len(s.turns) for s in sessions.values())
    if not lens:
        return Claim("sessions long enough for debt to matter", False, "no data",
                     f">={min_median} turns")
    med = lens[len(lens) // 2]
    return Claim("sessions long enough for debt to matter", med >= min_median,
                 f"{med:,} turns", f">={min_median} turns",
                 "debt multiple is 1 + R/50, so short sessions carry little")


def model_routing_is_marginal(sessions) -> Claim:
    """Claim: the thing people ask for is the smallest lever."""
    from .savings import model_routing

    total = sum(s.cost for s in sessions.values())
    if not total:
        return Claim("model routing is a minor lever", False, "no data", "<5% of spend")
    share = model_routing(sessions).saving / total
    return Claim("model routing is a minor lever", share < 0.05,
                 f"{share:.1%}", "<5% of spend")


def attribution_is_bounded(sessions) -> Claim:
    """The guard that caught three over-claiming bugs."""
    total, base, acc = decompose_read_cost(sessions)
    measured = sum(
        t.cache_read * rate(t.model).inp * CACHE_READ_MULT / M
        for s in sessions.values() for t in s.turns
    )
    ok = (base + acc) <= measured * 1.001 and abs((base + acc) - total) < 0.01
    return Claim("attribution never exceeds measured spend", ok,
                 f"${base + acc:,.0f}", f"<=${measured:,.0f}")


def horizon_is_calibrated(sessions) -> Claim:
    """Claim: enough local history to estimate remaining turns empirically.

    With fewer than MIN_SAMPLES sessions the estimator falls back to a flat prior
    calibrated to a long-session workload. On a short-session workload that prior
    over-estimates remaining turns and makes the router route too eagerly, so it
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
    from .simulate import Intervention, evaluate

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
)


def run(root: Path | str = DEFAULT_ROOT) -> list[Claim]:
    sessions = load_sessions(root)
    return [c(sessions) for c in CHECKS]


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="router.validate",
                                 description="Re-test the claims this repo depends on.")
    ap.add_argument("root", nargs="?", default=str(DEFAULT_ROOT))
    a = ap.parse_args(argv)

    claims = run(a.root)
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
