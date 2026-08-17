"""Does how long a session has run predict how much longer it will run?

The assumption under every attained-service scheduler
------------------------------------------------------
Serving systems that treat an agent program as the scheduling unit -- rather
than scheduling its individual model calls -- prioritise by **attained
service**: how much work a program has already consumed. Programs that have
consumed little go first, on the theory that they will finish soon and get out
of the way, while programs that have already consumed a lot are the ones likely
to keep consuming.

That theory is an empirical claim about the *conditional* distribution of work,
and it is true or false per workload:

* **Heavy-tailed.** The longer a session has run, the more it has left. Attained
  service is informative, prioritising by it works, and the tail is where the
  money is.
* **Memoryless.** Remaining work is independent of work done. Attained service
  carries no signal at all, and any policy built on it is sorting noise.
* **Bounded.** The longer it has run, the closer it is to finishing. Prioritise
  the *old* programs, which is the opposite of the usual rule.

Nothing in this repo had checked which of the three this workload is, and every
"restart the session" recommendation quietly assumed the first. This module
measures it.

Why it matters here rather than in a queue
-------------------------------------------
There is no queue on a laptop. One person runs one session at a time, so the
latency argument for program-aware scheduling does not apply and this module
does not make it.

What does apply is the **cost** version of the same question. A session's cost
per turn rises with its context, so the decision "carry on, or hand off to a
fresh session" is exactly a question about remaining service: carrying on is
right if the session is nearly done, and wrong if it has another four hundred
turns in it. `adder prefix` prices the restart; `adder horizon` estimates the
remaining turns. Neither asks whether attained service is a usable predictor at
all, which is the thing that decides whether either number can be trusted.

What is measured
----------------
For every session and every point in it, the pair (turns attained, turns
remaining). Then:

* `E[remaining | attained >= x]` as a curve. Flat means memoryless; rising
  means heavy-tailed; falling means bounded.
* The slope of that curve as one number, with an interval.
* The share of total spend sitting in sessions past each threshold, because a
  predictor that is right about sessions holding 2% of spend is not a lever.

Two ways to get this wrong, both of which this module did first
---------------------------------------------------------------
**Correlating attained against remaining over pooled positions.** Inside one
session those two sum to a constant, so they are mechanically anti-correlated:
a 600-turn session contributes positions running from (1, 599) to (600, 0). On
a workload built to be heavy-tailed that statistic came out at -0.75 and would
have been reported as "bounded -- leave long sessions alone", the exact inverse
of the right advice. Comparing *across thresholds* compares different sessions,
which is the comparison the question is actually about.

**Resampling positions instead of sessions.** A single 600-turn session
contributes 600 agreeing points; treating them as independent narrows the
interval by roughly a factor of 25. The session is the unit that was
independently drawn, so the session is the unit that gets resampled.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

from adder.util import render
from adder.util.stats import DEFAULT_SEED, mean, quantile, share

# Attained-service thresholds, in turns. Log-ish spacing because the question
# is about the tail, and a linear grid spends most of its rows on the head.
THRESHOLDS: tuple[int, ...] = (1, 5, 10, 25, 50, 100, 200, 400, 800)

# Below this many sessions past a threshold, the row is reported but does not
# inform the verdict: three sessions can produce any slope you like.
MIN_SESSIONS = 5


@dataclass(frozen=True)
class Point:
    """One position inside one session."""

    session: str
    attained: int          # turns completed so far
    remaining: int         # turns still to come
    cost_so_far: float
    cost_remaining: float


def points(sessions: dict) -> list[Point]:
    """Every (attained, remaining) pair in the workload.

    Every position in every session, not one per session. A single sample per
    session would answer "how long are sessions", which is a different and much
    less useful question than "given that it has run this long, what is left".
    """
    out: list[Point] = []
    for sid, s in sessions.items():
        turns = s.turns
        n = len(turns)
        if n < 2:
            continue
        costs = [t.cost() for t in turns]
        total = sum(costs)
        running = 0.0
        for i in range(n):
            running += costs[i]
            out.append(Point(sid, attained=i + 1, remaining=n - i - 1,
                             cost_so_far=running, cost_remaining=total - running))
    return out


@dataclass
class Row:
    """One attained-service threshold, and what is left beyond it."""

    threshold: int
    sessions: int = 0
    points: int = 0
    mean_remaining: float = 0.0
    p90_remaining: float = 0.0
    mean_cost_remaining: float = 0.0
    spend_share: float = 0.0

    @property
    def thin(self) -> bool:
        return self.sessions < MIN_SESSIONS


def curve(sessions: dict, pts: list[Point] | None = None,
          thresholds: tuple[int, ...] = THRESHOLDS) -> list[Row]:
    """`E[remaining | attained >= x]` for each threshold."""
    pts = points(sessions) if pts is None else pts
    total_spend = sum(s.cost for s in sessions.values()) or 0.0
    rows: list[Row] = []
    for x in thresholds:
        past = [p for p in pts if p.attained >= x]
        if not past:
            rows.append(Row(x))
            continue
        ids = {p.session for p in past}
        rows.append(Row(
            threshold=x,
            sessions=len(ids),
            points=len(past),
            mean_remaining=mean([float(p.remaining) for p in past]),
            p90_remaining=quantile([float(p.remaining) for p in past], 0.9),
            mean_cost_remaining=mean([p.cost_remaining for p in past]),
            spend_share=share(
                sum(s.cost for sid, s in sessions.items() if sid in ids), total_spend),
        ))
    return rows


def lengths_of(sessions: dict) -> list[int]:
    """Turn count per session, which is all the slope needs.

    The conditional mean has a closed form in the lengths alone. A session of
    length `n` contributes `n - x + 1` positions at threshold `x`, with
    remaining values `n - x` down to `0`, summing to `(n-x)(n-x+1)/2`. So the
    whole curve is computable from a list of integers rather than from every
    position in the workload.

    That is not a micro-optimisation. The interval resamples the workload two
    hundred times, and rebuilding every position each time is O(resamples x
    turns) -- minutes on a real transcript directory, for a number that takes
    milliseconds this way.
    """
    return [len(s.turns) for s in sessions.values() if len(s.turns) >= 2]


def _mean_remaining(lengths: list[int], x: int) -> tuple[int, float]:
    """`(sessions surviving to x, mean remaining turns past x)`, in closed form."""
    survivors = 0
    positions = 0
    total = 0.0
    for n in lengths:
        if n < x:
            continue
        survivors += 1
        m = n - x
        positions += m + 1
        total += m * (m + 1) / 2.0
    return survivors, (total / positions if positions else 0.0)


def slope_from_lengths(lengths: list[int],
                       thresholds: tuple[int, ...] = THRESHOLDS,
                       floor: float = 0.25) -> float:
    """The turns-per-turn slope, computed from session lengths alone."""
    need = max(MIN_SESSIONS, int(len(lengths) * floor))
    xs: list[float] = []
    ys: list[float] = []
    for x in thresholds:
        survivors, mean_rem = _mean_remaining(lengths, x)
        if survivors >= need:
            xs.append(float(x))
            ys.append(mean_rem)
    if len(xs) < 3:
        return 0.0
    mx, my = mean(xs), mean(ys)
    denom = sum((v - mx) ** 2 for v in xs)
    if denom <= 0:
        return 0.0
    return sum((v - mx) * (y - my) for v, y in zip(xs, ys, strict=True)) / denom


def _covered(rows: list[Row], n_sessions: int, floor: float = 0.25) -> list[Row]:
    """Rows where enough sessions survive for the mean to mean anything.

    The deepest thresholds are dominated by truncation rather than by the shape
    of the distribution: past the point where only a handful of sessions remain,
    the conditional mean is just those sessions running out, and every workload
    looks bounded there. Requiring a quarter of the population keeps the
    comparison on the part of the curve that is about the distribution.
    """
    need = max(MIN_SESSIONS, int(n_sessions * floor))
    return [r for r in rows if r.points and r.sessions >= need]


def tail_slope(sessions: dict, pts: list[Point] | None = None) -> float:
    """How fast `E[remaining | attained >= x]` falls, in turns per turn.

    This is the mean-residual-life curve, summarised by least squares over the
    thresholds where enough sessions survive to estimate a mean. Two reference
    values anchor it and need no calibration:

    * **-0.5** -- every session is the same length. Averaging over the positions
      past `x` halves the 1:1 decline, so -0.5 rather than -1.0 is the
      equal-length reference. How far in a session is tells you how close it is
      to finishing, and a rule of the form "restart after N turns" can work.
    * **~0.0** -- lengths are widely dispersed. Surviving to `x` says little
      about what is left, and a turn-count rule is sorting noise.

    What this deliberately does **not** claim
    -----------------------------------------
    A rising curve would mean a genuinely heavy tail, and this statistic will
    not report one. That is not caution, it is arithmetic: every real workload
    is finite, so past the median length the surviving sessions are simply
    running out and the conditional mean must fall to zero. Measured on four
    synthetic workloads built to be heavy-tailed, the summary came out negative
    on all of them, for that reason and not because they lacked a tail.

    Three earlier versions of this statistic each reported a confident answer
    that was an artefact:

    * correlating attained against remaining over pooled positions -- inside one
      session those sum to a constant, so they are mechanically anti-correlated;
      on a heavy-tailed fixture it returned -0.75 and would have advised the
      exact opposite of the truth;
    * a rank correlation over thresholds -- reads "bounded" for every workload,
      because the ordering is set by the truncated tail;
    * least squares over the full threshold range -- the deepest threshold has
      enormous leverage, so the same truncation decides the answer again.

    What survives is the comparison against the equal-length reference, over
    the supported range. That is a smaller claim than the one this module set
    out to make, and it is the one the data can carry.
    """
    del pts   # the curve depends only on session lengths
    return slope_from_lengths(lengths_of(sessions))


def tail_slope_ci(
    sessions: dict,
    *,
    resamples: int = 200,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """Interval for the slope, resampling **sessions**.

    Resampling positions would treat 600 points inside one session as 600
    independent observations and return an interval roughly a factor of
    sqrt(600) too narrow. The session is the unit that was independently drawn,
    so the session is the unit that gets resampled.
    """
    import random

    lengths = lengths_of(sessions)
    if len(lengths) < 2:
        return (-1.0, 1.0)
    rng = random.Random(seed)
    n = len(lengths)
    draws = [slope_from_lengths([lengths[rng.randrange(n)] for _ in range(n)])
             for _ in range(resamples)]
    return (quantile(draws, alpha / 2.0), quantile(draws, 1.0 - alpha / 2.0))


@dataclass
class Report:
    rows: list[Row] = field(default_factory=list)
    slope: float = 0.0
    slope_ci: tuple[float, float] = (0.0, 0.0)
    n_sessions: int = 0
    n_points: int = 0

    @property
    def verdict(self) -> str:
        """`uniform-length` or `dispersed`, decided on the interval.

        Deliberately two-way. A third category for a heavy tail would be
        unfalsifiable here -- see `tail_slope` -- and a verdict this module
        cannot support is a verdict it should not print.
        """
        _lo, hi = self.slope_ci
        # Half way between the equal-length reference (-0.5) and no
        # information at all (0.0).
        return "uniform-length" if hi < -0.25 else "dispersed"

    @property
    def informative(self) -> bool:
        """Whether turn count is a usable predictor of what is left."""
        return self.verdict == "uniform-length"

    def to_json(self) -> dict:
        return {
            "sessions": self.n_sessions,
            "points": self.n_points,
            "tail_slope": self.slope,
            "tail_slope_ci95": list(self.slope_ci),
            "verdict": self.verdict,
            "attained_service_is_informative": self.informative,
            "equal_length_reference": -0.5,
            "curve": [
                {"attained_at_least": r.threshold, "sessions": r.sessions,
                 "mean_remaining_turns": r.mean_remaining,
                 "p90_remaining_turns": r.p90_remaining,
                 "mean_cost_remaining_usd": r.mean_cost_remaining,
                 "spend_share": r.spend_share, "thin": r.thin}
                for r in self.rows
            ],
        }


def analyse(sessions: dict, *, resamples: int = 200,
            seed: int = DEFAULT_SEED) -> Report:
    pts = points(sessions)
    return Report(
        rows=curve(sessions, pts),
        slope=tail_slope(sessions, pts),
        slope_ci=tail_slope_ci(sessions, resamples=resamples, seed=seed),
        n_sessions=len(sessions),
        n_points=len(pts),
    )


def format_report(rep: Report) -> str:
    out: list[str] = []
    out += render.heading("attained service — does long-so-far mean long-to-go?",
                          rule="=")
    if not rep.n_points:
        out.append("  No session long enough to have a remaining half.")
        return "\n".join(out)

    out.append(render.kv("sessions", f"{rep.n_sessions:,}"))
    out.append(render.kv("positions measured", f"{rep.n_points:,}"))
    _lo, hi = rep.slope_ci
    out.append(render.kv("tail slope", f"{rep.slope:+.3f} turns/turn  "
                                       f"[{_lo:+.3f}, {hi:+.3f}]"))
    out.append(render.kv("verdict", rep.verdict))
    out.append(render.kv("equal-length reference", "-0.500 turns/turn"))
    out.append("")

    out += render.table(
        [[f">= {r.threshold}", f"{r.sessions:,}",
          f"{r.mean_remaining:,.0f}", f"{r.p90_remaining:,.0f}",
          render.money(r.mean_cost_remaining), render.pct(r.spend_share),
          "thin" if r.thin else ""]
         for r in rep.rows if r.points],
        ["attained", "sessions", "mean left", "p90 left", "cost left", "of spend", ""],
        align="<>>>>><",
    )

    out.append("")
    if rep.verdict == "uniform-length":
        out += render.wrap(
            f"At {rep.slope:+.2f} turns per turn this workload is close to the "
            "equal-length reference of -0.50: sessions are similar sizes, so how "
            "far one has run does tell you how close it is to finishing. A rule "
            "of the form 'hand off after N turns' can work here, and "
            "`adder prefix` prices the restart.")
    else:
        out += render.wrap(
            f"At {rep.slope:+.2f} turns per turn, against an equal-length "
            "reference of -0.50, session lengths here are widely dispersed. How "
            "far a session has run is a weak predictor of what is left, so a "
            "fixed turn-count restart rule is largely sorting noise. Decide "
            "restarts on context size and cost per turn, which are observed "
            "rather than inferred.")

    heavy = [r for r in rep.rows if not r.thin and r.spend_share > 0]
    if heavy:
        deepest = heavy[-1]
        out += render.wrap(
            f"Sessions that reach {deepest.threshold} turns hold "
            f"{render.pct(deepest.spend_share)} of total spend and still have "
            f"{render.money(deepest.mean_cost_remaining)} of cost ahead of them "
            "on average. That is the population any per-session lever has to "
            "move to matter.")

    out.append("")
    out += render.wrap(
        "The interval resamples sessions, not positions: 600 positions inside "
        "one session agree with each other and would otherwise make the slope "
        "look far more certain than it is.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    from adder.core import filters

    ap = argparse.ArgumentParser(
        prog="adder sched",
        description="Whether how far a session has run predicts how much is "
                    "left, and what that means for restarting one.",
    )
    ap.add_argument("--resamples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--json", action="store_true")
    filters.add_arguments(ap)
    args = ap.parse_args(argv)

    sessions, _w = filters.load(args, use_cache=True)
    if not sessions:
        msg = "  No sessions found."
        print(json.dumps({"sessions": 0}, indent=2) if args.json else msg)
        return 1

    rep = analyse(sessions, resamples=max(1, args.resamples), seed=args.seed)
    if args.json:
        print(json.dumps(rep.to_json(), indent=2, sort_keys=True))
    else:
        print(format_report(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
