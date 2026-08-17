"""Which comparison to run next, when every comparison costs real money.

The problem with "run thirty and look"
---------------------------------------
`adder ab` runs a controlled A/B on answer quality. It works, and it is the
only thing in this repo that settles a quality question rather than modelling
one. It is also the only thing here that spends money to produce data, and it
has been used the way everyone uses an A/B harness: pick two models, run some
tasks, read the result, run some more if it looks close.

That is the worst possible allocation of a fixed budget. Comparisons are not
equally informative. Comparing a model against one it obviously beats teaches
you nothing you did not already know; comparing two whose intervals overlap by
90% is where the entire uncertainty lives. Spending evenly across pairs buys a
ranking whose weakest link is exactly the pair nobody sampled.

Public preference leaderboards solved this with **active sampling**: rather than
drawing pairs uniformly, draw them in proportion to how much a new comparison
would reduce uncertainty about the ranking. The same idea applies here, with a
much smaller budget and therefore a much larger payoff.

What "most informative" means here
-----------------------------------
For a Bradley-Terry fit, one more comparison between `i` and `j` adds Fisher
information

    w_ij = p_ij * (1 - p_ij)

where `p_ij` is the current estimate that `i` beats `j`. That term is maximised
at `p = 0.5` and collapses toward zero as the outcome becomes predictable, which
is the formal version of "a coin flip is informative, a foregone conclusion is
not". Weighting by it alone, however, gives you a harness that endlessly
re-samples the closest pair even after it has been settled to a precision
nobody needs.

So the score used here is information **per unit of remaining uncertainty**:

    value(i,j) = p_ij * (1 - p_ij) * overlap(i, j) / sqrt(n_ij + 1)

* the first term is the information a comparison carries;
* `overlap` is how much the two confidence intervals still share, so a pair that
  has already separated stops being attractive even if its win probability is
  near 0.5;
* the last term is diminishing returns on a pair already sampled heavily, which
  is what stops the harness from spending its whole budget on one cell.

None of the three on its own produces sensible allocations, and each of the
three has an obvious failure mode that the other two cover. That is the entire
design.

What this cannot do
-------------------
It cannot tell you the pair is worth comparing *at all*. A pair of models you
would never route between is a waste of budget however uncertain it is, so the
candidate set is yours to supply. It also assumes comparisons are exchangeable,
which they are not if your task mix drifts mid-experiment -- run a design, then
re-plan, rather than treating a plan as a schedule for the next month.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from adder.pricing.bt import Battle, Rating, fit_with_ci, win_probability
from adder.util import render
from adder.util.stats import wilson_interval

# A pair whose intervals no longer overlap at all still gets a floor of
# attention, because "separated" is a statement about the current sample and a
# drifting task mix can un-separate it. Small enough that it never outranks a
# genuinely contested pair.
SETTLED_FLOOR = 0.02


@dataclass(frozen=True)
class Pair:
    """One candidate comparison, scored for how much it would teach you."""

    a: str
    b: str
    n: int
    p_win: float
    overlap: float
    value: float

    @property
    def settled(self) -> bool:
        return self.overlap <= 0.0

    @property
    def key(self) -> tuple[str, str]:
        return (self.a, self.b) if self.a <= self.b else (self.b, self.a)


def interval_overlap(x: Rating, y: Rating) -> float:
    """Shared width of two rating intervals, as a fraction of the narrower one.

    Zero when they are disjoint (the ordering is established), one when either
    interval sits entirely inside the other (the ordering is anyone's guess).
    A fraction rather than an absolute width so a pair of well-sampled models
    with tight intervals is not permanently outranked by a pair of barely
    sampled ones with enormous ones.
    """
    lo = max(x.lo, y.lo)
    hi = min(x.hi, y.hi)
    shared = max(0.0, hi - lo)
    narrower = min(x.hi - x.lo, y.hi - y.lo)
    if narrower <= 0:
        return 1.0 if shared > 0 else 0.0
    return min(1.0, shared / narrower)


def counts(battles) -> dict[tuple[str, str], int]:
    """How many comparisons each unordered pair already has."""
    out: dict[tuple[str, str], int] = {}
    for b in battles:
        k = (b.a, b.b) if b.a <= b.b else (b.b, b.a)
        out[k] = out.get(k, 0) + 1
    return out


def score_pairs(
    ratings: dict[str, Rating],
    battles,
    *,
    candidates: list[tuple[str, str]] | None = None,
) -> list[Pair]:
    """Rank every candidate comparison by how much it would reduce uncertainty."""
    import math

    seen = counts(battles)
    models = sorted(ratings)
    if candidates is None:
        candidates = [(a, b) for i, a in enumerate(models) for b in models[i + 1:]]

    out: list[Pair] = []
    for a, b in candidates:
        ra, rb = ratings.get(a), ratings.get(b)
        if ra is None or rb is None:
            continue
        p = win_probability(ra.rating, rb.rating)
        information = p * (1.0 - p)
        overlap = interval_overlap(ra, rb)
        n = seen.get((a, b) if a <= b else (b, a), 0)
        value = information * max(overlap, SETTLED_FLOOR) / math.sqrt(n + 1)
        out.append(Pair(a, b, n, p, overlap, value))
    out.sort(key=lambda pr: (-pr.value, pr.a, pr.b))
    return out


def allocate(pairs: list[Pair], budget: int) -> dict[tuple[str, str], int]:
    """Split `budget` comparisons across pairs in proportion to their value.

    Largest-remainder apportionment rather than rounding each share
    independently: rounding gives away or invents comparisons, and a plan that
    says "run 31 of your 30" is a plan nobody follows. Every pair with any
    value gets at least one comparison if the budget allows, because a plan
    that ignores a contested pair entirely is the failure this module exists
    to prevent.
    """
    if budget <= 0 or not pairs:
        return {}
    total = sum(p.value for p in pairs)
    if total <= 0:
        # Nothing to distinguish them: spread evenly rather than picking by
        # sort order, which would be an arbitrary choice dressed as a decision.
        share, extra = divmod(budget, len(pairs))
        return {p.key: share + (1 if i < extra else 0) for i, p in enumerate(pairs)}

    exact = {p.key: budget * p.value / total for p in pairs}
    floors = {k: int(v) for k, v in exact.items()}
    left = budget - sum(floors.values())
    for k, _ in sorted(exact.items(), key=lambda kv: -(kv[1] - int(kv[1])))[:left]:
        floors[k] += 1
    return {k: v for k, v in floors.items() if v > 0}


def comparisons_to_separate(p_win: float, *, alpha: float = 0.05,
                            cap: int = 100_000) -> int:
    """How many comparisons before a pair with this win rate separates.

    The number that decides whether an experiment is worth starting. A pair at
    p=0.55 needs on the order of a thousand comparisons before its Wilson
    interval clears one half, which at any realistic per-task cost means the
    honest answer is "this pair will not be separated, route on price".

    Returns `cap` when the pair is a true coin (`p = 0.5`), because no finite
    number of comparisons separates a tie -- and reporting a large number
    rather than raising is what lets the caller print "not in this budget".
    """
    if not 0.0 <= p_win <= 1.0:
        raise ValueError(f"p_win must be in [0,1], got {p_win}")
    edge = abs(p_win - 0.5)
    if edge < 1e-9:
        return cap
    n = 1
    while n < cap:
        lo, hi = wilson_interval(round(p_win * n), n, alpha=alpha)
        if lo > 0.5 or hi < 0.5:
            return n
        n = max(n + 1, int(n * 1.3))
    return cap


@dataclass
class Plan:
    pairs: list[Pair]
    allocation: dict[tuple[str, str], int]
    budget: int
    cost_per_comparison: float = 0.0

    @property
    def total_cost(self) -> float:
        return self.budget * self.cost_per_comparison

    @property
    def contested(self) -> list[Pair]:
        return [p for p in self.pairs if not p.settled]

    def to_json(self) -> dict:
        return {
            "budget": self.budget,
            "cost_per_comparison_usd": self.cost_per_comparison,
            "total_cost_usd": self.total_cost,
            "contested_pairs": len(self.contested),
            "plan": [
                {"a": a, "b": b, "comparisons": n}
                for (a, b), n in sorted(self.allocation.items(), key=lambda kv: -kv[1])
            ],
            "pairs": [
                {"a": p.a, "b": p.b, "n": p.n, "p_win": p.p_win,
                 "overlap": p.overlap, "value": p.value, "settled": p.settled}
                for p in self.pairs
            ],
        }


def plan(
    battles,
    *,
    budget: int = 60,
    cost_per_comparison: float = 0.0,
    candidates: list[tuple[str, str]] | None = None,
    resamples: int = 80,
) -> Plan:
    ratings = fit_with_ci(battles, resamples=resamples) if battles else {}
    pairs = score_pairs(ratings, battles, candidates=candidates)
    return Plan(pairs, allocate(pairs, budget), budget, cost_per_comparison)


def load_battles(path: Path) -> list[Battle]:
    """Read comparisons from JSONL: `{"a": ..., "b": ..., "winner": "a"|"b"|"tie"}`."""
    out: list[Battle] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: not JSON ({exc.msg})") from exc
            missing = [k for k in ("a", "b") if k not in d]
            if missing:
                raise ValueError(f"{path}:{lineno}: missing {', '.join(missing)}")
            out.append(Battle(str(d["a"]), str(d["b"]), str(d.get("winner", "tie"))))
    return out


def format_report(pl: Plan, *, top: int = 10) -> str:
    out: list[str] = []
    out += render.heading("experiment design — what to compare next", rule="=")
    if not pl.pairs:
        out.append("  No comparisons recorded yet, so there is nothing to rank.")
        out += render.wrap(
            "Seed the log with a handful of comparisons between the models you "
            "would actually route between, then re-run: the allocation below is "
            "only meaningful once there is an ordering to be uncertain about.")
        return "\n".join(out)

    out.append(render.kv("candidate pairs", str(len(pl.pairs))))
    out.append(render.kv("still contested", str(len(pl.contested))))
    out.append(render.kv("budget", f"{pl.budget} comparisons"))
    if pl.cost_per_comparison > 0:
        out.append(render.kv("at", f"{render.money(pl.cost_per_comparison)} each "
                                   f"= {render.money(pl.total_cost)}"))
    out.append("")

    rows = []
    for p in pl.pairs[:top]:
        n = pl.allocation.get(p.key, 0)
        rows.append([f"{p.a[:20]} vs {p.b[:20]}", str(p.n), f"{p.p_win:.2f}",
                     render.pct(p.overlap), str(n),
                     "settled" if p.settled else ""])
    out += render.table(
        rows, ["pair", "have", "p(win)", "overlap", "run", ""],
        align="<>>>><",
    )

    out.append("")
    hardest = max(pl.contested, key=lambda p: p.overlap, default=None)
    if hardest is not None:
        need = comparisons_to_separate(hardest.p_win)
        if need >= 100_000:
            out += render.wrap(
                f"{hardest.a} and {hardest.b} are a coin flip on this data. No "
                "finite number of comparisons separates a tie, so the honest "
                "conclusion is that they are interchangeable on quality and the "
                "choice is price.")
        elif pl.cost_per_comparison > 0:
            out += render.wrap(
                f"Separating {hardest.a} from {hardest.b} needs about {need:,} "
                f"comparisons ({render.money(need * pl.cost_per_comparison)}). "
                "If that is more than the routing decision is worth, stop "
                "measuring and route on price.")
        else:
            out += render.wrap(
                f"Separating {hardest.a} from {hardest.b} needs about {need:,} "
                "comparisons. Pass --cost to see what that is in dollars.")

    out += render.wrap(
        "Allocation is proportional to information per unit of remaining "
        "uncertainty, so contested pairs get the budget and foregone conclusions "
        "do not. Re-plan after running: a plan is not a schedule.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    import sys

    ap = argparse.ArgumentParser(
        prog="adder design",
        description="Allocate a comparison budget to the pairs that would "
                    "actually reduce uncertainty about the ranking.",
    )
    ap.add_argument("path", nargs="?", type=Path,
                    help="JSONL of recorded comparisons (a, b, winner)")
    ap.add_argument("--budget", type=int, default=60,
                    help="comparisons you are willing to run (default 60)")
    ap.add_argument("--cost", type=float, default=0.0,
                    help="USD per comparison, to price the plan")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    battles: list[Battle] = []
    if args.path is not None:
        if not args.path.exists():
            print(f"adder design: no such file: {args.path}", file=sys.stderr)
            return 1
        try:
            battles = load_battles(args.path)
        except ValueError as exc:
            print(f"adder design: {exc}", file=sys.stderr)
            return 2

    pl = plan(battles, budget=max(0, args.budget), cost_per_comparison=max(0.0, args.cost))
    if args.json:
        print(json.dumps(pl.to_json(), indent=2, sort_keys=True))
    else:
        print(format_report(pl, top=max(1, args.top)))
    return 0 if pl.pairs else 1


if __name__ == "__main__":
    raise SystemExit(main())
