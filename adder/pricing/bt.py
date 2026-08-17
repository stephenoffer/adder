"""Bradley-Terry: turn pairwise preferences into ratings that carry an interval.

Why this module exists
----------------------
Everything downstream of `catalog.py` treats a model's arena rating as a
number. It is not a number, it is an estimate with a standard error, and the
difference matters at exactly the point where routing decisions are made. On
the top of the coding board the 95% interval is roughly +/-10 points wide, so a
"17-point lead" between two models is two overlapping intervals and no lead at
all. A router that swaps models on that difference is trading real money for
noise, and it will do it confidently.

So this module implements the estimator the public leaderboards actually use --
Bradley-Terry fit by maximum likelihood, with bootstrap intervals -- rather than
the Elo update rule people assume they use. The distinction is not pedantry:
Elo is an online rule that weights recent games more heavily and depends on the
order the games arrived in, so re-running it on a shuffled log gives different
ratings. Bradley-Terry is a batch MLE, has no order dependence, and is
reproducible from the log alone. For a tool whose whole claim is "re-run the
measurement", that is the only defensible choice.

Three things the rest of the package needs from here
----------------------------------------------------
1. **A rating with an interval.** `fit_with_ci` returns both, from the same
   bootstrap the leaderboards use, seeded so two runs agree.
2. **A statistically honest ranking.** `ranks` assigns the same rank to models
   whose intervals overlap. A router asking "is B worse than A" gets
   "indistinguishable" as a first-class answer, which is the answer most of the
   time and the one a scalar comparison can never give.
3. **Tiers, not a total order.** `tiers` partitions models into k strength
   classes by dynamic programming. This is what makes routing tractable: with
   ~400 models and pairwise data that is under 0.1% dense, per-pair estimates
   are hopeless, but "top tier vs third tier" is well determined. Everything
   that talks about a strong model and a weak model means a tier, not a model.

What it is not: a claim that arena preference is the right target. It is a
proxy for agentic tool use and it is labelled MODELLED everywhere it surfaces.
This module makes the proxy honest about its own precision; it cannot make it
measure something else.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from adder.util.stats import quantile

# Arena convention: ratings are reported on a logistic scale of 400 points per
# factor of 10 in odds, anchored so the numbers land in the familiar range.
# The scale is cosmetic -- every decision here is made on the log-odds -- but
# printing 1247 next to a published 1251 is worth the four lines it costs.
SCALE = 400.0 / math.log(10.0)
ANCHOR = 1000.0

DEFAULT_SEED = 20260801
DEFAULT_RESAMPLES = 200


class Battle(NamedTuple):
    """One pairwise comparison. `winner` is 'a', 'b', or 'tie'."""

    a: str
    b: str
    winner: str = "tie"


def _validate(battles: Sequence[Battle]) -> None:
    for i, b in enumerate(battles):
        if b.winner not in ("a", "b", "tie"):
            raise ValueError(f"battle {i}: winner must be a/b/tie, got {b.winner!r}")
        if b.a == b.b:
            raise ValueError(f"battle {i}: a model cannot play itself ({b.a!r})")


def fit(
    battles: Sequence[Battle],
    *,
    prior: float = 1.0,
    tol: float = 1e-9,
    max_iter: int = 500,
) -> dict[str, float]:
    """Bradley-Terry strengths by MM, returned on the arena's 400-point scale.

    Ties count as half a win to each side. That is what the public boards do,
    and the alternative (dropping ties) throws away the single most common
    outcome between two models of similar strength -- which is precisely the
    comparison a router has to get right.

    `prior` adds pseudo-battles against a virtual average opponent. Without it
    a model that has never lost has infinite strength, and the fit either
    diverges or silently stops at `max_iter` with a number that depends on the
    iteration cap. With it, an undefeated model on three battles gets a high
    rating with a wide interval, which is the truth. Set `prior=0` only if you
    have verified the comparison graph is strongly connected.

    The MM (minorize-maximize) iteration is used rather than Newton because it
    is monotone, needs no line search, cannot overshoot, and is about fifteen
    lines. On the sizes involved -- hundreds of models, tens of thousands of
    battles -- it converges in well under a second.
    """
    _validate(battles)
    models = sorted({m for b in battles for m in (b.a, b.b)})
    if not models:
        return {}
    if len(models) == 1:
        return {models[0]: ANCHOR}

    index = {m: i for i, m in enumerate(models)}
    n = len(models)
    # wins[i][j]: credit i earned against j. Halves are why this is a float.
    wins = [[0.0] * n for _ in range(n)]
    for b in battles:
        i, j = index[b.a], index[b.b]
        if b.winner == "a":
            wins[i][j] += 1.0
        elif b.winner == "b":
            wins[j][i] += 1.0
        else:
            wins[i][j] += 0.5
            wins[j][i] += 0.5

    # The virtual opponent sits at strength 1 and plays `prior` drawn battles
    # against everyone, which is what keeps an undefeated model finite.
    total_wins = [sum(wins[i]) + prior / 2.0 for i in range(n)]
    played = [[wins[i][j] + wins[j][i] for j in range(n)] for i in range(n)]

    # A model with no credit at all has no maximum-likelihood strength: the
    # likelihood rises without bound as its strength goes to zero, so the fit
    # walks it to 0 and `log(0)` ends the run with `math.domain error` from
    # four frames down. This is the mirror of the undefeated case the prior
    # already exists to handle, and it is reachable under the documented
    # precondition -- "the comparison graph is strongly connected" is about who
    # *played* whom, and a model can play everyone and beat none of them.
    #
    # Refused rather than floored. Picking an epsilon would put a finite rating
    # on a model whose rating is genuinely unbounded below, and a made-up
    # number that looks like a measurement is the failure this package is built
    # to avoid. The prior is the principled fix and the message says so.
    if prior <= 0:
        winless = [models[i] for i in range(n) if total_wins[i] <= 0]
        if winless:
            raise ValueError(
                f"prior=0 cannot fit {', '.join(winless[:5])}"
                + (f" and {len(winless) - 5} more" if len(winless) > 5 else "")
                + ": a model that never won has no finite Bradley-Terry strength. "
                "Pass prior>0 (the default is 1.0) to keep it finite.")

    p = [1.0] * n
    for _ in range(max_iter):
        new = [0.0] * n
        for i in range(n):
            denom = prior / (p[i] + 1.0)
            for j in range(n):
                if i != j and played[i][j] > 0:
                    denom += played[i][j] / (p[i] + p[j])
            new[i] = total_wins[i] / denom if denom > 0 else p[i]
        # Normalise to geometric mean 1 so the scale cannot drift between
        # iterations and the convergence test means what it says.
        logs = [math.log(x) for x in new if x > 0]
        if logs:
            g = math.exp(sum(logs) / len(logs))
            if g > 0:
                new = [x / g for x in new]
        shift = max((abs(math.log(a / b))
                     for a, b in zip(new, p, strict=True) if a > 0 and b > 0),
                    default=0.0)
        p = new
        if shift < tol:
            break

    return {m: ANCHOR + SCALE * math.log(p[index[m]]) for m in models}


def win_probability(rating_a: float, rating_b: float) -> float:
    """P(a beats b) under the fitted model, on the 400-point scale."""
    return 1.0 / (1.0 + math.exp((rating_b - rating_a) / SCALE))


@dataclass(frozen=True)
class Rating:
    """A fitted strength with the interval that says how much to trust it."""

    model: str
    rating: float
    lo: float
    hi: float
    battles: int

    @property
    def half_width(self) -> float:
        return (self.hi - self.lo) / 2.0

    def overlaps(self, other: Rating) -> bool:
        """True when no ordering between these two is supported by the data."""
        return not (self.lo > other.hi or other.lo > self.hi)

    def beats(self, other: Rating) -> bool:
        """Strictly better, with the intervals disjoint. The only claim worth acting on."""
        return self.lo > other.hi


def _resample_outcomes(
    battles: Sequence[Battle],
    point: dict[str, float],
    tie_rate: float,
    rng: random.Random,
) -> list[Battle]:
    """Replay the same matchups, redrawing who won from the fitted model."""
    out = []
    for b in battles:
        if tie_rate > 0.0 and rng.random() < tie_rate:
            out.append(Battle(b.a, b.b, "tie"))
            continue
        p = win_probability(point.get(b.a, ANCHOR), point.get(b.b, ANCHOR))
        out.append(Battle(b.a, b.b, "a" if rng.random() < p else "b"))
    return out


def fit_with_ci(
    battles: Sequence[Battle],
    *,
    alpha: float = 0.05,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    prior: float = 1.0,
    method: str = "outcomes",
) -> dict[str, Rating]:
    """Fit, then resample to get an interval per model.

    Two resampling schemes, and the difference between them is not academic.

    `method="battles"` draws rows of the log with replacement. This is what the
    public boards do, and on a large log it is right. On a small one it fails
    silently and badly: a log of six battles that a single model swept has no
    outcome variation to resample, every resample refits to the same number,
    and the reported interval has **zero width**. A tool whose entire purpose is
    to stop confident wrong numbers cannot ship an estimator that answers "how
    sure are you" with "completely" after six observations.

    `method="outcomes"` (the default) holds the matchup schedule fixed and
    redraws each winner from the fitted probabilities, preserving the observed
    tie rate. The justification is that the schedule is not a random sample of
    anything -- it is whichever pairs happened to be played -- while the thing a
    rating interval is supposed to describe is which side won. It degrades
    correctly: six sweeps produce a high rating with a very wide interval,
    which is the honest reading of six observations. On large balanced logs the
    two methods agree to within a point or two.

    `resamples` defaults low -- 200 rather than the 2000 used for scalar
    statistics -- because each resample is a full MLE fit. 200 is enough to
    place a 95% percentile interval to within a point or two on the arena
    scale, which is finer than any decision downstream of it.
    """
    if method not in ("outcomes", "battles"):
        raise ValueError(f"method must be 'outcomes' or 'battles', got {method!r}")
    _validate(battles)
    point = fit(battles, prior=prior)
    if not point:
        return {}
    counts: dict[str, int] = {}
    for b in battles:
        counts[b.a] = counts.get(b.a, 0) + 1
        counts[b.b] = counts.get(b.b, 0) + 1
    tie_rate = sum(1 for b in battles if b.winner == "tie") / len(battles) if battles else 0.0

    draws: dict[str, list[float]] = {m: [] for m in point}
    if battles and resamples > 0:
        rng = random.Random(seed)
        n = len(battles)
        for _ in range(resamples):
            if method == "battles":
                sample = [battles[rng.randrange(n)] for _ in range(n)]
            else:
                sample = _resample_outcomes(battles, point, tie_rate, rng)
            fitted = fit(sample, prior=prior)
            for m in point:
                # A model can miss a resample entirely. Recording its point
                # estimate rather than skipping keeps every model's interval
                # built from the same number of draws, so the percentiles are
                # comparable across rows.
                draws[m].append(fitted.get(m, point[m]))

    out: dict[str, Rating] = {}
    for m, r in point.items():
        d = draws[m]
        lo = quantile(d, alpha / 2.0) if d else r
        hi = quantile(d, 1.0 - alpha / 2.0) if d else r
        out[m] = Rating(m, r, min(lo, r), max(hi, r), counts.get(m, 0))
    return out


def ranks(ratings: Iterable[Rating]) -> dict[str, int]:
    """Rank each model as 1 + the number of models that strictly beat it.

    This is the leaderboard's "rank (upper bound)" convention, and it is the
    one that makes the ranking honest: models whose intervals overlap share a
    rank instead of being ordered by a point estimate that could flip on the
    next thousand votes. A report that prints a strict 1..N ordering from
    overlapping intervals is inventing precision.
    """
    rs = list(ratings)
    return {r.model: 1 + sum(1 for other in rs if other.beats(r)) for r in rs}


def indistinguishable(ratings: Iterable[Rating], model: str) -> list[str]:
    """Every model whose interval overlaps `model`'s, cheapest-first is caller's job.

    This is the set a cost optimiser is allowed to choose from for free: if the
    data cannot tell these models apart on quality, the only remaining
    difference is price.
    """
    rs = {r.model: r for r in ratings}
    target = rs.get(model)
    if target is None:
        return []
    return sorted(m for m, r in rs.items() if m != model and r.overlaps(target))


def tiers(scores: dict[str, float], k: int = 10) -> dict[str, int]:
    """Partition models into `k` strength tiers, tier 0 being the strongest.

    Exact dynamic programming over the sorted ratings, minimising total
    within-tier squared deviation -- the one-dimensional k-means that has an
    optimal O(n^2 k) solution, so there is no initialisation to get unlucky
    with and no seed to report.

    Why tiers at all: pairwise preference data between any two specific models
    is under 0.1% dense, so "is model X better than model Y" is usually
    unanswerable while "is X in the top tier" is not. Grouping first is the
    standard fix for that sparsity, and it is what makes a strong-vs-weak
    routing rule estimable from public data.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    items = sorted(scores.items(), key=lambda kv: -kv[1])
    n = len(items)
    if n == 0:
        return {}
    if k >= n:
        return {m: i for i, (m, _) in enumerate(items)}

    xs = [v for _, v in items]
    prefix = [0.0] * (n + 1)
    prefix_sq = [0.0] * (n + 1)
    for i, x in enumerate(xs):
        prefix[i + 1] = prefix[i] + x
        prefix_sq[i + 1] = prefix_sq[i] + x * x

    def sse(i: int, j: int) -> float:
        """Within-segment squared deviation for xs[i:j]."""
        m = j - i
        if m <= 1:
            return 0.0
        total = prefix[j] - prefix[i]
        return max(0.0, (prefix_sq[j] - prefix_sq[i]) - total * total / m)

    inf = float("inf")
    cost = [[inf] * (k + 1) for _ in range(n + 1)]
    split = [[0] * (k + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for t in range(1, k + 1):
        for j in range(1, n + 1):
            for i in range(t - 1, j):
                if cost[i][t - 1] == inf:
                    continue
                c = cost[i][t - 1] + sse(i, j)
                if c < cost[j][t]:
                    cost[j][t] = c
                    split[j][t] = i

    out: dict[str, int] = {}
    j, t = n, k
    while t > 0:
        i = split[j][t]
        for idx in range(i, j):
            out[items[idx][0]] = t - 1
        j, t = i, t - 1
    return out


def tier_members(assignment: dict[str, int]) -> dict[int, list[str]]:
    """Invert a tier assignment, members sorted, for display and for selection."""
    out: dict[int, list[str]] = {}
    for m, t in assignment.items():
        out.setdefault(t, []).append(m)
    for t in out:
        out[t].sort()
    return out


def agreement(fitted: dict[str, float], published: dict[str, float]) -> float:
    """Rank correlation between a local fit and a published board.

    The check that keeps this module honest: if refitting the public battle log
    does not reproduce the public ordering, the fit is wrong and every tier
    built on it is wrong. `adder validate` runs this.
    """
    from adder.util.stats import spearman

    shared = sorted(set(fitted) & set(published))
    if len(shared) < 3:
        return 0.0
    return spearman([fitted[m] for m in shared], [published[m] for m in shared])
