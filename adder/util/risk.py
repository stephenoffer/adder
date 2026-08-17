"""Uncertainty arithmetic: the difference between "cheaper on average" and "cheaper".

Every gate in this repo compares two modelled costs and emits advice when the
difference is positive. That is the right comparison only if the inputs are
known. They are not. Three of them are estimates with real spread:

* `remaining_turns` is a forecast off a heavy-tailed length distribution,
* `p_fail` is a rate measured from a handful of logged outcomes,
* the summary a delegated read hands back is a modelled compression ratio.

A gate fed point estimates answers "is the saving positive at the midpoint",
which is not the question. The question is "is the saving positive", and the
honest answer to that is a distribution. When the distribution straddles zero,
advice that clears the midpoint is a coin flip wearing a dollar sign -- and
because a wrong recommendation costs the routing turn *plus* the redo, a
straddling recommendation loses money on average even when its midpoint is
positive.

So this module supplies three things the point-estimate gates were missing.

**Posterior bounds on a rate.** `p_fail` is Beta-posterior, already, in
`outcomes.evidence` -- but only its mean was ever used, and a mean cannot
distinguish 0.2 from four runs from 0.2 from four hundred. `beta_bounds` returns
the credible interval, so a gate can price the failure branch at the *upper*
bound of how often it fails, which is the only bound that cannot flatter a
recommendation.

**Worst-case evaluation over a box.** Every cost function here is multilinear in
its uncertain inputs: tokens x rate x turns, plus a probability multiplying a
branch. A multilinear function on a box attains its extrema at the box's
vertices, so the worst case is exact from 2^k evaluations rather than an
optimisation. `worst_case` does that enumeration and reports which corner lost,
because "this advice fails if the session ends within 40 turns" is actionable
and "worst case -$0.03" is not.

**Probability the advice is right.** `p_cheaper` integrates the saving over the
independent marginals on a stratified grid -- deterministic, no RNG, no seed to
leave unset. It is the number a `--min-confidence` flag can gate on.

The multilinearity claim is load-bearing and is tested rather than asserted; see
`tests/test_risk.py::test_multilinear_extrema_are_at_vertices`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from itertools import product
from math import exp, lgamma, log, log1p

# Two-sided credible level used everywhere a bound is taken without one being
# named. 0.10 puts 5% in each tail: tight enough that a well-evidenced tier
# still clears its gate, loose enough that four observations do not.
DEFAULT_ALPHA = 0.10

_MAX_ITER = 300
_EPS = 1e-12


# --------------------------------------------------------------------------
# Regularized incomplete beta, and the Beta quantile built on it.
#
# Written out rather than imported because `[project.dependencies]` stays empty
# and `statistics` has no beta quantile. Continued fraction is the standard
# modified-Lentz evaluation; the quantile is a bisection on a monotone CDF,
# which is slower than Newton and cannot diverge, and the whole thing runs a
# few hundred times per decision at most.
# --------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    """Continued-fraction expansion of the incomplete beta (modified Lentz)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _EPS:
        d = _EPS
    d = 1.0 / d
    h = d
    for m in range(1, _MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _EPS:
            d = _EPS
        c = 1.0 + aa / c
        if abs(c) < _EPS:
            c = _EPS
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _EPS:
            d = _EPS
        c = 1.0 + aa / c
        if abs(c) < _EPS:
            c = _EPS
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) = P(Beta(a,b) <= x)."""
    if a <= 0 or b <= 0:
        raise ValueError(f"beta parameters must be positive, got a={a}, b={b}")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log1p(-x))
    # Converge on whichever side of the mode keeps the fraction well-behaved.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_quantile(q: float, a: float, b: float) -> float:
    """Inverse CDF of Beta(a, b) by bisection. Monotone, so it cannot diverge."""
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"quantile must be in [0,1], got {q}")
    if q == 0.0:
        return 0.0
    if q == 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class Interval:
    """A point estimate that admits it is one.

    `lo`/`hi` are whatever bound the producer thinks is honest -- a credible
    interval for a rate, an empirical quantile for a forecast. `point` is what
    a mean-only caller would have used, kept so the two can be compared.
    """

    lo: float
    point: float
    hi: float

    def __post_init__(self) -> None:
        if not (self.lo <= self.point <= self.hi):
            raise ValueError(f"interval out of order: {self.lo} <= {self.point} <= {self.hi}")

    @property
    def width(self) -> float:
        return self.hi - self.lo

    @classmethod
    def exact(cls, x: float) -> Interval:
        """A known quantity, so the same machinery can carry it without special-casing."""
        return cls(x, x, x)

    def corners(self) -> tuple[float, float]:
        return (self.lo, self.hi)


def beta_bounds(
    fails: float,
    total: float,
    *,
    alpha: float = DEFAULT_ALPHA,
    prior_fail: float = 1.0,
    prior_ok: float = 1.0,
) -> Interval:
    """Credible interval for a failure rate, from a Beta-conjugate posterior.

    `fails` and `total` may be fractional: the outcome log weights observations
    by recency, and an interval that ignored that weighting would claim evidence
    the estimate does not have. Half-weight observations widen the interval,
    exactly as they should.

    The mean is the same number `outcomes.evidence` already reports; what is new
    is `hi`, which is what a gate should charge the failure branch at.
    """
    if total < 0 or fails < 0 or fails > total + _EPS:
        raise ValueError(f"need 0 <= fails <= total, got fails={fails}, total={total}")
    a = prior_fail + fails
    b = prior_ok + max(0.0, total - fails)
    mean = a / (a + b)
    lo = beta_quantile(alpha / 2.0, a, b)
    hi = beta_quantile(1.0 - alpha / 2.0, a, b)
    # Bisection is exact to 1e-12; clamp anyway so the ordering invariant on
    # Interval can never trip on a rounding artifact.
    return Interval(min(lo, mean), mean, max(hi, mean))


def beta_from_mean(mean: float, pseudo_count: float) -> tuple[float, float]:
    """Beta parameters whose mean is exactly `mean`, by moment matching.

    The obvious construction -- feed `mean * n` failures and `n` trials to
    `beta_bounds` -- does not do this. That routine adds a Beta(1,1) prior, so a
    0.15 belief backed by four pseudo-observations comes out with a mean of
    0.27, and the interval is no longer centred on the number the caller
    actually holds. When the input *is* the belief rather than a count of
    events, there is no prior left to add: match the moments and stop.
    """
    if not 0.0 <= mean <= 1.0:
        raise ValueError(f"mean must be in [0,1], got {mean}")
    n = max(0.2, pseudo_count)
    return max(0.05, mean * n), max(0.05, (1.0 - mean) * n)


def bounds_from_mean(mean: float, pseudo_count: float, *,
                     alpha: float = DEFAULT_ALPHA) -> Interval:
    """Credible interval around a stated belief with a stated weight."""
    a, b = beta_from_mean(mean, pseudo_count)
    lo = beta_quantile(alpha / 2.0, a, b)
    hi = beta_quantile(1.0 - alpha / 2.0, a, b)
    return Interval(min(lo, mean), mean, max(hi, mean))


def quantiles_from_mean(mean: float, pseudo_count: float, *,
                        strata: int = 8) -> list[float]:
    """Equally-weighted quantile ladder around a stated belief."""
    a, b = beta_from_mean(mean, pseudo_count)
    return [beta_quantile((i + 0.5) / strata, a, b) for i in range(strata)]


def empirical_bounds(values: Sequence[float], *, alpha: float = DEFAULT_ALPHA) -> Interval:
    """Quantile interval around the MEAN of an empirical sample.

    The mean, not the median, and the distinction is the whole point. Cost is
    linear in remaining turns, so the expected cost of carrying a token is set
    by `E[R]`, not by `median(R)`. Session length is heavy-tailed -- a handful
    of very long sessions carry most of the spend -- so the median sits well
    below the mean, and a carry cost computed from the median under-prices every
    long session, which is where all the money is. See `horizon.mean_remaining`.
    """
    xs = sorted(float(v) for v in values)
    if not xs:
        raise ValueError("empirical_bounds needs at least one observation")
    mean = sum(xs) / len(xs)
    lo = _quantile(xs, alpha / 2.0)
    hi = _quantile(xs, 1.0 - alpha / 2.0)
    # Clamp, exactly as `beta_bounds` does. On a heavy-tailed sample the mean can
    # sit outside its own quantile interval -- a hundred 20-turn sessions and one
    # 10,000-turn session put E[R] above the 95th percentile -- and that is the
    # distribution this function exists for, not an edge case. Ordering the
    # interval around the point estimate keeps the carry term priced off the
    # mean instead of raising ValueError at the one shape that matters.
    return Interval(min(lo, mean), mean, max(hi, mean))


def _quantile(sorted_xs: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile. `stats.quantile` is the one definition.

    Kept as a name because two functions here call it with an already-sorted
    sample, but the arithmetic is no longer restated: `stats` exists because
    three modules had three definitions of the p90, and a fourth living in the
    module that prices every gate's uncertainty is the worst place for one.
    """
    from adder.util.stats import quantile

    return quantile(sorted_xs, q)


# --------------------------------------------------------------------------
# Worst case over a box, and the probability the advice is right.
# --------------------------------------------------------------------------

def worst_case(
    saving: Callable[..., float],
    bounds: dict[str, Interval],
) -> tuple[float, dict[str, float]]:
    """Smallest value `saving` takes over the box, and the corner where it did.

    Exact, not approximate, for the functions this repo actually evaluates. Each
    cost here is multilinear in its uncertain arguments -- `tokens * rate *
    turns`, `p_fail * branch_cost` -- and a multilinear function is affine along
    every axis with the others held fixed, so its extrema over an axis-aligned
    box are attained at a vertex. Enumerating 2^k vertices therefore *is* the
    minimisation, with k = 3 or 4 in practice.

    For a function that is not multilinear this returns the worst vertex rather
    than the worst point, which is a bound in the optimistic direction; callers
    outside this repo should check their own function before trusting it.
    """
    if not bounds:
        return saving(), {}
    keys = sorted(bounds)
    best_val, best_corner = None, {}
    for combo in product(*(bounds[k].corners() for k in keys)):
        kw = dict(zip(keys, combo, strict=True))
        val = saving(**kw)
        if best_val is None or val < best_val:
            best_val, best_corner = val, kw
    return float(best_val), best_corner


def p_cheaper(
    saving: Callable[..., float],
    marginals: dict[str, Sequence[float]],
    *,
    strata: int = 8,
) -> float:
    """P(saving > 0) under independent marginals, by stratified quadrature.

    Deterministic on purpose. A Monte-Carlo estimate of this number would need a
    seed, and an unseeded seed is how a test starts passing on Tuesdays. Each
    marginal is supplied as a quantile ladder (see `quantiles`), the product grid
    is enumerated, and the answer is the share of grid points with a positive
    saving. With `strata` equally-weighted quantiles per axis, that share is the
    midpoint rule for the probability integral.
    """
    if not marginals:
        return 1.0 if saving() > 0 else 0.0
    keys = sorted(marginals)
    grids = [list(marginals[k]) for k in keys]
    if any(not g for g in grids):
        raise ValueError("every marginal needs at least one quantile point")
    total = wins = 0
    for combo in product(*grids):
        total += 1
        if saving(**dict(zip(keys, combo, strict=True))) > 0:
            wins += 1
    del strata  # kept in the signature for callers that build their own ladders
    return wins / total if total else 0.0


def beta_quantiles(fails: float, total: float, *, strata: int = 8,
                   prior_fail: float = 1.0, prior_ok: float = 1.0) -> list[float]:
    """Equally-weighted quantile ladder of a Beta posterior, for `p_cheaper`."""
    a = prior_fail + fails
    b = prior_ok + max(0.0, total - fails)
    return [beta_quantile((i + 0.5) / strata, a, b) for i in range(strata)]


def empirical_quantiles(values: Sequence[float], *, strata: int = 8) -> list[float]:
    """Equally-weighted quantile ladder of a sample, for `p_cheaper`."""
    xs = sorted(float(v) for v in values)
    if not xs:
        raise ValueError("empirical_quantiles needs at least one observation")
    return [_quantile(xs, (i + 0.5) / strata) for i in range(strata)]


# Probability that a recommendation must be cheaper before it is emitted.
#
# Half, and the reason it is not higher is that the tool makes hundreds of these
# decisions rather than one. Total spend is the *sum* of the outcomes, so it
# concentrates on the sum of the expectations: a rule that declines positive-
# expectation advice to protect any single decision loses money over the
# sequence, reliably, in exchange for a comfort it cannot actually deliver.
# Raising this to 0.80 on measured data here rejects delegations with $1.13 of
# expected saving against $0.21 of overhead -- advice that is right seven times
# in ten and pays 5x when it is right.
#
# What the threshold is for is the specific pathology that expectation alone
# cannot see: a mean dragged positive by a tail while the typical outcome loses.
# Below half, that is what is happening, and the recommendation should not be
# made. Callers who are optimising one session rather than a workload can raise
# it; `--min-confidence` exists for them.
#
# The pessimistic vertex is reported but never gated on. Requiring the saving to
# survive it means requiring every uncertain input to take its worst value at
# the same time, which for three independent inputs at a 5% tail each is a joint
# event of probability 0.000125. Refusing to act on a one-in-eight-thousand
# scenario is not caution; a tool that declines everything is trivially never
# more expensive and never worth installing.
DEFAULT_CONFIDENCE = 0.50


@dataclass(frozen=True)
class Guarantee:
    """Whether a recommendation is cheap, or merely cheap in expectation.

    Three separate claims, in ascending strength, because they are separately
    useful and collapsing them loses information:

    * `expected > overhead` -- cheaper at the point estimates. This is what every
      gate in this repo used to check on its own, and it is the weakest of the
      three: it says nothing about how wide the estimates are.
    * `safe` -- cheaper with probability at least `threshold`, integrating over
      the marginals. This is what the gate should use.
    * `dominant` -- cheaper at the pessimistic vertex, so cheaper under *any*
      combination of inputs the estimates admit. Rare, and worth saying so when
      it holds.
    """

    expected: float                    # saving at the point estimates
    worst: float                       # saving at the pessimistic vertex
    confidence: float                  # P(saving > overhead) under the marginals
    overhead: float = 0.0              # what emitting the advice costs
    corner: dict[str, float] = field(default_factory=dict)
    alpha: float = DEFAULT_ALPHA
    threshold: float = DEFAULT_CONFIDENCE

    @property
    def safe(self) -> bool:
        """Cheaper than doing nothing with at least `threshold` probability."""
        return self.expected > self.overhead and self.confidence >= self.threshold

    @property
    def dominant(self) -> bool:
        """Cheaper than doing nothing even at the worst admissible inputs."""
        return self.worst > self.overhead

    @property
    def margin(self) -> float:
        return self.worst - self.overhead

    def describe(self) -> str:
        head = (f"saves ${self.expected:,.4f} expected against ${self.overhead:,.4f} "
                f"of overhead, {self.confidence:.0%} confident")
        why = ", ".join(f"{k}={v:,.4g}" for k, v in sorted(self.corner.items()))
        # `safe` first, and not `dominant` first. A caller can set a threshold
        # above 1.0 to force every recommendation off, and a description that
        # led with dominance would report the good news about a decision that
        # was declined.
        if not self.safe:
            return f"not confident enough to recommend: {head}. It loses money when {why}"
        if self.dominant:
            return head + f"; cheaper even at the worst inputs (${self.worst:,.4f})"
        return head + f"; it would lose money only if {why}"


def guarantee(
    saving: Callable[..., float],
    bounds: dict[str, Interval],
    *,
    overhead: float = 0.0,
    marginals: dict[str, Sequence[float]] | None = None,
    alpha: float = DEFAULT_ALPHA,
    threshold: float = DEFAULT_CONFIDENCE,
) -> Guarantee:
    """Price a recommendation at its midpoint, its worst corner, and in probability.

    The probability is of clearing `overhead`, not of clearing zero. Advice that
    saves a cent 99% of the time and costs a routing turn every time is a losing
    proposition, and a gate that integrates against the wrong threshold would
    wave it through.
    """
    expected = saving(**{k: v.point for k, v in bounds.items()})
    worst, corner = worst_case(saving, bounds)
    if marginals:
        conf = p_cheaper(lambda **kw: saving(**kw) - overhead, marginals)
    else:
        conf = 1.0 if worst > overhead else 0.0
    return Guarantee(expected=expected, worst=worst, confidence=conf,
                     overhead=overhead, corner=corner, alpha=alpha,
                     threshold=threshold)


def shrink(values: Iterable[float], prior: float, weight: float) -> float:
    """James-Stein-flavoured shrink of a small sample toward a prior.

    Used where a per-project rate has to stand in for a global one: with `n`
    observations the sample gets weight `n / (n + weight)`. It is not the
    optimal shrinkage estimator -- that needs a variance this data does not
    reliably supply -- but it has the property that matters, which is that a
    two-observation project rate cannot swing a gate on its own.
    """
    xs = list(values)
    if not xs:
        return prior
    n = len(xs)
    lam = n / (n + max(0.0, weight))
    return lam * (sum(xs) / n) + (1.0 - lam) * prior
