"""Small statistics helpers, in one place because three modules disagreed.

`trace` ranked a sorted list and indexed it by `int(len*p)`; `carry` and
`prefix` each carried a private `_median` that returned 0.0 on an empty input;
`horizon` interpolated. Three definitions of "the p90" produce three different
p90s from the same data, and the one in `trace` is biased high -- it is the
nearest-rank estimator with no interpolation, so on 10 sessions the "p90" is
the 10th value, i.e. the maximum.

Every quantile here is the linear-interpolation estimator (the same one
`statistics.quantiles(method="inclusive")` uses), so `quantile(xs, 0.5)` and
`median(xs)` agree by construction. Nothing here does I/O and nothing here
depends on the rest of the package.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def _sorted(xs: Iterable[float]) -> list[float]:
    return sorted(float(x) for x in xs)


def quantile(xs: Iterable[float], q: float) -> float:
    """Linear-interpolated quantile. Empty input is 0.0, not an exception.

    A report that raises on an empty dataset is a report nobody can run on a
    fresh machine, and every caller here is a display path.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"quantile must be in [0,1], got {q}")
    a = _sorted(xs)
    if not a:
        return 0.0
    if len(a) == 1:
        return a[0]
    pos = q * (len(a) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(a) - 1)
    frac = pos - lo
    return a[lo] * (1.0 - frac) + a[hi] * frac


def median(xs: Iterable[float]) -> float:
    return quantile(xs, 0.5)


def mean(xs: Iterable[float]) -> float:
    a = list(xs)
    return sum(a) / len(a) if a else 0.0


def quantiles(xs: Iterable[float], qs: Sequence[float] = (0.5, 0.9)) -> list[float]:
    a = _sorted(xs)
    return [quantile(a, q) for q in qs]


def mad(xs: Iterable[float]) -> float:
    """Median absolute deviation. The scale estimator that outliers cannot move.

    Standard deviation is the wrong tool for finding an expensive turn: one
    $40 turn inflates sigma enough to hide itself. MAD does not move.
    """
    a = _sorted(xs)
    if not a:
        return 0.0
    m = quantile(a, 0.5)
    return quantile([abs(x - m) for x in a], 0.5)


# Scale factor making MAD a consistent estimator of sigma for normal data.
MAD_TO_SIGMA = 1.4826


def _scale(sorted_xs: Sequence[float]) -> tuple[float, float]:
    """`(median, scale)` for a robust z. Handles the zero-dispersion case.

    MAD is zero whenever more than half the sample is identical -- which is
    rare in real spend and universal in a synthetic fixture. Returning a scale
    of zero there means "never flag anything", so a detector built on it misses
    the one value that is obviously an outlier, which is the case it exists for.

    When there is no dispersion to measure, the median itself becomes the
    scale, and the score turns into "how many times the median is this value
    above the median". Constant data still scores 0 for every point, so nothing
    is invented; a lone extreme value scores high, which is correct.
    """
    med = quantile(sorted_xs, 0.5)
    scale = mad(sorted_xs) * MAD_TO_SIGMA
    if scale > 0:
        return med, scale
    return (med, med) if med > 0 else (med, 0.0)


def robust_z(x: float, xs: Iterable[float]) -> float:
    """How many robust deviations `x` sits above the median of `xs`.

    Returns 0.0 only when the sample cannot support a score at all: fewer than
    two values, or a sample that is entirely zero.
    """
    a = _sorted(xs)
    if len(a) < 2:
        return 0.0
    med, scale = _scale(a)
    return (x - med) / scale if scale > 0 else 0.0


def robust_z_series(xs: Iterable[float]) -> list[float]:
    """Robust z for every value, computing the median and scale once.

    `robust_z(x, xs)` sorts `xs` on every call. Scoring a series with it is
    O(n^2 log n), which on 24,000 turns is a minute and a half of a report
    that should take a second. Anything scanning a whole workload wants this.
    """
    a = list(xs)
    if len(a) < 2:
        return [0.0] * len(a)
    med, scale = _scale(_sorted(a))
    if scale <= 0:
        return [0.0] * len(a)
    return [(x - med) / scale for x in a]


def share(part: float, whole: float) -> float:
    """`part/whole`, or 0.0 when the whole is zero. The division every report does."""
    return part / whole if whole else 0.0


def gini(xs: Iterable[float]) -> float:
    """Concentration of a cost distribution, 0 (even) to 1 (one session has it all).

    `trace` already reports "the top 25% of sessions are 80% of spend"; this is
    the same claim without the arbitrary cut point, and it is the number that
    says whether a per-session lever can matter at all. A workload with a Gini
    near 0 has no expensive sessions to fix.
    """
    a = _sorted(x for x in xs if x > 0)
    n = len(a)
    if n < 2:
        return 0.0
    total = sum(a)
    if total <= 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(a))
    return (2.0 * cum) / (n * total) - (n + 1.0) / n


def geometric_mean(xs: Iterable[float]) -> float:
    """Geometric mean of positive values; 0.0 if there are none.

    Ratios (a 2x under-estimate and a 2x over-estimate) average to 1.0 here and
    to 1.25 arithmetically. Anything averaging multiples wants this one.
    """
    a = [float(x) for x in xs if x > 0]
    if not a:
        return 0.0
    return math.exp(sum(math.log(x) for x in a) / len(a))


def trimmed_mean(xs: Iterable[float], trim: float = 0.1) -> float:
    """Mean with the top and bottom `trim` share dropped.

    For rates measured off transcripts, where one pathological session is
    common and one pathological session should not set a threshold.
    """
    if not 0.0 <= trim < 0.5:
        raise ValueError(f"trim must be in [0,0.5), got {trim}")
    a = _sorted(xs)
    if not a:
        return 0.0
    k = int(len(a) * trim)
    core = a[k: len(a) - k] or a
    return sum(core) / len(core)


# ---------------------------------------------------------------------------
# Uncertainty.
#
# Everything above returns a point estimate. A point estimate is what got the
# original 1.78x inflation past review: "cache hit rate is 91%" reads as a
# fact, and nobody asks how many turns it was computed from. Below this line
# every function returns an interval or a p-value, and the reports that make a
# comparison are expected to use one. The rule the rest of the package follows:
# a difference whose interval spans zero is reported as "no difference
# measured", not as a small difference.
#
# All resampling here takes an explicit `seed` and defaults it to a constant.
# A report that prints a different confidence interval on every run is a report
# that cannot be diffed in CI, and a bootstrap with an unseeded RNG is exactly
# that.
# ---------------------------------------------------------------------------

DEFAULT_SEED = 20260801
DEFAULT_RESAMPLES = 2000


def _z_for(alpha: float) -> float:
    """Two-sided normal critical value. Table lookup, then a fallback.

    `statistics.NormalDist().inv_cdf` exists and is exact; the table is here so
    the common alphas are exact *and* free, and so the intent of a bare 1.96 in
    a report is written down somewhere.
    """
    table = {0.10: 1.6448536269514722, 0.05: 1.959963984540054,
             0.01: 2.5758293035489004, 0.001: 3.2905267314919255}
    if alpha in table:
        return table[alpha]
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    from statistics import NormalDist
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def normal_cdf(x: float) -> float:
    """Standard normal CDF via `math.erf`, so no dependency and no table."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bootstrap_ci(
    xs: Sequence[float],
    stat=mean,
    *,
    alpha: float = 0.05,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap interval for any statistic of one sample.

    The percentile method rather than BCa: BCa needs a jackknife pass per
    resample, and for the sample sizes here (tens to thousands of turns) the
    bias correction moves the interval less than the width of the number we
    print. Cheap and honest beats precise and unrunnable.

    An empty sample returns `(0.0, 0.0)` rather than raising, matching
    `quantile`: every caller is a display path.
    """
    if resamples < 1:
        # A zero-width interval at zero is not "no resamples", it is a claim of
        # perfect precision about a number that was never estimated -- the
        # confident-wrong-number failure this whole package is built to avoid.
        raise ValueError(f"resamples must be at least 1, got {resamples}")
    a = [float(x) for x in xs]
    if not a:
        return (0.0, 0.0)
    if len(a) == 1:
        return (a[0], a[0])
    import random
    rng = random.Random(seed)
    n = len(a)
    draws = sorted(stat([a[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples))
    return (quantile(draws, alpha / 2.0), quantile(draws, 1.0 - alpha / 2.0))


def paired_bootstrap_ci(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    alpha: float = 0.05,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """Interval for `mean(ys) - mean(xs)` when the two samples are paired.

    Paired is the case that matters here: the A/B harness runs both arms on the
    *same* recorded turns, so resampling the arms independently throws away the
    pairing and inflates the interval enough to hide a real effect. Resample
    turn indices, not values.
    """
    if len(xs) != len(ys):
        raise ValueError(f"paired samples must match: {len(xs)} vs {len(ys)}")
    if not xs:
        return (0.0, 0.0)
    d = [float(b) - float(a) for a, b in zip(xs, ys, strict=True)]
    return bootstrap_ci(d, mean, alpha=alpha, resamples=resamples, seed=seed)


def permutation_test(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = DEFAULT_SEED,
) -> float:
    """Two-sided p for "these two samples have the same mean", by relabelling.

    No normality assumption, which matters because per-turn cost is
    lognormal-ish with a fat tail and a t-test on it is a coin flip. Returns
    the (r+1)/(n+1) corrected p so a p of exactly 0 is never reported.
    """
    if resamples < 1:
        # `(0+1)/(0+1)` is 1.0: a p-value of exactly "no evidence of a
        # difference", produced without performing a single relabelling.
        raise ValueError(f"resamples must be at least 1, got {resamples}")
    a, b = [float(x) for x in xs], [float(y) for y in ys]
    if not a or not b:
        return 1.0
    import random
    rng = random.Random(seed)
    observed = abs(mean(a) - mean(b))
    pool = a + b
    n_a = len(a)
    hits = 0
    for _ in range(resamples):
        rng.shuffle(pool)
        if abs(mean(pool[:n_a]) - mean(pool[n_a:])) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (resamples + 1)


def paired_permutation_test(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = DEFAULT_SEED,
) -> float:
    """Two-sided p for a paired difference, by flipping the sign of each pair.

    The A/B and simulate harnesses both compare arms on identical turns; this
    is the test that matches that design.
    """
    if len(xs) != len(ys):
        raise ValueError(f"paired samples must match: {len(xs)} vs {len(ys)}")
    if resamples < 1:
        raise ValueError(f"resamples must be at least 1, got {resamples}")
    d = [float(b) - float(a) for a, b in zip(xs, ys, strict=True)]
    if not d:
        return 1.0
    import random
    rng = random.Random(seed)
    observed = abs(mean(d))
    hits = 0
    for _ in range(resamples):
        flipped = [x if rng.random() < 0.5 else -x for x in d]
        if abs(mean(flipped)) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (resamples + 1)


def wilson_interval(k: int, n: int, *, alpha: float = 0.05) -> tuple[float, float]:
    """Interval for a rate `k/n`. Wilson, not normal-approximation.

    The normal approximation is wrong exactly where routing lives: a router
    that failed 0 of 12 times gets the interval `(0, 0)` from Wald, which is a
    confident claim of perfection from twelve observations. Wilson gives
    `(0, 0.24)`, which is the truth.
    """
    if n <= 0:
        return (0.0, 1.0)
    z = _z_for(alpha)
    # Clamp before dividing. `k` reaches here from outcome logs and A/B arm
    # counts that are summed in several places, and a `k` outside `[0, n]` made
    # `p * (1 - p)` negative and `math.sqrt` raise a domain error -- a crash
    # from a display path, in the one function whose whole job is to keep a
    # small sample from over-claiming.
    k = max(0, min(int(k), int(n)))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # Snap the exact endpoints. Without this a clean run reports a lower bound
    # of 2.8e-17 instead of 0, and the formatter renders that as "0.0000%" with
    # a footnote nobody can explain.
    lo = 0.0 if k <= 0 else max(0.0, centre - half)
    hi = 1.0 if k >= n else min(1.0, centre + half)
    return (lo, hi)


def proportion_diff_ci(
    k1: int, n1: int, k2: int, n2: int, *, alpha: float = 0.05
) -> tuple[float, float]:
    """Interval for `k2/n2 - k1/n1` (Newcombe's hybrid-score method).

    Built from two Wilson intervals rather than a pooled normal, for the same
    reason: the arms of an escalation A/B are routinely small and lopsided.
    """
    if n1 <= 0 or n2 <= 0:
        return (-1.0, 1.0)
    l1, u1 = wilson_interval(k1, n1, alpha=alpha)
    l2, u2 = wilson_interval(k2, n2, alpha=alpha)
    p1, p2 = k1 / n1, k2 / n2
    lo = (p2 - p1) - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
    hi = (p2 - p1) + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
    return (max(-1.0, lo), min(1.0, hi))


def anytime_ci(xs: Sequence[float], *, alpha: float = 0.05) -> tuple[float, float]:
    """Interval that stays valid when you look at it after every new sample.

    A fixed-sample 95% interval is only 95% if you decided the sample size
    first. Nobody does that here: the natural way to use `adder verify` is to
    re-run it as sessions accumulate and stop when it looks good, which is
    optional stopping and inflates the false-positive rate to somewhere near
    30%. This is an empirical-Bernstein confidence sequence -- wider than the
    fixed-n interval, and it does not care how often you peek.
    """
    a = [float(x) for x in xs]
    n = len(a)
    if n < 2:
        return (float("-inf"), float("inf"))
    m = mean(a)
    var = sum((x - m) ** 2 for x in a) / (n - 1)
    rng = (max(a) - min(a)) or 1.0
    # Howard et al. style radius: variance term plus a range term, both O(1/n)
    # up to the log(1/alpha) peeking penalty.
    log_term = math.log(2.0 / alpha)
    half = math.sqrt(2.0 * var * log_term / n) + 3.0 * rng * log_term / n
    return (m - half, m + half)


def bh_fdr(pvals: Sequence[float], *, q: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg: which of these p-values survive at FDR `q`.

    `adder doctor` runs on the order of twenty checks. At alpha=0.05 that is a
    64% chance of at least one false finding, and a tool that invents one
    expensive-looking problem per run is a tool people stop reading.
    """
    n = len(pvals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvals[i])
    keep = [False] * n
    threshold_rank = -1
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= q * rank / n:
            threshold_rank = rank
    for rank, i in enumerate(order, start=1):
        if rank <= threshold_rank:
            keep[i] = True
    return keep


def hedges_g(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Standardised effect size with the small-sample correction.

    Reported next to a p-value so a "significant" result on 4,000 turns cannot
    hide that the effect is a thousandth of a standard deviation.
    """
    a, b = [float(x) for x in xs], [float(y) for y in ys]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = mean(a), mean(b)
    v1 = sum((x - m1) ** 2 for x in a) / (n1 - 1)
    v2 = sum((x - m2) ** 2 for x in b) / (n2 - 1)
    pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled <= 0:
        return 0.0
    d = (m2 - m1) / pooled
    correction = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    return d * correction


def samples_needed(p_baseline: float, lift: float, *, alpha: float = 0.05,
                   power: float = 0.8) -> int:
    """Per-arm sample size to detect `lift` on a rate of `p_baseline`.

    The number `adder ab` prints before you spend money running an experiment,
    because the alternative -- running 30 tasks per arm and reading the result
    -- is how a lever gets adopted on noise.
    """
    p1 = min(max(p_baseline, 1e-9), 1 - 1e-9)
    p2 = min(max(p_baseline + lift, 1e-9), 1 - 1e-9)
    if abs(p2 - p1) < 1e-12:
        return 0
    from statistics import NormalDist
    z_a = _z_for(alpha)
    z_b = NormalDist().inv_cdf(power)
    pbar = (p1 + p2) / 2.0
    num = (z_a * math.sqrt(2 * pbar * (1 - pbar)) +
           z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p2 - p1) ** 2)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Rank correlation. Used wherever a proxy is checked against an outcome.

    Rank rather than Pearson because the quantities being related here -- arena
    rating against measured failure rate, context size against cost -- are
    monotone but not linear, and Pearson understates a monotone curve.
    """
    n = len(xs)
    if n != len(ys):
        raise ValueError(f"spearman needs equal lengths: {n} vs {len(ys)}")
    if n < 2:
        return 0.0
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else 0.0


def _ranks(xs: Sequence[float]) -> list[float]:
    """Average ranks, so ties do not silently become an arbitrary order."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def kendall_tau(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Tau-b: concordance between two orderings, tie-corrected.

    The right statistic for "does this cheap proxy rank models the same way the
    expensive measurement does", which is the question every quality proxy in
    this package has to answer.
    """
    n = len(xs)
    if n != len(ys):
        raise ValueError(f"kendall_tau needs equal lengths: {n} vs {len(ys)}")
    if n < 2:
        return 0.0
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0 and dy == 0:
                tx += 1
                ty += 1
            elif dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx * dy > 0:
                conc += 1
            else:
                disc += 1
    n0 = n * (n - 1) / 2
    den = math.sqrt((n0 - tx) * (n0 - ty))
    return (conc - disc) / den if den > 0 else 0.0
