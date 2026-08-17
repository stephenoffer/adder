"""Separate what a model knows from how much it writes, and price the difference.

The confound, and why it is worse for a cost tool than for a leaderboard
------------------------------------------------------------------------
Head-to-head preference data has a known bias: judges prefer longer, more
heavily formatted answers, somewhat independently of whether they are better.
The established fix is to put style features -- response length, markdown
headers, list count -- into the Bradley-Terry regression as covariates, so the
strength coefficient reflects capability rather than the reader's taste for
bullet points.

On a leaderboard that is a fairness correction: the ranking stops rewarding
padding. Here it is a **cost** correction, and it bites twice as hard, because
the thing that inflated the rating is the same thing this tool bills you for:

* the extra tokens are billed as output on the turn that produced them; and
* they then sit in the context and are re-read as prefix on every turn after,
  which for a long session is the larger of the two.

So routing on an uncontrolled rating pays a premium *for the property that
inflated the number*. That is not a rounding error in a cost model, it is the
model recommending the expensive option for a reason it believes is quality.

What this module computes
-------------------------
A Bradley-Terry fit with style covariates:

    logit P(i beats j) = theta_i - theta_j + sum_k beta_k * (s_k(i) - s_k(j))

`theta` is the style-controlled strength. `beta` is how much the judges paid for
each unit of style, and it is worth reading on its own -- a large positive
length coefficient means your evaluation is substantially a length contest.

The difference between a model's uncontrolled and controlled strength is its
**style premium**: the rating it earns by writing more rather than by being
better. `premium_cost` turns that into dollars per answer using the same rates
as every other report here.

Honesty, inherited
------------------
This is observational, and the original analysis says so. Length correlates with
substance: a model may write more *because* it is doing more work, and the
regression will strip that out along with the padding. The controlled number is
therefore a **lower** bound on capability, not a corrected measurement of it, and
anything printed from it says so. It is still the right number to route on,
because a lower bound on capability paired with an exact number for cost is the
conservative direction for a spending decision.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from adder.pricing.bt import ANCHOR, SCALE, Battle, fit
from adder.util.stats import mean

# Style features, in the order their coefficients are reported. Chosen to match
# what the published control uses: length dominates, and the markdown counts
# capture the "wall of headers" effect that length alone misses.
FEATURES: tuple[str, ...] = ("length", "headers", "lists", "bold")

# Newton steps for the covariate fit. It is a low-dimensional convex problem
# (one coefficient per feature), so this converges in well under ten.
MAX_ITER = 60
TOL = 1e-9


@dataclass(frozen=True)
class Style:
    """Style measurements for one response.

    Counts, not ratios: the regression takes differences between the two sides
    of a battle, and normalising each side first would throw away the magnitude
    that makes a difference meaningful.
    """

    tokens: int = 0
    headers: int = 0
    lists: int = 0
    bold: int = 0

    def vector(self) -> tuple[float, ...]:
        # Length enters on a log scale. Judges do not prefer a 4,000-token
        # answer over a 2,000-token one as much as they prefer 2,000 over
        # 1,000, and a linear term fits the tail at the expense of the range
        # where nearly all the data is.
        return (
            math.log1p(max(0, self.tokens)),
            float(self.headers),
            float(self.lists),
            float(self.bold),
        )


def measure(text: str) -> Style:
    """Style features of a response, counted from its markdown.

    Deliberately crude. The point is not to parse markdown correctly, it is to
    capture the three things a reader visibly rewards: how much there is, how
    chunked it looks, and how much of it is shouting.
    """
    if not text:
        return Style()
    lines = text.splitlines()
    headers = sum(1 for ln in lines if ln.lstrip().startswith("#"))
    lists = sum(1 for ln in lines
                if ln.lstrip()[:2] in ("- ", "* ", "+ ")
                or (ln.lstrip()[:2].rstrip(".").isdigit() and ". " in ln[:5]))
    bold = text.count("**") // 2
    # Four characters per token is the same estimator the context reports use.
    return Style(tokens=max(1, len(text) // 4), headers=headers,
                 lists=lists, bold=bold)


@dataclass
class Controlled:
    """A style-controlled fit: strengths, and what style was worth."""

    strength: dict[str, float] = field(default_factory=dict)
    uncontrolled: dict[str, float] = field(default_factory=dict)
    beta: dict[str, float] = field(default_factory=dict)
    # Bootstrap interval per coefficient. Empty when it was not requested, and
    # `length_matters` is False in that case rather than falling back to a bare
    # threshold -- a cutoff on a point estimate fires on noise, which is how an
    # earlier version reported a length effect from a judge that had none
    # (beta = 0.061 against a threshold of 0.05).
    beta_ci: dict[str, tuple[float, float]] = field(default_factory=dict)
    battles: int = 0
    converged: bool = False
    # False when style never varied within a matchup, so style and skill are
    # collinear and the coefficients mean nothing.
    identified: bool = True

    def premium(self, model: str) -> float:
        """Rating points this model earns from style rather than capability.

        Positive means the uncontrolled board flatters it. Negative means the
        opposite -- a terse model that is better than it looks, which is the
        one a cost-driven router should be most interested in.
        """
        return self.uncontrolled.get(model, 0.0) - self.strength.get(model, 0.0)

    @property
    def length_matters(self) -> bool:
        """Whether the judges measurably rewarded length, on the interval.

        Three ways this is False, and they are different answers: the effect is
        absent, the fit is unidentified because style never varied within a
        matchup, or no interval was computed. None of the three licences the
        claim, so none of them returns True.
        """
        if not self.identified:
            return False
        lo, _hi = self.beta_ci.get("length", (0.0, 0.0))
        return lo > 0.0

    @property
    def measured(self) -> bool:
        """Whether an interval was computed at all."""
        return bool(self.beta_ci)

    def to_json(self) -> dict:
        return {
            "battles": self.battles,
            "converged": self.converged,
            "identified": self.identified,
            "beta": dict(self.beta),
            "beta_ci95": {k: list(v) for k, v in self.beta_ci.items()},
            "length_matters": self.length_matters,
            "models": {
                m: {"controlled": self.strength.get(m, 0.0),
                    "uncontrolled": self.uncontrolled.get(m, 0.0),
                    "style_premium": self.premium(m)}
                for m in sorted(self.strength)
            },
        }


def fit_controlled(
    battles: Sequence[Battle],
    styles: Sequence[tuple[Style, Style]],
    *,
    max_iter: int = MAX_ITER,
    tol: float = TOL,
    ridge: float = 1e-3,
    resamples: int = 0,
    seed: int = 20260801,
) -> Controlled:
    """Bradley-Terry with style covariates, fitted jointly by Newton-Raphson.

    The model is an ordinary logistic regression with no intercept: one column
    per model (+1 for the left side, -1 for the right) and one per style
    feature (the difference between the two responses). Fitting the strengths
    and the coefficients *jointly* is the point -- an earlier version alternated
    between the two blocks and had to round a continuous residual back into a
    discrete winner to reuse the plain fit, which threw away exactly the
    information the coefficients are estimated from. It returned a length
    coefficient of zero on data generated to have one.

    `ridge` is a small quadratic penalty. It is not regularisation for its own
    sake: strengths are only identified up to an additive constant, and style
    features are frequently collinear with a model (a model that always writes
    long, against opponents that always write short, cannot have its length
    effect separated from its skill). The penalty makes the system solvable and
    pushes unidentified directions toward zero rather than toward whatever the
    numerical noise suggests. `identified` reports when that has happened, so
    the caller can decline to draw a conclusion.
    """
    if len(battles) != len(styles):
        raise ValueError(
            f"need one style pair per battle: {len(battles)} vs {len(styles)}")
    result = Controlled(battles=len(battles))
    if not battles:
        return result

    result.uncontrolled = fit(battles)
    models = sorted(result.uncontrolled)
    index = {m: i for i, m in enumerate(models)}
    n_m, n_f = len(models), len(FEATURES)
    dim = n_m + n_f

    rows: list[tuple[list[float], float]] = []
    for b, (sa, sb) in zip(battles, styles, strict=True):
        x = [0.0] * dim
        x[index[b.a]] = 1.0
        x[index[b.b]] = -1.0
        va, vb = sa.vector(), sb.vector()
        for k in range(n_f):
            x[n_m + k] = va[k] - vb[k]
        rows.append((x, _outcome_of(b)))

    theta = [0.0] * dim
    for _ in range(max_iter):
        grad = [-ridge * t for t in theta]
        hess = [[0.0] * dim for _ in range(dim)]
        for i in range(dim):
            hess[i][i] = -ridge
        for x, y in rows:
            eta = sum(xi * ti for xi, ti in zip(x, theta, strict=True))
            p = _sigmoid(eta)
            w = max(p * (1.0 - p), 1e-9)
            resid = y - p
            nz = [i for i, xi in enumerate(x) if xi]
            for i in nz:
                grad[i] += resid * x[i]
                for j in nz:
                    hess[i][j] -= w * x[i] * x[j]
        step = _solve(hess, grad)
        if step is None:
            break
        theta = [t - s_ for t, s_ in zip(theta, step, strict=True)]
        if max(abs(v) for v in step) < tol:
            result.converged = True
            break

    # Strengths are identified only up to a constant; centre them and put them
    # on the same 400-point scale the uncontrolled fit uses.
    raw = theta[:n_m]
    centre = mean(raw)
    result.strength = {m: ANCHOR + SCALE * (raw[index[m]] - centre) for m in models}
    result.beta = dict(zip(FEATURES, theta[n_m:], strict=True))
    result.identified = _style_varies(rows, n_m, n_f)
    if resamples > 0:
        result.beta_ci = _beta_ci(battles, styles, resamples=resamples, seed=seed,
                                  max_iter=max_iter, tol=tol, ridge=ridge)
    return result


def _beta_ci(
    battles: Sequence[Battle],
    styles: Sequence[tuple[Style, Style]],
    *,
    resamples: int,
    seed: int,
    max_iter: int,
    tol: float,
    ridge: float,
) -> dict[str, tuple[float, float]]:
    """Percentile interval per coefficient, resampling battles.

    A coefficient without an interval is a number that fires a threshold on
    noise. The battle is the unit that was drawn, so the battle is the unit
    that gets resampled.
    """
    import random

    from adder.util.stats import quantile

    rng = random.Random(seed)
    n = len(battles)
    draws: dict[str, list[float]] = {f: [] for f in FEATURES}
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        sample_b = [battles[i] for i in idx]
        sample_s = [styles[i] for i in idx]
        try:
            fitted = fit_controlled(sample_b, sample_s, max_iter=max_iter,
                                    tol=tol, ridge=ridge)
        except ValueError:
            continue
        for f in FEATURES:
            draws[f].append(fitted.beta.get(f, 0.0))
    return {f: (quantile(v, 0.025), quantile(v, 0.975)) for f, v in draws.items() if v}


def _style_varies(rows: list[tuple[list[float], float]], n_m: int, n_f: int) -> bool:
    """Whether style differences vary independently of who is playing.

    If every battle between a given pair carries the same style difference,
    style and skill are collinear and no fit can separate them. Detecting that
    is the difference between reporting "length did not matter" and reporting
    "this data cannot say".
    """
    seen: dict[tuple[int, ...], set[tuple[float, ...]]] = {}
    for x, _ in rows:
        pair = tuple(i for i, v in enumerate(x[:n_m]) if v)
        seen.setdefault(pair, set()).add(tuple(x[n_m:n_m + n_f]))
    return any(len(v) > 1 for v in seen.values())


def _sigmoid(x: float) -> float:
    """Logistic, clamped so a confident fit cannot overflow."""
    if x < -60:
        return 0.0
    if x > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def _outcome_of(b: Battle) -> float:
    """Observed result as a number: 1 for a win, 0 for a loss, 0.5 for a tie.

    A fractional response is legitimate for a binomial likelihood, so ties need
    no special case and no dropping -- which matters, because a tie is the most
    common outcome between two models of similar strength and dropping it would
    discard exactly the comparisons the control exists to correct.
    """
    return 1.0 if b.winner == "a" else 0.0 if b.winner == "b" else 0.5


def _solve(hess: list[list[float]], grad: list[float]) -> list[float] | None:
    """Solve `hess @ step = grad` by Gaussian elimination. None if singular.

    Singular should not happen now that the fit carries a ridge penalty, but a
    return of None rather than an exception keeps a degenerate input producing
    a report instead of a traceback.
    """
    n = len(grad)
    a = [[*row, grad[i]] for i, row in enumerate(hess)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(n):
            if r == col:
                continue
            factor = a[r][col] / a[col][col]
            for c in range(col, n + 1):
                a[r][c] -= factor * a[col][c]
    return [a[i][n] / a[i][i] for i in range(n)]


def premium_cost(
    premium_points: float,
    *,
    extra_tokens: float,
    out_rate: float,
    cache_read_rate: float,
    remaining_turns: int,
) -> float:
    """Dollars per answer that a model's style premium actually costs.

    Both halves, because the second is usually the larger: the extra tokens are
    billed once as output, and then re-read as prefix on every remaining turn.
    A model whose rating advantage is 300 extra tokens per answer is not
    slightly more expensive, it is more expensive on every turn for the rest of
    the session.

    `premium_points` is carried only to make the caller state it; the cost
    depends on the tokens, not on the rating.
    """
    del premium_points
    if extra_tokens <= 0:
        return 0.0
    once = extra_tokens * out_rate / 1_000_000.0
    carried = extra_tokens * cache_read_rate * max(0, remaining_turns) / 1_000_000.0
    return once + carried


def mean_style(styles: Sequence[Style]) -> Style:
    """Average style of a set of responses, for per-model summaries."""
    if not styles:
        return Style()
    return Style(
        tokens=int(mean([float(s.tokens) for s in styles])),
        headers=round(mean([float(s.headers) for s in styles])),
        lists=round(mean([float(s.lists) for s in styles])),
        bold=round(mean([float(s.bold) for s in styles])),
    )
