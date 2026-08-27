"""The uncertainty layer. Every number here is checkable against a known value,
which is the point: a special function written out by hand to avoid a dependency
has to be pinned to something other than its own output."""
from __future__ import annotations

import math
from itertools import pairwise

import pytest

from adder.util.risk import (
    DEFAULT_CONFIDENCE,
    Interval,
    beta_bounds,
    beta_from_mean,
    beta_quantile,
    beta_quantiles,
    betainc,
    bounds_from_mean,
    empirical_bounds,
    empirical_quantiles,
    guarantee,
    p_cheaper,
    quantiles_from_mean,
    shrink,
    worst_case,
)


class TestIncompleteBeta:
    """Pinned against closed forms and against scipy-computed values."""

    @pytest.mark.parametrize("a,b,x,want", [
        (2, 3, 0.5, 0.6875),          # closed form: 1 - (1-x)^3 (1 + 3x)
        (1, 1, 0.37, 0.37),           # uniform
        (0.5, 0.5, 0.5, 0.5),         # arcsine, symmetric
        (5, 2, 0.9, 0.885735),
        (3, 7, 0.2, 0.26180250),   # binomial identity: sum_{j>=3} C(9,j) .2^j .8^{9-j}
    ])
    def test_known_values(self, a, b, x, want):
        assert betainc(a, b, x) == pytest.approx(want, abs=1e-6)

    def test_bounds_of_the_domain(self):
        assert betainc(2, 3, 0.0) == 0.0
        assert betainc(2, 3, 1.0) == 1.0

    def test_is_monotone(self):
        xs = [i / 50 for i in range(51)]
        vals = [betainc(2.5, 4.5, x) for x in xs]
        assert all(b >= a for a, b in pairwise(vals))

    def test_rejects_nonpositive_parameters(self):
        with pytest.raises(ValueError):
            betainc(0.0, 1.0, 0.5)

    def test_quantile_inverts_the_cdf(self):
        for q in (0.01, 0.1, 0.5, 0.9, 0.99):
            x = beta_quantile(q, 3.0, 7.0)
            assert betainc(3.0, 7.0, x) == pytest.approx(q, abs=1e-9)


class TestBetaBounds:
    def test_interval_brackets_the_mean(self):
        b = beta_bounds(2, 10)
        assert b.lo < b.point < b.hi

    def test_more_evidence_narrows_it(self):
        """The whole reason a mean is not enough: 20% over 10 runs and 20% over
        1,000 runs are the same number and different instructions."""
        thin = beta_bounds(2, 10)
        thick = beta_bounds(200, 1_000)
        assert thick.width < thin.width / 5

    def test_no_evidence_is_nearly_the_whole_unit_interval(self):
        b = beta_bounds(0, 0)
        assert b.lo < 0.06 and b.hi > 0.94 and b.point == pytest.approx(0.5)

    def test_accepts_fractional_weights(self):
        """Recency weighting produces fractional counts; half-weight evidence
        should widen the interval, not be rejected."""
        full = beta_bounds(4.0, 20.0)
        half = beta_bounds(2.0, 10.0)
        assert half.width > full.width

    def test_rejects_more_failures_than_trials(self):
        with pytest.raises(ValueError):
            beta_bounds(5, 2)


class TestMomentMatching:
    def test_mean_is_preserved_exactly(self):
        """`beta_bounds` adds a Beta(1,1); this must not, or the gate's midpoint
        stops matching the number reported beside it."""
        for m in (0.05, 0.15, 0.4, 0.9):
            a, b = beta_from_mean(m, 4.0)
            assert a / (a + b) == pytest.approx(m)
            assert bounds_from_mean(m, 4.0).point == pytest.approx(m)

    def test_more_pseudo_count_narrows(self):
        assert bounds_from_mean(0.2, 40.0).width < bounds_from_mean(0.2, 4.0).width

    def test_quantile_ladder_averages_to_about_the_mean(self):
        qs = quantiles_from_mean(0.25, 8.0, strata=16)
        assert sum(qs) / len(qs) == pytest.approx(0.25, abs=0.03)


class TestEmpiricalBounds:
    def test_uses_the_mean_not_the_median(self):
        """Heavy-tailed by construction: the mean must sit far above the median,
        because cost is linear in the quantity being summarised."""
        xs = [10] * 9 + [1000]
        b = empirical_bounds(xs)
        assert b.point == pytest.approx(109.0)
        assert b.point > sorted(xs)[len(xs) // 2]

    def test_rejects_an_empty_sample(self):
        with pytest.raises(ValueError):
            empirical_bounds([])

    def test_quantile_ladder_is_sorted_and_sized(self):
        qs = empirical_quantiles([5, 1, 9, 3], strata=4)
        assert len(qs) == 4 and qs == sorted(qs)


class TestWorstCase:
    def test_multilinear_extrema_are_at_vertices(self):
        """The claim `worst_case` rests on, checked rather than asserted.

        A multilinear function is affine along each axis with the others held
        fixed, so its minimum over a box is attained at a corner. If that were
        false, enumerating 2^k corners would be an approximation rather than the
        answer, and every guarantee built on it would be optimistic.
        """
        def f(x, y, z):
            return 3 * x * y - 2 * y * z + x - 5 * z + 1

        bounds = {"x": Interval(0, 1, 2), "y": Interval(-1, 0, 3), "z": Interval(1, 2, 4)}
        best, _ = worst_case(f, bounds)
        # Dense interior sweep must never beat the corner enumeration.
        n = 21
        grid_min = min(
            f(x=0 + 2 * i / (n - 1), y=-1 + 4 * j / (n - 1), z=1 + 3 * k / (n - 1))
            for i in range(n) for j in range(n) for k in range(n)
        )
        assert best <= grid_min + 1e-9

    def test_reports_the_losing_corner(self):
        best, corner = worst_case(lambda r: r * 2.0 - 1.0, {"r": Interval(0.1, 1.0, 5.0)})
        assert best == pytest.approx(-0.8) and corner == {"r": 0.1}

    def test_no_bounds_is_a_plain_call(self):
        assert worst_case(lambda: 7.0, {}) == (7.0, {})


class TestProbability:
    def test_always_positive_gives_one(self):
        assert p_cheaper(lambda r: r, {"r": [1.0, 2.0, 3.0]}) == 1.0

    def test_never_positive_gives_zero(self):
        assert p_cheaper(lambda r: -r, {"r": [1.0, 2.0]}) == 0.0

    def test_counts_the_share_of_the_grid(self):
        assert p_cheaper(lambda r: r - 2.5, {"r": [1.0, 2.0, 3.0, 4.0]}) == 0.5

    def test_is_deterministic(self):
        """No RNG anywhere in it, so two calls cannot disagree."""
        f = lambda r, p: r * 0.01 - p  # noqa: E731
        marg = {"r": empirical_quantiles([10, 50, 200, 900]), "p": beta_quantiles(2, 10)}
        assert p_cheaper(f, marg) == p_cheaper(f, marg)

    def test_rejects_an_empty_marginal(self):
        with pytest.raises(ValueError):
            p_cheaper(lambda r: r, {"r": []})


class TestGuarantee:
    def _g(self, **kw):
        return guarantee(
            lambda R, p: R * 0.001 - p * 0.5,
            {"R": Interval(50, 400, 900), "p": Interval(0.05, 0.2, 0.6)},
            marginals={"R": empirical_quantiles([50, 100, 200, 400, 900]),
                       "p": beta_quantiles(2, 10)},
            **kw,
        )

    def test_probability_is_of_clearing_overhead_not_zero(self):
        """Advice that saves a cent reliably and costs a turn every time is a
        losing proposition; the integral has to be against the right threshold."""
        cheap = self._g(overhead=0.0)
        dear = self._g(overhead=0.30)
        assert cheap.confidence > dear.confidence

    def test_safe_requires_both_expectation_and_confidence(self):
        g = self._g(overhead=0.0, threshold=1.01)
        assert g.expected > 0 and not g.safe

    def test_dominant_is_strictly_stronger_than_safe(self):
        g = self._g(overhead=0.0)
        assert not (g.dominant and not g.safe)

    def test_default_threshold_is_a_coin_flip_not_certainty(self):
        """A router taking hundreds of decisions is paid by the sum, so
        declining positive-expectation advice loses money over the sequence."""
        assert DEFAULT_CONFIDENCE == 0.50

    def test_describe_names_the_losing_corner(self):
        g = self._g(overhead=5.0)
        assert "R=50" in g.describe() and "loses money" in g.describe()

    def test_a_pinned_input_is_not_named_as_a_condition(self):
        """`--remaining 0` produced "it would lose money only if remaining=0".

        The corner is meant to name what would have to go wrong. An input with
        no width has already gone as wrong as it can, and it is already in the
        expected number, so naming it reads as a hypothetical the caller could
        avoid. Only the axes that still carry uncertainty belong there.
        """
        g = guarantee(
            lambda R, p: 1.0 - p - R * 0.0,
            {"R": Interval.exact(0.0), "p": Interval(0.05, 0.2, 0.6)},
            marginals={"R": [0.0], "p": beta_quantiles(2, 10)}, overhead=0.5,
        )
        assert g.safe and not g.dominant
        out = g.describe()
        assert "R=" not in out
        assert "p=" in out

    def test_a_pinned_decline_still_explains_itself(self):
        """Every axis pinned and the saving under the bar: no corner to blame."""
        g = guarantee(lambda R: R * 0.0 + 0.01, {"R": Interval.exact(3.0)},
                      overhead=1.0, marginals={"R": [3.0]})
        assert not g.safe
        assert "not confident enough" in g.describe()
        assert "loses money when" not in g.describe()


class TestInterval:
    def test_rejects_out_of_order(self):
        with pytest.raises(ValueError):
            Interval(1.0, 0.0, 2.0)

    def test_exact_has_no_width(self):
        assert Interval.exact(3.0).width == 0.0


class TestShrink:
    def test_a_thin_sample_stays_near_the_prior(self):
        assert shrink([1.0], 0.0, weight=10.0) == pytest.approx(1 / 11)

    def test_a_thick_sample_wins(self):
        assert shrink([1.0] * 1000, 0.0, weight=10.0) > 0.99

    def test_no_sample_is_the_prior(self):
        assert shrink([], 0.42, weight=10.0) == 0.42


def test_no_dependency_on_scipy():
    """The reason all of the above exists. If this ever passes trivially because
    someone imported scipy, the empty-dependencies rule has been broken."""
    import adder.util.risk as r

    assert not any(m in getattr(r, "__dict__", {}) for m in ("scipy", "numpy"))
    assert math.isfinite(betainc(2, 3, 0.5))


class TestEmpiricalBoundsOnAHeavyTail:
    """The mean can sit outside its own quantile interval, and here it usually does.

    `empirical_bounds` is documented as the estimator for session length --
    heavy-tailed, mean well above median, which is the whole reason it returns
    the mean rather than the median. On that shape the mean can exceed the upper
    quantile, and the unclamped constructor raised `ValueError: interval out of
    order` at exactly the distribution the function exists for. `beta_bounds`
    and `bounds_from_mean` already clamped; this one did not.
    """

    def test_a_mean_above_the_upper_quantile_does_not_raise(self):
        heavy = [1.0] * 100 + [10_000.0]
        got = empirical_bounds(heavy)
        assert got.point == pytest.approx(sum(heavy) / len(heavy))

    def test_the_interval_still_brackets_the_point(self):
        got = empirical_bounds([1.0] * 100 + [10_000.0])
        assert got.lo <= got.point <= got.hi

    def test_a_mean_below_the_lower_quantile_does_not_raise(self):
        """The mirror case: a heavy LEFT tail drags the mean under the p5."""
        got = empirical_bounds([10_000.0] * 100 + [1.0])
        assert got.lo <= got.point <= got.hi

    def test_a_well_behaved_sample_is_unchanged(self):
        """The clamp must not move an interval that was already ordered."""
        got = empirical_bounds([10.0, 20.0, 30.0, 40.0, 50.0])
        assert got.point == pytest.approx(30.0)
        assert got.lo < got.point < got.hi
