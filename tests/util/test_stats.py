"""The quantile estimator, pinned.

`trace` used to index a sorted list at `int(len*p)`, which returns the maximum
for p=0.9 on ten samples. These tests exist so that estimator cannot come back
by accident: a p90 that is really the max makes every "typical session" claim
in the README too large.
"""

from __future__ import annotations

import pytest

from adder.util import stats
from adder.util.stats import (
    MAD_TO_SIGMA,
    geometric_mean,
    gini,
    mad,
    mean,
    median,
    quantile,
    quantiles,
    robust_z,
    robust_z_series,
    share,
    trimmed_mean,
    wilson_interval,
)


class TestQuantile:
    def test_empty_is_zero_not_an_exception(self):
        assert quantile([], 0.5) == 0.0
        assert median([]) == 0.0
        assert mean([]) == 0.0

    def test_single_value(self):
        assert quantile([7], 0.0) == 7
        assert quantile([7], 1.0) == 7

    def test_endpoints_are_min_and_max(self):
        xs = [3, 1, 4, 1, 5, 9, 2, 6]
        assert quantile(xs, 0.0) == 1
        assert quantile(xs, 1.0) == 9

    def test_interpolates(self):
        # Midpoint of a 2-sample set is the average, not either sample.
        assert quantile([0, 10], 0.5) == pytest.approx(5.0)
        assert quantile([0, 10], 0.25) == pytest.approx(2.5)

    def test_p90_of_ten_is_not_the_maximum(self):
        """The nearest-rank bug, as an assertion."""
        xs = list(range(1, 11))          # 1..10
        assert quantile(xs, 0.9) == pytest.approx(9.1)
        assert quantile(xs, 0.9) < max(xs)

    def test_median_and_quantile_agree(self):
        xs = [5, 1, 9, 3, 7]
        assert median(xs) == quantile(xs, 0.5)

    def test_matches_statistics_median_on_even_length(self):
        import statistics

        xs = [1.0, 2.0, 3.0, 4.0]
        assert median(xs) == pytest.approx(statistics.median(xs))

    @pytest.mark.parametrize("q", [-0.01, 1.01])
    def test_out_of_range_rejected(self, q):
        with pytest.raises(ValueError):
            quantile([1, 2, 3], q)

    def test_quantiles_returns_one_per_request(self):
        assert len(quantiles([1, 2, 3, 4], (0.1, 0.5, 0.9))) == 3

    def test_unsorted_input_is_sorted_first(self):
        assert median([9, 1, 5]) == 5


class TestRobustScale:
    def test_mad_of_constant_is_zero(self):
        assert mad([4, 4, 4, 4]) == 0.0

    def test_one_outlier_does_not_move_mad(self):
        base = [10] * 20 + [11] * 20
        assert mad(base) == pytest.approx(mad([*base, 10_000]), abs=0.6)

    def test_robust_z_flags_the_outlier(self):
        xs = [10, 11, 9, 10, 12, 8, 10, 11, 500]
        assert robust_z(500, xs) > 10
        assert abs(robust_z(10, xs)) < 1

    def test_constant_data_scores_every_point_at_zero(self):
        assert robust_z(3, [3, 3, 3]) == 0.0
        assert robust_z_series([3, 3, 3]) == [0.0, 0.0, 0.0]

    def test_a_sample_too_small_to_score_is_zero(self):
        assert robust_z(5, [3]) == 0.0
        assert robust_z_series([3]) == [0.0]

    def test_an_all_zero_sample_has_no_scale(self):
        assert robust_z(5, [0, 0, 0]) == 0.0
        assert robust_z_series([0, 0, 0, 9]) == [0.0] * 4

    def test_zero_dispersion_falls_back_to_the_median_as_scale(self):
        """MAD is zero whenever most of the sample is identical. Scoring
        everything 0 there would make the detector miss the obvious outlier."""
        xs = [10] * 40 + [900]
        assert robust_z(900, xs) == pytest.approx((900 - 10) / 10)
        assert robust_z(10, xs) == 0.0

    def test_series_matches_the_scalar_form(self):
        xs = [10, 11, 9, 10, 12, 8, 10, 11, 500]
        assert robust_z_series(xs) == pytest.approx([robust_z(x, xs) for x in xs])

    def test_mad_to_sigma_is_the_standard_constant(self):
        assert pytest.approx(1.4826) == MAD_TO_SIGMA


class TestConcentration:
    def test_gini_of_equal_shares_is_zero(self):
        assert gini([5, 5, 5, 5]) == pytest.approx(0.0, abs=1e-9)

    def test_gini_rises_with_concentration(self):
        even = gini([10, 10, 10, 10])
        skewed = gini([1, 1, 1, 100])
        assert skewed > even
        assert 0.0 <= skewed <= 1.0

    def test_gini_of_one_session_is_zero(self):
        """A single session cannot be unequal with itself."""
        assert gini([42]) == 0.0

    def test_gini_ignores_non_positive(self):
        assert gini([0, 0, 5, 5]) == pytest.approx(gini([5, 5]))


class TestMeans:
    def test_geometric_mean_of_reciprocal_pair_is_one(self):
        assert geometric_mean([2.0, 0.5]) == pytest.approx(1.0)

    def test_geometric_mean_ignores_non_positive(self):
        assert geometric_mean([0, -1]) == 0.0

    def test_trimmed_mean_drops_the_tails(self):
        xs = [*range(10), 1000]
        assert trimmed_mean(xs, 0.1) < mean(xs)

    def test_trimmed_mean_of_short_input_keeps_everything(self):
        assert trimmed_mean([1, 2, 3], 0.4) == pytest.approx(2.0)

    def test_trim_must_be_below_a_half(self):
        with pytest.raises(ValueError):
            trimmed_mean([1, 2], 0.5)


class TestShare:
    def test_zero_whole_is_zero_not_a_crash(self):
        assert share(3, 0) == 0.0

    def test_normal_case(self):
        assert share(1, 4) == 0.25


# --- uncertainty -----------------------------------------------------------

def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    xs = [float(i) for i in range(100)]
    lo, hi = stats.bootstrap_ci(xs)
    assert lo < stats.mean(xs) < hi
    assert (lo, hi) == stats.bootstrap_ci(xs)


def test_bootstrap_ci_narrows_as_the_sample_grows():
    small = stats.bootstrap_ci([1.0, 2.0, 3.0, 4.0] * 5)
    large = stats.bootstrap_ci([1.0, 2.0, 3.0, 4.0] * 200)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_bootstrap_ci_handles_degenerate_samples():
    assert stats.bootstrap_ci([]) == (0.0, 0.0)
    assert stats.bootstrap_ci([7.0]) == (7.0, 7.0)


def test_paired_bootstrap_keeps_the_pairing():
    # A constant +1 shift on every pair: the paired interval must exclude zero
    # even though the two samples individually overlap completely.
    xs = [float(i) for i in range(50)]
    ys = [x + 1.0 for x in xs]
    lo, hi = stats.paired_bootstrap_ci(xs, ys)
    assert lo > 0.0
    assert abs((lo + hi) / 2 - 1.0) < 0.05


def test_paired_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        stats.paired_bootstrap_ci([1.0], [1.0, 2.0])


def test_permutation_test_separates_signal_from_noise():
    a = [0.0, 0.1, -0.1, 0.05, -0.05] * 8
    b = [5.0, 5.1, 4.9, 5.05, 4.95] * 8
    assert stats.permutation_test(a, b, resamples=2000) < 0.01
    assert stats.permutation_test(a, a, resamples=2000) > 0.5


def test_permutation_p_is_never_zero():
    a = [0.0] * 20
    b = [100.0] * 20
    assert stats.permutation_test(a, b, resamples=500) > 0.0


def test_paired_permutation_detects_a_consistent_shift():
    xs = [float(i) for i in range(30)]
    ys = [x + 0.5 for x in xs]
    assert stats.paired_permutation_test(xs, ys, resamples=2000) < 0.01


def test_wilson_never_claims_certainty_from_a_clean_run():
    lo, hi = stats.wilson_interval(0, 12)
    assert lo == 0.0
    assert 0.15 < hi < 0.30          # not (0, 0), which is what Wald returns


def test_wilson_tightens_with_evidence():
    narrow = stats.wilson_interval(500, 1000)
    wide = stats.wilson_interval(5, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_on_no_data_is_the_whole_interval():
    assert stats.wilson_interval(0, 0) == (0.0, 1.0)


def test_proportion_diff_ci_spans_zero_when_the_rates_match():
    lo, hi = stats.proportion_diff_ci(10, 20, 10, 20)
    assert lo < 0.0 < hi


def test_proportion_diff_ci_excludes_zero_on_a_large_clear_gap():
    lo, _hi = stats.proportion_diff_ci(100, 1000, 500, 1000)
    assert lo > 0.0


def test_anytime_ci_is_wider_than_the_fixed_sample_interval():
    xs = [float(i % 10) for i in range(200)]
    a_lo, a_hi = stats.anytime_ci(xs)
    b_lo, b_hi = stats.bootstrap_ci(xs)
    assert (a_hi - a_lo) > (b_hi - b_lo)
    assert a_lo < stats.mean(xs) < a_hi


def test_anytime_ci_is_uninformative_below_two_samples():
    lo, hi = stats.anytime_ci([1.0])
    assert lo == float("-inf") and hi == float("inf")


def test_bh_fdr_keeps_the_strong_and_drops_the_marginal():
    keep = stats.bh_fdr([0.001, 0.04, 0.6, 0.8, 0.9])
    assert keep[0] is True
    assert keep[2] is False and keep[3] is False


def test_bh_fdr_is_stricter_than_a_bare_alpha():
    # One finding at p=0.04 among twenty checks clears a bare alpha of 0.05 and
    # is exactly the false positive `doctor` would otherwise print every run.
    pvals = [0.04] + [0.9] * 19
    assert not any(stats.bh_fdr(pvals, q=0.05))
    # ...but the same p-value with nothing else competing does survive.
    assert stats.bh_fdr([0.04], q=0.05) == [True]


def test_bh_fdr_on_empty_input():
    assert stats.bh_fdr([]) == []


def test_hedges_g_signs_and_scales():
    a = [0.0, 1.0, 2.0, 3.0, 4.0]
    b = [10.0, 11.0, 12.0, 13.0, 14.0]
    assert stats.hedges_g(a, b) > 3.0
    assert stats.hedges_g(b, a) < -3.0
    assert stats.hedges_g(a, a) == 0.0


def test_samples_needed_grows_as_the_effect_shrinks():
    big = stats.samples_needed(0.30, 0.15)
    small = stats.samples_needed(0.30, 0.02)
    assert small > big > 0


def test_samples_needed_on_no_effect_is_zero():
    assert stats.samples_needed(0.3, 0.0) == 0


def test_spearman_is_one_on_a_monotone_curve_pearson_would_understate():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [x ** 3 for x in xs]
    assert stats.spearman(xs, ys) == pytest.approx(1.0)
    assert stats.spearman(xs, [-y for y in ys]) == pytest.approx(-1.0)


def test_spearman_handles_ties_without_ordering_by_accident():
    assert stats.spearman([1.0, 1.0, 1.0], [3.0, 2.0, 1.0]) == 0.0


def test_kendall_tau_agrees_with_a_known_ordering():
    assert stats.kendall_tau([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert stats.kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_rank_correlations_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        stats.spearman([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        stats.kendall_tau([1.0], [1.0, 2.0])


def test_normal_cdf_matches_known_points():
    assert stats.normal_cdf(0.0) == pytest.approx(0.5)
    assert stats.normal_cdf(1.96) == pytest.approx(0.975, abs=1e-3)


class TestWilsonOnOutOfRangeCounts:
    """`k` arrives from summed outcome-log counts, and sums drift out of range.

    Before the clamp, `k > n` made `p * (1 - p)` negative and `math.sqrt` raise
    a domain error -- a crash out of a display path, from the one function
    whose job is to stop a small sample over-claiming.
    """

    def test_k_above_n_does_not_raise(self):
        lo, hi = wilson_interval(5, 3)
        assert 0.0 <= lo <= hi <= 1.0
        assert hi == 1.0

    def test_negative_k_does_not_raise(self):
        lo, hi = wilson_interval(-1, 3)
        assert lo == 0.0 and 0.0 <= hi <= 1.0

    def test_in_range_counts_are_untouched(self):
        assert wilson_interval(0, 12) == pytest.approx((0.0, 0.2424940), abs=1e-6)


class TestAResampleCountOfZeroIsRefused:
    """A zero-width interval at zero is not "no resamples".

    It reads as perfect precision about a number nothing estimated, and a
    p-value of exactly 1.0 reads as "no evidence of a difference" -- both
    produced without performing a single resample. This module exists because
    "a point estimate is what got the original 1.78x inflation past review";
    a fabricated interval is the same failure with more decoration.
    """

    def test_bootstrap_refuses(self):
        with pytest.raises(ValueError, match="at least 1"):
            stats.bootstrap_ci([1.0, 2.0, 3.0], resamples=0)

    def test_permutation_refuses(self):
        with pytest.raises(ValueError, match="at least 1"):
            stats.permutation_test([1.0, 2.0], [3.0, 4.0], resamples=0)

    def test_paired_permutation_refuses(self):
        with pytest.raises(ValueError, match="at least 1"):
            stats.paired_permutation_test([1.0, 2.0], [3.0, 4.0], resamples=-1)

    def test_one_resample_is_allowed(self):
        lo, hi = stats.bootstrap_ci([1.0, 2.0, 3.0], resamples=1)
        assert lo <= hi

    def test_an_empty_sample_is_still_a_display_path(self):
        assert stats.bootstrap_ci([]) == (0.0, 0.0)
