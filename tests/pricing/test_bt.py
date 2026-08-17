"""The rating fit, pinned against cases where a scalar comparison lies.

The properties under test are the ones routing depends on: a model that wins
more gets a higher rating, a model with three battles gets a wide interval, two
models the data cannot separate share a rank, and the tier partition is the
optimal one rather than whatever a greedy pass produced.
"""

from __future__ import annotations

import math
import random

import pytest

from adder.pricing import bt
from adder.pricing.bt import Battle, fit


def _complete_graph_with_a_winless_model():
    """Every pair plays both ways, and C loses all four of its matches."""
    return [Battle("A", "B", "a"), Battle("B", "A", "b"),
            Battle("A", "C", "a"), Battle("C", "A", "b"),
            Battle("B", "C", "a"), Battle("C", "B", "b")]


def _round_robin(strengths: dict[str, float], n_each: int = 200, seed: int = 7):
    """Synthetic battles drawn from a known Bradley-Terry model.

    Recovering the strengths that generated the data is the only test that can
    tell a correct fit from a plausible one.
    """
    rng = random.Random(seed)
    models = sorted(strengths)
    out = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            p = bt.win_probability(strengths[a], strengths[b])
            for _ in range(n_each):
                out.append(Battle(a, b, "a" if rng.random() < p else "b"))
    return out


def test_fit_recovers_the_generating_strengths():
    truth = {"strong": 1200.0, "middle": 1100.0, "weak": 1000.0}
    fitted = bt.fit(_round_robin(truth))
    # Ratings are identified only up to an additive constant, so compare gaps.
    gap_fit = fitted["strong"] - fitted["weak"]
    gap_true = truth["strong"] - truth["weak"]
    assert abs(gap_fit - gap_true) < 25.0
    assert fitted["strong"] > fitted["middle"] > fitted["weak"]


def test_fit_is_order_independent():
    """The property Elo does not have, and the reason this is a batch MLE."""
    battles = _round_robin({"a": 1200.0, "b": 1000.0}, n_each=100)
    forward = bt.fit(battles)
    shuffled = list(battles)
    random.Random(99).shuffle(shuffled)
    backward = bt.fit(shuffled)
    for m in forward:
        assert forward[m] == pytest.approx(backward[m], abs=1e-6)


def test_ties_count_as_half_and_leave_equals_equal():
    battles = [Battle("a", "b", "tie")] * 40
    fitted = bt.fit(battles)
    assert fitted["a"] == pytest.approx(fitted["b"], abs=1e-6)


def test_an_undefeated_model_stays_finite():
    """Without the prior this diverges, and the rating becomes a function of max_iter."""
    battles = [Battle("a", "b", "a")] * 50
    fitted = bt.fit(battles)
    assert math.isfinite(fitted["a"])
    assert fitted["a"] > fitted["b"]


def test_prior_zero_is_available_for_connected_graphs():
    battles = _round_robin({"a": 1100.0, "b": 1000.0}, n_each=50)
    fitted = bt.fit(battles, prior=0.0)
    assert fitted["a"] > fitted["b"]


def test_empty_and_single_model_inputs():
    assert bt.fit([]) == {}
    assert bt.fit([Battle("solo", "solo2", "a")]).keys() == {"solo", "solo2"}


def test_a_model_cannot_play_itself():
    with pytest.raises(ValueError):
        bt.fit([Battle("a", "a", "a")])


def test_an_invalid_winner_label_is_rejected():
    with pytest.raises(ValueError):
        bt.fit([Battle("a", "b", "left")])


def test_win_probability_is_symmetric_and_monotone():
    assert bt.win_probability(1000.0, 1000.0) == pytest.approx(0.5)
    assert bt.win_probability(1400.0, 1000.0) > 0.9
    assert bt.win_probability(1000.0, 1400.0) < 0.1


def test_intervals_are_wide_on_little_data_and_narrow_on_a_lot():
    thin = bt.fit_with_ci([Battle("a", "b", "a")] * 6, resamples=60)
    thick = bt.fit_with_ci(_round_robin({"a": 1200.0, "b": 1000.0}, n_each=400),
                           resamples=60)
    assert thin["a"].half_width > thick["a"].half_width


def test_a_swept_log_does_not_report_a_zero_width_interval():
    """Six wins out of six is not certainty, and row-resampling says it is."""
    swept = [Battle("a", "b", "a")] * 6
    assert bt.fit_with_ci(swept, resamples=60)["a"].half_width > 20.0
    # The published-board method is still available, and still degenerates --
    # documented here so nobody "fixes" the default back.
    rows = bt.fit_with_ci(swept, resamples=60, method="battles")
    assert rows["a"].half_width == 0.0


def test_the_two_resampling_methods_agree_on_a_large_log():
    battles = _round_robin({"a": 1200.0, "b": 1000.0}, n_each=400)
    outcomes = bt.fit_with_ci(battles, resamples=80)["a"]
    rows = bt.fit_with_ci(battles, resamples=80, method="battles")["a"]
    assert abs(outcomes.half_width - rows.half_width) < 8.0


def test_an_unknown_resampling_method_is_rejected():
    with pytest.raises(ValueError):
        bt.fit_with_ci([Battle("a", "b", "a")], method="jackknife")


def test_the_observed_tie_rate_survives_resampling():
    """An all-tie log must keep producing equal ratings, not drift apart."""
    fitted = bt.fit_with_ci([Battle("a", "b", "tie")] * 100, resamples=40)
    assert fitted["a"].rating == pytest.approx(fitted["b"].rating, abs=1e-6)
    assert fitted["a"].overlaps(fitted["b"])


def test_ci_is_reproducible_across_runs():
    battles = _round_robin({"a": 1200.0, "b": 1000.0}, n_each=40)
    first = bt.fit_with_ci(battles, resamples=40)
    second = bt.fit_with_ci(battles, resamples=40)
    assert first["a"].lo == second["a"].lo
    assert first["a"].hi == second["a"].hi


def test_the_point_estimate_is_always_inside_its_own_interval():
    battles = _round_robin({"a": 1200.0, "b": 1150.0, "c": 1000.0}, n_each=30)
    for r in bt.fit_with_ci(battles, resamples=40).values():
        assert r.lo <= r.rating <= r.hi


def test_overlapping_intervals_share_a_rank():
    # Two models separated by noise only: a scalar comparison would order them.
    battles = _round_robin({"a": 1005.0, "b": 1000.0}, n_each=30)
    fitted = bt.fit_with_ci(battles, resamples=60)
    assert fitted["a"].overlaps(fitted["b"])
    assert not fitted["a"].beats(fitted["b"])
    assert set(bt.ranks(fitted.values()).values()) == {1}


def test_a_real_gap_produces_a_strict_ranking():
    battles = _round_robin({"a": 1400.0, "b": 1000.0}, n_each=200)
    fitted = bt.fit_with_ci(battles, resamples=60)
    assert fitted["a"].beats(fitted["b"])
    assert bt.ranks(fitted.values()) == {"a": 1, "b": 2}


def test_indistinguishable_lists_the_free_substitutions():
    battles = _round_robin({"a": 1002.0, "b": 1000.0, "far": 1500.0}, n_each=60)
    fitted = bt.fit_with_ci(battles, resamples=60)
    assert bt.indistinguishable(fitted.values(), "a") == ["b"]
    assert bt.indistinguishable(fitted.values(), "missing") == []


def test_battle_counts_are_recorded_per_model():
    fitted = bt.fit_with_ci([Battle("a", "b", "a"), Battle("a", "c", "b")],
                            resamples=10)
    assert fitted["a"].battles == 2
    assert fitted["c"].battles == 1


def test_tiers_are_the_optimal_partition_not_an_equal_split():
    # Three tight clusters with wide gaps: any correct 1-D k-means finds them,
    # an equal-count split does not.
    scores = {"a": 1500.0, "b": 1498.0, "c": 1200.0,
              "d": 1198.0, "e": 1196.0, "f": 900.0}
    assigned = bt.tiers(scores, k=3)
    members = bt.tier_members(assigned)
    assert members[0] == ["a", "b"]
    assert members[1] == ["c", "d", "e"]
    assert members[2] == ["f"]


def test_tier_zero_is_the_strongest():
    scores = {"top": 1400.0, "bottom": 900.0}
    assigned = bt.tiers(scores, k=2)
    assert assigned["top"] == 0
    assert assigned["bottom"] == 1


def test_more_tiers_than_models_gives_each_its_own():
    scores = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert bt.tiers(scores, k=10) == {"a": 0, "b": 1, "c": 2}


def test_tiers_on_empty_input_and_bad_k():
    assert bt.tiers({}, k=3) == {}
    with pytest.raises(ValueError):
        bt.tiers({"a": 1.0}, k=0)


def test_every_model_lands_in_exactly_one_tier():
    scores = {f"m{i}": float(i * i % 37) for i in range(40)}
    assigned = bt.tiers(scores, k=6)
    assert set(assigned) == set(scores)
    assert set(assigned.values()) <= set(range(6))


def test_agreement_detects_a_fit_that_reproduces_a_published_board():
    truth = {"a": 1300.0, "b": 1200.0, "c": 1100.0, "d": 1000.0}
    fitted = bt.fit(_round_robin(truth, n_each=200))
    assert bt.agreement(fitted, truth) == pytest.approx(1.0)


def test_agreement_needs_enough_shared_models_to_mean_anything():
    assert bt.agreement({"a": 1.0}, {"a": 1.0}) == 0.0


class TestDisconnectedGraphs:
    """Islands of models that never played each other.

    The docstring for `fit` claims the prior is what keeps this finite. That
    claim was never tested, and a disconnected comparison graph is not exotic --
    it is what a fresh log looks like before any cross-tier battle happens.
    """

    def test_two_islands_still_produce_finite_ratings(self):
        battles = ([Battle("a", "b", "a")] * 20 + [Battle("c", "d", "a")] * 20)
        fitted = bt.fit(battles)
        assert all(math.isfinite(v) for v in fitted.values())
        assert fitted["a"] > fitted["b"]
        assert fitted["c"] > fitted["d"]

    def test_ratings_across_islands_are_not_claimed_to_be_comparable(self):
        """Nothing links the islands, so the intervals must overlap."""
        battles = ([Battle("a", "b", "a")] * 20 + [Battle("c", "d", "a")] * 20)
        fitted = bt.fit_with_ci(battles, resamples=60)
        assert fitted["a"].overlaps(fitted["c"])

    def test_a_single_bridging_battle_links_them(self):
        battles = ([Battle("a", "b", "a")] * 20 + [Battle("c", "d", "a")] * 20
                   + [Battle("b", "c", "a")] * 20)
        fitted = bt.fit(battles)
        assert fitted["a"] > fitted["d"]


class TestTierStability:
    def test_the_partition_does_not_depend_on_input_order(self):
        scores = {"a": 1500.0, "b": 1498.0, "c": 1200.0, "d": 900.0}
        forward = bt.tiers(scores, k=3)
        backward = bt.tiers(dict(reversed(list(scores.items()))), k=3)
        assert forward == backward

    def test_identical_ratings_land_in_the_same_tier(self):
        assigned = bt.tiers({"a": 1200.0, "b": 1200.0, "c": 900.0}, k=2)
        assert assigned["a"] == assigned["b"]


# A model that never won has no finite Bradley-Terry strength.
#
# `fit` documents `prior=0` as available "if you have verified the comparison
# graph is strongly connected". That precondition is about who *played* whom, and
# it does not stop a model from playing everyone and beating none of them. When
# that happens the MM iteration walks its strength to zero and the run ends four
# frames down in `math.log(0)` with "expected a positive input, got 0.0" -- which
# names neither the model nor the fix.
#
# This is the exact mirror of the undefeated case the prior already exists for.
# It is refused rather than floored: an epsilon would put a finite rating on a
# model whose rating is genuinely unbounded below, and a made-up number that
# reads as a measurement is the failure this package exists to prevent.
class TestPriorZeroRefusesRatherThanCrashing:
    def test_it_raises_a_value_error(self):
        with pytest.raises(ValueError):
            fit(_complete_graph_with_a_winless_model(), prior=0.0)

    def test_the_message_names_the_model_that_cannot_be_fitted(self):
        with pytest.raises(ValueError, match="C"):
            fit(_complete_graph_with_a_winless_model(), prior=0.0)

    def test_the_message_names_the_fix(self):
        with pytest.raises(ValueError, match="prior"):
            fit(_complete_graph_with_a_winless_model(), prior=0.0)

    def test_a_total_sweep_is_refused_too(self):
        with pytest.raises(ValueError):
            fit([Battle("A", "B", "a")] * 5, prior=0.0)


class TestTheDefaultPriorStillFitsEverything:
    def test_a_winless_model_gets_a_finite_rating(self):
        got = fit(_complete_graph_with_a_winless_model())
        assert all(math.isfinite(v) for v in got.values())

    def test_and_is_ranked_last(self):
        got = fit(_complete_graph_with_a_winless_model())
        assert got["C"] < got["B"] < got["A"]

    def test_prior_zero_still_works_on_a_graph_where_everyone_wins(self):
        """The case the flag is documented for must keep working."""
        # `winner` names the field, so "b" here means the second model won.
        battles = ([Battle("a", "b", "a")] * 30 + [Battle("a", "b", "b")] * 20)
        got = fit(battles, prior=0.0)
        assert got["a"] > got["b"]
        assert all(math.isfinite(v) for v in got.values())
