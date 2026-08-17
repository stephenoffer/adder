"""Style control, pinned on recovering a coefficient it was given.

A regression that cannot recover a known coefficient from data generated with
it is not measuring anything, and the failure is silent -- it returns zeros and
a report that says "length did not matter". An earlier version of this fit did
exactly that, which is why the recovery test is the first one here.
"""

from __future__ import annotations

import json
import math
import random

import pytest

from adder.pricing import style as st
from adder.pricing.bt import Battle


def _generated(n=1200, *, beta_length=0.9, skill_gap=0.0, seed=5,
               lengths=(400, 900, 1800, 3200), constant_style=True):
    """Battles from a known model: skill gap plus a length preference."""
    rng = random.Random(seed)
    battles, styles = [], []
    for _ in range(n):
        la, lb = rng.choice(lengths), rng.choice(lengths)
        if constant_style:
            sa = st.Style(tokens=la, headers=2, lists=2, bold=1)
            sb = st.Style(tokens=lb, headers=2, lists=2, bold=1)
        else:
            sa = st.Style(tokens=la, headers=la // 500, lists=la // 400, bold=la // 600)
            sb = st.Style(tokens=lb, headers=lb // 500, lists=lb // 400, bold=lb // 600)
        eta = skill_gap + beta_length * (math.log1p(la) - math.log1p(lb))
        p = 1.0 / (1.0 + math.exp(-eta))
        battles.append(Battle("a_model", "b_model", "a" if rng.random() < p else "b"))
        styles.append((sa, sb))
    return battles, styles


class TestMeasure:
    def test_it_counts_the_things_a_reader_rewards(self):
        text = "# Title\n\n- one\n- two\n\n**bold** and **more**\n## Sub\n"
        s = st.measure(text)
        assert s.headers == 2
        assert s.lists == 2
        assert s.bold == 2
        assert s.tokens > 0

    def test_an_empty_response_has_no_style(self):
        assert st.measure("") == st.Style()

    def test_length_enters_on_a_log_scale(self):
        """Doubling a long answer should matter less than doubling a short one."""
        small = st.Style(tokens=100).vector()[0]
        mid = st.Style(tokens=200).vector()[0]
        big = st.Style(tokens=3200).vector()[0]
        huge = st.Style(tokens=6400).vector()[0]
        assert (mid - small) == pytest.approx(huge - big, abs=0.02)

    def test_the_feature_vector_matches_the_declared_order(self):
        assert len(st.Style().vector()) == len(st.FEATURES)


class TestRecovery:
    def test_it_recovers_a_known_length_coefficient(self):
        battles, styles = _generated(n=1500, beta_length=0.9)
        res = st.fit_controlled(battles, styles, resamples=25)
        assert res.converged
        assert res.beta["length"] == pytest.approx(0.9, abs=0.15)
        assert res.length_matters
        assert res.beta_ci["length"][0] > 0

    def test_a_judge_indifferent_to_length_yields_no_coefficient(self):
        """The interval, not a threshold: a bare cutoff fires on noise."""
        battles, styles = _generated(n=1500, beta_length=0.0)
        res = st.fit_controlled(battles, styles, resamples=25)
        assert abs(res.beta["length"]) < 0.15
        assert not res.length_matters
        lo, hi = res.beta_ci["length"]
        assert lo < 0.0 < hi

    def test_without_an_interval_no_effect_is_claimed(self):
        """`measured` and `length_matters` are different questions."""
        battles, styles = _generated(n=600, beta_length=0.9)
        res = st.fit_controlled(battles, styles)
        assert not res.measured
        assert not res.length_matters
        assert res.beta["length"] > 0.5      # the point estimate is still there

    def test_control_shrinks_a_verbose_model_lead_without_erasing_real_skill(self):
        rng = random.Random(11)
        battles, styles = [], []
        for _ in range(1500):
            la, lb = rng.choice([1800, 3200]), rng.choice([400, 900])
            sa = st.Style(tokens=la, headers=2, lists=2, bold=1)
            sb = st.Style(tokens=lb, headers=2, lists=2, bold=1)
            eta = 0.5 + 0.9 * (math.log1p(la) - math.log1p(lb))
            p = 1.0 / (1.0 + math.exp(-eta))
            battles.append(Battle("verbose", "terse", "a" if rng.random() < p else "b"))
            styles.append((sa, sb))
        res = st.fit_controlled(battles, styles)
        raw_gap = res.uncontrolled["verbose"] - res.uncontrolled["terse"]
        ctl_gap = res.strength["verbose"] - res.strength["terse"]
        assert raw_gap > ctl_gap > 0          # shrunk, not erased
        assert res.premium("verbose") > 50

    def test_a_terse_model_has_a_negative_premium(self):
        """The one a cost-driven router should care about most."""
        rng = random.Random(13)
        battles, styles = [], []
        for _ in range(1200):
            la, lb = rng.choice([300, 500]), rng.choice([2000, 3000])
            sa = st.Style(tokens=la, headers=2, lists=2, bold=1)
            sb = st.Style(tokens=lb, headers=2, lists=2, bold=1)
            eta = 0.9 * (math.log1p(la) - math.log1p(lb))
            p = 1.0 / (1.0 + math.exp(-eta))
            battles.append(Battle("terse", "verbose", "a" if rng.random() < p else "b"))
            styles.append((sa, sb))
        res = st.fit_controlled(battles, styles)
        assert res.premium("terse") < 0


class TestIdentifiability:
    def test_constant_style_within_a_matchup_is_not_identified(self):
        """Style and skill are collinear; "no effect" would be a lie."""
        battles = [Battle("a_model", "b_model", "a")] * 200
        styles = [(st.Style(tokens=2000), st.Style(tokens=500))] * 200
        res = st.fit_controlled(battles, styles, resamples=10)
        assert not res.identified
        assert not res.length_matters

    def test_varying_style_within_a_matchup_is_identified(self):
        battles, styles = _generated(n=300)
        assert st.fit_controlled(battles, styles).identified

    def test_a_mismatched_style_list_is_rejected(self):
        with pytest.raises(ValueError):
            st.fit_controlled([Battle("a", "b", "a")], [])

    def test_an_empty_log_fits_nothing(self):
        res = st.fit_controlled([], [])
        assert res.battles == 0
        assert res.strength == {}

    def test_ties_are_kept_rather_than_dropped(self):
        """Ties are the most common outcome between similar models."""
        battles = [Battle("a", "b", "tie")] * 200
        styles = [(st.Style(tokens=1000), st.Style(tokens=1000))] * 200
        res = st.fit_controlled(battles, styles)
        assert res.battles == 200
        assert res.strength["a"] == pytest.approx(res.strength["b"], abs=1.0)


class TestPremiumCost:
    def test_the_carry_term_dominates_on_a_long_session(self):
        once = st.premium_cost(50.0, extra_tokens=400, out_rate=75.0,
                               cache_read_rate=1.5, remaining_turns=0)
        carried = st.premium_cost(50.0, extra_tokens=400, out_rate=75.0,
                                  cache_read_rate=1.5, remaining_turns=300)
        assert carried > once * 2

    def test_no_extra_tokens_costs_nothing(self):
        assert st.premium_cost(50.0, extra_tokens=0, out_rate=75.0,
                               cache_read_rate=1.5, remaining_turns=100) == 0.0

    def test_it_scales_with_the_extra_tokens(self):
        small = st.premium_cost(0.0, extra_tokens=100, out_rate=75.0,
                                cache_read_rate=1.5, remaining_turns=50)
        large = st.premium_cost(0.0, extra_tokens=400, out_rate=75.0,
                                cache_read_rate=1.5, remaining_turns=50)
        assert large == pytest.approx(4 * small)


class TestSummary:
    def test_mean_style_averages_a_set_of_responses(self):
        m = st.mean_style([st.Style(tokens=100, headers=2),
                           st.Style(tokens=300, headers=4)])
        assert m.tokens == 200
        assert m.headers == 3

    def test_mean_style_of_nothing(self):
        assert st.mean_style([]) == st.Style()

    def test_json_is_finite_and_complete(self):
        battles, styles = _generated(n=400)
        payload = st.fit_controlled(battles, styles).to_json()
        text = json.dumps(payload)
        assert "NaN" not in text and "Infinity" not in text
        assert set(payload) >= {"beta", "models", "identified"}
        assert "style_premium" in next(iter(payload["models"].values()))
