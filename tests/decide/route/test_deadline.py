"""Deadline policies, pinned so the cheapest-looking one cannot win by cheating.

The trap this suite exists for: a policy that abandons work looks cheapest
unless unfinished work is charged for. Every cost here includes finishing what
was missed, at full price.
"""

from __future__ import annotations

import json

import pytest

from adder.decide.route import deadline as dl
from adder.decide.route.deadline import Workload


def _w(**kw):
    base = {"units": 100, "horizon": 12, "cost_cheap": 0.5, "cost_guaranteed": 1.0}
    base.update(kw)
    return Workload(**base)


class TestValidation:
    def test_a_negative_queue_is_rejected(self):
        with pytest.raises(ValueError):
            _w(units=-1)

    def test_a_zero_horizon_is_rejected(self):
        with pytest.raises(ValueError):
            _w(horizon=0)

    @pytest.mark.parametrize("field", ["batch_throughput", "stall_rate",
                                       "guaranteed_throughput"])
    def test_rates_are_range_checked(self, field):
        with pytest.raises(ValueError):
            _w(**{field: 1.5})

    def test_negative_prices_are_rejected(self):
        with pytest.raises(ValueError):
            _w(cost_cheap=-1.0)

    def test_the_discount_is_derived_not_assumed(self):
        assert _w(cost_cheap=0.5, cost_guaranteed=1.0).discount == pytest.approx(0.5)
        assert _w(cost_cheap=1.0, cost_guaranteed=1.0).discount == 0.0

    def test_a_free_guaranteed_path_has_no_discount_to_offer(self):
        assert _w(cost_guaranteed=0.0).discount == 0.0


class TestSimulate:
    def test_the_guaranteed_policy_always_finishes(self):
        run = dl.simulate(_w(), "guaranteed")
        assert run.met
        assert run.cheap_units == 0
        assert run.cost == pytest.approx(_w().all_guaranteed_cost)

    def test_a_never_stalling_cheap_path_reaches_the_floor(self):
        w = _w(stall_rate=0.0, batch_throughput=1.0)
        run = dl.simulate(w, "cheap")
        assert run.met
        assert run.cost == pytest.approx(w.floor_cost)

    def test_a_fully_stalled_cheap_path_finishes_nothing(self):
        run = dl.simulate(_w(stall_rate=1.0), "cheap")
        assert not run.met
        assert run.cheap_units == 0

    def test_unfinished_work_is_still_charged(self):
        """Otherwise the cheapest policy is the one that gives up."""
        w = _w(stall_rate=1.0)
        run = dl.simulate(w, "cheap")
        assert run.cost == pytest.approx(w.all_guaranteed_cost)

    def test_the_uniform_policy_finishes_even_with_a_flaky_cheap_path(self):
        run = dl.simulate(_w(stall_rate=0.9), "uniform")
        assert run.met

    def test_an_unknown_policy_is_rejected(self):
        with pytest.raises(ValueError):
            dl.simulate(_w(), "hope")

    def test_an_empty_queue_costs_nothing(self):
        assert dl.simulate(_w(units=0), "uniform").cost == 0.0

    def test_the_same_seed_gives_the_same_run(self):
        a = dl.simulate(_w(), "uniform", seed=7)
        b = dl.simulate(_w(), "uniform", seed=7)
        assert (a.cost, a.cheap_units, a.switches) == (b.cost, b.cheap_units, b.switches)

    def test_different_seeds_explore_different_stalls(self):
        costs = {dl.simulate(_w(stall_rate=0.5), "uniform", seed=s).cost
                 for s in range(8)}
        assert len(costs) > 1

    def test_a_minimum_run_reduces_switching(self):
        many = dl.simulate(_w(stall_rate=0.5), "uniform", seed=11)
        few = dl.simulate(_w(stall_rate=0.5, min_batch_run=4), "uniform", seed=11)
        assert few.switches <= many.switches


class TestPolicies:
    def test_the_uniform_policy_never_misses_a_deadline(self):
        out = dl.evaluate(_w(stall_rate=0.6), "uniform", trials=120)
        assert out.met_rate == 1.0

    def test_the_cheap_policy_misses_when_the_path_is_flaky(self):
        out = dl.evaluate(_w(stall_rate=0.7, batch_throughput=0.2), "cheap",
                          trials=120)
        assert out.met_rate < 1.0

    def test_the_uniform_policy_beats_paying_full_price(self):
        w = _w(stall_rate=0.2, horizon=24)
        uniform = dl.evaluate(w, "uniform", trials=120)
        guaranteed = dl.evaluate(w, "guaranteed", trials=120)
        assert uniform.cost_mean < guaranteed.cost_mean
        assert uniform.met_rate == 1.0

    def test_greedy_wins_when_the_guaranteed_path_is_instant(self):
        """The case the module refuses to have a favourite about.

        When the guaranteed path can absorb the whole remaining queue at once,
        the last-step sprint always rescues greedy, so it collects the entire
        discount and the proportional policy is simply paying more.
        """
        w = _w(stall_rate=0.3, horizon=24, guaranteed_throughput=1.0)
        greedy = dl.evaluate(w, "greedy", trials=200)
        uniform = dl.evaluate(w, "uniform", trials=200)
        assert greedy.met_rate == 1.0
        assert greedy.cost_mean < uniform.cost_mean

    def test_the_proportional_policy_wins_when_the_sprint_cannot_rescue(self):
        """Rate-limit the guaranteed path and greedy's late work has nowhere to go."""
        w = _w(stall_rate=0.6, horizon=24, guaranteed_throughput=0.2)
        greedy = dl.evaluate(w, "greedy", trials=200)
        uniform = dl.evaluate(w, "uniform", trials=200)
        assert uniform.met_rate >= greedy.met_rate

    def test_a_tight_window_leaves_no_room_for_the_discount(self):
        out = dl.evaluate(_w(horizon=1, stall_rate=0.5), "uniform", trials=60)
        assert out.cheap_share < 0.5

    def test_outcomes_are_reproducible(self):
        a = dl.evaluate(_w(), "uniform", trials=40)
        b = dl.evaluate(_w(), "uniform", trials=40)
        assert a.cost_mean == b.cost_mean
        assert a.met_rate == b.met_rate

    def test_p90_is_at_least_the_mean_on_a_skewed_run(self):
        out = dl.evaluate(_w(stall_rate=0.5), "cheap", trials=200)
        assert out.cost_p90 >= out.cost_mean - 1e-9

    def test_compare_covers_every_policy(self):
        assert {o.policy for o in dl.compare(_w(), trials=40)} == set(dl.POLICIES)


class TestBreakeven:
    def test_a_generous_window_has_a_usable_shortest_window(self):
        w = _w(horizon=48, stall_rate=0.2)
        be = dl.breakeven_horizon(w, trials=60)
        assert 1 <= be <= w.horizon

    def test_a_hopeless_window_reports_no_useful_horizon(self):
        w = _w(horizon=2, stall_rate=1.0)
        assert dl.breakeven_horizon(w, trials=40) == w.horizon + 1


class TestReport:
    def test_it_names_the_cheapest_policy_that_meets_the_deadline(self):
        text = dl.report(_w(stall_rate=0.2, horizon=24), trials=80)
        assert "Cheapest policy that meets every deadline" in text

    def test_it_explains_which_of_the_two_middles_to_prefer(self):
        text = dl.report(_w(stall_rate=0.3, horizon=24), trials=120)
        assert "rate-limited" in text

    def test_it_says_when_the_window_is_too_tight(self):
        text = dl.report(_w(horizon=1, stall_rate=1.0), trials=80)
        assert "stop optimising" in text or "too tight" in text

    def test_it_prices_the_naive_all_cheap_policy_honestly(self):
        text = dl.report(_w(stall_rate=0.8, batch_throughput=0.2), trials=80)
        assert "not the one you collect" in text or "finishes only" in text

    def test_it_always_discloses_the_modelled_inputs(self):
        assert "MODELLED" in dl.report(_w(), trials=40)

    def test_json_is_finite_and_complete(self):
        payload = dl.to_json(_w(), trials=40)
        text = json.dumps(payload)
        assert "NaN" not in text and "Infinity" not in text
        assert payload["modelled"] is True
        assert len(payload["policies"]) == len(dl.POLICIES)

    def test_the_recommended_policy_meets_the_deadline(self):
        payload = dl.to_json(_w(stall_rate=0.5), trials=80)
        best = next(p for p in payload["policies"] if p["policy"] == payload["best"])
        assert best["deadline_met_rate"] >= 0.99


class TestCli:
    def test_it_runs_and_prints(self, capsys, isolated_home):
        assert dl.main(["--trials", "40"]) == 0
        assert capsys.readouterr().out.strip()

    def test_json_parses(self, capsys, isolated_home):
        assert dl.main(["--trials", "40", "--json"]) == 0
        json.loads(capsys.readouterr().out)

    def test_a_bad_rate_is_a_usage_error_not_a_traceback(self, capsys, isolated_home):
        assert dl.main(["--stall-rate", "2.0"]) == 2

    def test_a_bad_horizon_is_a_usage_error(self, capsys, isolated_home):
        assert dl.main(["--horizon", "0"]) == 2
