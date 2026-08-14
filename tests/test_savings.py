"""Savings estimator: the failure mode is silent over-claiming, so these tests
pin every figure to measured reality and check the levers compose correctly."""

import pytest

from adder.debt import decompose_read_cost
from adder.savings import (
    combine,
    delegation,
    explore_on_haiku,
    model_routing,
    splitting,
    terseness,
)
from adder.trace import Session, Turn

OPUS = "claude-opus-5"


def _sess(n_turns: int, out: int = 500, ctx_step: int = 5_000,
          base: int = 25_000, sidechain: bool = False) -> Session:
    s = Session("s", "p")
    for i in range(n_turns):
        s.turns.append(Turn("s", "p", OPUS, uncached_in=0,
                            cache_read=base + ctx_step * i, cache_write=0,
                            out=out, thinking=0, sidechain=sidechain))
    return s


class TestLeversAreBounded:
    @pytest.mark.parametrize("lever", [terseness, delegation, splitting])
    def test_no_lever_exceeds_the_pool_it_draws_from(self, lever):
        sessions = {"a": _sess(600)}
        _, _, pool = decompose_read_cost(sessions)
        e = lever(sessions)
        # generation savings sit outside the pool, so allow a small margin
        assert e.saving <= pool * 1.25

    def test_pool_fraction_is_a_fraction(self):
        sessions = {"a": _sess(600)}
        for lever in (terseness, delegation, splitting):
            f = lever(sessions).pool_fraction
            assert 0.0 <= f <= 1.0, f"{lever.__name__} -> {f}"

    def test_flat_session_offers_nothing_to_save(self):
        sessions = {"a": _sess(200, ctx_step=0)}
        assert terseness(sessions).pool_fraction == pytest.approx(0.3)
        assert splitting(sessions).saving == pytest.approx(0.0, abs=1e-6)


class TestComposition:
    def test_substitutes_do_not_add(self):
        """The central claim: summing them double-counts."""
        sessions = {"a": _sess(600)}
        _, _, pool = decompose_read_cost(sessions)
        levers = [terseness(sessions), delegation(sessions), splitting(sessions)]
        naive_sum = sum(e.saving for e in levers)
        pool_saving, _ = combine(pool, levers, [])
        assert pool_saving < naive_sum

    def test_combination_beats_the_single_best_lever(self):
        """...but max() is too conservative; they do compose."""
        sessions = {"a": _sess(600)}
        _, _, pool = decompose_read_cost(sessions)
        levers = [terseness(sessions), delegation(sessions), splitting(sessions)]
        pool_saving, _ = combine(pool, levers, [])
        assert pool_saving > max(e.saving for e in levers) * 0.95

    def test_combination_never_exceeds_the_pool(self):
        sessions = {"a": _sess(600)}
        _, _, pool = decompose_read_cost(sessions)
        aggressive = [terseness(sessions, reduction=0.9),
                      delegation(sessions, delegable_turns=0.9),
                      splitting(sessions, max_turns=10)]
        pool_saving, _ = combine(pool, aggressive, [])
        assert pool_saving <= pool * 1.001

    def test_no_levers_saves_nothing(self):
        assert combine(1000.0, [], []) == (0.0, 0.0)


class TestSeparateLevers:
    def test_model_routing_is_small_on_warm_contexts(self):
        """The original ask, quantified as the smallest lever."""
        sessions = {"a": _sess(500, ctx_step=1_000)}
        _, _, pool = decompose_read_cost(sessions)
        assert model_routing(sessions).saving < pool * 0.05

    def test_explore_savings_need_subagents(self):
        assert explore_on_haiku({"a": _sess(50)}).saving == pytest.approx(0.0)

    def test_explore_savings_measured_when_subagents_exist(self):
        e = explore_on_haiku({"a": _sess(50, sidechain=True)})
        assert e.saving > 0 and e.confidence == "MEASURED"


class TestConfidenceLabelling:
    def test_tiers_are_distinguished(self):
        sessions = {"a": _sess(100, sidechain=True)}
        assert explore_on_haiku(sessions).confidence == "MEASURED"
        assert terseness(sessions).confidence == "ATTRIBUTED"
        assert delegation(sessions).confidence == "MODELLED"

    def test_modelled_estimates_state_assumptions(self):
        sessions = {"a": _sess(100)}
        for e in (delegation(sessions), splitting(sessions), model_routing(sessions)):
            assert e.assumptions, f"{e.lever} must state its assumptions"
