"""Savings estimator: the failure mode is silent over-claiming, so these tests
pin every figure to measured reality and check the levers compose correctly."""
from __future__ import annotations

import pytest

from adder.core.trace import Session, Turn
from adder.evaluate.claims.savings import (
    combine,
    delegation,
    explore_on_haiku,
    model_routing,
    splitting,
    terseness,
)
from adder.measure.spend.debt import decompose_read_cost

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


class TestContextLevers:
    """The two levers added with the context family, and where each belongs."""

    def test_compaction_is_a_substitute_not_an_addition(self, make_sessions):
        from adder.evaluate.claims.savings import levers

        pool, _separate = levers(make_sessions(3, 60))
        names = [e.lever for e in pool]
        assert any("Compact sessions" in n for n in names), names

    def test_memory_trim_is_separate_from_the_pool(self, make_sessions):
        from adder.evaluate.claims.savings import levers

        pool, separate = levers(make_sessions(3, 60))
        assert any("resident memory" in e.lever for e in separate)
        assert not any("resident memory" in e.lever for e in pool)

    def test_compaction_saving_is_never_negative(self, make_sessions):
        from adder.evaluate.claims.savings import compaction_discipline

        assert compaction_discipline(make_sessions(3, 60)).saving >= 0

    def test_compaction_pool_fraction_is_a_share(self, make_sessions):
        from adder.evaluate.claims.savings import compaction_discipline

        e = compaction_discipline(make_sessions(3, 60))
        assert 0.0 <= e.pool_fraction <= 1.0

    def test_memory_trim_counts_only_what_is_recoverable(self, tmp_path,
                                                         make_sessions):
        from adder.evaluate.claims.savings import memory_trim

        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "CLAUDE.md").write_text("a real instruction file\n" * 200)
        # Nothing duplicated and nothing over-long: a whole instruction file is
        # not a saving just because it is large.
        assert memory_trim(make_sessions(3, 60), repo).saving == 0.0


class TestSplittingModelsTheMainChain:
    """Where you restart a session says nothing about a subagent's window.

    `splitting` counted every turn, so a session that was 71% subagent turns
    entered the model at 3.5x its real conversational length, and it took the
    cache-read rate for the whole session from `turns[0]` -- a subagent turn in
    two of the 37 long sessions here, which priced their entire span at the
    subagent's model.
    """

    @staticmethod
    def _turn(ctx, *, side=False, model="claude-opus-5"):
        return Turn("s", "p", model, uncached_in=0, cache_read=ctx, cache_write=0,
                    out=10, thinking=0, sidechain=side, ts="2026-08-10T12:00:00Z")

    def _session(self, n_main, n_side=0, *, side_model="claude-haiku-4-5"):
        s = Session("s", "p")
        # a subagent turn first, which is what set the rate for everything
        s.turns = [self._turn(3_000, side=True, model=side_model)
                   for _ in range(n_side)]
        s.turns += [self._turn(50_000 + 200 * i) for i in range(n_main)]
        return {"s": s}

    def test_a_session_short_on_the_main_chain_is_not_split(self):
        """310 turns, but only 100 of them are the conversation."""
        got = splitting(self._session(100, n_side=210), max_turns=300)
        assert got.saving == 0.0

    def test_a_genuinely_long_session_still_counts(self):
        got = splitting(self._session(400), max_turns=300)
        assert got.saving > 0

    def test_subagent_turns_do_not_change_the_estimate(self):
        """They are the same conversation either way."""
        without = splitting(self._session(400), max_turns=300).saving
        with_side = splitting(self._session(400, n_side=50), max_turns=300).saving
        assert with_side == pytest.approx(without)

    def test_the_rate_does_not_come_from_a_subagent_turn(self):
        """Haiku reads at a fifth of Opus; taking its rate would shrink this."""
        cheap_first = splitting(self._session(400, n_side=1,
                                              side_model="claude-haiku-4-5"),
                                max_turns=300).saving
        plain = splitting(self._session(400), max_turns=300).saving
        assert cheap_first == pytest.approx(plain)
