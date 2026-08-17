"""Delegation as measured: run detection, subagent model choice, missed reads."""

from __future__ import annotations

import json

import pytest

from adder.core.trace import Session, Turn
from adder.measure.session.horizon import Horizon
from adder.measure.spend.agents import DELEGABLE_TOKENS, Run, analyse, cheaper_model, missed, runs

OPUS, HAIKU = "claude-opus-5", "claude-haiku-4-5"


class TestRunDetection:
    def test_contiguous_sidechain_turns_are_one_run(self, make_turn):
        s = Session("s", "p")
        s.turns = [make_turn(), make_turn(sidechain=True), make_turn(sidechain=True),
                   make_turn()]
        found = runs({"s": s})
        assert len(found) == 1
        assert found[0].n_turns == 2

    def test_a_main_chain_turn_between_them_splits_the_run(self, make_turn):
        s = Session("s", "p")
        s.turns = [make_turn(sidechain=True), make_turn(),
                   make_turn(sidechain=True)]
        assert len(runs({"s": s})) == 2

    def test_no_sidechain_turns_means_no_runs(self, make_session):
        assert runs({"s": make_session(20)}) == []

    def test_summary_is_the_last_turns_output(self, make_turn):
        s = Session("s", "p")
        s.turns = [make_turn(sidechain=True, out=100),
                   make_turn(sidechain=True, out=7_000)]
        assert runs({"s": s})[0].summary_tokens == 7_000

    def test_an_empty_run_has_no_model_and_no_cost(self):
        r = Run("s", "p")
        assert r.model == ""
        assert r.cost() == 0.0
        assert r.summary_tokens == 0
        assert r.when is None


class TestCheaperModel:
    def _run(self, make_turn, *, model=OPUS, read=50_000, out=2_000):
        r = Run("s", "p")
        r.turns = [make_turn(model=model, read=read, out=out, sidechain=True)]
        return r

    def test_a_small_opus_run_could_have_been_haiku(self, make_turn):
        model, saving = cheaper_model(self._run(make_turn, read=50_000))
        assert model == HAIKU
        assert saving > 0

    def test_a_run_too_large_for_haiku_is_not_offered_it(self, make_turn):
        model, _ = cheaper_model(self._run(make_turn, read=900_000))
        assert model != HAIKU

    def test_an_already_cheap_run_has_nothing_to_save(self, make_turn):
        model, saving = cheaper_model(self._run(make_turn, model=HAIKU))
        assert model is None
        assert saving == 0.0

    def test_a_run_with_no_context_is_skipped(self, make_turn):
        r = Run("s", "p")
        r.turns = [make_turn(read=0, uncached=0, write=0, sidechain=True)]
        assert cheaper_model(r) == (None, 0.0)

    def test_floor_pins_a_capability_minimum(self, make_turn):
        model, _ = cheaper_model(self._run(make_turn), floor="claude-sonnet-5")
        assert model != HAIKU


class TestMissed:
    def test_a_read_too_large_for_the_subagent_is_not_offered(self, make_turn):
        """Feasibility gates profitability: a 900K read does not fit in Haiku."""
        s = Session("s", "p")
        s.turns = [make_turn(read=10_000), make_turn(read=900_000)]
        assert missed({"s": s}, horizon=Horizon([500] * 20)) == []

    def test_a_large_admission_is_flagged(self, make_turn):
        s = Session("s", "p")
        s.turns = [make_turn(read=10_000),
                   make_turn(read=10_000 + DELEGABLE_TOKENS + 5_000)]
        found = missed({"s": s}, horizon=Horizon([500] * 20))
        assert len(found) == 1
        assert found[0].tokens > DELEGABLE_TOKENS

    def test_small_growth_is_not_flagged(self, make_session):
        assert missed({"s": make_session(30, growth=1_000)}) == []

    def test_sidechain_turns_are_not_counted_as_missed(self, make_turn):
        s = Session("s", "p")
        s.turns = [make_turn(read=10_000, sidechain=True),
                   make_turn(read=500_000, sidechain=True)]
        assert missed({"s": s}) == []

    def test_saving_uses_the_horizon_at_that_turn(self, make_turn):
        """A read late in a short session is worth less than one early on."""
        s = Session("s", "p")
        s.turns = [make_turn(read=10_000), make_turn(read=150_000)]
        long_h = missed({"s": s}, horizon=Horizon([2000] * 20))[0].saving
        short_h = missed({"s": s}, horizon=Horizon([3] * 20))[0].saving
        assert long_h > short_h

    def test_results_are_sorted_by_saving(self, make_turn):
        s = Session("s", "p")
        s.turns = [make_turn(read=10_000), make_turn(read=60_000),
                   make_turn(read=60_000 + 120_000)]
        found = missed({"s": s}, horizon=Horizon([500] * 20))
        assert found == sorted(found, key=lambda m: -m.saving)


class TestAnalyse:
    def test_share_of_spend(self, make_turn):
        s = Session("s", "p")
        s.turns = [make_turn(read=100_000), make_turn(read=100_000, sidechain=True)]
        rep = analyse({"s": s})
        assert rep.n_runs == 1
        assert rep.share == pytest.approx(0.5, rel=0.01)

    def test_no_sessions_is_not_a_division_error(self):
        rep = analyse({})
        assert rep.share == 0.0
        assert rep.n_runs == 0

    def test_by_model_groups_runs(self, make_turn):
        s = Session("s", "p")
        s.turns = [make_turn(sidechain=True, model=OPUS), make_turn(),
                   make_turn(sidechain=True, model=HAIKU)]
        assert set(analyse({"s": s}).by_model()) == {OPUS, HAIKU}

    def test_downgradable_reports_targets(self, make_turn):
        s = Session("s", "p")
        s.turns = [make_turn(sidechain=True, model=OPUS, read=40_000, out=1_000)]
        saving, moves = analyse({"s": s}).downgradable()
        assert saving > 0
        assert HAIKU in moves


class TestCli:
    def test_json(self, tmp_path, write_jsonl, capsys):
        from adder.measure.spend.agents import main

        write_jsonl([
            {"type": "assistant", "sessionId": "s", "isSidechain": True,
             "timestamp": "2026-08-01T10:00:00Z",
             "message": {"id": "m1", "model": "claude-opus-5",
                         "usage": {"input_tokens": 1,
                                   "cache_read_input_tokens": 40_000,
                                   "output_tokens": 1000}, "content": []}},
        ])
        assert main([str(tmp_path), "--json"]) == 0
        d = json.loads(capsys.readouterr().out)
        assert d["runs"] == 1
        assert d["downgradable_saving"] > 0

    def test_text_says_so_when_nothing_was_delegated(self, tmp_path, write_jsonl,
                                                     capsys):
        from adder.measure.spend.agents import main

        write_jsonl([{"type": "assistant", "sessionId": "s",
                      "timestamp": "2026-08-01T10:00:00Z",
                      "message": {"id": "m1", "model": "claude-opus-5",
                                  "usage": {"input_tokens": 1,
                                            "cache_read_input_tokens": 900,
                                            "output_tokens": 10}, "content": []}}])
        assert main([str(tmp_path)]) == 0
        assert "No subagent runs" in capsys.readouterr().out


class TestRunsAreSplitByAgent:
    """Two subagents in one session are two runs, however the records interleave.

    Subagent records carry the *parent's* session id, so every sidechain turn in
    a session sits in one contiguous block no matter how many agents wrote it --
    and a workflow that fans out never puts a main-chain turn between them to
    break the block. Grouping by adjacency merged 119 subagents into 4 runs on
    the author's corpus, one of them 509 turns long with the context collapsing
    four times inside it, each collapse an agent starting fresh.

    `agentId` is on every sidechain record and is the only field that separates
    them.
    """

    @staticmethod
    def _turn(agent, out=100, ctx=10_000):
        return Turn("s", "p", "claude-opus-5", uncached_in=0, cache_read=ctx,
                    cache_write=0, out=out, thinking=0, sidechain=True,
                    agent_id=agent)

    @staticmethod
    def _main():
        return Turn("s", "p", "claude-opus-5", uncached_in=0, cache_read=500_000,
                    cache_write=0, out=100, thinking=0, sidechain=False)

    def _sess(self, turns):
        s = Session("s", "p")
        s.turns = turns
        return {"s": s}

    def test_two_adjacent_agents_are_two_runs(self):
        got = runs(self._sess([self._turn("a"), self._turn("a"),
                               self._turn("b"), self._turn("b")]))
        assert len(got) == 2

    def test_interleaved_agents_are_still_separated(self):
        """A fan-out writes both agents' turns in whatever order they finish."""
        got = runs(self._sess([self._turn("a"), self._turn("b"),
                               self._turn("a"), self._turn("b")]))
        assert sorted(r.n_turns for r in got) == [2, 2]

    def test_each_run_reports_its_own_summary(self):
        """`summary_tokens` is what the parent admits; one per agent, not one total."""
        got = runs(self._sess([self._turn("a", out=11), self._turn("a", out=22),
                               self._turn("b", out=33), self._turn("b", out=44)]))
        assert sorted(r.summary_tokens for r in got) == [22, 44]

    def test_peak_context_is_per_agent(self):
        """It feeds the `fits()` gate in `cheaper_model`."""
        got = runs(self._sess([self._turn("a", ctx=10_000),
                               self._turn("b", ctx=900_000)]))
        assert sorted(r.peak_context for r in got) == [10_000, 900_000]

    def test_main_chain_turns_are_not_runs(self):
        assert runs(self._sess([self._main(), self._main()])) == []

    def test_records_without_an_agent_id_fall_back_to_adjacency(self):
        """Transcripts written before the field existed still yield runs."""
        got = runs(self._sess([self._turn(""), self._turn(""),
                               self._main(), self._turn("")]))
        assert sorted(r.n_turns for r in got) == [1, 2]
