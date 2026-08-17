"""Interruption economics, pinned on the term that decides them.

The claim under test: what makes cheap-but-interruptible capacity pay is not the
size of the discount, it is whether the work can checkpoint. A report that
priced the discount without pricing the lost context would recommend it for
every session, including the ones where an interruption destroys hours.
"""

from __future__ import annotations

import json

import pytest

from adder.decide.route import harvest as hv


class TestLosses:
    def test_a_longer_session_loses_more(self, make_session):
        short = hv.losses({"s": make_session(5, base=20_000, growth=1_000)})[0]
        long = hv.losses({"s": make_session(60, base=20_000, growth=1_000)})[0]
        assert long.expected_loss_cold > short.expected_loss_cold

    def test_a_handoff_removes_most_of_the_loss(self, make_session):
        row = hv.losses({"s": make_session(30, base=20_000, growth=500)},
                        handoff_tokens=2_000)[0]
        assert row.expected_loss_handoff < row.expected_loss_cold
        assert 0.0 < row.protected < 1.0

    def test_a_handoff_larger_than_the_context_removes_all_of_it(self, make_session):
        row = hv.losses({"s": make_session(10, base=1_000, growth=0)},
                        handoff_tokens=10_000_000)[0]
        assert row.expected_loss_handoff == 0.0
        assert row.protected == 1.0

    def test_a_one_turn_session_has_nothing_to_lose(self, make_session):
        assert hv.losses({"s": make_session(1)}) == []

    def test_no_sessions(self):
        assert hv.losses({}) == []


class TestEconomics:
    @staticmethod
    def _rep(n_turns=40, **kw):
        from adder.core.trace import Session, Turn

        sessions = {}
        for i in range(6):
            s = Session(f"s{i}", "p")
            for k in range(n_turns):
                s.turns.append(Turn(f"s{i}", "p", "claude-opus-5", uncached_in=0,
                                    cache_read=20_000 + 2_000 * k, cache_write=0,
                                    out=400, thinking=0, sidechain=False, ts=None))
            sessions[f"s{i}"] = s
        return hv.analyse(sessions, **kw)

    def test_checkpointing_is_what_makes_it_pay(self):
        rep = self._rep(n_turns=60, handoff_tokens=2_000, discount=0.5,
                        interruptions=1.0)
        assert rep.gain(checkpointed=True) > rep.gain(checkpointed=False)

    def test_a_higher_interruption_rate_erodes_the_gain(self):
        calm = self._rep(interruptions=0.1)
        stormy = self._rep(interruptions=10.0)
        assert stormy.gain(checkpointed=True) < calm.gain(checkpointed=True)

    def test_a_bigger_discount_helps(self):
        small = self._rep(discount=0.1)
        big = self._rep(discount=0.9)
        assert big.gain(checkpointed=True) > small.gain(checkpointed=True)

    def test_the_breakeven_rate_is_where_the_gain_crosses_zero(self):
        rep = self._rep()
        be = rep.breakeven_rate(checkpointed=True)
        if be not in (0.0, float("inf")):
            at = hv.Report(rows=rep.rows, spend=rep.spend,
                           handoff_tokens=rep.handoff_tokens,
                           discount=rep.discount, interruptions=be)
            assert at.gain(checkpointed=True) == pytest.approx(0.0, abs=1e-9)

    def test_nothing_to_lose_means_no_breakeven(self):
        rep = self._rep(handoff_tokens=10_000_000)
        assert rep.breakeven_rate(checkpointed=True) == float("inf")

    def test_the_protected_share_is_a_fraction(self):
        rep = self._rep()
        assert 0.0 <= rep.protected <= 1.0


class TestReport:
    def test_it_credits_the_checkpoint_when_that_is_the_deciding_term(self):
        rep = TestEconomics._rep(n_turns=60, interruptions=1.0)
        text = hv.format_report(rep)
        if rep.worth_it_checkpointed and not rep.worth_it_cold:
            assert "checkpoint is what makes this work" in text

    def test_it_refuses_when_the_interruption_rate_is_too_high(self):
        rep = TestEconomics._rep(interruptions=1_000.0)
        assert not rep.worth_it_checkpointed
        assert "Not worth it" in hv.format_report(rep)

    def test_it_always_states_the_uniform_assumption(self):
        assert "uniform" in hv.format_report(TestEconomics._rep())

    def test_an_empty_workload_says_so(self):
        assert "Nothing to model" in hv.format_report(hv.analyse({}))

    def test_json_is_finite_and_complete(self):
        payload = TestEconomics._rep().to_json()
        text = json.dumps(payload)
        assert "NaN" not in text
        assert payload["uniform_interruption_assumption"] is True
        assert "breakeven_interruptions_checkpointed" in payload


class TestCli:
    def test_it_runs_against_a_fixture(self, write_jsonl, capsys, isolated_home):
        recs = [{
            "type": "assistant", "sessionId": "s",
            "timestamp": f"2026-08-01T10:{i:02d}:00Z",
            "message": {"id": f"m{i}", "model": "claude-opus-5",
                        "usage": {"input_tokens": 2,
                                  "cache_read_input_tokens": 20_000 + 500 * i,
                                  "cache_creation_input_tokens": 100,
                                  "output_tokens": 400}}} for i in range(10)]
        root = write_jsonl(recs, into=None)
        assert hv.main([str(root)]) == 0
        assert capsys.readouterr().out.strip()

    def test_json_parses(self, write_jsonl, capsys, isolated_home):
        recs = [{
            "type": "assistant", "sessionId": "s",
            "timestamp": f"2026-08-01T10:{i:02d}:00Z",
            "message": {"id": f"m{i}", "model": "claude-opus-5",
                        "usage": {"input_tokens": 2,
                                  "cache_read_input_tokens": 20_000,
                                  "cache_creation_input_tokens": 100,
                                  "output_tokens": 400}}} for i in range(6)]
        root = write_jsonl(recs, into=None)
        assert hv.main([str(root), "--json"]) == 0
        json.loads(capsys.readouterr().out)

    def test_an_empty_root_exits_one(self, tmp_path, capsys, isolated_home):
        assert hv.main([str(tmp_path)]) == 1
        assert capsys.readouterr().out.strip()
