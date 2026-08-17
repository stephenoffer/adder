"""The mid-session context decision: carry on, compact, or restart.

These are the only numbers in the tool that are consumed while the decision is
still available, so the bar is different from a report's: a wrong verdict here
throws away a working context. The tie rule -- carry on unless an alternative
is strictly positive -- is asserted directly, because both alternatives destroy
information that nothing in this repo prices.
"""

from __future__ import annotations

import json

import pytest

from adder.core.trace import Session
from adder.measure.session import live


@pytest.fixture
def report():
    def _make(**kw):
        base = {"turns": 100, "context": 300_000, "spent": 10.0, "per_turn": 0.1,
                "projected_remaining": 200, "projected_total": 30.0,
                "model": "claude-opus-5", "out_per_turn": 400, "ttl": "5m",
                "expected_remaining": 300.0, "read_mult": 0.10,
                "opening_cost": 0.13}
        base.update(kw)
        return live.LiveReport(**base)
    return _make


class TestCompactionNet:
    def test_a_long_horizon_makes_compacting_pay(self, report):
        assert report(expected_remaining=500).compaction_net() > 0

    def test_no_horizon_makes_compacting_a_loss(self, report):
        # Both terms: `carry_turns` falls back to the median when no mean is
        # given, so zeroing one of them is not a session with no future.
        assert report(expected_remaining=0,
                      projected_remaining=0).compaction_net() < 0

    def test_it_scales_with_the_context(self, report):
        big = report(context=800_000).compaction_net()
        small = report(context=80_000).compaction_net()
        assert big > small

    def test_a_worse_cache_makes_compacting_pay_sooner(self, report):
        assert (report(read_mult=0.5).compaction_net()
                > report(read_mult=0.1).compaction_net())

    def test_the_one_hour_ttl_makes_the_rebuild_dearer(self, report):
        assert report(ttl="1h").compaction_net() < report(ttl="5m").compaction_net()


class TestRestartNet:
    def test_a_large_context_and_long_horizon_favours_restarting(self, report):
        assert report(context=500_000, expected_remaining=400).restart_net() > 0

    def test_a_context_at_the_handoff_size_has_nothing_to_free(self, report):
        assert report(context=2_000).restart_net() < 0

    def test_a_dearer_opening_makes_restarting_worse(self, report):
        assert (report(opening_cost=5.0).restart_net()
                < report(opening_cost=0.1).restart_net())

    def test_restarting_beats_compacting_at_a_large_context(self, report):
        r = report(context=600_000, expected_remaining=400)
        assert r.restart_net() > r.compaction_net()


class TestVerdict:
    def test_a_short_session_is_told_to_carry_on(self, report):
        verdict, worth = report(context=40_000, expected_remaining=5).context_verdict()
        assert verdict == "carry on"
        assert worth == 0.0

    def test_a_long_full_session_is_told_to_act(self, report):
        verdict, worth = report(context=600_000,
                                expected_remaining=400).context_verdict()
        assert verdict in ("compact", "restart")
        assert worth > 0

    def test_the_worth_is_the_better_of_the_two(self, report):
        r = report(context=600_000, expected_remaining=400)
        _, worth = r.context_verdict()
        assert worth == pytest.approx(max(r.compaction_net(), r.restart_net()))


class TestDuplicateReads:
    def _transcript(self, tmp_path, n=2):
        recs = []
        for i in range(n):
            recs.append({"type": "assistant", "sessionId": "s",
                         "timestamp": f"2026-08-01T10:{i:02d}:00Z",
                         "message": {"id": f"m{i}", "model": "claude-opus-5",
                                     "usage": {"input_tokens": 1,
                                               "cache_read_input_tokens": 30_000,
                                               "cache_creation_input_tokens": 0,
                                               "output_tokens": 100},
                                     "content": [{"type": "tool_use", "id": f"u{i}",
                                                  "name": "Read",
                                                  "input": {"file_path": "/a.py"}}]}})
            recs.append({"type": "user", "sessionId": "s",
                         "message": {"content": [{"type": "tool_result",
                                                  "tool_use_id": f"u{i}",
                                                  "content": "x" * 4_000}]}})
        p = tmp_path / "s.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in recs))
        return p

    def test_a_repeated_read_is_found(self, tmp_path):
        ids, tokens = live.duplicate_reads(self._transcript(tmp_path, 2))
        assert ids == 1
        assert tokens > 0

    def test_a_single_read_is_not(self, tmp_path):
        assert live.duplicate_reads(self._transcript(tmp_path, 1)) == (0, 0)

    def test_no_transcript_is_survivable(self):
        assert live.duplicate_reads(None) == (0, 0)


class TestOpeningFromSession:
    def test_it_uses_the_sessions_own_first_turn(self, make_turn):
        from adder.measure.window.prefix import Opening

        s = Session("s", "p")
        s.turns = [make_turn(read=20_000, write=5_000, uncached=100)]
        op = Opening.from_session(s)
        assert op.measured
        assert op.floor_tokens == s.turns[0].context
        assert op.read_tokens == 20_000

    def test_an_empty_session_falls_back_to_the_prior(self):
        from adder.measure.window.prefix import Opening

        assert not Opening.from_session(Session("s", "p")).measured
