"""The session ranking, and the sort keys it promises."""

from __future__ import annotations

import json

import pytest

from adder.measure.spend.sessions import SORTS, Row, rank, report


class TestRank:
    def test_default_sort_is_by_cost(self, make_sessions, make_session):
        sessions = make_sessions(2, 10)
        sessions["big"] = make_session(200, sid="big")
        assert rank(sessions)[0].session.id == "big"

    def test_every_declared_sort_works(self, make_sessions):
        sessions = make_sessions(3, 20)
        for key in SORTS:
            assert len(rank(sessions, key)) == 3

    def test_unknown_sort_is_an_error(self, make_sessions):
        with pytest.raises(ValueError):
            rank(make_sessions(1, 5), "vibes")

    def test_per_turn_finds_the_expensive_short_session(self, make_session):
        cheap = make_session(400, sid="long", base=5_000, growth=0)
        pricey = make_session(10, sid="dense", base=900_000, growth=0)
        rows = rank({"long": cheap, "dense": pricey}, "per-turn")
        assert rows[0].session.id == "dense"
        assert rank({"long": cheap, "dense": pricey}, "cost")[0].session.id == "long"

    def test_recent_sorts_by_start_time(self, make_session):
        old = make_session(5, sid="old")
        new = make_session(5, sid="new", minutes_apart=1)
        for t in new.turns:
            t.ts = t.ts.replace("2026-08-01", "2026-08-09")
        assert rank({"old": old, "new": new}, "recent")[0].session.id == "new"


class TestRow:
    def test_per_turn_of_an_empty_session_does_not_divide_by_zero(self):
        from adder.core.trace import Session

        assert Row(Session("x", "p")).per_turn == 0.0

    def test_rebuild_cost_is_zero_without_rebuilds(self, make_session):
        assert Row(make_session(20)).rebuild_cost == 0.0

    def test_rebuild_cost_prices_only_the_excess(self, make_session, make_turn):
        s = make_session(5)
        s.turns.append(make_turn(read=0, write=200_000, minutes=99))
        row = Row(s)
        assert row.rebuilds == 1
        # The excess over a cache read, not the whole write.
        assert 0 < row.rebuild_cost < 200_000 * 5 * 1.25 / 1e6


class TestReport:
    def test_empty(self):
        assert "No sessions" in report({})

    def test_shows_a_row_per_session_up_to_top(self, make_sessions):
        text = report(make_sessions(5, 10), top=2)
        assert "3 more" in text

    def test_reports_concentration(self, make_sessions):
        assert "concentration" in report(make_sessions(3, 10))


class TestCli:
    def test_json(self, tmp_path, capsys, write_jsonl):
        from adder.measure.spend.sessions import main

        write_jsonl([{"type": "assistant", "sessionId": "s",
                      "timestamp": "2026-08-01T10:00:00Z",
                      "message": {"id": "m1", "model": "claude-opus-5",
                                  "usage": {"input_tokens": 1,
                                            "cache_read_input_tokens": 5000,
                                            "output_tokens": 100},
                                  "content": []}}])
        assert main([str(tmp_path), "--json"]) == 0
        d = json.loads(capsys.readouterr().out)
        assert d["rows"][0]["id"] == "s"
        assert d["sort"] == "cost"

    def test_no_sessions_exits_one(self, tmp_path, capsys):
        from adder.measure.spend.sessions import main

        assert main([str(tmp_path)]) == 1
        assert "No sessions" in capsys.readouterr().out
