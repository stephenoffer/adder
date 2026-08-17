"""Export: the columns, the formats, and the promise that no prompt text leaves.

The privacy assertion is the one that matters. Transcripts hold source code and
prompts; an export that carried any of it would end up pasted into a ticket.
"""

from __future__ import annotations

import csv
import io
import json

import pytest

from adder.measure.spend.export import (
    DAY_COLUMNS,
    FORMATS,
    GRAINS,
    SESSION_COLUMNS,
    TURN_COLUMNS,
    render,
    rows_for,
)


class TestGrains:
    def test_turn_grain_is_one_row_per_turn(self, make_sessions):
        sessions = make_sessions(2, 10)
        cols, rows = rows_for(sessions, "turn")
        assert cols == TURN_COLUMNS
        assert len(rows) == 20

    def test_session_grain_is_one_row_per_session(self, make_sessions):
        cols, rows = rows_for(make_sessions(3, 10), "session")
        assert cols == SESSION_COLUMNS
        assert len(rows) == 3

    def test_day_grain_buckets_by_date(self, make_session):
        s = make_session(60, minutes_apart=60)      # spans more than one day
        cols, rows = rows_for({"s": s}, "day")
        assert cols == DAY_COLUMNS
        assert len(rows) >= 2
        assert sum(r["turns"] for r in rows) == 60

    def test_day_grain_keeps_undated_turns_in_their_own_bucket(self, make_session,
                                                               make_turn):
        s = make_session(3)
        s.turns.append(make_turn(ts=None))
        _, rows = rows_for({"s": s}, "day")
        assert any(r["day"] == "undated" for r in rows)

    def test_unknown_grain_is_an_error(self, make_sessions):
        with pytest.raises(ValueError):
            rows_for(make_sessions(1, 3), "fortnight")

    def test_every_declared_grain_runs(self, make_sessions):
        for g in GRAINS:
            assert rows_for(make_sessions(1, 5), g)


class TestTotalsReconcile:
    def test_turn_costs_sum_to_session_costs(self, make_sessions):
        sessions = make_sessions(2, 20)
        _, turns = rows_for(sessions, "turn")
        _, sess = rows_for(sessions, "session")
        assert sum(r["cost"] for r in turns) == pytest.approx(
            sum(r["cost"] for r in sess), rel=1e-6)

    def test_day_costs_sum_to_turn_costs(self, make_sessions):
        sessions = make_sessions(2, 20)
        _, turns = rows_for(sessions, "turn")
        _, days = rows_for(sessions, "day")
        assert sum(r["cost"] for r in turns) == pytest.approx(
            sum(r["cost"] for r in days), rel=1e-6)


class TestFormats:
    def test_csv_round_trips(self, make_sessions):
        cols, rows = rows_for(make_sessions(1, 5), "turn")
        parsed = list(csv.DictReader(io.StringIO(render(cols, rows, "csv"))))
        assert len(parsed) == 5
        assert list(parsed[0]) == list(cols)

    def test_jsonl_is_one_object_per_line(self, make_sessions):
        cols, rows = rows_for(make_sessions(1, 4), "turn")
        lines = render(cols, rows, "jsonl").strip().splitlines()
        assert len(lines) == 4
        assert json.loads(lines[0])["model"]

    def test_json_carries_the_column_order(self, make_sessions):
        cols, rows = rows_for(make_sessions(1, 2), "session")
        d = json.loads(render(cols, rows, "json"))
        assert d["columns"] == list(cols)

    def test_field_names_match_across_formats(self, make_sessions):
        cols, rows = rows_for(make_sessions(1, 2), "turn")
        from_csv = set(next(csv.DictReader(io.StringIO(render(cols, rows, "csv")))))
        from_jsonl = set(json.loads(render(cols, rows, "jsonl").splitlines()[0]))
        assert from_csv == from_jsonl

    def test_unknown_format_is_an_error(self, make_sessions):
        cols, rows = rows_for(make_sessions(1, 1), "turn")
        with pytest.raises(ValueError):
            render(cols, rows, "parquet")

    def test_every_declared_format_runs(self, make_sessions):
        cols, rows = rows_for(make_sessions(1, 2), "turn")
        for f in FORMATS:
            assert render(cols, rows, f)


class TestPrivacy:
    def test_no_column_can_hold_message_content(self):
        """A column list is the enforcement point; keep it inspectable."""
        banned = {"text", "content", "prompt", "message", "input", "output_text"}
        for cols in (TURN_COLUMNS, SESSION_COLUMNS, DAY_COLUMNS):
            assert not (set(cols) & banned)

    def test_prompt_text_does_not_appear_in_an_export(self, tmp_path, write_jsonl,
                                                      capsys):
        from adder.measure.spend.export import main

        secret = "SUPER-SECRET-PROMPT-TEXT"
        write_jsonl([
            {"type": "user", "sessionId": "s", "timestamp": "2026-08-01T10:00:00Z",
             "message": {"content": [{"type": "text", "text": secret}]}},
            {"type": "assistant", "sessionId": "s",
             "timestamp": "2026-08-01T10:00:01Z",
             "message": {"id": "m1", "model": "claude-opus-5",
                         "usage": {"input_tokens": 1, "cache_read_input_tokens": 900,
                                   "output_tokens": 10},
                         "content": [{"type": "text", "text": secret}]}},
        ])
        assert main([str(tmp_path)]) == 0
        assert secret not in capsys.readouterr().out


class TestWriting:
    def _root(self, write_jsonl):
        return write_jsonl([
            {"type": "assistant", "sessionId": "s",
             "timestamp": "2026-08-01T10:00:00Z",
             "message": {"id": "m1", "model": "claude-opus-5",
                         "usage": {"input_tokens": 1, "cache_read_input_tokens": 900,
                                   "output_tokens": 10}, "content": []}}])

    def test_writes_to_a_named_file(self, write_jsonl, tmp_path):
        from adder.measure.spend.export import main

        root = self._root(write_jsonl)
        dest = tmp_path / "out" / "turns.csv"
        assert main([str(root), "-o", str(dest)]) == 0
        assert dest.read_text().startswith("timestamp,")

    def test_refuses_to_overwrite_without_force(self, write_jsonl, tmp_path, capsys):
        from adder.measure.spend.export import main

        root = self._root(write_jsonl)
        dest = tmp_path / "taken.csv"
        dest.write_text("mine")
        assert main([str(root), "-o", str(dest)]) == 1
        assert dest.read_text() == "mine"
        assert "--force" in capsys.readouterr().err

    def test_force_replaces(self, write_jsonl, tmp_path):
        from adder.measure.spend.export import main

        root = self._root(write_jsonl)
        dest = tmp_path / "taken.csv"
        dest.write_text("mine")
        assert main([str(root), "-o", str(dest), "--force"]) == 0
        assert dest.read_text() != "mine"
