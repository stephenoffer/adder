"""Deduplication: the measurement bug that inflated every headline figure 1.7x.

Claude Code writes one JSONL record per content block, each repeating the whole
message's `usage`. Summing lines multi-counts every turn that used a tool.
"""

import json

import pytest

from adder.trace import Session, Turn, iter_file, load_sessions, summarize

OPUS = "claude-opus-5"


def _rec(mid, out=100, blocks=None, ts="2026-08-14T10:00:00Z", **usage):
    u = {"input_tokens": 2, "cache_read_input_tokens": 1000,
         "cache_creation_input_tokens": 0, "output_tokens": out}
    u.update(usage)
    return {"type": "assistant", "timestamp": ts, "sessionId": "s",
            "message": {"id": mid, "model": OPUS, "usage": u,
                        "content": blocks or [{"type": "text", "text": "hi"}]}}


def _write(tmp_path, records, name="s.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in records))
    return p


class TestDedup:
    def test_repeated_block_records_count_once(self, tmp_path):
        p = _write(tmp_path, [_rec("m1"), _rec("m1"), _rec("m1")])
        assert len(list(iter_file(p))) == 1

    def test_distinct_messages_are_kept(self, tmp_path):
        p = _write(tmp_path, [_rec("m1"), _rec("m2")])
        assert len(list(iter_file(p))) == 2

    def test_final_streamed_usage_wins(self, tmp_path):
        """Partial records carry a running output count; the last one completes it."""
        p = _write(tmp_path, [_rec("m1", out=3), _rec("m1", out=3), _rec("m1", out=1044)])
        turns = list(iter_file(p))
        assert len(turns) == 1 and turns[0].out == 1044

    def test_tool_names_merge_across_block_records(self, tmp_path):
        a = _rec("m1", blocks=[{"type": "tool_use", "name": "Bash"}])
        b = _rec("m1", out=200, blocks=[{"type": "tool_use", "name": "Read"}])
        turns = list(iter_file(_write(tmp_path, [a, b])))
        assert set(turns[0].tools) == {"Bash", "Read"}

    def test_records_without_an_id_are_not_collapsed(self, tmp_path):
        r = _rec("m1")
        del r["message"]["id"]
        p = _write(tmp_path, [dict(r), dict(r)])
        assert len(list(iter_file(p))) == 2

    def test_order_is_preserved(self, tmp_path):
        p = _write(tmp_path, [_rec("m1", out=1), _rec("m2", out=2), _rec("m3", out=3)])
        assert [t.out for t in iter_file(p)] == [1, 2, 3]


class TestTTLAndSpeed:
    def test_one_hour_bucket_is_detected(self, tmp_path):
        r = _rec("m1")
        r["message"]["usage"]["cache_creation"] = {
            "ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 5000}
        turns = list(iter_file(_write(tmp_path, [r])))
        assert turns[0].ttl == "1h"

    def test_five_minute_is_the_default(self, tmp_path):
        turns = list(iter_file(_write(tmp_path, [_rec("m1")])))
        assert turns[0].ttl == "5m"

    def test_one_hour_write_costs_more(self, tmp_path):
        base = _rec("m1", cache_creation_input_tokens=100_000)
        slow = _rec("m2", cache_creation_input_tokens=100_000)
        slow["message"]["usage"]["cache_creation"] = {
            "ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 100_000}
        turns = list(iter_file(_write(tmp_path, [base, slow])))
        assert turns[1].cost() > turns[0].cost()

    def test_fast_mode_is_priced_at_the_premium(self, tmp_path):
        fast = _rec("m1", out=1000)
        fast["message"]["usage"]["speed"] = "fast"
        std = _rec("m2", out=1000)
        turns = list(iter_file(_write(tmp_path, [fast, std])))
        assert turns[0].speed == "fast"
        assert turns[0].cost() == pytest.approx(2 * turns[1].cost(), rel=1e-6)


class TestSessionMetrics:
    def _sess(self):
        s = Session("s", "p")
        for i, ctx in enumerate((100_000, 200_000, 150_000)):
            s.turns.append(Turn("s", "p", OPUS, 0, ctx, 0, 100, 0, False,
                                ts=f"2026-08-14T10:{i:02d}:00Z"))
        return s

    def test_gaps_are_measured_in_seconds(self):
        assert self._sess().gaps() == [60.0, 60.0]
        assert self._sess().median_gap() == 60.0

    def test_base_context_is_the_floor(self):
        assert self._sess().base_context == 100_000

    def test_cost_on_date_matches_property(self):
        s = self._sess()
        assert s.cost_on() == pytest.approx(s.cost)

    def test_no_timestamps_means_no_gaps(self):
        s = Session("s", "p")
        s.turns.append(Turn("s", "p", OPUS, 0, 100, 0, 1, 0, False))
        assert s.gaps() == [] and s.median_gap() == 0.0


class TestParseCache:
    def test_cache_returns_the_same_totals(self, tmp_path, monkeypatch):
        import adder.trace as tr

        monkeypatch.setattr(tr, "CACHE_PATH", tmp_path / ".cache")
        _write(tmp_path, [_rec("m1"), _rec("m2")])
        cold = load_sessions(tmp_path, use_cache=True)
        warm = load_sessions(tmp_path, use_cache=True)
        assert cold["s"].n_turns == warm["s"].n_turns == 2

    def test_a_corrupt_cache_is_ignored(self, tmp_path, monkeypatch):
        import adder.trace as tr

        bad = tmp_path / ".cache"
        bad.write_bytes(b"not a pickle")
        monkeypatch.setattr(tr, "CACHE_PATH", bad)
        _write(tmp_path, [_rec("m1")])
        assert load_sessions(tmp_path, use_cache=True)["s"].n_turns == 1

    def test_summarize_counts_deduped_turns(self, tmp_path, monkeypatch):
        import adder.trace as tr

        monkeypatch.setattr(tr, "CACHE_PATH", tmp_path / ".cache")
        _write(tmp_path, [_rec("m1"), _rec("m1"), _rec("m2")])
        s, _ = summarize(tmp_path)
        assert s.n_turns == 2
