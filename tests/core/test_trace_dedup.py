"""Deduplication: the measurement bug that inflated every headline figure 1.7x.

Claude Code writes one JSONL record per content block, each repeating the whole
message's `usage`. Summing lines multi-counts every turn that used a tool.
"""
from __future__ import annotations

import json

import pytest

from adder.core.trace import Session, Turn, iter_file, load_sessions, summarize

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
        import adder.core.trace as tr

        monkeypatch.setattr(tr, "CACHE_PATH", tmp_path / ".cache")
        _write(tmp_path, [_rec("m1"), _rec("m2")])
        cold = load_sessions(tmp_path, use_cache=True)
        warm = load_sessions(tmp_path, use_cache=True)
        assert cold["s"].n_turns == warm["s"].n_turns == 2

    def test_a_corrupt_cache_is_ignored(self, tmp_path, monkeypatch):
        import adder.core.trace as tr

        bad = tmp_path / ".cache"
        bad.write_bytes(b"not a pickle")
        monkeypatch.setattr(tr, "CACHE_PATH", bad)
        _write(tmp_path, [_rec("m1")])
        assert load_sessions(tmp_path, use_cache=True)["s"].n_turns == 1

    def test_summarize_counts_deduped_turns(self, tmp_path, monkeypatch):
        import adder.core.trace as tr

        monkeypatch.setattr(tr, "CACHE_PATH", tmp_path / ".cache")
        _write(tmp_path, [_rec("m1"), _rec("m1"), _rec("m2")])
        s, _ = summarize(tmp_path)
        assert s.n_turns == 2


class TestTheParseCacheIsOnByDefault:
    """The cache defaulted off while its setting defaulted on.

    So memoization only happened where a caller had remembered to ask, and the
    paths that most needed it had not: `horizon.load` is reached from
    `live.analyse`, which both hooks call, so every prompt submission and every
    guarded read paid a full re-parse of every transcript on the machine.
    Measured on 222 local transcripts: 2,339ms cold against 81ms warm.
    """

    @staticmethod
    def _records():
        return [{"type": "assistant", "sessionId": "s",
                 "message": {"id": "m1", "model": "claude-opus-5",
                             "usage": {"input_tokens": 10, "cache_read_input_tokens": 100,
                                       "output_tokens": 5}}}]

    def test_a_second_load_reads_the_cache(self, isolated_home, write_jsonl, tmp_path):
        from adder.core.trace import CACHE_PATH, load_sessions

        d = write_jsonl(self._records(), into=tmp_path / "proj")
        load_sessions(d)
        assert CACHE_PATH.exists(), "the default must memoize, not re-parse"

    def test_it_can_still_be_forced_off(self, isolated_home, write_jsonl, tmp_path):
        from adder.core.trace import CACHE_PATH, load_sessions

        d = write_jsonl(self._records(), into=tmp_path / "proj")
        load_sessions(d, use_cache=False)
        assert not CACHE_PATH.exists()

    def test_the_setting_wins_over_the_default(self, isolated_home, write_jsonl,
                                               tmp_path, monkeypatch):
        from adder.core.trace import CACHE_PATH, load_sessions

        monkeypatch.setenv("ADDER_CACHE", "0")
        d = write_jsonl(self._records(), into=tmp_path / "proj")
        load_sessions(d)
        assert not CACHE_PATH.exists()

    def test_a_changed_file_is_re_read(self, isolated_home, write_jsonl, tmp_path):
        """Keyed by (mtime, size), so a cache hit can never serve stale turns."""
        from adder.core.trace import load_sessions

        d = tmp_path / "proj"
        write_jsonl(self._records(), into=d)
        assert sum(len(s.turns) for s in load_sessions(d).values()) == 1
        more = [*self._records(),
                {"type": "assistant", "sessionId": "s",
                 "message": {"id": "m2", "model": "claude-opus-5",
                             "usage": {"input_tokens": 1, "output_tokens": 1}}}]
        write_jsonl(more, into=d)
        assert sum(len(s.turns) for s in load_sessions(d).values()) == 2


class TestMalformedRecordsDoNotEndTheFile:
    """A record this reader cannot price costs one turn, never the rest.

    `iter_file` is consumed inside `load_sessions`, which has no handler, so an
    exception raised while building a `Turn` silently truncated the transcript
    at that record and reported the shortfall as a smaller bill -- the exact
    failure mode this package treats as unacceptable.
    """

    def _rec(self, i, **over):
        rec = {"type": "assistant", "sessionId": "s",
               "timestamp": f"2026-08-10T12:0{i}:00Z",
               "message": {"id": f"m{i}", "model": "claude-opus-5",
                           "usage": {"input_tokens": 1, "output_tokens": 10}}}
        for k, v in over.items():
            if k in ("model", "usage"):
                rec["message"][k] = v
            else:
                rec[k] = v
        return rec

    def test_a_non_string_model_does_not_stop_the_read(self, write_jsonl):
        from adder.core.trace import iter_file

        d = write_jsonl([self._rec(0), self._rec(1, model=5),
                         self._rec(2, model={"a": 1}), self._rec(3)])
        turns = list(iter_file(d / "s.jsonl"))
        assert [t.msg_id for t in turns] == ["m0", "m3"]

    def test_a_non_dict_usage_does_not_stop_the_read(self, write_jsonl):
        from adder.core.trace import iter_file

        d = write_jsonl([self._rec(0), self._rec(1, usage="lots"),
                         self._rec(2, usage=[1, 2]), self._rec(3)])
        assert [t.msg_id for t in iter_file(d / "s.jsonl")] == ["m0", "m3"]

    def test_a_string_cache_creation_bucket_is_counted_as_zero(self, write_jsonl):
        from adder.core.trace import iter_file

        r = self._rec(0, usage={"input_tokens": 1, "output_tokens": 10,
                                "cache_creation": {"ephemeral_5m_input_tokens": "3",
                                                   "ephemeral_1h_input_tokens": {}}})
        turns = list(iter_file(write_jsonl([r]) / "s.jsonl"))
        assert [t.ttl for t in turns] == ["5m"]

    def test_a_non_dict_output_tokens_details_is_not_fatal(self, write_jsonl):
        from adder.core.trace import iter_file

        r = self._rec(0, usage={"input_tokens": 1, "output_tokens": 10,
                                "output_tokens_details": "none"})
        assert [t.thinking for t in iter_file(write_jsonl([r]) / "s.jsonl")] == [0]

    def test_an_unhashable_tool_name_stays_hashable(self, write_jsonl):
        from adder.core.trace import iter_file

        r = self._rec(0)
        r["message"]["content"] = [{"type": "tool_use", "name": {"weird": 1}}]
        turns = list(iter_file(write_jsonl([r]) / "s.jsonl"))
        assert all(isinstance(name, str) for name in turns[0].tools)


class TestMixedTimezoneStamps:
    """A session assembled from a Claude Code file and a foreign log has both.

    `ingest` accepts OpenAI and OTel exports whose timestamps carry no UTC
    offset. Ordering those against Claude Code's `Z`-stamped ones raised
    `TypeError: can't compare offset-naive and offset-aware datetimes` from
    `gaps()`, which decides the cache TTL, and from `started`/`ended`.
    """

    def _sess(self):
        from adder.core.trace import Session, Turn

        s = Session("s", "p")
        s.turns = [
            Turn("s", "p", "claude-opus-5", 0, 1000, 0, 10, 0, False,
                 ts="2026-08-10T12:00:00Z"),
            Turn("s", "p", "claude-opus-5", 0, 1000, 0, 10, 0, False,
                 ts="2026-08-10T13:00:00"),
        ]
        return s

    def test_gaps_orders_mixed_stamps(self):
        assert len(self._sess().gaps()) == 1

    def test_started_and_ended_do_not_raise(self):
        s = self._sess()
        assert s.started is not None and s.ended is not None
        assert s.wall_seconds >= 0.0
