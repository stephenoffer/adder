"""Growth attribution: the finding that reversed the repo's headline advice."""
from __future__ import annotations

import json

import pytest

from adder.core.trace import Session, Turn
from adder.measure.window.context import (
    _est_tokens,
    _text_of,
    measured_growth,
    output_share_of_growth,
    scan,
)

OPUS = "claude-opus-5"


def _write(tmp_path, records):
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records))
    return tmp_path


def _sess(contexts, outs, sidechain=False):
    s = Session("s", "p")
    for c, o in zip(contexts, outs, strict=True):
        s.turns.append(Turn("s", "p", OPUS, 0, c, 0, o, 0, sidechain))
    return s


class TestFlatten:
    def test_plain_string(self):
        assert _text_of("hello") == "hello"

    def test_nested_tool_result(self):
        assert "deep" in _text_of([{"type": "tool_result", "content": [{"text": "deep"}]}])

    def test_tool_input_is_counted(self):
        assert _text_of([{"input": {"cmd": "ls"}}])

    def test_garbage_is_not_fatal(self):
        assert _text_of(None) == "" and _text_of(12) == ""

    def test_token_estimate_is_chars_over_four(self):
        assert _est_tokens("a" * 400) == 100


class TestMeasuredGrowth:
    def test_sums_positive_deltas_only(self):
        s = _sess([100, 200, 150, 300], [0, 0, 0, 0])
        # +100, -50 (ignored), +150
        assert measured_growth({"s": s}) == 250

    def test_no_turns_is_zero(self):
        assert measured_growth({}) == 0

    def test_output_share_uses_main_chain_only(self):
        s = _sess([0, 1000], [500, 500])
        assert output_share_of_growth({"s": s}) == pytest.approx(1.0)

    def test_sidechain_output_is_excluded(self):
        s = _sess([0, 1000], [500, 500], sidechain=True)
        assert output_share_of_growth({"s": s}) == 0.0

    def test_zero_growth_gives_zero_share(self):
        assert output_share_of_growth({"s": _sess([100], [500])}) == 0.0


class TestScan:
    def test_tool_results_are_attributed_to_their_tool(self, tmp_path):
        root = _write(tmp_path, [
            {"type": "assistant", "message": {"id": "m1", "model": OPUS, "content": [
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "x" * 4000}]}},
        ])
        g = scan(root)
        assert g.by_tool["Bash"] == 1000
        assert g.tool_calls["Bash"] == 1
        assert g.tool_results == 1000

    def test_user_text_is_separated_from_tool_output(self, tmp_path):
        root = _write(tmp_path, [
            {"type": "user", "message": {"content": [{"type": "text", "text": "y" * 400}]}},
        ])
        g = scan(root)
        assert g.user_messages == 100 and g.tool_results == 0

    def test_thinking_text_is_empty_but_still_billed(self, tmp_path):
        """Opus 5 returns empty thinking text; estimating from text undercounts.

        This is why `report` uses billed output tokens rather than text length.
        """
        root = _write(tmp_path, [
            {"type": "assistant", "message": {"id": "m1", "model": OPUS, "content": [
                {"type": "thinking", "thinking": ""}]}},
        ])
        g = scan(root)
        assert g.assistant_text == 0

    def test_duplicate_block_records_do_not_double_count_text(self, tmp_path):
        """Same message id written once per content block."""
        msg = {"id": "m1", "model": OPUS, "content": [{"type": "text", "text": "z" * 400}]}
        root = _write(tmp_path, [
            {"type": "assistant", "message": msg},
            {"type": "assistant", "message": msg},
        ])
        assert scan(root).assistant_text == 100

    def test_malformed_lines_are_skipped(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_text('not json\n{"type":"user","message":{"content":"%s"}}\n' % ("k" * 400))
        assert scan(tmp_path).user_messages > 0


class TestWindowScoping:
    """A windowed report must not print filtered billing beside unfiltered
    attribution: two numbers from different populations, laid out as though
    they described each other."""

    def _records(self, tmp_path):
        import json as _json

        recs = []
        for day, session in (("2026-01-05", "old"), ("2026-08-14", "new")):
            recs.append({
                "type": "assistant", "sessionId": session,
                "timestamp": f"{day}T10:00:00Z",
                "message": {"id": f"m-{session}", "model": "claude-opus-5",
                            "usage": {"input_tokens": 1, "output_tokens": 10},
                            "content": [{"type": "tool_use", "id": f"u-{session}",
                                         "name": "Bash", "input": {}}]}})
            recs.append({
                "type": "user", "sessionId": session,
                "timestamp": f"{day}T10:00:01Z",
                "message": {"content": [{"type": "tool_result",
                                         "tool_use_id": f"u-{session}",
                                         "content": "x" * 4000}]}})
        (tmp_path / "s.jsonl").write_text("\n".join(_json.dumps(r) for r in recs))
        return tmp_path

    def test_a_date_window_scopes_the_attribution(self, tmp_path):
        from datetime import date

        from adder.core.filters import Window
        from adder.measure.window.context import scan

        root = self._records(tmp_path)
        everything = scan(root)
        recent = scan(root, window=Window(since=date(2026, 6, 1)))
        assert recent.tool_results * 2 == everything.tool_results

    def test_a_session_window_scopes_the_attribution(self, tmp_path):
        from adder.core.filters import Window
        from adder.measure.window.context import scan

        root = self._records(tmp_path)
        assert scan(root, window=Window(sessions=("new",))).tool_results > 0
        assert (scan(root, window=Window(sessions=("new",))).tool_results
                < scan(root).tool_results)

    def test_no_window_keeps_everything(self, tmp_path):
        from adder.core.filters import Window
        from adder.measure.window.context import scan

        root = self._records(tmp_path)
        assert scan(root, window=Window()).tool_results == scan(root).tool_results


class TestSidechainsDoNotCountAsMainChainGrowth:
    """A subagent's context is not the session's context.

    Sidechain turns are interleaved with the parent's in the same session and
    run in their own short-lived context. Pairing them with main-chain turns
    walks the context down to a few thousand tokens and back up to several
    hundred thousand, and records that climb as growth that never happened.
    `output_share_of_growth` already excluded sidechains from its numerator, so
    only the denominator was contaminated and the share came out several times
    too small -- and that share scales every verbosity claim in `adder debt`.
    """

    @staticmethod
    def _sess(turns):
        s = Session("s", "p")
        s.turns = turns
        return s

    @staticmethod
    def _turn(read, *, side=False, out=1_000):
        return Turn("s", "p", "claude-opus-5", uncached_in=0, cache_read=read,
                    cache_write=0, out=out, thinking=0, sidechain=side)

    def test_a_sidechain_turn_does_not_invent_growth(self):
        main = self._sess([self._turn(100_000), self._turn(110_000),
                           self._turn(120_000)])
        mixed = self._sess([self._turn(100_000), self._turn(5_000, side=True),
                            self._turn(110_000), self._turn(120_000)])
        assert measured_growth({"s": mixed}) == measured_growth({"s": main})

    def test_growth_is_the_main_chain_delta(self):
        mixed = self._sess([self._turn(100_000), self._turn(5_000, side=True),
                            self._turn(110_000), self._turn(120_000)])
        assert measured_growth({"s": mixed}) == 20_000

    def test_the_share_is_measured_over_one_population(self):
        """Numerator and denominator must both exclude sidechains."""
        main = self._sess([self._turn(100_000), self._turn(110_000),
                           self._turn(120_000)])
        mixed = self._sess([self._turn(100_000), self._turn(5_000, side=True),
                            self._turn(110_000), self._turn(120_000)])
        assert output_share_of_growth({"s": mixed}) == pytest.approx(
            output_share_of_growth({"s": main}))

    def test_a_session_that_is_all_sidechain_has_no_main_chain_growth(self):
        only = self._sess([self._turn(5_000, side=True),
                           self._turn(9_000, side=True)])
        assert measured_growth({"s": only}) == 0


class TestAReplayedRecordIsNotMoreGrowth:
    """Two mechanisms replay records into this scan, and both inflate a ratio.

    Claude Code writes one JSONL record per content block, repeating the
    message envelope on each; a resumed session writes a NEW transcript that
    restates earlier turns. The shares this scan produces feed
    `output_share_of_growth`, which scales every terseness claim in the repo --
    and an inflation of exactly this kind is what put that number at 105% once
    already (see `debt.py`).
    """

    def _records(self):
        use = {"type": "assistant", "sessionId": "s",
               "timestamp": "2026-08-01T10:00:00Z",
               "message": {"id": "m1", "model": "claude-opus-5", "content": [
                   {"type": "text", "text": "x" * 400},
                   {"type": "tool_use", "id": "t1", "name": "Bash",
                    "input": {"command": "ls"}}]}}
        res = {"type": "user", "sessionId": "s",
               "timestamp": "2026-08-01T10:01:00Z",
               "message": {"content": [
                   {"type": "tool_result", "tool_use_id": "t1",
                    "content": "y" * 4_000}]}}
        return use, res

    def test_a_replayed_turn_is_counted_once(self, write_jsonl):
        from adder.measure.window.context import scan

        use, res = self._records()
        once = scan(write_jsonl([use, res], name="a.jsonl"))
        d = write_jsonl([use, res], name="a.jsonl")
        write_jsonl([use, res], name="b.jsonl", into=d)   # the resumed session
        twice = scan(d)
        assert twice.tool_results == once.tool_results
        assert twice.assistant_text == once.assistant_text
        assert twice.tool_calls == once.tool_calls

    def test_two_genuinely_distinct_calls_both_count(self, write_jsonl):
        from adder.measure.window.context import scan

        use, res = self._records()
        second_use = json.loads(json.dumps(use))
        second_use["message"]["id"] = "m2"
        second_use["message"]["content"][1]["id"] = "t2"
        second_res = json.loads(json.dumps(res))
        second_res["message"]["content"][0]["tool_use_id"] = "t2"
        g = scan(write_jsonl([use, res, second_use, second_res]))
        assert g.tool_calls["Bash"] == 2
        assert g.tool_results == 2 * 1_000
