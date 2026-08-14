"""Growth attribution: the finding that reversed the repo's headline advice."""

import json

import pytest

from adder.context import (
    _est_tokens,
    _text_of,
    measured_growth,
    output_share_of_growth,
    scan,
)
from adder.trace import Session, Turn

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
