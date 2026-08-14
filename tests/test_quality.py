"""Performance proxies: the guard that stops a cost cut from hiding a regression."""

import json

from router.quality import QualityStats, regressions, scan

OPUS = "claude-opus-5"


def _write(tmp_path, records):
    (tmp_path / "s.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    return tmp_path


class TestScan:
    def test_tool_errors_are_counted(self, tmp_path):
        root = _write(tmp_path, [
            {"type": "assistant", "message": {"id": "m1", "model": OPUS, "content": [
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "boom",
                 "is_error": True}]}},
        ])
        q = scan(root)
        assert q.tool_calls == 1 and q.tool_errors == 1
        assert q.tool_error_rate == 1.0

    def test_corrections_are_detected(self, tmp_path):
        root = _write(tmp_path, [
            {"type": "user", "message": {"content": [{"type": "text", "text": "no, that's wrong"}]}},
            {"type": "user", "message": {"content": [{"type": "text", "text": "add a test"}]}},
        ])
        q = scan(root)
        assert q.user_prompts == 2 and q.corrections == 1
        assert q.correction_rate == 0.5

    def test_tool_replies_are_not_human_prompts(self, tmp_path):
        _write(tmp_path, [
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "output"}]}},
        ])
        assert scan(tmp_path).user_prompts == 0

    def test_injected_meta_is_not_a_human_prompt(self, tmp_path):
        root = _write(tmp_path, [
            {"type": "user", "message": {"content": [
                {"type": "text", "text": "<command-name>/goal</command-name>"}]}},
        ])
        assert scan(root).user_prompts == 0

    def test_rework_ratio_counts_repeat_edits(self, tmp_path):
        edit = {"type": "tool_use", "id": "e", "name": "Edit",
                "input": {"file_path": "/a.py"}}
        root = _write(tmp_path, [
            {"type": "assistant", "message": {"id": "m1", "model": OPUS,
                                              "content": [edit, edit]}},
        ])
        q = scan(root)
        assert q.edits == 2 and q.rework_ratio == 2.0

    def test_empty_root_is_safe(self, tmp_path):
        q = scan(tmp_path)
        assert q.turns == 0 and q.tool_error_rate == 0.0


class TestRegressions:
    def _stats(self, **kw):
        q = QualityStats(**kw)
        return q

    def test_no_regression_when_metrics_improve(self):
        before = self._stats(tool_calls=100, tool_errors=10, user_prompts=10, turns=100)
        after = self._stats(tool_calls=100, tool_errors=5, user_prompts=10, turns=100)
        assert regressions(before, after) == []

    def test_rising_error_rate_is_flagged(self):
        before = self._stats(tool_calls=100, tool_errors=5, user_prompts=10, turns=100)
        after = self._stats(tool_calls=100, tool_errors=20, user_prompts=10, turns=100)
        regs = regressions(before, after)
        assert any("tool_error_rate" in r for r in regs)

    def test_small_moves_are_treated_as_noise(self):
        before = self._stats(tool_calls=1000, tool_errors=100, user_prompts=10, turns=100)
        after = self._stats(tool_calls=1000, tool_errors=105, user_prompts=10, turns=100)
        assert regressions(before, after) == []

    def test_more_turns_per_prompt_is_a_regression(self):
        before = self._stats(turns=100, user_prompts=10, tool_calls=1)
        after = self._stats(turns=200, user_prompts=10, tool_calls=1)
        assert any("turns_per_prompt" in r for r in regressions(before, after))
