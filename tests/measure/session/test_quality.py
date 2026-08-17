"""Performance proxies: the guard that stops a cost cut from hiding a regression."""
from __future__ import annotations

import json

from adder.measure.session.quality import QualityStats, regressions, scan

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
        # Distinct block ids, because two edits are two calls. Reusing one id
        # for both described the *same* call twice, which is the shape a
        # resumed transcript replays and which the scan now deduplicates.
        def edit(uid):
            return {"type": "tool_use", "id": uid, "name": "Edit",
                    "input": {"file_path": "/a.py"}}

        root = _write(tmp_path, [
            {"type": "assistant", "message": {"id": "m1", "model": OPUS,
                                              "content": [edit("e1"), edit("e2")]}},
        ])
        q = scan(root)
        assert q.edits == 2 and q.rework_ratio == 2.0

    def test_a_replayed_record_is_not_a_second_edit(self, tmp_path):
        """A resumed session restates earlier turns in a new transcript."""
        rec = {"type": "assistant", "message": {
            "id": "m1", "model": OPUS,
            "content": [{"type": "tool_use", "id": "e1", "name": "Edit",
                         "input": {"file_path": "/a.py"}}]}}
        root = _write(tmp_path, [rec, rec])
        q = scan(root)
        assert q.edits == 1 and q.tool_calls == 1 and q.turns == 1

    def test_a_replayed_tool_error_is_counted_once(self, tmp_path):
        use = {"type": "assistant", "message": {
            "id": "m1", "model": OPUS,
            "content": [{"type": "tool_use", "id": "t1", "name": "Bash",
                         "input": {"command": "false"}}]}}
        err = {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "boom",
             "is_error": True}]}}
        root = _write(tmp_path, [use, err, use, err])
        q = scan(root)
        assert q.tool_calls == 1 and q.tool_errors == 1

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


class TestApiErrors:
    """`<synthetic>` records are client-side failures: unbilled, but real."""

    def _write(self, tmp_path, records):
        import json as _json

        (tmp_path / "s.jsonl").write_text(
            "\n".join(_json.dumps(r) for r in records))
        return tmp_path

    def _assistant(self, mid, model="claude-opus-5"):
        return {"type": "assistant", "sessionId": "s",
                "timestamp": "2026-08-01T10:00:00Z",
                "message": {"id": mid, "model": model,
                            "usage": {"input_tokens": 1, "output_tokens": 10},
                            "content": [{"type": "text", "text": "hi"}]}}

    def test_counted_separately_from_turns(self, tmp_path):
        from adder.measure.session.quality import scan

        root = self._write(tmp_path, [
            self._assistant("m1"),
            self._assistant("m2", model="<synthetic>"),
        ])
        q = scan(root)
        assert q.turns == 1
        assert q.api_errors == 1
        assert q.api_error_rate == 1.0

    def test_absent_when_nothing_failed(self, tmp_path):
        from adder.measure.session.quality import scan

        q = scan(self._write(tmp_path, [self._assistant("m1")]))
        assert q.api_errors == 0
        assert q.api_error_rate == 0.0

    def test_excluded_from_the_regression_check(self):
        """Network flakiness must not fail a cost change."""
        from adder.measure.session.quality import WORSE_IF_HIGHER

        assert "api_error_rate" not in WORSE_IF_HIGHER

    def test_no_turns_is_not_a_division_error(self):
        from adder.measure.session.quality import QualityStats

        assert QualityStats().api_error_rate == 0.0


class TestWindowScoping:
    def _root(self, tmp_path):
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
                                         "content": "ok"}]}})
        (tmp_path / "s.jsonl").write_text("\n".join(_json.dumps(r) for r in recs))
        return tmp_path

    def test_a_window_scopes_the_proxies(self, tmp_path):
        from adder.core.filters import Window
        from adder.measure.session.quality import scan

        root = self._root(tmp_path)
        assert scan(root).tool_calls == 2
        assert scan(root, window=Window(sessions=("new",))).tool_calls == 1

    def test_the_bare_dates_still_work_for_compare(self, tmp_path):
        from datetime import date

        from adder.measure.session.quality import compare

        before, after = compare(self._root(tmp_path), date(2026, 6, 1))
        assert before.tool_calls == 1
        assert after.tool_calls == 1


class TestARegressionFromZero:
    """The clearest possible degradation was the one this could not see.

    `regressions` skipped any metric whose baseline was zero, because the
    relative move is undefined there. Undefined is not the same as zero: a tool
    error rate going 0 -> 30% and a correction rate going 0 -> 50% both returned
    "nothing regressed" from the function whose entire job is to falsify "we cut
    cost and nothing got worse".
    """

    @staticmethod
    def _stats(**kw):
        s = QualityStats(**kw)
        s.edited_files.update({"a": 5, "b": 5})
        return s

    def _clean(self):
        return self._stats(turns=100, tool_calls=200, tool_errors=0,
                           user_prompts=50, corrections=0, interrupts=0, edits=10)

    def _degraded(self):
        return self._stats(turns=100, tool_calls=200, tool_errors=60,
                           user_prompts=50, corrections=25, interrupts=10, edits=10)

    def test_a_metric_appearing_from_zero_is_a_regression(self):
        assert regressions(self._clean(), self._degraded())

    def test_every_appearing_metric_is_named(self):
        got = " ".join(regressions(self._clean(), self._degraded()))
        assert "tool_error_rate" in got
        assert "correction_rate" in got
        assert "interrupt_rate" in got

    def test_the_message_says_there_was_no_baseline(self):
        got = regressions(self._clean(), self._degraded())
        assert all("no baseline" in g for g in got)

    def test_an_unchanged_workload_still_reports_nothing(self):
        assert regressions(self._clean(), self._clean()) == []

    def test_an_improvement_is_not_a_regression(self):
        assert regressions(self._degraded(), self._clean()) == []

    def test_a_nonzero_baseline_still_uses_the_relative_test(self):
        before = self._stats(turns=100, tool_calls=200, tool_errors=10,
                             user_prompts=50, edits=10)
        after = self._stats(turns=100, tool_calls=200, tool_errors=11,
                            user_prompts=50, edits=10)
        assert regressions(before, after) == []      # +10% is inside tolerance
