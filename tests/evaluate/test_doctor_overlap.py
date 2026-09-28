"""Two doctor numbers that disagreed with the modules they quote.

The tools headline divided by tool results alone, the error `carried_cost`
records fixing, so it said "Read is 100% of context growth" on a history where
the assistant's own output was most of it. And the "at stake" total added up
checks that price the same re-read tokens, while the footer below it said the
levers do not add.
"""

from __future__ import annotations

import json
import re

from adder.core.trace import load_sessions
from adder.evaluate.doctor import Check, check_tools, report, stake_range
from adder.measure.window.tools import billed_output, scan


def _history(tmp_path):
    recs = []
    for i in range(20):
        recs.append({
            "type": "assistant", "sessionId": "s", "timestamp": f"2026-08-01T10:{i:02d}:00Z",
            "message": {"id": f"m{i}", "model": "claude-opus-5", "content": [
                {"type": "tool_use", "id": f"u{i}", "name": "Read",
                 "input": {"file_path": f"/w/f{i}.py"}}],
                "usage": {"input_tokens": 1, "cache_read_input_tokens": 50_000 + 3_000 * i,
                          "output_tokens": 2_000}}})
        recs.append({
            "type": "user", "sessionId": "s", "timestamp": f"2026-08-01T10:{i:02d}:30Z",
            "message": {"content": [{"type": "tool_result", "tool_use_id": f"u{i}",
                                     "content": "x" * 4_000}]}})
    (tmp_path / "s.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    return tmp_path


def test_the_tools_share_counts_assistant_output(tmp_path, isolated_home):
    root = _history(tmp_path)
    sessions = load_sessions(root, use_cache=False)
    rep = scan(root)
    worst = rep.by_tool["Read"]
    want = rep.share_of_growth(worst, billed_output(sessions))
    assert want < rep.share_of_growth(worst)          # the fixture discriminates
    headline = check_tools(root, sessions, 1_000.0).headline
    assert f"{want:.0%} of context growth" in headline
    assert "100%" not in headline


class TestTheStakeIsARange:
    def test_overlapping_findings_are_not_summed_into_one_figure(self):
        fixes = [Check("tools", False, "a", action="x", dollars=300.0),
                 Check("reread", False, "b", action="y", dollars=200.0)]
        assert stake_range(fixes) == (300.0, 500.0)
        text = report(fixes)
        assert "$300.00 to $500.00 at stake" in text
        assert not re.search(r"—[^\n]*, \$500\.00 at stake", text)

    def test_one_finding_is_one_figure(self):
        text = report([Check("tools", False, "a", action="x", dollars=300.0)])
        assert "$300.00 at stake" in text and " to " not in text.split("at stake")[0][-30:]

    def test_the_json_floor_and_ceiling(self, tmp_path, capsys, isolated_home):
        from adder.evaluate.doctor import main

        main([str(_history(tmp_path)), "--json"])
        d = json.loads(capsys.readouterr().out)
        assert d["at_stake"] <= d["at_stake_upper"]
