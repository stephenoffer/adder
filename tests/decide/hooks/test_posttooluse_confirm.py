"""The PostToolUse hook: silent, fail-open, and the only place a read lands."""

from __future__ import annotations

import io
import json

from adder.decide import guard as lib
from adder.decide.hooks import posttooluse_confirm as hook


def _run(payload, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    assert hook.main() == 0
    return capsys.readouterr().out


def test_it_promotes_a_pending_read_and_prints_nothing(tmp_path, monkeypatch, capsys,
                                                       isolated_home):
    state_path = tmp_path / "g.json"
    monkeypatch.setenv("ADDER_GUARD_STATE", str(state_path))
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    st = lib.load_state("s1", state_path)
    st.post_seen = 1
    lib.observe("Read", {"file_path": str(f)}, st, lib.Verdict(False, "x"))
    lib.save_state("s1", st, state_path)
    out = _run({"session_id": "s1", "tool_name": "Read", "tool_input": {"file_path": str(f)},
                "tool_response": {"type": "text", "file": {
                    "filePath": str(f), "numLines": 1, "startLine": 1, "totalLines": 1}}},
               monkeypatch, capsys)
    assert out == ""
    assert str(f) in lib.load_state("s1", state_path).reads


def test_garbage_is_ignored(monkeypatch, capsys):
    for payload in ([], {"tool_name": "Read", "tool_input": "no"}, {"tool_name": "Edit"}):
        assert _run(payload, monkeypatch, capsys) == ""
