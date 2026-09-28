"""Each hook prices the session that called it, and survives a bad payload.

Two sessions open in one repository is the normal case, and both hooks used to
guess the session from the working directory: the newest transcript in the
project. A 25-turn session was told it had spent $287 over 300 turns, which was
the other session's bill, and from a subdirectory the guess found nothing. The
payload names the transcript; these tests hold the hooks to it.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import pathlib

import pytest

from adder.measure.session import live

HOOKS = pathlib.Path(__file__).resolve().parents[3] / "adder" / "decide" / "hooks"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_hook_{name}", HOOKS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _transcript(path: pathlib.Path, sid: str, n: int) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps({
        "type": "assistant", "sessionId": sid, "timestamp": f"2026-08-01T10:{i % 60:02d}:00Z",
        "message": {"id": f"{sid}-{i}", "model": "claude-opus-5", "content": [],
                    "usage": {"input_tokens": 1, "cache_read_input_tokens": 1_000 * i,
                              "output_tokens": 10}}}) for i in range(n)))
    return path


@pytest.fixture
def two_sessions(tmp_path):
    proj = tmp_path / "projects" / live.slug_for("/w/repo")
    mine = _transcript(proj / "AAAA.jsonl", "AAAA", 25)
    other = _transcript(proj / "BBBB.jsonl", "BBBB", 300)
    os.utime(mine, (1, 1))                      # the other session wrote last
    return tmp_path / "projects", mine, other


class TestTheNamedTranscriptWins:
    def test_over_a_newer_one_in_the_same_project(self, two_sessions):
        root, mine, _ = two_sessions
        assert live.current_session("/w/repo", root).id == "BBBB"      # the old guess
        s = live.current_session("/w/repo", root, transcript=str(mine))
        assert (s.id, s.n_turns) == ("AAAA", 25)

    def test_from_a_subdirectory_the_guess_cannot_find(self, two_sessions):
        root, mine, _ = two_sessions
        assert live.current_session("/w/repo/src", root) is None
        assert live.current_session("/w/repo/src", root, transcript=mine).id == "AAAA"

    @pytest.mark.parametrize("bad", ["", "/no/such/file.jsonl", "/etc"])
    def test_a_missing_or_odd_path_falls_back_to_the_guess(self, two_sessions, bad):
        root, _, _ = two_sessions
        assert live.current_session("/w/repo", root, transcript=bad).id == "BBBB"


class TestTheHooksPassItThrough:
    def _capture(self, monkeypatch):
        seen = {}

        def fake(cwd=None, root=None, *, transcript=None):
            seen["transcript"] = transcript
            return None

        monkeypatch.setattr("adder.measure.session.live.current_session", fake)
        return seen

    def test_the_cost_advisor(self, monkeypatch, isolated_home, tmp_path):
        monkeypatch.setenv("ADDER_STATE", str(tmp_path / "advisor.json"))
        seen = self._capture(monkeypatch)
        mod = _load("session_cost_advisor")
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
            {"cwd": "/w/repo", "transcript_path": "/t/AAAA.jsonl"})))
        assert mod.main() == 0
        assert seen["transcript"] == "/t/AAAA.jsonl"

    def test_the_read_guard(self, monkeypatch, isolated_home, tmp_path):
        monkeypatch.setenv("ADDER_GUARD_STATE", str(tmp_path / "guard.json"))
        monkeypatch.setenv("ADDER_SIZE_MODEL", str(tmp_path / "sizes.json"))
        seen = self._capture(monkeypatch)
        mod = _load("pretooluse_read_guard")
        monkeypatch.setattr(mod, "WATCHED", mod.WATCHED)
        big = tmp_path / "big.py"
        big.write_text("x = 1\n" * 40_000)
        from adder.decide import guard
        monkeypatch.setattr(guard, "needs_pricing", lambda *a, **k: True)
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({
            "session_id": "AAAA", "cwd": "/w/repo", "transcript_path": "/t/AAAA.jsonl",
            "tool_name": "Read", "tool_input": {"file_path": str(big)}})))
        assert mod.main() == 0
        assert seen["transcript"] == "/t/AAAA.jsonl"


class TestAPayloadThatIsNotAnObject:
    @pytest.mark.parametrize("raw", ["[]", "null", '"s"', '{"tool_name": ["Read"]}',
                                     '{"tool_name": "Read", "tool_input": []}'])
    def test_the_read_guard_stays_silent(self, raw, monkeypatch, capsys, isolated_home,
                                         tmp_path):
        monkeypatch.setenv("ADDER_GUARD_STATE", str(tmp_path / "guard.json"))
        mod = _load("pretooluse_read_guard")
        monkeypatch.setattr("sys.stdin", io.StringIO(raw))
        assert mod.main() == 0
        assert capsys.readouterr().out == ""

    @pytest.mark.parametrize("raw", ["[]", '"s"'])
    def test_the_cost_advisor_stays_silent(self, raw, monkeypatch, capsys, tmp_path):
        monkeypatch.setenv("ADDER_STATE", str(tmp_path / "advisor.json"))
        mod = _load("session_cost_advisor")
        monkeypatch.setattr("sys.stdin", io.StringIO(raw))
        assert mod.main() == 0


def test_compaction_clears_a_subagents_own_memory(monkeypatch, isolated_home, tmp_path):
    """Keyed by `session_id` alone, a subagent's compaction cleared its parent
    and left the subagent refusing a re-read of a file it no longer held."""
    monkeypatch.setenv("ADDER_GUARD_STATE", str(tmp_path / "guard.json"))
    monkeypatch.setenv("ADDER_SIZE_MODEL", str(tmp_path / "sizes.json"))
    from adder.decide import guard

    cleared = []
    monkeypatch.setattr(guard, "load_state", lambda key: guard.GuardState())
    monkeypatch.setattr(guard, "save_state", lambda key, state: cleared.append(key))
    mod = _load("precompact_learn")
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"session_id": "BBBB", "agent_id": "A1"})))
    mod.main()
    assert cleared[:1] == ["BBBB:A1"]
