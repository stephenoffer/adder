"""Reading the transcripts Codex CLI, Gemini CLI and OpenCode write to disk.

The number this file protects is the Codex one. Codex writes each
`token_count` event at least twice with identical totals, so a reader that
sums `last_token_usage` per event reports close to double the real bill -- on
the transcripts this reader was written against, 1.96x to 1.99x. The check
that matters is that the turns add back up to Codex's own cumulative total.

Every fixture is synthetic and follows the shape of the real files.
"""

from __future__ import annotations

import json
from pathlib import Path

from adder.core import harness, ingest, native, settings
from adder.core.filters import root_of
from adder.core.trace import iter_file, load_sessions, transcripts


def _codex_usage(inp, cached, out, reasoning=0):
    return {"input_tokens": inp, "cached_input_tokens": cached,
            "output_tokens": out, "reasoning_output_tokens": reasoning,
            "total_tokens": inp + out}


def _codex_file(path: Path, calls: list[tuple[int, int, int]], *,
                model="gpt-5.3-codex", repeat=2, session="019c-test") -> Path:
    """A rollout file with `calls` as (input, cached, output), each event repeated."""
    rows = [
        {"timestamp": "2026-02-13T22:37:01.000Z", "type": "session_meta",
         "payload": {"id": session, "cwd": "/Users/jo/app", "model_provider": "openai"}},
        {"timestamp": "2026-02-13T22:37:01.100Z", "type": "event_msg",
         "payload": {"type": "token_count", "info": None}},
    ]
    total = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0,
             "reasoning_output_tokens": 0, "total_tokens": 0}
    for i, (inp, cached, out) in enumerate(calls):
        rows.append({"timestamp": f"2026-02-13T22:38:{i:02d}.000Z", "type": "turn_context",
                     "payload": {"model": model, "cwd": "/Users/jo/app",
                                 "collaboration_mode": {"settings": {
                                     "reasoning_effort": "medium"}}}})
        rows.append({"timestamp": f"2026-02-13T22:38:{i:02d}.500Z", "type": "response_item",
                     "payload": {"type": "function_call", "name": "exec_command",
                                 "arguments": "{}", "call_id": f"c{i}"}})
        last = _codex_usage(inp, cached, out)
        for k in total:
            total[k] += last[k]
        for _ in range(repeat):
            rows.append({"timestamp": f"2026-02-13T22:38:{i:02d}.900Z", "type": "event_msg",
                         "payload": {"type": "token_count",
                                     "info": {"total_token_usage": dict(total),
                                              "last_token_usage": last,
                                              "model_context_window": 258400}}})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


class TestCodex:
    def test_a_repeated_token_count_is_one_call(self, tmp_path):
        p = _codex_file(tmp_path / "rollout-a.jsonl",
                        [(50_000, 48_000, 300), (51_000, 50_000, 200)])
        turns = list(iter_file(p))
        assert len(turns) == 2
        billed = sum(t.context + t.out for t in turns)
        assert billed == 50_300 + 51_200      # Codex's own total_tokens

    def test_the_cached_prefix_is_subtracted_from_input(self, tmp_path):
        p = _codex_file(tmp_path / "rollout-a.jsonl", [(50_000, 48_000, 300)])
        (t,) = list(iter_file(p))
        assert (t.uncached_in, t.cache_read, t.cache_write) == (2_000, 48_000, 0)

    def test_model_session_project_and_tools_come_from_other_records(self, tmp_path):
        p = _codex_file(tmp_path / "rollout-a.jsonl", [(10_000, 0, 100)])
        (t,) = list(iter_file(p))
        assert t.model == "gpt-5.3-codex"
        assert t.session == "019c-test"
        assert t.project == "-Users-jo-app"
        assert t.tools == ("exec_command",)
        assert t.effort == "medium"

    def test_a_resumed_rollout_does_not_count_twice(self, tmp_path):
        calls = [(20_000, 0, 100), (21_000, 19_000, 100)]
        _codex_file(tmp_path / "a" / "rollout-1.jsonl", calls)
        _codex_file(tmp_path / "a" / "rollout-2.jsonl", calls)
        sessions = load_sessions(tmp_path / "a", use_cache=False)
        assert sum(len(s.turns) for s in sessions.values()) == 2

    def test_a_turn_with_no_model_yet_is_not_priced_as_something(self, tmp_path):
        p = tmp_path / "rollout-a.jsonl"
        p.write_text(json.dumps({"type": "event_msg", "payload": {
            "type": "token_count", "info": {
                "total_token_usage": _codex_usage(10, 0, 5),
                "last_token_usage": _codex_usage(10, 0, 5)}}}) + "\n")
        assert list(iter_file(p)) == []


def _gemini_file(path: Path) -> Path:
    doc = {
        "sessionId": "g-1", "projectHash": "abc123",
        "startTime": "2026-03-01T10:00:00Z", "lastUpdated": "2026-03-01T10:05:00Z",
        "messages": [
            {"id": "u1", "type": "user", "timestamp": "2026-03-01T10:00:00Z",
             "content": "hi"},
            {"id": "m1", "type": "gemini", "timestamp": "2026-03-01T10:00:05Z",
             "content": "ok", "model": "gemini-2.5-pro",
             "tokens": {"input": 30_000, "output": 400, "cached": 25_000,
                        "thoughts": 600, "tool": 0, "total": 31_000},
             "toolCalls": [{"id": "t1", "name": "read_file"}]},
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2))
    return path


class TestGeminiCli:
    def test_cached_is_inside_input_and_thoughts_bill_as_output(self, tmp_path):
        p = _gemini_file(tmp_path / "abc123" / "chats" / "session-1.json")
        (t,) = list(iter_file(p))
        assert (t.uncached_in, t.cache_read) == (5_000, 25_000)
        assert (t.out, t.thinking) == (1_000, 600)
        assert t.model == "gemini-2.5-pro"
        assert t.session == "g-1" and t.tools == ("read_file",)

    def test_user_messages_are_not_turns(self, tmp_path):
        p = _gemini_file(tmp_path / "abc123" / "chats" / "session-1.json")
        assert len(list(iter_file(p))) == 1


def _opencode_file(path: Path, **over) -> Path:
    msg = {
        "id": "msg_01", "sessionID": "ses_01", "role": "assistant",
        "modelID": "claude-sonnet-4-5", "providerID": "anthropic",
        "time": {"created": 1767225600000},
        "path": {"cwd": "/Users/jo/app", "root": "/Users/jo/app"},
        "tokens": {"input": 1_200, "output": 300, "reasoning": 50,
                   "cache": {"read": 40_000, "write": 2_000}},
        **over,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(msg, indent=2))
    return path


class TestOpenCode:
    def test_input_is_already_net_of_the_cache(self, tmp_path):
        p = _opencode_file(tmp_path / "ses_01" / "msg_01.json")
        (t,) = list(iter_file(p))
        assert (t.uncached_in, t.cache_read, t.cache_write) == (1_200, 40_000, 2_000)
        # Reasoning is kept on the side, never added: an under-count, not a double.
        assert (t.out, t.thinking) == (300, 50)
        assert t.session == "ses_01" and t.project == "-Users-jo-app"
        assert t.ts is not None and t.ts.startswith("2026-01-01")

    def test_a_user_message_is_not_a_turn(self, tmp_path):
        p = _opencode_file(tmp_path / "ses_01" / "msg_02.json", role="user")
        assert list(iter_file(p)) == []

    def test_a_session_directory_groups_into_one_session(self, tmp_path):
        _opencode_file(tmp_path / "ses_01" / "msg_01.json")
        _opencode_file(tmp_path / "ses_01" / "msg_02.json", id="msg_02")
        sessions = load_sessions(tmp_path, use_cache=False)
        assert list(sessions) == ["ses_01"]
        assert len(sessions["ses_01"].turns) == 2


class TestDetection:
    def test_claude_code_and_api_logs_are_not_claimed(self, tmp_path):
        p = tmp_path / "x.jsonl"
        p.write_text(json.dumps({"type": "assistant", "message": {}}) + "\n")
        assert native.detect(p) is None
        assert native.iter_turns(p) is None

    def test_doctor_classifies_native_files_by_agent(self, tmp_path):
        paths = [_codex_file(tmp_path / "rollout-a.jsonl", [(10, 0, 5)]),
                 _gemini_file(tmp_path / "h" / "chats" / "session-1.json")]
        tally, silent = ingest.classify_files(paths)
        assert tally == {"codex": 1, "gemini-cli": 1} and silent == []


class TestFindingTheFiles:
    def test_an_agent_name_is_its_transcript_directory(self, isolated_home):
        assert native.root_for("codex") == isolated_home / ".codex" / "sessions"
        assert native.root_for("Gemini") == isolated_home / ".gemini" / "tmp"
        assert native.root_for("nope") is None

    def test_root_argument_takes_an_agent_name(self, isolated_home):
        import argparse

        got = root_of(argparse.Namespace(root="codex"))
        assert got == isolated_home / ".codex" / "sessions"

    def test_a_real_directory_beats_an_agent_name(self, isolated_home):
        import argparse

        (isolated_home.parent / "codex").mkdir()
        assert root_of(argparse.Namespace(root="codex")) == Path("codex")

    def test_transcripts_resolves_a_name_too(self, isolated_home):
        p = _codex_file(isolated_home / ".codex" / "sessions" / "2026" / "r.jsonl",
                        [(10, 0, 5)])
        assert transcripts("codex") == [p]

    def test_agent_of_root_reads_path_components(self):
        assert native.agent_of_root(Path("/mnt/b/jo/.codex/sessions/2026")) == "codex"
        assert native.agent_of_root(Path("/tmp/logs")) is None


class TestDefaults:
    def test_claude_stays_the_default_when_it_has_sessions(self, isolated_home):
        claude = isolated_home / ".claude" / "projects" / "-x"
        claude.mkdir(parents=True)
        (claude / "s.jsonl").write_text("{}\n")
        _codex_file(isolated_home / ".codex" / "sessions" / "r.jsonl", [(10, 0, 5)])
        r = settings.resolve(env={})
        assert Path(r["root"].value) == isolated_home / ".claude" / "projects"
        assert r["harness"].value == "claude-code"

    def test_a_codex_only_machine_reads_codex(self, isolated_home):
        _codex_file(isolated_home / ".codex" / "sessions" / "r.jsonl", [(10, 0, 5)])
        r = settings.resolve(env={})
        assert Path(r["root"].value) == isolated_home / ".codex" / "sessions"
        assert r["harness"].value == "codex"
        assert "derived" in r["harness"].source

    def test_an_empty_machine_keeps_the_claude_default(self, isolated_home):
        r = settings.resolve(env={})
        assert Path(r["root"].value) == isolated_home / ".claude" / "projects"

    def test_an_explicit_harness_wins_over_the_root(self, isolated_home):
        r = settings.resolve(env={"ADDER_ROOT": "~/.codex/sessions",
                                  "ADDER_HARNESS": "any"})
        assert r["harness"].value == "any"

    def test_opencode_is_a_harness_that_pins_nothing(self):
        h = harness.get("opencode")
        assert h.name == "opencode" and not h.pins_main_session


def _claude_file(path: Path, model: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "type": "assistant", "timestamp": "2026-08-01T09:00:00Z",
        "message": {"id": "m1", "model": model, "role": "assistant",
                    "usage": {"input_tokens": 10, "output_tokens": 5}}}) + "\n")
    return path


class TestTheModelIsReadNotRemembered:
    def test_the_model_is_the_newest_sessions(self, isolated_home):
        import os

        old = _claude_file(isolated_home / ".claude" / "projects" / "-a" / "1.jsonl",
                           "claude-sonnet-4-5")
        new = _claude_file(isolated_home / ".claude" / "projects" / "-a" / "2.jsonl",
                           "claude-opus-5")
        os.utime(old, (1, 1))
        os.utime(new, (2, 2))
        r = settings.resolve(env={}, derive_model=True)
        assert r["model"].value == "claude-opus-5"
        assert r["model"].source == "your newest session"
        assert settings.session_model() == "claude-opus-5"

    def test_a_codex_machine_is_priced_as_codex(self, isolated_home):
        _codex_file(isolated_home / ".codex" / "sessions" / "r.jsonl", [(10, 0, 5)])
        assert settings.session_model() == "gpt-5.3-codex"

    def test_a_configured_model_wins(self, isolated_home):
        _codex_file(isolated_home / ".codex" / "sessions" / "r.jsonl", [(10, 0, 5)])
        got = settings.get("model", env={"ADDER_MODEL": "gpt-5"})
        assert got == "gpt-5"

    def test_no_sessions_keeps_the_shipped_default(self, isolated_home):
        assert settings.session_model() == settings.BY_NAME["model"].default

    def test_subagent_files_do_not_speak_for_the_session(self, isolated_home):
        import os

        main = _claude_file(isolated_home / ".claude" / "projects" / "-a" / "s.jsonl",
                            "claude-opus-5")
        sub = _claude_file(isolated_home / ".claude" / "projects" / "-a" / "s"
                           / "subagents" / "agent-1.jsonl", "claude-haiku-4-5")
        os.utime(main, (1, 1))
        os.utime(sub, (2, 2))
        assert settings.session_model() == "claude-opus-5"


class TestTheCommandLineRootMovesTheHarness:
    def test_naming_an_agent_on_the_command_line_sets_the_harness(self, isolated_home):
        import argparse

        root_of(argparse.Namespace(root="codex"))
        r = settings.resolve(env={})
        assert r["root"].source == "command line"
        assert r["harness"].value == "codex"

    def test_an_explicit_harness_still_wins(self, isolated_home):
        import argparse

        root_of(argparse.Namespace(root="codex"))
        assert settings.get("harness", env={"ADDER_HARNESS": "any"}) == "any"


class TestTheAgentRegistry:
    def test_every_agent_is_fully_described(self):
        for a in native.AGENTS:
            assert a.name and a.harness and a.root
            assert native.get(a.name) is a
            assert all(native.get(k) is a for k in a.aliases)

    def test_every_agent_name_is_a_harness(self):
        for a in native.AGENTS:
            assert harness.get(a.name).name == a.harness
            for k in a.aliases:
                assert harness.get(k).name == a.harness

    def test_claude_code_is_listed_but_left_to_trace(self, tmp_path):
        p = tmp_path / "x.jsonl"
        p.write_text(json.dumps({"type": "assistant", "message": {}}) + "\n")
        assert native.owner(p) is None
        assert native.iter_turns(p, agent="claude") is None

    def test_a_subclass_is_all_a_new_agent_needs(self, tmp_path):
        from adder.core.ingest import Usage

        class Toy(native.Agent):
            name = harness = "toy"
            root = (".toy",)

            def usages(self, path):
                for i in (1, 1, 2):                    # a repeated call
                    yield Usage(uncached_in=10, cache_read=0, cache_write=0,
                                out=5, model="gpt-5", msg_id=f"toy:{i}"), "s", "p"

        turns = list(Toy().turns(tmp_path / "f"))
        assert [t.msg_id for t in turns] == ["toy:1", "toy:2"]


class TestLookupsStayCheap:
    """`plan` asks for the ladder per recorded turn. When every lookup resolved
    the `root` default, which walks each agent's directory, a 67-session replay
    did not finish in fifteen minutes."""

    def test_one_setting_does_not_compute_the_root_default(self, isolated_home, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("asked for `ladder`, walked the transcript directories")
        monkeypatch.setattr(native, "discover", boom)
        assert settings.get("ladder", env={}) == ""

    def test_a_derived_setting_still_reads_what_it_derives_from(self, isolated_home):
        _codex_file(isolated_home / ".codex" / "sessions" / "r.jsonl", [(10, 0, 5)])
        assert settings.get("harness", env={}) == "codex"

    def test_has_sessions_remembers_yes_and_never_no(self, tmp_path):
        root = tmp_path / "projects"
        root.mkdir()
        assert not native.has_sessions(root)
        (root / "s.jsonl").write_text("{}\n")
        assert native.has_sessions(root)           # a cached "no" would miss this
        (root / "s.jsonl").unlink()
        assert native.has_sessions(root)           # remembered for the process
        native.forget_latest_model()
        assert not native.has_sessions(root)
