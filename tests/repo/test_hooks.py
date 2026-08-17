"""The hooks, which are the only part of the tool that can PREVENT spend.

Everything else in this repo measures after the fact. `pretooluse_read_guard`
runs while the decision is still reversible, which makes it the one component
whose failure is silent and expensive: a guard that is misconfigured still looks
installed.

The judgement it makes now lives in `adder.decide.guard` and is tested against
directly in `tests/decide/test_guard.py`. What is left here is the part that
can only be tested through the hook itself: that it reads stdin without dying,
that it never parses a transcript for a call it has no opinion on, that it
writes its state where it was told to and nowhere else, and that its install
snippet describes the tools it actually handles.

The hooks live under `.claude/` rather than in the package, so they are loaded
by path. `.claude/` is tracked and shipped, so it is testable and it is tested.
"""

from __future__ import annotations

import importlib.util
import io
import json
import pathlib

import pytest

HOOKS = pathlib.Path(__file__).resolve().parents[2] / ".claude" / "hooks"


def _load(name: str):
    """Import a hook by path, fresh, so module-level env reads re-run."""
    spec = importlib.util.spec_from_file_location(f"_hook_{name}", HOOKS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def guard(monkeypatch, tmp_path):
    """The hook, with its state file pointed away from the real home directory.

    Without the redirect this suite would write to `~/.claude/.adder-guard.json`
    on the machine running it, which CLAUDE.md forbids and which would also make
    the tests depend on whatever the developer's own sessions had recorded.
    """
    monkeypatch.setenv("ADDER_GUARD_STATE", str(tmp_path / "guard.json"))
    monkeypatch.setenv("ADDER_SIZE_MODEL", str(tmp_path / "sizes.json"))
    return _load("pretooluse_read_guard")


def _run(mod, payload, monkeypatch, capsys) -> dict | None:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    assert mod.main() == 0, "a hook must never signal failure to the harness"
    out = capsys.readouterr().out.strip()
    return json.loads(out) if out else None


class _Sess:
    def __init__(self, n_turns=200):
        self.n_turns = n_turns
        self.id = "sess"


class _Report:
    """Enough of `LiveReport` to price a call.

    `carry_turns` and not `projected_remaining`: the median answers "how much
    longer will this run" and the mean is the one a cost is linear in, which is
    the distinction `horizon.mean_remaining` exists to make.
    """

    def __init__(self, remaining, model="claude-opus-5", read_mult=0.10):
        self.projected_remaining = remaining
        self.expected_remaining = float(remaining)
        self.carry_turns = float(remaining)
        self.model = model
        self.context = 100_000
        # The realised re-read multiplier. The hook prices the carry term with
        # it, and a stub missing it is exactly the silent disabling `_swallow`
        # warns about.
        self.read_mult = read_mult


def _wire(monkeypatch, remaining, turns=200):
    monkeypatch.setattr("adder.measure.session.live.current_session",
                        lambda *a, **k: _Sess(turns))
    monkeypatch.setattr("adder.measure.session.live.analyse",
                        lambda s, **k: _Report(remaining))


class TestTheInstallContract:
    """A hook that names one set of tools in its docs and handles another is a
    hook that looks installed and does nothing for half its matcher."""

    def test_the_install_snippet_matches_the_guarded_set(self, guard):
        from adder.decide.guard import GUARDED

        doc = guard.__doc__ or ""
        for tool in GUARDED:
            assert tool in doc, f"{tool} is guarded but not in the install snippet"

    def test_the_hook_defers_to_the_library_for_the_tool_set(self, guard):
        """The list must not be duplicated here; two copies drift."""
        assert "GUARDED" in pathlib.Path(guard.__file__).read_text()

    def test_it_advises_rather_than_blocking_by_default(self):
        from adder.decide.guard import Settings

        assert Settings.resolve(env={}).block is False


class TestEarlyExits:
    """None of these may touch the transcript; they run on every tool call."""

    def test_a_tool_it_does_not_guard_is_ignored(self, guard, monkeypatch, capsys):
        assert _run(guard, {"tool_name": "Edit", "tool_input": {}},
                    monkeypatch, capsys) is None

    def test_malformed_stdin_is_survived(self, guard, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin", io.StringIO("{not json"))
        assert guard.main() == 0
        assert capsys.readouterr().out.strip() == ""

    def test_an_empty_payload_is_survived(self, guard, monkeypatch, capsys):
        assert _run(guard, {}, monkeypatch, capsys) is None

    def test_a_small_read_never_looks_up_the_session(self, guard, monkeypatch,
                                                     capsys, tmp_path):
        """The floor is an I/O guard: parsing a transcript per Read is not free."""
        def explode(*a, **k):                      # pragma: no cover - must not run
            raise AssertionError("looked up the session for a trivial read")

        monkeypatch.setattr("adder.measure.session.live.current_session", explode)
        f = tmp_path / "small.py"
        f.write_text("x" * 100)
        assert _run(guard, {"tool_name": "Read", "tool_input": {"file_path": str(f)}},
                    monkeypatch, capsys) is None

    def test_a_bounded_command_never_looks_up_the_session(self, guard, monkeypatch,
                                                          capsys):
        """22,761 of this corpus's tool calls are Bash. Almost all are bounded,
        and none of them may cost a transcript parse."""
        def explode(*a, **k):                      # pragma: no cover - must not run
            raise AssertionError("looked up the session for a bounded command")

        monkeypatch.setattr("adder.measure.session.live.current_session", explode)
        assert _run(guard, {"tool_name": "Bash",
                            "tool_input": {"command": "wc -l src/*.py"}},
                    monkeypatch, capsys) is None

    def test_a_young_session_is_left_alone(self, guard, monkeypatch, capsys, tmp_path):
        _wire(monkeypatch, remaining=400, turns=2)
        f = tmp_path / "f.py"
        f.write_text("x" * 400_000)
        assert _run(guard, {"tool_name": "Read", "tool_input": {"file_path": str(f)}},
                    monkeypatch, capsys) is None


class TestThroughTheHook:
    """The integration path: stdin to stdout, with the session stubbed."""

    def _read(self, guard, monkeypatch, capsys, tmp_path, chars, remaining, turns=200):
        _wire(monkeypatch, remaining, turns)
        f = tmp_path / "f.py"
        f.write_text("x" * chars)
        return _run(guard, {"tool_name": "Read", "session_id": "s1",
                            "tool_input": {"file_path": str(f)}}, monkeypatch, capsys)

    def test_a_read_the_old_constant_waved_through_now_fires(
            self, guard, monkeypatch, capsys, tmp_path):
        """6,000 tokens is under the old 15,000 floor and $1+ to carry for 400 turns."""
        got = self._read(guard, monkeypatch, capsys, tmp_path,
                         chars=24_000, remaining=400)
        assert got is not None, "an expensive read must not be silent"
        assert "read guard" in got["hookSpecificOutput"]["additionalContext"]

    def test_the_same_read_late_in_a_short_session_stays_quiet(
            self, guard, monkeypatch, capsys, tmp_path):
        """Nothing to amortize over means nothing to warn about."""
        assert self._read(guard, monkeypatch, capsys, tmp_path,
                          chars=24_000, remaining=1) is None

    def test_the_message_names_the_cost_and_the_alternative(
            self, guard, monkeypatch, capsys, tmp_path):
        got = self._read(guard, monkeypatch, capsys, tmp_path,
                         chars=400_000, remaining=400)
        msg = got["hookSpecificOutput"]["additionalContext"]
        assert "$" in msg and "subagent" in msg and "delegate" in msg

    def test_it_advises_rather_than_blocks_by_default(
            self, guard, monkeypatch, capsys, tmp_path):
        got = self._read(guard, monkeypatch, capsys, tmp_path,
                         chars=400_000, remaining=400)
        assert "permissionDecision" not in got["hookSpecificOutput"]

    def test_blocking_asks_rather_than_denies(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setenv("ADDER_GUARD_STATE", str(tmp_path / "guard.json"))
        monkeypatch.setenv("ADDER_GUARD_BLOCK", "1")
        guard = _load("pretooluse_read_guard")
        got = self._read(guard, monkeypatch, capsys, tmp_path,
                         chars=400_000, remaining=400)
        out = got["hookSpecificOutput"]
        assert out["permissionDecision"] == "ask", "a guard must never deny silently"
        assert out["permissionDecisionReason"]

    def test_a_broken_session_lookup_never_breaks_the_turn(
            self, guard, monkeypatch, capsys, tmp_path):
        def explode(*a, **k):
            raise RuntimeError("transcript unreadable")

        monkeypatch.setattr("adder.measure.session.live.current_session", explode)
        f = tmp_path / "f.py"
        f.write_text("x" * 400_000)
        assert _run(guard, {"tool_name": "Read", "tool_input": {"file_path": str(f)}},
                    monkeypatch, capsys) is None


class TestItRemembersReads:
    """The duplicate-read saving needs memory between calls, and that memory is
    the only thing this hook writes."""

    def test_a_read_is_recorded_even_when_the_guard_stays_quiet(
            self, guard, monkeypatch, capsys, tmp_path):
        """The first read is small and silent; the *second* one is the saving,
        and it can only be caught if the first was remembered."""
        state_path = tmp_path / "guard.json"
        f = tmp_path / "small.py"
        f.write_text("x" * 100)
        _run(guard, {"tool_name": "Read", "session_id": "s1",
                     "tool_input": {"file_path": str(f)}}, monkeypatch, capsys)
        assert str(f) in json.loads(state_path.read_text())["s1"]["reads"]

    def test_the_second_read_of_an_unchanged_file_fires(
            self, guard, monkeypatch, capsys, tmp_path):
        _wire(monkeypatch, remaining=300)
        f = tmp_path / "big.py"
        f.write_text("x" * 40_000)
        payload = {"tool_name": "Read", "session_id": "s1",
                   "tool_input": {"file_path": str(f)}}
        _run(guard, payload, monkeypatch, capsys)
        again = _run(guard, payload, monkeypatch, capsys)
        assert again is not None
        assert "already in this context" in \
            again["hookSpecificOutput"]["additionalContext"]

    def test_it_writes_state_only_where_it_was_told_to(self, guard, tmp_path):
        from adder.decide.guard import Settings

        assert str(tmp_path) in str(Settings.resolve().state_path), \
            "must not default into the real home during a test"

    def test_one_session_cannot_see_another_sessions_reads(
            self, guard, monkeypatch, capsys, tmp_path):
        _wire(monkeypatch, remaining=300)
        f = tmp_path / "big.py"
        f.write_text("x" * 40_000)
        _run(guard, {"tool_name": "Read", "session_id": "s1",
                     "tool_input": {"file_path": str(f)}}, monkeypatch, capsys)
        other = _run(guard, {"tool_name": "Read", "session_id": "s2",
                             "tool_input": {"file_path": str(f)}}, monkeypatch, capsys)
        assert other is None or "already in this context" not in json.dumps(other)


class TestSessionCostAdvisor:
    def test_malformed_stdin_is_survived(self, monkeypatch, capsys):
        mod = _load("session_cost_advisor")
        monkeypatch.setattr("sys.stdin", io.StringIO("{not json"))
        assert mod.main() == 0

    def test_it_writes_state_only_where_it_was_told_to(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ADDER_STATE", str(tmp_path / "advisor.json"))
        mod = _load("session_cost_advisor")
        assert str(tmp_path) in str(mod.STATE), "must not default into the real home"


class TestItWatchesWrites:
    """A `Write` is never advised about and always remembered."""

    def test_a_write_is_recorded_without_advising(self, guard, monkeypatch, capsys,
                                                  tmp_path):
        state_path = tmp_path / "guard.json"
        f = tmp_path / "new.py"
        f.write_text("x" * 40_000)
        got = _run(guard, {"tool_name": "Write", "session_id": "s1",
                           "tool_input": {"file_path": str(f), "content": "x"}},
                   monkeypatch, capsys)
        assert got is None, "admitting a write costs nothing; there is nothing to say"
        assert str(f) in json.loads(state_path.read_text())["s1"]["wrote"]

    def test_reading_back_what_this_session_wrote_fires(self, guard, monkeypatch,
                                                        capsys, tmp_path):
        _wire(monkeypatch, remaining=300)
        f = tmp_path / "new.py"
        f.write_text("x" * 40_000)
        _run(guard, {"tool_name": "Write", "session_id": "s1",
                     "tool_input": {"file_path": str(f), "content": "x"}},
             monkeypatch, capsys)
        back = _run(guard, {"tool_name": "Read", "session_id": "s1",
                            "tool_input": {"file_path": str(f)}}, monkeypatch, capsys)
        assert back is not None
        assert "written by this session" in \
            back["hookSpecificOutput"]["additionalContext"]

    def test_a_write_never_looks_up_the_session(self, guard, monkeypatch, capsys,
                                                tmp_path):
        def explode(*a, **k):                      # pragma: no cover - must not run
            raise AssertionError("parsed a transcript for a Write")

        monkeypatch.setattr("adder.measure.session.live.current_session", explode)
        f = tmp_path / "new.py"
        f.write_text("x")
        assert _run(guard, {"tool_name": "Write", "session_id": "s1",
                            "tool_input": {"file_path": str(f)}},
                    monkeypatch, capsys) is None


class TestTheAgentsAndTheGuardAgree:
    """`.claude/agents/` and the guard both state a return budget, and they are
    two copies of one number. The guard prices a `Task` against it; the agent
    files instruct the subagent to meet it. If they drift, the tool advises one
    thing and the agent is told another."""

    AGENTS = pathlib.Path(__file__).resolve().parents[2] / ".claude" / "agents"

    def test_every_routing_tier_states_the_budget(self):
        from adder.decide.guard import BRIEF_TARGET_TOKENS

        want = f"{BRIEF_TARGET_TOKENS:,} tokens"
        for name in ("route-t0", "route-t1", "route-t2"):
            text = (self.AGENTS / f"{name}.md").read_text()
            assert want in text, f"{name} does not state the {want} return budget"

    def test_the_budget_is_a_ceiling_not_a_target_to_hit(self):
        for name in ("route-t0", "route-t1", "route-t2"):
            text = (self.AGENTS / f"{name}.md").read_text()
            assert "Return under" in text

    def test_every_tier_says_why_a_return_is_expensive(self):
        """A budget with no reason attached is a rule people round off."""
        for name in ("route-t0", "route-t1", "route-t2", "Explore"):
            text = (self.AGENTS / f"{name}.md").read_text().lower()
            assert "every remaining turn" in text or "every later turn" in text

    def test_the_read_only_tiers_cannot_write(self):
        """Tier 0 exists so that escalating away from it is always safe."""
        for name in ("route-t0", "Explore"):
            text = (self.AGENTS / f"{name}.md").read_text()
            assert "disallowedTools: Write, Edit, NotebookEdit" in text


class TestTheAdvisorChargesItself:
    """The prompt hook injects ~155 tokens that are then carried for the rest of
    the session. The read guard learned this the expensive way -- 903 fires,
    none of them charged for the sentence they injected."""

    def test_it_prices_its_own_message(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ADDER_STATE", str(tmp_path / "advisor.json"))
        mod = _load("session_cost_advisor")
        assert mod.ADVICE_TAKEN == 0.5

    def test_the_uptake_assumption_is_shared_with_the_guard(self, monkeypatch,
                                                            tmp_path):
        """Two hooks disagreeing about how often advice is taken would price the
        same sentence two ways."""
        from adder.decide.guard import Settings

        monkeypatch.setenv("ADDER_STATE", str(tmp_path / "advisor.json"))
        mod = _load("session_cost_advisor")
        assert Settings.resolve(env={}).advice_taken == mod.ADVICE_TAKEN

    def test_the_env_name_is_the_guards(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ADDER_STATE", str(tmp_path / "advisor.json"))
        monkeypatch.setenv("ADDER_GUARD_ADVICE_TAKEN", "0.25")
        assert _load("session_cost_advisor").ADVICE_TAKEN == 0.25

    def test_it_charges_before_it_speaks(self, monkeypatch, tmp_path):
        """The gate has to be in the path that emits, not merely computed."""
        monkeypatch.setenv("ADDER_STATE", str(tmp_path / "advisor.json"))
        src = pathlib.Path(_load("session_cost_advisor").__file__).read_text()
        emit = src.index("additionalContext\": message")
        gate = src.index("worth * ADVICE_TAKEN <= overhead")
        assert gate < emit, "priced after emitting is not priced at all"


class TestTheLatencyBudget:
    """A PreToolUse hook runs on every tool call, so its own latency is part of
    what it costs. This is a regression test for a real one: `live.analyse`
    re-fitted the session-length distribution over every transcript on the
    machine on every guarded read, and the hook took 2,136ms. Latency is not
    dollars — but a two-second hook is one people uninstall, and an uninstalled
    guard saves nothing."""

    def test_a_call_it_has_no_opinion_on_never_imports_adder(self, guard,
                                                             monkeypatch, capsys):
        """Importing `adder` costs about 27ms on top of the interpreter's own
        32ms, and most calls are ones the guard says nothing about."""
        import builtins

        real = builtins.__import__
        loaded = []

        def watching(name, *a, **k):
            if name.startswith("adder"):
                loaded.append(name)
            return real(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", watching)
        _run(guard, {"tool_name": "TodoWrite", "tool_input": {}}, monkeypatch, capsys)
        assert not loaded, f"imported {loaded} for a tool it does not watch"

    def test_the_module_imports_no_heavy_stdlib(self, guard):
        """`pathlib` alone is ~10ms, and this file runs once per tool call."""
        src = pathlib.Path(guard.__file__).read_text()
        head = src.split("def ", 1)[0]
        assert "from pathlib import" not in head and "import pathlib" not in head

    def test_the_transcript_is_read_through_the_cache(self):
        """The defect was a default: the parse cache was off unless a caller
        asked, and the hook path had not."""
        import inspect

        from adder.core.trace import load_sessions

        assert inspect.signature(load_sessions).parameters["use_cache"].default is None

    def test_the_horizon_is_not_refitted_per_call(self):
        import inspect

        from adder.measure.session.horizon import load

        assert inspect.signature(load).parameters["use_cache"].default is True


class TestPreCompactLearner:
    """The model is refreshed when somebody remembers to run `--learn`.
    Compaction is the moment that fixes: it only happens in a long session, a
    stale model costs most there, and the session is already stopping."""

    @pytest.fixture
    def learner(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ADDER_SIZE_MODEL", str(tmp_path / "sizes.json"))
        return _load("precompact_learn")

    def test_it_emits_nothing(self, learner, monkeypatch, capsys):
        """It must not inject tokens into a context mid-compaction, and it must
        not depend on what a PreCompact hook is allowed to return."""
        assert _run(learner, {"trigger": "auto"}, monkeypatch, capsys) is None

    def test_it_refreshes_the_model(self, learner, monkeypatch, capsys, tmp_path):
        import adder.core.shapes as shapes

        called = []
        monkeypatch.setattr(shapes, "refresh", lambda *a, **k: called.append(1))
        _run(learner, {"trigger": "manual"}, monkeypatch, capsys)
        assert called

    def test_a_fresh_model_is_not_rescanned(self, learner, monkeypatch, capsys,
                                            tmp_path):
        """Compacting twice in an hour must not scan twice."""
        import time

        from adder.core.shapes import SizeModel

        SizeModel(shapes={"cat": (1, 2, 5)}, built=time.time(), calls=5).save(
            tmp_path / "sizes.json")

        def explode(*a, **k):                  # pragma: no cover - must not run
            raise AssertionError("rescanned a model that was still fresh")

        monkeypatch.setattr(SizeModel, "learn", classmethod(explode))
        assert _run(learner, {"trigger": "auto"}, monkeypatch, capsys) is None

    def test_malformed_stdin_is_survived(self, learner, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin", io.StringIO("{not json"))
        assert learner.main() == 0

    def test_a_broken_refresh_never_signals_failure(self, learner, monkeypatch,
                                                    capsys):
        import adder.core.shapes as shapes

        def explode(*a, **k):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(shapes, "refresh", explode)
        assert _run(learner, {}, monkeypatch, capsys) is None

    def test_it_keeps_pathlib_off_the_import_path(self, learner):
        """Same reason as the read guard: this runs as its own process."""
        src = pathlib.Path(learner.__file__).read_text()
        head = src.split("def ", 1)[0]
        assert "import pathlib" not in head and "from pathlib import" not in head


class TestItRemembersBeforeItCanPrice:
    """A read in a session's first five turns is still a read.

    The hook returns early when there are too few turns to project a horizon
    from. It used to return *before* recording the call, so every read in a
    session's opening was forgotten -- and those are precisely the reads a
    later turn re-reads, which is the one saving in this project that needs no
    model to justify it. The running per-shape total the aggregate rule is
    built on has the same dependency: it only works if the small early calls
    are counted.
    """

    def test_a_read_in_a_short_session_is_remembered(
            self, guard, monkeypatch, capsys, tmp_path):
        from adder.decide import guard as lib

        _wire(monkeypatch, remaining=400, turns=2)
        state_path = tmp_path / "guard.json"
        monkeypatch.setattr(lib.Settings, "resolve",
                            classmethod(lambda cls, **kw: lib.Settings(
                                state_path=state_path)))
        f = tmp_path / "f.py"
        f.write_text("x" * 400_000)
        payload = {"tool_name": "Read", "session_id": "s1",
                   "tool_input": {"file_path": str(f)}}
        assert _run(guard, payload, monkeypatch, capsys) is None
        assert str(f) in lib.load_state("s1", state_path).reads

    def test_a_bash_call_in_a_short_session_still_accumulates(
            self, guard, monkeypatch, capsys, tmp_path):
        from adder.decide import guard as lib

        _wire(monkeypatch, remaining=400, turns=2)
        state_path = tmp_path / "guard.json"
        monkeypatch.setattr(lib.Settings, "resolve",
                            classmethod(lambda cls, **kw: lib.Settings(
                                state_path=state_path)))
        payload = {"tool_name": "Bash", "session_id": "s2",
                   "tool_input": {"command": "cat /etc/hosts"}}
        _run(guard, payload, monkeypatch, capsys)
        assert lib.load_state("s2", state_path).shape_calls
