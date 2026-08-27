"""The duplicate-read rule when the harness reads with `cat` instead of `Read`.

`bypassPermissions` is not a corner case -- it is how agent harnesses run
unattended, and its own guidance routes file access to the shell because the
dedicated tools prompt. On exactly that configuration the guard's cheapest
saving used to see nothing: `state.reads` was written only from `Read`'s
`file_path`, so a session that read every file with `cat` and `sed -n` reported
zero re-reads. Zero and "this cannot be observed here" printed the same.

What is asserted here is the identity bookkeeping and nothing else. The
refusal's *shape* -- once per target, always with a way through, off by default
-- is `test_guard_enforce.py`'s subject and is not restated.

The asymmetry to hold on to: a slice may never be *recorded* as a whole read,
because that would make the guard refuse the call that would have got the rest;
but a slice of a file already held whole may be refused, because those lines
are demonstrably already there.
"""

from __future__ import annotations

import pytest

from adder.core.shapes import SizeModel
from adder.decide.guard import (
    GuardState,
    Settings,
    Verdict,
    decide,
    needs_pricing,
    observe,
)

OPUS = "claude-opus-5"


@pytest.fixture
def sizes():
    # Small on purpose. The size rule firing would mask the rule under test,
    # and the duplicate rule is the one that must work on a `cat` nobody would
    # otherwise have interrupted about.
    return SizeModel(shapes={"cat": (120, 400, 40)}, heads={"cat": (120, 400, 40)},
                     tools={}, built=1.0, calls=80)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "pyproject.toml").write_text("x = 1\n" * 2_000)
    (tmp_path / "other.py").write_text("y = 2\n" * 2_000)
    return tmp_path


@pytest.fixture
def cfg(tmp_path):
    return Settings(enforce="certain", state_path=tmp_path / "state.json")


def run(command, state, cfg, sizes, repo, turns: int = 300) -> Verdict:
    inp = {"command": command}
    v = decide("Bash", inp, model=OPUS, remaining_turns=turns, cfg=cfg,
               sizes=sizes, state=state, cwd=str(repo))
    observe("Bash", inp, state, v, sizes=sizes, cwd=str(repo))
    return v


class TestWhatTheShellPutInTheContext:
    def test_a_cat_is_remembered_like_a_read(self, repo, cfg, sizes):
        state = GuardState()
        run("cat pyproject.toml", state, cfg, sizes, repo)
        assert str(repo / "pyproject.toml") in state.reads

    def test_a_relative_path_is_keyed_where_the_session_was(self, repo, cfg, sizes):
        state = GuardState()
        run("cat pyproject.toml", state, cfg, sizes, repo)
        # Not `pyproject.toml`: a bare name is a different file in every
        # directory, and `Read` records an absolute one.
        assert list(state.reads) == [str(repo / "pyproject.toml")]

    def test_a_slice_is_not_a_whole_read(self, repo, cfg, sizes):
        state = GuardState()
        run("sed -n '1,50p' pyproject.toml", state, cfg, sizes, repo)
        assert state.reads == {}

    def test_a_grep_is_not_a_whole_read(self, repo, cfg, sizes):
        state = GuardState()
        run("grep -n 'x = 1' pyproject.toml", state, cfg, sizes, repo)
        assert state.reads == {}

    def test_a_write_is_not_a_read(self, repo, cfg, sizes):
        state = GuardState()
        run("cat pyproject.toml > /dev/null", state, cfg, sizes, repo)
        assert state.reads == {}

    def test_a_refused_command_admitted_nothing(self, repo, cfg, sizes):
        # The premise: it was denied, so it never ran, so there is nothing to
        # remember -- and remembering it would make the guard's own refusal the
        # evidence for the next one.
        state = GuardState()
        observe("Bash", {"command": "cat pyproject.toml"}, state,
                Verdict(True, "refused", deny=True), sizes=sizes, cwd=str(repo))
        assert state.reads == {}


class TestTheSecondShellRead:
    def test_cat_twice_is_refused(self, repo, cfg, sizes):
        state = GuardState()
        assert not run("cat pyproject.toml", state, cfg, sizes, repo).fire
        v = run("cat pyproject.toml", state, cfg, sizes, repo)
        assert v.fire and v.deny and v.certain
        assert v.kind == "duplicate"

    def test_the_refusal_names_the_file_and_a_way_through(self, repo, cfg, sizes):
        state = GuardState()
        run("cat pyproject.toml", state, cfg, sizes, repo)
        v = run("cat pyproject.toml", state, cfg, sizes, repo)
        assert "pyproject.toml" in v.message
        assert Verdict.ESCAPE in v.message
        # The caller is working in a shell. Advice about `limit:` is advice
        # about a tool it is not using.
        assert "limit" not in v.message

    def test_a_slice_of_a_file_already_held_whole_is_refused(self, repo, cfg, sizes):
        state = GuardState()
        run("cat pyproject.toml", state, cfg, sizes, repo)
        v = run("sed -n '40,80p' pyproject.toml", state, cfg, sizes, repo)
        assert v.fire and v.deny

    def test_a_read_after_a_cat_is_caught_too(self, repo, cfg, sizes):
        state = GuardState()
        run("cat pyproject.toml", state, cfg, sizes, repo)
        inp = {"file_path": str(repo / "pyproject.toml")}
        v = decide("Read", inp, model=OPUS, remaining_turns=300, cfg=cfg,
                   sizes=sizes, state=state, cwd=str(repo))
        assert v.fire and v.kind == "duplicate"

    def test_a_cat_after_a_read_is_caught_too(self, repo, cfg, sizes):
        state = GuardState()
        observe("Read", {"file_path": str(repo / "pyproject.toml")}, state,
                Verdict(False, "first read"), cwd=str(repo))
        v = run("cat pyproject.toml", state, cfg, sizes, repo)
        assert v.fire and v.deny

    def test_one_file_is_refused_once_across_both_tools(self, repo, cfg, sizes):
        # `Read:{path}` is one entry in the refuse-once ledger, not two. A
        # guard that refuses `cat f` and then refuses `Read f` has refused the
        # same file twice, which is what the ledger exists to prevent.
        state = GuardState()
        run("cat pyproject.toml", state, cfg, sizes, repo)
        assert run("cat pyproject.toml", state, cfg, sizes, repo).deny
        inp = {"file_path": str(repo / "pyproject.toml")}
        v = decide("Read", inp, model=OPUS, remaining_turns=300, cfg=cfg,
                   sizes=sizes, state=state, cwd=str(repo))
        assert not v.deny

    def test_an_edited_file_may_be_re_read(self, repo, cfg, sizes):
        state = GuardState()
        run("cat pyproject.toml", state, cfg, sizes, repo)
        (repo / "pyproject.toml").write_text("z = 3\n" * 2_000)
        assert not run("cat pyproject.toml", state, cfg, sizes, repo).fire

    def test_a_command_that_also_reads_something_new_is_left_alone(self, repo, cfg, sizes):
        # Refusing it would cost the half that brings something to save the
        # half that does not.
        state = GuardState()
        run("cat pyproject.toml", state, cfg, sizes, repo)
        assert not run("cat pyproject.toml other.py", state, cfg, sizes, repo).fire

    def test_compaction_drops_the_premise(self, repo, cfg, sizes):
        state = GuardState()
        run("cat pyproject.toml", state, cfg, sizes, repo)
        state.forget_context()
        assert not run("cat pyproject.toml", state, cfg, sizes, repo).fire

    def test_saving_is_booked_as_prevented_not_as_advice(self, repo, cfg, sizes):
        state = GuardState()
        run("cat pyproject.toml", state, cfg, sizes, repo)
        run("cat pyproject.toml", state, cfg, sizes, repo)
        assert state.prevented > 0


class TestOffByDefault:
    def test_nothing_is_refused_unless_enforcing(self, repo, sizes, tmp_path):
        cfg = Settings(state_path=tmp_path / "state.json")
        state = GuardState()
        run("cat pyproject.toml", state, cfg, sizes, repo)
        v = run("cat pyproject.toml", state, cfg, sizes, repo)
        assert not v.deny

    def test_a_large_duplicate_is_still_advised_about(self, repo, sizes, tmp_path):
        cfg = Settings(state_path=tmp_path / "state.json", min_tokens=100)
        state = GuardState()
        run("cat pyproject.toml", state, cfg, sizes, repo)
        v = run("cat pyproject.toml", state, cfg, sizes, repo)
        assert v.fire and not v.deny and v.kind == "duplicate"


class TestTheCheapPath:
    def test_a_shell_re_read_is_worth_pricing(self, repo, sizes):
        state = GuardState()
        observe("Bash", {"command": "cat pyproject.toml"}, state,
                Verdict(False, "first"), sizes=sizes, cwd=str(repo))
        assert needs_pricing("Bash", {"command": "cat pyproject.toml"},
                             sizes=sizes, state=state, cwd=str(repo))

    def test_a_first_read_is_not(self, repo, sizes):
        # Nothing to match on yet, and the shape is small, so this call is not
        # worth parsing a transcript for.
        assert not needs_pricing("Bash", {"command": "cat pyproject.toml"},
                                 sizes=sizes, state=GuardState(), cwd=str(repo))

    def test_having_advised_about_the_shape_does_not_blind_it(self, repo, sizes):
        # `advised` stops the guard repeating itself about a command *shape*.
        # It is not a reason to stop noticing that this particular file is
        # already in the context.
        state = GuardState(advised=["cat"])
        observe("Bash", {"command": "cat pyproject.toml"}, state,
                Verdict(False, "first"), sizes=sizes, cwd=str(repo))
        assert needs_pricing("Bash", {"command": "cat pyproject.toml"},
                             sizes=sizes, state=state, cwd=str(repo))
