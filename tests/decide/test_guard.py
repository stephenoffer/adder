"""The one component that can prevent spend, and the one whose failure is silent.

A guard that stops guarding still lets every tool call succeed, so nothing
about the session looks wrong. That is why the decision was pulled out of the
hook and into `adder.decide.guard`, and why these tests assert on the *reason*
as well as the outcome: "the guard was quiet" is only acceptable when it was
quiet for a stated reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adder.core.shapes import SizeModel
from adder.decide.guard import (
    GUARDED,
    OBSERVED,
    GuardState,
    Settings,
    Verdict,
    decide,
    hook_path,
    install_snippet,
    installed_in,
    load_state,
    needs_pricing,
    observe,
    report,
    save_state,
)

OPUS = "claude-opus-5"


@pytest.fixture
def sizes():
    """A model that has seen `cat` return a lot and `echo` return nothing."""
    return SizeModel(
        shapes={"cat": (200, 40_000, 40), "echo": (5, 20, 40)},
        heads={"cat": (200, 40_000, 40), "echo": (5, 20, 40)},
        tools={}, built=1.0, calls=80,
    )


@pytest.fixture
def cfg():
    """Defaults, not the developer's environment."""
    return Settings()


class TestSettings:
    """Resolved when the guard runs. An import-time constant is untestable and
    un-overridable, which `render.color_enabled` already established here."""

    def test_defaults_are_the_documented_ones(self):
        s = Settings()
        assert (s.min_tokens, s.min_cost, s.block) == (2_000, 0.25, False)

    def test_blocking_is_off_unless_asked_for(self):
        assert Settings.resolve(env={}).block is False, \
            "a guard must not start blocking by default"

    def test_the_documented_adder_name_is_honoured(self):
        assert Settings.resolve(env={"ADDER_GUARD_MIN_TOKENS": "77"}).min_tokens == 77

    def test_the_pre_rename_router_name_still_works(self):
        """Silently ignoring an existing config would be its own outage."""
        assert Settings.resolve(env={"ROUTER_GUARD_HARD": "12345"}).hard_tokens == 12345

    def test_adder_wins_when_both_are_set(self):
        got = Settings.resolve(env={"ROUTER_GUARD_HARD": "1", "ADDER_GUARD_HARD": "2"})
        assert got.hard_tokens == 2

    def test_a_junk_value_falls_back_rather_than_raising(self):
        assert Settings.resolve(env={"ADDER_GUARD_MIN_TOKENS": "banana"}).min_tokens == 2_000

    def test_resolution_is_not_cached_across_calls(self):
        assert Settings.resolve(env={"ADDER_GUARD_MAX_FIRES": "3"}).max_fires == 3
        assert Settings.resolve(env={"ADDER_GUARD_MAX_FIRES": "9"}).max_fires == 9


class TestNeverSpeaksWithoutReason:
    def test_an_unguarded_tool_is_ignored(self, cfg, sizes):
        v = decide("Edit", {"file_path": "/etc/hosts"}, model=OPUS,
                   remaining_turns=400, sizes=sizes, cfg=cfg)
        assert not v.fire and "not a tool" in v.reason

    def test_a_command_bounded_by_shape_is_left_alone(self, cfg, sizes):
        """`wc -l` is small whatever the input, and carries no number to price."""
        v = decide("Bash", {"command": "wc -l a.ts b.ts"}, model=OPUS,
                   remaining_turns=400, sizes=sizes, cfg=cfg)
        assert not v.fire and "bounded by construction" in v.reason

    def test_a_numeric_bound_is_priced_rather_than_waved_through(self, cfg, sizes):
        """`head -50` is quiet because fifty lines is small, not because the
        word `head` appears. `sed -n '1,600p'` was waved through by the
        structural rule and returned 6,079 tokens."""
        v = decide("Bash", {"command": "cat huge.log | head -50"}, model=OPUS,
                   remaining_turns=400, sizes=sizes, cfg=cfg)
        assert not v.fire and "floor" in v.reason

    def test_a_large_numeric_bound_is_not_a_free_pass(self, cfg, sizes):
        v = decide("Bash", {"command": "sed -n '1,600p' big.py"}, model=OPUS,
                   remaining_turns=400, sizes=sizes, cfg=cfg)
        assert v.fire, "600 lines is roughly 6,000 tokens, not 'bounded'"

    def test_a_bound_caps_a_learned_estimate(self, cfg, sizes):
        """`cat` here has returned 40K tokens and this call inherits that
        through the program backoff — but fifty lines is fifty lines."""
        capped = sizes.predict_command("cat huge.log | head -50")
        assert capped.p90 < sizes.predict_command("cat huge.log").p90

    def test_a_small_prediction_is_below_the_floor(self, cfg, sizes):
        v = decide("Bash", {"command": "echo hello"}, model=OPUS,
                   remaining_turns=400, sizes=sizes, cfg=cfg)
        assert not v.fire and "floor" in v.reason

    def test_nothing_to_amortize_over_means_nothing_to_warn_about(self, cfg, sizes,
                                                                  tmp_path):
        """6,000 tokens is over the floor and under a quarter to carry through
        one more turn, so there is nothing here worth a sentence."""
        f = tmp_path / "f.py"
        f.write_text("x" * 24_000)
        v = decide("Read", {"file_path": str(f)}, model=OPUS, remaining_turns=1,
                   sizes=sizes, cfg=cfg)
        assert not v.fire and "floor" in v.reason

    def test_a_grep_that_returns_paths_is_not_content(self, cfg, sizes):
        v = decide("Grep", {"pattern": "x", "output_mode": "files_with_matches"},
                   model=OPUS, remaining_turns=400, sizes=sizes, cfg=cfg)
        assert not v.fire and "paths or counts" in v.reason

    def test_a_grep_the_caller_already_bounded(self, cfg, sizes):
        v = decide("Grep", {"pattern": "x", "output_mode": "content", "head_limit": 50},
                   model=OPUS, remaining_turns=400, sizes=sizes, cfg=cfg)
        assert not v.fire and "already bounded" in v.reason

    def test_every_pass_states_a_reason(self, cfg, sizes):
        """'Why was the guard silent on that 40K read' has to have an answer."""
        for tool, inp in [("Edit", {}), ("Bash", {"command": "wc -l f"}),
                          ("Bash", {"command": "echo hi"}),
                          ("Grep", {"pattern": "x"})]:
            assert decide(tool, inp, model=OPUS, remaining_turns=400,
                          sizes=sizes, cfg=cfg).reason


class TestTheDollarGate:
    """The decision is a cost, not a token count: one constant cannot be right
    at both ends of a session."""

    def test_an_expensive_read_fires(self, cfg, sizes, tmp_path):
        f = tmp_path / "f.py"
        f.write_text("x" * 400_000)
        v = decide("Read", {"file_path": str(f)}, model=OPUS, remaining_turns=400,
                   sizes=sizes, cfg=cfg)
        assert v.fire and "read guard" in v.message

    def test_the_same_read_late_in_a_short_session_stays_quiet(self, cfg, sizes, tmp_path):
        f = tmp_path / "f.py"
        f.write_text("x" * 24_000)
        assert not decide("Read", {"file_path": str(f)}, model=OPUS, remaining_turns=1,
                          sizes=sizes, cfg=cfg).fire

    def test_the_message_names_the_cost_and_the_alternative(self, cfg, sizes, tmp_path):
        f = tmp_path / "f.py"
        f.write_text("x" * 400_000)
        msg = decide("Read", {"file_path": str(f)}, model=OPUS, remaining_turns=400,
                     sizes=sizes, cfg=cfg).message
        assert "$" in msg and "subagent" in msg and "delegate" in msg

    def test_it_advises_rather_than_blocks_by_default(self, cfg, sizes, tmp_path):
        f = tmp_path / "f.py"
        f.write_text("x" * 400_000)
        v = decide("Read", {"file_path": str(f)}, model=OPUS, remaining_turns=400,
                   sizes=sizes, cfg=cfg)
        assert "permissionDecision" not in v.payload()["hookSpecificOutput"]

    def test_blocking_asks_rather_than_denies(self, sizes, tmp_path):
        f = tmp_path / "f.py"
        f.write_text("x" * 400_000)
        v = decide("Read", {"file_path": str(f)}, model=OPUS, remaining_turns=400,
                   sizes=sizes, cfg=Settings(block=True))
        out = v.payload()["hookSpecificOutput"]
        assert out["permissionDecision"] == "ask", "a guard must never deny silently"
        assert out["permissionDecisionReason"]

    def test_it_does_not_delegate_what_the_subagent_cannot_hold(self, cfg, sizes,
                                                                tmp_path):
        """`placement_cost` refuses when the read is larger than the subagent's
        context window. Firing anyway would advise an option that does not
        exist and quote a saving from it."""
        f = tmp_path / "f.py"
        f.write_text("x" * 4_000_000)
        v = decide("Read", {"file_path": str(f)}, model=OPUS, remaining_turns=400,
                   sizes=sizes, cfg=cfg)
        assert not v.fire and "cannot delegate" in v.reason

    def test_it_quotes_no_saving_it_cannot_price(self, cfg, sizes, tmp_path):
        f = tmp_path / "f.py"
        f.write_text("x" * 4_000_000)
        assert decide("Read", {"file_path": str(f)}, model=OPUS, remaining_turns=400,
                      sizes=sizes, cfg=cfg).saving == 0.0


class TestTheAdviceIsNotFree:
    """The injected sentence is admitted to the context like any other token.
    The old guard fired 903 times without ever charging for that."""

    def test_the_overhead_is_priced(self, cfg, sizes, tmp_path):
        f = tmp_path / "f.py"
        f.write_text("x" * 400_000)
        v = decide("Read", {"file_path": str(f)}, model=OPUS, remaining_turns=400,
                   sizes=sizes, cfg=cfg)
        assert v.overhead > 0, "injecting a sentence is not free"

    def test_it_stays_quiet_when_the_advice_costs_more_than_it_saves(self, sizes, tmp_path):
        """Assume advice is never taken and no fire can ever pay for itself."""
        f = tmp_path / "f.py"
        f.write_text("x" * 400_000)
        v = decide("Read", {"file_path": str(f)}, model=OPUS, remaining_turns=400,
                   sizes=sizes, cfg=Settings(advice_taken=0.0))
        assert not v.fire and "costs" in v.reason

    def test_the_net_is_the_discounted_saving_less_the_overhead(self):
        v = Verdict(True, "x", saving=1.0, overhead=0.1, advice_taken=0.5)
        assert v.net == pytest.approx(0.4)

    def test_a_fire_always_has_positive_expected_value(self, cfg, sizes, tmp_path):
        f = tmp_path / "f.py"
        f.write_text("x" * 400_000)
        v = decide("Read", {"file_path": str(f)}, model=OPUS, remaining_turns=400,
                   sizes=sizes, cfg=cfg)
        assert v.fire and v.net > 0


class TestDuplicateReads:
    """19.2% of unbounded reads of text files on the author's machine re-read a
    file already in the context. It is the only certain saving in this project.

    Quoted at 27.4% until the image fix: 138 of the 182 duplicates in that
    corpus were screenshots, capped near 1,600 tokens whatever their byte
    size."""

    @staticmethod
    def _seen(f):
        state = GuardState()
        observe("Read", {"file_path": str(f)}, state, Verdict(False, "first read"))
        return state

    def test_a_re_read_of_an_unchanged_file_fires(self, cfg, sizes, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x" * 40_000)
        v = decide("Read", {"file_path": str(f)}, model=OPUS, remaining_turns=300,
                   sizes=sizes, state=self._seen(f), cfg=cfg)
        assert v.fire and v.kind == "duplicate"
        assert "already in this context" in v.message

    def test_a_re_read_after_an_edit_is_legitimate(self, cfg, sizes, tmp_path):
        """Re-reading a file you just changed is the correct thing to do, and a
        guard that nags about it is a guard people uninstall."""
        f = tmp_path / "a.py"
        f.write_text("x" * 40_000)
        state = self._seen(f)
        f.write_text("y" * 40_000)                     # edited: mtime moves
        import os
        os.utime(f, (0, 0))
        v = decide("Read", {"file_path": str(f)}, model=OPUS, remaining_turns=300,
                   sizes=sizes, state=state, cfg=cfg)
        assert v.kind != "duplicate"

    def test_a_first_read_is_not_a_duplicate(self, cfg, sizes, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x" * 40_000)
        v = decide("Read", {"file_path": str(f)}, model=OPUS, remaining_turns=300,
                   sizes=sizes, state=GuardState(), cfg=cfg)
        assert v.kind != "duplicate"

    def test_a_bounded_re_read_is_not_a_duplicate(self, cfg, sizes, tmp_path):
        """`offset`/`limit` asks for a different slice; it is not the same read."""
        f = tmp_path / "a.py"
        f.write_text("x" * 40_000)
        v = decide("Read", {"file_path": str(f), "limit": 50}, model=OPUS,
                   remaining_turns=300, sizes=sizes, state=self._seen(f), cfg=cfg)
        assert v.kind != "duplicate"

    def test_a_tiny_duplicate_is_not_worth_a_sentence(self, cfg, sizes, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x" * 40)
        v = decide("Read", {"file_path": str(f)}, model=OPUS, remaining_turns=300,
                   sizes=sizes, state=self._seen(f), cfg=cfg)
        assert not v.fire

    def test_the_saving_is_the_whole_carry_not_a_delegation_margin(self, cfg, sizes,
                                                                   tmp_path):
        """Nothing has to be delegated: the tokens are already there."""
        f = tmp_path / "a.py"
        f.write_text("x" * 400_000)
        v = decide("Read", {"file_path": str(f)}, model=OPUS, remaining_turns=300,
                   sizes=sizes, state=self._seen(f), cfg=cfg)
        assert v.saving == pytest.approx(v.inline) and v.delegated == 0.0


class TestItDoesNotRepeatItself:
    def test_a_shape_is_advised_once_per_session(self, cfg, sizes, tmp_path):
        state = GuardState()
        inp = {"command": "cat one.py"}
        first = decide("Bash", inp, model=OPUS, remaining_turns=400, sizes=sizes,
                       state=state, cfg=cfg)
        assert first.fire
        observe("Bash", inp, state, first)
        second = decide("Bash", {"command": "cat two.py"}, model=OPUS,
                        remaining_turns=400, sizes=sizes, state=state, cfg=cfg)
        assert not second.fire and "already advised" in second.reason

    def test_there_is_a_ceiling_on_how_often_it_speaks(self, sizes, tmp_path):
        f = tmp_path / "f.py"
        f.write_text("x" * 400_000)
        state = GuardState(fires=15)
        v = decide("Read", {"file_path": str(f)}, model=OPUS, remaining_turns=400,
                   sizes=sizes, state=state, cfg=Settings(max_fires=15))
        assert not v.fire and "15 times" in v.reason


class TestNeedsPricing:
    """The hot path. Parsing a transcript per tool call is not affordable at the
    22,761 Bash calls this corpus contains."""

    def test_a_bounded_command_never_reaches_the_transcript(self, sizes):
        assert not needs_pricing("Bash", {"command": "wc -l f"}, sizes=sizes)

    def test_a_small_command_never_reaches_the_transcript(self, sizes):
        assert not needs_pricing("Bash", {"command": "echo hi"}, sizes=sizes)

    def test_an_unguarded_tool_never_reaches_the_transcript(self, sizes):
        assert not needs_pricing("Edit", {"file_path": "/x"}, sizes=sizes)

    def test_a_big_command_does(self, sizes):
        assert needs_pricing("Bash", {"command": "cat big.py"}, sizes=sizes)

    def test_a_possible_duplicate_always_does(self, sizes, tmp_path):
        """Cheap to check and certain to pay when it hits."""
        f = tmp_path / "a.py"
        f.write_text("x")
        state = GuardState(reads={str(f): 0.0})
        assert needs_pricing("Read", {"file_path": str(f)}, sizes=sizes, state=state)

    def test_an_already_advised_shape_does_not(self, sizes):
        state = GuardState(advised=["cat"])
        assert not needs_pricing("Bash", {"command": "cat f.py"}, sizes=sizes, state=state)


class TestState:
    def test_a_read_is_remembered_with_its_mtime(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x")
        state = observe("Read", {"file_path": str(f)}, GuardState(),
                        Verdict(False, "x"))
        assert state.reads[str(f)] == f.stat().st_mtime

    def test_a_fire_is_booked_on_both_sides(self):
        state = observe("Bash", {"command": "cat f"}, GuardState(),
                        Verdict(True, "x", saving=1.0, overhead=0.01))
        assert state.fires == 1 and state.saving == 1.0 and state.overhead == 0.01

    def test_the_read_memory_is_bounded(self, tmp_path):
        state = GuardState()
        for i in range(500):
            observe("Read", {"file_path": f"/tmp/f{i}.py"}, state, Verdict(False, "x"))
        assert len(state.reads) <= 400, "a dedup memory, not a record"

    def test_round_trip(self, tmp_path):
        p = tmp_path / "guard.json"
        state = GuardState(reads={"/a": 1.0}, advised=["cat"], fires=2,
                           saving=1.0, overhead=0.1)
        save_state("sess", state, p)
        back = load_state("sess", p)
        assert back.reads == {"/a": 1.0} and back.advised == ["cat"] and back.fires == 2

    def test_sessions_do_not_see_each_other(self, tmp_path):
        p = tmp_path / "guard.json"
        save_state("a", GuardState(fires=3), p)
        save_state("b", GuardState(fires=0), p)
        assert load_state("a", p).fires == 3 and load_state("b", p).fires == 0

    @pytest.mark.parametrize("blob", ["", "{", "[]", "null", '{"sess": 5}'])
    def test_a_corrupt_state_file_is_survived(self, tmp_path, blob):
        p = tmp_path / "guard.json"
        p.write_text(blob)
        assert load_state("sess", p) == GuardState()

    def test_a_corrupt_state_file_is_overwritten_not_appended_to(self, tmp_path):
        p = tmp_path / "guard.json"
        p.write_text("{not json")
        save_state("sess", GuardState(fires=1), p)
        assert json.loads(p.read_text())["sess"]["fires"] == 1

    def test_the_file_cannot_grow_without_bound(self, tmp_path):
        p = tmp_path / "guard.json"
        for i in range(260):
            save_state(f"s{i}", GuardState(fires=i), p)
        assert len(json.loads(p.read_text())) <= 200

    def test_an_unwritable_state_path_never_raises(self, tmp_path):
        """A hook that raises has stopped guarding, and nothing looks wrong."""
        target = tmp_path / "afile"
        target.write_text("x")
        save_state("sess", GuardState(), target / "nested.json")

    def test_solvency_is_the_discounted_saving_against_the_overhead(self):
        assert GuardState(saving=1.0, overhead=0.4).solvent(0.5)
        assert not GuardState(saving=1.0, overhead=0.6).solvent(0.5)


class TestFailOpen:
    """Every one of these would leave the tool call working and the guard dead."""

    def test_a_missing_state_file_is_not_an_error(self, tmp_path):
        assert load_state("nobody", tmp_path / "absent.json") == GuardState()

    def test_a_verdict_that_does_not_fire_emits_nothing(self):
        assert Verdict(False, "quiet").payload() == {}

    def test_deciding_with_no_model_and_no_state_works(self):
        assert decide("Bash", {"command": "cat f"}, model=OPUS,
                      remaining_turns=100).reason

    def test_an_empty_tool_input_is_survived(self, cfg, sizes):
        for tool in GUARDED:
            assert decide(tool, {}, model=OPUS, remaining_turns=100,
                          sizes=sizes, cfg=cfg).reason


class TestReadAfterWrite:
    """A `Write` puts the whole file in the context as its own tool input.
    Reading it back admits every one of those tokens a second time.

    `reread.py` cannot see this case: it compares result digests, and a write's
    result is "file written", not the content."""

    @staticmethod
    def _wrote(f, when=1_000.0):
        state = GuardState()
        observe("Write", {"file_path": str(f)}, state, Verdict(False, "watched"),
                now=when)
        return state

    def test_reading_back_a_file_this_session_wrote_fires(self, cfg, sizes, tmp_path):
        f = tmp_path / "new.py"
        f.write_text("x" * 40_000)
        state = self._wrote(f, when=f.stat().st_mtime)
        v = decide("Read", {"file_path": str(f)}, model=OPUS, remaining_turns=300,
                   sizes=sizes, state=state, cfg=cfg)
        assert v.fire and v.kind == "duplicate"
        assert "written by this session" in v.message

    def test_a_file_changed_after_our_write_is_worth_reading(self, cfg, sizes, tmp_path):
        """Something outside the session touched it; the context is stale."""
        f = tmp_path / "new.py"
        f.write_text("x" * 40_000)
        state = self._wrote(f, when=f.stat().st_mtime - 3_600)
        v = decide("Read", {"file_path": str(f)}, model=OPUS, remaining_turns=300,
                   sizes=sizes, state=state, cfg=cfg)
        assert v.kind != "duplicate"

    def test_an_edit_is_not_a_write(self, cfg, sizes, tmp_path):
        """An edit puts a hunk in the context, not a file, so re-reading an
        edited file can be the only way to see the rest of it. Advising against
        that would be advising against the correct move."""
        f = tmp_path / "a.py"
        f.write_text("x" * 40_000)
        state = GuardState()
        observe("Edit", {"file_path": str(f)}, state, Verdict(False, "watched"))
        assert not state.wrote
        v = decide("Read", {"file_path": str(f)}, model=OPUS, remaining_turns=300,
                   sizes=sizes, state=state, cfg=cfg)
        assert v.kind != "duplicate"

    def test_writing_supersedes_an_earlier_read(self, cfg, sizes, tmp_path):
        """The read memory is about a version of the file that no longer exists."""
        f = tmp_path / "a.py"
        f.write_text("x" * 40_000)
        state = GuardState()
        observe("Read", {"file_path": str(f)}, state, Verdict(False, "first"))
        observe("Write", {"file_path": str(f)}, state, Verdict(False, "watched"),
                now=1_000.0)
        assert str(f) not in state.reads

    def test_the_write_memory_is_bounded(self):
        state = GuardState()
        for i in range(500):
            observe("Write", {"file_path": f"/tmp/w{i}.py"}, state,
                    Verdict(False, "watched"), now=float(i))
        assert len(state.wrote) <= 400

    def test_writes_survive_a_state_round_trip(self, tmp_path):
        p = tmp_path / "guard.json"
        save_state("s", GuardState(wrote={"/a": 5.0}), p)
        assert load_state("s", p).wrote == {"/a": 5.0}

    def test_write_is_watched_but_never_advised_about(self, cfg, sizes):
        """Admitting a write costs nothing: the content is already in context."""
        assert "Write" in OBSERVED and "Write" not in GUARDED
        assert not decide("Write", {"file_path": "/tmp/x"}, model=OPUS,
                          remaining_turns=300, sizes=sizes, cfg=cfg).fire


class TestInstallation:
    """An uninstalled guard, a broken guard and a correctly quiet guard all
    produce exactly the same experience. This is the only thing that tells them
    apart, which makes it the most valuable line in the report."""

    @pytest.fixture
    def fake_home(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        (home / ".claude").mkdir(parents=True)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
        monkeypatch.chdir(tmp_path)
        return home

    def test_nothing_declared_is_reported_as_not_installed(self, fake_home):
        assert installed_in() == []

    def test_a_user_level_declaration_is_found(self, fake_home):
        target = fake_home / ".claude" / "settings.json"
        target.write_text(install_snippet())
        assert installed_in() == [target]

    def test_a_project_declaration_is_found(self, fake_home, tmp_path):
        proj = tmp_path / ".claude"
        proj.mkdir()
        (proj / "settings.json").write_text(install_snippet())
        assert (proj / "settings.json") in installed_in()

    def test_unparseable_settings_are_skipped_not_raised(self, fake_home):
        (fake_home / ".claude" / "settings.json").write_text("{not json")
        assert installed_in() == []

    def test_an_unrelated_hook_is_not_mistaken_for_this_one(self, fake_home):
        (fake_home / ".claude" / "settings.json").write_text(json.dumps(
            {"hooks": {"PreToolUse": [{"matcher": "Read",
                                       "hooks": [{"type": "command",
                                                  "command": "python3 other.py"}]}]}}))
        assert installed_in() == []

    def test_the_snippet_matches_every_tool_the_guard_watches(self):
        """A matcher narrower than OBSERVED silently disables part of the guard."""
        blob = json.loads(install_snippet())
        matcher = blob["hooks"]["PreToolUse"][0]["matcher"]
        assert set(matcher.split("|")) == set(OBSERVED)

    def test_the_snippet_points_at_a_hook_that_exists(self):
        assert hook_path().is_file()

    def test_the_report_leads_with_the_bad_news(self, fake_home):
        assert "NO — nothing is preventing spend" in report()

    def test_install_prints_and_writes_nothing(self, fake_home, capsys):
        """`adder config --init` set the precedent: this tool prints
        configuration for a person to place. It matters more here, because a
        hook changes what every session does."""
        from adder.decide.guard import main

        before = sorted(fake_home.rglob("*"))
        assert main(["--install"]) == 0
        assert "PreToolUse" in capsys.readouterr().out
        assert sorted(fake_home.rglob("*")) == before


class TestSubagentReturns:
    """A subagent cannot be advised to use a subagent.

    Pricing a `Task` through `placement_cost` quoted "vs $0.20 delegated to a
    subagent" for a call that *was* the subagent — modelling delegating a
    delegation, and quoting a saving from an option already taken."""

    @staticmethod
    def _sizes(p50, p90, n=30):
        return SizeModel(tools={"Task": (p50, p90, n)}, built=1.0, calls=n)

    def test_a_fat_return_is_priced_against_a_brief(self, cfg):
        v = decide("Task", {"description": "audit"}, model=OPUS, remaining_turns=400,
                   sizes=self._sizes(800, 9_000), cfg=cfg)
        assert v.fire and v.kind == "brief"
        assert "1,000 tokens" in v.message and "the findings" in v.message

    def test_it_never_advises_delegating_a_delegation(self, cfg):
        v = decide("Task", {"description": "audit"}, model=OPUS, remaining_turns=400,
                   sizes=self._sizes(800, 9_000), cfg=cfg)
        assert "delegated to a subagent" not in v.message

    def test_a_return_already_within_a_brief_is_left_alone(self, cfg):
        v = decide("Task", {"description": "audit"}, model=OPUS, remaining_turns=400,
                   sizes=self._sizes(200, 600), cfg=cfg)
        assert not v.fire and "within a brief" in v.reason

    def test_the_saving_is_the_difference_from_the_brief(self, cfg):
        from adder.decide.guard import BRIEF_TARGET_TOKENS
        from adder.pricing.cost import admitted_token_cost

        v = decide("Task", {"description": "audit"}, model=OPUS, remaining_turns=400,
                   sizes=self._sizes(800, 9_000), cfg=cfg)
        want = admitted_token_cost(BRIEF_TARGET_TOKENS, OPUS, 400)
        assert v.delegated == pytest.approx(want)
        assert v.saving == pytest.approx(v.inline - want)

    def test_a_short_session_has_nothing_to_amortize(self, cfg):
        assert not decide("Task", {"description": "audit"}, model=OPUS,
                          remaining_turns=1, sizes=self._sizes(800, 9_000),
                          cfg=cfg).fire

    def test_agent_is_treated_the_same_as_task(self, cfg):
        sizes = SizeModel(tools={"Agent": (800, 9_000, 30)}, built=1.0, calls=30)
        assert decide("Agent", {"description": "x"}, model=OPUS, remaining_turns=400,
                      sizes=sizes, cfg=cfg).kind == "brief"

    def test_webfetch_still_uses_the_placement_model(self, cfg):
        """Unlike a Task, a page fetch genuinely can be handed to a subagent."""
        sizes = SizeModel(tools={"WebFetch": (4_000, 40_000, 30)}, built=1.0, calls=30)
        v = decide("WebFetch", {"url": "https://example.com"}, model=OPUS,
                   remaining_turns=400, sizes=sizes, cfg=cfg)
        assert v.fire and v.kind == "size" and "subagent" in v.message


class TestTheAggregate:
    """The per-call view is structurally blind to the largest single channel.

    Across 222 local transcripts, 32 session-and-shape pairs exceed 20K
    cumulative tokens and together account for 19.7% of every Bash result token
    in the corpus. The biggest is `sed -n 'A,Bp'` — a *bounded* read, correctly
    waved through every time: 246 calls at a 513-token average, 126,222 tokens
    into one session. Every one of those calls really was small.
    """

    @staticmethod
    def _sizes(p50=513):
        return SizeModel(shapes={"sed+range": (p50, p50 * 2, 246)},
                         heads={"sed": (p50, p50 * 2, 246)}, built=1.0, calls=246)

    @staticmethod
    def _run(sizes, cfg, n, remaining=300):
        state = GuardState()
        inp = {"command": "sed -n '1,200p' big.py"}
        fired = []
        for _ in range(n):
            v = decide("Bash", inp, model=OPUS, remaining_turns=remaining,
                       sizes=sizes, state=state, cfg=cfg)
            observe("Bash", inp, state, v, sizes=sizes)
            if v.fire:
                fired.append(v)
        return state, fired

    def test_many_small_bounded_calls_eventually_fire(self, cfg):
        _, fired = self._run(self._sizes(), cfg, 60)
        assert fired and fired[0].kind == "aggregate"

    def test_a_single_call_of_that_shape_says_nothing(self, cfg):
        _, fired = self._run(self._sizes(), cfg, 1)
        assert not fired, "each of these calls is genuinely small"

    def test_bounded_calls_are_counted(self, cfg):
        """If they were not, the rule would never fire on the shape that
        accounts for most of the money."""
        state, _ = self._run(self._sizes(), cfg, 10)
        assert state.admitted["sed+range"] == 5_130
        assert state.shape_calls["sed+range"] == 10

    def test_it_accumulates_the_median_not_the_tail(self, cfg):
        """A running sum wants the expected value; the p90 would inflate the
        total by the ratio between them."""
        state, _ = self._run(self._sizes(p50=100), cfg, 4)
        assert state.admitted["sed+range"] == 400

    def test_it_says_it_once(self, cfg):
        _, fired = self._run(self._sizes(), cfg, 120)
        assert len(fired) == 1, "advice about a habit repeats for free and helps once"

    def test_the_message_names_the_count_and_the_total(self, cfg):
        _, fired = self._run(self._sizes(), cfg, 60)
        msg = fired[0].message
        assert "times this session" in msg and "$" in msg
        assert "sed+range" in msg

    def test_a_short_session_has_nothing_to_amortize(self, cfg):
        _, fired = self._run(self._sizes(), cfg, 60, remaining=1)
        assert not fired

    def test_it_claims_only_half_the_carry(self, cfg):
        """The tokens already admitted cannot be un-admitted. Only the calls
        still to come can be avoided, and assuming that is all of them would
        over-claim."""
        _, fired = self._run(self._sizes(), cfg, 60)
        assert fired[0].saving == pytest.approx(fired[0].inline * 0.5)

    def test_the_shape_table_is_bounded(self, cfg):
        state = GuardState()
        sizes = SizeModel(heads={}, shapes={}, built=1.0, calls=0)
        for i in range(500):
            observe("Bash", {"command": f"prog{i} --flag"}, state,
                    Verdict(False, "x"), sizes=sizes)
        assert len(state.admitted) <= 400 and len(state.shape_calls) <= 400

    def test_the_counters_survive_a_round_trip(self, tmp_path):
        p = tmp_path / "guard.json"
        save_state("s", GuardState(admitted={"cat": 9}, shape_calls={"cat": 3}), p)
        back = load_state("s", p)
        assert back.admitted == {"cat": 9} and back.shape_calls == {"cat": 3}

    def test_a_crossed_aggregate_is_worth_pricing_even_when_bounded(self):
        """The hot path must let it through, or the rule can never fire."""
        state = GuardState(admitted={"sed+range": 50_000})
        assert needs_pricing("Bash", {"command": "sed -n '1,20p' f"},
                             sizes=self._sizes(), state=state)


class TestTheAdviceIsActionable:
    """"Bound it" is advice nobody can act on without doing the arithmetic
    themselves, and the arithmetic depends on where in the session they are."""

    @staticmethod
    def _msg(tmp_path, remaining, cfg, sizes):
        f = tmp_path / "big.py"
        f.write_text("x" * 400_000)
        return decide("Read", {"file_path": str(f)}, model=OPUS,
                      remaining_turns=remaining, sizes=sizes, cfg=cfg).message

    def test_it_names_a_line_count(self, cfg, sizes, tmp_path):
        assert "limit:" in self._msg(tmp_path, 300, cfg, sizes)

    def test_the_count_shrinks_as_the_session_lengthens(self, cfg, sizes, tmp_path):
        """The same constant cannot be right at both ends of a session — which
        is the reason the trigger is a cost, and it applies to the advice too."""
        import re

        def lines(rem):
            m = re.search(r"limit: ([\d,]+)", self._msg(tmp_path, rem, cfg, sizes))
            return int(m.group(1).replace(",", ""))

        assert lines(50) > lines(300) > lines(900)

    def test_the_suggested_read_actually_clears_the_floor(self, cfg):
        """The number has to be true, not merely smaller."""
        from adder.core.shapes import READ_TOKENS_PER_LINE
        from adder.decide.guard import _affordable_lines
        from adder.pricing.cost import admitted_token_cost

        for remaining in (25, 100, 400, 1_200):
            n = _affordable_lines(OPUS, remaining, cfg, None, 0)
            cost = admitted_token_cost(int(n * READ_TOKENS_PER_LINE[1]), OPUS, remaining)
            assert cost < cfg.min_cost, f"{n} lines still costs ${cost:.2f}"

    def test_one_more_line_would_not_clear_it(self, cfg):
        """It is the largest affordable read, not an arbitrary safe one."""
        from adder.core.shapes import READ_TOKENS_PER_LINE
        from adder.decide.guard import _affordable_lines
        from adder.pricing.cost import admitted_token_cost

        remaining = 400
        n = _affordable_lines(OPUS, remaining, cfg, None, 0)
        over = admitted_token_cost(int((n + 1) * READ_TOKENS_PER_LINE[1]), OPUS,
                                   remaining)
        assert over >= cfg.min_cost

    def test_a_command_gets_a_command_shaped_hint(self, cfg, sizes):
        v = decide("Bash", {"command": "cat big.py"}, model=OPUS, remaining_turns=400,
                   sizes=sizes, cfg=cfg)
        assert "head -50" in v.message and "limit:" not in v.message


class TestExplain:
    """`--explain` answers "why did it say nothing about *this*", so it has to
    accept whatever the reader has in front of them."""

    def test_a_path_is_read_as_a_read(self):
        from adder.decide.guard import _as_call

        tool, inp = _as_call("/tmp/thing.py")
        assert tool == "Read" and inp["file_path"].endswith("thing.py")

    def test_anything_else_is_read_as_a_command(self):
        from adder.decide.guard import _as_call

        assert _as_call("cat foo.py | head -5")[0] == "Bash"

    def test_a_tool_name_is_read_as_that_tool(self):
        from adder.decide.guard import _as_call

        assert _as_call("Task")[0] == "Task"

    def test_a_path_with_spaces_is_still_a_command(self):
        """`ls /tmp` starts with a slash only by accident of argument order."""
        from adder.decide.guard import _as_call

        assert _as_call("./scripts/adder help")[0] == "Bash"


class TestReplay:
    """Replaying the guard over transcripts already paid for is the only honest
    check on the one component that speaks without being asked."""

    @staticmethod
    def _records(calls):
        """`calls` is a list of (tool, input) pairs, in order."""
        out = []
        for i, (tool, inp) in enumerate(calls):
            out.append({"type": "assistant", "sessionId": "s1",
                        "message": {"id": f"m{i}", "model": "claude-opus-5",
                                    "content": [{"type": "tool_use", "id": f"u{i}",
                                                 "name": tool, "input": inp}]}})
        return out

    def test_an_empty_tree_replays_to_nothing(self, tmp_path, isolated_home):
        from adder.decide.guard import replay

        r = replay(tmp_path)
        assert r.calls == 0 and r.fires == 0 and r.net == 0.0

    def test_it_counts_what_it_would_say(self, write_jsonl, tmp_path, isolated_home):
        from adder.decide.guard import replay

        f = tmp_path / "big.py"
        f.write_text("x" * 4_000_00)
        d = write_jsonl(self._records([("Read", {"file_path": str(f)})] * 3),
                        into=tmp_path / "proj")
        r = replay(d, cfg=Settings())
        assert r.calls == 3 and r.fires >= 1

    def test_it_never_writes_state(self, write_jsonl, tmp_path, isolated_home):
        """Replaying must not disturb a live session's memory."""
        from adder.decide.guard import replay

        f = tmp_path / "big.py"
        f.write_text("x" * 400_000)
        d = write_jsonl(self._records([("Read", {"file_path": str(f)})] * 2),
                        into=tmp_path / "proj")
        state_path = Settings.resolve().state_path
        before = state_path.exists()
        replay(d, cfg=Settings())
        assert state_path.exists() == before

    def test_the_net_is_the_saving_less_the_overhead(self, write_jsonl, tmp_path,
                                                     isolated_home):
        from adder.decide.guard import replay

        f = tmp_path / "big.py"
        f.write_text("x" * 400_000)
        d = write_jsonl(self._records([("Read", {"file_path": str(f)})] * 4),
                        into=tmp_path / "proj")
        r = replay(d, cfg=Settings())
        assert r.net == pytest.approx(r.saving - r.overhead)

    def test_most_calls_never_cost_a_transcript_parse(self, write_jsonl, tmp_path,
                                                     isolated_home):
        """The latency budget, measured rather than asserted."""
        from adder.decide.guard import replay

        calls = [("Bash", {"command": "wc -l f"})] * 50
        d = write_jsonl(self._records(calls), into=tmp_path / "proj")
        r = replay(d, cfg=Settings())
        assert r.lookup_rate == 0.0

    def test_uptake_scales_the_saving_and_not_the_cost(self, write_jsonl, tmp_path,
                                                      isolated_home):
        """The advice is paid for whether or not it is taken."""
        from adder.decide.guard import replay

        f = tmp_path / "big.py"
        f.write_text("x" * 400_000)
        d = write_jsonl(self._records([("Read", {"file_path": str(f)})] * 3),
                        into=tmp_path / "proj")
        half = replay(d, cfg=Settings(), advice_taken=0.5)
        full = replay(d, cfg=Settings(), advice_taken=1.0)
        assert full.saving == pytest.approx(half.saving * 2)
        assert full.overhead == pytest.approx(half.overhead)


class TestItsOwnFootprint:
    """The guard is charged for everything it says and everything it keeps."""

    def test_the_message_is_capped(self, cfg, sizes, tmp_path):
        from adder.decide.guard import MAX_MESSAGE_TOKENS
        from adder.util.text import est_tokens

        f = tmp_path / ("deeply/" * 30)
        f.mkdir(parents=True)
        target = f / ("a_very_long_file_name" * 8 + ".py")
        target.write_text("x" * 400_000)
        v = decide("Read", {"file_path": str(target)}, model=OPUS,
                   remaining_turns=400, sizes=sizes, cfg=cfg)
        text = v.payload()["hookSpecificOutput"]["additionalContext"]
        assert est_tokens(text) <= MAX_MESSAGE_TOKENS

    def test_clipping_does_not_cut_a_word_in_half(self):
        from adder.decide.guard import MAX_MESSAGE_TOKENS

        long = Verdict(True, "x", message="word " * (MAX_MESSAGE_TOKENS * 2))
        assert long.clipped().message.endswith("…")
        assert "wor…" not in long.clipped().message

    def test_a_short_message_is_untouched(self):
        v = Verdict(True, "x", message="short enough")
        assert v.clipped() is v

    def test_stale_sessions_are_dropped_by_age(self, tmp_path):
        import time

        p = tmp_path / "guard.json"
        old = GuardState(fires=1)
        old.touched = time.time() - 30 * 86_400
        save_state("ancient", old, p)
        # Rewrite the record with the aged stamp save_state would have replaced.
        blob = json.loads(p.read_text())
        blob["ancient"]["touched"] = time.time() - 30 * 86_400
        p.write_text(json.dumps(blob))
        save_state("fresh", GuardState(), p)
        assert "ancient" not in json.loads(p.read_text())

    def test_a_live_session_is_never_pruned(self, tmp_path):
        p = tmp_path / "guard.json"
        save_state("live", GuardState(fires=2), p)
        save_state("live", GuardState(fires=3), p)
        assert json.loads(p.read_text())["live"]["fires"] == 3

    def test_deciding_writes_nothing(self, cfg, sizes, tmp_path, monkeypatch):
        """`decide` is pure. Only the hook persists, and only after deciding."""
        f = tmp_path / "f.py"
        f.write_text("x" * 400_000)
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
        decide("Read", {"file_path": str(f)}, model=OPUS, remaining_turns=400,
               sizes=sizes, cfg=cfg)
        assert not list(home.rglob("*"))


class TestItKeepsNoContent:
    """The guard writes a state file under the user's home. CLAUDE.md's rule
    about transcripts -- real ones contain source code, file paths and prompts
    -- applies to anything this tool writes about them. The state is a dedup
    memory, and a dedup memory needs identities, never contents."""

    def test_a_read_records_a_path_and_an_mtime_only(self, tmp_path):
        f = tmp_path / "secret.py"
        f.write_text("API_KEY = 'hunter2'")
        state = observe("Read", {"file_path": str(f)}, GuardState(),
                        Verdict(False, "x"))
        assert "hunter2" not in json.dumps(state.to_json())

    def test_a_write_records_no_content(self, tmp_path):
        state = observe("Write", {"file_path": "/tmp/x", "content": "hunter2"},
                        GuardState(), Verdict(False, "x"), now=1.0)
        assert "hunter2" not in json.dumps(state.to_json())

    def test_a_command_is_reduced_to_its_shape(self, tmp_path):
        """`shape()` drops arguments, so a command carrying a token or a
        password never reaches disk."""
        sizes = SizeModel(shapes={}, heads={}, built=1.0, calls=0)
        state = observe("Bash", {"command": "curl -H 'Authorization: Bearer sk-secret' x"},
                        GuardState(), Verdict(False, "x"), sizes=sizes)
        blob = json.dumps(state.to_json())
        assert "sk-secret" not in blob and "Authorization" not in blob

    def test_an_advised_shape_carries_no_arguments(self):
        state = observe("Bash", {"command": "cat /home/me/.env"}, GuardState(),
                        Verdict(True, "x", saving=1.0, overhead=0.01))
        assert state.advised == ["cat"]

    def test_every_guarded_tool_has_a_hint(self):
        from adder.decide.guard import GUARDED, _bounded_hint

        for tool in GUARDED:
            assert _bounded_hint(tool) != "bound the output", \
                f"{tool} falls through to the generic hint"


class TestUptake:
    """`guard_advice_taken = 0.5` is the one number the solvency gate rests on
    and the only one nothing measured. It is measurable: a fire is recorded,
    and the transcript afterwards says whether the behaviour asked for
    happened. Not proof of causation, and not presented as any."""

    @staticmethod
    def _calls(rows):
        """rows: (ts, tool, input) — as `iter_calls` reads them."""
        out = []
        for i, (ts, tool, inp) in enumerate(rows):
            out.append({"type": "assistant", "sessionId": "s1", "timestamp": ts,
                        "message": {"id": f"m{i}", "model": "claude-opus-5",
                                    "content": [{"type": "tool_use", "id": f"u{i}",
                                                 "name": tool, "input": inp}]}})
        return out

    def test_no_fires_means_no_measurement(self, tmp_path, isolated_home):
        from adder.decide.guard import uptake

        u = uptake(tmp_path, log=tmp_path / "absent.jsonl")
        assert not u.measured and "assumption stands" in u.describe()

    def test_a_fire_records_a_shape_not_a_command(self, tmp_path):
        from adder.decide.guard import load_fires, record_fire

        log = tmp_path / "fires.jsonl"
        record_fire("s1", "Bash", {"command": "curl -H 'Bearer sk-secret' x"},
                    Verdict(True, "x", kind="size", tokens=9), path=log, now=1.0)
        rows = load_fires(log)
        assert rows[0]["shape"] == "curl" and "sk-secret" not in json.dumps(rows)

    def test_a_fire_records_a_basename_not_a_path(self, tmp_path):
        from adder.decide.guard import load_fires, record_fire

        log = tmp_path / "fires.jsonl"
        record_fire("s1", "Read", {"file_path": "/home/me/private/keys.env"},
                    Verdict(True, "x", kind="duplicate"), path=log, now=1.0)
        assert load_fires(log)[0]["name"] == "keys.env"
        assert "private" not in json.dumps(load_fires(log))

    def test_a_corrupt_line_is_skipped_not_fatal(self, tmp_path):
        from adder.decide.guard import load_fires

        log = tmp_path / "fires.jsonl"
        log.write_text('{"session": "s1"}\n{not json\n{"session": "s2"}\n')
        assert len(load_fires(log)) == 2

    def test_an_unwritable_log_never_raises(self, tmp_path):
        from adder.decide.guard import record_fire

        blocker = tmp_path / "afile"
        blocker.write_text("x")
        record_fire("s", "Bash", {"command": "cat f"}, Verdict(True, "x"),
                    path=blocker / "nested.jsonl", now=1.0)

    def test_bounding_after_a_finding_counts_as_taken(self, write_jsonl, tmp_path,
                                                      isolated_home):
        from adder.decide.guard import record_fire, uptake

        log = tmp_path / "fires.jsonl"
        d = write_jsonl(self._calls([
            ("2026-08-01T00:00:00Z", "Bash", {"command": "cat a.py"}),
            ("2026-08-01T00:00:02Z", "Bash", {"command": "cat b.py | head -20"}),
            ("2026-08-01T00:00:03Z", "Bash", {"command": "cat c.py | head -20"}),
        ]), into=tmp_path / "proj")
        import datetime
        ts = datetime.datetime(2026, 8, 1, 0, 0, 1,
                               tzinfo=datetime.timezone.utc).timestamp()
        record_fire("s1", "Bash", {"command": "cat a.py"},
                    Verdict(True, "x", kind="size"), path=log, now=ts)
        u = uptake(d, log=log)
        assert u.fires == 1 and u.changed == 1
        assert u.after > u.before

    def test_carrying_on_unbounded_counts_as_not_taken(self, write_jsonl, tmp_path,
                                                       isolated_home):
        from adder.decide.guard import record_fire, uptake

        log = tmp_path / "fires.jsonl"
        d = write_jsonl(self._calls([
            ("2026-08-01T00:00:00Z", "Bash", {"command": "cat a.py"}),
            ("2026-08-01T00:00:02Z", "Bash", {"command": "cat b.py"}),
        ]), into=tmp_path / "proj")
        import datetime
        ts = datetime.datetime(2026, 8, 1, 0, 0, 1,
                               tzinfo=datetime.timezone.utc).timestamp()
        record_fire("s1", "Bash", {"command": "cat a.py"},
                    Verdict(True, "x", kind="size"), path=log, now=ts)
        u = uptake(d, log=log)
        assert u.fires == 1 and u.changed == 0

    def test_a_duplicate_not_read_again_counts_as_taken(self, write_jsonl, tmp_path,
                                                        isolated_home):
        from adder.decide.guard import record_fire, uptake

        log = tmp_path / "fires.jsonl"
        d = write_jsonl(self._calls([
            ("2026-08-01T00:00:00Z", "Read", {"file_path": "/x/thing.py"}),
        ]), into=tmp_path / "proj")
        import datetime
        ts = datetime.datetime(2026, 8, 1, 0, 0, 1,
                               tzinfo=datetime.timezone.utc).timestamp()
        record_fire("s1", "Read", {"file_path": "/x/thing.py"},
                    Verdict(True, "x", kind="duplicate"), path=log, now=ts)
        u = uptake(d, log=log)
        assert u.fires == 1 and u.changed == 1

    def test_a_fire_with_nothing_after_it_is_not_judged(self, write_jsonl, tmp_path,
                                                        isolated_home):
        """Silence is not evidence either way."""
        from adder.decide.guard import record_fire, uptake

        log = tmp_path / "fires.jsonl"
        d = write_jsonl(self._calls([
            ("2026-08-01T00:00:00Z", "Bash", {"command": "cat a.py"}),
        ]), into=tmp_path / "proj")
        import datetime
        ts = datetime.datetime(2026, 8, 1, 0, 0, 9,
                               tzinfo=datetime.timezone.utc).timestamp()
        record_fire("s1", "Bash", {"command": "cat a.py"},
                    Verdict(True, "x", kind="size"), path=log, now=ts)
        assert uptake(d, log=log).fires == 0

    def test_ten_findings_are_needed_before_it_beats_the_assumption(self):
        from adder.decide.guard import Uptake

        assert not Uptake(fires=9, changed=9).measured
        assert Uptake(fires=10, changed=5).measured


class TestACorruptStateFileDoesNotSilenceTheGuard:
    """The state file is written from a hook and is small enough to hand-edit.

    Every failure here is invisible: the hook's only handler is a blanket
    `except`, so an exception out of `save_state` or `ledger` is not an error
    anybody sees -- it is the guard quietly no longer remembering anything.
    """

    def test_a_non_dict_session_entry_does_not_break_the_write(self, tmp_path):
        import json

        from adder.decide.guard import GuardState, load_state, save_state

        p = tmp_path / "guard.json"
        p.write_text(json.dumps({"old": "not-a-dict", "odd": {"touched": "yesterday"}}))
        save_state("new", GuardState(fires=2), p)
        assert load_state("new", p).fires == 2

    def test_a_non_numeric_field_does_not_break_the_ledger(self, tmp_path):
        import json

        from adder.decide.guard import ledger

        p = tmp_path / "guard.json"
        p.write_text(json.dumps({"s": {"fires": "lots", "saving": None,
                                       "overhead": {}}}))
        led = ledger(p)
        assert led["fires"] == 0 and led["saving"] == 0.0


class TestABoundedReadIsNotAWholeRead:
    """`limit`/`offset` admit a slice, and the guard must not claim otherwise.

    Remembering a bounded read as a whole one made the guard tell the model a
    later complete read was `already in this context and has not changed on
    disk` -- talking it out of the only move that would get the rest of the
    file. The guard is the one component here that changes behaviour.
    """

    def _file(self, tmp_path):
        f = tmp_path / "big.py"
        f.write_text("x" * 400_000)
        return f

    def test_a_limited_read_is_not_remembered_as_the_file(self, tmp_path):
        from adder.decide.guard import GuardState, Verdict, observe

        f = self._file(tmp_path)
        st = GuardState()
        observe("Read", {"file_path": str(f), "limit": 50}, st, Verdict(False, ""))
        assert st.reads == {}

    def test_an_offset_read_is_not_remembered_as_the_file(self, tmp_path):
        from adder.decide.guard import GuardState, Verdict, observe

        f = self._file(tmp_path)
        st = GuardState()
        observe("Read", {"file_path": str(f), "offset": 400}, st, Verdict(False, ""))
        assert st.reads == {}

    def test_a_whole_read_still_is(self, tmp_path):
        from adder.decide.guard import GuardState, Verdict, _already_known, observe

        f = self._file(tmp_path)
        st = GuardState()
        observe("Read", {"file_path": str(f)}, st, Verdict(False, ""))
        assert _already_known(str(f), st)

    def test_the_guard_stays_quiet_after_only_a_slice_was_read(self, tmp_path):
        from adder.decide.guard import GuardState, Verdict, _already_known, observe

        f = self._file(tmp_path)
        st = GuardState()
        observe("Read", {"file_path": str(f), "limit": 50}, st, Verdict(False, ""))
        assert _already_known(str(f), st) == ""
