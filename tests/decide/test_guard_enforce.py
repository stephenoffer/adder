"""Refusing, rather than advising: the level where adder stops being a report.

Every other test in this directory asks whether the guard said the right thing.
These ask whether it may *do* something, which is a different bar. An advisory
guard that is wrong costs a sentence. An enforcing guard that is wrong costs a
turn, or a session -- so the properties asserted here are mostly about the ways
a refusal must be survivable:

* it never happens twice for the same target,
* it always carries a reason and a way through,
* it is off unless somebody turned it on,
* and it is dropped the moment compaction makes its premise false.

The saving is only allowed to be booked undiscounted because the call did not
happen. That is the one number in this project with no uptake assumption behind
it, and `test_a_refusal_is_not_discounted` is what keeps it that way.
"""

from __future__ import annotations

import pytest

from adder.core.shapes import SizeModel
from adder.decide.guard import (
    ENFORCE_LEVELS,
    GuardState,
    Settings,
    Verdict,
    decide,
    observe,
)

OPUS = "claude-opus-5"


@pytest.fixture
def sizes():
    return SizeModel(
        shapes={"cat": (200, 40_000, 40), "echo": (5, 20, 40)},
        heads={"cat": (200, 40_000, 40), "echo": (5, 20, 40)},
        tools={}, built=1.0, calls=80,
    )


@pytest.fixture
def big(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x" * 40_000)
    return f


def _seen(f) -> GuardState:
    """State in which `f` has already been read whole."""
    state = GuardState()
    observe("Read", {"file_path": str(f)}, state, Verdict(False, "first read"))
    return state


def _read(f, state, cfg, sizes, turns: int = 300) -> Verdict:
    return decide("Read", {"file_path": str(f)}, model=OPUS, remaining_turns=turns,
                  sizes=sizes, state=state, cfg=cfg)


class TestItIsOffUntilItIsTurnedOn:
    """The default has to stay advisory. Someone who upgrades and reads no
    release notes must not find their tool calls being refused."""

    def test_the_default_settings_do_not_enforce(self):
        assert Settings().enforce == "off"
        assert Settings().enforcing is False

    def test_a_bare_environment_does_not_enforce(self):
        assert Settings.resolve(env={}).enforcing is False

    def test_the_env_var_turns_it_on(self):
        assert Settings.resolve(env={"ADDER_GUARD_ENFORCE": "certain"}).enforcing is True

    def test_an_unknown_level_reads_as_off(self):
        """A typo in a config file must not be what starts denying calls."""
        assert Settings.resolve(env={"ADDER_GUARD_ENFORCE": "aggressive"}).enforce == "off"

    def test_the_levels_are_ordered_least_to_most(self):
        assert ENFORCE_LEVELS == ("off", "certain", "full")

    def test_advising_still_happens_with_enforcement_off(self, sizes, big):
        v = _read(big, _seen(big), Settings(), sizes)
        assert v.fire and v.action == "advise"


class TestTheCertainClass:
    """Content already in the context. Refusing this cannot lose information,
    which is the only reason refusing is allowed at all."""

    def test_a_duplicate_read_is_refused(self, sizes, big):
        v = _read(big, _seen(big), Settings(enforce="certain"), sizes)
        assert v.fire and v.action == "deny" and v.certain

    def test_the_refusal_reaches_the_hook_as_a_deny(self, sizes, big):
        out = _read(big, _seen(big), Settings(enforce="certain"), sizes).payload()
        assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert out["hookSpecificOutput"]["permissionDecisionReason"]

    def test_a_refusal_always_says_how_to_proceed(self, sizes, big):
        """A wall is a bug; a redirection is the feature."""
        v = _read(big, _seen(big), Settings(enforce="certain"), sizes)
        assert Verdict.ESCAPE in v.message

    def test_the_way_through_survives_the_message_clipper(self):
        """The escape clause is at the end of the sentence, so a clipper that
        does not know about it turns "ask again" into a wall."""
        long = Verdict(True, "refused", deny=True, message="word " * 400)
        assert Verdict.ESCAPE in long.clipped().message

    def test_a_refusal_is_still_bounded(self):
        long = Verdict(True, "refused", deny=True, message="word " * 400)
        assert len(long.clipped().message) < len("word " * 400)

    def test_a_read_after_an_edit_is_still_allowed(self, sizes, tmp_path):
        import os
        f = tmp_path / "a.py"
        f.write_text("x" * 40_000)
        state = _seen(f)
        f.write_text("y" * 40_000)
        os.utime(f, (0, 0))
        assert _read(f, state, Settings(enforce="certain"), sizes).action != "deny"

    def test_a_first_read_is_never_refused(self, sizes, big):
        assert _read(big, GuardState(), Settings(enforce="certain"), sizes).action != "deny"

    def test_a_small_duplicate_is_refused_even_below_the_advice_floor(self, sizes,
                                                                      tmp_path):
        """The size floor exists to stop the guard *interrupting*. A refusal is
        not an interruption: nothing is being weighed and nothing is lost, so
        the only test it has to pass is that it beats the cost of its own
        sentence."""
        f = tmp_path / "small.py"
        f.write_text("x" * 12_000)                     # ~3K tokens, under the floor
        cfg = Settings(min_tokens=100_000, enforce="certain")
        assert _read(f, _seen(f), cfg, sizes).action == "deny"

    def test_a_duplicate_worth_less_than_the_sentence_is_left_alone(self, sizes,
                                                                    tmp_path):
        f = tmp_path / "tiny.py"
        f.write_text("x" * 40)
        assert not _read(f, _seen(f), Settings(enforce="certain"), sizes).fire


class TestRefusingIsSurvivable:
    """The property that makes enforcement safe to ship. The guard cannot be
    argued with, so it has to yield on its own."""

    def test_the_second_attempt_at_the_same_read_goes_through(self, sizes, big):
        cfg, state = Settings(enforce="certain"), _seen(big)
        first = _read(big, state, cfg, sizes)
        observe("Read", {"file_path": str(big)}, state, first)
        assert first.action == "deny"
        assert _read(big, state, cfg, sizes).action != "deny"

    def test_a_refusal_does_not_record_the_read_as_admitted(self, sizes, big):
        """The denied call never happened, so nothing entered the context. If
        the guard remembered it anyway, its own refusal would become the
        evidence for the next one."""
        state = GuardState()
        observe("Read", {"file_path": str(big)}, state,
                Verdict(True, "refused", deny=True, target=f"Read:{big}"))
        assert str(big) not in state.reads

    def test_compaction_drops_every_already_in_context_claim(self, sizes, big):
        """Compaction is what makes the premise false. `precompact_learn`
        calls this; without it the guard refuses reads of content the model no
        longer has, which is the one way enforcement costs more than it
        saves."""
        state = _seen(big)
        assert _read(big, state, Settings(enforce="certain"), sizes).action == "deny"
        state.forget_context()
        assert _read(big, state, Settings(enforce="certain"), sizes).action != "deny"

    def test_forgetting_keeps_the_spend_record(self, sizes):
        state = GuardState(saving=3.0, overhead=0.1, prevented=2.0, fires=4)
        state.forget_context()
        assert (state.saving, state.overhead, state.prevented, state.fires) == \
            (3.0, 0.1, 2.0, 4)

    def test_the_refusal_ledger_survives_a_round_trip(self, big):
        state = GuardState()
        observe("Read", {"file_path": str(big)}, state,
                Verdict(True, "refused", deny=True, saving=1.5,
                        target=f"Read:{big}"))
        back = GuardState.from_json(state.to_json())
        assert f"Read:{big}" in back.denied and back.prevented == pytest.approx(1.5)


class TestTheFullLevel:
    """Refusing a large first read. A weaker claim than the duplicate one --
    it rests on a horizon and on a subagent returning a brief -- so it is a
    separate opt-in and it always names the cheaper call."""

    def test_certain_does_not_refuse_a_large_first_read(self, sizes, big):
        v = _read(big, GuardState(), Settings(enforce="certain"), sizes)
        assert v.fire and v.action == "advise"

    def test_full_refuses_it_and_names_the_alternative(self, sizes, big):
        v = _read(big, GuardState(), Settings(enforce="full"), sizes)
        assert v.fire and v.action == "deny"
        assert "limit" in v.message or "delegate" in v.message

    def test_full_yields_on_the_second_attempt(self, sizes, big):
        cfg, state = Settings(enforce="full"), GuardState()
        first = _read(big, state, cfg, sizes)
        observe("Read", {"file_path": str(big)}, state, first)
        assert first.action == "deny"
        assert _read(big, state, cfg, sizes).action != "deny"

    def test_a_read_with_no_cheaper_equal_is_not_refused(self, sizes, big):
        """Late in a session there are too few turns left for delegation to
        pay, and then the correct output is silence, not a refusal."""
        v = _read(big, GuardState(), Settings(enforce="full"), sizes, turns=1)
        assert v.action != "deny"


class TestTheAccounting:
    """A refusal is the only saving here with no uptake assumption attached.
    Keeping it that way is the point of the separate counter."""

    def test_a_refusal_is_not_discounted(self, sizes, big):
        v = _read(big, _seen(big), Settings(enforce="certain"), sizes)
        assert v.uptake == 1.0
        assert v.net == pytest.approx(v.saving - v.overhead)

    def test_advice_is_still_discounted(self, sizes, big):
        v = _read(big, _seen(big), Settings(), sizes)
        assert v.uptake == 0.5

    def test_prevented_is_counted_apart_from_promised(self, sizes, big):
        state = _seen(big)
        v = _read(big, state, Settings(enforce="certain"), sizes)
        observe("Read", {"file_path": str(big)}, state, v)
        assert state.prevented == pytest.approx(v.saving)
        assert state.prevented <= state.saving

    def test_advice_promises_but_prevents_nothing(self, sizes, big):
        state = _seen(big)
        v = _read(big, state, Settings(), sizes)
        observe("Read", {"file_path": str(big)}, state, v)
        assert state.saving > 0 and state.prevented == 0.0


class TestWhatReachesDisk:
    """`shape()` exists so a command carrying a token never lands in the state
    file. The refusal ledger is a new thing that reaches disk, and it keeps the
    same promise."""

    def test_a_command_target_is_a_shape_not_a_command(self):
        from adder.decide.guard import _target
        got = _target("Bash", {"command": "curl -H 'Authorization: Bearer sekrit' x"})
        assert "sekrit" not in got and got.startswith("Bash:")

    def test_a_pattern_is_hashed_rather_than_stored(self):
        from adder.decide.guard import _target
        assert "sekrit" not in _target("Grep", {"pattern": "sekrit"})

    def test_targets_of_different_calls_do_not_collide(self):
        from adder.decide.guard import _target
        assert _target("Grep", {"pattern": "a"}) != _target("Grep", {"pattern": "b"})
