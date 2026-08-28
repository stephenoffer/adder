"""Shadow mode: the refusal decision, run in full, carried out on nothing.

Every advisory dollar this tool reports is multiplied by `guard_advice_taken`,
which is 0.5 and assumed on any machine that has not measured it -- the docs
call it the weakest number in the project. Enforcement then asks a user to hand
refusal authority to the guard on the strength of that assumption.

Shadow mode is the answer to that. It computes exactly the refusal `certain`
would make, records it, and refuses nothing, so the trade is measured on this
machine before anything is denied. That gives it two hard obligations, and this
file is mostly those two:

* **It must cost nothing.** No message reaches the model, so no tokens are
  admitted, so the overhead is zero and the fire ceiling does not apply. A mode
  that charged for its own measurement would be a worse assumption than the one
  it replaces.
* **It must be able to say no.** A shadow refusal the session went around is
  the closest thing to evidence that the refusal was wrong -- under enforcement
  it would have cost a turn -- and a report that only counted the savings would
  be an advertisement rather than a measurement.
"""

from __future__ import annotations

import pytest

from adder.core.shapes import SizeModel
from adder.decide.guard import (
    GuardState,
    Settings,
    Verdict,
    decide,
    last_session,
    main,
    observe,
    record_fire,
    save_state,
    shadow,
)

OPUS = "claude-opus-5"
SHADOW = Settings(enforce="shadow")
CERTAIN = Settings(enforce="certain")


@pytest.fixture
def sizes():
    return SizeModel(
        shapes={"cat": (200, 40_000, 40)}, heads={"cat": (200, 40_000, 40)},
        tools={}, built=1.0, calls=80,
    )


@pytest.fixture
def big(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x" * 40_000)
    return f


def _seen(f) -> GuardState:
    state = GuardState()
    observe("Read", {"file_path": str(f)}, state, Verdict(False, "first read"))
    return state


def _read(f, state, cfg, sizes) -> Verdict:
    return decide("Read", {"file_path": str(f)}, model=OPUS, remaining_turns=300,
                  sizes=sizes, state=state, cfg=cfg)


class TestItRefusesNothing:
    def test_the_verdict_is_a_shadow_not_a_denial(self, sizes, big):
        v = _read(big, _seen(big), SHADOW, sizes)
        assert v.fire and v.shadow and not v.deny and v.action == "shadow"

    def test_nothing_reaches_the_model(self, sizes, big):
        """The payload is the entire interface to the session. Empty means the
        turn proceeds exactly as it would have with the guard uninstalled."""
        assert _read(big, _seen(big), SHADOW, sizes).payload() == {}

    def test_it_finds_the_same_call_certain_would_refuse(self, sizes, big):
        """Otherwise it is measuring a different guard than the one it is
        recommending, which is the failure `bench` was pulled up on."""
        assert _read(big, _seen(big), CERTAIN, sizes).deny
        assert _read(big, _seen(big), SHADOW, sizes).shadow

    def test_a_read_it_shadowed_is_still_remembered_as_read(self, sizes, big):
        """A denied read never happened, so `observe` deliberately forgets it.
        A shadowed one *did* happen -- the tokens are in the context -- and
        forgetting it would make the next duplicate invisible."""
        state = _seen(big)
        v = _read(big, state, SHADOW, sizes)
        observe("Read", {"file_path": str(big)}, state, v)
        assert str(big) in state.reads


class TestItCostsNothing:
    def test_the_overhead_is_zero(self, sizes, big):
        assert _read(big, _seen(big), SHADOW, sizes).overhead == 0.0

    def test_no_part_of_the_saving_is_credited(self, sizes, big):
        """`uptake` is 0 because nothing was realised. The counterfactual
        saving is still carried -- it is what the report is about -- but the
        ledger may not book a cent of it."""
        v = _read(big, _seen(big), SHADOW, sizes)
        assert v.saving > 0 and v.uptake == 0.0

    def test_it_does_not_spend_the_fire_budget(self, sizes, big):
        """The ceiling bounds how often the guard may interrupt. Applying it
        here would truncate the measurement at `guard_max_fires` findings a
        session and still print as if it were complete."""
        state = _seen(big)
        state.fires = 999
        assert _read(big, state, Settings(enforce="shadow", max_fires=15),
                     sizes).shadow

    def test_the_visible_ledger_stays_empty(self, sizes, big):
        """A machine in shadow mode must be able to report, truthfully, that
        its guard has said nothing and promised nothing."""
        state = _seen(big)
        v = _read(big, state, SHADOW, sizes)
        observe("Read", {"file_path": str(big)}, state, v)
        assert (state.fires, state.saving, state.overhead, state.prevented) == \
            (0, 0.0, 0.0, 0.0)
        assert state.shadow_fires == 1 and state.shadow_saving > 0


class TestItCanSayNo:
    def test_asking_again_is_recorded_as_a_contradiction(self, sizes, big):
        """The evidence against. Under enforcement this is the escape hatch
        firing, which means the model had a reason the guard could not see."""
        state = _seen(big)
        v = _read(big, state, SHADOW, sizes)
        observe("Read", {"file_path": str(big)}, state, v)
        observe("Read", {"file_path": str(big)}, state,
                Verdict(False, "second ask"))
        assert sum(state.contradicted.values()) == 1

    def test_the_call_that_created_the_record_is_not_its_own_contradiction(
            self, sizes, big):
        state = _seen(big)
        v = _read(big, state, SHADOW, sizes)
        observe("Read", {"file_path": str(big)}, state, v)
        assert not state.contradicted

    def test_reading_the_same_file_through_the_shell_counts(self, sizes, big):
        """`Read:` and `Bash:` are different targets, so nothing matching on
        the target alone would see this -- and it is the exact case the
        duplicate rule exists for, since the harness picks which tool reads."""
        state = _seen(big)
        v = _read(big, state, SHADOW, sizes)
        observe("Read", {"file_path": str(big)}, state, v)
        observe("Bash", {"command": f"cat {big}"}, state,
                Verdict(False, "went round it"))
        assert sum(state.contradicted.values()) == 1

    def test_an_unrelated_call_is_not_a_contradiction(self, sizes, big, tmp_path):
        other = tmp_path / "b.py"
        other.write_text("y")
        state = _seen(big)
        v = _read(big, state, SHADOW, sizes)
        observe("Read", {"file_path": str(big)}, state, v)
        observe("Read", {"file_path": str(other)}, state, Verdict(False, "other"))
        assert not state.contradicted

    def test_compaction_drops_the_shadow_record(self, sizes, big):
        """After compaction the tokens the refusal called redundant are gone,
        so a later read is the session recovering them rather than the guard
        having been wrong. Counting it would libel the measurement."""
        state = _seen(big)
        v = _read(big, state, SHADOW, sizes)
        observe("Read", {"file_path": str(big)}, state, v)
        state.forget_context()
        observe("Read", {"file_path": str(big)}, state, Verdict(False, "re-read"))
        assert not state.contradicted


class TestTheReport:
    def test_it_aggregates_across_sessions(self, tmp_path):
        path = tmp_path / "state.json"
        for i in range(3):
            st = GuardState(shadow_fires=2, shadow_saving=1.0,
                            contradicted={f"Read:{i}": 1})
            save_state(f"s{i}", st, path)
        sh = shadow(path)
        assert sh.sessions == 3 and sh.fires == 6
        assert sh.would_save == pytest.approx(3.0)
        assert sh.contradicted == 3 and sh.contradictions == 3

    def test_the_realised_figure_writes_off_every_contradiction(self, tmp_path):
        """Deliberately the harsh reading: a contradicted refusal did not fail
        to save, it would have cost a turn. A lower bound is the only honest
        number to put next to `adder auto on`."""
        path = tmp_path / "state.json"
        save_state("s", GuardState(shadow_fires=4, shadow_saving=8.0,
                                   contradicted={"Read:a": 1}), path)
        sh = shadow(path)
        assert sh.contradiction_rate == 0.25
        assert sh.realised == pytest.approx(6.0)

    def test_a_missing_state_file_is_empty_rather_than_fatal(self, tmp_path):
        assert shadow(tmp_path / "nope.json").fires == 0

    def test_ten_findings_before_it_claims_a_rate(self, tmp_path):
        path = tmp_path / "state.json"
        save_state("s", GuardState(shadow_fires=9, shadow_saving=1.0), path)
        assert not shadow(path).measured
        save_state("s", GuardState(shadow_fires=10, shadow_saving=1.0), path)
        assert shadow(path).measured


class TestTheLastSession:
    """`adder guard --last`, which exists because of how a refusal gets turned off.

    A user who suspects the guard blocked something they needed has, today, no
    way to look. What they can do is set `guard_enforce=off`, and that is what
    they will do -- so the cheapest thing this tool can offer is the list.
    """

    @pytest.fixture
    def log(self, tmp_path):
        path = tmp_path / "fires.jsonl"
        record_fire("older", "Read", {"file_path": "/x/a.py"},
                    Verdict(True, "size", kind="size", tokens=9),
                    path=path, now=100.0)
        record_fire("newest", "Read", {"file_path": "/x/b.py"},
                    Verdict(True, "duplicate", kind="duplicate", tokens=10,
                            saving=1.0, deny=True, target="Read:/x/b.py"),
                    path=path, now=200.0)
        record_fire("newest", "Bash", {"command": "cat /x/c.py"},
                    Verdict(True, "duplicate", kind="duplicate", tokens=11,
                            saving=2.0, shadow=True, target="Bash:cat"),
                    path=path, now=201.0)
        return path

    def test_it_reports_only_the_newest_session(self, log):
        session, rows = last_session(log)
        assert session == "newest" and len(rows) == 2

    def test_the_rows_are_in_the_order_they_happened(self, log):
        _, rows = last_session(log)
        assert [r["ts"] for r in rows] == [200.0, 201.0]

    def test_a_refusal_is_labelled_as_one(self, log):
        _, rows = last_session(log)
        assert [r["action"] for r in rows] == ["deny", "shadow"]

    def test_an_empty_log_is_not_fatal(self, tmp_path):
        assert last_session(tmp_path / "nope.jsonl") == ("", [])

    def test_nothing_in_the_output_is_an_argument(self, log, capsys,
                                                  monkeypatch, tmp_path):
        """`record_fire` promises that a shape reaches disk and a command does
        not. A report that reconstructed one would break the promise on the way
        back out."""
        monkeypatch.setenv("ADDER_HOME", str(log.parent))
        (log.parent / "adder-guard-fires.jsonl").write_text(
            log.read_text(), encoding="utf-8")
        assert main(["--last"]) == 0
        assert "/x/c.py" not in capsys.readouterr().out
