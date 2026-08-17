"""Calibration must be conservative with little data and responsive with lots."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from adder.decide.track.outcomes import (
    MIN_EVIDENCE,
    Outcome,
    calibration,
    evidence,
    load,
    main,
    p_fail,
    readiness,
    record,
)


@pytest.fixture
def log(tmp_path):
    return tmp_path / "outcomes.jsonl"


def _mk(tier="T0", escalated=False, project="p"):
    return Outcome(tier=tier, model="claude-haiku-4-5", project=project, escalated=escalated)


class TestPrior:
    def test_no_history_is_maximally_cautious(self, log):
        assert p_fail("T0", log=log) == pytest.approx(0.5)

    def test_single_failure_does_not_swing_the_gate(self, log):
        record(_mk(escalated=True), log)
        assert 0.5 < p_fail("T0", log=log) < 0.7   # smoothed, not 1.0


class TestConvergence:
    def test_converges_toward_observed_rate(self, log):
        for _ in range(90):
            record(_mk(escalated=False), log)
        for _ in range(10):
            record(_mk(escalated=True), log)
        assert p_fail("T0", log=log) == pytest.approx(0.108, abs=0.02)

    def test_tiers_are_scored_separately(self, log):
        for _ in range(20):
            record(_mk(tier="T0", escalated=True), log)
            record(_mk(tier="T1", escalated=False), log)
        assert p_fail("T0", log=log) > p_fail("T1", log=log)

    def test_falls_back_to_global_when_project_is_sparse(self, log):
        for _ in range(30):
            record(_mk(project="big", escalated=False), log)
        record(_mk(project="new", escalated=True), log)
        # One datapoint in "new" must not dominate; global history is used.
        assert p_fail("T0", project="new", log=log) < 0.3


class TestRobustness:
    def test_corrupt_lines_are_skipped(self, log):
        record(_mk(), log)
        log.write_text(log.read_text() + "{not json\n")
        record(_mk(), log)
        assert len(load(log)) == 2

    def test_record_never_raises(self, tmp_path):
        record(_mk(), tmp_path / "no" / "such" / "dir" / "x.jsonl")   # must not raise

    def test_calibration_reports_all_tiers(self, log):
        record(_mk(tier="T0"), log)
        record(_mk(tier="T2", escalated=True), log)
        cal = calibration(log)
        assert set(cal) == {"T0", "T2"} and cal["T2"]["escalated"] == 1


class TestEvidence:
    """`p_fail` alone cannot say whether it is a measurement or an admission."""

    def test_an_empty_log_is_labelled_a_prior_not_a_measurement(self, log):
        e = evidence("T0", "p", log)
        assert e.p_fail == 0.5 and e.scope == "prior" and not e.informative
        assert "prior" in e.describe()

    def test_a_thin_log_is_real_but_not_yet_actionable(self, log):
        for _ in range(6):
            record(_mk(), log)
        e = evidence("T0", "p", log)
        assert e.scope != "prior" and e.n == 6
        assert not e.informative, "6 runs must not be enough to override a classifier"

    def test_enough_recent_history_becomes_actionable(self, log):
        for _ in range(30):
            record(_mk(), log)
        e = evidence("T0", "p", log)
        assert e.informative and e.n == 30 and e.weight > MIN_EVIDENCE

    def test_scope_falls_back_to_global_and_says_so(self, log):
        for _ in range(30):
            record(_mk(project="old"), log)
        record(_mk(project="new"), log)
        e = evidence("T0", "new", log)
        assert e.scope == "global" and "global" in e.describe()

    def test_stale_history_stops_being_actionable(self, log):
        import time

        old = time.time() - 365 * 86400
        for _ in range(30):
            o = _mk()
            o.ts = old
            record(o, log)
        e = evidence("T0", "p", log)
        assert e.n == 30 and not e.informative, "a year-old log is not current evidence"

    def test_p_fail_still_returns_the_bare_number(self, log):
        for _ in range(20):
            record(_mk(escalated=True), log)
        # Unweighted: the recency weight is read off the wall clock inside each
        # call, so the weighted form is not bit-for-bit comparable with itself.
        kw = {"recency_weighted": False}
        assert p_fail("T0", "p", log, **kw) == evidence("T0", "p", log, **kw).p_fail


class TestTimestampCoercion:
    """`ts` is epoch seconds here and an ISO string everywhere else in the repo.

    A row carrying the wrong one used to load cleanly and then raise inside the
    recency weighting, where both callers swallow the exception — so the
    failure was not an error message, it was the outcome log quietly ceasing to
    influence routing.
    """

    def test_an_iso_timestamp_is_accepted(self, log):
        row = {"tier": "T0", "model": "m", "project": "p", "escalated": True,
               "ts": "2026-08-14T10:00:00Z"}
        Path(log).write_text(json.dumps(row) + "\n")
        rows = load(log)
        assert len(rows) == 1 and isinstance(rows[0].ts, float)
        assert evidence("T0", "p", log).n == 1        # and does not raise

    def test_an_unparseable_timestamp_drops_the_row(self, log):
        rows = [{"tier": "T0", "model": "m", "project": "p", "escalated": True,
                 "ts": "not a time"},
                {"tier": "T0", "model": "m", "project": "p", "escalated": False,
                 "ts": 1_780_000_000.0}]
        Path(log).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        assert len(load(log)) == 1

    def test_a_boolean_is_not_a_timestamp(self, log):
        Path(log).write_text(json.dumps(
            {"tier": "T0", "model": "m", "project": "p", "escalated": True,
             "ts": True}) + "\n")
        assert load(log) == []

    def test_a_recorded_outcome_still_round_trips(self, log):
        record(_mk(escalated=True), log)
        rows = load(log)
        assert len(rows) == 1 and isinstance(rows[0].ts, float)


class TestRecordFromTheCommandLine:
    """The adaptive half of the tool was behind a Python snippet nobody ran."""

    def test_one_run_lands_in_the_log(self, log, capsys):
        assert main(["record", "--tier", "T1", "--model", "claude-sonnet-5",
                     "--project", "demo", "--log", str(log)]) == 0
        rows = load(log)
        assert len(rows) == 1
        assert rows[0].tier == "T1" and not rows[0].escalated
        assert "recorded" in capsys.readouterr().out

    def test_an_escalation_is_recorded_as_one(self, log):
        main(["record", "--tier", "T0", "--escalated", "--log", str(log)])
        assert load(log)[0].escalated

    def test_the_optional_fields_survive_the_round_trip(self, log):
        main(["record", "--tier", "T2", "--model", "m", "--project", "p",
              "--cost", "0.25", "--context", "400000", "--remaining", "120",
              "--effort", "high", "--reason", "why", "--log", str(log)])
        o = load(log)[0]
        assert (o.cost, o.context_tokens, o.remaining_turns) == (0.25, 400_000, 120)
        assert o.effort == "high" and o.reason == "why"

    def test_a_tier_is_required(self, log):
        with pytest.raises(SystemExit):
            main(["record", "--log", str(log)])

    def test_recording_enough_runs_makes_the_tier_actionable(self, log):
        for _ in range(15):
            main(["record", "--tier", "T1", "--project", "demo", "--log", str(log)])
        assert evidence("T1", "demo", log).informative

    def test_it_never_writes_where_it_was_not_told_to(self, tmp_path, capsys):
        """The default log is the user's home. A test must not be able to reach it."""
        target = tmp_path / "nested" / "o.jsonl"
        main(["record", "--tier", "T0", "--log", str(target)])
        assert target.exists() and len(load(target)) == 1


class TestReadiness:
    """An empty log makes a silent router. This is the report that unsilences it."""

    def test_an_empty_log_names_the_shortfall_rather_than_a_p_fail(self, log):
        rows = readiness(log)
        assert rows and all(r["is_prior"] for r in rows)
        assert all(r["shortfall"] == MIN_EVIDENCE for r in rows)
        assert all(r["verdict"] == "needs history" for r in rows)

    def test_the_shortfall_shrinks_as_runs_arrive(self, log):
        before = readiness(log)[1]["shortfall"]
        for _ in range(6):
            record(_mk(tier="T1"), log)
        after = readiness(log, project="p")[1]["shortfall"]
        assert 0 < after < before

    def test_enough_clean_runs_make_a_tier_usable(self, log):
        for _ in range(20):
            record(_mk(tier="T1"), log)
        t1 = next(r for r in readiness(log, project="p") if r["tier"] == "T1")
        assert t1["verdict"] == "usable" and not t1["is_prior"]

    def test_a_tier_that_keeps_failing_is_not_usable_however_much_history(self, log):
        for _ in range(40):
            record(_mk(tier="T1", escalated=True), log)
        t1 = next(r for r in readiness(log, project="p") if r["tier"] == "T1")
        assert t1["verdict"] == "fails its own break-even"
        assert t1["p_fail"] >= t1["break_even"]

    def test_the_break_even_tightens_as_context_grows(self, log):
        """A bigger context makes the turn that catches a failure more expensive."""
        small = readiness(log, context_tokens=50_000)[0]["break_even"]
        large = readiness(log, context_tokens=900_000)[0]["break_even"]
        assert large < small

    def test_the_report_renders_without_a_log(self, tmp_path, capsys):
        assert main(["--log", str(tmp_path / "absent.jsonl")]) == 0
        out = capsys.readouterr().out
        assert "No outcomes recorded yet" in out
        assert "adder outcomes record" in out, "say how to fix it, not just that it is broken"
        assert "never buys a downgrade" in out

    def test_json_carries_both_halves(self, log, capsys):
        import json as _json

        record(_mk(tier="T1"), log)
        assert main(["--log", str(log), "--json"]) == 0
        got = _json.loads(capsys.readouterr().out)
        assert "calibration" in got and "readiness" in got
        assert {r["tier"] for r in got["readiness"]} == {"T0", "T1"}
