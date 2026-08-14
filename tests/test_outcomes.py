"""Calibration must be conservative with little data and responsive with lots."""

import pytest

from router.outcomes import Outcome, calibration, load, p_fail, record


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
