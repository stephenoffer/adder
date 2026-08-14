"""The A/B harness must not manufacture passes. These test the scorer itself."""

import pytest

from adder.ab import TASKS, ArmResult, Outcome, report, wilson_lower_bound


class TestCheckers:
    def _check(self, task_id):
        return next(t for t in TASKS if t.id == task_id).check

    def test_accepts_correct_terse_answer(self):
        assert self._check("breakeven")("50")

    def test_accepts_correct_answer_in_a_sentence(self):
        assert self._check("breakeven")("It returns 50 turns.")

    def test_rejects_wrong_number(self):
        assert not self._check("breakeven")("It returns 25 turns.")

    def test_tolerates_digit_separators(self):
        assert self._check("default-remaining")("1 return 450")
        assert self._check("default-remaining")("450")

    def test_rejects_empty_and_refusal(self):
        c = self._check("breakeven")
        assert not c("")
        assert not c("I cannot determine that from the file.")

    def test_negation_guard_rejects_the_opposite_answer(self):
        """'falsy' checkers must not pass on 'truthy'."""
        c = self._check("gate-falsy")
        assert c("falsy")
        assert not c("truthy")

    def test_negation_guard_rejects_hedged_both_answers(self):
        c = self._check("gate-falsy")
        assert not c("It could be truthy or falsy depending on context.")

    def test_direction_checker_rejects_wrong_direction(self):
        c = self._check("abstain-direction")
        assert c("up")
        assert not c("down")
        assert not c("It routes down to a cheaper model.")

    def test_multi_needle_requires_all_parts(self):
        c = self._check("sonnet-intro-expiry")
        assert c("2026-08-31")
        assert not c("2026-09-30")

    @pytest.mark.parametrize("t", TASKS)
    def test_every_task_source_exists(self, t):
        assert t.context(), f"{t.id}: source {t.source} missing or empty"

    @pytest.mark.parametrize("t", TASKS)
    def test_no_checker_passes_on_empty_output(self, t):
        """A model that says nothing must never score a point."""
        assert not t.check(""), f"{t.id} passes on empty output"

    @pytest.mark.parametrize("t", TASKS)
    def test_no_checker_passes_on_refusal(self, t):
        assert not t.check("I don't know."), f"{t.id} passes on a refusal"


class TestWilson:
    def test_zero_sample_is_zero(self):
        assert wilson_lower_bound(0, 0) == 0.0

    def test_perfect_small_sample_is_not_conclusive(self):
        """12/12 must not read as 100% certainty."""
        assert wilson_lower_bound(12, 12) < 0.80

    def test_more_data_tightens_the_bound(self):
        assert wilson_lower_bound(120, 120) > wilson_lower_bound(12, 12)

    def test_bound_is_below_point_estimate(self):
        assert wilson_lower_bound(9, 12) < 9 / 12


class TestReport:
    def _arm(self, model, passed, n, cost):
        a = ArmResult(model)
        a.outcomes = [Outcome("t", model, i < passed, cost=cost / n) for i in range(n)]
        return a

    def test_flags_quality_loss(self):
        cheap = self._arm("haiku", 6, 12, 0.001)
        strong = self._arm("opus", 12, 12, 0.010)
        assert "do not route this task class down" in report([cheap, strong])

    def test_reports_no_loss_when_matched(self):
        cheap = self._arm("haiku", 12, 12, 0.001)
        strong = self._arm("opus", 12, 12, 0.010)
        assert "no measured quality loss" in report([cheap, strong])

    def test_always_states_the_scope_limit(self):
        r = report([self._arm("haiku", 12, 12, 0.001), self._arm("opus", 12, 12, 0.01)])
        assert "smoke test, not proof" in r and "tier T0" in r
