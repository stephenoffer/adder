"""The four gates: feasibility, placement, escalation risk, and overhead."""

import pytest

from adder.classify import Tier
from adder.cost import escalation_is_profitable, placement_cost, switch_is_profitable
from adder.policy import choose_effort, decide
from adder.savings import cache_discipline, effort_reduction, tool_output_discipline
from adder.trace import Session, Turn

OPUS, HAIKU = "claude-opus-5", "claude-haiku-4-5"


def _sess(n=100, ctx=200_000, out=500):
    s = Session("s", "p")
    for i in range(n):
        s.turns.append(Turn("s", "p", OPUS, 0, ctx + i * 1000, 0, out, 0, False,
                            ts=f"2026-08-14T10:{i % 60:02d}:00Z"))
    return s


class TestFeasibilityGate:
    def test_switch_to_a_model_that_cannot_hold_the_context_is_refused(self):
        d = switch_is_profitable(OPUS, HAIKU, 544_000, 100_000)
        assert not d and "context limit" in d.reason

    def test_pure_economics_can_be_probed_separately(self):
        d = switch_is_profitable(OPUS, HAIKU, 544_000, 100_000, check_context=False)
        assert d          # 100K output clears the break-even, ignoring feasibility

    def test_delegation_refuses_a_read_larger_than_the_subagent_window(self):
        _, _, d = placement_cost(
            tokens_read=500_000, summary_tokens=5_000, remaining_turns=300,
            main_model=OPUS, sub_model=HAIKU)
        assert not d and "cannot delegate" in d.reason

    def test_escalation_refuses_a_cheap_tier_that_cannot_hold_the_context(self):
        d = escalation_is_profitable(HAIKU, OPUS, ctx_tokens=544_000,
                                     est_out_tokens=500, p_fail=0.0)
        assert not d and "exceeds" in d.reason

    def test_a_huge_read_escalates_the_tier_for_feasibility(self):
        p = decide("what is in the log", context_tokens=100_000, remaining_turns=300,
                   est_read_tokens=400_000, p_fail=0.0)
        assert p.tier >= Tier.T1
        assert any("feasibility" in w for w in p.warnings)


class TestEscalationGate:
    def test_a_tier_that_always_fails_is_not_recommended(self):
        p = decide("what does prices.py do", context_tokens=400_000,
                   remaining_turns=300, p_fail=1.0)
        assert p.model == OPUS

    def test_a_reliable_cheap_tier_is_used(self):
        p = decide("what does prices.py do", context_tokens=400_000,
                   remaining_turns=300, p_fail=0.0)
        assert p.model == HAIKU

    def test_p_fail_is_reported_on_the_plan(self):
        p = decide("what does prices.py do", context_tokens=400_000,
                   remaining_turns=300, p_fail=0.25)
        assert p.p_fail == 0.25

    def test_p_fail_out_of_range_is_rejected(self):
        with pytest.raises(ValueError):
            escalation_is_profitable(HAIKU, OPUS, ctx_tokens=1000,
                                     est_out_tokens=1, p_fail=1.5)


class TestEffortChoice:
    def test_haiku_gets_no_effort_flag(self):
        assert choose_effort(Tier.T0, HAIKU) == "default"

    def test_opus_gets_the_tier_effort(self):
        assert choose_effort(Tier.T2, OPUS) == "high"
        assert choose_effort(Tier.T3, OPUS) == "xhigh"

    def test_plans_always_name_an_effort(self):
        p = decide("refactor the whole system", context_tokens=100_000,
                   remaining_turns=100)
        assert p.effort


class TestNewLevers:
    def test_tool_discipline_targets_the_read_half_of_the_pool(self):
        sessions = {"a": _sess()}
        e = tool_output_discipline(sessions, ".")
        assert e.saving >= 0 and 0.0 <= e.pool_fraction <= 1.0

    def test_effort_reduction_is_bounded_and_labelled(self):
        e = effort_reduction({"a": _sess()})
        assert e.confidence == "MODELLED"
        assert 0.0 <= e.pool_fraction <= 1.0
        assert "prior" in e.assumptions

    def test_effort_reduction_rejects_unknown_levels(self):
        with pytest.raises(ValueError):
            effort_reduction({"a": _sess()}, to_effort="turbo")

    def test_cache_discipline_is_measured_and_never_negative(self):
        e = cache_discipline({"a": _sess()})
        assert e.confidence == "MEASURED" and e.saving >= 0
