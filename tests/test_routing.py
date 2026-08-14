"""Classifier and policy behaviour, including the cases that must NOT route."""

import pytest

from router.classify import Tier, classify
from router.policy import decide, routing_overhead

OPUS = "claude-opus-5"


class TestClassifierPrecision:
    @pytest.mark.parametrize("task", [
        "what does cost.py do",
        "where is the retry logic",
        "list the files in router/",
        "which function computes the rate",
    ])
    def test_short_read_only_goes_cheap(self, task):
        v = classify(task)
        assert v.tier == Tier.T0 and v.read_only and not v.abstained

    @pytest.mark.parametrize("task", [
        "refactor auth across the codebase",
        "why is the cache invalidating",
        "design a migration plan for the storage layer",
        "investigate the race condition in the worker pool",
        "root-cause the performance regression",
    ])
    def test_hard_vocabulary_never_routes_down(self, task):
        assert classify(task).tier >= Tier.T2

    def test_stack_trace_forces_strong_model(self):
        v = classify("this fails:\nTraceback (most recent call last):\n  File x")
        assert v.tier >= Tier.T2

    def test_ambiguous_abstains_upward(self):
        v = classify("make it better")
        assert v.tier == Tier.T2 and v.abstained

    def test_empty_task_routes_up(self):
        assert classify("").tier == Tier.T2

    def test_multi_step_mutation_is_not_cheap(self):
        v = classify("add a retry then update the tests and finally bump the version")
        assert v.tier >= Tier.T2

    def test_read_only_flag_only_on_nonmutating(self):
        assert not classify("fix the typo in README.md").read_only

    def test_is_deterministic(self):
        t = "refactor the auth module"
        assert classify(t).tier == classify(t).tier

    def test_is_fast_and_offline(self):
        import time
        start = time.perf_counter()
        for _ in range(2000):
            classify("refactor the auth module across the service")
        per_call_ms = (time.perf_counter() - start) / 2000 * 1000
        assert per_call_ms < 10.0


class TestRoutingDeclines:
    """The failure mode that sinks naive routers: routing when it costs more."""

    def test_declines_when_saving_below_overhead(self):
        p = decide("what does prices.py do", context_tokens=900_000, remaining_turns=3)
        assert not p.worth_it
        assert "does not clear routing overhead" in p.render()

    def test_routes_when_saving_clears_overhead(self):
        p = decide("what does prices.py do", context_tokens=400_000, remaining_turns=500)
        assert p.worth_it and p.action == "delegate"

    def test_overhead_scales_with_context(self):
        assert routing_overhead(900_000, OPUS) > routing_overhead(100_000, OPUS)

    def test_overhead_is_material_in_big_sessions(self):
        """~$0.25 just to spend a turn deciding, at 500K context."""
        assert 0.20 < routing_overhead(500_000, OPUS) < 0.35


class TestPolicyRouting:
    def test_hard_task_delegates_to_strong_tier(self):
        p = decide("refactor auth across the codebase",
                   context_tokens=400_000, remaining_turns=400)
        assert p.agent == "route-t2" and p.model == "claude-opus-5"

    def test_cheap_task_delegates_to_haiku(self):
        p = decide("what does prices.py do", context_tokens=400_000, remaining_turns=400)
        assert p.agent == "route-t0" and p.model == "claude-haiku-4-5"

    def test_never_downgrades_a_hard_task(self):
        p = decide("redesign the storage layer", context_tokens=5_000, remaining_turns=2)
        assert p.action != "downgrade"

    def test_plan_explains_itself(self):
        p = decide("what does prices.py do", context_tokens=400_000, remaining_turns=400)
        assert p.reasons and all(isinstance(r, str) for r in p.reasons)

    def test_zero_remaining_turns_still_safe(self):
        p = decide("what does x do", context_tokens=10_000, remaining_turns=0)
        assert isinstance(p.saving, float)


class TestNoTaskGuard:
    """If argument substitution fails, refuse rather than delegate nothing."""

    @pytest.mark.parametrize("task", ["", "   ", "$ARGUMENTS", "${ARGUMENTS}", "$1"])
    def test_missing_task_refuses_to_route(self, task):
        p = decide(task, context_tokens=400_000, remaining_turns=500)
        assert p.action == "inline" and not p.worth_it and p.confidence == 0.0
        assert "no task text" in " ".join(p.reasons)

    def test_real_task_still_routes(self):
        p = decide("what does prices.py do", context_tokens=400_000, remaining_turns=500)
        assert p.action == "delegate"
