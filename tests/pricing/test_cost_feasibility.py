"""A switch has to hold the turn it is recommended for, not only the prefix.

`switch_is_profitable` checked `fits(to_model, ctx_tokens)`. A 199K context on
Haiku's 200K window passed, while the task was about to read 30K more into it
and write 20K back -- `policy.decide` then returned a Haiku downgrade that
needed ~249K in a 200K window.
"""
from __future__ import annotations

from adder.pricing.cost import (
    SUBAGENT_OPENING_TOKENS,
    delegation_need,
    placement_cost,
    switch_is_profitable,
)
from adder.pricing.registry import context_window

OPUS, HAIKU = "claude-opus-5", "claude-haiku-4-5"


class TestSwitchFeasibility:
    def test_reads_and_output_count_against_the_window(self):
        d = switch_is_profitable(OPUS, HAIKU, 199_000, 20_000, extra_tokens=30_000)
        assert not d and "impossible" in d.reason
        assert "249,000" in d.reason

    def test_output_alone_can_overflow_a_context_that_fits(self):
        window = context_window(HAIKU)
        d = switch_is_profitable(OPUS, HAIKU, window - 100, 5_000)
        assert not d and "impossible" in d.reason

    def test_a_turn_that_fits_is_still_priced(self):
        d = switch_is_profitable(OPUS, HAIKU, 2_000, 4_000, extra_tokens=1_000)
        assert d and d.saving > 0

    def test_the_arithmetic_probe_still_ignores_the_window(self):
        d = switch_is_profitable(OPUS, HAIKU, 199_000, 20_000, extra_tokens=30_000,
                                 check_context=False)
        assert "impossible" not in d.reason


class TestDelegationNeed:
    def test_it_is_what_placement_checks(self):
        """One expression for both callers, so they cannot disagree."""
        window = context_window(HAIKU)
        read = window - SUBAGENT_OPENING_TOKENS - 200 + 1
        assert delegation_need(read, 200) == window + 1
        _, _, d = placement_cost(tokens_read=read, summary_tokens=200,
                                 remaining_turns=10, main_model=OPUS, sub_model=HAIKU)
        assert not d and "cannot delegate" in d.reason

    def test_negative_inputs_do_not_shrink_it(self):
        assert delegation_need(-5, -5) == SUBAGENT_OPENING_TOKENS
