"""Tests that lock in the empirical claims the design rests on.

If any of these fail, the plan's conclusions no longer hold and the router's
gates are unsafe.
"""

from datetime import date

import pytest

from router.cost import (
    M,
    admitted_token_cost,
    escalation_is_profitable,
    placement_cost,
    switch_is_profitable,
    turn_cost,
)
from router.prices import CACHE_READ_MULT, rate

OPUS, HAIKU, SONNET = "claude-opus-5", "claude-haiku-4-5", "claude-sonnet-5"


class TestPricing:
    def test_sonnet5_intro_expires(self):
        assert rate(SONNET, date(2026, 8, 31)) == (2, 10)
        assert rate(SONNET, date(2026, 9, 1)) == (3, 15)

    def test_dated_transcript_ids_resolve(self):
        assert rate("claude-haiku-4-5-20251001") == (1, 5)

    def test_aliases(self):
        assert rate("opus") == (5, 25)
        assert rate("haiku") == (1, 5)


class TestTheInversion:
    """The finding that reframed the whole project."""

    def test_opus_cached_read_is_half_of_haiku_uncached(self):
        opus_cached = rate(OPUS).inp * CACHE_READ_MULT      # $0.50 / MTok
        haiku_uncached = rate(HAIKU).inp                     # $1.00 / MTok
        assert opus_cached == pytest.approx(0.50)
        assert haiku_uncached == pytest.approx(1.00)
        assert haiku_uncached == pytest.approx(2 * opus_cached)

    def test_reading_1m_cached_on_opus_beats_haiku_fresh(self):
        stay = turn_cost(OPUS, cache_read=1_000_000)
        move = turn_cost(HAIKU, uncached_in=1_000_000)
        assert stay == pytest.approx(0.50)
        assert move == pytest.approx(1.00)
        assert move > stay


class TestAmortization:
    @pytest.mark.parametrize(
        "turns,expected",
        [(50, 0.3125), (200, 1.0625), (1000, 5.0625), (3478, 17.4525)],
    )
    def test_10k_tokens_in_opus_context(self, turns, expected):
        assert admitted_token_cost(10_000, OPUS, turns) == pytest.approx(expected, rel=1e-3)

    def test_cost_grows_linearly_in_remaining_turns(self):
        a = admitted_token_cost(10_000, OPUS, 100)
        b = admitted_token_cost(10_000, OPUS, 200)
        assert (b - a) == pytest.approx(10_000 * 5 * CACHE_READ_MULT * 100 / M)

    def test_zero_remaining_turns_is_just_the_write(self):
        assert admitted_token_cost(10_000, OPUS, 0) == pytest.approx(10_000 * 5 * 1.25 / M)


class TestCacheGate:
    def test_breakeven_is_ctx_over_40_optimistic_case(self):
        """Opus->Haiku, uncached switch: profitable iff out > ctx/40.

        `check_context=False` isolates the pure break-even arithmetic; 400K does
        not fit Haiku, and the feasibility gate is tested separately below.
        """
        ctx = 400_000
        below = switch_is_profitable(OPUS, HAIKU, ctx, ctx // 40 - 500, check_context=False)
        above = switch_is_profitable(OPUS, HAIKU, ctx, ctx // 40 + 500, check_context=False)
        assert not below and above

    def test_realistic_write_multiplier_is_stricter(self):
        """1.25x cache write moves break-even from ctx/40 to ~ctx/26.7."""
        ctx = 400_000
        at_40 = ctx // 40 + 100
        assert switch_is_profitable(OPUS, HAIKU, ctx, at_40, check_context=False)
        assert not switch_is_profitable(
            OPUS, HAIKU, ctx, at_40, switch_in_mult=1.25, check_context=False)

    def test_declines_on_measured_median_session(self):
        """544K ctx, 818 avg output tokens -> must refuse. This is the case
        every naive router gets wrong."""
        d = switch_is_profitable(OPUS, HAIKU, 544_000, 818)
        assert not d
        assert d.saving < 0
        assert "13,600" in d.reason or "needs >" in d.reason

    def test_cold_short_context_allows_the_switch(self):
        """The narrow band where per-turn model routing is genuinely right."""
        assert switch_is_profitable(OPUS, HAIKU, 2_000, 4_000)

    def test_decision_explains_itself(self):
        d = switch_is_profitable(OPUS, HAIKU, 544_000, 818)
        assert "544,000" in d.reason and "claude-haiku" in d.reason


class TestPlacement:
    def test_delegating_a_big_read_wins_in_a_long_session(self):
        inline, sub, d = placement_cost(
            tokens_read=50_000, summary_tokens=500,
            remaining_turns=3478, main_model=OPUS,
        )
        assert d and inline > 80 and sub < 1.5
        assert inline / sub > 50          # plan claimed ~300x at 500-tok summary

    def test_delegation_wins_even_with_zero_amortization(self):
        """Two independent axes, not one: a cheap model reading fresh ($1/MTok)
        already beats Opus admitting the same tokens ($6.25/MTok cache write),
        before any amortization. Delegation is not purely a long-session play."""
        inline, sub, d = placement_cost(
            tokens_read=50_000, summary_tokens=500,
            remaining_turns=0, main_model=OPUS,
        )
        assert d and inline > sub

    def test_delegating_loses_without_compression(self):
        """The real failure mode: a subagent that returns nearly everything it
        read adds a round trip and saves nothing."""
        _, _, d = placement_cost(
            tokens_read=5_000, summary_tokens=4_500,
            remaining_turns=0, main_model=OPUS,
        )
        assert not d and "keep inline" in d.reason

    def test_compression_ratio_drives_the_decision(self):
        ratios = []
        for summary in (500, 2_000, 4_500):
            _, _, d = placement_cost(
                tokens_read=5_000, summary_tokens=summary,
                remaining_turns=10, main_model=OPUS,
            )
            ratios.append(d.saving)
        assert ratios == sorted(ratios, reverse=True)

    def test_there_is_a_crossover(self):
        prev = None
        for turns in (0, 5, 20, 100):
            _, _, d = placement_cost(
                tokens_read=20_000, summary_tokens=800,
                remaining_turns=turns, main_model=OPUS,
            )
            if prev is not None:
                assert d.saving >= prev      # monotonically better with more turns
            prev = d.saving


class TestEscalationGate:
    def test_low_failure_rate_justifies_trying_cheap(self):
        assert escalation_is_profitable(
            HAIKU, OPUS, ctx_tokens=20_000, est_out_tokens=2_000, p_fail=0.1
        )

    def test_high_failure_rate_does_not(self):
        d = escalation_is_profitable(
            HAIKU, OPUS, ctx_tokens=20_000, est_out_tokens=2_000, p_fail=0.9
        )
        assert not d and "go straight to" in d.reason

    def test_failed_cheap_attempt_costs_more_than_never_trying(self):
        """The double-pay trap: cheap+expensive > expensive alone."""
        ctx, out = 20_000, 2_000
        cheap = turn_cost(HAIKU, uncached_in=ctx, out=out)
        exp = turn_cost(OPUS, uncached_in=ctx, out=out)
        assert cheap + exp > exp

    def test_rejects_invalid_probability(self):
        with pytest.raises(ValueError):
            escalation_is_profitable(HAIKU, OPUS, ctx_tokens=1, est_out_tokens=1, p_fail=1.5)
