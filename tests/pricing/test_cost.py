"""Tests that lock in the empirical claims the design rests on.

If any of these fail, the plan's conclusions no longer hold and adder's
gates are unsafe.
"""
from __future__ import annotations

from datetime import date

import pytest

from adder.pricing.cost import (
    M,
    admitted_token_cost,
    choose_ttl,
    effort_saving,
    escalation_is_profitable,
    fanout_cost,
    placement_cost,
    switch_is_profitable,
    turn_cost,
)
from adder.pricing.prices import CACHE_READ_MULT, rate
from adder.pricing.registry import resolve

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


class TestDelegationCanFail:
    """The term placement was missing. A delegated read that comes back useless
    costs the subagent run, the turn that noticed, AND the inline read anyway --
    so it is strictly worse than never having delegated."""

    def _place(self, **kw):
        return placement_cost(tokens_read=50_000, summary_tokens=5_000,
                              remaining_turns=300, main_model=OPUS,
                              sub_model=HAIKU, **kw)

    def test_zero_redo_risk_reproduces_the_old_number(self):
        assert self._place(p_redo=0.0)[1] == pytest.approx(self._place()[1])

    def test_redo_risk_makes_delegation_dearer(self):
        assert self._place(p_redo=0.25)[1] > self._place(p_redo=0.0)[1]

    def test_the_catch_turn_is_charged(self):
        """A bad summary does not announce itself; a main-session turn has to
        read it, judge it, and dispatch again."""
        cheap = self._place(p_redo=0.25, redo_overhead=0.0)[1]
        dear = self._place(p_redo=0.25, redo_overhead=0.50)[1]
        assert dear - cheap == pytest.approx(0.25 * 0.50)

    def test_certain_failure_costs_more_than_never_delegating(self):
        inline, sub, d = self._place(p_redo=1.0, redo_overhead=0.10)
        assert sub > inline and not d

    def test_a_large_enough_redo_risk_flips_the_verdict(self):
        assert bool(self._place(p_redo=0.0)[2])
        assert not bool(self._place(p_redo=0.9, redo_overhead=0.5)[2])

    def test_rejects_an_invalid_probability(self):
        with pytest.raises(ValueError):
            self._place(p_redo=1.5)


class TestCarryOverride:
    """`admitted_token_cost` takes a measured carry model, and must be exactly
    the old arithmetic when it is not given one."""

    def test_none_is_the_documented_default(self):
        from adder.measure.window.carry import Carry

        assert admitted_token_cost(10_000, OPUS, 300, carry=Carry()) == pytest.approx(
            admitted_token_cost(10_000, OPUS, 300))

    def test_a_measured_multiplier_changes_the_answer(self):
        from adder.measure.window.carry import Carry

        dear = Carry(read_mult=0.20, source="measured")
        assert admitted_token_cost(10_000, OPUS, 300, carry=dear) > \
            admitted_token_cost(10_000, OPUS, 300)

    def test_placement_passes_the_carry_through_to_both_sides(self):
        """Inline and the admitted summary both carry; a carry model that only
        reached one of them would bias every comparison."""
        from adder.measure.window.carry import Carry

        dear = Carry(read_mult=0.40, source="measured")
        inline_d, sub_d, _ = placement_cost(
            tokens_read=50_000, summary_tokens=5_000, remaining_turns=300,
            main_model=OPUS, sub_model=HAIKU, carry=dear)
        inline_p, sub_p, _ = placement_cost(
            tokens_read=50_000, summary_tokens=5_000, remaining_turns=300,
            main_model=OPUS, sub_model=HAIKU)
        assert inline_d > inline_p and sub_d > sub_p


class TestOneSharedExpression:
    """`cost.py` and `select.py` must not carry separate copies of this.

    They did, briefly. The duplicate is how they came to disagree about whether
    the carry term could be corrected at all: one grew a fitted-carry hook and
    the other could not accept one. These tests exist to make a second
    divergence fail loudly rather than quietly produce two different dollar
    figures for the same tokens.
    """

    def test_the_two_entry_points_agree_exactly(self):
        from adder.decide.route.select import Need, cost_of
        from adder.pricing.catalog import load

        opus = load().get("claude-opus-5")
        for n, turns in ((10_000, 100), (50_000, 300), (200_000, 40), (1, 1)):
            claude_side = admitted_token_cost(n, "claude-opus-5", turns)
            catalog_side = cost_of(opus, Need(
                est_read_tokens=n, remaining_turns=turns, est_out_tokens=0,
                context_tokens=0, summary_tokens=1)).inline
            assert catalog_side == pytest.approx(claude_side, rel=1e-12), (
                f"{n} tokens over {turns} turns"
            )

    def test_the_split_adds_up(self):
        from adder.pricing.cost import Rates, admitted_cost

        got = admitted_cost(10_000, Rates(inp=5, out=25, cache_read=0.5,
                                          cache_write=6.25), reads=100)
        assert got.total == pytest.approx(got.write + got.reads)
        assert got.write == pytest.approx(10_000 * 6.25 / 1e6)
        assert got.reads == pytest.approx(10_000 * 0.5 * 100 / 1e6)

    def test_reads_is_a_count_not_a_turn_count(self):
        """A fitted carry model discounts re-reads for compaction survival."""
        from adder.pricing.cost import Rates, admitted_cost

        r = Rates(inp=5, out=25, cache_read=0.5, cache_write=6.25)
        assert admitted_cost(1_000, r, reads=16.4).reads == pytest.approx(
            admitted_cost(1_000, r, reads=16).reads * 16.4 / 16)

    def test_negative_reads_cannot_pay_you(self):
        from adder.pricing.cost import Rates, admitted_cost

        got = admitted_cost(1_000, Rates(5, 25, 0.5, 6.25), reads=-50)
        assert got.reads == 0.0 and got.total == got.write

    def test_claude_rates_come_from_the_dated_table(self):
        from adder.pricing.cost import Rates

        early = Rates.claude("claude-sonnet-5", on=date(2026, 8, 31))
        late = Rates.claude("claude-sonnet-5", on=date(2026, 9, 1))
        assert early.inp < late.inp                      # intro rate expires
        assert late.cache_read == pytest.approx(late.inp * 0.10)
        assert late.cache_write == pytest.approx(late.inp * 1.25)


# A lever the model does not have is not a lever, whatever it would save.
#
# Every other gate in `cost.py` checks feasibility before profitability: a model
# that cannot hold the context is refused rather than priced. `effort_saving` had
# no such check. Haiku 4.5 rejects the `effort` parameter outright -- `prices.py`
# records that as `efforts=()` with a comment saying so -- and lowering it was
# quoted as a $0.09 saving on a 400-turn horizon. Opus 4.6 takes low/medium/high/
# max and not xhigh, and raising to xhigh was quoted the same way. Both are 400s,
# not savings.
#
# The check is deliberately limited to the first-party table, which is
# authoritative about which levels a model accepts. A catalog entry carries no
# effort levels at all, and refusing on absence would switch the lever off for
# every non-Claude model on the strength of a field nobody populates.
class TestAModelThatRejectsEffort:
    def test_haiku_takes_no_effort_levels_at_all(self):
        assert resolve("claude-haiku-4-5").efforts == ()

    def test_lowering_effort_on_it_is_refused(self):
        _, d = effort_saving(5_000, "claude-haiku-4-5", from_effort="high",
                             to_effort="medium", remaining_turns=400)
        assert not d.ok

    def test_and_is_not_quoted_a_saving(self):
        saved, d = effort_saving(5_000, "claude-haiku-4-5", from_effort="high",
                                 to_effort="medium", remaining_turns=400)
        assert saved == 0.0
        assert d.saving == 0.0

    def test_the_reason_names_the_model_and_what_it_accepts(self):
        _, d = effort_saving(5_000, "claude-haiku-4-5", to_effort="medium")
        assert "claude-haiku-4-5" in d.reason
        assert "effort" in d.reason


class TestALevelTheModelDoesNotHave:
    def test_opus_4_6_does_not_take_xhigh(self):
        assert "xhigh" not in resolve("claude-opus-4-6").efforts

    def test_so_moving_to_xhigh_is_refused(self):
        saved, d = effort_saving(5_000, "claude-opus-4-6", from_effort="high",
                                 to_effort="xhigh", remaining_turns=400)
        assert not d.ok and saved == 0.0

    def test_a_level_it_does_take_still_prices(self):
        saved, d = effort_saving(5_000, "claude-opus-4-6", from_effort="high",
                                 to_effort="medium", remaining_turns=400)
        assert d.ok and saved > 0


class TestTheLeverStillWorksWhereItExists:
    def test_opus_5_takes_the_whole_ladder(self):
        saved, d = effort_saving(5_000, "claude-opus-5", from_effort="high",
                                 to_effort="medium", remaining_turns=400)
        assert d.ok and saved > 0

    def test_raising_effort_is_still_reported_as_not_cheaper(self):
        """The pre-existing branch: a supported level that costs more."""
        _, d = effort_saving(5_000, "claude-opus-5", from_effort="high",
                             to_effort="xhigh", remaining_turns=400)
        assert not d.ok
        assert "not cheaper" in d.reason

    def test_an_unknown_level_is_still_a_ValueError_not_a_refusal(self):
        with pytest.raises(ValueError):
            effort_saving(1_000, "claude-opus-5", to_effort="turbo")


class TestTheGateBoundaries:
    """Mutation testing found 15 of 23 comparisons here untested.

    Flipping `>` to `>=`, or `<=` to `<`, in the gates that decide whether adder
    emits advice left the whole suite green. These are the load-bearing
    comparisons in the package -- each one is the difference between "recommend"
    and "stay quiet" -- so each gets its boundary pinned rather than its happy
    path re-checked.
    """

    def test_switching_to_the_model_you_are_on_costs_nothing(self):
        """It is not a switch, so there is no prefix to invalidate.

        The arithmetic charges the target a full uncached re-read of the
        context, which is what losing a model-scoped prefix costs -- and which
        staying put does not do. Unguarded it reported a $0.45 loss for a no-op.
        """
        d = switch_is_profitable(OPUS, OPUS, 100_000, 1_000, check_context=False)
        assert d.saving == 0.0 and not d.ok
        assert "no switch" in d.reason

    def test_a_dated_variant_is_the_same_model(self):
        d = switch_is_profitable(OPUS, "claude-opus-5[1m]", 100_000, 1_000,
                                 check_context=False)
        assert d.saving == 0.0

    def test_a_real_switch_is_still_priced(self):
        d = switch_is_profitable(OPUS, HAIKU, 100_000, 1_000, check_context=False)
        assert d.saving != 0.0

    def test_a_switch_to_a_dearer_model_loses(self):
        d = switch_is_profitable(HAIKU, OPUS, 100_000, 1_000, check_context=False)
        assert not d.ok and d.saving < 0

    def test_breakeven_is_infinite_when_output_is_no_cheaper(self):
        """`denom <= 0`: same output rate means the switch never pays back."""
        from adder.pricing.cost import _breakeven_out
        from adder.pricing.registry import rate as _rate
        r = _rate(OPUS)
        assert _breakeven_out(100_000, r, r, 1.0) == float("inf")

    def test_placement_declines_a_zero_saving(self):
        """Delegating something that costs exactly the same is not a saving."""
        _, _, d = placement_cost(tokens_read=0, summary_tokens=1,
                                 remaining_turns=0, main_model=OPUS,
                                 sub_model=OPUS, brief_tokens=0)
        assert not d.ok

    def test_escalation_declines_at_its_own_break_even(self):
        """At exactly `max_tolerable_p_fail` the expected saving is zero."""
        from adder.pricing.cost import max_tolerable_p_fail
        cap = max_tolerable_p_fail(HAIKU, OPUS, ctx_tokens=50_000,
                                   est_out_tokens=1_000)
        at = escalation_is_profitable(HAIKU, OPUS, ctx_tokens=50_000,
                                      est_out_tokens=1_000, p_fail=cap)
        assert not at.ok
        just_under = escalation_is_profitable(HAIKU, OPUS, ctx_tokens=50_000,
                                              est_out_tokens=1_000,
                                              p_fail=max(0.0, cap - 1e-6))
        assert just_under.ok

    def test_max_tolerable_p_fail_is_zero_when_the_denominator_vanishes(self):
        from adder.pricing.cost import max_tolerable_p_fail
        assert max_tolerable_p_fail(OPUS, OPUS, ctx_tokens=0, est_out_tokens=0) == 0.0

    def test_a_gap_exactly_at_the_ttl_has_not_expired(self):
        """`gap > secs`, not `>=`: a cache idle for exactly its lifetime is alive."""
        at, _, _ = choose_ttl(100_000, OPUS, turns=10, gap_seconds=300)
        past, _, _ = choose_ttl(100_000, OPUS, turns=10, gap_seconds=301)
        assert at == "5m" and past == "1h"

    def test_one_turn_never_stages_a_fan_out(self):
        """`n < 2`: there is nothing to stagger against."""
        _, _, d = fanout_cost(1, 200_000, OPUS)
        assert not d.ok

    def test_two_calls_do_stage(self):
        _, _, d = fanout_cost(2, 200_000, OPUS)
        assert d.ok and d.saving > 0

    def test_effort_declines_a_move_that_saves_nothing(self):
        """`delta <= 0`: the same level is not a reduction."""
        _, d = effort_saving(1_000, OPUS, from_effort="high", to_effort="high")
        assert not d.ok
