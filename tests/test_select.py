"""Cross-provider selection: the gates, the arithmetic, and the honest labels.

The failure mode being tested for throughout is not "picks a slightly worse
model". It is "produces a confident recommendation from a number it should not
have trusted": an unrated model treated as good, an aggregator price treated as
a list price, an inline placement that the harness cannot actually perform.
"""

import pytest

from router.catalog import Catalog, Entry
from router.select import (
    UNUSABLE_GIVEN_LOSS,
    Need,
    combos,
    cost_of,
    p_fail_from_elo,
    p_loss_from_elo,
    rank,
    win_probability,
)

OPUS = Entry(key="claude-opus-5", id="anthropic/claude-opus-5", org="Anthropic",
             license="Proprietary", inp=5.0, out=25.0, cache_read=0.5,
             cache_write=6.25, context=1_000_000, params=("tools", "reasoning"),
             elo={"webdev": 1690.0}, verified=True, modalities=("text", "image"))
HAIKU = Entry(key="claude-haiku-4.5", id="anthropic/claude-haiku-4.5", org="Anthropic",
              license="Proprietary", inp=1.0, out=5.0, cache_read=0.1,
              cache_write=1.25, context=200_000, params=("tools",),
              elo={"webdev": 1420.0}, verified=True)
OSS = Entry(key="qwen4", id="qwen/qwen4-max", org="Alibaba", license="Apache 2.0",
            inp=1.2, out=6.0, cache_read=0.12, cache_write=1.5,
            context=262_144, params=("tools", "reasoning"),
            elo={"webdev": 1600.0}, verified=False)
UNRATED = Entry(key="mystery", id="who/mystery-1", org="Nobody", license="MIT",
                inp=0.05, out=0.2, context=1_000_000, params=("tools",),
                verified=False)
NOCACHE = Entry(key="nocache", id="vendor/nocache", org="Vendor", license="Proprietary",
                inp=1.0, out=4.0, context=1_000_000, params=("tools",),
                elo={"webdev": 1600.0}, verified=False)

CAT = Catalog([OPUS, HAIKU, OSS, UNRATED, NOCACHE])


class TestEloArithmetic:
    def test_equal_ratings_are_a_coin_flip(self):
        assert win_probability(1500, 1500) == pytest.approx(0.5)

    def test_four_hundred_points_is_ten_to_one(self):
        assert win_probability(1900, 1500) == pytest.approx(10 / 11, abs=1e-6)

    def test_preference_loss_and_failure_are_not_the_same_number(self):
        """A model that loses half its comparisons does not fail half its tasks."""
        loss = p_loss_from_elo(OSS, OPUS)
        fail = p_fail_from_elo(OSS, OPUS)
        assert fail == pytest.approx(loss * UNUSABLE_GIVEN_LOSS)
        assert fail < loss

    def test_difficulty_widens_the_gap(self):
        easy = p_fail_from_elo(HAIKU, OPUS, difficulty=0.4)
        hard = p_fail_from_elo(HAIKU, OPUS, difficulty=1.4)
        assert hard > easy

    def test_no_rating_means_no_estimate_rather_than_a_default(self):
        assert p_fail_from_elo(UNRATED, OPUS) is None


class TestCosting:
    def test_the_carry_term_dominates_a_long_session(self):
        """Re-reading admitted tokens outweighs generating them by turn 100."""
        c = cost_of(OPUS, Need(est_read_tokens=50_000, remaining_turns=200))
        assert c.carry > 0.5 * c.inline

    def test_delegation_wins_when_the_session_still_has_far_to_run(self):
        near_end = cost_of(OPUS, Need(est_read_tokens=50_000, remaining_turns=1))
        early = cost_of(OPUS, Need(est_read_tokens=50_000, remaining_turns=300))
        assert early.placement == "delegate"
        assert near_end.inline < early.inline

    def test_switching_a_warm_session_pays_to_rebuild_the_prefix(self):
        """The prompt cache is model-scoped; a switch re-writes the whole prefix."""
        need = Need(context_tokens=400_000, remaining_turns=50, harness="any")
        switched = cost_of(HAIKU, need, session=OPUS)
        assert switched.inline > 0.4 * 400_000 * HAIKU.cache_write / 1e6

    def test_a_provider_without_published_cache_rates_is_flagged(self):
        assert cost_of(NOCACHE, Need()).assumed_cache
        assert not cost_of(OPUS, Need()).assumed_cache

    def test_a_model_that_cannot_hold_the_session_is_not_priced_inline(self):
        c = cost_of(HAIKU, Need(context_tokens=500_000, harness="any"))
        assert not c.inline_feasible
        assert c.best == c.delegated and c.placement == "delegate"


class TestRanking:
    def test_unrated_models_are_excluded_by_default(self):
        """The cheapest thing in the catalog is usually the least known thing."""
        ids = {p.entry.key for p in rank(Need(), cat=CAT)}
        assert "mystery" not in ids
        assert "mystery" in {p.entry.key for p in
                             rank(Need(), cat=CAT, include_unrated=True)}

    def test_the_quality_floor_filters_on_rating_not_price(self):
        picks = rank(Need(), cat=CAT, quality_floor=1500)
        assert {p.entry.key for p in picks} <= {"claude-opus-5", "qwen4", "nocache"}

    def test_claude_code_cannot_put_a_third_party_model_in_the_session(self):
        """A GPT or open-weight model can be a subagent; it cannot be the session."""
        picks = {p.entry.key: p for p in rank(Need(harness="claude-code"), cat=CAT)}
        assert picks["qwen4"].placement == "delegate"
        assert any("subagent only" in r for r in picks["qwen4"].reasons)

    def test_any_harness_allows_a_third_party_model_inline(self):
        """Two things have to be true for a non-Claude model to run inline:
        the harness must route natively, and delegating must not pay off.
        Delegation stops paying when the work cannot be compressed on the way
        back -- a summary the same size as the read is not a summary."""
        need = Need(harness="any", session_model="qwen4", remaining_turns=1,
                    est_read_tokens=2_000, summary_tokens=2_000)
        picks = {p.entry.key: p for p in rank(need, cat=CAT)}
        assert picks["qwen4"].placement == "inline"

    def test_delegation_wins_whenever_the_summary_is_smaller_than_the_read(self):
        need = Need(harness="any", session_model="qwen4", remaining_turns=1,
                    est_read_tokens=2_000)
        assert {p.entry.key: p for p in rank(need, cat=CAT)}["qwen4"].placement == "delegate"

    def test_switching_the_session_model_is_priced_into_the_placement(self):
        """Even on `any`, moving a warm 100K session to another vendor rebuilds it."""
        need = Need(harness="any", session_model="claude-opus-5", remaining_turns=1,
                    context_tokens=100_000, est_read_tokens=2_000)
        assert {p.entry.key: p for p in rank(need, cat=CAT)}["qwen4"].placement == "delegate"

    def test_open_weights_only_drops_the_proprietary_models(self):
        keys = {p.entry.key for p in rank(Need(open_weights_only=True), cat=CAT)}
        assert keys == {"qwen4"}

    def test_unverified_prices_are_labelled_every_time(self):
        picks = {p.entry.key: p for p in rank(Need(), cat=CAT)}
        assert any("aggregator" in w for w in picks["qwen4"].warnings)
        assert not any("aggregator" in w for w in picks["claude-opus-5"].warnings)

    def test_refusing_unverified_prices_leaves_only_first_party(self):
        keys = {p.entry.key for p in rank(Need(allow_unverified=False), cat=CAT)}
        assert keys and all(k.startswith("claude") for k in keys)

    def test_a_model_too_small_for_the_task_is_not_a_candidate(self):
        keys = {p.entry.key for p in rank(Need(est_read_tokens=400_000), cat=CAT)}
        assert "claude-haiku-4.5" not in keys and "qwen4" not in keys

    def test_tool_use_is_required_by_default(self):
        no_tools = Entry(key="chatty", id="v/chatty", org="V", inp=0.1, out=0.1,
                         context=1_000_000, elo={"webdev": 1650.0})
        cat = Catalog([OPUS, no_tools])
        assert "chatty" not in {p.entry.key for p in rank(Need(), cat=cat)}
        assert "chatty" in {p.entry.key for p in rank(Need(needs_tools=False), cat=cat)}

    def test_missing_reference_is_an_error_not_a_silent_default(self):
        with pytest.raises(KeyError):
            rank(Need(reference="does-not-exist"), cat=CAT)


class TestCombinations:
    def test_a_cascade_costs_the_cheap_run_plus_the_escalations(self):
        need = Need(est_read_tokens=40_000, remaining_turns=100)
        plans = combos(need, cat=CAT, cheap="qwen4", measured_p_fail=0.2,
                       detection=1.0)
        by_shape = {c.shape: c for c in plans}
        low = next(c for c in plans if c.shape == "single" and c.models == [OSS.id])
        strong = next(c for c in plans if c.shape == "single" and c.models == [OPUS.id])
        assert by_shape["cascade"].expected_cost == pytest.approx(
            low.expected_cost + 0.2 * strong.expected_cost)

    def test_perfect_detection_makes_a_cascade_as_good_as_the_strong_model(self):
        plans = {c.shape: c for c in combos(Need(), cat=CAT, cheap="qwen4",
                                            detection=1.0)}
        assert plans["cascade"].quality == pytest.approx(OPUS.rating())

    def test_undetected_failures_are_priced_into_the_quality(self):
        """The assumption that decides a cascade is detection, not price."""
        good = {c.shape: c for c in combos(Need(), cat=CAT, cheap="qwen4",
                                           detection=1.0)}["cascade"]
        bad = {c.shape: c for c in combos(Need(), cat=CAT, cheap="qwen4",
                                          detection=0.3)}["cascade"]
        assert bad.quality < good.quality
        assert bad.expected_cost < good.expected_cost   # cheaper *because* worse

    def test_every_plan_states_the_assumption_that_decides_it(self):
        for c in combos(Need(), cat=CAT, cheap="qwen4"):
            assert c.assumption

    def test_measured_p_fail_overrides_the_elo_estimate(self):
        modelled = {c.shape: c for c in combos(Need(), cat=CAT, cheap="qwen4")}
        measured = {c.shape: c for c in combos(Need(), cat=CAT, cheap="qwen4",
                                               measured_p_fail=0.0)}
        assert measured["cascade"].expected_cost < modelled["cascade"].expected_cost

    def test_plans_are_ranked_by_expected_cost(self):
        costs = [c.expected_cost for c in combos(Need(), cat=CAT, cheap="qwen4")]
        assert costs == sorted(costs)

    def test_a_single_strong_pass_is_always_on_the_menu(self):
        """Doing it properly once must stay comparable, not be optimised away."""
        plans = combos(Need(), cat=CAT, cheap="qwen4")
        assert any(c.shape == "single" and c.models == [OPUS.id] for c in plans)
