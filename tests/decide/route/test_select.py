"""Cross-provider selection: the gates, the arithmetic, and the honest labels.

The failure mode being tested for throughout is not "picks a slightly worse
model". It is "produces a confident recommendation from a number it should not
have trusted": an unrated model treated as good, an aggregator price treated as
a list price, an inline placement that the harness cannot actually perform.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from adder.decide.route.select import (
    UNUSABLE_GIVEN_LOSS,
    UNUSABLE_RANGE,
    Need,
    blend_p_fail,
    calibrate_unusable_given_loss,
    combos,
    cost_of,
    p_fail_from_elo,
    p_loss_from_elo,
    rank,
    ratings_overlap,
    sensitivity,
    win_probability,
)
from adder.pricing.catalog import Catalog, Entry

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


class TestBlendedFailureRate:
    """Measured tier history and the arena gap are different questions.

    The outcome log knows how often *this tier* escalates here. The arena knows
    how much weaker a *substitute* is than the model that tier names. Using
    either alone throws away the other; `blend_p_fail` composes them as
    independent failure modes.
    """

    def test_no_history_leaves_the_elo_estimate_alone(self):
        assert blend_p_fail(0.0, 0.2) == pytest.approx(0.2)

    def test_a_tier_that_always_escalates_stays_hopeless(self):
        """However good the substitute looks, the task still fails."""
        assert blend_p_fail(1.0, 0.0) == pytest.approx(1.0)

    def test_the_blend_is_never_lower_than_either_input(self):
        for m in (0.0, 0.15, 0.4, 0.9):
            for g in (0.0, 0.1, 0.5, 1.0):
                assert blend_p_fail(m, g) >= max(m, g) - 1e-9

    def test_it_stays_a_probability(self):
        assert 0.0 <= blend_p_fail(1.5, 2.0) <= 1.0
        assert blend_p_fail(-1.0, -1.0) == 0.0

    def test_measured_tier_history_raises_the_cascade_estimate(self):
        """A tier with a 30% escalation rate makes every substitute riskier."""
        plain = {c.shape: c for c in combos(Need(), cat=CAT, cheap="qwen4")}
        with_history = {c.shape: c for c in
                        combos(Need(), cat=CAT, cheap="qwen4", tier_p_fail=0.3)}
        assert with_history["cascade"].expected_cost > plain["cascade"].expected_cost

    def test_a_hard_override_still_wins_over_the_blend(self):
        """`measured_p_fail` replaces the estimate; `tier_p_fail` composes with it."""
        forced = {c.shape: c for c in
                  combos(Need(), cat=CAT, cheap="qwen4",
                         measured_p_fail=0.0, tier_p_fail=0.9)}
        assert "p_fail 0%" in forced["cascade"].detail


# Two models the arena cannot separate: 17 points apart, intervals overlapping.
NEAR = Entry(key="near", id="vendor/near", org="Vendor", license="Apache 2.0",
             inp=1.0, out=5.0, cache_read=0.1, cache_write=1.25,
             context=1_000_000, params=("tools",),
             elo={"webdev": 1673.0}, elo_lo={"webdev": 1662.0},
             elo_hi={"webdev": 1685.0}, verified=False)
OPUS_CI = Entry(key="claude-opus-5", id="anthropic/claude-opus-5", org="Anthropic",
                license="Proprietary", inp=5.0, out=25.0, cache_read=0.5,
                cache_write=6.25, context=1_000_000, params=("tools", "reasoning"),
                elo={"webdev": 1690.0}, elo_lo={"webdev": 1681.0},
                elo_hi={"webdev": 1701.0}, verified=True, modalities=("text", "image"))


class TestConfidenceIntervals:
    """A 17-point lead between two overlapping intervals is not a lead.

    On the live board the 95% half-width at the top is about 10 points, so the
    gap between the first and second model is inside the noise. Deriving a
    confident preference rate from it is inventing precision the source itself
    declines to claim.
    """

    def test_overlapping_intervals_are_detected(self):
        assert ratings_overlap(NEAR, OPUS_CI)

    def test_clearly_separated_ratings_do_not_overlap(self):
        far = replace(NEAR, elo={"webdev": 1400.0}, elo_lo={"webdev": 1390.0},
                      elo_hi={"webdev": 1410.0})
        assert not ratings_overlap(far, OPUS_CI)

    def test_a_missing_interval_is_not_an_overlap_claim(self):
        assert not ratings_overlap(Entry(key="x", id="x", elo={"webdev": 1690.0}),
                                   OPUS_CI)

    def test_the_conservative_bound_never_flatters_the_substitute(self):
        point = p_loss_from_elo(NEAR, OPUS_CI, conservative=False)
        bound = p_loss_from_elo(NEAR, OPUS_CI)
        assert bound > point

    def test_the_bound_compares_worst_case_against_best_case(self):
        expected = 1 - win_probability(NEAR.elo_lo["webdev"], OPUS_CI.elo_hi["webdev"])
        assert p_loss_from_elo(NEAR, OPUS_CI) == pytest.approx(expected)

    def test_the_correction_is_modest_not_dramatic(self):
        """Worth stating honestly: this moves the estimate by points, not halves."""
        point = p_loss_from_elo(NEAR, OPUS_CI, conservative=False)
        assert p_loss_from_elo(NEAR, OPUS_CI) - point < 0.10

    def test_a_ranking_says_when_the_arena_cannot_separate_two_models(self):
        cat = Catalog([OPUS_CI, NEAR])
        pick = {p.entry.key: p for p in rank(Need(), cat=cat)}["near"]
        assert any("cannot separate" in r for r in pick.reasons)


class TestEffortVariantDisclosure:
    def test_a_rating_earned_at_max_effort_is_flagged_against_default_pricing(self):
        """The arena ranks efforts separately; the price table has one price."""
        cat = Catalog([OPUS_CI, replace(NEAR, rating_variant="vendor/near-max")])
        pick = {p.entry.key: p for p in rank(Need(), cat=cat)}["near"]
        assert any("higher reasoning effort" in w for w in pick.warnings)
        assert any("vendor/near-max" in r for r in pick.reasons)

    def test_the_disclosure_is_one_line_however_many_rows_carry_it(self):
        """A caveat repeated per model is noise; the row keeps the specifics."""
        cat = Catalog([OPUS_CI,
                       replace(NEAR, rating_variant="vendor/near-max"),
                       replace(NEAR, key="near2", id="vendor/near2",
                               rating_variant="vendor/near2-max")])
        picks = rank(Need(), cat=cat)
        distinct = {w for p in picks for w in p.warnings if "reasoning effort" in w}
        assert len(distinct) == 1

    def test_an_unvaried_rating_is_not_flagged(self):
        cat = Catalog([OPUS_CI, replace(NEAR, rating_variant="near")])
        pick = {p.entry.key: p for p in rank(Need(), cat=cat)}["near"]
        assert not any("reasoning effort" in w for w in pick.warnings)


class TestCacheMissCorrection:
    """`adder carry` measures the realised read multiplier; this consumes it."""

    def test_the_default_changes_nothing(self):
        base = cost_of(OPUS, Need(est_read_tokens=50_000, remaining_turns=200))
        same = cost_of(OPUS, Need(est_read_tokens=50_000, remaining_turns=200,
                                  cache_miss_correction=1.0))
        assert base.inline == same.inline

    def test_a_measured_miss_rate_raises_the_carry_term(self):
        base = cost_of(OPUS, Need(est_read_tokens=50_000, remaining_turns=200))
        worse = cost_of(OPUS, Need(est_read_tokens=50_000, remaining_turns=200,
                                   cache_miss_correction=1.15))
        assert worse.carry == pytest.approx(base.carry * 1.15)

    def test_it_scales_each_vendor_published_rate_not_a_borrowed_one(self):
        """Miss *frequency* is a workload property; the *rate* stays the vendor's."""
        a = cost_of(OSS, Need(est_read_tokens=50_000, remaining_turns=200,
                              cache_miss_correction=2.0))
        b = cost_of(OSS, Need(est_read_tokens=50_000, remaining_turns=200))
        assert a.carry == pytest.approx(b.carry * 2.0)
        assert b.carry == pytest.approx(50_000 * OSS.cache_read * 200 / 1e6)


class TestResidencyPlumbing:
    """The second half of the fitted carry model: how many re-reads happen.

    On this machine no compaction lands inside a typical horizon, so the
    corrected count equals the turn count and nothing moves. The plumbing still
    has to exist and be exercised, because the workloads where it does move are
    exactly the long sessions that hold the spend.
    """

    def test_the_default_is_every_remaining_turn(self):
        base = cost_of(OPUS, Need(est_read_tokens=10_000, remaining_turns=200))
        pinned = cost_of(OPUS, Need(est_read_tokens=10_000, remaining_turns=200,
                                    expected_reads=200.0))
        assert base.carry == pytest.approx(pinned.carry)

    def test_a_token_that_does_not_survive_costs_less_to_carry(self):
        base = cost_of(OPUS, Need(est_read_tokens=10_000, remaining_turns=200))
        short = cost_of(OPUS, Need(est_read_tokens=10_000, remaining_turns=200,
                                   expected_reads=16.0))
        assert short.carry == pytest.approx(base.carry * 16 / 200)

    def test_the_two_corrections_are_independent(self):
        """Miss rate scales the price of a read; residency scales the count."""
        both = cost_of(OPUS, Need(est_read_tokens=10_000, remaining_turns=200,
                                  expected_reads=100.0, cache_miss_correction=2.0))
        neither = cost_of(OPUS, Need(est_read_tokens=10_000, remaining_turns=200))
        assert both.carry == pytest.approx(neither.carry * (100 / 200) * 2.0)


class TestPanelHonesty:
    def test_the_panel_shape_does_not_invent_a_quality_number(self):
        """Best-of-N needs independent failures. Runs of one model are not."""
        panel = next(c for c in combos(Need(), cat=CAT, cheap="qwen4")
                     if c.shape == "panel")
        assert panel.quality is None
        assert panel.expected_cost > 0
        assert "independent" in panel.assumption


class TestCalibratingThePrior:
    """`UNUSABLE_GIVEN_LOSS` converts a preference loss into a redo.

    It is the weakest link in the module -- it scales every cascade cost and
    every substitute verdict linearly, and it was chosen by judgement. But the
    outcome log records how often a tier really escalated, and the arena says
    how often that tier's model loses to the escalation target. The ratio is
    the constant, so wherever there is history it stops being a guess.
    """

    def test_it_is_the_ratio_of_observed_to_modelled(self):
        p_loss = p_loss_from_elo(HAIKU, OPUS, conservative=False)
        fitted, basis = calibrate_unusable_given_loss(HAIKU, OPUS, 0.30)
        assert fitted == pytest.approx(0.30 / p_loss)
        assert "fitted" in basis

    def test_a_model_compared_with_itself_falls_back_to_the_prior(self):
        """Bradley-Terry gives 0.5 at a zero gap, not 0.

        So this case cannot be caught by a small-quotient threshold: the fit
        would quietly divide the observed rate by a coin flip and report the
        result as measured. It has to be rejected by identity.
        """
        fitted, basis = calibrate_unusable_given_loss(OPUS, OPUS, 0.30)
        assert fitted == UNUSABLE_GIVEN_LOSS
        assert "nothing to escalate to" in basis

    def test_a_gap_inside_the_error_bars_is_not_fittable(self):
        """Dividing by a difference the arena does not claim amplifies noise."""
        near = replace(OPUS, key="near", id="v/near", elo={"webdev": 1685.0},
                       elo_lo={"webdev": 1670.0}, elo_hi={"webdev": 1700.0})
        ref = replace(OPUS, elo={"webdev": 1690.0}, elo_lo={"webdev": 1681.0},
                      elo_hi={"webdev": 1701.0})
        fitted, basis = calibrate_unusable_given_loss(near, ref, 0.30)
        assert fitted == UNUSABLE_GIVEN_LOSS and "error bars" in basis

    def test_an_unrated_model_falls_back_to_the_prior(self):
        fitted, basis = calibrate_unusable_given_loss(UNRATED, OPUS, 0.30)
        assert fitted == UNUSABLE_GIVEN_LOSS and "prior" in basis

    def test_the_fit_stays_a_probability(self):
        """A tier that escalates more often than it loses comparisons happens."""
        fitted, _ = calibrate_unusable_given_loss(HAIKU, OPUS, 1.0)
        assert 0.0 < fitted <= 1.0

    def test_a_calibrated_prior_changes_the_failure_estimate(self):
        low = p_fail_from_elo(OSS, OPUS, unusable_given_loss=0.15)
        high = p_fail_from_elo(OSS, OPUS, unusable_given_loss=0.60)
        assert high > low


class TestSensitivityToTheUnmeasuredPrior:
    """The useful question about an invented constant is whether it matters."""

    def test_a_wide_cost_gap_is_reported_as_stable(self):
        cheap = replace(OSS, inp=0.02, out=0.1)
        cat = Catalog([OPUS, cheap])
        got = sensitivity(Need(), cat=cat, cheap="qwen4")
        assert got.stable
        assert "stable" in got.render()

    def test_a_flip_inside_the_range_is_found_and_located(self):
        """When the winner changes mid-range the output must say so, not average."""
        # Priced so the cascade's escalation term crosses the single-pass cost
        # inside the plausible band.
        knife = replace(OSS, key="knife", id="v/knife", inp=2.6, out=13.0,
                        elo={"webdev": 1500.0})
        cat = Catalog([OPUS, knife])
        got = sensitivity(Need(est_read_tokens=60_000, remaining_turns=100),
                          cat=cat, cheap="knife", steps=25)
        if not got.stable:
            assert got.flips and UNUSABLE_RANGE[0] < got.flips[0][0] < UNUSABLE_RANGE[1]
            assert "UNSTABLE" in got.render()
            assert "coin flip" in got.render()

    def test_the_sweep_covers_the_whole_plausible_band(self):
        cat = Catalog([OPUS, OSS])
        got = sensitivity(Need(), cat=cat, cheap="qwen4")
        assert (got.low, got.high) == UNUSABLE_RANGE

    def test_it_reports_both_ends_not_just_a_verdict(self):
        cat = Catalog([OPUS, OSS])
        got = sensitivity(Need(), cat=cat, cheap="qwen4")
        assert got.winner_low and got.winner_high


class TestAnUnknownContextWindowIsNotAPass:
    """`registry.ModelSpec.fits` states the rule this gate applies.

    "Unknown context is False, not True. A feasibility gate that passes because
    it does not know is not a gate." This copy of it had the opposite default,
    so all 53 bundled entries with no published window were quoted as able to
    hold any session at all -- and inline is the placement whose price is the
    reason to pick one.
    """

    def _entry(self, **kw):
        from adder.pricing.catalog import Entry

        base = {"key": "x", "id": "vendor/x", "org": "vendor",
                "inp": 1.0, "out": 5.0, "context": None}
        base.update(kw)
        return Entry.from_json(base)

    def _need(self):
        from adder.decide.route.select import Need

        return Need(context_tokens=500_000, remaining_turns=100,
                    est_read_tokens=10_000, harness="any")

    def test_no_published_window_is_infeasible_inline(self):
        from adder.decide.route.select import cost_of

        c = cost_of(self._entry(), self._need(), session=self._entry(context=1_000_000))
        assert not c.inline_feasible
        assert "no published context window" in c.inline_blocked

    def test_a_window_that_fits_is_feasible(self):
        from adder.decide.route.select import cost_of

        e = self._entry(context=1_000_000)
        assert cost_of(e, self._need(), session=e).inline_feasible

    def test_a_window_that_does_not_fit_says_so(self):
        from adder.decide.route.select import cost_of

        e = self._entry(context=200_000)
        c = cost_of(e, self._need(), session=self._entry(context=1_000_000))
        assert not c.inline_feasible and "200,000" in c.inline_blocked

    def test_a_model_with_no_placement_at_all_costs_infinity(self):
        from adder.decide.route.select import Costed

        c = Costed(self._entry(), inline=1.0, delegated=2.0, subagent=1.0, carry=0.0,
                   inline_feasible=False, delegate_feasible=False)
        assert c.best == float("inf")
        assert not c.usable
