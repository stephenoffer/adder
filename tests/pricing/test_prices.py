"""Capability metadata: the gates that stop a 'saving' from being a 400 error."""
from __future__ import annotations

from datetime import date

import pytest

from adder.pricing.prices import (
    BATCH_MULT,
    CACHE_WRITE_MULT,
    MODELS,
    UnknownModelError,
    UnsupportedSpeedError,
    cache_min,
    caches,
    cheapest_that_fits,
    context_limit,
    fits,
    intro_expiry,
    is_known,
    rate,
    resolve,
    supports_effort,
    tier_order,
)

OPUS, HAIKU, SONNET = "claude-opus-5", "claude-haiku-4-5", "claude-sonnet-5"


class TestResolution:
    def test_longest_prefix_wins(self):
        """claude-sonnet-4-6 must not resolve as claude-sonnet-5."""
        assert resolve("claude-sonnet-4-6-20251101").id == "claude-sonnet-4-6"

    def test_claude_code_context_suffix(self):
        """Claude Code writes variants like claude-opus-5[1m]."""
        assert resolve("claude-opus-5[1m]").id == "claude-opus-5"

    def test_unknown_model_raises(self):
        with pytest.raises(UnknownModelError):
            resolve("gpt-4")
        assert not is_known("gpt-4")


class TestContextLimits:
    def test_haiku_cannot_hold_the_median_session(self):
        """544K is the measured median peak context; Haiku holds 200K."""
        assert context_limit(HAIKU) == 200_000
        assert not fits(HAIKU, 544_000)

    def test_opus_holds_a_million(self):
        assert fits(OPUS, 999_000) and not fits(OPUS, 1_000_001)

    def test_cheapest_that_fits_skips_models_that_cannot(self):
        assert cheapest_that_fits(50_000) == HAIKU
        assert cheapest_that_fits(544_000) == SONNET

    def test_capability_floor_is_respected(self):
        assert cheapest_that_fits(10_000, at_least=OPUS).startswith("claude-opus")


class TestCacheMinimums:
    def test_minimum_is_not_monotonic_across_generations(self):
        """Opus 5 caches a 512-token prefix; Opus 4.6 and Haiku need 4096."""
        assert cache_min(OPUS) == 512
        assert cache_min("claude-opus-4-6") == 4096
        assert cache_min(HAIKU) == 4096

    def test_short_brief_silently_does_not_cache_on_haiku(self):
        assert caches(OPUS, 1_000)
        assert not caches(HAIKU, 1_000)


class TestRates:
    def test_intro_rate_expires(self, scheduled_intro):
        assert rate(SONNET, date(2026, 8, 31)) == (2, 10)
        assert rate(SONNET, date(2026, 9, 1)) == (3, 15)
        assert intro_expiry(SONNET) == date(2026, 8, 31)
        assert intro_expiry(OPUS) is None

    def test_fast_mode_doubles_opus(self):
        assert rate(OPUS, speed="fast") == (10, 50)
        with pytest.raises(UnsupportedSpeedError):
            rate(HAIKU, speed="fast")

    def test_one_hour_cache_write_costs_more_than_five_minute(self):
        assert CACHE_WRITE_MULT["1h"] > CACHE_WRITE_MULT["5m"]
        assert BATCH_MULT == 0.5

    def test_tier_order_is_cheapest_first(self):
        order = tier_order()
        assert order[0] == HAIKU
        assert MODELS[order[0]].base.inp <= MODELS[order[-1]].base.inp


class TestPointReleasesHaveTheirOwnRows:
    """`claude-opus-5-5` resolved by prefix to `claude-opus-5` and billed at its
    rates: 1.25x on input and output, 2.5x on cache reads."""

    @pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-opus-5-5[1m]"])
    def test_opus_five_five_is_not_opus_five(self, model):
        assert resolve(model).id == "claude-opus-5-5"
        assert rate(model, date(2026, 9, 1)) == (4, 20)

    def test_a_dated_opus_five_is_still_opus_five(self):
        assert resolve("claude-opus-5-20260401").id == "claude-opus-5"

    @pytest.mark.parametrize("model,per_m", [
        ("claude-opus-5-5", 0.20), ("claude-fable-5-1", 0.25), ("claude-opus-5", 0.50),
        ("claude-mythos-5-1", 0.25), ("claude-mythos-5", 1.00),
    ])
    def test_cache_reads_bill_at_the_published_rate(self, model, per_m):
        from adder.pricing.cost import turn_cost
        assert turn_cost(model, cache_read=1_000_000) == pytest.approx(per_m)

    @pytest.mark.parametrize("model", ["claude-sonnet-5-1", "claude-opus-5-7[1m]",
                                       "claude-haiku-4-5-1"])
    def test_an_unlisted_point_release_is_unknown_not_its_parent(self, model):
        with pytest.raises(UnknownModelError):
            resolve(model)


class TestSonnetFiveKeptItsLaunchPrice:
    """The announced revert to $3/$15 on 2026-09-01 was cancelled; modelling it
    billed every Sonnet 5 turn from that date at 1.5x."""

    @pytest.mark.parametrize("on", [date(2026, 8, 15), date(2026, 9, 1), date(2027, 1, 1)])
    def test_it_is_two_and_ten_on_either_side_of_the_old_date(self, on):
        assert rate(SONNET, on) == (2, 10)

    def test_it_has_no_expiry_to_warn_about(self):
        assert intro_expiry(SONNET) is None


class TestEffort:
    def test_haiku_rejects_effort(self):
        assert not supports_effort(HAIKU, "low")

    def test_opus_five_supports_the_full_ladder(self):
        for lvl in ("low", "medium", "high", "xhigh", "max"):
            assert supports_effort(OPUS, lvl)

    def test_opus_four_six_has_no_xhigh(self):
        assert not supports_effort("claude-opus-4-6", "xhigh")
