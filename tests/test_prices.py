"""Capability metadata: the gates that stop a 'saving' from being a 400 error."""

from datetime import date

import pytest

from router.prices import (
    BATCH_MULT,
    CACHE_WRITE_MULT,
    MODELS,
    UnknownModel,
    UnsupportedSpeed,
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
        with pytest.raises(UnknownModel):
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
    def test_intro_rate_expires(self):
        assert rate(SONNET, date(2026, 8, 31)) == (2, 10)
        assert rate(SONNET, date(2026, 9, 1)) == (3, 15)
        assert intro_expiry(SONNET) == date(2026, 8, 31)
        assert intro_expiry(OPUS) is None

    def test_fast_mode_doubles_opus(self):
        assert rate(OPUS, speed="fast") == (10, 50)
        with pytest.raises(UnsupportedSpeed):
            rate(HAIKU, speed="fast")

    def test_one_hour_cache_write_costs_more_than_five_minute(self):
        assert CACHE_WRITE_MULT["1h"] > CACHE_WRITE_MULT["5m"]
        assert BATCH_MULT == 0.5

    def test_tier_order_is_cheapest_first(self):
        order = tier_order()
        assert order[0] == HAIKU
        assert MODELS[order[0]].base.inp <= MODELS[order[-1]].base.inp


class TestEffort:
    def test_haiku_rejects_effort(self):
        assert not supports_effort(HAIKU, "low")

    def test_opus_five_supports_the_full_ladder(self):
        for lvl in ("low", "medium", "high", "xhigh", "max"):
            assert supports_effort(OPUS, lvl)

    def test_opus_four_six_has_no_xhigh(self):
        assert not supports_effort("claude-opus-4-6", "xhigh")
