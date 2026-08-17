"""The same arithmetic, run against four providers that bill differently.

These are the regression guards for model-agnosticism. Every one of them fails
if some future change reintroduces a constant where a provider lookup belongs --
which is how the cost model started, with Anthropic's 0.10x read and 1.25x
write applied to whatever model id it was handed.

The Claude numbers here are also a pin: making the tool work for other vendors
was not allowed to move a single figure for the vendor it was measured on.
"""

from __future__ import annotations

import pytest

from adder.pricing import providers
from adder.pricing.cost import (
    M,
    Rates,
    admitted_token_cost,
    batch_mult,
    cache_storage_cost,
    choose_ttl,
    fanout_cost,
    marginal_turn_cost,
    turn_cost,
)
from adder.pricing.registry import ModelSpec, resolve

OPUS = "claude-opus-5"
GPT = "gpt-5"
GEMINI = "gemini-3-flash"
DEEPSEEK = "deepseek-v4-pro"

ALL = [OPUS, GPT, GEMINI, DEEPSEEK]


def no_cache_spec(inp=2.0, out=8.0, context=200_000) -> ModelSpec:
    """A model on an endpoint that does not cache. The conservative default."""
    return ModelSpec(id="uncached-1", org="Nobody", provider=providers.UNKNOWN,
                     inp=inp, out=out, context=context)


class TestClaudeNumbersDidNotMove:
    """Making this work for other vendors must not change the vendor it was
    measured on. These are the figures the README and docs quote."""

    def test_a_turn_prices_exactly_as_before(self):
        got = turn_cost(OPUS, uncached_in=2_000, cache_read=500_000,
                        cache_write=10_000, out=800)
        expect = (2_000 * 5 + 500_000 * 0.5 + 10_000 * 6.25 + 800 * 25) / M
        assert got == pytest.approx(expect)

    def test_the_carry_term_is_unchanged(self):
        """10K tokens over 1,000 turns on Opus 5: ~$5.06."""
        assert admitted_token_cost(10_000, OPUS, 1_000) == pytest.approx(5.0625)

    def test_the_write_premium_is_still_real(self):
        r = Rates.for_model(OPUS)
        assert r.cache_write == pytest.approx(r.inp * 1.25)
        assert r.cache_read == pytest.approx(r.inp * 0.10)

    def test_the_one_hour_ttl_still_costs_twice(self):
        r5 = Rates.for_model(OPUS, ttl="5m")
        r1h = Rates.for_model(OPUS, ttl="1h")
        assert r1h.cache_write == pytest.approx(r5.cache_write * 1.6)   # 2.00/1.25


class TestAutomaticCachingHasNoWritePremium:
    @pytest.mark.parametrize("model", [GPT, GEMINI, DEEPSEEK])
    def test_a_cache_write_is_billed_as_ordinary_input(self, model):
        """Applying Anthropic's 1.25x here invents a quarter of the write side
        of the bill."""
        r = Rates.for_model(model)
        assert r.cache_write == pytest.approx(r.inp)

    @pytest.mark.parametrize("model", [GPT, GEMINI, DEEPSEEK])
    def test_and_a_turn_costs_less_than_the_anthropic_shaped_estimate(self, model):
        r = Rates.for_model(model)
        got = turn_cost(model, cache_write=100_000)
        anthropic_shaped = 100_000 * r.inp * 1.25 / M
        assert got < anthropic_shaped
        assert got == pytest.approx(100_000 * r.inp / M)


class TestNoCacheIsTenTimesTheCarry:
    def test_the_carry_term_is_priced_at_full_input_rate(self):
        """The correction that matters most, and the one that used to point the
        wrong way: borrowing 0.10x for an endpoint with no cache understates
        carry tenfold, and carry is ~76% of spend."""
        spec = no_cache_spec(inp=2.0)
        assert spec.cache_read_rate() == 2.0
        assert spec.cache_read_rate() == spec.rate().inp

    def test_a_marginal_turn_is_an_order_of_magnitude_dearer(self):
        """Same price per token, different cache: the marginal turn differs 10x.

        This is the number on screen when someone decides whether to keep
        going, so getting it wrong is not academic.
        """
        cached = Rates(inp=2.0, out=8.0, cache_read=0.2, cache_write=2.5)
        uncached = Rates(inp=2.0, out=8.0, cache_read=2.0, cache_write=2.0)
        ctx = 500_000
        a = (ctx * cached.cache_read + 800 * cached.out) / M
        b = (ctx * uncached.cache_read + 800 * uncached.out) / M
        assert b > 9 * a

    def test_marginal_turn_cost_uses_the_providers_read_rate(self):
        for model in ALL:
            r = Rates.for_model(model)
            got = marginal_turn_cost(400_000, 800, model)
            assert got == pytest.approx((400_000 * r.cache_read + 800 * r.out) / M)


class TestTtlIsOnlyALeverWhereItExists:
    def test_anthropic_gets_a_real_choice(self):
        ttl, saving, why = choose_ttl(500_000, OPUS, turns=100, gap_seconds=900)
        assert ttl == "1h"
        assert saving > 0
        assert "5m" in why

    @pytest.mark.parametrize("model", [GPT, GEMINI, DEEPSEEK])
    def test_an_automatic_provider_is_told_there_is_nothing_to_choose(self, model):
        """A dollar figure for a change nobody can make is worse than saying so."""
        _ttl, saving, why = choose_ttl(500_000, model, turns=100, gap_seconds=900)
        assert saving == 0.0
        assert "not " in why and "selectable" in why

    def test_and_the_reason_names_the_real_lever_instead(self):
        _, _, why = choose_ttl(500_000, GPT, turns=100, gap_seconds=9_000)
        assert "turn latency" in why


class TestFanoutAdviceMatchesTheProvider:
    def test_staggering_pays_where_writes_carry_a_premium(self):
        _, _, d = fanout_cost(5, 50_000, OPUS)
        assert d.ok and d.saving > 0

    def test_a_prefix_under_the_minimum_still_caches_nothing(self):
        _, _, d = fanout_cost(5, 100, "claude-haiku-4-5")
        assert not d.ok
        assert "cache minimum" in d.reason


class TestBatchTiers:
    @pytest.mark.parametrize("model,expect", [(OPUS, 0.50), (GPT, 0.50),
                                              (GEMINI, 0.50)])
    def test_a_published_batch_tier_is_used(self, model, expect):
        assert batch_mult(model) == pytest.approx(expect)

    def test_a_provider_with_no_batch_tier_offers_no_discount(self):
        """Recommending "batch it" where there is no batch API is a suggestion
        to use a product that does not exist."""
        assert batch_mult(DEEPSEEK) == 1.0


class TestStorageIsOnlyBilledWhereItIsBilled:
    def test_google_charges_for_an_idle_hour(self):
        """The one cost term driven by elapsed time rather than tokens moved.
        No amount of prompt discipline reduces it."""
        assert cache_storage_cost(500_000, GEMINI, hours=1.0) > 0

    @pytest.mark.parametrize("model", [OPUS, GPT, DEEPSEEK])
    def test_nobody_else_does(self, model):
        assert cache_storage_cost(500_000, model, hours=10.0) == 0.0


class TestEveryModelPricesAtAll:
    @pytest.mark.parametrize("model", ALL)
    def test_the_cost_model_answers_for_any_provider(self, model):
        """The whole point. Before the registry these raised `UnknownModel`
        for everything that was not Claude."""
        assert turn_cost(model, uncached_in=1_000, out=100) > 0
        assert admitted_token_cost(10_000, model, 100) > 0
        assert marginal_turn_cost(100_000, 500, model) > 0

    @pytest.mark.parametrize("model", ALL)
    def test_rates_are_internally_consistent(self, model):
        r = Rates.for_model(model)
        assert 0 < r.cache_read <= r.inp, "a cache read never costs more than input"
        assert r.cache_write >= r.inp * 0.99, "a write never costs less than input"
        assert r.out > 0

    @pytest.mark.parametrize("model", ALL)
    def test_a_spec_can_always_explain_where_its_rates_came_from(self, model):
        assert resolve(model).rate_provenance
