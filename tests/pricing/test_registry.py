"""Resolving any model, from any provider, to something the cost model can use.

Two failure modes dominate this file, and they are opposites:

* Refusing to price a model that is perfectly well known, which is what the
  Claude-only table did to every other vendor.
* Pricing a model that is *not* known, off whatever nearby key happened to
  match. That one is worse: an unknown model reported as unknown is a caveat
  in the output, but an unknown model priced off a wildcard is a wrong number
  presented as a measurement.
"""

from __future__ import annotations

import json

import pytest

from adder.pricing import providers, registry
from adder.pricing.catalog import SCHEMA

OPUS, HAIKU, SONNET = "claude-opus-5", "claude-haiku-4-5", "claude-sonnet-5"


@pytest.fixture(autouse=True)
def _clean_registry_cache():
    """Resolution is memoized on the catalog layer sources; tests move those."""
    registry.reset_cache()
    yield
    registry.reset_cache()


class TestFirstPartyStillWins:
    def test_claude_resolves_from_the_hand_checked_table(self):
        s = registry.resolve(OPUS)
        assert s.first_party and s.verified
        assert (s.inp, s.out) == (5, 25)
        assert s.source == "first-party:prices.py"

    def test_the_intro_rate_still_expires(self):
        """Only the first-party layer has a time dimension, and it keeps it."""
        from datetime import date
        assert registry.rate(SONNET, date(2026, 8, 31)) == (2, 10)
        assert registry.rate(SONNET, date(2026, 9, 1)) == (3, 15)

    def test_per_model_cache_minimums_are_not_flattened_to_a_provider_default(self):
        """The minimum is model-scoped and NOT monotonic across generations.

        A provider-level default would quietly say 1024 for all of them and
        make every sub-4K delegation on Haiku look like it caches when it does
        not.
        """
        assert registry.cache_min("claude-opus-5") == 512
        assert registry.cache_min("claude-haiku-4-5") == 4096
        assert registry.cache_min("claude-opus-4-6") == 4096

    def test_the_context_suffix_and_dated_ids_still_resolve(self):
        assert registry.resolve("claude-opus-5[1m]").id == "claude-opus-5"
        assert registry.resolve("claude-haiku-4-5-20251001").id == "claude-haiku-4-5"

    def test_sonnet_4_6_never_resolves_as_sonnet_5(self):
        assert registry.resolve("claude-sonnet-4-6-20260101").id == "claude-sonnet-4-6"


class TestOtherProviders:
    @pytest.mark.parametrize("model", ["gpt-5", "gemini-3-flash", "deepseek-v4-pro"])
    def test_a_non_claude_model_prices_instead_of_raising(self, model):
        """The whole point: the Claude-only table raised `UnknownModel` here."""
        s = registry.resolve(model)
        assert s.priced and not s.first_party

    def test_a_vendor_prefixed_routing_slug_resolves(self):
        assert registry.resolve("anthropic/claude-opus-4.6").id == "claude-opus-4-6"

    def test_cache_economics_come_from_the_right_provider(self):
        """Anthropic pays a write premium; OpenAI does not. Same arithmetic,
        different provider, and the difference is not a rounding error."""
        claude = registry.resolve(OPUS)
        gpt = registry.resolve("gpt-5")
        assert claude.cache_write_rate() > claude.rate().inp     # 1.25x premium
        assert gpt.cache_write_rate() == pytest.approx(gpt.rate().inp)


class TestNoCacheIsPricedAsNoCache:
    def test_re_reads_cost_full_input_rate_when_there_is_no_cache(self):
        """The correction that matters most in a long session.

        Borrowing Anthropic's 0.10x for a provider with no cache understates
        the carry term tenfold, and carry is ~76% of spend on the measured
        transcripts. That is not a bad recommendation, it is one pointed the
        wrong way.
        """
        spec = registry.ModelSpec(id="m", org="Nobody", provider=providers.UNKNOWN,
                           inp=2.0, out=8.0, context=100_000)
        assert spec.cache_read_rate() == 2.0
        assert spec.cache_write_rate() == 2.0
        assert not spec.caches(1_000_000)

    def test_a_published_cache_rate_is_evidence_the_endpoint_caches(self):
        """Half the hosted open-weight endpoints have no provider entry but do
        publish a materially discounted cached-input price. Ignoring that
        overstates carry and refuses delegations that are in fact profitable.
        """
        s = registry.resolve("kimi-k2")
        assert s.provider.caches
        assert s.cache_read_rate() < s.rate().inp

    def test_a_cache_rate_equal_to_input_is_not_a_cache(self):
        """Some aggregators echo the input rate into the cache field."""
        prov = registry._sanity_checked_provider(
            providers.UNKNOWN, registry.Entry(key="k", id="k", inp=1.0, cache_read=1.0))
        assert not prov.caches


class TestWriteRateVersusStorageRate:
    def test_a_per_hour_storage_rate_is_not_used_as_a_write_rate(self):
        """The aggregator puts both in one field.

        Google's `cache_write` is per-million *per hour* of storage. For
        `gemini-3.7-flash` that is 0.0208 against an input rate of 0.75. Taken
        for a write rate it prices a cache write at 2.8% of input, a 36x
        understatement of the term that decides whether a token should enter
        the context at all.
        """
        s = registry.resolve("gemini-3.7-flash")
        assert s.cache_storage_abs == pytest.approx(0.020833, rel=1e-3)
        assert s.cache_write_abs is None
        assert s.cache_write_rate() == pytest.approx(s.rate().inp)

    def test_storage_is_kept_rather_than_discarded(self):
        """It is a real charge; it is just a different one."""
        assert registry.resolve("gemini-2.5-pro").cache_storage_abs == pytest.approx(0.375)


class TestFastMode:
    def test_a_cache_read_doubles_when_the_input_rate_doubles(self):
        """A cache read is a fraction *of the input rate*. Pricing reads at the
        standard rate while pricing output at the fast rate understated a
        fast-mode session by most of its carry term."""
        s = registry.resolve(OPUS)
        assert s.cache_read_rate(speed="fast") == pytest.approx(
            2 * s.cache_read_rate())
        assert s.cache_write_rate(speed="fast") == pytest.approx(
            2 * s.cache_write_rate())

    def test_a_model_without_fast_mode_refuses_rather_than_guessing(self):
        from adder.pricing.prices import UnsupportedSpeedError
        with pytest.raises(UnsupportedSpeedError):
            registry.resolve("gpt-5").rate(speed="fast")


class TestPrefixMatchingDoesNotInventPrices:
    def test_an_unknown_generation_does_not_match_a_floating_alias(self):
        """The bundled catalog holds `~openai/gpt-latest`, which normalizes to
        the bare key `gpt`. Under a plain `startswith` every unrecognised
        OpenAI id matched it and was silently priced at that alias's rate, with
        no warning, because resolution had *succeeded*.
        """
        assert not registry.is_known("gpt-9-turbo")
        assert not registry.is_known("gpt-77")

    def test_a_real_variant_still_resolves_to_its_own_record(self):
        assert registry.resolve("gpt-5-mini").id != registry.resolve("gpt-5").id

    def test_a_mid_token_match_is_rejected(self):
        """`gpt-5` is a prefix of the string `gpt-50` and not of the model."""
        assert not registry.is_known("gpt-50")

    def test_a_model_nobody_has_heard_of_is_unknown(self):
        assert not registry.is_known("frobnicator-7")
        with pytest.raises(registry.UnknownModelError):
            registry.resolve("frobnicator-7")

    def test_an_empty_id_is_unknown_rather_than_a_wildcard(self):
        with pytest.raises(registry.UnknownModelError):
            registry.resolve("")


class TestUnknownIsNotUnpriced:
    def test_a_known_model_with_no_published_price_refuses_to_invent_one(self,
                                                                        tmp_path,
                                                                        monkeypatch):
        cat = tmp_path / "cat.json"
        cat.write_text(json.dumps({
            "schema": SCHEMA, "provenance": {},
            "models": [{"key": "mystery-1", "id": "mystery-1", "org": "Nobody",
                        "context": 100_000}],
        }))
        monkeypatch.setenv("ADDER_CATALOG", str(cat))
        registry.reset_cache()
        assert registry.is_known("mystery-1")
        assert not registry.is_priced("mystery-1")
        with pytest.raises(registry.UnpricedModelError):
            registry.resolve("mystery-1").rate()


class TestFeasibilityGates:
    def test_an_unknown_context_window_does_not_pass_the_fits_gate(self):
        """A gate that passes because it does not know is not a gate."""
        spec = registry.ModelSpec(id="m", org="x", provider=providers.UNKNOWN, inp=1, out=1,
                           context=None)
        assert not spec.fits(1)

    def test_context_limit_reports_none_rather_than_infinity(self, tmp_path,
                                                             monkeypatch):
        cat = tmp_path / "cat.json"
        cat.write_text(json.dumps({
            "schema": SCHEMA, "provenance": {},
            "models": [{"key": "mystery-1", "id": "mystery-1", "inp": 1, "out": 2}],
        }))
        monkeypatch.setenv("ADDER_CATALOG", str(cat))
        registry.reset_cache()
        assert registry.context_limit("mystery-1") is None

    def test_a_prefix_under_the_minimum_does_not_cache(self):
        assert not registry.caches("claude-haiku-4-5", 4095)
        assert registry.caches("claude-haiku-4-5", 4096)


class TestCandidates:
    def test_candidates_are_cheapest_first_and_all_fit(self):
        got = registry.candidates(min_context=200_000)
        assert got, "the bundled catalog should hold something this large"
        assert all(s.fits(200_000) for s in got)
        rates = [s.inp for s in got]
        assert rates == sorted(rates)

    def test_caching_only_excludes_endpoints_with_no_prompt_cache(self):
        got = registry.candidates(min_context=100_000, caching_only=True)
        assert all(s.provider.caches for s in got)

    def test_cheapest_that_fits_respects_the_context_gate(self):
        s = registry.cheapest_that_fits(900_000)
        assert s is None or s.fits(900_000)


class TestProvenanceIsReportable:
    def test_every_spec_can_say_where_its_cache_rates_came_from(self):
        assert registry.resolve("gpt-5").rate_provenance == "published"
        assert "multiplier" in registry.resolve(OPUS).rate_provenance
        spec = registry.ModelSpec(id="m", org="x", provider=providers.UNKNOWN, inp=1, out=1)
        assert "no cache" in spec.rate_provenance


class TestCacheInvalidation:
    def test_pointing_at_a_new_catalog_is_seen(self, tmp_path, monkeypatch):
        """Resolution is memoized. Without keying on the layer sources, a test
        that writes a catalog file sees whatever the previous test loaded."""
        cat = tmp_path / "cat.json"
        cat.write_text(json.dumps({
            "schema": SCHEMA, "provenance": {},
            "models": [{"key": "only-me", "id": "only-me", "inp": 9, "out": 9}],
        }))
        monkeypatch.setenv("ADDER_CATALOG", str(cat))
        registry.reset_cache()
        assert registry.resolve("only-me").inp == 9
        assert not registry.is_known("gpt-5")      # replaced the whole stack
