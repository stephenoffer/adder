"""What each provider charges for the *shape* of caching, not the price.

The bug these lock down: every cost path in the repo used to apply Anthropic's
0.10x read / 1.25x write to whatever model id it was handed. On an
automatic-caching provider that invents a write premium nobody charges; on a
provider with no cache at all it understates the carry term -- the dominant
term in a long session -- by a factor of ten.
"""

from __future__ import annotations

import json

import pytest

from adder.pricing import providers


class TestCacheStyle:
    def test_anthropic_writes_cost_more_than_input(self):
        """Explicit caching is the only style where writing is a decision."""
        assert providers.ANTHROPIC.cache_style == providers.EXPLICIT
        assert providers.ANTHROPIC.has_write_premium
        assert providers.ANTHROPIC.write_mult("5m") == 1.25
        assert providers.ANTHROPIC.write_mult("1h") == 2.00

    @pytest.mark.parametrize("prov", [providers.OPENAI, providers.GOOGLE, providers.DEEPSEEK])
    def test_automatic_caching_has_no_write_premium(self, prov):
        """There is no premium because there was no decision to make.

        This is the half of the correction that saves money rather than
        costing it: charging a 1.25x write premium on an OpenAI transcript
        invents 25% of the write side of the bill.
        """
        assert prov.cache_style == providers.AUTOMATIC
        assert not prov.has_write_premium
        assert prov.write_mult() == 1.00

    def test_a_provider_with_no_cache_says_so(self):
        assert not providers.UNKNOWN.caches
        assert not providers.COHERE.caches

    def test_write_mult_is_one_when_there_is_no_cache(self):
        """No cache means a "write" is just ordinary input, billed once."""
        assert providers.UNKNOWN.write_mult() == 1.0
        assert providers.UNKNOWN.write_mult("5m") == 1.0


class TestUnknownTtlIsNotACrash:
    def test_an_unknown_ttl_label_falls_back_rather_than_raising(self):
        """A transcript from one provider read under another's TTL vocabulary
        is a data problem, not a crash."""
        assert providers.ANTHROPIC.write_mult("37y") == 1.25      # the cheapest it offers
        assert providers.OPENAI.write_mult("1h") == 1.00

    def test_the_fallback_never_invents_spend(self):
        """An unknown label resolves to the cheapest write, never the dearest.

        The other direction would let a mislabelled record inflate a report.
        """
        for prov in (providers.ANTHROPIC, providers.OPENAI, providers.GOOGLE, providers.AMAZON):
            table = prov.cache_write_mult or {1: 1.0}
            assert prov.write_mult("nonsense") == min(table.values())


class TestStorageIsItsOwnTerm:
    def test_only_google_bills_for_holding_a_cache_entry(self):
        """Every other cost term is driven by tokens moved. This one is driven
        by elapsed time, and no amount of prompt discipline reduces it."""
        assert providers.GOOGLE.cache_storage_per_mtok_hour
        assert providers.ANTHROPIC.cache_storage_per_mtok_hour is None
        assert providers.OPENAI.cache_storage_per_mtok_hour is None


class TestBatch:
    def test_a_provider_with_no_batch_tier_reports_none_not_a_discount(self):
        """`None` is not 0.5. Recommending "batch it" on a provider with no
        batch API is a suggestion to use a product that does not exist."""
        assert providers.DEEPSEEK.batch_mult is None
        assert providers.ANTHROPIC.batch_mult == 0.50


class TestResolution:
    @pytest.mark.parametrize("org,expect", [
        ("Anthropic", "anthropic"), ("anthropic", "anthropic"),
        ("OpenAI", "openai"), ("Azure", "openai"),
        ("Google", "google"), ("Google DeepMind", "google"),
        ("DeepSeek", "deepseek"), ("xAI", "spacexai"), ("SpaceXAI", "spacexai"),
        ("Mistral AI", "mistral"), ("AWS", "amazon"),
    ])
    def test_org_spellings_collapse_to_one_provider(self, org, expect):
        assert providers.get(org).name == expect

    def test_an_unheard_of_vendor_gets_the_no_cache_default(self):
        """Conservative on purpose: assuming a cache that does not exist
        understates carry by 10x and points the advice the wrong way."""
        assert providers.get("Weyland-Yutani") is providers.UNKNOWN
        assert not providers.get("Weyland-Yutani").caches

    @pytest.mark.parametrize("model,expect", [
        ("claude-opus-5", "anthropic"),
        ("gpt-5-mini", "openai"),
        ("o3-pro", "openai"),
        ("gemini-3-flash", "google"),
        ("deepseek-v4-pro", "deepseek"),
        ("grok-4", "spacexai"),
        ("codestral-2508", "mistral"),
        ("anthropic/claude-opus-4.6", "anthropic"),
    ])
    def test_a_bare_model_id_finds_its_provider(self, model, expect):
        """`--model gpt-5` on the command line carries no org, and must still
        price with OpenAI's economics rather than Anthropic's."""
        assert providers.for_model(model).name == expect

    def test_an_explicit_org_beats_an_id_guess(self):
        assert providers.for_model("some-internal-name", org="OpenAI").name == "openai"

    def test_longest_id_prefix_wins(self):
        """`x-ai/grok-4` must not resolve on a shorter, wronger prefix."""
        assert providers.for_model("x-ai/grok-4").name == "spacexai"


class TestOverrides:
    def test_a_site_can_pin_one_field_without_forking_the_table(self, tmp_path,
                                                                monkeypatch):
        """A file that mentions one field must not blank the rest of the record.

        The failure this prevents is silent: a negotiated cache read rate that
        also, invisibly, reset the TTL table to empty and turned every TTL
        decision into a coin flip.
        """
        f = tmp_path / "prov.json"
        f.write_text(json.dumps({"providers": {"anthropic": {"cache_read_mult": 0.05}}}))
        monkeypatch.setenv("ADDER_PROVIDERS", str(f))
        got = providers.get("anthropic")
        assert got.cache_read_mult == 0.05
        assert got.cache_write_mult == {"5m": 1.25, "1h": 2.00}   # untouched
        assert got.cache_style == providers.EXPLICIT
        assert got.max_breakpoints == 4

    def test_a_new_vendor_can_be_added_without_a_code_change(self, tmp_path,
                                                             monkeypatch):
        f = tmp_path / "prov.json"
        f.write_text(json.dumps({"providers": {"acme": {
            "cache_style": "automatic", "cache_read_mult": 0.2,
            "cache_write_mult": {"auto": 1.0}, "batch_mult": 0.4}}}))
        monkeypatch.setenv("ADDER_PROVIDERS", str(f))
        got = providers.get("acme")
        assert got.caches and got.cache_read_mult == 0.2
        assert got.batch_mult == 0.4

    def test_a_corrupt_override_degrades_rather_than_crashing(self, tmp_path,
                                                              monkeypatch):
        """A broken local file must not take down every report."""
        f = tmp_path / "prov.json"
        f.write_text("{not json at all")
        monkeypatch.setenv("ADDER_PROVIDERS", str(f))
        assert providers.get("anthropic").cache_write_mult == {"5m": 1.25, "1h": 2.00}

    def test_a_missing_override_file_is_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ADDER_PROVIDERS", str(tmp_path / "nope.json"))
        assert providers.get("openai").cache_style == providers.AUTOMATIC

    def test_a_garbage_typed_field_is_dropped_not_coerced_to_nonsense(
            self, tmp_path, monkeypatch):
        """This file is advertised as hand-editable, so the types cannot be
        trusted. A rate that cannot be a number becomes None, which every gate
        already reads as "unknown" -- never as "free"."""
        f = tmp_path / "prov.json"
        f.write_text(json.dumps({"providers": {"acme": {
            "cache_read_mult": "not-a-number", "cache_style": "telepathy"}}}))
        monkeypatch.setenv("ADDER_PROVIDERS", str(f))
        got = providers.get("acme")
        assert got.cache_read_mult is None
        assert got.cache_style == providers.NONE      # an unknown style is not a cache


class TestRoundTrip:
    def test_every_builtin_survives_json(self):
        for name in providers.known_orgs():
            p = providers.get(name)
            assert providers.Provider.from_json(p.to_json()) == p


class TestAnOverrideAppliesHoweverTheModelWasNamed:
    """`ADDER_PROVIDERS` exists so a site with negotiated rates need not fork.

    `for_model` resolved the org through `all_providers()` (overrides applied)
    and the id prefix through `_BUILTIN` (overrides ignored), so one model got
    two different cache economics depending on whether it arrived from the
    catalog with an `org` or was typed on the command line as `--model gpt-5`.
    """

    @pytest.fixture
    def negotiated(self, tmp_path, monkeypatch):
        import json

        path = tmp_path / "providers.json"
        path.write_text(json.dumps({"providers": {
            "openai": {"cache_read_mult": 0.01, "notes": "negotiated"}}}),
            encoding="utf-8")
        monkeypatch.setenv("ADDER_PROVIDERS", str(path))
        return path

    def test_the_override_loads(self, negotiated):
        from adder.pricing.providers import all_providers

        assert all_providers()["openai"].cache_read_mult == 0.01

    def test_the_org_path_sees_it(self, negotiated):
        from adder.pricing.providers import for_model

        assert for_model("gpt-5", "openai").cache_read_mult == 0.01

    def test_the_id_prefix_path_sees_it_too(self, negotiated):
        from adder.pricing.providers import for_model

        assert for_model("gpt-5").cache_read_mult == 0.01

    def test_the_two_paths_agree(self, negotiated):
        from adder.pricing.providers import for_model

        assert for_model("gpt-5") == for_model("gpt-5", "openai")

    def test_an_unoverridden_vendor_is_untouched(self, negotiated):
        from adder.pricing.providers import ANTHROPIC, for_model

        assert for_model("claude-opus-5").cache_read_mult == ANTHROPIC.cache_read_mult
