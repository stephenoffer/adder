"""Catalog joining, layering, and the gates that keep a stale price honest."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from adder.catalog import (
    SCHEMA,
    Catalog,
    Entry,
    load,
    merge,
    normalize_key,
)


def _stamp(days_ago: float = 0.0) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).replace(
        microsecond=0).isoformat()


class TestNormalization:
    def test_arena_and_aggregator_agree_after_normalizing(self):
        """The two public sources name the same model four different ways."""
        assert normalize_key("claude-opus-4-6-thinking") == "claude-opus-4.6"
        assert normalize_key("anthropic/claude-opus-4.6") == "claude-opus-4.6"
        assert normalize_key("Claude Opus 4.6 (high)") == "claude-opus-4.6"
        assert normalize_key("claude-opus-4-6-20260101") == "claude-opus-4.6"

    def test_effort_variants_collapse_to_one_model(self):
        """The arena ranks efforts separately; the catalog holds one entry."""
        keys = {normalize_key(k) for k in (
            "claude-opus-5-max", "claude-opus-5-high", "claude-opus-5-thinking",
            "claude-opus-5")}
        assert keys == {"claude-opus-5"}

    def test_thinking_budget_suffix_is_stripped(self):
        assert normalize_key("claude-sonnet-4-5-20250929-high-32k") == "claude-sonnet-4.5"

    def test_distinct_models_do_not_collide(self):
        assert normalize_key("gpt-5.5-mini") != normalize_key("gpt-5.5")
        assert normalize_key("claude-sonnet-4.6") != normalize_key("claude-sonnet-5")


class TestMerge:
    def test_a_failed_field_does_not_erase_a_good_one(self):
        """A refresh that misses a price must not blank the price we had."""
        base = Entry(key="m", id="m", inp=3.0, out=15.0, context=200_000)
        over = Entry(key="m", id="m", elo={"text": 1400.0})
        got = merge(base, over)
        assert got.inp == 3.0 and got.context == 200_000 and got.elo["text"] == 1400.0

    def test_elo_boards_accumulate_across_sources(self):
        a = Entry(key="m", id="m", elo={"text": 1400.0})
        b = Entry(key="m", id="m", elo={"webdev": 1500.0})
        assert merge(a, b).elo == {"text": 1400.0, "webdev": 1500.0}

    def test_verified_is_sticky(self):
        """An unverified overlay cannot downgrade a first-party price claim."""
        base = Entry(key="m", id="m", inp=5.0, verified=True)
        assert merge(base, Entry(key="m", id="m", inp=4.0, verified=False)).verified

    def test_sources_are_recorded_without_duplicates(self):
        a = Entry(key="m", id="m", sources=("lmarena",))
        b = Entry(key="m", id="m", sources=("lmarena", "openrouter"))
        assert merge(a, b).sources == ("lmarena", "openrouter")


class TestEntry:
    def test_open_weights_is_derived_from_the_license(self):
        assert Entry(key="a", id="a", license="Apache-2.0").open_weights
        assert not Entry(key="b", id="b", license="Proprietary").open_weights
        assert not Entry(key="c", id="c").open_weights  # unknown is not open

    def test_rating_prefers_code_boards_over_prose(self):
        """A coding router that ranks on the text board picks the wrong model."""
        e = Entry(key="m", id="m", elo={"text": 1500.0, "webdev": 1600.0})
        assert e.rating() == 1600.0

    def test_rating_is_none_when_nothing_rated_it(self):
        assert Entry(key="m", id="m").rating() is None

    def test_missing_price_is_unknown_not_free(self):
        assert not Entry(key="m", id="m", out=5.0).priced
        assert Entry(key="m", id="m", inp=0.0, out=0.0).priced

    def test_age_is_a_number(self):
        assert Entry(key="m", id="m", fetched_at=_stamp(10)).age_days() == pytest.approx(
            10, abs=0.1)
        assert Entry(key="m", id="m").age_days() is None


class TestCatalog:
    def test_lookup_normalizes_the_query(self):
        c = Catalog([Entry(key="claude-opus-5", id="anthropic/claude-opus-5")])
        assert c.get("anthropic/claude-opus-5") is not None
        assert c.get("claude-opus-5-thinking") is not None

    def test_find_gates_on_context_and_tools(self):
        c = Catalog([
            Entry(key="small", id="small", inp=1, out=2, context=8_000, params=("tools",)),
            Entry(key="big", id="big", inp=1, out=2, context=500_000, params=("tools",)),
            Entry(key="notools", id="notools", inp=1, out=2, context=500_000),
        ])
        got = {e.key for e in c.find(min_context=200_000, needs_tools=True)}
        assert got == {"big"}

    def test_unpriced_models_are_excluded_by_default(self):
        c = Catalog([Entry(key="m", id="m", context=100_000)])
        assert c.find() == []
        assert len(c.find(priced_only=False)) == 1

    def test_staleness_uses_the_freshest_source(self):
        c = Catalog([Entry(key="a", id="a", fetched_at=_stamp(200)),
                     Entry(key="b", id="b", fetched_at=_stamp(2))])
        assert c.age_days() == pytest.approx(2, abs=0.1)
        assert not c.is_stale(max_age_days=21)

    def test_an_empty_catalog_is_stale_not_fresh(self):
        """No data must never read as up-to-date data."""
        assert Catalog().is_stale()

    def test_roundtrip(self, tmp_path):
        c = Catalog([Entry(key="m", id="m", inp=1.0, out=2.0, elo={"text": 1400.0},
                           context=1000, verified=True)])
        p = c.save(tmp_path / "cat.json")
        back = Catalog.from_json(json.loads(p.read_text()))
        e = back.get("m")
        assert e.inp == 1.0 and e.elo["text"] == 1400.0 and e.verified

    def test_a_future_schema_refuses_to_load_silently(self):
        with pytest.raises(ValueError, match="schema"):
            Catalog.from_json({"schema": SCHEMA + 99, "models": []})


class TestLayering:
    def test_project_override_beats_the_user_cache(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ADDER_HOME", str(tmp_path / "home"))
        Catalog([Entry(key="m", id="m", inp=9.0, out=9.0, context=1000)]).save(
            tmp_path / "home" / "catalog.json")
        Catalog([Entry(key="m", id="m", inp=1.0, out=1.0)]).save(
            tmp_path / "proj" / ".adder" / "catalog.json")
        cat = load(cwd=tmp_path / "proj", include_first_party=False)
        e = cat.get("m")
        assert e.inp == 1.0          # override wins the price
        assert e.context == 1000     # and keeps everything it did not say

    def test_first_party_claude_rates_beat_anything_scraped(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ADDER_HOME", str(tmp_path / "home"))
        Catalog([Entry(key="claude-opus-5", id="anthropic/claude-opus-5",
                       inp=999.0, out=999.0, verified=False)]).save(
            tmp_path / "home" / "catalog.json")
        e = load(cwd=tmp_path).get("claude-opus-5")
        assert e.inp == 5.0 and e.verified

    def test_a_corrupt_cache_degrades_instead_of_crashing(self, tmp_path, monkeypatch):
        """A bad cache file must not take down every cost report on the machine."""
        monkeypatch.setenv("ADDER_HOME", str(tmp_path / "home"))
        bad = tmp_path / "home" / "catalog.json"
        bad.parent.mkdir(parents=True)
        bad.write_text("{not json")
        assert load(cwd=tmp_path).get("claude-opus-5") is not None

    def test_first_party_layer_covers_every_priced_claude_model(self):
        from adder.prices import MODELS

        cat = load()
        for mid in MODELS:
            e = cat.get(mid)
            assert e is not None and e.verified, mid
