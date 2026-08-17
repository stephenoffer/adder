"""Catalog joining, layering, and the gates that keep a stale price honest."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest

from adder.pricing.catalog import (
    SCHEMA,
    Catalog,
    Entry,
    first_party,
    load,
    merge,
    normalize_key,
    normalize_org,
)
from adder.pricing.prices import rate
from adder.pricing.registry import resolve

# Sonnet 5's introductory window, which the dated-rate tests below straddle.
DURING = date(2026, 8, 15)
AFTER = date(2026, 9, 1)


def _entry(elo):
    """An entry carrying only arena ratings, for the board-selection tests."""
    return Entry(key="m", id="m", elo=dict(elo))


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
        from adder.pricing.prices import MODELS

        cat = load()
        for mid in MODELS:
            e = cat.get(mid)
            assert e is not None and e.verified, mid


class TestOrgNormalization:
    """An aggregator decorates alias routes; the harness gate reads the result.

    `~anthropic/claude-haiku-latest` is an Anthropic model on a floating alias.
    Left decorated, the organisation reads as a vendor that does not exist, the
    Claude Code harness gate refuses it inline placement, and the refusal names
    "~anthropic" to the user.
    """

    def test_alias_decoration_is_stripped(self):
        assert normalize_org("~anthropic") == "anthropic"
        assert normalize_org("  Anthropic ") == "Anthropic"

    def test_an_undecorated_name_is_untouched(self):
        assert normalize_org("Alibaba") == "Alibaba"
        assert normalize_org("") == ""

    def test_a_decorated_entry_is_recognised_as_its_real_vendor(self):
        e = Entry.from_json({"key": "k", "id": "~anthropic/claude-haiku-latest",
                             "org": "~anthropic"})
        assert e.org.lower() == "anthropic"


class TestHostileCatalogFiles:
    """A project override is hand-edited by design, so its types are not trusted."""

    def test_a_price_written_as_a_string_is_coerced(self):
        e = Entry.from_json({"key": "k", "id": "k", "inp": "5", "out": 25})
        assert e.inp == 5.0 and e.priced

    def test_an_unparseable_price_becomes_unknown_not_cheap(self):
        e = Entry.from_json({"key": "k", "id": "k", "inp": "free", "out": 25})
        assert e.inp is None and not e.priced

    def test_a_boolean_is_not_a_price(self):
        assert Entry.from_json({"key": "k", "id": "k", "inp": True}).inp is None

    def test_nan_and_infinity_are_rejected(self):
        for bad in (float("nan"), float("inf")):
            assert Entry.from_json({"key": "k", "id": "k", "inp": bad}).inp is None

    def test_a_bad_context_does_not_become_a_window(self):
        assert Entry.from_json({"key": "k", "id": "k", "context": "big"}).context is None

    def test_a_string_price_does_not_crash_a_ranking(self, tmp_path, monkeypatch):
        """The failure this replaces was a bare TypeError from inside a cost model."""
        from adder.decide.route.select import Need, rank

        path = tmp_path / "hand-edited.json"
        Catalog([Entry.from_json({"key": "junk", "id": "junk", "inp": "5", "out": 25,
                                  "context": 1_000_000, "params": ["tools"],
                                  "elo": {"webdev": 1700}})]).save(path)
        monkeypatch.setenv("ADDER_CATALOG", str(path))
        picks = rank(Need(), cat=load())
        assert all(isinstance(p.cost, float) for p in picks)


class TestRatingIntervals:
    """A rating without its interval invites precision the arena does not claim."""

    def test_the_interval_travels_with_the_rating(self):
        e = Entry(key="m", id="m", elo={"webdev": 1690.0},
                  elo_lo={"webdev": 1681.0}, elo_hi={"webdev": 1701.0})
        assert e.rating() == 1690.0
        assert e.rating_interval() == (1681.0, 1701.0)

    def test_a_rating_with_no_published_interval_says_so(self):
        assert Entry(key="m", id="m", elo={"webdev": 1690.0}).rating_interval() is None

    def test_the_interval_follows_the_board_the_rating_came_from(self):
        e = Entry(key="m", id="m", elo={"text": 1500.0, "webdev": 1600.0},
                  elo_lo={"text": 1490.0, "webdev": 1590.0},
                  elo_hi={"text": 1510.0, "webdev": 1610.0})
        assert e.rating_board() == "webdev"
        assert e.rating_interval() == (1590.0, 1610.0)
        assert e.rating_interval(boards=("text",)) == (1490.0, 1510.0)

    def test_intervals_merge_alongside_ratings(self):
        a = Entry(key="m", id="m", elo={"text": 1400.0}, elo_lo={"text": 1390.0},
                  elo_hi={"text": 1410.0})
        b = Entry(key="m", id="m", elo={"webdev": 1500.0}, elo_lo={"webdev": 1490.0},
                  elo_hi={"webdev": 1510.0}, rating_variant="m-max")
        got = merge(a, b)
        assert got.elo_lo == {"text": 1390.0, "webdev": 1490.0}
        assert got.rating_variant == "m-max"

    def test_the_variant_that_earned_the_rating_is_recorded(self):
        """The arena ranks efforts separately; the price table has one price."""
        e = Entry(key="m", id="m", elo={"webdev": 1690.0}, rating_variant="m-max")
        assert e.rating_variant == "m-max"


# Zero is a price. None is the absence of one, and the catalog conflated them.
#
# `Entry` makes every price Optional so that "this model costs nothing to call"
# and "nobody published a price for this model" stay distinguishable -- the whole
# `priced_only` gate turns on it, and the module docstring says so: "None means
# 'not published', never 'free'."
#
# Two filters written as truthiness tests erased that distinction:
#
# * `to_json` dropped any falsy value, so a free model was written to disk with no
#   price and reloaded as unpriced. A catalog that had been saved and reloaded
#   silently stopped offering every free model it knew about.
# * `merge` overlaid only truthy values, so pinning a price to 0 in a project
#   override did nothing and the base price was served instead -- an override that
#   looked applied and was not.
class TestAFreePriceSurvivesSerialization:
    def test_zero_is_written_to_the_file(self):
        e = Entry(key="m", id="m", inp=0.0, out=0.0)
        assert e.to_json()["inp"] == 0.0
        assert e.to_json()["out"] == 0.0

    def test_zero_round_trips_as_a_price_not_as_absence(self):
        e = Entry(key="m", id="m", inp=0.0, out=0.0, context=128_000)
        back = Entry.from_json(e.to_json())
        assert (back.inp, back.out) == (0.0, 0.0)
        assert back.priced

    def test_an_unpriced_model_still_round_trips_as_unpriced(self):
        back = Entry.from_json(Entry(key="m", id="m").to_json())
        assert back.inp is None and not back.priced

    def test_a_free_model_survives_a_whole_catalog_round_trip(self):
        """`find(priced_only=True)` is what drops an unpriced model."""
        cat = Catalog([Entry(key="m", id="m", inp=0.0, out=0.0, context=128_000,
                             params=("tools",))])
        back = Catalog.from_json(json.loads(json.dumps(cat.to_json())))
        assert [e.id for e in back.find(priced_only=True)] == ["m"]

    def test_defaults_are_still_omitted_so_files_stay_small(self):
        e = Entry(key="m", id="m", name="", votes=0, verified=False)
        assert sorted(e.to_json()) == ["id", "key"]


class TestAnOverrideCanPinAPriceToZero:
    def test_pinning_free_overrides_the_base_price(self):
        got = merge(Entry(key="m", id="m", inp=5.0, out=25.0),
                    Entry(key="m", id="m", inp=0.0, out=0.0))
        assert (got.inp, got.out) == (0.0, 0.0)

    def test_an_absent_override_still_leaves_the_base_price_alone(self):
        """The invariant `merge` exists for: a refresh that fails to price a
        model must not blank the price we already had."""
        got = merge(Entry(key="m", id="m", inp=5.0, out=25.0),
                    Entry(key="m", id="m"))
        assert (got.inp, got.out) == (5.0, 25.0)


class TestRatingIntervalsMergeOnTheirOwn:
    def test_intervals_apply_without_a_rating_in_the_same_overlay(self):
        """`ratings_overlap` reads these; dropping them switches off the
        "the arena cannot separate these two" guard without saying so."""
        got = merge(Entry(key="m", id="m", elo_lo={"w": 10.0}, elo_hi={"w": 20.0}),
                    Entry(key="m", id="m", elo_lo={"w": 1.0}, elo_hi={"w": 2.0}))
        assert got.elo_lo == {"w": 1.0}
        assert got.elo_hi == {"w": 2.0}

    def test_an_overlay_with_no_intervals_leaves_the_base_ones(self):
        got = merge(Entry(key="m", id="m", elo_lo={"w": 10.0}, elo_hi={"w": 20.0}),
                    Entry(key="m", id="m", elo={"w": 1500.0}))
        assert got.elo_lo == {"w": 10.0}
        assert got.elo == {"w": 1500.0}


# Which board a rating comes from must depend on the data, not on load order.
#
# One board carries several keys -- `webdev-hard` and `webdev-easy` both answer to
# `webdev` -- and `rating_board` returned whichever appeared first when iterating
# the dict. That order is the order the sources happened to merge in, which
# changes between refreshes, so the same catalog could rate a model 1600 on one
# run and 1400 on the next.
#
# It is not a cosmetic difference. `rating()` sets the quality floor in `rank`,
# feeds `p_loss_from_elo`, and decides every `substitutes()` verdict, so a
# 200-point swing that comes from nothing is a routing decision that comes from
# nothing.
class TestOrderIndependence:
    def test_the_same_ratings_give_the_same_board_either_way(self):
        a = _entry([("webdev-hard", 1600.0), ("webdev-easy", 1400.0)])
        b = _entry([("webdev-easy", 1400.0), ("webdev-hard", 1600.0)])
        assert a.rating_board() == b.rating_board()

    def test_and_therefore_the_same_rating(self):
        a = _entry([("webdev-hard", 1600.0), ("webdev-easy", 1400.0)])
        b = _entry([("webdev-easy", 1400.0), ("webdev-hard", 1600.0)])
        assert a.rating() == b.rating()

    def test_ties_are_broken_by_name_not_by_insertion(self):
        a = _entry([("webdev-z", 1500.0), ("webdev-a", 1500.0)])
        b = _entry([("webdev-a", 1500.0), ("webdev-z", 1500.0)])
        assert a.rating_board() == b.rating_board() == "webdev-a"

    def test_the_fallback_board_is_order_independent_too(self):
        """No coding board present: the best of whatever is there."""
        a = _entry([("vision", 1700.0), ("audio", 1500.0)])
        b = _entry([("audio", 1500.0), ("vision", 1700.0)])
        assert a.rating_board() == b.rating_board() == "vision"


class TestTheSelectionRuleItself:
    def test_the_best_rating_on_the_board_wins(self):
        assert _entry([("webdev-easy", 1400.0),
                       ("webdev-hard", 1600.0)]).rating() == 1600.0

    def test_a_preferred_board_beats_a_higher_rating_elsewhere(self):
        """Board preference is the first key; `rating()` is code-shaped first."""
        e = _entry([("vision", 1900.0), ("webdev", 1500.0)])
        assert e.rating_board() == "webdev"

    def test_no_ratings_means_no_board(self):
        assert _entry([]).rating_board() is None
        assert _entry([]).rating() is None

    def test_an_exact_board_name_still_matches(self):
        assert _entry([("webdev", 1500.0)]).rating_board() == "webdev"


# The catalog's Claude layer honours the same dates the price table does.
#
# `prices.py` exists because Claude rates move: Sonnet 5 ships at an introductory
# $2/$10 that reverts to $3/$15 after 2026-08-31, and the module's own docstring
# says "every lookup takes an `on` date". The catalog's first-party layer read
# `m.base` instead, so during the introductory window `adder pick` priced Sonnet 5
# at $3/$15 while `adder trace` and `adder policy` priced the same model at
# $2/$10 -- a 50% penalty applied to one model in the cross-vendor comparison, and
# two halves of one tool disagreeing about one price.
#
# This layer is generated at load time rather than read off disk, which is exactly
# why it can carry a date honestly where a scraped price cannot.
class TestTheCatalogAgreesWithThePriceTable:
    @pytest.mark.parametrize("on", [DURING, AFTER])
    def test_the_input_rate_matches(self, on):
        e = first_party(on).get("claude-sonnet-5")
        assert e.inp == rate("claude-sonnet-5", on).inp

    @pytest.mark.parametrize("on", [DURING, AFTER])
    def test_the_output_rate_matches(self, on):
        e = first_party(on).get("claude-sonnet-5")
        assert e.out == rate("claude-sonnet-5", on).out

    def test_the_intro_rate_is_cheaper_than_the_reverted_one(self):
        """Guards the test itself: if these were equal it would prove nothing."""
        assert first_party(DURING).get("claude-sonnet-5").inp < \
            first_party(AFTER).get("claude-sonnet-5").inp

    def test_cache_rates_are_derived_from_the_dated_input_rate(self):
        during = first_party(DURING).get("claude-sonnet-5")
        after = first_party(AFTER).get("claude-sonnet-5")
        assert during.cache_read < after.cache_read

    def test_a_model_with_no_intro_rate_is_unaffected(self):
        assert first_party(DURING).get("claude-opus-5").inp == \
            first_party(AFTER).get("claude-opus-5").inp


class TestTheDateReachesTheWholeStack:
    def test_load_passes_the_date_through(self):
        assert load(on=DURING).get("claude-sonnet-5").inp == \
            rate("claude-sonnet-5", DURING).inp
        assert load(on=AFTER).get("claude-sonnet-5").inp == \
            rate("claude-sonnet-5", AFTER).inp


class TestTheRegistrySnapshotAgreesWithItsOwnMethod:
    def test_spec_fields_match_spec_rate(self):
        """`candidates()` sorts on `.inp`, so a stale snapshot reaches the ladder."""
        spec = resolve("claude-sonnet-5")
        assert (spec.inp, spec.out) == tuple(spec.rate())


# A free model has to survive the whole path, not just the parser.
#
# `parse_openrouter` has always produced `inp=0.0` for a `:free` row, and
# `Entry.priced` has always accepted it. The loss happened on the way to disk:
# `to_json` dropped every falsy value, so the price vanished and the entry
# reloaded as unpriced, where `find(priced_only=True)` drops it.
#
# This is not hypothetical. The shipped `data/catalog.json` carries 16 `:free`
# models -- `google/gemma-4-31b-it:free`, `openai/gpt-oss-20b:free`, three
# NVIDIA Nemotron free tiers -- with no price of any kind, so `adder pick`
# has been excluding the cheapest models in the catalog from every ranking.
# They come back on the next `adder models refresh`.
class TestAFreeModelSurvivesTheWholeRefreshPath:
    @staticmethod
    def _page():
        return json.dumps({"data": [{
            "id": "openai/gpt-oss-20b:free", "name": "OpenAI: gpt-oss-20b (free)",
            "context_length": 131072,
            "architecture": {"input_modalities": ["text"],
                             "output_modalities": ["text"]},
            "pricing": {"prompt": "0", "completion": "0"},
            "supported_parameters": ["tools"],
        }]})

    def test_the_parser_reads_a_zero_price_as_a_price(self):
        from adder.pricing.sources import parse_openrouter

        e = parse_openrouter(self._page())[0]
        assert (e.inp, e.out) == (0.0, 0.0)
        assert e.priced

    def test_it_is_still_priced_after_a_save_and_reload(self, tmp_path):
        from adder.pricing.sources import parse_openrouter

        cat = Catalog(parse_openrouter(self._page()))
        path = cat.save(tmp_path / "catalog.json")
        back = Catalog.from_json(json.loads(path.read_text()))
        assert [e.id for e in back.find(priced_only=True, needs_tools=True)] == [
            "openai/gpt-oss-20b:free"]

    def test_a_negative_sentinel_price_is_still_rejected(self):
        """`openrouter/auto` prices at -1 to mean "resolved at request time"."""
        from adder.pricing.sources import parse_openrouter

        page = json.dumps({"data": [{
            "id": "openrouter/auto", "name": "Auto",
            "architecture": {"output_modalities": ["text"]},
            "pricing": {"prompt": "-1", "completion": "-1"},
            "supported_parameters": ["tools"]}]})
        assert not parse_openrouter(page)[0].priced


class TestTheProjectOverrideIsFoundFromASubdirectory:
    """`.adder/catalog.json` is searched upward, like `.adder.json`.

    `settings.project_file` walks up "the way git finds `.git`, so a repo-level
    setting applies from any subdirectory of it". This looked only in the
    current directory, so the two project-level override mechanisms disagreed
    about what "this project" means: the settings file applied from anywhere in
    a repo and a pinned price silently stopped applying one directory down --
    while the module docstring advertises the override as the way to pin a
    price without forking the file.
    """

    @pytest.fixture
    def repo(self, tmp_path, monkeypatch):
        import json as _json

        (tmp_path / ".adder").mkdir()
        (tmp_path / ".adder" / "catalog.json").write_text(_json.dumps({
            "schema": 1,
            "models": [{"key": "gpt-5", "id": "gpt-5", "org": "openai",
                        "inp": 0.5, "out": 1.0, "context": 400_000}],
        }), encoding="utf-8")
        (tmp_path / "sub" / "deeper").mkdir(parents=True)
        monkeypatch.delenv("ADDER_CATALOG", raising=False)
        return tmp_path

    def test_found_at_the_root(self, repo, monkeypatch):
        from adder.pricing.catalog import project_override

        monkeypatch.chdir(repo)
        assert project_override() == repo / ".adder" / "catalog.json"

    def test_found_from_a_subdirectory(self, repo, monkeypatch):
        from adder.pricing.catalog import project_override

        monkeypatch.chdir(repo / "sub" / "deeper")
        assert project_override() == repo / ".adder" / "catalog.json"

    def test_the_pinned_price_applies_from_a_subdirectory(self, repo, monkeypatch):
        from adder.pricing import registry
        from adder.pricing.catalog import load

        monkeypatch.chdir(repo / "sub" / "deeper")
        registry.reset_cache()
        assert load().get("gpt-5").inp == 0.5

    def test_nothing_found_still_names_where_it_looked(self, tmp_path, monkeypatch):
        from adder.pricing.catalog import project_override

        deep = tmp_path / "a" / "b"
        deep.mkdir(parents=True)
        monkeypatch.chdir(deep)
        assert project_override().name == "catalog.json"
