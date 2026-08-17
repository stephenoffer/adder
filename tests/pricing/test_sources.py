"""Parsers for the public sources. No socket is opened anywhere in this file.

The fetchers are one `urlopen` call each and are not worth testing. The parsers
are where a source silently changing shape turns into a wrong recommendation,
so they are tested against captures of the real payload shapes -- including the
malformed ones that a live refresh actually returns.
"""
from __future__ import annotations

import json

import pytest

from adder.pricing.sources import (
    FetchFailedError,
    OfflineError,
    fetch,
    lmarena_entries,
    parse_lmarena,
    parse_openrouter,
    refresh,
)


class TestArenaParser:
    def test_every_board_is_extracted_from_the_escaped_payload(self, arena_page):
        boards = parse_lmarena(arena_page)
        assert set(boards) == {"text-overall-style_control", "webdev-overall-raw"}
        assert len(boards["text-overall-style_control"]) == 2

    def test_a_layout_change_raises_instead_of_returning_nothing(self):
        """An empty parse must not look like 'no model is any good'."""
        with pytest.raises(FetchFailedError, match="layout changed"):
            parse_lmarena("<html><body>redesigned</body></html>")

    def test_bytes_and_str_both_parse(self, arena_page):
        assert parse_lmarena(arena_page.encode()) == parse_lmarena(arena_page)

    def test_effort_variants_fold_into_one_entry_keeping_the_best(self, arena_page):
        """The arena ranks -high and -max separately; routing wants the model."""
        entries = {e.key: e for e in lmarena_entries(parse_lmarena(arena_page))}
        assert "claude-opus-5" in entries
        e = entries["claude-opus-5"]
        assert e.elo["webdev"] == 1691.0      # the better of -high and -max
        assert e.elo["text"] == 1500.5

    def test_board_ratings_are_kept_apart(self, arena_page):
        """Folding boards together would rank a prose model as a coding model."""
        e = {x.key: x for x in lmarena_entries(parse_lmarena(arena_page))}["claude-opus-5"]
        assert e.rating() == 1691.0           # code board wins for a coding router
        assert e.rating(boards=("text",)) == 1500.5

    def test_license_carries_through_so_open_weights_is_answerable(self, arena_page):
        entries = {e.key: e for e in lmarena_entries(parse_lmarena(arena_page))}
        assert entries["qwen4"].open_weights
        assert not entries["claude-opus-5"].open_weights

    def test_arena_prices_are_never_marked_verified(self, arena_page):
        for e in lmarena_entries(parse_lmarena(arena_page)):
            assert not e.verified


class TestAggregatorParser:
    def test_prices_convert_from_per_token_to_per_million(self, openrouter_page):
        e = {x.key: x for x in parse_openrouter(openrouter_page)}["claude-opus-5"]
        assert e.inp == 5.0 and e.out == 25.0
        assert e.cache_read == 0.5 and e.cache_write == 6.25

    def test_negative_sentinel_prices_become_unknown(self, openrouter_page):
        """Left alone, a -1 sorts first and the router recommends being paid."""
        e = {x.key: x for x in parse_openrouter(openrouter_page)}["auto"]
        assert e.inp is None and e.out is None and not e.priced

    def test_image_generators_are_not_routing_targets(self, openrouter_page):
        assert "gpt-image-2" not in {x.key for x in parse_openrouter(openrouter_page)}

    def test_capability_metadata_survives(self, openrouter_page):
        e = {x.key: x for x in parse_openrouter(openrouter_page)}["claude-opus-5"]
        assert e.supports_tools and e.supports_reasoning
        assert e.context == 1_000_000 and e.max_output == 128_000
        assert e.coding == 80.0 and e.released == "2026-01-01"

    def test_unexpected_payload_shape_raises(self):
        with pytest.raises(FetchFailedError, match="payload shape"):
            parse_openrouter('{"data": {"not": "a list"}}')


class TestRefresh:
    def test_replaying_captures_needs_no_network(self, tmp_path, monkeypatch, arena_page, openrouter_page):
        monkeypatch.setenv("ADDER_OFFLINE", "1")
        (tmp_path / "a.html").write_text(arena_page)
        (tmp_path / "o.json").write_text(openrouter_page)
        cat, results = refresh(offline_files={"lmarena": tmp_path / "a.html",
                                              "openrouter": tmp_path / "o.json"})
        assert all(r.ok for r in results)
        e = cat.get("claude-opus-5")
        assert e.cache_read == 0.5 and e.rating() == 1691.0
        assert set(e.sources) == {"openrouter", "lmarena"}

    def test_one_dead_source_degrades_instead_of_failing(self, tmp_path, monkeypatch, openrouter_page):
        monkeypatch.setenv("ADDER_OFFLINE", "1")
        (tmp_path / "o.json").write_text(openrouter_page)
        cat, results = refresh(offline_files={"openrouter": tmp_path / "o.json"})
        by = {r.name: r for r in results}
        assert by["openrouter"].ok and not by["lmarena"].ok
        assert cat.get("claude-opus-5") is not None
        assert cat.provenance["sources"][1]["error"]

    def test_offline_env_var_blocks_the_socket(self, monkeypatch):
        monkeypatch.setenv("ADDER_OFFLINE", "1")
        with pytest.raises(OfflineError):
            fetch("https://example.invalid/")

    def test_provenance_records_what_actually_happened(self, tmp_path, monkeypatch, openrouter_page):
        monkeypatch.setenv("ADDER_OFFLINE", "1")
        (tmp_path / "o.json").write_text(openrouter_page)
        cat, _ = refresh(offline_files={"openrouter": tmp_path / "o.json"})
        assert cat.provenance["refreshed_at"]
        names = {s["name"] for s in cat.provenance["sources"]}
        assert names == {"openrouter", "lmarena"}


class TestIntervalsAndVariants:
    """What the arena publishes about its own uncertainty, kept rather than dropped."""

    def test_the_published_interval_survives_the_fold(self, arena_page):
        e = {x.key: x for x in lmarena_entries(parse_lmarena(arena_page))}["claude-opus-5"]
        assert e.elo_lo["text"] == 1490.0 and e.elo_hi["text"] == 1511.0

    def test_a_rating_with_no_interval_carries_none(self, arena_page):
        """The webdev rows in the fixture publish no bounds; do not invent any."""
        e = {x.key: x for x in lmarena_entries(parse_lmarena(arena_page))}["claude-opus-5"]
        assert "webdev" not in e.elo_lo
        assert e.rating_interval() is None      # webdev wins the board preference

    def test_the_variant_that_earned_the_rating_is_recorded(self, arena_page):
        e = {x.key: x for x in lmarena_entries(parse_lmarena(arena_page))}["claude-opus-5"]
        assert e.rating_variant == "claude-opus-5-max"

    def test_an_aggregator_alias_route_reports_its_real_vendor(self):
        """`~anthropic` is a floating alias, not a different company."""
        raw = json.dumps({"data": [{
            "id": "~anthropic/claude-haiku-latest", "name": "Claude Haiku Latest",
            "architecture": {"output_modalities": ["text"]},
            "pricing": {"prompt": "0.000001", "completion": "0.000005"},
            "supported_parameters": ["tools"]}]})
        e = parse_openrouter(raw)[0]
        assert e.org.lower() == "anthropic"


class TestTheOfflineSwitchUsesItsOwnVocabulary:
    """`ADDER_OFFLINE=0` means "not offline" everywhere the setting is defined.

    `settings._as_bool` reads 0/false/no/off as False, and `adder config`
    documents `offline` as a boolean. `fetch` tested the raw environment string
    for truthiness, so the one obvious way to turn the switch off for a single
    command -- `ADDER_OFFLINE=0 adder models refresh` -- refused every fetch.
    """

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_the_switch_is_on(self, value, monkeypatch):
        from adder.pricing.sources import OfflineError, fetch, is_offline

        monkeypatch.setenv("ADDER_OFFLINE", value)
        assert is_offline() is True
        with pytest.raises(OfflineError):
            fetch("https://example.invalid/x")

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
    def test_the_switch_is_off(self, value, monkeypatch):
        from adder.pricing.sources import is_offline

        monkeypatch.setenv("ADDER_OFFLINE", value)
        assert is_offline() is False

    def test_it_agrees_with_the_setting(self, monkeypatch):
        from adder.core.settings import get
        from adder.pricing.sources import is_offline

        for value in ("0", "1", "false", "true", "no", "yes"):
            monkeypatch.setenv("ADDER_OFFLINE", value)
            assert is_offline() == bool(get("offline")), value


def test_a_truncated_response_is_a_fetch_failure_not_a_traceback(monkeypatch):
    """`IncompleteRead` is an `http.client.HTTPException`: neither an OSError
    nor a URLError, so it escaped the handler entirely."""
    import http.client
    import urllib.request

    from adder.pricing.sources import FetchFailedError, fetch

    monkeypatch.delenv("ADDER_OFFLINE", raising=False)

    class _Torn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, *a):
            raise http.client.IncompleteRead(b"half")

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Torn())
    with pytest.raises(FetchFailedError):
        fetch("https://example.invalid/x")
