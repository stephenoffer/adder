"""Parsers for the public sources. No socket is opened anywhere in this file.

The fetchers are one `urlopen` call each and are not worth testing. The parsers
are where a source silently changing shape turns into a wrong recommendation,
so they are tested against captures of the real payload shapes -- including the
malformed ones that a live refresh actually returns.
"""

import json

import pytest

from adder.sources import (
    FetchFailed,
    Offline,
    fetch,
    lmarena_entries,
    parse_lmarena,
    parse_openrouter,
    refresh,
)

# The arena page is a server-rendered React stream: the payload is real JSON,
# escaped once, embedded in a JS string. This is that shape, minimised.
ARENA_PAGE = (
    '<!DOCTYPE html><html><body><script>self.__next_f.push([1,"'
    '{\\"id\\":\\"leaderboard-sets/public/leaderboards/text-overall-style_control'
    '/leaderboard-snapshots/latest\\",\\"entries\\":['
    '{\\"rank\\":1,\\"modelKey\\":\\"claude-opus-5-max-text\\",'
    '\\"modelDisplayName\\":\\"claude-opus-5-max\\",\\"rating\\":1500.5,'
    '\\"votes\\":21533,\\"modelOrganization\\":\\"Anthropic\\",'
    '\\"license\\":\\"Proprietary\\",\\"inputPricePerMillion\\":5,'
    '\\"outputPricePerMillion\\":25,\\"contextLength\\":1000000},'
    '{\\"rank\\":2,\\"modelDisplayName\\":\\"qwen4-max\\",\\"rating\\":1480.0,'
    '\\"votes\\":9000,\\"modelOrganization\\":\\"Alibaba\\",'
    '\\"license\\":\\"Apache 2.0\\",\\"inputPricePerMillion\\":1.2,'
    '\\"outputPricePerMillion\\":6,\\"contextLength\\":262144}]}'
    '{\\"id\\":\\"leaderboard-sets/public/leaderboards/webdev-overall-raw'
    '/leaderboard-snapshots/latest\\",\\"entries\\":['
    '{\\"rank\\":1,\\"modelDisplayName\\":\\"claude-opus-5-high\\",'
    '\\"rating\\":1690.0,\\"votes\\":12000,\\"modelOrganization\\":\\"Anthropic\\"},'
    '{\\"rank\\":2,\\"modelDisplayName\\":\\"claude-opus-5-max\\",'
    '\\"rating\\":1691.0,\\"votes\\":12000,\\"modelOrganization\\":\\"Anthropic\\"}]}'
    '"])</script></body></html>'
)

OPENROUTER_PAGE = json.dumps({"data": [
    {
        "id": "anthropic/claude-opus-5", "name": "Anthropic: Claude Opus 5",
        "created": 1767225600, "context_length": 1000000,
        "architecture": {"input_modalities": ["text", "image"],
                         "output_modalities": ["text"]},
        "pricing": {"prompt": "0.000005", "completion": "0.000025",
                    "input_cache_read": "0.0000005",
                    "input_cache_write": "0.00000625"},
        "top_provider": {"max_completion_tokens": 128000},
        "supported_parameters": ["tools", "reasoning", "max_tokens"],
        "benchmarks": {"artificial_analysis": {"intelligence_index": 71.2,
                                               "coding_index": 80.0,
                                               "agentic_index": 62.0}},
    },
    {
        # Meta-model: negative price means "resolved at request time".
        "id": "openrouter/auto", "name": "Auto Router",
        "context_length": 2000000,
        "architecture": {"output_modalities": ["text"]},
        "pricing": {"prompt": "-1", "completion": "-1"},
        "supported_parameters": ["tools"],
    },
    {
        # Image generator: not a routing target for a coding agent.
        "id": "openai/gpt-image-2", "name": "OpenAI: GPT Image 2",
        "architecture": {"output_modalities": ["image"]},
        "pricing": {"prompt": "0.00001", "completion": "0.00004"},
    },
]})


class TestArenaParser:
    def test_every_board_is_extracted_from_the_escaped_payload(self):
        boards = parse_lmarena(ARENA_PAGE)
        assert set(boards) == {"text-overall-style_control", "webdev-overall-raw"}
        assert len(boards["text-overall-style_control"]) == 2

    def test_a_layout_change_raises_instead_of_returning_nothing(self):
        """An empty parse must not look like 'no model is any good'."""
        with pytest.raises(FetchFailed, match="layout changed"):
            parse_lmarena("<html><body>redesigned</body></html>")

    def test_bytes_and_str_both_parse(self):
        assert parse_lmarena(ARENA_PAGE.encode()) == parse_lmarena(ARENA_PAGE)

    def test_effort_variants_fold_into_one_entry_keeping_the_best(self):
        """The arena ranks -high and -max separately; routing wants the model."""
        entries = {e.key: e for e in lmarena_entries(parse_lmarena(ARENA_PAGE))}
        assert "claude-opus-5" in entries
        e = entries["claude-opus-5"]
        assert e.elo["webdev"] == 1691.0      # the better of -high and -max
        assert e.elo["text"] == 1500.5

    def test_board_ratings_are_kept_apart(self):
        """Folding boards together would rank a prose model as a coding model."""
        e = {x.key: x for x in lmarena_entries(parse_lmarena(ARENA_PAGE))}["claude-opus-5"]
        assert e.rating() == 1691.0           # code board wins for a coding router
        assert e.rating(boards=("text",)) == 1500.5

    def test_license_carries_through_so_open_weights_is_answerable(self):
        entries = {e.key: e for e in lmarena_entries(parse_lmarena(ARENA_PAGE))}
        assert entries["qwen4"].open_weights
        assert not entries["claude-opus-5"].open_weights

    def test_arena_prices_are_never_marked_verified(self):
        for e in lmarena_entries(parse_lmarena(ARENA_PAGE)):
            assert not e.verified


class TestAggregatorParser:
    def test_prices_convert_from_per_token_to_per_million(self):
        e = {x.key: x for x in parse_openrouter(OPENROUTER_PAGE)}["claude-opus-5"]
        assert e.inp == 5.0 and e.out == 25.0
        assert e.cache_read == 0.5 and e.cache_write == 6.25

    def test_negative_sentinel_prices_become_unknown(self):
        """Left alone, a -1 sorts first and the router recommends being paid."""
        e = {x.key: x for x in parse_openrouter(OPENROUTER_PAGE)}["auto"]
        assert e.inp is None and e.out is None and not e.priced

    def test_image_generators_are_not_routing_targets(self):
        assert "gpt-image-2" not in {x.key for x in parse_openrouter(OPENROUTER_PAGE)}

    def test_capability_metadata_survives(self):
        e = {x.key: x for x in parse_openrouter(OPENROUTER_PAGE)}["claude-opus-5"]
        assert e.supports_tools and e.supports_reasoning
        assert e.context == 1_000_000 and e.max_output == 128_000
        assert e.coding == 80.0 and e.released == "2026-01-01"

    def test_unexpected_payload_shape_raises(self):
        with pytest.raises(FetchFailed, match="payload shape"):
            parse_openrouter('{"data": {"not": "a list"}}')


class TestRefresh:
    def test_replaying_captures_needs_no_network(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ADDER_OFFLINE", "1")
        (tmp_path / "a.html").write_text(ARENA_PAGE)
        (tmp_path / "o.json").write_text(OPENROUTER_PAGE)
        cat, results = refresh(offline_files={"lmarena": tmp_path / "a.html",
                                              "openrouter": tmp_path / "o.json"})
        assert all(r.ok for r in results)
        e = cat.get("claude-opus-5")
        assert e.cache_read == 0.5 and e.rating() == 1691.0
        assert set(e.sources) == {"openrouter", "lmarena"}

    def test_one_dead_source_degrades_instead_of_failing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ADDER_OFFLINE", "1")
        (tmp_path / "o.json").write_text(OPENROUTER_PAGE)
        cat, results = refresh(offline_files={"openrouter": tmp_path / "o.json"})
        by = {r.name: r for r in results}
        assert by["openrouter"].ok and not by["lmarena"].ok
        assert cat.get("claude-opus-5") is not None
        assert cat.provenance["sources"][1]["error"]

    def test_offline_env_var_blocks_the_socket(self, monkeypatch):
        monkeypatch.setenv("ADDER_OFFLINE", "1")
        with pytest.raises(Offline):
            fetch("https://example.invalid/")

    def test_provenance_records_what_actually_happened(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ADDER_OFFLINE", "1")
        (tmp_path / "o.json").write_text(OPENROUTER_PAGE)
        cat, _ = refresh(offline_files={"openrouter": tmp_path / "o.json"})
        assert cat.provenance["refreshed_at"]
        names = {s["name"] for s in cat.provenance["sources"]}
        assert names == {"openrouter", "lmarena"}
