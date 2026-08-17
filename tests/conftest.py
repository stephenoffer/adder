"""Shared builders for synthetic sessions and transcripts.

Every test file used to grow its own `_sess()` helper, and they had drifted:
some produced turns with timestamps, some did not, and a test that passed
against one shape failed against the other for reasons that had nothing to do
with what it was testing.

These are fixtures rather than plain functions so that a test declares what it
needs in its signature, and so nothing here can be imported into library code
by accident.

Nothing in this file reads `~/.claude`. A test that touches real transcripts is
marked `transcripts` and skipped in CI; see `pyproject.toml`.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from adder.core.trace import Session, Turn

OPUS = "claude-opus-5"
HAIKU = "claude-haiku-4-5"
SONNET = "claude-sonnet-5"

START = datetime(2026, 8, 1, 9, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def make_turn():
    def _make(*, model=OPUS, session="s", project="proj", read=20_000, write=0,
              uncached=0, out=400, thinking=0, sidechain=False, minutes=0,
              tools=(), ttl="5m", speed="standard", effort="high", msg_id="",
              ts=...):
        # `...` means "derive from `minutes`"; `None` means "no timestamp at
        # all", which is a case several reports have to handle and a default of
        # None would make untestable.
        when = (START + timedelta(minutes=minutes)).isoformat() if ts is ... else ts
        return Turn(session, project, model, uncached_in=uncached, cache_read=read,
                    cache_write=write, out=out, thinking=thinking,
                    sidechain=sidechain, ts=when, ttl=ttl, speed=speed,
                    msg_id=msg_id, tools=tuple(tools), effort=effort)
    return _make


@pytest.fixture
def make_session(make_turn):
    """A session whose context grows the way a real one does.

    Growth matters: several reports key off the difference between the smallest
    context (the irreducible base prompt) and the largest, and a session whose
    turns all carry the same context makes those reports return zero for
    reasons unrelated to the code under test.
    """
    def _make(n_turns=50, *, model=OPUS, sid="s", project="proj", base=20_000,
              growth=2_000, out=400, sidechain=False, minutes_apart=2, **kw):
        s = Session(sid, project)
        for i in range(n_turns):
            s.turns.append(make_turn(model=model, session=sid, project=project,
                                     read=base + growth * i, out=out,
                                     sidechain=sidechain,
                                     minutes=i * minutes_apart, **kw))
        return s
    return _make


@pytest.fixture
def make_sessions(make_session):
    def _make(n=3, n_turns=50, **kw):
        return {f"s{i}": make_session(n_turns, sid=f"s{i}", **kw) for i in range(n)}
    return _make


@pytest.fixture
def write_jsonl(tmp_path):
    """Write raw transcript records and return the directory holding them."""
    def _write(records, name="s.jsonl", into=None):
        d = into or tmp_path
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text("\n".join(json.dumps(r) for r in records))
        return d
    return _write


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Point every user-level file adder writes or reads at a temp directory.

    Without this a test can pick up the developer's real outcome log, ledger,
    parse cache, or config, and then pass or fail for reasons that do not exist
    in CI.
    """
    import adder.core.settings as settings
    import adder.core.trace as trace

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(trace, "CACHE_PATH", home / ".adder-trace-cache")
    monkeypatch.setattr(settings, "USER_FILE", home / "adder.json")
    monkeypatch.setenv("ADDER_LOG", str(home / "outcomes.jsonl"))
    monkeypatch.setenv("ADDER_LEDGER", str(home / "ledger.jsonl"))
    monkeypatch.setenv("ADDER_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    return home


# --------------------------------------------------------------------------
# Recorded upstream pages. Two test modules parse these -- `pricing/test_sources`
# checks the parsers, `decide/test_models` checks the refresh path that calls
# them -- and a test module may not import another test module, so they live
# here. Both are minimised captures of a real response; neither is fetched.
# --------------------------------------------------------------------------

# The arena page is a server-rendered React stream: the payload is real JSON,
# escaped once, embedded in a JS string. This is that shape, minimised.
_ARENA_PAGE = (
    '<!DOCTYPE html><html><body><script>self.__next_f.push([1,"'
    '{\\"id\\":\\"leaderboard-sets/public/leaderboards/text-overall-style_control'
    '/leaderboard-snapshots/latest\\",\\"entries\\":['
    '{\\"rank\\":1,\\"modelKey\\":\\"claude-opus-5-max-text\\",'
    '\\"modelDisplayName\\":\\"claude-opus-5-max\\",\\"rating\\":1500.5,'
    '\\"ratingLower\\":1490.0,\\"ratingUpper\\":1511.0,'
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

_OPENROUTER_PAGE = json.dumps({"data": [
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


@pytest.fixture
def arena_page() -> str:
    """A minimised capture of the LMArena leaderboard page."""
    return _ARENA_PAGE


@pytest.fixture
def openrouter_page() -> str:
    """A minimised capture of the OpenRouter models endpoint."""
    return _OPENROUTER_PAGE


@pytest.fixture
def tz(monkeypatch):
    """Pin the process timezone for one test, and restore it afterwards.

    Days are local in this package -- `parse_date("today")` is `date.today()` --
    so anything asserting which day a timestamp falls on has to say which zone
    it means, or it passes on the author's machine and fails in CI.
    """
    import time

    def _set(name: str):
        monkeypatch.setenv("TZ", name)
        time.tzset()

    yield _set
    time.tzset()          # monkeypatch has restored TZ by now
