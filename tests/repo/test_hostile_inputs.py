"""Every reader takes input this tool does not own, and must not raise on it.

Three surfaces read untrusted data: transcript records under `~/.claude`, which
adder reads and never writes; a catalog file users are invited to hand-edit; and
tool inputs, which are whatever the model emitted. Each has a stated promise
about surviving that -- `ingest._int` says "Never raises on a log field",
`Entry.from_json` says "the types cannot be trusted", `guard` says "Everything
here fails open" -- and each raised on values reachable from ordinary use.

The worst of them is the guard, because its only handler is a blanket `except`
in the hook. An exception there is not an error anybody sees; it is the guard
quietly declining to guard, which this package calls its worst failure mode
precisely because it is invisible.
"""
from __future__ import annotations

import itertools
from typing import ClassVar

import pytest

from adder.core.ingest import usage_from
from adder.core.shapes import empty_model, is_bounded, read_estimate, shape
from adder.core.trace import _turn_from_record
from adder.decide.guard import GuardState, decide, needs_pricing, observe
from adder.pricing.catalog import Entry

# Values a JSON log or a tool call can actually carry. `Infinity` and `NaN` are
# here because `json.loads` accepts both literals by default.
HOSTILE = [None, "", 12, -5, 1.9, True, False, [], {}, "abc", float("nan"),
           float("inf"), float("-inf"), 10**20, {"a": 1}, [1, 2], "1e5",
           "\x00bad", "/etc/hosts"]


def _pairs(n=2):
    return itertools.product(HOSTILE, repeat=n)


class TestTheTranscriptReader:
    def test_no_usage_value_can_raise(self):
        for a, b in _pairs():
            rec = {"type": "assistant", "sessionId": "s",
                   "timestamp": "2026-08-10T12:00:00Z",
                   "message": {"id": "m", "model": "claude-opus-5",
                               "usage": {"input_tokens": a,
                                         "cache_read_input_tokens": b,
                                         "cache_creation_input_tokens": a,
                                         "output_tokens": b}}}
            t = _turn_from_record(rec, "p", "proj", True, None)
            if t is not None:
                t.cost(), t.context, t.input_cost()

    def test_counts_are_non_negative_ints(self):
        for a, b in _pairs():
            rec = {"type": "assistant", "sessionId": "s",
                   "message": {"id": "m", "model": "claude-opus-5",
                               "usage": {"input_tokens": a, "output_tokens": b}}}
            t = _turn_from_record(rec, "p", "proj", True, None)
            if t is not None:
                assert isinstance(t.uncached_in, int) and t.uncached_in >= 0
                assert isinstance(t.out, int) and t.out >= 0


class TestTheCatalogReader:
    def test_no_field_can_raise(self):
        for a, b in _pairs():
            e = Entry.from_json({"key": "k", "id": "i", "inp": a, "out": b,
                                 "context": a, "modalities": b, "params": a,
                                 "elo": b, "elo_lo": a, "elo_hi": b, "votes": a,
                                 "sources": b, "name": a, "org": b, "license": a})
            e.to_json(), e.priced, e.rating(), e.open_weights

    def test_a_string_modality_is_one_modality(self):
        """`tuple("text")` is four characters, not one modality."""
        assert Entry.from_json({"key": "k", "modalities": "text"}).modalities == ("text",)

    def test_a_non_iterable_modality_is_dropped(self):
        assert Entry.from_json({"key": "k", "modalities": 12}).modalities == ()

    def test_a_non_mapping_rating_is_dropped(self):
        assert Entry.from_json({"key": "k", "elo": "abc"}).elo == {}


class TestTheIngestAdapters:
    SHAPES: ClassVar[list] = [
        lambda a, b: {"model": "gpt-5", "usage": {"prompt_tokens": a,
                      "completion_tokens": b,
                      "prompt_tokens_details": {"cached_tokens": a}}},
        lambda a, b: {"model": "gpt-5", "usage": {"input_tokens": a,
                      "output_tokens": b,
                      "input_tokens_details": {"cached_tokens": a}}},
        lambda a, b: {"type": "message", "model": "claude-opus-5",
                      "usage": {"input_tokens": a, "output_tokens": b,
                                "cache_read_input_tokens": a,
                                "cache_creation_input_tokens": b}},
        lambda a, b: {"model": "gemini-3", "usageMetadata": {
            "promptTokenCount": a, "candidatesTokenCount": b,
            "cachedContentTokenCount": a}},
    ]

    def test_no_adapter_raises(self):
        for mk in self.SHAPES:
            for a, b in _pairs():
                u = usage_from(mk(a, b))
                if u is None:
                    continue
                for f in ("uncached_in", "cache_read", "cache_write", "out",
                          "thinking"):
                    v = getattr(u, f)
                    assert isinstance(v, int) and v >= 0, f"{f}={v!r}"


class TestTheGuard:
    TOOLS: ClassVar[tuple] = ("Read", "Bash", "Grep", "Glob", "WebFetch", "Task", "Write", "Edit")

    def test_no_tool_input_can_raise(self):
        for tool in self.TOOLS:
            for a, b in _pairs():
                inp = {"file_path": a, "limit": b, "command": a, "pattern": b,
                       "output_mode": a, "head_limit": b, "url": a, "offset": b}
                needs_pricing(tool, inp)
                v = decide(tool, inp, model="claude-opus-5", remaining_turns=400)
                observe(tool, inp, GuardState(), v, sizes=empty_model())
                v.payload()

    @pytest.mark.parametrize("cmd", [12, 1.9, True, None, [], {}])
    def test_the_shell_parser_takes_anything(self, cmd):
        shape(cmd)
        is_bounded(cmd)

    @pytest.mark.parametrize("fp", [12, None, [], {}, "\x00bad"])
    def test_the_read_sizer_takes_anything(self, fp):
        read_estimate({"file_path": fp, "limit": float("inf"), "offset": fp})


class TestTheCatalogsUncoercedFields:
    """Three fields escaped the coercion the loader's own docstring promises.

    "the types cannot be trusted ... Coerce what is recoverable and drop what
    is not." `rating_variant` reaches `.endswith` inside `select.rank`,
    `fetched_at` reaches `.replace` inside `age_days`, and `released` is
    printed as text -- and all three were carried through verbatim.
    """

    def test_rating_variant_is_always_a_string(self):
        for v in HOSTILE:
            e = Entry.from_json({"key": "k", "rating_variant": v})
            assert isinstance(e.rating_variant, str)
            e.rating_variant.endswith(e.key)      # what select.rank does

    def test_fetched_at_is_a_string_or_none(self):
        for v in HOSTILE:
            e = Entry.from_json({"key": "k", "fetched_at": v})
            assert e.fetched_at is None or isinstance(e.fetched_at, str)
            e.age_days()                          # must not raise

    def test_released_is_a_string_or_none(self):
        for v in HOSTILE:
            e = Entry.from_json({"key": "k", "released": v})
            assert e.released is None or isinstance(e.released, str)

    def test_a_naive_fetched_at_still_ages(self):
        """`2026-08-01` is an ordinary thing to hand-type into an override."""
        assert Entry.from_json({"key": "k", "fetched_at": "2026-08-01"}).age_days() >= 0
