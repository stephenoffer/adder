"""Read a usage record written by anything, not just Claude Code.

Why this exists
---------------
`trace.py` parses one format: the JSONL Claude Code writes to
`~/.claude/projects`. Everything downstream -- every report, every gate, every
dollar in this repo -- is computed from `Turn` records produced by that one
parser. So "adder works with any model" was only ever half true: the cost model
could be taught other providers, but there was no way to get a non-Claude
session *into* it.

This module is the other half. It normalizes a usage record from any of the
common shapes into the same `Turn`, so an OpenAI agent loop, a Gemini session,
a LiteLLM proxy log, or an OpenTelemetry trace gets the same carry analysis,
the same placement gates, and the same reports.

The bug that makes this hard, and that a naive adapter gets wrong
-----------------------------------------------------------------
**Providers disagree about whether the cached prefix is included in the input
count.**

* Anthropic: `input_tokens` is the *uncached* input only. Cached tokens are
  reported separately in `cache_read_input_tokens`. The three are disjoint and
  sum to the context.
* OpenAI: `prompt_tokens` is the *whole* prompt, and
  `prompt_tokens_details.cached_tokens` is the part of that total which was
  served from cache. They overlap.
* Google: same as OpenAI. `promptTokenCount` includes
  `cachedContentTokenCount`.

Read an OpenAI record with Anthropic's semantics and every cached token is
counted twice: once at full input rate inside `prompt_tokens`, and once more at
the cache read rate. On a long agent session the cached prefix *is* the bill, so
this is not a small error -- it roughly doubles the reported input side and it
does it silently, because both numbers are present and both look plausible.
Every adapter below therefore states which convention it is converting *from*,
and the overlapping ones subtract.

The second thing that differs, and why `providers.py` had to come first
----------------------------------------------------------------------
Anthropic reports `cache_creation_input_tokens` because a cache write is an
explicit, separately-priced act. OpenAI and Google report no write count at all,
because under automatic caching there is no write to price -- an uncached prefix
is billed as ordinary input and the cache happens as a side effect. So these
adapters leave `cache_write` at zero and let the uncached tokens carry it, which
is correct precisely because `providers.py` prices an automatic-caching write at
the input rate. Inventing a `cache_write` here to "look like Anthropic" would
have re-introduced a 1.25x premium nobody was charged.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from adder.core.trace import Turn
from adder.pricing.registry import is_known, provider_for

# Format identifiers, in the order `sniff` tries them.
CLAUDE_CODE = "claude-code"
ANTHROPIC_API = "anthropic-api"
OPENAI_CHAT = "openai-chat"
OPENAI_RESPONSES = "openai-responses"
GEMINI = "gemini"
OTEL = "otel"
GENERIC = "generic"

FORMATS = (CLAUDE_CODE, ANTHROPIC_API, OPENAI_CHAT, OPENAI_RESPONSES, GEMINI,
           OTEL, GENERIC)


def _int(v: Any) -> int:
    """A non-negative int, or 0. Never raises on a log field.

    NaN and infinity are rejected before the conversion, not after. `int(inf)`
    raises `OverflowError` and `int(nan)` raises `ValueError`, and only the
    second was caught -- so a single `Infinity` anywhere in a usage block took
    down the whole read with an exception this function's own docstring
    promises it does not raise. That is reachable from ordinary input:
    `json.loads` accepts the `Infinity` and `NaN` literals by default, so any
    proxy that serialises a float division emits a record that lands here.

    `catalog._num` already screened both out for the same reason. This is that
    screen, applied to the other coercion helper in the package.
    """
    if v is None or isinstance(v, bool):
        return 0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0
    if f != f or f in (float("inf"), float("-inf")):
        return 0
    try:
        n = int(f)
    except (TypeError, ValueError, OverflowError):
        return 0
    return n if n > 0 else 0


def _first(d: dict, *keys: str) -> Any:
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _sub(d: Any, *keys: str) -> dict:
    """A nested mapping from an untrusted record, or an empty one.

    `(d.get("usage") or {}).get(...)` reads as safe and is not: a record whose
    `usage` is the string `"lots"` -- or whose `prompt_tokens_details` is a
    number, which a proxy that flattens its logs will emit -- is truthy, so the
    `or {}` never fires and the `.get` raises `AttributeError` out of an
    adapter whose whole contract is to normalise whatever it is handed.
    """
    if not isinstance(d, dict):
        return {}
    for k in keys:
        d = d.get(k)
        if not isinstance(d, dict):
            return {}
    return d


# ---------------------------------------------------------------------------
# Sniffing
# ---------------------------------------------------------------------------

def sniff(d: dict) -> str | None:
    """Which shape is this record? None if it carries no usage at all.

    Per record rather than per file on purpose. A proxy log interleaves calls
    to three providers in one file, and a format decided once from the first
    line would mis-read the other two -- with the input-token convention being
    exactly the thing it got wrong.
    """
    if not isinstance(d, dict):
        return None

    if d.get("type") == "assistant" and isinstance(d.get("message"), dict):
        return CLAUDE_CODE
    if d.get("type") == "message" and "usage" in d and "model" in d:
        return ANTHROPIC_API

    if "usageMetadata" in d or "usage_metadata" in d:
        return GEMINI

    # OTel: either a flat attribute dict or a span with one.
    attrs = d.get("attributes") if isinstance(d.get("attributes"), dict) else d
    if isinstance(attrs, dict) and any(
        k.startswith("gen_ai.usage.") for k in attrs
    ):
        return OTEL

    usage = d.get("usage")
    if isinstance(usage, dict):
        if "prompt_tokens" in usage or "completion_tokens" in usage:
            return OPENAI_CHAT
        if "input_tokens" in usage or "output_tokens" in usage:
            # Both the Responses API and the Anthropic Messages API use these
            # names, and they mean different things -- see `_from_openai_responses`.
            # Either cache field is a tell: both are Anthropic-only spellings.
            #
            # Keying on `cache_creation_input_tokens` alone was not enough. A
            # turn that read a cached prefix and wrote nothing carries only
            # `cache_read_input_tokens`, and any proxy that drops zero-valued
            # fields emits exactly that. Such a record sniffed as OpenAI, and
            # the Responses adapter looks for the cached count under
            # `input_tokens_details.cached_tokens` -- so it found none and the
            # whole cached prefix was silently dropped. On a long session that
            # is ~76% of the bill going to zero, with every number still
            # looking plausible.
            if ("cache_creation_input_tokens" in usage
                    or "cache_read_input_tokens" in usage):
                return ANTHROPIC_API
            return OPENAI_RESPONSES

    if any(k in d for k in ("input_tokens", "prompt_tokens", "output_tokens",
                            "completion_tokens")):
        return GENERIC
    return None


# ---------------------------------------------------------------------------
# Adapters. Each returns the five token counts plus metadata, or None.
# ---------------------------------------------------------------------------

class Usage:
    """Normalized to Anthropic's disjoint convention, which `Turn` expects.

    `uncached_in`, `cache_read` and `cache_write` never overlap. Every adapter
    is responsible for getting into this convention from whatever its provider
    uses; nothing downstream has to know which provider it came from.
    """

    __slots__ = (
        "cache_read",
        "cache_write",
        "effort",
        "model",
        "msg_id",
        "out",
        "session",
        "sidechain",
        "speed",
        "thinking",
        "tools",
        "ts",
        "ttl",
        "uncached_in",
    )

    def __init__(self, **kw: Any):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def to_turn(self, *, session: str, project: str) -> Turn:
        # Every identity field is coerced here rather than in six adapters. A
        # foreign log puts objects in these routinely -- a LiteLLM router
        # rewrites `id` to a dict when it retries -- and a `msg_id` that is not
        # a string is a dict key in `iter_turns` and a set element in
        # `trace.load_sessions`, both of which raise `unhashable type` a long
        # way from the line that caused it.
        return Turn(
            session=str(self.session or session),
            project=project,
            model=self.model,
            uncached_in=self.uncached_in or 0,
            cache_read=self.cache_read or 0,
            cache_write=self.cache_write or 0,
            out=self.out or 0,
            thinking=self.thinking or 0,
            sidechain=bool(self.sidechain),
            ts=self.ts if isinstance(self.ts, str) else None,
            ttl=str(self.ttl or _default_ttl(self.model)),
            speed=str(self.speed or "standard"),
            msg_id=str(self.msg_id or ""),
            tools=tuple(str(x) for x in (self.tools or ())),
            effort=str(self.effort or ""),
        )


def _default_ttl(model: str | None) -> str:
    """The TTL label to record when the source format does not say.

    Not the string "5m". A record from a provider whose cache lifetime is a
    single non-selectable value should carry *that* label, so the TTL-choice
    reports can tell "this workload picked the short TTL" apart from "this
    provider only has one".
    """
    return provider_for(model or "").default_ttl


def _from_openai_chat(d: dict) -> Usage | None:
    """Chat Completions. `prompt_tokens` INCLUDES the cached part; subtract it.

    This is the conversion that matters. `prompt_tokens: 50000` with
    `cached_tokens: 48000` is a turn that paid full rate for 2,000 tokens and
    the cache rate for 48,000. Carried across verbatim it would report 50,000
    uncached tokens *and* 48,000 cache reads -- a context of 98,000 that never
    existed, priced about twice what it cost.
    """
    u = _sub(d, "usage")
    prompt = _int(_first(u, "prompt_tokens"))
    cached = _int(_sub(u, "prompt_tokens_details").get("cached_tokens"))
    reasoning = _int(_sub(u, "completion_tokens_details").get("reasoning_tokens"))
    cached = min(cached, prompt)          # a log that disagrees with itself
    return Usage(
        uncached_in=prompt - cached,
        cache_read=cached,
        cache_write=0,                    # automatic caching: no write to price
        out=_int(_first(u, "completion_tokens")),
        thinking=reasoning,
        model=_first(d, "model") or "",
        msg_id=_first(d, "id", "request_id") or "",
        ts=_ts(d),
        session=_first(d, "session_id", "conversation_id", "thread_id"),
        effort=str(_first(d, "reasoning_effort") or ""),
        tools=_openai_tools(d),
    )


def _from_openai_responses(d: dict) -> Usage | None:
    """Responses API. Same overlapping convention as Chat Completions.

    The field names (`input_tokens`, `output_tokens`) collide exactly with
    Anthropic's, which mean the opposite thing -- Anthropic's `input_tokens`
    excludes the cache, this one includes it. Two formats, one spelling, and
    the difference is a factor of two on the input side. `sniff` separates
    them on `cache_creation_input_tokens`, which only Anthropic emits.
    """
    u = _sub(d, "usage")
    inp = _int(_first(u, "input_tokens"))
    cached = _int(_sub(u, "input_tokens_details").get("cached_tokens"))
    reasoning = _int(_sub(u, "output_tokens_details").get("reasoning_tokens"))
    cached = min(cached, inp)
    return Usage(
        uncached_in=inp - cached,
        cache_read=cached,
        cache_write=0,
        out=_int(_first(u, "output_tokens")),
        thinking=reasoning,
        model=_first(d, "model") or "",
        msg_id=_first(d, "id") or "",
        ts=_ts(d),
        session=_first(d, "conversation", "previous_response_id"),
        effort=str(_sub(d, "reasoning").get("effort") or ""),
        tools=_responses_tools(d),
    )


def _from_anthropic_api(d: dict) -> Usage | None:
    """The Messages API directly, rather than through Claude Code.

    Already in the disjoint convention, so nothing is subtracted. Worth having
    as its own adapter anyway: anyone driving the API themselves has no
    `~/.claude/projects` directory to point adder at.
    """
    u = _sub(d, "usage")
    return Usage(
        uncached_in=_int(u.get("input_tokens")),
        cache_read=_int(u.get("cache_read_input_tokens")),
        cache_write=_int(u.get("cache_creation_input_tokens")),
        out=_int(u.get("output_tokens")),
        thinking=_int(_sub(u, "output_tokens_details").get("thinking_tokens")),
        model=d.get("model") or "",
        msg_id=str(d.get("id") or ""),
        ts=_ts(d),
        ttl=_anthropic_ttl(u),
    )


def _anthropic_ttl(u: dict) -> str | None:
    """Dominant write TTL, from the 5m/1h split when the record carries one."""
    detail = u.get("cache_creation") if isinstance(u, dict) else None
    if not isinstance(detail, dict):
        return None
    five = _int(_first(detail, "ephemeral_5m_input_tokens"))
    hour = _int(_first(detail, "ephemeral_1h_input_tokens"))
    if not five and not hour:
        return None
    return "1h" if hour > five else "5m"


def _from_gemini(d: dict) -> Usage | None:
    """`promptTokenCount` INCLUDES `cachedContentTokenCount`; subtract it.

    Gemini also reports `thoughtsTokenCount` *outside* `candidatesTokenCount`,
    unlike every other provider here, where reasoning is part of the output
    count. Adding thinking to output on top would double-count it, so it is
    recorded on the side and the output count is left alone.
    """
    u = _sub(d, "usageMetadata") or _sub(d, "usage_metadata")
    prompt = _int(_first(u, "promptTokenCount", "prompt_token_count"))
    cached = _int(_first(u, "cachedContentTokenCount", "cached_content_token_count"))
    out = _int(_first(u, "candidatesTokenCount", "candidates_token_count"))
    think = _int(_first(u, "thoughtsTokenCount", "thoughts_token_count"))
    cached = min(cached, prompt)
    return Usage(
        uncached_in=prompt - cached,
        cache_read=cached,
        cache_write=0,
        out=out + think,          # bill both; `thinking` records the split
        thinking=think,
        model=_first(d, "modelVersion", "model_version", "model") or "",
        msg_id=_first(d, "responseId", "response_id") or "",
        ts=_ts(d),
    )


_OTEL_KEYS = {
    "uncached_in": ("gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens"),
    "cache_read": ("gen_ai.usage.cache_read_input_tokens",
                   "gen_ai.usage.cached_input_tokens"),
    "cache_write": ("gen_ai.usage.cache_creation_input_tokens",),
    "out": ("gen_ai.usage.output_tokens", "gen_ai.usage.completion_tokens"),
    "thinking": ("gen_ai.usage.reasoning_tokens",),
}


def _from_otel(d: dict) -> Usage | None:
    """OpenTelemetry GenAI semantic conventions.

    The convention does not say whether `gen_ai.usage.input_tokens` includes
    the cached part, and instrumentations differ, because each one mirrors the
    SDK underneath it. So the provider decides: for the vendors whose native
    field overlaps, the cached count is subtracted; for the rest it is not.
    Guessing uniformly in either direction would be wrong for half the corpus.
    """
    a = d.get("attributes") if isinstance(d.get("attributes"), dict) else d
    got = {}
    for field, keys in _OTEL_KEYS.items():
        got[field] = _int(_first(a, *keys))
    model = _first(a, "gen_ai.response.model", "gen_ai.request.model") or ""
    if _overlaps(model):
        got["uncached_in"] = max(0, got["uncached_in"] - got["cache_read"])
    return Usage(
        **got,
        model=model,
        msg_id=_first(a, "gen_ai.response.id") or _first(d, "span_id", "spanId") or "",
        ts=_ts(d),
        session=_first(a, "gen_ai.conversation.id", "session.id"),
        effort=str(_first(a, "gen_ai.request.reasoning_effort") or ""),
    )


def _overlaps(model: str) -> bool:
    """Does this provider's input count include the cached prefix?

    Anthropic reports the three counts disjointly. OpenAI and Google report a
    total with the cached part inside it. Anything unrecognised is treated as
    disjoint, because subtracting a cache read that was never included would
    understate the input side, and understating spend is the failure this repo
    is least willing to ship.
    """
    return provider_for(model).name in ("openai", "google")


def _from_generic(d: dict) -> Usage | None:
    """A flat record with recognisable names, e.g. a hand-rolled log.

    Deliberately the last resort. It reads whichever spelling it finds and
    applies the same provider-driven overlap rule as the OTel adapter.
    """
    u = d.get("usage") if isinstance(d.get("usage"), dict) else d
    inp = _int(_first(u, "input_tokens", "prompt_tokens", "in_tokens"))
    cached = _int(_first(u, "cache_read_input_tokens", "cached_tokens",
                         "cache_read", "cached_input_tokens"))
    write = _int(_first(u, "cache_creation_input_tokens", "cache_write"))
    out = _int(_first(u, "output_tokens", "completion_tokens", "out_tokens"))
    model = _first(d, "model", "model_id", "modelId") or ""
    if _overlaps(model):
        cached = min(cached, inp)
        inp -= cached
    return Usage(
        uncached_in=inp, cache_read=cached, cache_write=write, out=out,
        thinking=_int(_first(u, "reasoning_tokens", "thinking_tokens")),
        model=model,
        msg_id=_first(d, "id", "message_id", "request_id") or "",
        ts=_ts(d),
        session=_first(d, "session", "session_id", "conversation_id"),
    )


_ADAPTERS = {
    ANTHROPIC_API: _from_anthropic_api,
    OPENAI_CHAT: _from_openai_chat,
    OPENAI_RESPONSES: _from_openai_responses,
    GEMINI: _from_gemini,
    OTEL: _from_otel,
    GENERIC: _from_generic,
}


def _ts(d: dict) -> str | None:
    """An ISO timestamp, from whichever field the format uses.

    Epoch seconds are converted rather than dropped: `created: 1767225600` is a
    timestamp, and without it every idle-gap and TTL report silently has
    nothing to measure.
    """
    for k in ("timestamp", "created_at", "start_time", "startTime", "time"):
        v = d.get(k)
        if isinstance(v, str) and v:
            return v
    for k in ("created", "created_at", "start_time_unix_nano"):
        v = d.get(k)
        if isinstance(v, (int, float)) and v > 0:
            from datetime import datetime, timezone
            secs = float(v)
            # OTel records nanoseconds; anything this large is not seconds.
            if secs > 1e17:
                secs /= 1e9
            try:
                return datetime.fromtimestamp(secs, tz=timezone.utc).isoformat()
            except (OverflowError, OSError, ValueError):
                return None
    return None


def _openai_tools(d: dict) -> tuple[str, ...]:
    """Tool names invoked by this response, for the per-tool carry report."""
    names: list[str] = []
    for choice in d.get("choices") or ():
        if not isinstance(choice, dict):
            continue
        msg = choice.get("message") or {}
        for call in msg.get("tool_calls") or ():
            if isinstance(call, dict):
                fn = call.get("function") or {}
                name = fn.get("name") or call.get("name")
                if name:
                    names.append(str(name))
    return tuple(dict.fromkeys(names))


def _responses_tools(d: dict) -> tuple[str, ...]:
    """Tool names invoked by a Responses API turn.

    A different shape from Chat Completions -- calls are `function_call` items
    in the flat `output` array rather than `tool_calls` on a choice's message --
    and the adapter carried no extraction at all, so every Responses record
    arrived with no tools. `adder tools` and `trace --by tool` then filed all of
    that spend under "(no tool call)", which reads as "this workload calls no
    tools" rather than "this format was not read".
    """
    names: list[str] = []
    for item in d.get("output") or ():
        if not isinstance(item, dict):
            continue
        if item.get("type") not in ("function_call", "custom_tool_call",
                                    "tool_call"):
            continue
        name = item.get("name") or (item.get("function") or {}).get("name")
        if name:
            names.append(str(name))
    return tuple(dict.fromkeys(names))


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

def usage_from(d: dict, *, fmt: str | None = None) -> Usage | None:
    """Normalize one record, or None if it carries no usage.

    `fmt` forces an adapter. Use it when a log is known to be one shape and the
    sniffer would have to guess -- but prefer letting it sniff, because a mixed
    proxy log is the common case and the whole point of sniffing per record.
    """
    kind = fmt or sniff(d)
    if kind is None or kind == CLAUDE_CODE:
        # Claude Code has its own parser in `trace`, which also handles the
        # per-content-block deduplication this module has no view of.
        return None
    fn = _ADAPTERS.get(kind)
    if fn is None:
        return None
    got = fn(d)
    if got is None:
        return None
    # A model id that is not a string cannot be resolved, and reaching the
    # registry with one raises `TypeError: unhashable type` from inside an
    # `lru_cache` -- far from the log line that caused it. Foreign logs put
    # objects in this field routinely: LiteLLM writes `{"model": {"name": ...}}`
    # when a router rewrites the request.
    got.model = got.model if isinstance(got.model, str) else ""
    if not got.model:
        return None
    if not any((got.uncached_in, got.cache_read, got.cache_write, got.out)):
        return None
    return got


def turn_from(d: dict, *, session: str, project: str = "",
              fmt: str | None = None) -> Turn | None:
    got = usage_from(d, fmt=fmt)
    return None if got is None else got.to_turn(session=session, project=project)


def iter_records(path: Path) -> Iterator[dict]:
    """Yield JSON objects from a `.json` or `.jsonl` file.

    Handles the three shapes these logs actually come in: one object per line,
    a top-level array, and a single object. A file that is none of them yields
    nothing rather than raising -- a malformed log should degrade one file, not
    the run.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    stripped = text.lstrip()
    if stripped.startswith("["):
        try:
            for row in json.loads(text):
                if isinstance(row, dict):
                    yield row
            return
        except (json.JSONDecodeError, ValueError):
            return
    saw = False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(row, dict):
            saw = True
            yield row
        elif isinstance(row, list):
            for r in row:
                if isinstance(r, dict):
                    saw = True
                    yield r
    if not saw and stripped.startswith("{"):
        try:
            row = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return
        if isinstance(row, dict):
            yield row


def iter_turns(path: Path, *, fmt: str | None = None,
               skip_unknown: bool = True,
               unknown: dict[str, int] | None = None) -> Iterator[Turn]:
    """Priced turns from one foreign log file.

    Deduplication is by `msg_id` where the format supplies one, because a
    retried request and a streamed response can both appear more than once --
    the same hazard that inflated the Claude Code numbers 1.78x before it was
    caught. Records with no id are kept in order and never merged, since
    without an id there is no way to tell a duplicate from a second call.
    """
    project = path.parent.name
    best: dict[str, Turn] = {}
    order: list[Turn] = []
    # Where each message id sits in `order`. `order.index(prev)` was both
    # quadratic and wrong: `Turn` is a plain dataclass, so `index` compares by
    # value and finds the *first equal* turn, which on a log with two identical
    # calls is not the one being replaced. `trace.iter_file` learned the same
    # lesson; this is that fix, applied to the foreign-log reader.
    at: dict[str, int] = {}
    for d in iter_records(path):
        t = turn_from(d, session=path.stem, project=project, fmt=fmt)
        if t is None:
            continue
        if not is_known(t.model):
            if unknown is not None:
                unknown[t.model] = unknown.get(t.model, 0) + 1
            if skip_unknown:
                continue
        if not t.msg_id:
            order.append(t)
            continue
        prev = best.get(t.msg_id)
        if prev is None:
            best[t.msg_id] = t
            at[t.msg_id] = len(order)
            order.append(t)
        elif t.out > prev.out:
            # A later record completed the stream. Replace in place so the
            # original ordering survives.
            order[at[t.msg_id]] = t
            best[t.msg_id] = t
    yield from order


def detect_formats(paths: Iterable[Path]) -> dict[str, int]:
    """How many records of each shape a set of files holds.

    For `adder doctor` and for telling a user why their log produced nothing:
    "1,200 records, all of them unrecognised" is a far more useful answer than
    an empty report.
    """
    tally: dict[str, int] = {}
    for p in paths:
        for d in iter_records(p):
            kind = sniff(d) or "unrecognised"
            tally[kind] = tally.get(kind, 0) + 1
    return tally
