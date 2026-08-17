"""Per-provider billing mechanics: how caching is paid for, not what it costs.

Why this is not `catalog.py` and not `prices.py`
------------------------------------------------
`prices.py` holds first-party Claude rates. `catalog.py` holds ~500 models
scraped from public sources. Neither holds the thing that actually decides an
agent-session cost model: **how a provider bills a cached prefix**.

That is a property of the provider, not the model, and it differs in ways that
are not a rounding error:

* **Anthropic** caching is *explicit*. You place a breakpoint, pay a 1.25x
  (5m) or 2.00x (1h) premium to write it, and read it back at 0.10x. Writing
  is a decision with a price, so "where do the breakpoints go" is a real lever.
* **OpenAI** caching is *automatic*. There is no breakpoint and no write
  premium -- an uncached prefix is billed at plain input rate and the cache
  happens as a side effect. Reads are discounted (0.25x on the 4.x family,
  0.10x on the 5.x family). "Place a breakpoint" is not a lever that exists.
* **Google** ships both: implicit caching (automatic, free to write, ~0.25x
  reads) and explicit context caching, which additionally bills *storage per
  hour* -- a term no other provider has, and one that makes a long idle gap
  cost money even when nothing is being read.
* **DeepSeek** caches to disk automatically, free to write, ~0.1x to read.
* Plenty of hosted open-weight endpoints do not cache at all. For those the
  carry term is not `0.10x * remaining_turns`, it is `1.0x * remaining_turns`
  -- ten times larger, and the single biggest number in the report.

The bug this module exists to prevent
-------------------------------------
Before it, `cost.py` applied Anthropic's `CACHE_READ_MULT = 0.10` and
`CACHE_WRITE_MULT = 1.25` to whatever model id it was handed. On a provider
with no caching that understates the carry term by 10x, which is not a
recommendation error, it is a recommendation pointed the wrong way: adder
would tell you to admit tokens to a context that could not amortize them.

What is verified and what is not
--------------------------------
`verified=True` means the mechanic was read off the provider's published
pricing page and is the kind of thing that changes with a launch, not a week.
`verified=False` means it is a documented default for a family of hosted
endpoints and should be treated as MODELLED -- the reports say so, and
`ADDER_PROVIDERS=<path>` overrides the whole table for a site that knows
better. Nothing here is a price; prices come from the catalog.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Cache styles
# ---------------------------------------------------------------------------
# The distinction is not cosmetic. It decides which levers adder is allowed to
# recommend at all:
#
#   explicit  -> breakpoint placement, TTL choice, and fan-out staggering are
#                all real decisions with prices attached.
#   automatic -> the provider caches for you. Breakpoint advice is noise.
#                Fan-out staggering still matters (the entry has to exist
#                before it can be read) and TTL is not selectable.
#   none      -> there is no cache. Every re-read costs full input rate, so the
#                carry term is ~10x what an Anthropic-shaped model assumes and
#                context discipline is the only lever left.
EXPLICIT = "explicit"
AUTOMATIC = "automatic"
NONE = "none"
CACHE_STYLES = (EXPLICIT, AUTOMATIC, NONE)


@dataclass(frozen=True)
class Provider:
    """How one provider bills the things an agent session does repeatedly."""

    name: str
    cache_style: str = NONE

    # Multipliers on the input rate, used ONLY when the catalog publishes no
    # absolute cache rate for a model. Absolute always wins: these are the
    # shape of the provider's pricing, not the price.
    cache_read_mult: float | None = None
    # Write premium by TTL label. `1.0` means "no premium" -- the normal case
    # for automatic caching, where an uncached prefix is simply input.
    cache_write_mult: dict[str, float] | None = None
    # Seconds each TTL label survives. An automatic cache still has a lifetime;
    # it is just not selectable, so it appears here with one entry.
    ttl_seconds: dict[str, int] | None = None
    # Smallest prefix that caches at all. Below it, caching silently does not
    # happen -- no error, no discount.
    cache_min: int = 0
    # Cache-write granularity. OpenAI caches in 128-token increments, so a
    # prefix of 1200 caches 1152 of it. Zero means "no quantization".
    cache_block: int = 0
    # Explicit-cache storage, USD per million tokens per hour. Google is the
    # only major provider that bills this, and omitting it makes a long idle
    # session look free when it is not.
    cache_storage_per_mtok_hour: float | None = None

    # Async batch discount, as a multiplier on the whole bill. None means the
    # provider publishes no batch tier.
    batch_mult: float | None = None

    # Reasoning-effort vocabulary. Providers disagree on both the names and the
    # type: Anthropic takes labels, Google takes an integer thinking budget.
    efforts: tuple[str, ...] = ()
    effort_is_budget: bool = False

    # Hard cap on cache breakpoints per request (explicit caching only).
    max_breakpoints: int = 0

    verified: bool = False
    notes: str = ""

    # ---- derived ------------------------------------------------------
    @property
    def caches(self) -> bool:
        return self.cache_style != NONE

    @property
    def has_write_premium(self) -> bool:
        """Does writing a cache entry cost more than plain input?

        Only true for explicit caching. This is what makes "should I cache
        this?" a question with a wrong answer; under automatic caching the
        answer is always yes and there is nothing to decide.
        """
        return any(m > 1.0 for m in (self.cache_write_mult or {}).values())

    @property
    def default_ttl(self) -> str:
        """The TTL label to assume when a record does not say."""
        if not self.ttl_seconds:
            return "5m"
        return min(self.ttl_seconds, key=lambda k: self.ttl_seconds[k])

    def write_mult(self, ttl: str | None = None) -> float:
        """Write premium for `ttl`, falling back to the provider's default.

        Unknown TTL labels do not raise. A transcript from one provider read
        under another's TTL vocabulary is a data problem, not a crash, and the
        conservative reading of an unknown label is the cheapest write the
        provider offers -- anything else invents spend.
        """
        table = self.cache_write_mult or {}
        if not table:
            return 1.0
        if ttl and ttl in table:
            return table[ttl]
        return table.get(self.default_ttl, min(table.values()))

    def ttl_for(self, ttl: str | None = None) -> int | None:
        table = self.ttl_seconds or {}
        if not table:
            return None
        if ttl and ttl in table:
            return table[ttl]
        return table.get(self.default_ttl)

    def supports_effort(self, level: str) -> bool:
        return level in self.efforts

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cache_style": self.cache_style,
            "cache_read_mult": self.cache_read_mult,
            "cache_write_mult": dict(self.cache_write_mult or {}),
            "ttl_seconds": dict(self.ttl_seconds or {}),
            "cache_min": self.cache_min,
            "cache_block": self.cache_block,
            "cache_storage_per_mtok_hour": self.cache_storage_per_mtok_hour,
            "batch_mult": self.batch_mult,
            "efforts": list(self.efforts),
            "effort_is_budget": self.effort_is_budget,
            "max_breakpoints": self.max_breakpoints,
            "verified": self.verified,
            "notes": self.notes,
        }

    @staticmethod
    def from_json(d: dict[str, Any]) -> Provider:
        """Build a provider from a file a human is invited to hand-edit.

        Same reasoning as `catalog.Entry.from_json`: coerce what is
        recoverable, drop what is not. A `cache_read_mult` written as a string
        used to load fine and then raise from inside a cost comparison.
        """
        style = str(d.get("cache_style", NONE)).lower()
        if style not in CACHE_STYLES:
            style = NONE
        return Provider(
            name=str(d.get("name", "")),
            cache_style=style,
            cache_read_mult=_num(d.get("cache_read_mult")),
            cache_write_mult=_mult_map(d.get("cache_write_mult")),
            ttl_seconds=_int_map(d.get("ttl_seconds")),
            cache_min=_int(d.get("cache_min")) or 0,
            cache_block=_int(d.get("cache_block")) or 0,
            cache_storage_per_mtok_hour=_num(d.get("cache_storage_per_mtok_hour")),
            batch_mult=_num(d.get("batch_mult")),
            efforts=tuple(str(e) for e in (d.get("efforts") or ())),
            effort_is_budget=bool(d.get("effort_is_budget", False)),
            max_breakpoints=_int(d.get("max_breakpoints")) or 0,
            verified=bool(d.get("verified", False)),
            notes=str(d.get("notes", "")),
        )


def _num(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def _int(v: Any) -> int | None:
    f = _num(v)
    return None if f is None else int(f)


def _mult_map(v: Any) -> dict[str, float] | None:
    if not isinstance(v, dict):
        return None
    out = {str(k): _num(x) for k, x in v.items()}
    clean = {k: x for k, x in out.items() if x is not None}
    return clean or None


def _int_map(v: Any) -> dict[str, int] | None:
    if not isinstance(v, dict):
        return None
    out = {str(k): _int(x) for k, x in v.items()}
    clean = {k: x for k, x in out.items() if x is not None}
    return clean or None


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------
# Keys are lowercase canonical vendor names, matching `catalog.normalize_org`.

ANTHROPIC = Provider(
    name="anthropic",
    cache_style=EXPLICIT,
    cache_read_mult=0.10,
    cache_write_mult={"5m": 1.25, "1h": 2.00},
    ttl_seconds={"5m": 300, "1h": 3600},
    cache_min=1024,            # per-model override lives in prices.py
    batch_mult=0.50,
    efforts=("low", "medium", "high", "xhigh", "max"),
    max_breakpoints=4,
    verified=True,
    notes="explicit cache_control breakpoints; write premium is a real decision",
)

OPENAI = Provider(
    name="openai",
    cache_style=AUTOMATIC,
    # 0.10x on the 5.x family, 0.25x on 4.x. The per-model absolute rate in the
    # catalog wins wherever it exists; this is the fallback shape.
    cache_read_mult=0.25,
    cache_write_mult={"auto": 1.00},   # no premium: an uncached prefix is input
    ttl_seconds={"auto": 600},         # ~5-10 min of inactivity
    cache_min=1024,
    cache_block=128,                   # caches in 128-token increments
    batch_mult=0.50,
    efforts=("minimal", "low", "medium", "high"),
    verified=True,
    notes="automatic prompt caching; no breakpoints and no write premium to optimise",
)

GOOGLE = Provider(
    name="google",
    cache_style=AUTOMATIC,
    cache_read_mult=0.25,
    cache_write_mult={"auto": 1.00},
    ttl_seconds={"auto": 3600},
    cache_min=1024,
    # Explicit context caching bills storage per hour on top of reads. This is
    # the term that makes an idle session cost money on Google and nowhere else.
    cache_storage_per_mtok_hour=1.00,
    batch_mult=0.50,
    efforts=("low", "medium", "high"),
    effort_is_budget=True,             # the API takes an integer thinking budget
    verified=False,
    notes="implicit caching is automatic and free to write; explicit caching adds "
          "per-hour storage. Storage rate is MODELLED and varies by model family.",
)

DEEPSEEK = Provider(
    name="deepseek",
    cache_style=AUTOMATIC,
    cache_read_mult=0.10,
    cache_write_mult={"auto": 1.00},
    ttl_seconds={"auto": 3600},
    cache_min=0,
    cache_block=64,
    batch_mult=None,
    efforts=(),
    verified=False,
    notes="automatic disk cache, free to write; hit/miss rates published per model",
)

XAI = Provider(
    name="spacexai",
    cache_style=AUTOMATIC,
    cache_read_mult=0.25,
    cache_write_mult={"auto": 1.00},
    ttl_seconds={"auto": 600},
    cache_min=0,
    batch_mult=None,
    efforts=("low", "high"),
    verified=False,
    notes="automatic caching; discount MODELLED from the published cached-input rate",
)

MISTRAL = Provider(
    name="mistral",
    cache_style=AUTOMATIC,
    cache_read_mult=0.25,
    cache_write_mult={"auto": 1.00},
    ttl_seconds={"auto": 300},
    batch_mult=0.50,
    verified=False,
    notes="MODELLED: cache behaviour varies by endpoint",
)

AMAZON = Provider(
    name="amazon",
    cache_style=EXPLICIT,
    cache_read_mult=0.10,
    cache_write_mult={"5m": 1.25},
    ttl_seconds={"5m": 300},
    cache_min=1024,
    batch_mult=0.50,
    max_breakpoints=4,
    verified=False,
    notes="Bedrock-hosted prompt caching mirrors the explicit model; MODELLED",
)

COHERE = Provider(name="cohere", cache_style=NONE, verified=False,
                  notes="no published prompt cache")

# Hosted open-weight endpoints. The default is NO CACHE, deliberately.
#
# This is the conservative direction and it is the one that used to be wrong.
# Assuming a cache that does not exist understates the carry term by 10x and
# makes adder recommend admitting tokens a context cannot amortize. Assuming no
# cache when there is one overstates carry, which produces advice that is
# merely too cautious. Between a report that is too cautious and one that is
# confidently pointed the wrong way, this repo picks too cautious.
UNKNOWN = Provider(
    name="unknown",
    cache_style=NONE,
    verified=False,
    notes="no cache economics known for this vendor; carry is priced at full "
          "input rate, which is the conservative direction",
)

_BUILTIN: dict[str, Provider] = {
    p.name: p
    for p in (ANTHROPIC, OPENAI, GOOGLE, DEEPSEEK, XAI, MISTRAL, AMAZON, COHERE)
}

# Vendors that route through another provider's economics, or that the catalog
# spells differently from the pricing page. Values are keys of `_BUILTIN`.
_ORG_ALIASES = {
    "anthropic": "anthropic",
    "claude": "anthropic",
    "openai": "openai",
    "azure": "openai",
    "azure openai": "openai",
    "google": "google",
    "google deepmind": "google",
    "deepmind": "google",
    "vertex": "google",
    "deepseek": "deepseek",
    "xai": "spacexai",
    "x.ai": "spacexai",
    "spacexai": "spacexai",
    "grok": "spacexai",
    "mistral": "mistral",
    "mistral ai": "mistral",
    "mistralai": "mistral",
    "amazon": "amazon",
    "aws": "amazon",
    "bedrock": "amazon",
    "cohere": "cohere",
}

# Model-id prefixes, for when the org is missing but the id is recognisable.
# Longest prefix wins, same rule as model resolution.
_ID_PREFIXES = (
    ("claude-", "anthropic"),
    ("anthropic/", "anthropic"),
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("chatgpt", "openai"),
    ("openai/", "openai"),
    ("gemini", "google"),
    ("google/", "google"),
    ("deepseek", "deepseek"),
    ("grok", "spacexai"),
    ("x-ai/", "spacexai"),
    ("mistral", "mistral"),
    ("codestral", "mistral"),
    ("devstral", "mistral"),
    ("magistral", "mistral"),
    ("amazon/", "amazon"),
    ("nova-", "amazon"),
    ("cohere/", "cohere"),
    ("command-", "cohere"),
)


def _overrides() -> dict[str, Provider]:
    """`ADDER_PROVIDERS=<path>` replaces or extends the built-in table.

    A site that has negotiated rates, or a provider that shipped a cache tier
    after this table was written, should not need a fork. Same layering rule as
    the catalog: the file wins field-by-field over the built-in.
    """
    path = os.environ.get("ADDER_PROVIDERS")
    if not path:
        return {}
    try:
        raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # A broken override degrades to the built-in table rather than taking
        # down every report. `adder doctor` reports the failure loudly.
        return {}
    out: dict[str, Provider] = {}
    entries = raw.get("providers", raw) if isinstance(raw, dict) else {}
    if not isinstance(entries, dict):
        return {}
    for name, d in entries.items():
        if not isinstance(d, dict):
            continue
        d = {**d, "name": str(name).lower()}
        base = _BUILTIN.get(str(name).lower())
        over = Provider.from_json(d)
        out[str(name).lower()] = _merge(base, over, d) if base else over
    return out


def _merge(base: Provider, over: Provider, raw: dict[str, Any]) -> Provider:
    """Field-wise overlay, keyed on what the file actually mentioned.

    Not a blanket `replace`: a file that pins only `cache_read_mult` must not
    silently reset the TTL table or blank `verified` back to its default.
    """
    fields: dict[str, Any] = {}
    for f in ("cache_style", "cache_read_mult", "cache_write_mult", "ttl_seconds",
              "cache_min", "cache_block", "cache_storage_per_mtok_hour",
              "batch_mult", "efforts", "effort_is_budget", "max_breakpoints",
              "verified", "notes"):
        if f in raw:
            fields[f] = getattr(over, f)
    return replace(base, **fields)


def all_providers() -> dict[str, Provider]:
    """The effective table: built-ins overlaid with any local override."""
    return {**_BUILTIN, **_overrides()}


def get(name: str) -> Provider:
    """Provider by canonical name. Unknown vendors get the no-cache default."""
    key = (name or "").strip().lower()
    key = _ORG_ALIASES.get(key, key)
    return all_providers().get(key, UNKNOWN)


def for_org(org: str) -> Provider:
    return get(org)


def for_model(model_id: str, org: str = "") -> Provider:
    """Provider for a model, preferring an explicit org over an id guess.

    The org comes from the catalog and is right when it is present. The id
    prefix is the fallback for a bare model name typed on the command line,
    and it is why `--model gpt-5` prices with OpenAI's automatic caching
    rather than Anthropic's write premium.
    """
    if org:
        p = get(org)
        if p is not UNKNOWN:
            return p
    mid = (model_id or "").strip().lower()
    best: tuple[int, str] | None = None
    for prefix, name in _ID_PREFIXES:
        if mid.startswith(prefix) and (best is None or len(prefix) > best[0]):
            best = (len(prefix), name)
    if best is not None:
        # `all_providers()`, not `_BUILTIN`. Reading the built-in table
        # directly meant an `ADDER_PROVIDERS` override applied to a catalog
        # entry carrying `org="openai"` and was ignored for the identical model
        # resolved by id prefix -- `--model gpt-5` typed on the command line.
        # A site that has negotiated rates then got two different cache
        # economics for one model depending on how it was named.
        return all_providers().get(best[1], _BUILTIN[best[1]])
    return UNKNOWN


def known_orgs() -> list[str]:
    return sorted(all_providers())
