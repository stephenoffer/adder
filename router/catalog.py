"""Provider-agnostic model catalog: what exists, what it costs, how good it is.

Why this is separate from `prices.py`
-------------------------------------
`prices.py` is a hand-maintained table of first-party Claude rates. It is
authoritative and it goes stale the day Anthropic ships anything. That is
tolerable for one vendor and untenable across five: the set of models worth
routing to now spans Anthropic, OpenAI, Google, and a dozen open-weight
families, and it changes weekly.

So the catalog is *data*, not code. It is refreshed from public sources
(`rt models refresh`), cached on disk, and merged in layers:

    bundled snapshot  <  user cache  <  project override  <  first-party table

Later layers win field-by-field, so a project can pin one price without
forking the whole file, and hand-maintained Claude rates always beat a
scraped one.

Three properties the rest of the router depends on
--------------------------------------------------
1. **Provenance.** Every field carries where it came from and when. A price
   scraped from an aggregator is not the same claim as a first-party list
   price, and `verified` says which one you have. The cost gates refuse to
   report a saving computed from an unverified price unless told otherwise --
   the failure mode of a model router is not a bad recommendation, it is a
   confident bad recommendation.
2. **Staleness is a number, not a vibe.** `age_days()` is on the record. A
   catalog nobody has refreshed in three months should degrade recommendations,
   not silently serve 2026-Q1 prices as current.
3. **Absolute cache rates, not multipliers.** Anthropic charges 0.1x input for
   a cache read and 1.25x/2x to write. Other providers do not: some charge
   nothing to write, some do not cache at all. Since the dominant term in an
   agent session is `cache_read * remaining_turns`, a multiplier borrowed from
   Anthropic and applied to Gemini is not a rounding error, it is the whole
   answer. Cache rates here are USD per million tokens, absolute, or None for
   "this provider published nothing".
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = 1

# Boards carried from the arena, in the order a coding router should trust them.
CODING_BOARDS = ("webdev", "text", "document")

# Effort/variant suffixes an arena key carries that a model id does not.
_VARIANT = (
    "max", "xhigh", "high", "medium", "low", "minimal",
    "thinking", "nothinking", "no-thinking", "reasoning", "nonreasoning",
    "search", "text", "chat", "preview", "latest", "exp", "experimental",
)
_PAREN = re.compile(r"\s*\([^)]*\)")
_DATED = re.compile(r"-(\d{8}|\d{4}-\d{2}-\d{2}|\d{2}-\d{2})(?=$|[.-])")
_THINK_BUDGET = re.compile(r"-(thinking|high|max|low|medium|xhigh)-\d+k$")


def normalize_key(s: str) -> str:
    """Collapse a vendor slug or arena display name to a joinable key.

    The two public sources disagree on nearly every surface detail: the arena
    writes `claude-opus-4-6-thinking`, the aggregator writes
    `anthropic/claude-opus-4.6`. Without this, the join rate between them is
    about a third of what it should be, and the models that fall out are
    disproportionately the new ones -- exactly the ones worth routing to.
    """
    s = (s or "").lower().strip()
    s = s.split("/")[-1]
    s = _PAREN.sub("", s)
    s = s.replace(" ", "-").replace("_", "-")
    s = _THINK_BUDGET.sub("", s)
    for _ in range(4):
        s = _DATED.sub("", s)
    changed = True
    while changed:
        changed = False
        for v in _VARIANT:
            if s.endswith("-" + v) and len(s) > len(v) + 1:
                s = s[: -len(v) - 1]
                changed = True
    # Version separators: `-4-5` and `-4.5` are the same model.
    s = re.sub(r"(?<=\d)-(?=\d)", ".", s)
    for _ in range(2):
        s = _DATED.sub("", s)
    return s.strip("-")


@dataclass(frozen=True)
class Entry:
    """One model, joined across every source that knows about it."""

    key: str                              # normalized join key
    id: str                               # canonical routing id
    name: str = ""
    org: str = ""                         # Anthropic, OpenAI, Alibaba, ...
    license: str = ""                     # "Proprietary" or an SPDX-ish string

    # USD per million tokens. None means "not published", never "free".
    inp: float | None = None
    out: float | None = None
    cache_read: float | None = None
    cache_write: float | None = None

    context: int | None = None
    max_output: int | None = None
    modalities: tuple[str, ...] = ()
    params: tuple[str, ...] = ()          # tools, reasoning, structured_outputs

    # Quality proxies. `elo` is board -> arena rating; indices are 0-100.
    elo: dict[str, float] = field(default_factory=dict)
    votes: int = 0
    intelligence: float | None = None
    coding: float | None = None
    agentic: float | None = None

    released: str | None = None           # ISO date
    verified: bool = False                # price from a first-party source
    sources: tuple[str, ...] = ()
    fetched_at: str | None = None         # ISO timestamp of the newest source

    # ---- derived -------------------------------------------------------
    @property
    def open_weights(self) -> bool:
        lic = (self.license or "").lower()
        return bool(lic) and "proprietary" not in lic

    @property
    def supports_tools(self) -> bool:
        return "tools" in self.params

    @property
    def supports_reasoning(self) -> bool:
        return any(p.startswith("reasoning") or p == "include_reasoning"
                   for p in self.params)

    @property
    def priced(self) -> bool:
        return self.inp is not None and self.out is not None

    def rating(self, boards: Iterable[str] = CODING_BOARDS) -> float | None:
        """Best available arena rating, preferring code-shaped boards."""
        for b in boards:
            for full, v in self.elo.items():
                if full.split("-")[0] == b or full == b:
                    return v
        return max(self.elo.values()) if self.elo else None

    def fits(self, tokens: int) -> bool:
        return self.context is not None and tokens <= self.context

    def age_days(self, on: date | None = None) -> float | None:
        if not self.fetched_at:
            return None
        try:
            t = datetime.fromisoformat(self.fetched_at.replace("Z", "+00:00"))
        except ValueError:
            return None
        now = datetime.combine(on, datetime.min.time(), tzinfo=timezone.utc) \
            if on else datetime.now(timezone.utc)
        return max(0.0, (now - t).total_seconds() / 86400.0)

    def to_json(self) -> dict[str, Any]:
        d = {
            "key": self.key, "id": self.id, "name": self.name, "org": self.org,
            "license": self.license, "inp": self.inp, "out": self.out,
            "cache_read": self.cache_read, "cache_write": self.cache_write,
            "context": self.context, "max_output": self.max_output,
            "modalities": list(self.modalities), "params": list(self.params),
            "elo": dict(self.elo), "votes": self.votes,
            "intelligence": self.intelligence, "coding": self.coding,
            "agentic": self.agentic, "released": self.released,
            "verified": self.verified, "sources": list(self.sources),
            "fetched_at": self.fetched_at,
        }
        return {k: v for k, v in d.items() if v not in (None, "", [], {}, 0, False)
                or k in ("key", "id")}

    @staticmethod
    def from_json(d: dict[str, Any]) -> Entry:
        return Entry(
            key=d["key"], id=d.get("id", d["key"]), name=d.get("name", ""),
            org=d.get("org", ""), license=d.get("license", ""),
            inp=d.get("inp"), out=d.get("out"),
            cache_read=d.get("cache_read"), cache_write=d.get("cache_write"),
            context=d.get("context"), max_output=d.get("max_output"),
            modalities=tuple(d.get("modalities", ())),
            params=tuple(d.get("params", ())),
            elo={k: float(v) for k, v in (d.get("elo") or {}).items()},
            votes=int(d.get("votes", 0) or 0),
            intelligence=d.get("intelligence"), coding=d.get("coding"),
            agentic=d.get("agentic"), released=d.get("released"),
            verified=bool(d.get("verified", False)),
            sources=tuple(d.get("sources", ())), fetched_at=d.get("fetched_at"),
        )


def merge(base: Entry, over: Entry) -> Entry:
    """Field-wise overlay: `over` wins wherever it actually has a value.

    Deliberately not a dict update. A refresh that fails to price one model
    must not blank the price we already had, and a project override that pins
    a single rate must not erase everything else about the record.
    """
    out: dict[str, Any] = {}
    for f in ("name", "org", "license", "inp", "out", "cache_read", "cache_write",
              "context", "max_output", "released", "id"):
        v = getattr(over, f)
        if v not in (None, "", 0):
            out[f] = v
    for f in ("modalities", "params"):
        v = getattr(over, f)
        if v:
            out[f] = v
    for f in ("intelligence", "coding", "agentic"):
        v = getattr(over, f)
        if v is not None:
            out[f] = v
    if over.elo:
        out["elo"] = {**base.elo, **over.elo}
    if over.votes:
        out["votes"] = max(base.votes, over.votes)
    out["verified"] = base.verified or over.verified
    out["sources"] = tuple(dict.fromkeys(base.sources + over.sources))
    newest = max([x for x in (base.fetched_at, over.fetched_at) if x], default=None)
    if newest:
        out["fetched_at"] = newest
    return replace(base, **out)


class Catalog:
    """A set of entries plus where they came from."""

    def __init__(self, entries: Iterable[Entry] = (), *,
                 provenance: dict[str, Any] | None = None):
        self._by_key: dict[str, Entry] = {}
        for e in entries:
            self.add(e)
        self.provenance: dict[str, Any] = provenance or {}

    # ---- container -----------------------------------------------------
    def add(self, e: Entry) -> None:
        prev = self._by_key.get(e.key)
        self._by_key[e.key] = merge(prev, e) if prev else e

    def __len__(self) -> int:
        return len(self._by_key)

    def __iter__(self) -> Iterator[Entry]:
        return iter(sorted(self._by_key.values(), key=lambda e: e.key))

    def __contains__(self, key: str) -> bool:
        return normalize_key(key) in self._by_key

    def get(self, key: str) -> Entry | None:
        return self._by_key.get(normalize_key(key))

    def overlay(self, other: Catalog) -> Catalog:
        c = Catalog(self)
        for e in other:
            c.add(e)
        c.provenance = {**self.provenance, **other.provenance}
        return c

    # ---- queries -------------------------------------------------------
    def find(
        self,
        *,
        org: str | None = None,
        open_weights: bool | None = None,
        needs_tools: bool = False,
        min_context: int = 0,
        priced_only: bool = True,
        rated_only: bool = False,
        verified_only: bool = False,
        max_input_price: float | None = None,
    ) -> list[Entry]:
        out = []
        for e in self:
            if org and e.org.lower() != org.lower():
                continue
            if open_weights is not None and e.open_weights != open_weights:
                continue
            if needs_tools and not e.supports_tools:
                continue
            if min_context and not e.fits(min_context):
                continue
            if priced_only and not e.priced:
                continue
            if rated_only and e.rating() is None:
                continue
            if verified_only and not e.verified:
                continue
            if max_input_price is not None and (e.inp is None or e.inp > max_input_price):
                continue
            out.append(e)
        return out

    def age_days(self, on: date | None = None) -> float | None:
        ages = [a for a in (e.age_days(on) for e in self) if a is not None]
        return min(ages) if ages else None

    def is_stale(self, *, max_age_days: float = 21.0, on: date | None = None) -> bool:
        a = self.age_days(on)
        return a is None or a > max_age_days

    # ---- serialization -------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "provenance": self.provenance,
            "models": [e.to_json() for e in self],
        }

    @staticmethod
    def from_json(d: dict[str, Any]) -> Catalog:
        if int(d.get("schema", 0)) != SCHEMA:
            raise ValueError(
                f"catalog schema {d.get('schema')!r} != {SCHEMA}; "
                "run `rt models refresh` to rebuild it")
        return Catalog((Entry.from_json(m) for m in d.get("models", [])),
                       provenance=d.get("provenance", {}))

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_json(), indent=1, sort_keys=False))
        tmp.replace(path)
        return path


# --------------------------------------------------------------------------
# Layered loading
# --------------------------------------------------------------------------

BUNDLED = Path(__file__).with_name("data") / "catalog.json"


def home() -> Path:
    """Where a refreshed catalog is cached. Overridable for tests and CI."""
    env = os.environ.get("LLM_ROUTER_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".claude" / "llm-router"


def user_cache() -> Path:
    return home() / "catalog.json"


def project_override(cwd: Path | None = None) -> Path:
    return (cwd or Path.cwd()) / ".llm-router" / "catalog.json"


def _load_file(p: Path) -> Catalog | None:
    try:
        if not p.is_file():
            return None
        return Catalog.from_json(json.loads(p.read_text()))
    except (OSError, ValueError, json.JSONDecodeError, KeyError):
        # A corrupt cache must degrade to the bundled snapshot, never crash a
        # cost report. The refresh command reports the failure loudly instead.
        return None


def first_party() -> Catalog:
    """Claude rates from the hand-maintained table, marked verified.

    This layer wins over anything scraped. An aggregator's Anthropic prices
    are usually right and occasionally a month behind; the local table is the
    one a human checked against the published list.
    """
    from .prices import CACHE_READ_MULT, CACHE_WRITE_MULT, MODELS

    stamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    out = []
    for mid, m in MODELS.items():
        r = m.base
        out.append(Entry(
            key=normalize_key(mid), id=mid, name=mid, org="Anthropic",
            license="Proprietary",
            inp=r.inp, out=r.out,
            cache_read=round(r.inp * CACHE_READ_MULT, 6),
            cache_write=round(r.inp * CACHE_WRITE_MULT["5m"], 6),
            context=m.context, max_output=m.max_output,
            modalities=("text", "image"),
            params=("tools", "reasoning") if m.efforts else ("tools",),
            verified=True, sources=("first-party:prices.py",), fetched_at=stamp,
        ))
    return Catalog(out, provenance={"first-party": {"fetched_at": stamp}})


def load(*, cwd: Path | None = None, include_first_party: bool = True) -> Catalog:
    """Bundled snapshot, then user cache, then project override, then Claude.

    Never touches the network. Refresh is an explicit command.
    """
    cat = Catalog()
    for p in (BUNDLED, user_cache(), project_override(cwd)):
        got = _load_file(p)
        if got is not None:
            cat = cat.overlay(got)
    if include_first_party:
        cat = cat.overlay(first_party())
    return cat
