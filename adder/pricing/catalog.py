"""Provider-agnostic model catalog: what exists, what it costs, how good it is.

Why this is separate from `prices.py`
-------------------------------------
`prices.py` is a hand-maintained table of first-party Claude rates. It is
authoritative and it goes stale the day Anthropic ships anything. That is
tolerable for one vendor and untenable across five: the set of models worth
routing to now spans Anthropic, OpenAI, Google, and a dozen open-weight
families, and it changes weekly.

So the catalog is *data*, not code. It is refreshed from public sources
(`adder models refresh`), cached on disk, and merged in layers:

    bundled snapshot  <  user cache  <  project override  <  first-party table

Later layers win field-by-field, so a project can pin one price without
forking the whole file, and hand-maintained Claude rates always beat a
scraped one.

Three properties the rest of adder depends on
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


# Vendor prefixes an aggregator decorates its routing slugs with. `~anthropic`
# is a floating-alias route, not a different company -- and the harness gate
# compares the organisation against "anthropic", so leaving the tilde on turns
# an Anthropic model into a third-party one that is refused inline placement
# with a message naming a vendor that does not exist.
_ORG_DECORATION = "~!@"


def normalize_org(s: str) -> str:
    """Canonical vendor name from whatever a source spells it as."""
    s = (s or "").strip().lstrip(_ORG_DECORATION).strip()
    return s


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


def _strs(v: Any) -> tuple[str, ...]:
    """A tuple of strings from whatever a hand-edited file put here.

    `tuple(v)` raises on a number and silently explodes a string into its
    characters -- `"text"` became `('t','e','x','t')`, four modalities. Both
    are reachable from a file this tool invites people to edit, and neither is
    caught by `_load_file`, whose whole job is to degrade to the bundled
    snapshot rather than take down a cost report.
    """
    if v is None or isinstance(v, (str, bytes)):
        return (str(v),) if v else ()
    try:
        return tuple(str(x) for x in v)
    except TypeError:
        return ()


def _floats(v: Any) -> dict[str, float]:
    """A name -> number mapping, dropping anything that is not one."""
    if not isinstance(v, dict):
        return {}
    out = {}
    for k, raw in v.items():
        f = _num(raw)
        if f is not None:
            out[str(k)] = f
    return out


def _num(v: Any) -> float | None:
    """A float, or None. Booleans are not numbers here, whatever Python thinks."""
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


def _iso(v: Any) -> str | None:
    """A date/timestamp field as a string, or None. Never a number or a list.

    `released` and `fetched_at` are read back as ISO text -- `age_days` calls
    `.replace` on one and `merge` compares the other with `max` -- so a number
    or a mapping here fails a long way from the file it came out of.
    """
    return v if isinstance(v, str) and v.strip() else None


# Fields whose default is a falsy scalar rather than None. Omitting one of these
# from the serialized form round-trips to the same value; omitting a zero PRICE
# does not. `inp=0.0` is a model that costs nothing to call and `inp=None` is a
# model nobody published a price for, and keeping those two apart is the whole
# reason the price fields are Optional -- `find(priced_only=True)` drops the
# second and must not drop the first. A blanket `v not in (..., 0, False)` filter
# erased every free model's price on the way to disk, so a catalog that had been
# saved and reloaded silently stopped offering them.
_FALSY_DEFAULTS: dict[str, Any] = {"votes": 0, "verified": False}


def _worth_writing(key: str, value: Any) -> bool:
    """Is this field worth a line in the file, or is it just its own default?"""
    if value is None:
        return False
    if isinstance(value, (str, list, dict, tuple)) and not value:
        return False
    return not (key in _FALSY_DEFAULTS and value == _FALSY_DEFAULTS[key])


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
    #
    # `elo_lo`/`elo_hi` are the arena's own 95% interval for that rating. They
    # are not decoration: on the top of the webdev board the half-width is
    # around 10 points, so a 17-point "lead" is two overlapping intervals and
    # not a difference at all. Anything that compares two ratings has to be
    # able to say "indistinguishable", and it cannot do that without these.
    elo: dict[str, float] = field(default_factory=dict)
    elo_lo: dict[str, float] = field(default_factory=dict)
    elo_hi: dict[str, float] = field(default_factory=dict)
    # Which arena variant produced `elo` -- e.g. `claude-opus-5-max`. The arena
    # ranks reasoning efforts separately; the price table has one price. Keeping
    # the label means a report can disclose that it is quoting max-effort
    # quality next to default-effort pricing.
    rating_variant: str = ""
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
        b = self.rating_board(boards)
        return None if b is None else self.elo[b]

    def rating_board(self, boards: Iterable[str] = CODING_BOARDS) -> str | None:
        """Which board `rating()` came from. Deterministic for a given set of ratings.

        One board can carry several keys -- `webdev-hard` and `webdev-easy` both
        answer to `webdev` -- and this used to return whichever one came first
        in the dict. That order is the order the sources happened to merge in,
        not a property of the data, so the same catalog could rate a model 1600
        or 1400 depending on which refresh wrote it last. `rating()` feeds the
        quality floor, `p_loss_from_elo` and every substitute verdict, so a
        200-point swing on nothing is a routing decision on nothing.

        Ties inside a board are broken by key name, so the answer depends only
        on the ratings themselves.
        """
        if not self.elo:
            return None
        for b in boards:
            matching = [full for full in self.elo
                        if full == b or full.split("-")[0] == b]
            if matching:
                # Best rating on that board, as the docstring on `rating` says.
                return min(matching, key=lambda k: (-self.elo[k], k))
        return min(self.elo, key=lambda k: (-self.elo[k], k))

    def rating_interval(
        self, boards: Iterable[str] = CODING_BOARDS
    ) -> tuple[float, float] | None:
        """The arena's 95% interval for `rating()`, or None if unpublished."""
        b = self.rating_board(boards)
        if b is None:
            return None
        lo, hi = self.elo_lo.get(b), self.elo_hi.get(b)
        return None if lo is None or hi is None else (lo, hi)

    def fits(self, tokens: int) -> bool:
        return self.context is not None and tokens <= self.context

    def age_days(self, on: date | None = None) -> float | None:
        """How long ago this entry's newest source was fetched, in days.

        A stamp with no UTC offset -- `2026-08-01`, or an ISO datetime somebody
        hand-typed into an override file -- is read as UTC rather than raising.
        `datetime.fromisoformat` parses both happily and then the subtraction
        against an aware `now` raises `TypeError`, which the `except ValueError`
        here did not catch, so an ordinary date in a catalog file took down
        every caller of `is_stale()` -- which is `adder models`, `adder pick`
        and `policy.substitutes`.
        """
        if not self.fetched_at:
            return None
        try:
            t = datetime.fromisoformat(str(self.fetched_at).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
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
            "elo": dict(self.elo), "elo_lo": dict(self.elo_lo),
            "elo_hi": dict(self.elo_hi), "rating_variant": self.rating_variant,
            "votes": self.votes,
            "intelligence": self.intelligence, "coding": self.coding,
            "agentic": self.agentic, "released": self.released,
            "verified": self.verified, "sources": list(self.sources),
            "fetched_at": self.fetched_at,
        }
        return {k: v for k, v in d.items() if k in ("key", "id") or _worth_writing(k, v)}

    @staticmethod
    def from_json(d: dict[str, Any]) -> Entry:
        """Build an entry from a catalog file, coercing what a human might type.

        This reads a file a user is invited to hand-edit -- pinning one price in
        a project override is an advertised feature -- so the types cannot be
        trusted. A price written `"5"` instead of `5` used to load fine and then
        raise a bare `TypeError` from inside a cost comparison. Coerce what is
        recoverable and drop what is not: a field that cannot be a number
        becomes None, which every gate already reads as "unknown", not "free".
        """
        return Entry(
            key=str(d["key"]), id=str(d.get("id", d["key"])),
            name=str(d.get("name") or ""),
            org=normalize_org(str(d.get("org") or "")),
            license=str(d.get("license") or ""),
            inp=_num(d.get("inp")), out=_num(d.get("out")),
            cache_read=_num(d.get("cache_read")), cache_write=_num(d.get("cache_write")),
            context=_int(d.get("context")), max_output=_int(d.get("max_output")),
            modalities=_strs(d.get("modalities")),
            params=_strs(d.get("params")),
            elo=_floats(d.get("elo")),
            elo_lo=_floats(d.get("elo_lo")),
            elo_hi=_floats(d.get("elo_hi")),
            # Coerced like every other string here. `select.rank` calls
            # `.endswith` on it to decide whether the rating came from a
            # different effort variant, so a non-string raised AttributeError
            # from inside the ranking -- in a loader whose docstring says "the
            # types cannot be trusted".
            rating_variant=str(d.get("rating_variant") or ""),
            votes=_int(d.get("votes")) or 0,
            intelligence=_num(d.get("intelligence")), coding=_num(d.get("coding")),
            agentic=_num(d.get("agentic")), released=_iso(d.get("released")),
            verified=bool(d.get("verified", False)),
            sources=_strs(d.get("sources")), fetched_at=_iso(d.get("fetched_at")),
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
        # `is not None`, not a falsiness test. Pinning a price to 0 is the
        # advertised way to say "this model is free to call", and a truthiness
        # check silently discarded it and served the base price instead -- the
        # override looked applied and was not.
        if v is not None and v != "":
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
        if over.rating_variant:
            out["rating_variant"] = over.rating_variant
    # Merged independently of `elo`. The intervals are what `ratings_overlap`
    # and the conservative Elo comparison read, so a source that publishes
    # bounds for a rating already on the record was having them dropped -- and
    # a missing interval does not fail loudly, it quietly switches off the
    # "the arena cannot separate these two" guard.
    if over.elo_lo:
        out["elo_lo"] = {**base.elo_lo, **over.elo_lo}
    if over.elo_hi:
        out["elo_hi"] = {**base.elo_hi, **over.elo_hi}
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
        """How long ago the *fetched* data was fetched.

        The first-party layer is generated at load time, so it is always
        "fresh" and counting it would make every catalog look current forever
        -- the exact failure this number exists to catch. Provenance from the
        refresh wins; otherwise fall back to the entries that actually came
        from a source.
        """
        stamp = self.provenance.get("refreshed_at")
        if stamp:
            probe = Entry(key="_", id="_", fetched_at=stamp)
            got = probe.age_days(on)
            if got is not None:
                return got
        ages = [
            a for a in (
                e.age_days(on) for e in self
                if not all(s.startswith("first-party") for s in (e.sources or ("?",)))
            ) if a is not None
        ]
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
                "run `adder models refresh` to rebuild it")
        return Catalog((Entry.from_json(m) for m in d.get("models", [])),
                       provenance=d.get("provenance", {}))

    def save(self, path: Path) -> Path:
        """Write the catalog atomically, under a name no other writer shares.

        `adder models refresh` can be running in two sessions at once. With a
        fixed `.tmp` name they share one scratch path, and the first `replace`
        pulls it out from under the second, which then fails on a file it
        thought it owned. Measured at 45% of writes lost under three concurrent
        writers, and the callers here treat a write failure as "the cache is
        just stale", so it does not surface.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(self.to_json(), indent=1, sort_keys=False),
                           encoding="utf-8")
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
        return path


# --------------------------------------------------------------------------
# Layered loading
# --------------------------------------------------------------------------

BUNDLED = Path(__file__).with_name("data") / "catalog.json"


def home() -> Path:
    """Where a refreshed catalog is cached. Overridable for tests and CI."""
    env = os.environ.get("ADDER_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".claude" / "adder"


def user_cache() -> Path:
    return home() / "catalog.json"


def project_override(cwd: Path | None = None) -> Path:
    """Nearest `.adder/catalog.json` at or above `cwd`.

    Searched upward, the way `settings.project_file` finds `.adder.json` and
    for the same stated reason -- "so a repo-level setting applies from any
    subdirectory of it". This looked only in the current directory, so the two
    project-level override mechanisms disagreed about what "this project"
    means: `.adder.json` applied from anywhere in a repo and a pinned price
    silently stopped applying the moment you were one directory down. The
    module docstring advertises the override as the way to pin a price without
    forking the file.

    Returns the path in the current directory when nothing is found, so a
    caller can still report where it looked.
    """
    start = Path(cwd) if cwd is not None else Path.cwd()
    try:
        start = start.expanduser().resolve()
    except OSError:
        return (cwd or Path.cwd()) / ".adder" / "catalog.json"
    for d in [start, *start.parents]:
        candidate = d / ".adder" / "catalog.json"
        if candidate.is_file():
            return candidate
    return start / ".adder" / "catalog.json"


def _load_file(p: Path) -> Catalog | None:
    try:
        if not p.is_file():
            return None
        return Catalog.from_json(json.loads(p.read_text(encoding="utf-8")))
    except (OSError, ValueError, json.JSONDecodeError, KeyError):
        # A corrupt cache must degrade to the bundled snapshot, never crash a
        # cost report. The refresh command reports the failure loudly instead.
        return None


def first_party(on: date | None = None) -> Catalog:
    """Claude rates from the hand-maintained table, marked verified.

    This layer wins over anything scraped. An aggregator's Anthropic prices
    are usually right and occasionally a month behind; the local table is the
    one a human checked against the published list.

    The rate is taken **as of a date**, not from `base`. `prices.py` exists
    because Claude rates move -- Sonnet 5 ships at an introductory $2/$10 that
    reverts to $3/$15 after 2026-08-31 -- and every other consumer of that table
    honours it. This layer did not, so `adder pick` priced Sonnet 5 at $3/$15
    while `adder trace` and `adder policy` priced the same model at $2/$10, and
    the cross-vendor comparison quietly carried a 50% penalty against it. Two
    halves of one tool disagreeing about the price of one model is worse than
    either number on its own.

    Unlike a scraped price, this layer is generated at load time rather than
    read off disk, so it *can* carry a date honestly.
    """
    from adder.pricing.prices import CACHE_READ_MULT, CACHE_WRITE_MULT, MODELS

    stamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    out = []
    for mid, m in MODELS.items():
        r = m.rate(on)
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


def load(*, cwd: Path | None = None, include_first_party: bool = True,
         on: date | None = None) -> Catalog:
    """Bundled snapshot, then user cache, then project override, then Claude.

    Never touches the network. Refresh is an explicit command.

    `ADDER_CATALOG=<path>` replaces the whole stack with one file. That is
    how a report is made reproducible: a recommendation that depends on
    whatever happened to be cached on the machine that ran it is not a result
    anyone can check.
    """
    pinned = os.environ.get("ADDER_CATALOG")
    if pinned:
        cat = _load_file(Path(pinned).expanduser()) or Catalog()
        return cat.overlay(first_party(on)) if include_first_party else cat

    cat = Catalog()
    for p in (BUNDLED, user_cache(), project_override(cwd)):
        got = _load_file(p)
        if got is not None:
            cat = cat.overlay(got)
    if include_first_party:
        cat = cat.overlay(first_party(on))
    return cat
