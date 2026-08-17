"""Refresh the catalog from public sources. The only module that opens a socket.

Everything else in this repo is offline by construction, and that is a feature
worth protecting: a cost report that silently depends on a third party is a
cost report that breaks in CI at the worst moment. So the network lives here,
behind one explicit command, with three rules:

* **Opt-in.** Nothing calls `refresh()` implicitly. `ADDER_OFFLINE=1`
  makes even the explicit call refuse.
* **Fail soft, report loud.** One source down degrades the catalog to whatever
  the other one knew; it never leaves a half-written file or a crashed report.
  The provenance block records what succeeded and what did not.
* **Reproducible offline.** Every fetcher has a matching parser that takes raw
  bytes, so `adder models refresh --from lmarena=page.html` replays a saved
  capture. The parsers are what the tests exercise; the sockets are not.

There is no `main()` here on purpose. The command surface lives in
`models.py`, so that the module that can open a socket is a library with one
job and cannot be invoked by accident.

The two sources, and why these two
----------------------------------
**LMArena** publishes head-to-head Elo per model, per board, with vote counts
and confidence intervals. It is the only public quality signal that is (a)
updated within days of a launch, (b) comparable across vendors, and (c) not
self-reported by the vendor. Its board split matters: `webdev` ranks code
generation and correlates with what an agent session actually does, while
`text-overall` rewards prose. Routing a coding agent on the text board picks
the wrong model, confidently.

**OpenRouter's model index** publishes price, context window, cache rates,
modalities, and supported parameters across ~400 models from every major lab,
without an API key. It is an aggregator, so its prices are *reported*, not
authoritative: entries from it are marked `verified=False`, and first-party
Claude rates always overwrite them.

Neither source is a benchmark of agentic tool use, which is the thing we
actually care about. Arena Elo is a proxy, and everything downstream that uses
it is labelled MODELLED for exactly that reason.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adder.pricing.catalog import Catalog, Entry, normalize_key, normalize_org

LMARENA_URL = "https://lmarena.ai/leaderboard"
OPENROUTER_URL = "https://openrouter.ai/api/v1/models"

USER_AGENT = "adder/0.2 (+https://github.com/; catalog refresh)"
TIMEOUT = 30
MAX_BYTES = 32 * 1024 * 1024


class OfflineError(RuntimeError):
    pass


class FetchFailedError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# The values that mean "yes" for `ADDER_OFFLINE`, matching `settings._as_bool`.
#
# `pricing` sits below `core` and may not import the settings module, so the
# vocabulary is restated here rather than shared. It has to be restated
# *correctly*: a bare truthiness check made `ADDER_OFFLINE=0` -- which the
# setting defines as "not offline", and which is the obvious way to turn the
# switch off for one command -- refuse every fetch.
_TRUE = ("1", "true", "yes", "on")


def is_offline(env: dict[str, str] | None = None) -> bool:
    """Whether the offline switch is on, using the `offline` setting's vocabulary."""
    raw = (os.environ if env is None else env).get("ADDER_OFFLINE", "")
    return str(raw).strip().lower() in _TRUE


def fetch(url: str, *, timeout: int = TIMEOUT) -> bytes:
    """GET with a cap. Raises `OfflineError` when the network is switched off."""
    if is_offline():
        raise OfflineError("ADDER_OFFLINE is set; refusing to fetch " + url)
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/html;q=0.9",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read(MAX_BYTES)
    except (urllib.error.URLError, OSError, ValueError,
            http.client.HTTPException) as e:
        # `HTTPException` -- a truncated body is `IncompleteRead` -- is not an
        # OSError and is not a URLError, so a half-delivered response escaped
        # this handler entirely and `models refresh` reported a traceback
        # instead of "that source failed".
        raise FetchFailedError(f"{url}: {e}") from e


# --------------------------------------------------------------------------
# LMArena
# --------------------------------------------------------------------------

_BOARD_ID = re.compile(
    r"leaderboard-sets/public/leaderboards/([a-z0-9_-]+)/leaderboard-snapshots/latest")

# Boards that rank text-ish models. The rest rank image and video generators,
# which are not routing targets for a coding agent.
TEXT_BOARDS = ("text", "webdev", "vision", "document", "search", "image_to_webdev")

# The board a coding router should attribute a model's headline rating to.
CODING_BOARD = "webdev"


def _unescape(payload: str) -> str:
    return payload.replace('\\"', '"').replace("\\\\", "\\")


def _slice_array(s: str, start: int) -> str | None:
    """Return the JSON array beginning at `start`, by bracket matching."""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def parse_lmarena(html: str | bytes) -> dict[str, list[dict[str, Any]]]:
    """Pull every leaderboard out of the page's embedded payload.

    The page is a server-rendered React stream, so the data is present but
    JSON-escaped inside a JS string. Extracting it is a scrape, and scrapes
    break: a parse that finds no boards raises rather than quietly returning
    an empty catalog that would look like "no models are any good".
    """
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")
    flat = _unescape(html)
    boards: dict[str, list[dict[str, Any]]] = {}
    for m in _BOARD_ID.finditer(flat):
        name = m.group(1)
        j = flat.find('"entries":[', m.end())
        if j < 0:
            continue
        arr = _slice_array(flat, j + len('"entries":'))
        if not arr:
            continue
        try:
            entries = json.loads(arr)
        except json.JSONDecodeError:
            continue
        if entries:
            boards[name] = entries
    if not boards:
        raise FetchFailedError(
            "lmarena: no leaderboards found in the page; the layout changed "
            "(re-run with --from lmarena=<saved.html> after checking)")
    return boards


def lmarena_entries(boards: dict[str, list[dict[str, Any]]],
                    *, only: Iterable[str] = TEXT_BOARDS,
                    stamp: str | None = None) -> list[Entry]:
    """Fold per-board rows into one entry per model.

    A model appears once per board and often several times per board under
    different reasoning efforts. We keep the *best* rating per board, because
    the routing question is "how good can this model be", with effort priced
    separately by the cost model.
    """
    stamp = stamp or _now()
    want = {b.split("-")[0] for b in only}
    acc: dict[str, dict[str, Any]] = {}
    for board, rows in boards.items():
        short = board.split("-")[0]
        if short not in want:
            continue
        for r in rows:
            disp = r.get("modelDisplayName") or r.get("modelKey") or ""
            key = normalize_key(disp)
            if not key:
                continue
            a = acc.setdefault(key, {"elo": {}, "lo": {}, "hi": {}, "votes": 0,
                                     "variant": "", "row": r})
            rating = r.get("rating")
            if isinstance(rating, (int, float)) and rating > a["elo"].get(
                    short, float("-inf")):
                a["elo"][short] = float(rating)
                # The interval belongs to the rating it was published with.
                # Taking the best rating from one variant and the interval from
                # another would be a wider claim than either row makes.
                lo, hi = r.get("ratingLower"), r.get("ratingUpper")
                if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                    a["lo"][short], a["hi"][short] = float(lo), float(hi)
                else:
                    a["lo"].pop(short, None)
                    a["hi"].pop(short, None)
                if short == CODING_BOARD or not a["variant"]:
                    a["variant"] = disp
                if short == "text" or "row" not in a:
                    a["row"] = r
            a["votes"] = max(a["votes"], int(r.get("votes") or 0))

    out = []
    for key, a in acc.items():
        r = a["row"]
        out.append(Entry(
            key=key, id=key,
            name=r.get("modelDisplayName") or key,
            org=normalize_org(r.get("modelOrganization") or ""),
            license=r.get("license") or "",
            inp=_f(r.get("inputPricePerMillion")),
            out=_f(r.get("outputPricePerMillion")),
            context=_i(r.get("contextLength")),
            elo=a["elo"], elo_lo=a["lo"], elo_hi=a["hi"],
            rating_variant=a["variant"], votes=a["votes"],
            verified=False, sources=("lmarena",), fetched_at=stamp,
        ))
    return out


def _f(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _i(v: Any) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# OpenRouter
# --------------------------------------------------------------------------

def parse_openrouter(raw: str | bytes, *, stamp: str | None = None) -> list[Entry]:
    """Price, context, cache rates, modalities, and tool support per model.

    Prices arrive as USD *per token* in strings; everything else in this repo
    is per million, so convert once here rather than at every call site. A
    price of exactly 0 is kept as 0.0 (some open-weight endpoints really are
    free) but a missing key becomes None, which the gates treat as "unknown"
    rather than "cheap".
    """
    stamp = stamp or _now()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    data = json.loads(raw)
    rows = data.get("data") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise FetchFailedError("openrouter: unexpected payload shape")

    out = []
    for m in rows:
        mid = str(m.get("id") or "")
        if not mid:
            continue
        arch = m.get("architecture") or {}
        if "text" not in (arch.get("output_modalities") or ["text"]):
            continue                      # image/video generators are not routable here
        p = m.get("pricing") or {}
        top = m.get("top_provider") or {}
        bench = (m.get("benchmarks") or {}).get("artificial_analysis") or {}
        created = m.get("created")
        released = None
        if isinstance(created, (int, float)) and created > 0:
            released = datetime.fromtimestamp(created, tz=timezone.utc).date().isoformat()
        out.append(Entry(
            key=normalize_key(mid),
            id=mid,
            name=str(m.get("name") or mid),
            org=normalize_org(
                str(m.get("name") or "").split(":")[0]
                if ":" in str(m.get("name") or "") else mid.split("/")[0]),
            inp=_per_m(p.get("prompt")),
            out=_per_m(p.get("completion")),
            cache_read=_per_m(p.get("input_cache_read")),
            cache_write=_per_m(p.get("input_cache_write")),
            context=_i(m.get("context_length")) or _i(top.get("context_length")),
            max_output=_i(top.get("max_completion_tokens")),
            modalities=tuple(arch.get("input_modalities") or ()),
            params=tuple(m.get("supported_parameters") or ()),
            intelligence=_f(bench.get("intelligence_index")),
            coding=_f(bench.get("coding_index")),
            agentic=_f(bench.get("agentic_index")),
            released=released,
            verified=False, sources=("openrouter",), fetched_at=stamp,
        ))
    return out


def _per_m(v: Any) -> float | None:
    """Per-token string -> USD per million, rejecting sentinels.

    Aggregator rows for meta-models (`openrouter/auto` and friends) carry
    negative prices to mean "resolved at request time". Left alone, a negative
    rate sorts first in every ranking and produces a router that recommends
    being paid to run inference.
    """
    f = _f(v)
    if f is None or f < 0:
        return None
    return round(f * 1_000_000, 6)


# --------------------------------------------------------------------------
# Refresh
# --------------------------------------------------------------------------

@dataclass
class SourceResult:
    name: str
    ok: bool
    count: int = 0
    error: str = ""
    origin: str = ""


def refresh(
    *,
    offline_files: dict[str, Path] | None = None,
    timeout: int = TIMEOUT,
    sources: Iterable[str] = ("lmarena", "openrouter"),
) -> tuple[Catalog, list[SourceResult]]:
    """Build a catalog from public data. Returns whatever succeeded.

    Order matters: the aggregator lands first because it carries the structural
    facts (context window, cache rates, tool support), then the arena overlays
    quality and license. Both are unverified; `catalog.load()` puts the
    first-party Claude table on top afterwards.
    """
    offline_files = offline_files or {}
    stamp = _now()
    cat = Catalog()
    results: list[SourceResult] = []

    def _read(name: str, url: str) -> tuple[bytes, str]:
        p = offline_files.get(name)
        if p is not None:
            return Path(p).read_bytes(), f"file:{p}"
        return fetch(url, timeout=timeout), url

    if "openrouter" in sources:
        try:
            raw, origin = _read("openrouter", OPENROUTER_URL)
            entries = parse_openrouter(raw, stamp=stamp)
            for e in entries:
                cat.add(e)
            results.append(SourceResult("openrouter", True, len(entries), origin=origin))
        except (FetchFailedError, OfflineError, OSError, ValueError) as e:
            results.append(SourceResult("openrouter", False, error=str(e)))

    if "lmarena" in sources:
        try:
            raw, origin = _read("lmarena", LMARENA_URL)
            boards = parse_lmarena(raw)
            entries = lmarena_entries(boards, stamp=stamp)
            for e in entries:
                cat.add(e)
            results.append(SourceResult(
                "lmarena", True, len(entries),
                origin=f"{origin} ({len(boards)} boards)"))
        except (FetchFailedError, OfflineError, OSError, ValueError) as e:
            results.append(SourceResult("lmarena", False, error=str(e)))

    cat.provenance = {
        "refreshed_at": stamp,
        "sources": [
            {"name": r.name, "ok": r.ok, "models": r.count,
             "origin": r.origin, "error": r.error}
            for r in results
        ],
    }
    return cat, results
