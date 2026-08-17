"""One resolution point for "what does this model cost and how does it cache?".

The problem this fixes
----------------------
`prices.py` is a hand-maintained Claude table and `catalog.py` is a 500-model
scrape, and until this module existed nothing joined them. Every cost function
in the repo resolved models through `prices.resolve`, which raises
`UnknownModelError` for anything that is not Claude. So the cost model -- the piece
the whole tool is built on -- could not price a single OpenAI, Google, or
open-weight model, and `select.py` had to carry a second, subtly different copy
of the same arithmetic to work around it. That is how the two came to disagree
about whether the carry term could be corrected at all.

Resolution order, most authoritative first
------------------------------------------
1. **First-party table** (`prices.py`). Hand-checked against the published
   list price, date-aware, and the only layer that knows about introductory
   rates and fast mode. Wins for Claude, always.
2. **Catalog** (`catalog.py`). Everything else, joined from public sources,
   carrying its own provenance and staleness.
3. **Provider defaults** (`providers.py`). Fills in the cache mechanics the
   catalog does not publish -- which is most of them: 273 of 510 bundled
   entries have no cache read rate at all.

What a caller gets back
-----------------------
A `ModelSpec` that answers every question the cost model asks, for any model,
with the provenance of each answer attached. Two properties are load-bearing:

* **A model with no known cache is priced with no cache.** `cache_read_rate`
  returns the full input rate when the provider does not cache, so the carry
  term comes out ~10x larger rather than silently borrowing Anthropic's 0.10x.
  A carry term pointed the wrong way is worse than a large one.
* **Absolute beats derived.** Where the catalog publishes a real cache rate it
  is used verbatim. Multipliers are the fallback and are labelled as such, so a
  report can say which of the two it quoted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from datetime import date
from functools import lru_cache
from pathlib import Path

from adder.pricing import prices as _p
from adder.pricing import providers as _prov
from adder.pricing.catalog import Catalog, Entry, load, normalize_key
from adder.pricing.providers import Provider

M = 1_000_000.0


class UnknownModelError(KeyError):
    """Raised when neither the first-party table nor the catalog knows a model.

    Deliberately a subclass of nothing in `prices`: callers that want the old
    Claude-only behaviour keep importing `prices.UnknownModelError`, and callers
    that want the universal one import this. `is_known` covers both.
    """


@dataclass(frozen=True)
class ModelSpec:
    """Everything the cost model needs about one model, from any provider."""

    id: str
    org: str
    provider: Provider

    inp: float | None = None            # USD per million input tokens
    out: float | None = None
    # Absolute cache rates as published. None means "the provider published
    # nothing", never "free".
    cache_read_abs: float | None = None
    cache_write_abs: float | None = None
    # Explicit-cache storage, USD per million tokens per hour. Only Google
    # publishes one; it is kept apart from the write rate because it is billed
    # for elapsed time rather than for tokens moved.
    cache_storage_abs: float | None = None

    context: int | None = None
    max_output: int | None = None
    cache_min: int = 0
    efforts: tuple[str, ...] = ()

    verified: bool = False              # price came from a first-party source
    first_party: bool = False           # resolved from prices.py, not the catalog
    source: str = ""                    # where the price came from
    age_days: float | None = None       # how stale the catalog layer is
    entry: Entry | None = None          # the catalog record, when there was one

    # ---- pricing ------------------------------------------------------
    def rate(self, on: date | None = None, *, speed: str = "standard"):
        """Input/output rate as of a date. Only the first-party layer is dated.

        Catalog prices carry no time dimension -- a scrape is a snapshot, not a
        schedule -- so `on` is honoured where it is known and ignored where it
        cannot be. That is better than pretending a scraped price was valid on
        an arbitrary date.
        """
        if self.first_party:
            return _p.resolve(self.id).rate(on, speed=speed)
        if speed == "fast":
            raise _p.UnsupportedSpeedError(f"{self.id} has no fast mode")
        if self.inp is None or self.out is None:
            raise UnpricedModelError(
                f"{self.id} has no published price; a cost computed from it "
                "would be a guess dressed as a measurement")
        return _p.Rate(self.inp, self.out)

    @property
    def priced(self) -> bool:
        return self.inp is not None and self.out is not None

    def cache_read_rate(self, on: date | None = None, *,
                        speed: str = "standard") -> float:
        """USD per million tokens to re-read a cached prefix.

        Falls all the way through to the *full input rate* when the provider
        has no cache. That is not a placeholder: on an endpoint that does not
        cache, re-reading the prefix genuinely costs full input rate, and it is
        the single largest term in a long session.

        `speed` is not decoration. Fast mode bills the same model at double the
        input rate, and a cache read is a fraction *of the input rate* -- so it
        doubles too. Pricing reads at the standard rate while pricing output at
        the fast rate understates a fast-mode session by most of the carry
        term, which on these transcripts is ~76% of the bill.
        """
        inp = self.rate(on, speed=speed).inp
        if self.cache_read_abs is not None:
            return self.cache_read_abs
        mult = self.provider.cache_read_mult
        if not self.provider.caches or mult is None:
            return inp
        return inp * mult

    def cache_write_rate(self, ttl: str | None = None, on: date | None = None,
                         *, speed: str = "standard") -> float:
        """USD per million tokens to put a prefix into the cache.

        Under automatic caching this equals the input rate: there is no premium
        because there was no decision. Under explicit caching it carries the
        provider's TTL-dependent premium.
        """
        inp = self.rate(on, speed=speed).inp
        if not self.provider.caches:
            return inp
        # An absolute published write rate only describes the provider's
        # default TTL *at standard speed*. Asking for a different TTL, or for
        # fast mode, means going back to the multiplier -- the only form that
        # scales with the input rate.
        if (self.cache_write_abs is not None and speed == "standard"
                and (ttl is None or ttl == self.provider.default_ttl)):
            return self.cache_write_abs
        return inp * self.provider.write_mult(ttl)

    def batch_mult(self) -> float | None:
        return self.provider.batch_mult

    # ---- capability gates ---------------------------------------------
    def fits(self, tokens: int) -> bool:
        """Can this model hold `tokens` at all?

        Unknown context is False, not True. A feasibility gate that passes
        because it does not know is not a gate.
        """
        return self.context is not None and tokens <= self.context

    def caches(self, prefix_tokens: int) -> bool:
        """Will a prefix this size actually cache?

        Two ways to fail: the provider has no cache, or the prefix is under the
        minimum, in which case caching silently does not happen -- no error and
        no discount.
        """
        return self.provider.caches and prefix_tokens >= max(0, self.cache_min)

    def supports_effort(self, level: str) -> bool:
        return level in self.efforts

    @property
    def cache_style(self) -> str:
        return self.provider.cache_style

    @property
    def rate_provenance(self) -> str:
        """How the cache rates in this spec were arrived at.

        Reports quote this rather than the number alone. "0.10x, because that
        is what Anthropic charges and we had nothing else" is a different claim
        from "0.10x, published".
        """
        if self.cache_read_abs is not None:
            return "published"
        if not self.provider.caches:
            return "no cache: re-reads billed at full input rate"
        if self.provider.verified:
            return f"derived from {self.provider.name}'s published multiplier"
        return f"MODELLED from a {self.provider.name} default"


class UnpricedModelError(ValueError):
    """The model exists but nobody published what it costs."""


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def _cache_key() -> tuple:
    """Everything that can change what `load()` returns.

    The catalog is read from disk and joined on every resolve, which a 600-turn
    trace does thousands of times. Caching it is not optional at that volume,
    but caching it *without* keying on the layer sources makes tests that point
    `ADDER_CATALOG` at a fixture see whatever the previous test loaded.
    """
    return (
        os.environ.get("ADDER_CATALOG", ""),
        os.environ.get("ADDER_HOME", ""),
        os.environ.get("ADDER_PROVIDERS", ""),
        str(Path.cwd()),
    )


@lru_cache(maxsize=8)
def _catalog_for(key: tuple) -> Catalog:
    return load()


def catalog() -> Catalog:
    """The effective catalog, cached per layer-source combination."""
    return _catalog_for(_cache_key())


def reset_cache() -> None:
    """Drop memoized catalog and resolution state.

    Tests that write a catalog file mid-run need this; so does `models
    refresh`, which changes the file underneath a live process.
    """
    _catalog_for.cache_clear()
    _resolve_cached.cache_clear()


def _from_first_party(model: str) -> ModelSpec | None:
    try:
        m = _p.resolve(model)
    except _p.UnknownModelError:
        return None
    # Today's rate, not `base`. `rate()` below re-resolves the date on every
    # call, so these two fields are a snapshot of it -- and a snapshot taken
    # from `base` disagrees with the method beside it for the whole of an
    # introductory window. `candidates()` sorts on `.inp`, so the disagreement
    # reaches the ladder as well as the report.
    r = m.rate(None)
    prov = _prov.ANTHROPIC
    return ModelSpec(
        id=m.id, org="Anthropic", provider=prov,
        inp=r.inp, out=r.out,
        cache_read_abs=None,            # derived: the multiplier is the published form
        cache_write_abs=None,
        context=m.context, max_output=m.max_output,
        cache_min=m.cache_min,          # per-model, and NOT monotonic across generations
        efforts=m.efforts,
        verified=True, first_party=True, source="first-party:prices.py",
    )


def _from_catalog(model: str) -> ModelSpec | None:
    cat = catalog()
    e = cat.get(model)
    if e is None:
        e = _best_prefix(cat, model)
    if e is None:
        return None
    prov = _sanity_checked_provider(_prov.for_model(e.id or model, e.org), e)
    write_abs, storage = _split_write_and_storage(prov, e)
    return ModelSpec(
        id=e.id or model, org=e.org, provider=prov,
        inp=e.inp, out=e.out,
        cache_read_abs=e.cache_read, cache_write_abs=write_abs,
        cache_storage_abs=storage,
        context=e.context, max_output=e.max_output,
        cache_min=prov.cache_min,
        efforts=prov.efforts if e.supports_reasoning else (),
        verified=e.verified, first_party=False,
        source=", ".join(e.sources) or "catalog",
        age_days=e.age_days(), entry=e,
    )


def _sanity_checked_provider(prov: Provider, e: Entry) -> Provider:
    """Let published evidence override a provider default of "does not cache".

    The default for an unrecognised vendor is `NONE`, which prices every
    re-read at full input rate. That is the right *default* -- it is the
    conservative direction -- but it is the wrong answer when the catalog is
    sitting there publishing a cache read rate five times below the input rate.
    Half the hosted open-weight endpoints in the bundled snapshot are in
    exactly that position: no vendor entry in the provider table, but a real,
    scraped, materially-discounted cached-input price.

    Ignoring that evidence overstates the carry term for ~90 models, which
    makes adder refuse delegations that are in fact profitable. So: a cache
    read rate meaningfully below the input rate is treated as proof the
    endpoint caches, and the style is inferred as automatic -- the safe
    assumption, since it grants no breakpoint or TTL lever that might not
    exist.
    """
    if prov.caches or e.cache_read is None or not e.inp:
        return prov
    if e.cache_read >= e.inp * 0.9:
        # Not a discount worth calling a cache. Some aggregators echo the input
        # rate into the cache field when the endpoint has no cache at all.
        return prov
    return replace(
        prov,
        cache_style=_prov.AUTOMATIC,
        cache_read_mult=e.cache_read / e.inp,
        cache_write_mult={"auto": 1.00},
        ttl_seconds={"auto": 300},
        verified=False,
        notes=(f"{prov.notes}; caching inferred from a published cached-input "
               f"rate {e.cache_read:g} against an input rate of {e.inp:g}").lstrip("; "),
    )


def _split_write_and_storage(prov: Provider, e: Entry) -> tuple[float | None, float | None]:
    """Tell a cache *write* rate apart from a cache *storage* rate.

    Aggregators put both in one field. Google's `cache_write` is per-million
    *per hour* of storage for explicit context caching -- for `gemini-3.7-flash`
    that is 0.020833 against an input rate of 0.75. Taking it for a write rate
    prices a cache write at 2.8% of input, when under implicit caching a write
    is simply input: a 36x understatement of the term that decides whether a
    token should enter the context at all.

    The tell is an ordering that cannot be true of a real write rate: you
    cannot pay less to create a cache entry than to read one back. Anything
    below the read rate is storage, and is carried as storage.
    """
    w = e.cache_write
    if w is None:
        return None, None
    # Under automatic caching the write rate *is* the input rate, by
    # construction -- there is no premium because there was no decision to
    # make. So any published "cache write" number for such a provider is
    # describing something else, and the only something else anyone bills is
    # storage. Below the input rate it is carried as storage; above it, it is
    # not a term this model recognises and is dropped rather than guessed at.
    if prov.cache_style == _prov.AUTOMATIC:
        return None, (w if e.inp is not None and w < e.inp else None)
    if e.cache_read is not None and w < e.cache_read:
        return None, w
    return w, None


# Aggregators publish floating aliases -- `~openai/gpt-latest`, `*/auto` --
# whose price is whatever the vendor points them at this week. They normalize
# to very short keys (`gpt-latest` -> `gpt`), which makes them the greediest
# possible prefix target.
_FLOATING = ("latest", "auto", "default", "preview")


def _is_floating_alias(e: Entry) -> bool:
    ident = f"{e.id} {e.name}".lower()
    return "~" in ident or any(f"-{w}" in ident or ident.endswith(w) for w in _FLOATING)


def _best_prefix(cat: Catalog, model: str) -> Entry | None:
    """Longest-prefix match over normalized keys, with two guards.

    The naive version of this cost real money before it ever shipped. The
    bundled catalog contains `~openai/gpt-latest`, a floating alias that
    normalizes to the bare key `gpt`. Under a plain `startswith` every
    unrecognised OpenAI id -- `gpt-9-turbo`, anything newer than the snapshot --
    matched it and was silently priced at that alias's $5/$30, with no warning,
    because resolution had succeeded. An unknown model reported as unknown is a
    caveat in the output; an unknown model priced off a wildcard is a wrong
    number presented as a measurement, which is the one failure this repo
    treats as unacceptable.

    So two guards:

    1. **Version boundary.** The unmatched remainder may name a variant
       (`-turbo`, `-mini`) but never a new version number. `gpt-9-turbo` does
       not resolve as `gpt`, because `9` says it is a different generation
       whose price nobody here knows. Dated and effort suffixes are already
       stripped by `normalize_key`, so this only ever rejects real ambiguity.
    2. **No floating aliases.** They are never a prefix target. Their whole
       nature is that the model behind them changes without the key changing.
    """
    key = normalize_key(model)
    if not key:
        return None
    best: Entry | None = None
    for e in cat:
        if not e.key or e.key == key:
            continue
        if not key.startswith(e.key):
            continue
        rest = key[len(e.key):]
        if rest[:1] not in ("-", ".", "_", ":", "["):
            continue                       # matched mid-token, e.g. `gpt-5` in `gpt-50`
        if rest[1:2].isdigit():
            continue                       # a different generation, not a variant
        if _is_floating_alias(e):
            continue
        if best is None or len(e.key) > len(best.key):
            best = e
    return best


@lru_cache(maxsize=2048)
def _resolve_cached(model: str, key: tuple) -> ModelSpec:
    spec = _from_first_party(model) or _from_catalog(model)
    if spec is None:
        raise UnknownModelError(
            f"unknown model {model!r}: not in the first-party table and not in "
            f"the catalog ({len(catalog())} models). Run `adder models refresh`, "
            f"or add it to .adder/catalog.json")
    return spec


def resolve(model: str) -> ModelSpec:
    """Resolve any model id, from any provider, to a full spec.

    Aliases, dated suffixes (`-20251001`), context-variant suffixes (`[1m]`),
    and vendor-prefixed routing slugs (`anthropic/claude-opus-5`) all resolve.
    """
    if not model:
        raise UnknownModelError("empty model id")
    return _resolve_cached(model, _cache_key())


def is_known(model: str) -> bool:
    try:
        resolve(model)
        return True
    except UnknownModelError:
        return False


def is_priced(model: str) -> bool:
    """Known *and* carrying a price. The gate reports need before quoting USD."""
    try:
        return resolve(model).priced
    except UnknownModelError:
        return False


def provider_for(model: str) -> Provider:
    try:
        return resolve(model).provider
    except UnknownModelError:
        return _prov.UNKNOWN


# ---------------------------------------------------------------------------
# Convenience wrappers mirroring the `prices` surface, but universal
# ---------------------------------------------------------------------------

def rate(model: str, on: date | None = None, *, speed: str = "standard"):
    return resolve(model).rate(on, speed=speed)


def context_limit(model: str) -> int | None:
    """None means unknown, and callers must not read that as unlimited."""
    return resolve(model).context


def context_window(model: str, default: int = 0) -> int:
    """The context limit as a *number*, for arithmetic. Never None.

    `context_limit` returns None for the 53 bundled catalog entries whose
    window nobody publishes, and eight call sites did arithmetic or `:,`
    formatting on it directly: `min(need, context_limit(m))` raised
    `TypeError: '<' not supported between NoneType and int`, and an f-string
    raised `unsupported format string passed to NoneType.__format__`. Both are
    reachable by pointing the `ladder` setting at any such model, which is a
    documented thing to do.

    The caller names the fallback, because the safe direction differs: a
    feasibility check wants 0 (unknown does not fit), a cost cap wants the
    request size (do not clip what you cannot measure).
    """
    got = resolve(model).context
    return int(got) if got else int(default)


def limit_str(model: str) -> str:
    """`"200,000-token"` or `"undeclared"`. Never `"None-token"`.

    The display half of the same problem. `cost._limit_str` had this right and
    was the only place that did; every report that wanted to name a window in a
    sentence formatted the raw `int | None` instead.
    """
    try:
        n = resolve(model).context
    except UnknownModelError:
        return "undeclared (this model is not in the catalog)"
    return f"{n:,}-token" if n else "undeclared (catalog has no context window for it)"


def fits(model: str, tokens: int) -> bool:
    return resolve(model).fits(tokens)


def cache_min(model: str) -> int:
    return resolve(model).cache_min


def caches(model: str, prefix_tokens: int) -> bool:
    return resolve(model).caches(prefix_tokens)


def supports_effort(model: str, level: str) -> bool:
    return resolve(model).supports_effort(level)


def candidates(
    *,
    min_context: int = 0,
    needs_tools: bool = False,
    org: str | None = None,
    priced_only: bool = True,
    caching_only: bool = False,
) -> list[ModelSpec]:
    """Every model that could take the work, cheapest input rate first.

    This replaces the hand-written three-tier Claude ladder that was wrong
    within a week of any launch. `caching_only` exists because in a long
    session a model with no prompt cache is usually not a candidate at any
    price -- the carry term buries the rate difference.
    """
    out: list[ModelSpec] = []
    for e in catalog().find(org=org, needs_tools=needs_tools,
                            min_context=min_context, priced_only=priced_only):
        try:
            spec = resolve(e.key)
        except UnknownModelError:
            continue
        if caching_only and not spec.provider.caches:
            continue
        if min_context and not spec.fits(min_context):
            continue
        out.append(spec)
    return sorted(out, key=lambda s: (s.inp if s.inp is not None else float("inf"),
                                      s.out if s.out is not None else float("inf"),
                                      s.id))


def cheapest_that_fits(tokens: int, *, needs_tools: bool = False,
                       caching_only: bool = False) -> ModelSpec | None:
    got = candidates(min_context=tokens, needs_tools=needs_tools,
                     caching_only=caching_only)
    return got[0] if got else None
