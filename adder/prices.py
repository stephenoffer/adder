"""Date-aware Claude price and capability table.

Rates are USD per million tokens, first-party Claude API list price.

Three things here are load-bearing and usually omitted from cost models:

1. **Time.** Claude Sonnet 5 ships at an introductory $2/$10 that reverts to
   $3/$15 after 2026-08-31. Any threshold tuned against the intro rate is wrong
   the day it expires, so every lookup takes an `on` date.

2. **Context limits.** Haiku 4.5 holds 200K tokens; the measured median session
   context here is 544K. A router that recommends downgrading a 544K
   conversation to Haiku is recommending a 400 error, not a saving. `fits()`
   makes that checkable, and the cost gates refuse rather than "save".

3. **Cache minimums.** The minimum cacheable prefix is model-dependent and
   **not monotonic** across generations: 512 tokens on Opus 5, but 4096 on
   Opus 4.6 and Haiku 4.5. A prefix below the minimum silently does not cache
   -- no error, `cache_creation_input_tokens: 0`. A 2K-token subagent brief
   caches on Opus 5 and does not on Haiku, which changes the arithmetic of
   every delegation decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import NamedTuple


class Rate(NamedTuple):
    """USD per million tokens."""

    inp: float
    out: float


@dataclass(frozen=True)
class Model:
    id: str
    base: Rate
    intro: Rate | None = None
    intro_until: date | None = None
    context: int = 1_000_000
    max_output: int = 128_000
    # Minimum cacheable prefix. Below this, a cache_control marker is a no-op.
    cache_min: int = 1024
    # Fast mode (Claude API only, Opus 5 / Opus 4.8) runs the same model at a
    # premium rate. None means fast mode is unavailable.
    fast: Rate | None = None
    # Effort levels the model accepts, cheapest reasoning first.
    efforts: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

    def rate(self, on: date | None = None, *, speed: str = "standard") -> Rate:
        if speed == "fast":
            if self.fast is None:
                raise UnsupportedSpeed(f"{self.id} has no fast mode")
            return self.fast
        on = on or date.today()
        if self.intro and self.intro_until and on <= self.intro_until:
            return self.intro
        return self.base

    def fits(self, tokens: int) -> bool:
        return tokens <= self.context


class UnknownModel(KeyError):
    pass


class UnsupportedSpeed(ValueError):
    pass


# Ordered cheapest-first; `tier_order` depends on this.
MODELS: dict[str, Model] = {
    "claude-haiku-4-5": Model(
        "claude-haiku-4-5", Rate(1, 5),
        context=200_000, max_output=64_000, cache_min=4096,
        efforts=(),                      # Haiku 4.5 rejects `effort`
    ),
    "claude-sonnet-5": Model(
        "claude-sonnet-5",
        base=Rate(3, 15),
        intro=Rate(2, 10),
        intro_until=date(2026, 8, 31),
        cache_min=1024,
    ),
    "claude-sonnet-4-6": Model("claude-sonnet-4-6", Rate(3, 15), cache_min=1024),
    "claude-opus-5": Model(
        "claude-opus-5", Rate(5, 25),
        cache_min=512,                   # halved vs 4.8; short prefixes now cache
        fast=Rate(10, 50),
    ),
    "claude-opus-4-8": Model("claude-opus-4-8", Rate(5, 25), cache_min=1024, fast=Rate(10, 50)),
    "claude-opus-4-7": Model("claude-opus-4-7", Rate(5, 25), cache_min=2048),
    "claude-opus-4-6": Model("claude-opus-4-6", Rate(5, 25), cache_min=4096,
                             efforts=("low", "medium", "high", "max")),
    "claude-fable-5": Model("claude-fable-5", Rate(10, 50), cache_min=512),
    "claude-mythos-5": Model("claude-mythos-5", Rate(10, 50), cache_min=512),
}

# Claude Code aliases -> concrete ids.
ALIASES = {
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-5",
    "opus": "claude-opus-5",
    "fable": "claude-fable-5",
}

# Cache pricing multipliers, applied to the input rate.
CACHE_READ_MULT = 0.10
CACHE_WRITE_MULT = {"5m": 1.25, "1h": 2.00}
TTL_SECONDS = {"5m": 300, "1h": 3600}

# Batch API: 50% off all token usage, at the cost of async delivery.
BATCH_MULT = 0.50

# A cache breakpoint walks back at most this many content blocks looking for a
# prior entry. Agentic turns that add more blocks than this silently miss.
CACHE_LOOKBACK_BLOCKS = 20


def resolve(model: str) -> Model:
    """Resolve an alias, exact id, or dated variant (e.g. `-20251001` suffix)."""
    if model in ALIASES:
        model = ALIASES[model]
    if model in MODELS:
        return MODELS[model]
    # Transcripts carry dated ids like claude-haiku-4-5-20251001, and Claude
    # Code carries suffixed variants like claude-opus-5[1m]. Longest prefix
    # wins so claude-sonnet-4-6 never matches as claude-sonnet-5.
    best: Model | None = None
    for mid, m in MODELS.items():
        if model.startswith(mid) and (best is None or len(mid) > len(best.id)):
            best = m
    if best is not None:
        return best
    raise UnknownModel(
        f"unknown model {model!r}; known: {sorted(MODELS) + sorted(ALIASES)}"
    )


def rate(model: str, on: date | None = None, *, speed: str = "standard") -> Rate:
    return resolve(model).rate(on, speed=speed)


def is_known(model: str) -> bool:
    try:
        resolve(model)
        return True
    except UnknownModel:
        return False


def context_limit(model: str) -> int:
    return resolve(model).context


def fits(model: str, tokens: int) -> bool:
    """Can `model` hold `tokens` of context at all?

    The gate every naive downgrade misses: Haiku 4.5 tops out at 200K, and the
    measured median session context here is 544K.
    """
    return resolve(model).fits(tokens)


def cache_min(model: str) -> int:
    """Smallest prefix that will actually cache. Below this, caching is a no-op."""
    return resolve(model).cache_min


def caches(model: str, prefix_tokens: int) -> bool:
    return prefix_tokens >= cache_min(model)


def supports_effort(model: str, level: str) -> bool:
    return level in resolve(model).efforts


def tier_order() -> list[str]:
    """Model ids cheapest input-rate first, ties broken by output rate."""
    return sorted(MODELS, key=lambda m: (MODELS[m].base.inp, MODELS[m].base.out))


def cheapest_that_fits(tokens: int, *, at_least: str | None = None) -> str | None:
    """Cheapest model whose context window holds `tokens`.

    `at_least` pins a capability floor: never return something cheaper than it.
    """
    floor = resolve(at_least).base.inp if at_least else 0.0
    for mid in tier_order():
        m = MODELS[mid]
        if m.base.inp >= floor and m.fits(tokens):
            return mid
    return None


def intro_expiry(model: str) -> date | None:
    """When this model's introductory rate ends, if it has one."""
    m = resolve(model)
    return m.intro_until if m.intro else None
