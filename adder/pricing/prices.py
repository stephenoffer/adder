"""Date-aware Claude price and capability table.

Rates are USD per million tokens, first-party Claude API list price.

Three things here are load-bearing and usually omitted from cost models:

1. **Time.** Rates move, and not always the way they were announced. Sonnet 5
   launched at an "introductory" $2/$10 due to revert to $3/$15 after
   2026-08-31; the revert was cancelled and $2/$10 became the list price. This
   table modelled the announced revert and billed every Sonnet 5 turn from
   1 September at 1.5x. Every lookup still takes an `on` date, because the next
   introductory rate may well expire as announced.

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

import re
from dataclasses import dataclass
from datetime import date, timedelta
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
    # Cache read as a fraction of the input rate, where the published rate is
    # not the family's 0.10. None means `CACHE_READ_MULT`.
    cache_read_mult: float | None = None

    def rate(self, on: date | None = None, *, speed: str = "standard") -> Rate:
        if speed == "fast":
            if self.fast is None:
                raise UnsupportedSpeedError(f"{self.id} has no fast mode")
            return self.fast
        on = on or date.today()
        if self.intro and self.intro_until and on <= self.intro_until:
            return self.intro
        return self.base

    def fits(self, tokens: int) -> bool:
        return tokens <= self.context


class UnknownModelError(KeyError):
    pass


class UnsupportedSpeedError(ValueError):
    pass


# Ordered cheapest-first; `tier_order` depends on this.
MODELS: dict[str, Model] = {
    "claude-haiku-4-5": Model(
        "claude-haiku-4-5", Rate(1, 5),
        context=200_000, max_output=64_000, cache_min=4096,
        efforts=(),                      # Haiku 4.5 rejects `effort`
    ),
    # $2/$10 is the list price, not an introductory one: the scheduled move
    # to $3/$15 on 2026-09-01 was cancelled (pricing page, footnote 3).
    "claude-sonnet-5": Model("claude-sonnet-5", Rate(2, 10), cache_min=1024),
    "claude-sonnet-4-6": Model("claude-sonnet-4-6", Rate(3, 15), cache_min=1024),
    # Opus 5.5 is cheaper than Opus 5 and was priced as Opus 5 until it had a
    # row: longest prefix resolved `claude-opus-5-5` to `claude-opus-5`, so
    # every turn on it billed 1.25x its input and output and 2.5x its cache
    # reads, which are most of a session. Published: $4/$20, cache read $0.20,
    # fast $8/$40. The cache minimum is not published separately; it is taken
    # from Opus 5, whose tokenizer and feature set it shares.
    "claude-opus-5-5": Model(
        "claude-opus-5-5", Rate(4, 20),
        cache_min=512, fast=Rate(8, 40), cache_read_mult=0.05,
    ),
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
    # Same list price as Fable 5; cache reads publish at $0.25, not $1.00.
    "claude-fable-5-1": Model("claude-fable-5-1", Rate(10, 50), cache_min=512,
                              cache_read_mult=0.025),
    "claude-mythos-5": Model("claude-mythos-5", Rate(10, 50), cache_min=512),
    # Without its own row this resolved to `claude-mythos-5` by prefix and
    # billed cache reads at 4x, marked verified.
    "claude-mythos-5-1": Model("claude-mythos-5-1", Rate(10, 50), cache_min=512,
                               cache_read_mult=0.025),
}

# Claude Code writes an assistant record with this model id when the *client*
# produced the message rather than the API: "API Error: Connection closed
# mid-response", an interrupted stream, a local error placeholder. Their usage
# block is all zeros and nothing was billed for them.
#
# They matter twice. Counted as turns they inflate the turn count and depress
# every per-turn average; counted as an unknown model they raise a "this report
# is a lower bound" warning about spend that does not exist. They are neither,
# so they get their own predicate and `trace` reports them as what they are --
# a count of API failures, which is a quality signal rather than a cost one.
SYNTHETIC_MODELS = frozenset({"<synthetic>"})


def is_synthetic(model: str) -> bool:
    return model in SYNTHETIC_MODELS


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
    # wins so claude-sonnet-4-6 never matches as claude-sonnet-5. A prefix
    # followed by a version number is a different model, not a variant:
    # `claude-opus-5-5` priced as `claude-opus-5`, and `claude-mythos-5-1` as
    # `claude-mythos-5`, both silently and both marked verified. An unknown
    # point release has to surface as unknown.
    best: Model | None = None
    for mid, m in MODELS.items():
        if not model.startswith(mid) or _is_point_release(model[len(mid):]):
            continue
        if best is None or len(mid) > len(best.id):
            best = m
    if best is not None:
        return best
    raise UnknownModelError(
        f"unknown model {model!r}; known: {sorted(MODELS) + sorted(ALIASES)}"
    )


_POINT_RELEASE = re.compile(r"-\d{1,7}(?!\d)")


def _is_point_release(rest: str) -> bool:
    """Whether what follows a matched prefix names another version.

    `-5`, `-1` and `-1[1m]` do; `-20251001` (a date stamp), `[1m]` and
    `-fast` do not.
    """
    return _POINT_RELEASE.match(rest) is not None


def rate(model: str, on: date | None = None, *, speed: str = "standard") -> Rate:
    return resolve(model).rate(on, speed=speed)


def is_known(model: str) -> bool:
    try:
        resolve(model)
        return True
    except UnknownModelError:
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


def expiring_soon(within_days: int = 30, on: date | None = None
                  ) -> list[tuple[str, date, Rate, Rate]]:
    """Models whose introductory rate ends within `within_days`.

    Returns `(model_id, when, intro_rate, base_rate)`, soonest first. Every
    threshold in this repo -- the delegate-above size, the escalation
    break-even, the session-model choice -- is a ratio of two prices, so a rate
    change is a re-tune, not a footnote. Reports surface this rather than
    letting a number quietly become wrong overnight.
    """
    today = on or date.today()
    out = []
    for mid, m in MODELS.items():
        if not (m.intro and m.intro_until):
            continue
        if today <= m.intro_until <= today + timedelta(days=within_days):
            out.append((mid, m.intro_until, m.intro, m.base))
    return sorted(out, key=lambda row: row[1])
