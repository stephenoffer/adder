"""Date-aware Claude price table.

Rates are USD per million tokens, first-party Claude API list price.

Time matters here: Claude Sonnet 5 ships at an introductory $2/$10 that reverts
to $3/$15 after 2026-08-31. Any threshold tuned against the intro rate is wrong
the day it expires, so every lookup takes an `on` date.
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

    def rate(self, on: date | None = None) -> Rate:
        on = on or date.today()
        if self.intro and self.intro_until and on <= self.intro_until:
            return self.intro
        return self.base


# Ordered cheapest-first; `tier_order` depends on this.
MODELS: dict[str, Model] = {
    "claude-haiku-4-5": Model("claude-haiku-4-5", Rate(1, 5), context=200_000),
    "claude-sonnet-5": Model(
        "claude-sonnet-5",
        base=Rate(3, 15),
        intro=Rate(2, 10),
        intro_until=date(2026, 8, 31),
    ),
    "claude-sonnet-4-6": Model("claude-sonnet-4-6", Rate(3, 15)),
    "claude-opus-5": Model("claude-opus-5", Rate(5, 25)),
    "claude-opus-4-8": Model("claude-opus-4-8", Rate(5, 25)),
    "claude-opus-4-7": Model("claude-opus-4-7", Rate(5, 25)),
    "claude-opus-4-6": Model("claude-opus-4-6", Rate(5, 25)),
    "claude-fable-5": Model("claude-fable-5", Rate(10, 50)),
    "claude-mythos-5": Model("claude-mythos-5", Rate(10, 50)),
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


class UnknownModel(KeyError):
    pass


def resolve(model: str) -> Model:
    """Resolve an alias, exact id, or dated variant (e.g. `-20251001` suffix)."""
    if model in ALIASES:
        model = ALIASES[model]
    if model in MODELS:
        return MODELS[model]
    # Transcripts carry dated ids like claude-haiku-4-5-20251001.
    for mid, m in MODELS.items():
        if model.startswith(mid):
            return m
    raise UnknownModel(
        f"unknown model {model!r}; known: {sorted(MODELS) + sorted(ALIASES)}"
    )


def rate(model: str, on: date | None = None) -> Rate:
    return resolve(model).rate(on)


def is_known(model: str) -> bool:
    try:
        resolve(model)
        return True
    except UnknownModel:
        return False
