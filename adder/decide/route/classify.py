"""Task complexity classification from text alone - deliberately modest.

Why this is small on purpose
----------------------------
"Fix the login bug" is four words and unbounded work. Text features cannot
predict how deep a coding task goes, which is why published routers plateau on
agentic benchmarks. So this classifier does not try to grade everything. It
fires only on high-precision extremes and abstains otherwise, leaving the real
work to the escalation loop, which observes actual failure instead of guessing.

Abstaining routes UP: a misrouted hard task costs a full retry, a misrouted easy
one costs pennies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum
from itertools import pairwise

# The rungs, as a table rather than a literal buried in a property.
#
# A ladder written into code is correct the day it is written and quietly wrong
# after the next launch. Keeping it here means `adder models ladder` can diff it
# against the live catalog and show the drift, and a project that wants a
# different rung can rebind one entry instead of forking the enum. Defaults
# stay pinned: the catalog *reports*, it does not silently repoint dispatch.
DEFAULT_LADDER: dict[str, str] = {
    "T0": "claude-haiku-4-5",
    "T1": "claude-sonnet-5",
    "T2": "claude-opus-5",
    "T3": "claude-opus-5",
}


def ladder() -> dict[str, str]:
    """The ladder in effect, from the `ladder` setting if one is set.

    The default is Claude because that is what this tool reads transcripts from
    and what its measurements were taken on. That default is wrong for anybody
    running Codex or Gemini CLI, and before this there was no way to say so
    short of editing the source -- which is a poor answer for the one setting
    that decides where every dispatched task actually goes.

    Written `T0=gpt-5-mini,T1=gpt-5,T2=gpt-5-pro`. Unnamed rungs keep their
    default rather than disappearing, so a partial override cannot leave a tier
    pointing at nothing. Unparseable entries are ignored: a typo in a config
    file must not silently repoint dispatch either.
    """
    from adder.core.settings import get as _setting

    out = dict(DEFAULT_LADDER)
    try:
        raw = str(_setting("ladder") or "").strip()
    except (KeyError, OSError, ValueError):
        # A broken config file must not take dispatch down with it. The pinned
        # default is always a working ladder.
        return out
    if not raw:
        return out
    for part in raw.split(","):
        rung, _, model = part.partition("=")
        rung, model = rung.strip().upper(), model.strip()
        if rung in out and model:
            out[rung] = model
    return out


# Kept as a name for the *pinned* rungs. `ladder()` is what dispatch and the
# drift report should call: config is resolved per use rather than at import,
# because settings are deliberately uncached and a ladder frozen at import
# would ignore the environment a caller just set.
LADDER: dict[str, str] = DEFAULT_LADDER


def ladder_warnings(on=None) -> list[str]:
    """Ways the configured ladder is not a ladder.

    A ladder is only useful if climbing it costs more. Once the rungs became
    configurable, three ways to break that became reachable, and all three are
    silent: a rung naming a model nothing can price, a rung whose context
    window is smaller than the rung below it, and -- the one that actually
    happened while testing -- a partial override that repoints T0..T2 at a new
    vendor and leaves T3 on the old default, so the "most capable" rung ends up
    cheaper than the one under it.

    None of these raise. Dispatch still works with a crooked ladder; it just
    stops being an argument for anything, and the reader deserves to be told
    which rung to look at.
    """
    from adder.pricing.registry import UnknownModelError, UnpricedModelError, resolve

    out: list[str] = []
    priced: list[tuple[str, float, int | None]] = []
    for rung in sorted(ladder()):
        model = ladder()[rung]
        try:
            spec = resolve(model)
            r = spec.rate(on)
        except UnknownModelError:
            out.append(f"{rung} names {model!r}, which is not in the catalog or "
                       f"the first-party table; nothing dispatched there can be priced")
            continue
        except UnpricedModelError:
            out.append(f"{rung} names {model!r}, which nobody publishes a price "
                       f"for; every estimate through this rung is a guess")
            continue
        priced.append((rung, r.inp + r.out, spec.context))

    for (lo, lo_cost, lo_ctx), (hi, hi_cost, hi_ctx) in pairwise(priced):
        if hi_cost < lo_cost:
            out.append(f"{hi} ({ladder()[hi]}) is cheaper than {lo} "
                       f"({ladder()[lo]}); the ladder does not climb, so "
                       f"escalating from {lo} to {hi} saves money instead of "
                       f"spending it and every tier comparison inverts")
        if lo_ctx and hi_ctx and hi_ctx < lo_ctx:
            out.append(f"{hi} holds {hi_ctx:,} tokens but {lo} holds {lo_ctx:,}; "
                       f"escalating can fail on context alone")
    return out


class Tier(IntEnum):
    T0 = 0   # haiku,  read-only
    T1 = 1   # sonnet, scoped edits
    T2 = 2   # opus,   multi-file / ambiguous
    T3 = 3   # opus xhigh, long-horizon

    @property
    def model(self) -> str:
        return ladder()[self.name]

    @property
    def effort(self) -> str:
        return {0: "low", 1: "medium", 2: "high", 3: "xhigh"}[int(self)]

    @property
    def agent(self) -> str:
        return {0: "route-t0", 1: "route-t1", 2: "route-t2", 3: "route-t2"}[int(self)]


# How hard the work at each rung is, as a multiplier on an Elo gap. A lookup
# tolerates a much weaker model than a multi-file refactor does.
#
# Lives here rather than in `policy` because it is keyed on `Tier` and two
# modules need it: `policy.substitutes` prices a cross-vendor swap with it, and
# `frontier` sets its quality floor from it. `frontier` used to pass the
# classifier's *confidence* instead, which is not difficulty and is close to its
# inverse in the case that matters -- an abstention (confidence 0.3) is the
# classifier saying it cannot tell how deep the task goes, and it routes such
# tasks UP. Read as a difficulty, 0.3 is the easiest setting there is, so the
# tasks the classifier understood least were the ones offered the weakest
# models with the widest quality tolerance.
TIER_DIFFICULTY: dict[Tier, float] = {
    Tier.T0: 0.4, Tier.T1: 0.7, Tier.T2: 1.0, Tier.T3: 1.4,
}


# High-precision cheap signals: read-only questions with a bounded answer.
_TRIVIAL = re.compile(
    r"^\s*(what|where|which|who|when|does|is|are|list|show|find|locate|read|"
    r"print|cat|grep|count|how many)\b", re.I)

# High-precision expensive signals: open-ended, cross-cutting, or design work.
_HARD = re.compile(
    r"\b(architect|design|refactor|redesign|migrat\w+|rewrite|overhaul|"
    r"debug|root[- ]cause|investigat\w+|why (is|does|are|did)|"
    r"across (the )?(codebase|repo|service|system)|end[- ]to[- ]end|"
    r"performance|concurren\w+|race condition|deadlock|security|threat model)\b", re.I)

_MUTATING = re.compile(
    r"\b(fix|change|update|edit|add|remove|delete|rename|implement|write|"
    r"create|refactor|patch|bump|install|configure|set up)\b", re.I)

_MULTI_STEP = re.compile(r"\b(then|after that|and also|next,|finally)\b|^\s*\d+[.)]\s", re.I | re.M)
_CODE_FENCE = re.compile(r"```")
_STACK_TRACE = re.compile(r"(Traceback \(most recent|^\s+at [\w.$]+\(|Exception|Error:)", re.M)
_PATHLIKE = re.compile(r"[\w./-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|rb|md|json|ya?ml|toml|sh)\b")


@dataclass
class Verdict:
    tier: Tier
    confidence: float           # 0..1; low means "abstained, routed up"
    reasons: list[str] = field(default_factory=list)
    read_only: bool = False

    @property
    def abstained(self) -> bool:
        return self.confidence < 0.5


def classify(task: str) -> Verdict:
    """Classify a task description. Pure function, no I/O, no network."""
    t = (task or "").strip()
    if not t:
        return Verdict(Tier.T2, 0.0, ["empty task; defaulting up"], read_only=False)

    words = len(t.split())
    reasons: list[str] = []

    hard = bool(_HARD.search(t))
    mutating = bool(_MUTATING.search(t))
    trivial_shape = bool(_TRIVIAL.match(t))
    multi = bool(_MULTI_STEP.search(t))
    files = len(set(_PATHLIKE.findall(t)))
    trace = bool(_STACK_TRACE.search(t))
    fenced = bool(_CODE_FENCE.search(t))

    # --- high-precision expensive signals: decide first, never route these down
    if hard:
        reasons.append("matches design/debug/cross-cutting vocabulary")
        tier = Tier.T3 if (multi or words > 120) else Tier.T2
        if multi:
            reasons.append("multi-step phrasing")
        return Verdict(tier, 0.85, reasons)

    if trace:
        reasons.append("contains a stack trace or error output")
        return Verdict(Tier.T2, 0.8, reasons)

    if multi and mutating:
        reasons.append("multi-step mutating request")
        return Verdict(Tier.T2, 0.7, reasons)

    # --- high-precision cheap signal: short, read-only, single-target lookup
    if trivial_shape and not mutating and words <= 30 and not fenced:
        reasons.append("short read-only question")
        if files <= 1:
            return Verdict(Tier.T0, 0.85, reasons, read_only=True)
        reasons.append(f"spans {files} files")
        return Verdict(Tier.T1, 0.6, reasons, read_only=True)

    # --- scoped mutation of a single named file
    if mutating and files == 1 and words <= 60 and not multi:
        reasons.append("scoped edit to one named file")
        return Verdict(Tier.T1, 0.65, reasons)

    # --- abstain: route up
    reasons.append("no high-precision signal; abstaining and routing up")
    return Verdict(Tier.T2, 0.3, reasons)


def difficulty_of(task: str) -> float:
    """The difficulty multiplier implied by a task's classified tier."""
    return TIER_DIFFICULTY[classify(task).tier]


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(prog="adder classify")
    ap.add_argument("task", nargs="*", help="task text (or stdin)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    text = " ".join(a.task) if a.task else sys.stdin.read()

    v = classify(text)
    if a.json:
        print(json.dumps({
            "tier": v.tier.name, "model": v.tier.model, "effort": v.tier.effort,
            "agent": v.tier.agent, "confidence": v.confidence,
            "read_only": v.read_only, "abstained": v.abstained, "reasons": v.reasons,
        }))
    else:
        print(f"{v.tier.name} ({v.tier.model}, effort={v.tier.effort}) "
              f"confidence={v.confidence:.2f}{' ABSTAINED' if v.abstained else ''}")
        for r in v.reasons:
            print(f"  - {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
