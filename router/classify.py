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

# The rungs, as a table rather than a literal buried in a property.
#
# A ladder written into code is correct the day it is written and quietly wrong
# after the next launch. Keeping it here means `rt models ladder` can diff it
# against the live catalog and show the drift, and a project that wants a
# different rung can rebind one entry instead of forking the enum. Defaults
# stay pinned: the catalog *reports*, it does not silently repoint dispatch.
LADDER: dict[str, str] = {
    "T0": "claude-haiku-4-5",
    "T1": "claude-sonnet-5",
    "T2": "claude-opus-5",
    "T3": "claude-opus-5",
}


class Tier(IntEnum):
    T0 = 0   # haiku,  read-only
    T1 = 1   # sonnet, scoped edits
    T2 = 2   # opus,   multi-file / ambiguous
    T3 = 3   # opus xhigh, long-horizon

    @property
    def model(self) -> str:
        return LADDER[self.name]

    @property
    def effort(self) -> str:
        return {0: "low", 1: "medium", 2: "high", 3: "xhigh"}[int(self)]

    @property
    def agent(self) -> str:
        return {0: "route-t0", 1: "route-t1", 2: "route-t2", 3: "route-t2"}[int(self)]


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
_PATHLIKE = re.compile(r"[\w./-]+\.(py|ts|tsx|js|jsx|go|rs|java|rb|md|json|ya?ml|toml|sh)\b")


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


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(prog="router.classify")
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
