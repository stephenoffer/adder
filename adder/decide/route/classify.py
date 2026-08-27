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

Where that argument does not hold
---------------------------------
It assumes failure is visible, which is true of coding work and false of
recall. A weak model asked for every hardcoded credential in a tree returns
three of the seven, confidently, and nothing fails -- so the cost of the
misroute is not a retry, it is a wrong audit that reads as a right one. The
signal that separates the two is not difficulty, it is whether an incomplete
answer is detectable, and `_QUANTIFIER` plus `_plural_target` is the cheap,
high-precision version of it. Those tasks abstain regardless of how easy the
sentence looks.
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

# The subset of `_TRIVIAL` that asks for a *set* rather than a fact. "what does
# map_batches do" has one right answer and a wrong one is obvious; "find the
# hardcoded credentials" has an answer whose size nobody knows in advance.
_ENUMERATE = re.compile(
    r"^\s*(list|show|find|locate|grep|search|count|enumerate|which|how many)\b", re.I)

# Exhaustiveness, stated. Paired with a plural target this is the one cheap,
# high-precision way to spot a task whose failure mode is silent.
_QUANTIFIER = re.compile(
    r"\b(every|all|each|any|entire|whole|everywhere|anywhere|throughout|"
    r"exhaustiv\w*|comprehensiv\w*)\b", re.I)

# High-precision expensive signals: the *work* is open-ended, cross-cutting or
# investigative. Every entry here names something to be done, not something to
# be done to. See `_HARD_TOPIC` for why that distinction is load-bearing.
_HARD = re.compile(
    r"\b(architect|design|refactor|redesign|migrat\w+|rewrite|overhaul|"
    r"root[- ]cause|investigat\w+|why (is|does|are|did)|"
    r"across (the )?(codebase|repo|service|system)|end[- ]to[- ]end|"
    r"threat model)\b", re.I)

# Domain topics, which used to live in `_HARD` and decide on their own.
#
# They are nouns. "where is the security module" is a one-line grep and was
# priced as a threat model; "add a docstring to the debug helper" is a
# one-line edit and was priced as a debugging session. On a repository whose
# subject matter *is* performance, security, concurrency and debugging, a
# vocabulary match is evidence about the repository, not about the task, and it
# fired on roughly every request at about five times the cheapest rung's price.
# So a topic word no longer decides anything; it only explains the abstention.
_HARD_TOPIC = re.compile(
    r"\b(performance|concurren\w+|race condition|deadlock|security|"
    r"debug\w*|profil\w+|scalab\w+)\b", re.I)

_MUTATING = re.compile(
    r"\b(fix|change|update|edit|add|remove|delete|rename|implement|write|"
    r"create|refactor|patch|bump|install|configure|set up)\b", re.I)

_MULTI_STEP = re.compile(r"\b(then|after that|and also|next,|finally)\b|^\s*\d+[.)]\s", re.I | re.M)
_CODE_FENCE = re.compile(r"```")

# A stack trace, not the word "Exception".
#
# This matched the bare words `Exception` and `Error:` anywhere in the text, so
# "rename the Exception class" -- a mechanical edit -- read as a crash report
# and went to the top rung at 0.80 confidence. A trace has shape: a Python
# `Traceback` header, an indented `at frame(` line, a `File "x", line n` line,
# or an exception name followed by a colon and a message.
_STACK_TRACE = re.compile(
    r"Traceback \(most recent"
    r"|^\s+at [\w.$]+\("
    r'|^\s*File "[^"]+", line \d+'
    r"|\b[\w.]*(?:Error|Exception)\s*:\s*\S", re.M)

_PATHLIKE = re.compile(r"[\w./-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|rb|md|json|ya?ml|toml|sh)\b")

# Words ending in "s" that are not plural nouns.
#
# The plural test below is one regex, and without this list it reads "which
# function computes the rate" as a set query -- because "computes" ends in s --
# which is exactly the singular lookup the cheapest rung exists for. Third
# person verbs are the whole failure mode; these are the ones that actually
# turn up in task phrasing. Words ending in `ss`, `us` or `is` are excluded
# arithmetically rather than listed.
_NOT_PLURAL = frozenset((
    "does", "goes", "says", "means", "runs", "works", "fails", "breaks",
    "crashes", "hangs", "spills", "leaks", "computes", "matches", "returns",
    "takes", "makes", "uses", "needs", "gets", "calls", "handles", "sets",
    "gives", "keeps", "comes", "looks", "seems", "happens", "causes",
    "contains", "includes", "requires", "produces", "provides", "parses",
    "raises", "reads", "writes", "prints", "sends", "loads", "saves",
    "stores", "emits", "shows", "finds", "lists", "counts", "holds", "knows",
    "leaves", "lives", "sits", "stands", "starts", "stops",
    "thus", "yes", "less", "unless", "perhaps", "always", "across", "versus",
    "its",
))

_WORD_S = re.compile(r"\b([A-Za-z][\w-]{2,}s)\b")


def _plural_target(task: str) -> bool:
    """Does the text name a set of things rather than a single one?

    Deliberately crude -- a plural noun is a word ending in "s" that is not one
    of the verb forms in `_NOT_PLURAL` and does not end in `ss`/`us`/`is`. It
    is a precision filter, not a parser: its only job is to keep a quantifier
    from firing on "how many times does it retry".
    """
    for w in _WORD_S.findall(task):
        lw = w.lower()
        if lw in _NOT_PLURAL or lw.endswith(("ss", "us", "is")):
            continue
        return True
    return False


@dataclass
class Verdict:
    tier: Tier
    confidence: float           # 0..1; low means "abstained, routed up"
    reasons: list[str] = field(default_factory=list)
    # True only when no verb in the text asks for a change and the sentence is
    # shaped like a question. It is a *permission*, not a description, so it is
    # claimed conservatively -- and it is consumed: `policy.right_size` refuses
    # the T0 rung without it, because `route-t0` dispatches to an agent holding
    # Read, Grep, Glob and Bash and no write tool at all. It was published in
    # `--json` long before anything in here read it, which left an integrator
    # free to gate a write permission on a field the tool itself did not use.
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
    topic = bool(_HARD_TOPIC.search(t))
    mutating = bool(_MUTATING.search(t))
    trivial_shape = bool(_TRIVIAL.match(t))
    enumerating = bool(_ENUMERATE.match(t))
    quantified = bool(_QUANTIFIER.search(t))
    plural = _plural_target(t)
    multi = bool(_MULTI_STEP.search(t))
    files = len(set(_PATHLIKE.findall(t)))
    trace = bool(_STACK_TRACE.search(t))
    fenced = bool(_CODE_FENCE.search(t))

    # Conservative: a question-shaped sentence containing no verb that asks for
    # a change. Anything else -- including every abstention on a bare imperative
    # like "make it better" -- withholds the claim rather than guessing at it.
    read_only = trivial_shape and not mutating and not fenced

    # --- high-precision expensive signals: decide first, never route these down
    if hard:
        reasons.append("matches design/investigation/cross-cutting vocabulary")
        tier = Tier.T3 if (multi or words > 120) else Tier.T2
        if multi:
            reasons.append("multi-step phrasing")
        return Verdict(tier, 0.85, reasons, read_only=read_only)

    if trace:
        reasons.append("contains a stack trace or error output")
        return Verdict(Tier.T2, 0.8, reasons, read_only=read_only)

    if multi and mutating:
        reasons.append("multi-step mutating request")
        return Verdict(Tier.T2, 0.7, reasons)

    # --- recall-critical: the answer is a set, and a short one looks right
    #
    # The justification for routing an easy-looking task down is that a
    # misrouted easy one costs pennies. That holds when failure is visible,
    # which it is for coding work: the test fails, you retry. It is false for
    # recall. A weak model that finds three of seven hardcoded credentials
    # returns a confident list of three, nothing fails, and nothing retries --
    # so the cost of being wrong is not a retry, it is the audit being wrong.
    #
    # The signal is not difficulty, it is whether an incomplete answer is
    # detectable. A stated quantifier over a plural target is the cheap,
    # high-precision version of that, and it forces the abstention rather than
    # picking a rung: the classifier genuinely cannot tell how much work is
    # behind "every hardcoded credential across all 3167 python files".
    if quantified and plural:
        reasons.append(
            "quantifier over a plural target: the answer is a set, so an "
            "incomplete one is not detectable and recall outranks price")
        return Verdict(Tier.T2, 0.3, reasons, read_only=read_only)

    # --- high-precision cheap signal: short, read-only, single-target lookup
    if trivial_shape and not mutating and words <= 30 and not fenced:
        if enumerating and plural:
            # No quantifier, so this is weaker than the rule above and gets a
            # weaker answer rather than an abstention. Completeness is still
            # part of the answer, which is enough to keep it off the rung whose
            # under-reporting nothing would catch.
            reasons.append("enumerates a plural target; completeness is part of "
                           "the answer, so not the weakest rung")
            return Verdict(Tier.T1, 0.55, reasons, read_only=True)
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
    if topic:
        reasons.append("domain vocabulary matched but did not decide: a topic "
                       "word is evidence about the repository, not the task")
    return Verdict(Tier.T2, 0.3, reasons, read_only=read_only)


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
