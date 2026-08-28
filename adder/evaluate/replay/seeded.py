"""A source file with known defects in it, and a way to score who found them.

Why this is separate from everything else here
----------------------------------------------
`adder quality` and `adder verify` are this tool grading its own homework. Both
read the same transcripts, through the same parser, and price them with the
same cost model that produced the number they are checking. If that model is
wrong -- about result sizes, about the carry multiplier, about which records to
deduplicate -- the check is wrong in exactly the same direction, and it agrees
with itself.

That asymmetry runs the wrong way for what is at stake. Cost is the easy thing
to measure and it is measured five ways; quality is the thing that would
actually be lost by routing work to a cheaper tier, and it was measured by the
same machinery as the cost. So this file exists to supply a signal that shares
no code with any of it: a fixture with K defects planted in it, a prompt, and a
count of how many of the K came back.

Nothing here imports `cost`, `trace`, `shapes` or `carry`. The only number it
produces is `found / K`, and the only thing it needs to compute that is text.

Why a synthetic file rather than a real bug
-------------------------------------------
Because the ground truth has to be complete. A real file has an unknown number
of defects, so a model that reports four findings on it cannot be scored: there
is no K. Recall is the measurement that matters here -- an incomplete audit
that reads like a complete one is the failure mode the whole routing argument
turns on -- and recall needs a denominator that is known rather than estimated.

Each defect is real, ordinary, and independently findable. None of them is a
riddle: the point is not to be hard, it is to be *countable*. A tier that
cannot find eight obvious defects in eighty lines is not one to hand an audit
to, and a tier that finds all eight has demonstrated something a cost model
cannot demonstrate at all.

What a score does and does not license
--------------------------------------
It licenses a statement about *recall on defect-finding over supplied source*,
which is the task class the classifier abstains on and the one whose failure is
silent. It says nothing about multi-step work, tool use, or long-horizon
coherence -- the same scope limit `ab` states about its own comprehension
tasks, for the same reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# The prompt every arm is given, verbatim. Deliberately not a hint: it does not
# say how many defects there are, because a model told "find eight" will report
# eight things whether or not it found them, and the number it was told is then
# doing the work the measurement was supposed to do.
PROMPT = (
    "The file below is a small utility module. Find every defect in it. "
    "For each one, name the function it is in and say what is wrong in a "
    "sentence. Do not suggest style improvements; report only things that "
    "are wrong."
)


@dataclass(frozen=True)
class Seed:
    """One planted defect, and what naming it looks like.

    `symbol` anchors the finding: a reply that discusses an off-by-one without
    saying where has not found this one, and crediting it would let a model
    score by writing plausible sentences about software in general. `any_of`
    then has to match some word for the defect itself, so naming the function
    for an unrelated reason does not count either.

    Both halves are needed and neither is sufficient. That is stricter than it
    needs to be for a good answer and exactly as strict as it needs to be for a
    bad one.
    """
    id: str
    symbol: str
    any_of: tuple[str, ...]
    what: str

    def found_in(self, reply: str) -> bool:
        text = _normalise(reply)
        if _normalise(self.symbol) not in text:
            return False
        # A marker that normalises to nothing -- `"//"` was one -- matches every
        # string, so the seed it belongs to is credited to any reply that names
        # the function. Dropped rather than allowed: a scorer with one free
        # point in it is a scorer that flatters whatever it measures.
        needles = [n for n in (_normalise(w) for w in self.any_of) if n]
        return any(n in text for n in needles)


def _normalise(text: str) -> str:
    """Lowercase, with runs of non-word characters collapsed to single spaces.

    So that `write_atomic`, `write atomic` and `` `write_atomic()` `` are one
    string. A scorer that is defeated by a backtick is measuring formatting.
    """
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


# The fixture. Eighty-odd lines, eight defects, one per function, each of a
# different kind so that a model which is only good at spotting one kind scores
# as what it is.
#
# It is a string rather than a file on disk for two reasons. A `.py` file in the
# package would be imported by anything that walks the tree, and `ruff` would
# be right about every line of it; and a data file needs a package-data glob,
# which is exactly the omission that left four releases shipping no hooks.
SOURCE = '''\
"""Small helpers for a request cache. Not part of adder -- a fixture."""

import json
import os
import time


def parse_window(spec):
    """`"3-7"` -> the pages 3 through 7 inclusive."""
    lo, hi = spec.split("-")
    return list(range(int(lo), int(hi)))


def average_cost(costs):
    """Mean cost across a batch."""
    return sum(costs) / len(costs)


def merge_settings(base, override):
    """`base` updated with `override`."""
    base.update(override)
    return base


def read_config(path):
    """The config file as a dict, or an empty one."""
    try:
        return json.load(open(path))
    except:
        return {}


def token_price(tokens, dollars_per_million):
    """What `tokens` cost at this rate."""
    return tokens * dollars_per_million // 1_000_000


def cache_key(prompt, temperature):
    """A key for the response cache."""
    return f"{prompt}:{temperature}"


def retry_fetch(fetch, url):
    """Call `fetch(url)` until it works."""
    while True:
        try:
            return fetch(url)
        except OSError:
            time.sleep(1)


def write_atomic(path, text):
    """Write `text` to `path` without a torn read."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)
'''

SEEDS: tuple[Seed, ...] = (
    Seed("range-off-by-one", "parse_window",
         ("off by one", "off-by-one", "exclusive", "inclusive", "hi 1", "last page",
          "excludes", "misses"),
         "range() is half-open, so the last page in the span is dropped"),
    Seed("empty-division", "average_cost",
         ("empty", "zero", "zerodivision", "divide by zero", "division by zero",
          "len 0"),
         "an empty batch divides by zero"),
    Seed("caller-mutation", "merge_settings",
         ("mutat", "in place", "in-place", "modifies", "side effect", "caller"),
         "it edits the caller's dict rather than returning a new one"),
    Seed("bare-except", "read_config",
         ("bare except", "bare-except", "swallow", "catches everything",
          "keyboardinterrupt", "too broad", "broad except", "hides"),
         "a bare except hides every error, including the ones that matter"),
    Seed("leaked-handle", "read_config",
         ("close", "leak", "handle", "context manager", "with open", "file object"),
         "`open()` inside `json.load` is never closed"),
    Seed("integer-division", "token_price",
         ("integer division", "floor division", "integer divide", "truncat",
          "rounds down", "rounded down", "always 0", "always zero", "loses"),
         "`//` truncates every price below a dollar to zero"),
    Seed("key-collision", "cache_key",
         ("model", "collide", "collision", "same key", "different model",
          "distinguish", "identical"),
         "the key omits the model, so two models share one cache entry"),
    Seed("unbounded-retry", "retry_fetch",
         ("forever", "infinite", "unbounded", "no limit", "never gives up",
          "no timeout", "no backoff", "hang"),
         "there is no attempt limit, so a permanent failure loops forever"),
    Seed("shared-temp-path", "write_atomic",
         ("race", "concurrent", "collide", "same name", "unique", "two processes",
          "parallel"),
         "a fixed `.tmp` name is a shared path between concurrent writers"),
)

K = len(SEEDS)


@dataclass
class Recall:
    """One model's answer, scored against the planted list."""
    model: str
    found: set[str] = field(default_factory=set)
    reply: str = ""
    error: str = ""
    cost: float = 0.0

    @property
    def n(self) -> int:
        return K

    @property
    def rate(self) -> float:
        return len(self.found) / K if K else 0.0

    @property
    def missed(self) -> list[Seed]:
        return [s for s in SEEDS if s.id not in self.found]


def score(reply: str, seeds: tuple[Seed, ...] = SEEDS) -> set[str]:
    """Which planted defects this reply names. Pure text; no model, no network."""
    return {s.id for s in seeds if s.found_in(reply or "")}


def report(results: list[Recall]) -> str:
    """The recall table, with the misses named.

    The misses are printed rather than summarised because they are the whole
    point: "Haiku found 6 of 9" is a number, and "Haiku missed the unbounded
    retry and the shared temp path" is a decision about what to route to it.
    """
    from adder.util.stats import wilson_interval

    lines = [f"  {'model':<24}{'found':>10}{'recall':>10}{'95% CI low':>12}"
             f"{'cost':>10}", "  " + "-" * 68]
    for r in results:
        lo = wilson_interval(len(r.found), K)[0]
        lines.append(f"  {r.model:<24}{len(r.found):>4}/{K:<5}{r.rate:>9.0%}"
                     f"{lo:>12.0%}${r.cost:>9.4f}")
    lines.append("")
    for r in results:
        if r.error:
            lines.append(f"  {r.model}: {r.error[:70]}")
            continue
        if r.missed:
            lines.append(f"  {r.model} missed:")
            for s in r.missed:
                lines.append(f"    {s.symbol:<18}{s.what}")
    lines += [
        "",
        f"  {K} defects, planted. This shares no code with the cost model: it is",
        "  a fixture, a prompt and a string match, so it cannot agree with the",
        "  saving by construction the way `adder quality` and `adder verify` can.",
        "",
        "  Scope: recall on defect-finding over supplied source. That is the task",
        "  class whose failure is silent -- an incomplete answer reads exactly",
        "  like a complete one -- and it is the reason the classifier abstains on",
        "  it. It licenses nothing about multi-step or agentic work.",
    ]
    return "\n".join(lines)
