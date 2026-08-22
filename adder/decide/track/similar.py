"""What happened last time on a task like this one.

The gap this closes
-------------------
`outcomes.evidence` answers "how often does T0 fail *here*", scoped by project.
That is the right question one level too coarse. A project's task mix is not
homogeneous: the same repository gets "where is the retry logic" and "make the
scheduler preemptible", and a single failure rate for T0 across both is an
average over two populations with nothing in common. The gate that consumes it
is then either too timid for the lookups or too bold for the refactors, and
which one it is depends on the mix that week.

The routing literature settled this argument in the other direction. The
methods that hold up on a controlled benchmark are not the trained ones -- the
matrix-factorization and graph routers are matched, and sometimes beaten, by
clustering the queries and scoring each model *per cluster*, with no network to
train. What that says is that most of the recoverable signal is in "which kind
of task is this", and that kind is recoverable from the text without a model.

This module is that, with the one substitution the constraints here force and
the data here permit. There is no embedding model available -- no dependency, no
network, no model call -- so similarity is over the task's *vocabulary* rather
than its semantics: a MinHash sketch of terms, Jaccard between sketches. And
rather than cluster into k buckets and score each, it takes the k nearest
neighbours of the task in hand, because the log is small enough that the extra
resolution is free and a nearest-neighbour estimate needs no cluster count
chosen in advance.

What it deliberately does not do
--------------------------------
**It does not predict quality.** It predicts the thing already measured: how
often a rung, on tasks whose vocabulary resembles this one, escalated. Every
other number in this repo is a re-priced observation, and a router that starts
guessing at answer quality would be the first one that is not.

**It does not store the task.** `Outcome` has always carried `task_hash` and
never the text, which is deliberate -- the log lives in the user's home
directory and a task description contains their code and their prompts. A
MinHash sketch keeps that property: each of its slots is a minimum over the
whole term set, so the terms are not recoverable from it, and it is the same
fixed size whether the task was six words or six hundred.

**It is not allowed to buy a downgrade on thin data.** The asymmetry the rest of
the router runs on applies here too, and harder, because this estimator is
sharper and therefore easier to fool. See `sharpen`.

Why Jaccard over vocabulary is enough, and where it is not
----------------------------------------------------------
Task descriptions are short and written by a model to a fixed prompt, so they
are unusually lexically consistent: "find where X is configured" and "locate the
config for Y" share `find/locate`, `config`, and the shape. That is the regime
vocabulary overlap works in.

Where it fails is paraphrase with no shared words, and it fails *silently* -- the
neighbours simply are not found, the estimator returns nothing, and the caller
falls back to the tier-wide rate it used before. That is the correct failure: a
similarity measure that cannot tell it has missed must be built so that missing
costs nothing.
"""

from __future__ import annotations

import re
from dataclasses import replace
from hashlib import blake2b
from itertools import pairwise
from pathlib import Path

from adder.decide.track.outcomes import (
    PRIOR_FAIL,
    PRIOR_OK,
    Evidence,
    Outcome,
    load,
    recency_weight,
)

# Slots in the sketch. Each is an independent hash function, so the variance of
# the Jaccard estimate is J(1-J)/SLOTS -- about +/-0.12 at J=0.5. That is coarse
# for ranking two near-identical candidates and plenty for the only question
# asked of it, which is whether a row is on topic at all.
#
# 16 slots of 4 bytes is exactly one blake2b digest, so a term costs one hash
# call rather than sixteen. That is why the number is 16 and not 24.
SLOTS = 16
_DIGEST = SLOTS * 4

# Below this, a row is not a neighbour. Two task descriptions that share a
# fifth of their vocabulary are usually the same kind of work; a tenth is two
# sentences that both said "the".
SIM_FLOOR = 0.2

# How many neighbours to weigh, and the fewest that may form an estimate at
# all. Four is not much, which is why four only ever earns the *pessimistic*
# reading -- `sharpen` decides what a thin estimate is allowed to do.
NEIGHBOURS = 24
MIN_NEIGHBOURS = 4

# Terms below this length carry no topic. Kept as a length rule rather than a
# stopword list because a stopword list is a language commitment, and task
# descriptions in this log are not all English.
MIN_TERM = 3

_WORD = re.compile(r"[a-z0-9_]+")

# Vocabulary that is in every task description because the harness asks for it,
# and therefore separates nothing. Left small on purpose: each entry is a claim
# that a word is noise everywhere, and that claim gets weaker the longer the
# list gets.
_NOISE = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "into", "then",
    "task", "please", "should", "make", "sure", "code", "file", "files",
})


def terms(text: str) -> list[str]:
    """Topic-bearing terms, plus adjacent bigrams.

    Bigrams are included because the unigrams alone conflate the two halves of
    this log: "read the config" and "write the config" share every word that
    survives filtering, and they are opposite tiers. `read_config` against
    `write_config` does not.
    """
    words = [w for w in _WORD.findall((text or "").lower())
             if len(w) >= MIN_TERM and w not in _NOISE]
    out = list(words)
    out += [f"{a}_{b}" for a, b in pairwise(words)]
    return out


def sketch(text: str) -> tuple[int, ...]:
    """A fixed-size, irreversible MinHash sketch of the task's vocabulary.

    Empty when there is nothing to hash, which every caller reads as "no
    opinion" rather than as a sketch that matches nothing.
    """
    ts = terms(text)
    if not ts:
        return ()
    best = [0xFFFFFFFF] * SLOTS
    for t in ts:
        d = blake2b(t.encode("utf-8"), digest_size=_DIGEST).digest()
        for i in range(SLOTS):
            h = int.from_bytes(d[4 * i:4 * i + 4], "big")
            if h < best[i]:
                best[i] = h
    return tuple(best)


def similarity(a, b) -> float:
    """Estimated Jaccard overlap of two term sets, from their sketches.

    Unbiased at any set size: slot `i` of a sketch is the minimum of one hash
    over the whole term set, and two sets share that minimum with probability
    exactly their Jaccard index. This is why the sketch is k independent minima
    rather than the k smallest values of one hash -- the latter is cheaper and
    biased *upward* on short texts, which is the entire population here, and an
    inflated similarity is how an unrelated row becomes a neighbour.

    Sketches of different lengths are compared over their common prefix. Slots
    are independent, so a prefix is a smaller sketch rather than a broken one,
    and a log written by an older version stays readable.
    """
    n = min(len(a or ()), len(b or ()))
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if a[i] == b[i]) / n


def neighbours(
    task: str,
    rows: list[Outcome],
    *,
    tier: str | None = None,
    floor: float = SIM_FLOOR,
    k: int = NEIGHBOURS,
) -> list[tuple[float, Outcome]]:
    """The `k` most vocabulary-similar recorded outcomes, most similar first.

    Not scoped by project, and that is the point rather than an oversight: the
    task is the scope. A refactor in another repository of yours is better
    evidence about how a rung handles refactors than a lookup in this one, and
    the whole reason this module exists is that the project scope was mixing
    those two together.
    """
    sk = sketch(task)
    if not sk:
        return []
    scored = []
    for o in rows:
        if tier is not None and o.tier != tier:
            continue
        if not o.sketch:
            continue
        s = similarity(sk, o.sketch)
        if s >= floor:
            scored.append((s, o))
    # Sorted by similarity, then by recency, so the order is total and the
    # report does not shuffle between runs on ties.
    scored.sort(key=lambda p: (-p[0], -p[1].ts))
    return scored[:k]


def evidence_like(
    task: str,
    tier: str,
    rows: list[Outcome] | None = None,
    *,
    log: Path | str | None = None,
    now: float | None = None,
    floor: float = SIM_FLOOR,
    k: int = NEIGHBOURS,
) -> Evidence | None:
    """Escalation rate for a tier over tasks like this one, or None.

    None means "no opinion" and never "zero": too few neighbours, no sketches in
    the log yet, or nothing to hash. Callers keep whatever they had.

    The weight is the same recency decay the tier-wide estimator uses, times the
    similarity, so a row that is both old and only loosely on topic contributes
    almost nothing without having to be excluded by a second threshold. The same
    Beta(1,1) prior is applied to the same weighted mass, which is what makes the
    two estimates comparable at all -- `sharpen` puts them side by side, and it
    could not if one of them were smoothed differently.
    """
    import time

    rs = rows if rows is not None else load(log)
    near = neighbours(task, rs, tier=tier, floor=floor, k=k)
    if len(near) < MIN_NEIGHBOURS:
        return None

    at = time.time() if now is None else now
    mass = 0.0
    fails = 0.0
    for sim, o in near:
        w = recency_weight(o, at) * sim
        mass += w
        if o.escalated:
            fails += w
    if mass <= 0.0:
        return None
    p = (fails + PRIOR_FAIL) / (mass + PRIOR_FAIL + PRIOR_OK)
    return Evidence(p, len(near), mass, "neighbours", fails)


def sharpen(tier_ev: Evidence | None, nb: Evidence | None) -> Evidence | None:
    """Combine the tier-wide rate with the neighbour rate, asymmetrically.

    The rest of the router already refuses to let a cheap rung be chosen without
    evidence, on the grounds that the two errors do not cost the same: routing
    up wastes the rate difference, routing down wastes a whole failed run plus
    the turn that noticed. That argument does not weaken because the estimator
    got sharper -- it gets stronger, because a nearest-neighbour rate over four
    rows is much easier to push around than a tier-wide rate over four hundred,
    and it is the direction that costs money that is easy to push.

    So there are three cases and only one of them is symmetric:

    * **The neighbour estimate carries real mass** (`informative`): it replaces
      the tier-wide one outright, in either direction. It is a strictly better
      conditioned measurement of the same event and it has the evidence to say so.
    * **It is thin and pessimistic**: the rate is taken, the tier-wide mass is
      kept. Acting on a raised `p_fail` can only decline a downgrade, and
      declining a downgrade is free -- the worst case is paying for the model
      that would otherwise have been chosen anyway. `fails` is moved with the
      rate so the credible interval still describes the mean it is printed next
      to, and the resulting interval is narrower than the neighbour data alone
      supports. That is deliberate and it is the same trade `Evidence.upper`
      makes: when the two errors are priced differently, so is the estimate.
    * **It is thin and optimistic**: discarded. This is the case that would buy
      a downgrade on four rows, and it is exactly the case the asymmetry exists
      to refuse.
    """
    if nb is None:
        return tier_ev
    if tier_ev is None or tier_ev.scope == "prior":
        # Nothing to sharpen. The neighbour estimate stands on its own terms:
        # informative if it has the mass, and left non-informative if not, which
        # keeps it out of a downgrade without discarding what it measured.
        return nb
    if nb.informative:
        return nb
    if nb.p_fail > tier_ev.p_fail:
        return replace(tier_ev, p_fail=nb.p_fail,
                       fails=nb.p_fail * tier_ev.weight,
                       n=nb.n, scope=f"{tier_ev.scope}+neighbours")
    return tier_ev


def coverage(rows: list[Outcome] | None = None, log: Path | str | None = None) -> tuple[int, int]:
    """`(rows with a sketch, rows total)`.

    Worth reporting rather than assuming. Sketches began being written after the
    log did, and rows predating them are invisible to every function above --
    silently, since a missing sketch is skipped rather than an error. A user
    whose log is all legacy rows should be told that this half of the router is
    asleep, not left to conclude that their tasks have no neighbours.
    """
    rs = rows if rows is not None else load(log)
    return sum(1 for o in rs if o.sketch), len(rs)


def report(task: str, *, log: Path | str | None = None, now: float | None = None,
           floor: float = SIM_FLOOR, k: int = NEIGHBOURS) -> str:
    from adder.util.render import money

    rows = load(log)
    have, total = coverage(rows)
    out = [f'  Tasks like: "{task[:70]}"', ""]
    if not total:
        out.append("  The outcome log is empty. `adder outcomes import --write`")
        out.append("  backfills it from transcripts already on disk.")
        return "\n".join(out) + "\n"
    out.append(f"  {have:,} of {total:,} logged outcomes carry a vocabulary sketch "
               f"({have / total:.0%})")
    if not have:
        out.append("  Rows written before sketches existed cannot be compared. New")
        out.append("  outcomes carry one; re-run `adder outcomes import --write` to")
        out.append("  pick up delegations this log has not seen yet.")
        return "\n".join(out) + "\n"
    out.append("")

    from adder.decide.track.outcomes import evidence as tier_evidence

    any_tier = False
    for tier in ("T0", "T1", "T2", "T3"):
        nb = evidence_like(task, tier, rows, now=now, floor=floor, k=k)
        if nb is None:
            continue
        any_tier = True
        wide = tier_evidence(tier, None, outcomes=rows, now=now)
        used = sharpen(wide, nb)
        out.append(f"  {tier}: {nb.p_fail:.0%} escalated over {nb.n} similar runs "
                   f"(mass {nb.weight:.1f}) against {wide.p_fail:.0%} tier-wide")
        if used is not None and used.scope != nb.scope:
            out.append(f"       the gate uses {used.p_fail:.0%} ({used.scope})")
    if not any_tier:
        out.append(f"  No tier has {MIN_NEIGHBOURS} runs above {floor:.0%} similarity. "
                   f"The router")
        out.append("  falls back to the tier-wide rate, which is what it used before.")
        return "\n".join(out) + "\n"

    out.append("")
    near = neighbours(task, rows, floor=floor, k=8)
    if near:
        out.append("  Nearest recorded runs")
        for sim, o in near:
            flag = "escalated" if o.escalated else "held"
            out.append(f"    {sim:.0%}  {o.tier:3}  {flag:9}  "
                       f"{money(o.cost)}  {o.project[:28]}")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(
        prog="adder similar",
        description="what happened on tasks whose vocabulary resembles this one")
    ap.add_argument("task", nargs="*", help="task text (or stdin)")
    ap.add_argument("--log", default=None, help="outcome log (default: the `log` setting)")
    ap.add_argument("--floor", type=float, default=SIM_FLOOR,
                    help=f"minimum similarity to count as a neighbour (default {SIM_FLOOR})")
    ap.add_argument("--top", type=int, default=NEIGHBOURS, metavar="K",
                    help=f"neighbours to weigh (default {NEIGHBOURS})")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    a = ap.parse_args(argv)

    task = " ".join(a.task) if a.task else sys.stdin.read()
    if not task.strip():
        print("adder similar: no task text given", file=sys.stderr)
        return 2

    if a.json:
        rows = load(a.log)
        have, total = coverage(rows)
        tiers = {}
        for tier in ("T0", "T1", "T2", "T3"):
            nb = evidence_like(task, tier, rows, floor=a.floor, k=a.top)
            if nb is not None:
                tiers[tier] = {"p_fail": nb.p_fail, "n": nb.n,
                               "weight": round(nb.weight, 3)}
        print(json.dumps({
            "task_chars": len(task),
            "sketched_rows": have, "total_rows": total,
            "floor": a.floor, "neighbours": a.top,
            "by_tier": tiers,
        }))
        return 0

    print()
    print(report(task, log=a.log, floor=a.floor, k=a.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
