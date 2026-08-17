"""What a session restart actually costs, measured rather than assumed.

Every "just start a fresh session" recommendation in this repo was priced by one
of two models, and both were wrong:

  * `plan.replay` charged **nothing**. A split reset the context to the floor
    and the next turn carried on. That makes restarting free, and a lever that
    is free at any cadence will be pushed to any cadence by an optimiser.
  * `carry.optimal_split` charged a **full rebuild**: the whole floor rewritten
    at 1.25x or 2.00x. That makes restarting expensive, and expensive restarts
    are why the closed form kept answering "a few hundred turns".

The transcripts settle it, because every session records what its own opening
turn cost. Measured on this machine, over the 46 openings that followed a turn
inside the 5m TTL:

    opening context   27,953 tok      of which 74% arrived as a **cache read**

A session opening is not a rebuild. The expensive part of the floor -- system
prompt, tool schemas, CLAUDE.md -- is identical across sessions, so it is
already resident and is served at 0.10x. Only the session-specific tail (the
first prompt, its attachments) is written. Carrying a 2,000-token handoff, that
is $0.103 against the $0.300 the rebuild model assumed: **2.9x cheaper**. The
optimal cadence goes as `sqrt(W)`, so it moves by 1.7x on its own -- 33 turns
to 19.

Why this is the cache lever and hit rate is not
-----------------------------------------------
`adder cache` reports a 99.2% hit rate and $0 of recoverable rebuild waste: the
cache is already being *hit*. What it is not being used for is the thing it is
uniquely good at, which is **holding a prefix across sessions so that discarding
a context is cheap**. Once a restart costs a fifth of what it was thought to,
the cadence that minimises average turn cost drops from the hundreds into the
tens, and the average context a turn has to carry drops with it. That is where
the order of magnitude is: not in missing fewer caches, but in re-reading a far
smaller context because throwing one away stopped being expensive. Measured
here, holding everything else fixed, that is a **6.1x** cut to input cost per
turn against the 536-turn sessions this workload actually runs.

What is measured here and what is not
-------------------------------------
MEASURED: the split of an opening turn into uncached / read / written tokens,
and its cost. That is read straight off the transcript.

NOT MEASURED: that a mid-task restart opens as cheaply as the openings on
record. Those were mostly session starts, whose first prompt is short. A restart
in the middle of a task has to be told more, which is the `handoff_tokens` term,
and it is charged as a write at the model's rate. Its default is a prior.

One observation is deliberately *not* relied on: openings stay warm even after
gaps of days, which no TTL explains. The cause is not established -- it may be
that an identical prefix is resident from other work on the same account -- so
the numbers here are taken from openings that followed a turn within the 5m TTL
(n=46 with at least two turns). That is also the case a restart regime creates,
since restarting mid-work means the previous turn was seconds ago.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from adder.core import settings as _settings
from adder.core.filters import root_of as _root_of
from adder.core.trace import DEFAULT_ROOT, _ordered, load_sessions
from adder.pricing.cost import Rates
from adder.pricing.prices import CACHE_WRITE_MULT, TTL_SECONDS
from adder.pricing.registry import provider_for, rate

M = 1_000_000.0

# Openings are only counted as evidence when the previous turn anywhere in the
# workload, on the same model, was inside this window. That is the condition a
# restart regime satisfies by construction.
DEFAULT_WITHIN = float(TTL_SECONDS["5m"])

# Tokens a mid-task restart has to be handed to carry the thread. MODELLED.
DEFAULT_HANDOFF = 2_000

MIN_OPENINGS = 5

# Fallbacks for a machine with no transcripts. These are the *pessimistic*
# assumption -- a cold rebuild -- so that an uninstrumented caller is never
# told restarting is cheap on the strength of a prior.
PRIOR_FLOOR = 27_000


@dataclass(frozen=True)
class Opening:
    """What it costs to open a session, measured off the openings on record.

    `source` is load-bearing: a warm share fitted to 70 openings and one taken
    from a prior produce the same dataclass, and a caller about to divide a
    restart cadence by its square root needs to know which it holds.
    """

    floor_tokens: int = PRIOR_FLOOR       # median opening context
    read_tokens: int = 0                  # of it, served from cache
    write_tokens: int = PRIOR_FLOOR       # of it, written
    uncached_tokens: int = 0              # of it, paid at full rate
    openings: int = 0
    warm_openings: int = 0
    source: str = "prior"

    @property
    def measured(self) -> bool:
        return self.source == "measured"

    @property
    def warm_share(self) -> float:
        """Share of the opening context that arrived as a cache read."""
        return self.read_tokens / self.floor_tokens if self.floor_tokens else 0.0

    def cost(self, model: str, *, ttl: str = "1h", handoff_tokens: int = 0,
             on: date | None = None) -> float:
        """USD to open a session on `model`, including any handoff written in."""
        r = Rates.for_model(model, ttl=ttl, on=on)
        return (
            self.uncached_tokens * r.inp
            + self.read_tokens * r.cache_read
            + (self.write_tokens + max(0, handoff_tokens)) * r.cache_write
        ) / M

    def rebuild_cost(self, model: str, *, ttl: str = "1h", handoff_tokens: int = 0,
                     on: date | None = None) -> float:
        """What the same restart costs if the prefix is cold: everything written.

        This is the number `carry.optimal_split` used to assume, and the number
        that applies when a restart lands outside the TTL of the last turn.
        """
        r = Rates.for_model(model, ttl=ttl, on=on)
        return ((self.floor_tokens + max(0, handoff_tokens)) * r.cache_write) / M

    def discount(self, model: str, *, ttl: str = "1h",
                 handoff_tokens: int = 0, on: date | None = None) -> float:
        """How many times cheaper a warm restart is than a cold one."""
        cold = self.rebuild_cost(model, ttl=ttl, handoff_tokens=handoff_tokens, on=on)
        warm = self.cost(model, ttl=ttl, handoff_tokens=handoff_tokens, on=on)
        return cold / warm if warm else 1.0

    def describe(self) -> str:
        if not self.measured:
            return ("opening model: prior (no local transcripts); assuming a cold "
                    f"{self.floor_tokens:,}-token rebuild on every restart")
        return (f"opening model from {self.openings} session openings: "
                f"{self.floor_tokens:,} tok, {self.warm_share:.0%} served from "
                f"cache, {self.warm_openings} of them warm")

    @classmethod
    def from_session(cls, sess) -> Opening:
        """The opening this one session actually paid for.

        `measure` needs many sessions to find a median, which is the right
        input for a policy and the wrong one for a live decision: the question
        mid-session is "what would restarting *this* cost", and the answer is
        recorded on the session's own first turn. One observation, labelled as
        one observation.
        """
        if not getattr(sess, "turns", None):
            return cls()
        # The conversation's opening. A session that begins with a delegated
        # turn recorded that subagent's small, cold context as the session's
        # floor -- 4 of 105 sessions here -- and this module's whole output is
        # what re-opening the conversation costs.
        t = sess.main_turns[0]
        return cls(
            floor_tokens=max(1, t.context),
            read_tokens=t.cache_read,
            write_tokens=t.cache_write,
            uncached_tokens=t.uncached_in,
            openings=1,
            warm_openings=1 if t.cache_read > t.cache_write else 0,
            source="measured",
        )

    @classmethod
    def default(cls) -> Opening:
        return cls()


def _median(xs) -> float:
    """The shared median. `stats.median` is the one definition in this repo.

    `util.stats` opens by naming this exact pair -- "`carry` and `prefix` each
    carried a private `_median`" -- as the duplication it exists to replace,
    and both copies were still here. They agreed with it by luck rather than by
    construction: nothing stopped one of them drifting, which is how three
    different p90s from the same data happened the first time.
    """
    from adder.util.stats import median

    return median(xs)


def _preceding_gap(sessions) -> dict[str, float | None]:
    """Seconds from each session's opening turn back to the previous turn anywhere.

    Anywhere, not in the same session: the prefix a restart reads was written by
    whatever ran last on that model. Restricting the search to one session would
    call every opening cold by construction and measure nothing.
    """
    # `_ordered` on every stamp before it is sorted, bisected or subtracted. A
    # workload assembled from a Claude Code transcript and a foreign log holds
    # both offset-aware and naive datetimes, and `sort` raises on the mix --
    # from a helper three reports call, none of which has a handler.
    times: dict[str, list] = {}
    for s in sessions.values():
        for t in s.turns:
            if t.when:
                times.setdefault(t.model, []).append(_ordered(t.when))
    for v in times.values():
        v.sort()

    out: dict[str, float | None] = {}
    for s in sessions.values():
        if not s.turns:
            continue
        t = s.main_turns[0]
        if not t.when:
            out[s.id] = None
            continue
        when = _ordered(t.when)
        stamps = times.get(t.model, [])
        i = bisect.bisect_left(stamps, when)
        prev = next((stamps[j] for j in range(i - 1, -1, -1)
                     if stamps[j] < when), None)
        out[s.id] = (when - prev).total_seconds() if prev else None
    return out


def measure(sessions, *, within: float = DEFAULT_WITHIN,
            min_turns: int = 2) -> Opening:
    """Fit the opening model to recorded session openings.

    Only openings that followed a turn within `within` seconds are counted, for
    the reason in the module docstring: those are the ones whose warmth has an
    explanation, and they are the case a restart regime produces.
    """
    gaps = _preceding_gap(sessions)
    rows = []
    warm = 0
    for s in sessions.values():
        if len(s.main_turns) < min_turns:
            continue
        t = s.main_turns[0]
        if t.context <= 0:
            continue
        gap = gaps.get(s.id)
        if gap is None or gap > within:
            continue
        rows.append(t)
        if t.cache_read > t.cache_write:
            warm += 1

    if len(rows) < MIN_OPENINGS:
        return Opening.default()

    floor = int(_median([t.context for t in rows]))
    # Shares, not medians, for the split: three independently-taken medians do
    # not have to sum to the median context, and a split that does not add up
    # prices a restart against a session that never existed.
    total_ctx = sum(t.context for t in rows) or 1
    read = round(floor * sum(t.cache_read for t in rows) / total_ctx)
    unc = round(floor * sum(t.uncached_in for t in rows) / total_ctx)
    return Opening(
        floor_tokens=floor,
        read_tokens=read,
        uncached_tokens=unc,
        write_tokens=max(0, floor - read - unc),
        openings=len(rows),
        warm_openings=warm,
        source="measured",
    )


def warmth_by_gap(sessions) -> list[tuple[str, int, float]]:
    """(bucket, n, median warm share) over how long before the opening turn.

    The evidence table. It is reported rather than summarised because the long-
    gap rows are the ones this module declines to rely on, and a reader should
    be able to see them and disagree.
    """
    gaps = _preceding_gap(sessions)
    buckets: dict[str, list[float]] = {}
    order = ("<5m", "5m-1h", ">1h", "nothing before")
    for s in sessions.values():
        if not s.turns:
            continue
        t = s.main_turns[0]
        if t.context <= 0:
            continue
        g = gaps.get(s.id)
        # Bucket edges are this model's own cache lifetimes. A provider whose
        # cache lives ten minutes has its warmth decided at ten minutes, and
        # bucketing it at Anthropic's 5m/1h boundaries would sort openings into
        # groups that mean nothing on that provider.
        lives = sorted((provider_for(t.model).ttl_seconds or TTL_SECONDS).values())
        short = lives[0] if lives else TTL_SECONDS["5m"]
        long = lives[-1] if lives else TTL_SECONDS["1h"]
        if g is None:
            key = "nothing before"
        elif g < short:
            key = "<5m"
        elif g < long:
            key = "5m-1h"
        else:
            key = ">1h"
        buckets.setdefault(key, []).append(t.cache_read / t.context)
    return [(k, len(buckets[k]), _median(buckets[k])) for k in order if k in buckets]


def cadence(op: Opening, *, model: str, growth: float, read_mult: float,
            floor_tokens: int | None = None, ttl: str = "1h",
            handoff_tokens: int = DEFAULT_HANDOFF, warm: bool = True,
            observed_turns: int = 0, on: date | None = None
            ) -> tuple[int, float, float]:
    """(turns per session, average turn cost at it, average turn cost never splitting).

    The closed form is `carry.optimal_split`'s: average per-turn input cost over
    a cycle of `k` turns is `m*r*F + m*r*g*(k+1)/2 + W/k`, minimised at
    `k* = sqrt(2W/(m*r*g))`. The only thing that changes here is `W`, which is
    taken from what an opening actually cost instead of from a rebuild that does
    not happen. Because the optimum goes as `sqrt(W)`, a 3x cheaper restart is a
    1.7x shorter session, not a 3x shorter one.

    The "never splitting" leg is priced over `observed_turns`, because the
    comparison that matters is against how long sessions here actually run, not
    against a horizon chosen to flatter the result. That number should be
    cost-weighted: half the sessions on this machine are under 91 turns and
    account for a rounding error of the spend, while the ones holding the money
    run into four figures, and the average context they carry is the entire
    question.
    """
    from adder.measure.window.carry import optimal_split

    F = op.floor_tokens if floor_tokens is None else floor_tokens
    W = (op.cost(model, ttl=ttl, handoff_tokens=handoff_tokens, on=on) if warm
         else op.rebuild_cost(model, ttl=ttl, handoff_tokens=handoff_tokens, on=on))
    k, _saving, _why = optimal_split(model=model, floor_tokens=F, ttl=ttl,
                                     handoff_tokens=handoff_tokens,
                                     restart_cost=W, growth=growth,
                                     read_mult=read_mult, on=on)
    r = rate(model, on).inp
    per_tok = read_mult * r / M
    at_k = per_tok * F + per_tok * growth * (k + 1) / 2.0 + W / k
    n = observed_turns if observed_turns > 0 else k * 4
    never = per_tok * F + per_tok * growth * (n + 1) / 2.0
    return k, at_k, never


def weighted_median_turns(sessions) -> int:
    """Cost-weighted median session length: the horizon the spend actually sits at.

    A plain median counts a 5-turn session and an 1,800-turn one equally. The
    short ones are numerous and nearly free; weighting by cost puts the horizon
    where the money is.

    Length is the MAIN chain's, which is the population every consumer of this
    number is indexed against. `bench.expected_reads` feeds it straight into
    `Horizon.mean_remaining`, and `Horizon.from_sessions` counts main-chain
    turns for a measured reason it states: a subagent turn does not re-read the
    main context, so counting it asks where a session sits "using a ruler it was
    not measured with" -- 716 records for a 207-turn conversation, on this
    corpus. `memory.Pricing.turns` has the same requirement: a resident token is
    re-read once per turn of the conversation, not once per subagent step.
    """
    rows = [(len(s.main_turns), s.cost) for s in sessions.values() if s.turns]
    total = sum(c for _, c in rows)
    if not rows or total <= 0:
        return 0
    rows.sort()
    acc = 0.0
    for n, c in rows:
        acc += c
        if acc >= total / 2:
            return n
    return rows[-1][0]


def report(root: Path | str = DEFAULT_ROOT, *, model: str | None = None,
           ttl: str = "1h", handoff_tokens: int = DEFAULT_HANDOFF,
           on: date | None = None) -> str:
    from adder.measure.window.carry import Carry

    model = model or _settings.session_model()
    sessions = load_sessions(root, use_cache=True)
    op = measure(sessions)
    carry = Carry.measure(sessions)

    lines = ["  The shared prefix", "", f"  {op.describe()}", ""]
    if not op.measured:
        lines.append("  Nothing below is measured. Point this at a transcript "
                     "directory to fit it.")
        return "\n".join(lines)

    lines.append(f"  opening context   {op.floor_tokens:>10,} tok")
    _r = Rates.for_model(model, ttl=ttl, on=on)
    _rm = (_r.cache_read / _r.inp) if _r.inp else 1.0
    _wm = (_r.cache_write / _r.inp) if _r.inp else 1.0
    lines.append(f"    cache read      {op.read_tokens:>10,} tok   "
                 f"{op.warm_share:>5.0%}  @{_rm:.2f}x -- the shared floor, "
                 f"already resident")
    lines.append(f"    written         {op.write_tokens:>10,} tok   "
                 f"{op.write_tokens / op.floor_tokens:>5.0%}  @{_wm:.2f}x "
                 f"-- the part that is this session's")
    if op.uncached_tokens:
        lines.append(f"    uncached        {op.uncached_tokens:>10,} tok")

    warm = op.cost(model, ttl=ttl, handoff_tokens=handoff_tokens, on=on)
    cold = op.rebuild_cost(model, ttl=ttl, handoff_tokens=handoff_tokens, on=on)
    lines += ["", f"  One restart on {model}, carrying a "
                  f"{handoff_tokens:,}-token handoff:",
              f"    measured (prefix warm)   ${warm:>8,.4f}",
              f"    assumed  (full rebuild)  ${cold:>8,.4f}   "
              f"{op.discount(model, ttl=ttl, handoff_tokens=handoff_tokens, on=on):.1f}x "
              f"more than it costs"]

    lines += ["", "  Evidence: share of the opening context served from cache, by how",
              "  long before it anything last ran on the same model.", "",
              f"  {'gap':<18}{'openings':>9}{'from cache':>13}"]
    for bucket, n, share in warmth_by_gap(sessions):
        lines.append(f"  {bucket:<18}{n:>9,}{share:>12.0%}")
    lines.append("  Only the first row is relied on; the rest is unexplained and "
                 "would only help.")

    g = max(1.0, carry.growth)
    observed = weighted_median_turns(sessions)
    k_warm, at_warm, never = cadence(op, model=model, growth=g,
                                     read_mult=carry.read_mult, ttl=ttl,
                                     handoff_tokens=handoff_tokens, warm=True,
                                     observed_turns=observed, on=on)
    k_cold, at_cold, _ = cadence(op, model=model, growth=g,
                                 read_mult=carry.read_mult, ttl=ttl,
                                 handoff_tokens=handoff_tokens, warm=False,
                                 observed_turns=observed, on=on)
    lines += ["", "  What that does to the restart cadence "
                  "(k* = sqrt(2W/(m*r*g)), so it moves as sqrt(W)):", "",
              f"    on the assumed rebuild   {k_cold:>6,} turns   "
              f"${at_cold:>7,.4f} per turn",
              f"    on the measured opening  {k_warm:>6,} turns   "
              f"${at_warm:>7,.4f} per turn",
              f"    as run ({observed:,} turns){'':>10}${never:>7,.4f} per turn"]
    if at_warm > 0:
        lines.append(f"\n  Restarting at the measured optimum is {never / at_warm:.1f}x "
                     f"cheaper per turn than running a session to {observed:,} turns,")
        lines.append(f"  on a context growing {g:,.0f} tok/turn with a "
                     f"{carry.read_mult:.3f}x realized re-read multiplier. That is the "
                     f"input side only:")
        lines.append("  output is generated once and is untouched by any of this.")
    lines += ["",
              "  MODELLED: the handoff. A mid-task restart has to be told what the",
              f"  previous one knew, and {handoff_tokens:,} tokens is a prior, not a "
              "measurement.",
              "  Vary it with --handoff; the cadence moves as its square root too."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="adder prefix",
        description="Measure what a session restart actually costs.")
    ap.add_argument("root", nargs="?", default=None,
                    help="transcript directory (default: the `root` setting)")
    ap.add_argument("--model", default=_settings.session_model())
    ap.add_argument("--ttl", default="1h", choices=sorted(CACHE_WRITE_MULT))
    ap.add_argument("--handoff", type=int, default=DEFAULT_HANDOFF, metavar="TOK",
                    help=f"tokens handed to a restarted session "
                         f"(default: {DEFAULT_HANDOFF})")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    a = ap.parse_args(argv)
    # `root_of`: the argument if one was given, else the `root`
    # setting. Resolved here so two commands cannot disagree
    # about which transcript directory `adder config` names.
    a.root = str(_root_of(a))

    if a.json:
        import json

        from adder.core.trace import load_sessions

        sessions = load_sessions(a.root)
        op = measure(sessions)
        print(json.dumps({
            "measured": op.measured,
            "source": op.source,
            "floor_tokens": op.floor_tokens,
            "read_tokens": op.read_tokens,
            "write_tokens": op.write_tokens,
            "uncached_tokens": op.uncached_tokens,
            "warm_share": round(op.warm_share, 5),
            "openings": op.openings,
            "warm_openings": op.warm_openings,
            "model": a.model,
            "ttl": a.ttl,
            "handoff_tokens": a.handoff,
            "restart_cost": round(op.cost(a.model, ttl=a.ttl,
                                          handoff_tokens=a.handoff), 4),
            "cold_rebuild_cost": round(op.rebuild_cost(a.model, ttl=a.ttl,
                                                       handoff_tokens=a.handoff), 4),
            "discount": round(op.discount(a.model, ttl=a.ttl), 4),
            "warmth_by_gap": [
                {"bucket": label, "sessions": n, "warm_share": round(share, 4)}
                for label, n, share in warmth_by_gap(sessions)
            ],
        }))
        return 0

    print()
    print(report(a.root, model=a.model, ttl=a.ttl, handoff_tokens=a.handoff))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
