"""The fast path bills at 2x. Did the speed arrive?

The claim nobody checked
------------------------
`prices.py` has carried a 2x multiplier for the fast serving path since it was
written. Every report in this package prices it correctly and not one of them
has ever asked the obvious question: **what did the 2x buy?**

There is a good reason to be suspicious of the answer, and it is not specific to
this provider. The systems literature has repeatedly found that headline
inference speedups are measured in the configuration that flatters them --
batch size one, a research prototype, a short prompt -- and shrink toward
nothing under production conditions, because the system becomes compute-bound
and the verification step dominates. A 3x claim measured at batch size 1 is not
a lie, it is a measurement of a setting nobody runs in.

The same scepticism applies here, and the same remedy: measure it where you
actually run it.

What can honestly be measured from a transcript
------------------------------------------------
Transcripts do not record generation time. What they record is a timestamp per
turn, so the only available clock is the gap between one assistant turn and the
next. That gap contains tool execution, and it contains the human reading the
answer and typing a reply.

So the throughput computed here is **not** tokens per second of generation. It
is tokens per second of wall clock, and it understates the true figure by
whatever else happened in the gap. That is stated wherever it is printed.

It is still usable for one purpose, which is the purpose here: a **paired**
comparison. Compare fast turns against standard turns of the same model, in
similar contexts, and the tool-and-human overhead is a roughly common term that
partly cancels. What survives is a ratio, and the ratio is what the 2x premium
has to justify. An absolute throughput number from this data would be wrong, and
this module does not print one as if it were a measurement.

The likeliest finding
---------------------
That you have never used the fast path at all. On the machine this was written
against, 30,718 of 30,718 turns were standard. When that is the case the honest
report is not an empty table: it is the prospective calculation. Given the 2x
premium and your own measured spend, the fast path would cost this much extra
per session, and it is worth it only if a second of your time is worth more
than that. That is a decision, and it needs no fast turns to make.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from itertools import pairwise

from adder.util import render
from adder.util.stats import DEFAULT_SEED, mean, median, share


def _fast_multiplier() -> float:
    """The premium the fast path bills, read from the price table.

    Read rather than restated, so a change to the rates cannot leave this
    report auditing against a number the rest of the tool no longer uses.
    """
    from adder.pricing import prices

    return float(getattr(prices, "FAST_MULT", 2.0))


FAST_MULTIPLIER = _fast_multiplier()

# Gaps longer than this are not a model being slow, they are a person going to
# lunch. Including them turns the throughput estimate into a measure of working
# habits.
MAX_GAP_S = 120.0

# Below this many paired turns the ratio is noise and is reported as such.
MIN_PAIRED = 20


@dataclass(frozen=True)
class Sample:
    """One turn's observed wall-clock throughput."""

    model: str
    speed: str
    out_tokens: int
    seconds: float
    cost: float

    @property
    def tokens_per_second(self) -> float:
        return self.out_tokens / self.seconds if self.seconds > 0 else 0.0


def samples(sessions: dict, *, max_gap_s: float = MAX_GAP_S) -> list[Sample]:
    """Wall-clock throughput per turn, from consecutive timestamps.

    Turns with no usable gap are dropped rather than guessed at, and gaps beyond
    `max_gap_s` are dropped because they measure the human, not the model.
    """
    out: list[Sample] = []
    for s in sessions.values():
        turns = [t for t in s.turns if t.when is not None]
        for prev, cur in pairwise(turns):
            gap = (cur.when - prev.when).total_seconds()
            if not 0.0 < gap <= max_gap_s or cur.out <= 0:
                continue
            out.append(Sample(model=cur.model, speed=cur.speed,
                              out_tokens=cur.out, seconds=gap, cost=cur.cost()))
    return out


@dataclass
class Comparison:
    """Fast against standard, for one model."""

    model: str
    fast_n: int = 0
    standard_n: int = 0
    fast_tps: float = 0.0
    standard_tps: float = 0.0
    ratio_ci: tuple[float, float] = (0.0, 0.0)

    @property
    def ratio(self) -> float:
        return self.fast_tps / self.standard_tps if self.standard_tps > 0 else 0.0

    @property
    def paired(self) -> bool:
        return min(self.fast_n, self.standard_n) >= MIN_PAIRED

    @property
    def faster(self) -> bool:
        """True only when the interval clears parity."""
        return self.paired and self.ratio_ci[0] > 1.0

    @property
    def clears_the_premium(self) -> bool:
        """True only when the measured speedup clears what the premium costs."""
        return self.paired and self.ratio_ci[0] > FAST_MULTIPLIER


@dataclass
class Report:
    comparisons: list[Comparison] = field(default_factory=list)
    fast_turns: int = 0
    total_turns: int = 0
    spend: float = 0.0
    sessions: int = 0
    median_session_cost: float = 0.0
    # The output half of a median session, which is the only part the fast
    # premium applies to.
    median_session_output_cost: float = 0.0

    @property
    def fast_share(self) -> float:
        return share(self.fast_turns, self.total_turns)

    @property
    def ever_used(self) -> bool:
        return self.fast_turns > 0

    @property
    def premium_per_session(self) -> float:
        """What running everything on the fast path would add to a session.

        The output half only. Cached input is billed at the same rate either
        way, so doubling the whole session bill would overstate it -- and
        overstating the cost of a lever is the same failure as understating it.
        """
        return self.median_session_output_cost * (FAST_MULTIPLIER - 1.0)

    def to_json(self) -> dict:
        return {
            "fast_turns": self.fast_turns,
            "total_turns": self.total_turns,
            "fast_share": self.fast_share,
            "ever_used": self.ever_used,
            "multiplier": FAST_MULTIPLIER,
            "median_session_cost_usd": self.median_session_cost,
            "median_session_output_cost_usd": self.median_session_output_cost,
            "premium_per_session_usd": self.premium_per_session,
            "wall_clock_only": True,
            "models": [
                {"model": c.model, "fast_turns": c.fast_n,
                 "standard_turns": c.standard_n,
                 "fast_tokens_per_second": c.fast_tps,
                 "standard_tokens_per_second": c.standard_tps,
                 "ratio": c.ratio, "ratio_ci95": list(c.ratio_ci),
                 "paired": c.paired, "faster": c.faster,
                 "clears_the_premium": c.clears_the_premium}
                for c in self.comparisons
            ],
        }


def compare(rows: list[Sample], *, seed: int = DEFAULT_SEED) -> list[Comparison]:
    """Fast against standard, per model.

    Per model rather than pooled, because a workload that runs one model fast
    and another standard would otherwise produce a "speedup" that is entirely a
    difference between the two models.
    """
    by_model: dict[str, list[Sample]] = {}
    for r in rows:
        by_model.setdefault(r.model, []).append(r)

    out: list[Comparison] = []
    for model, group in sorted(by_model.items()):
        fast = [r.tokens_per_second for r in group if r.speed == "fast"]
        std = [r.tokens_per_second for r in group if r.speed != "fast"]
        c = Comparison(model=model, fast_n=len(fast), standard_n=len(std),
                       fast_tps=mean(fast), standard_tps=mean(std))
        if c.paired:
            # Interval on the ratio, by resampling each arm and dividing. The
            # arms are independent samples of turns, so they resample apart.
            import random

            rng = random.Random(seed)
            draws = []
            for _ in range(200):
                a = mean([fast[rng.randrange(len(fast))] for _ in range(len(fast))])
                b = mean([std[rng.randrange(len(std))] for _ in range(len(std))])
                draws.append(a / b if b > 0 else 0.0)
            draws.sort()
            c = Comparison(model=model, fast_n=len(fast), standard_n=len(std),
                           fast_tps=c.fast_tps, standard_tps=c.standard_tps,
                           ratio_ci=(draws[int(0.025 * len(draws))],
                                     draws[min(len(draws) - 1, int(0.975 * len(draws)))]))
        out.append(c)
    return out


def analyse(sessions: dict, *, max_gap_s: float = MAX_GAP_S,
            seed: int = DEFAULT_SEED) -> Report:
    rows = samples(sessions, max_gap_s=max_gap_s)
    live = [s for s in sessions.values() if s.turns]
    costs = [s.cost for s in live]
    # The premium applies to output only, so the output half is measured
    # separately rather than approximated by the whole bill -- which would have
    # overstated the cost of the lever by roughly the size of the input term,
    # and the input term is most of the bill here.
    out_costs = [sum(t.output_cost() for t in s.turns) for s in live]
    return Report(
        comparisons=compare(rows, seed=seed),
        fast_turns=sum(1 for r in rows if r.speed == "fast"),
        total_turns=len(rows),
        spend=sum(costs),
        sessions=len(sessions),
        median_session_cost=median(costs),
        median_session_output_cost=median(out_costs),
    )


def format_report(rep: Report) -> str:
    out: list[str] = []
    out += render.heading("fast path — was the premium worth it?", rule="=")
    if not rep.total_turns:
        out.append("  No turn has a usable gap to the next one, so nothing can "
                   "be timed.")
        return "\n".join(out)

    out.append(render.kv("turns timed", f"{rep.total_turns:,}"))
    out.append(render.kv("on the fast path",
                         f"{rep.fast_turns:,} ({render.pct(rep.fast_share)})"))
    out.append(render.kv("fast path bills at", f"{FAST_MULTIPLIER:g}x output"))

    if not rep.ever_used:
        out.append("")
        out += render.wrap(
            f"You have never used it. All {rep.total_turns:,} timed turns ran on "
            "the standard path, so there is nothing to audit and no speedup to "
            "credit or dispute.")
        out.append("")
        out += render.heading("what it would cost")
        out.append(render.kv("median session", render.money(rep.median_session_cost)))
        out.append(render.kv("of which output",
                             render.money(rep.median_session_output_cost)))
        out.append(render.kv("premium, always fast",
                             render.money(rep.premium_per_session) + " per session"))
        out += render.wrap(
            "That is the output half of the bill only; cached input costs the "
            "same either way. Whether it is worth paying depends on what a "
            "minute of waiting is worth to you, which this tool cannot know — "
            "but it is a number, and it is the one to decide against.")
        return "\n".join(out)

    out.append("")
    out += render.table(
        [[c.model[:26], f"{c.fast_n:,}", f"{c.standard_n:,}",
          f"{c.fast_tps:,.0f}", f"{c.standard_tps:,.0f}",
          f"{c.ratio:.2f}x" if c.standard_tps else "—",
          f"[{c.ratio_ci[0]:.2f}, {c.ratio_ci[1]:.2f}]" if c.paired else "thin"]
         for c in rep.comparisons],
        ["model", "fast", "std", "fast tok/s", "std tok/s", "ratio", "95% CI"],
        align="<>>>>><",
    )

    measured = [c for c in rep.comparisons if c.paired]
    out.append("")
    if not measured:
        out += render.wrap(
            f"No model has {MIN_PAIRED} turns on both paths, so no comparison "
            "here is worth reading. Run a session on each and come back.")
    else:
        clears = [c for c in measured if c.clears_the_premium]
        faster = [c for c in measured if c.faster]
        if clears:
            names = ", ".join(c.model for c in clears)
            out += render.wrap(
                f"The measured speedup clears the {FAST_MULTIPLIER:g}x premium for: "
                f"{names}. On those, the fast path is buying more than it costs.")
        elif faster:
            names = ", ".join(c.model for c in faster)
            out += render.wrap(
                f"Measurably faster but not by {FAST_MULTIPLIER:g}x: {names}. You are "
                "paying double for less than double, which may still be the right "
                "trade if the latency matters, but it is not a saving.")
        else:
            out += render.wrap(
                f"No model shows a speedup whose interval clears parity. On this "
                f"data the {FAST_MULTIPLIER:g}x premium bought nothing measurable.")

    out.append("")
    out += render.wrap(
        "WALL CLOCK, NOT GENERATION. Transcripts record one timestamp per turn, "
        "so the only clock available includes tool execution and the time you "
        "spent reading. Absolute throughput here is therefore understated for "
        "both paths; only the paired ratio is meaningful, and only because the "
        "overhead is a roughly common term.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    from adder.core import filters

    ap = argparse.ArgumentParser(
        prog="adder speed",
        description="Audit the fast serving path against the premium it bills.",
    )
    ap.add_argument("--max-gap", type=float, default=MAX_GAP_S,
                    help="ignore gaps longer than this many seconds (default 120)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--json", action="store_true")
    filters.add_arguments(ap)
    args = ap.parse_args(argv)

    sessions, _w = filters.load(args, use_cache=True)
    if not sessions:
        print(json.dumps({"total_turns": 0}, indent=2) if args.json
              else "  No sessions found.")
        return 1

    rep = analyse(sessions, max_gap_s=args.max_gap, seed=args.seed)
    if args.json:
        print(json.dumps(rep.to_json(), indent=2, sort_keys=True))
    else:
        print(format_report(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
