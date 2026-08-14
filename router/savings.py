"""What each lever is worth, unified around one root cause.

The causal model
----------------
Measured: assistant output is ~105% of context growth, and context re-reads are
~78% of spend. So there is one root cause, not four:

    output tokens accumulate in the main context, and are re-read every turn

That gives exactly three ways to reduce the bill, all attacking the same pool:

    1. TERSENESS   - generate fewer tokens
    2. DELEGATION  - generate them in a context that gets thrown away
    3. SPLITTING   - reset the accumulation

They are therefore **substitutes, not complements**. Reporting their sum would
double-count; the joint ceiling is the pool itself.

Model routing is a fourth, separate lever that touches only generation price. It
is included because it is what people ask for, and it is small.

Confidence tiers:
  MEASURED   - recomputed from recorded tokens; no assumptions.
  ATTRIBUTED - exact arithmetic assigning recorded spend to a cause.
  MODELLED   - depends on a stated counterfactual.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .cost import turn_cost
from .debt import decompose_read_cost, verbosity_saving
from .prices import CACHE_READ_MULT, rate
from .trace import DEFAULT_ROOT, Session, load_sessions

M = 1_000_000.0


@dataclass
class Estimate:
    lever: str
    saving: float
    confidence: str
    basis: str
    assumptions: str = ""
    pool_fraction: float = 0.0   # share of the accumulated-output pool it removes
    generation_saving: float = 0.0

    def line(self, total: float) -> str:
        pct = f"{100 * self.saving / total:.1f}%" if total else "-"
        return f"${self.saving:>9,.0f}  {pct:>6}  [{self.confidence:<10}] {self.lever}"


def terseness(sessions, *, reduction: float = 0.30, on: date | None = None) -> Estimate:
    """ATTRIBUTED: cut output by `reduction`, bounded by measured spend."""
    v = verbosity_saving(sessions, reduction=reduction, on=on)
    return Estimate(
        f"Write {reduction:.0%} less (leverage {v.leverage:.1f}x downstream)",
        v.total,
        "ATTRIBUTED",
        "reduction x measured accumulated read cost, plus generation saving",
        "assumes output can be cut proportionally without losing task quality",
        pool_fraction=reduction,
        generation_saving=v.generation_saved,
    )


def delegation(sessions, *, delegable_turns: float = 0.25,
               summary_ratio: float = 0.10, sub_model: str = "claude-haiku-4-5",
               on: date | None = None) -> Estimate:
    """MODELLED: run some turns in a throwaway context instead of the main one.

    NOT modelled as avoiding large file reads -- measured growth is diffuse
    (median 960 tok/turn; reads >=20K are only 7% of growth), so there are few
    big reads to move. The real mechanism is that a subagent's *own output*
    never enters the main context; only its summary does.
    """
    _, _, accumulated = decompose_read_cost(sessions, on)
    kept = accumulated * delegable_turns * (1 - summary_ratio)

    gen = 0.0
    for s in sessions.values():
        for t in s.turns:
            r_main, r_sub = rate(t.model, on), rate(sub_model, on)
            if r_sub.out < r_main.out:
                gen += t.out * delegable_turns * (r_main.out - r_sub.out) / M
    return Estimate(
        f"Delegate {delegable_turns:.0%} of turns to subagents "
        f"({summary_ratio:.0%} summary returned)",
        kept + gen,
        "MODELLED",
        "measured accumulated read cost, re-priced with delegated output excluded",
        f"assumes {delegable_turns:.0%} of turns are delegable and compress to "
        f"{summary_ratio:.0%}; both are estimates",
        pool_fraction=delegable_turns * (1 - summary_ratio),
        generation_saving=gen,
    )


def splitting(sessions, *, max_turns: int = 300, on: date | None = None) -> Estimate:
    """MODELLED: cap session length so accumulation resets.

    Uses each session's own measured baseline as the restart floor -- justified
    because baseline context is only ~5-8% of median context (measured).
    """
    saving = 0.0
    for s in sessions.values():
        if s.n_turns <= max_turns or not s.turns:
            continue
        r = rate(s.turns[0].model, on).inp * CACHE_READ_MULT
        actual = sum(t.context for t in s.turns) * r / M
        floor = min(t.context for t in s.turns)
        slope = max(0.0, (s.peak_context - floor) / max(1, s.n_turns))
        simulated = sum(
            (floor + slope * (i % max_turns)) * r / M for i in range(s.n_turns)
        )
        saving += max(0.0, actual - simulated)
    _, _, acc = decompose_read_cost(sessions, on)
    return Estimate(
        f"Split sessions longer than {max_turns} turns",
        saving,
        "MODELLED",
        "re-simulates context growth with periodic resets to measured baseline",
        "assumes work is separable at turn boundaries; real handoffs cost more",
        pool_fraction=min(1.0, saving / acc) if acc else 0.0,
    )


def model_routing(sessions, on: date | None = None) -> Estimate:
    """MODELLED: per-turn downgrade where the cache gate permits. The original ask."""
    from .cost import switch_is_profitable

    saving = 0.0
    eligible = 0
    for s in sessions.values():
        for t in s.turns:
            if t.model.startswith("claude-haiku"):
                continue
            d = switch_is_profitable("claude-opus-5", "claude-haiku-4-5",
                                     t.context, t.out, on=on)
            if d:
                saving += d.saving
                eligible += 1
    return Estimate(
        f"Per-turn model downgrade where the cache gate permits ({eligible:,} turns)",
        saving,
        "MODELLED",
        "cache-gated switch applied per recorded turn",
        "assumes a cheaper model would have sufficed on those turns "
        "(not verifiable from transcripts)",
    )


def explore_on_haiku(sessions, cheap: str = "claude-haiku-4-5",
                     on: date | None = None) -> Estimate:
    """MEASURED: rerun existing subagent turns at a cheap model's rates."""
    actual = saved = 0.0
    n = 0
    for s in sessions.values():
        for t in s.turns:
            if not t.sidechain or rate(t.model, on).inp <= rate(cheap, on).inp:
                continue
            actual += t.cost(on)
            saved += turn_cost(cheap, uncached_in=t.uncached_in, cache_read=t.cache_read,
                               cache_write=t.cache_write, out=t.out, on=on)
            n += 1
    return Estimate(
        f"Run subagents/Explore on Haiku ({n:,} existing subagent turns)",
        actual - saved,
        "MEASURED",
        f"recomputed {n:,} recorded subagent turns at Haiku rates",
        "",
    )


def combine(pool: float, levers: list[Estimate], separate: list[Estimate]) -> tuple[float, float]:
    """Combine substitute levers multiplicatively on the residual pool.

    They are substitutes, so their savings do not add: applying terseness leaves
    less pool for splitting to remove. Residual after all of them is the product
    of each one's residual, which is both more accurate than max() (too
    conservative) and than the sum (double-counts).
    """
    residual = 1.0
    for e in levers:
        residual *= max(0.0, 1.0 - min(1.0, e.pool_fraction))
    pool_saving = pool * (1.0 - residual)
    gen = sum(e.generation_saving for e in levers) + sum(e.saving for e in separate)
    return pool_saving, gen


def report(root: Path | str = DEFAULT_ROOT, *, max_turns: int = 300,
           on: date | None = None) -> None:
    sessions = load_sessions(root)
    total = sum(s.cost(on) if callable(getattr(s, "cost", None)) else s.cost
                for s in sessions.values())
    if not total:
        print(f"No priced turns found under {root}")
        return

    read_total, baseline, accumulated = decompose_read_cost(sessions, on)
    pool = [terseness(sessions, on=on), delegation(sessions, on=on),
            splitting(sessions, max_turns=max_turns, on=on)]
    separate = [model_routing(sessions, on), explore_on_haiku(sessions, on=on)]

    print(f"\n  Measured spend ${total:,.0f} across {len(sessions)} sessions")
    print(f"  Root cause: ${accumulated:,.0f} of it is prior assistant output being re-read")
    print(f"  (measured cache-read ${read_total:,.0f}; only ${baseline:,.0f} is irreducible baseline)\n")

    print("  SUBSTITUTES - all three attack the same pool; they do not add")
    print(f"  {'saving':>10} {'of tot':>6}  {'confidence':<12} lever")
    print("  " + "-" * 80)
    for e in sorted(pool, key=lambda e: -e.saving):
        print("  " + e.line(total))
    print(f"  {'':>10} {'':>6}  joint ceiling = the pool itself: ${accumulated:,.0f} "
          f"({100*accumulated/total:.0f}%)")

    print("\n  SEPARATE LEVERS (additive with the above)")
    for e in sorted(separate, key=lambda e: -e.saving):
        print("  " + e.line(total))

    pool_saving, gen = combine(accumulated, pool, separate)
    realistic = pool_saving + gen
    print(f"\n  COMBINED (substitutes compose multiplicatively on the residual):")
    print(f"    pool removed      ${pool_saving:>9,.0f} of ${accumulated:,.0f}")
    print(f"    generation + separate ${gen:>5,.0f}")
    print(f"    TOTAL             ${realistic:>9,.0f}   ({100*realistic/total:.0f}% of measured spend)")

    print("\n  Assumptions behind modelled figures:")
    for e in pool + separate:
        if e.confidence != "MEASURED" and e.assumptions:
            print(f"    - {e.lever.split('(')[0].strip()}: {e.assumptions}")
    print()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(prog="router.savings")
    ap.add_argument("root", nargs="?", default=str(DEFAULT_ROOT))
    ap.add_argument("--max-turns", type=int, default=300)
    a = ap.parse_args()
    report(a.root, max_turns=a.max_turns)
