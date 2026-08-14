"""What each lever is worth, unified around one pool.

The causal model
----------------
Context re-reads are ~78% of spend, so the bill is dominated by one pool:
**tokens that were admitted to the main context and are re-read every turn.**

Measured on deduplicated records, that pool has two roughly equal halves:

    ~50%  assistant output    -- the model's own prior words
    ~50%  read content        -- tool results (Bash dominates), user input

That split matters because it decides which advice can work. Terseness only
reaches the first half; narrowing tool output only reaches the second. An
earlier version of this analysis put output at ~105% of growth and concluded
verbosity was the whole story -- that came from multi-counted records (see
`trace.iter_file`) and overstated the terseness lever roughly twofold.

Levers that attack the pool are **substitutes, not complements**: applying one
leaves less pool for the next. Reporting their sum double-counts, so they are
composed multiplicatively on the residual and the joint ceiling is the pool.

Model routing and subagent-model choice are separate: they change the *price* of
tokens rather than the *number* of them, so they add rather than compete.

Confidence tiers:
  MEASURED   - recomputed from recorded tokens; no assumptions.
  ATTRIBUTED - exact arithmetic assigning recorded spend to a cause.
  MODELLED   - depends on a stated counterfactual.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .cost import EFFORT_OUTPUT_MULT, turn_cost
from .debt import decompose_read_cost, output_share_of_growth, verbosity_saving
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
    pool_fraction: float = 0.0   # share of the accumulated pool it removes
    generation_saving: float = 0.0

    def line(self, total: float) -> str:
        pct = f"{100 * self.saving / total:.1f}%" if total else "-"
        return f"${self.saving:>9,.0f}  {pct:>6}  [{self.confidence:<10}] {self.lever}"


def terseness(sessions, *, reduction: float = 0.30, on: date | None = None) -> Estimate:
    """ATTRIBUTED: cut output by `reduction`, bounded by the measured output share."""
    share = output_share_of_growth(sessions)
    v = verbosity_saving(sessions, reduction=reduction, output_share=share, on=on)
    return Estimate(
        f"Write {reduction:.0%} less (leverage {v.leverage:.1f}x downstream)",
        v.total,
        "ATTRIBUTED",
        f"reduction x measured {share:.0%} output share of growth x accumulated read cost",
        "assumes output can be cut proportionally without losing task quality; "
        "reaches only the output half of the pool",
        pool_fraction=reduction * share,
        generation_saving=v.generation_saved,
    )


def tool_output_discipline(sessions, root: Path | str = DEFAULT_ROOT, *,
                           reduction: float = 0.40,
                           on: date | None = None) -> Estimate:
    """ATTRIBUTED: admit less tool output to context.

    The other half of the pool, and the half no writing-style instruction can
    reach. Measured here: `Bash` results alone are the single largest source of
    read content. Piping verbose commands through `head`, selecting columns, and
    asking for counts instead of listings cuts it without losing information the
    task needs -- the full output is still on disk if it is wanted.
    """
    _, _, accumulated = decompose_read_cost(sessions, on)
    share = output_share_of_growth(sessions)
    read_share = max(0.0, 1.0 - share)
    saving = accumulated * read_share * reduction
    return Estimate(
        f"Cut tool output admitted to context by {reduction:.0%}",
        saving,
        "ATTRIBUTED",
        f"measured {read_share:.0%} read share of growth x accumulated read cost",
        f"assumes {reduction:.0%} of tool output is redundant to the task "
        "(piping through head/wc, narrower greps); unverified per-call",
        pool_fraction=read_share * reduction,
    )


def delegation(sessions, *, delegable_turns: float = 0.25,
               summary_ratio: float = 0.10, sub_model: str = "claude-haiku-4-5",
               on: date | None = None) -> Estimate:
    """MODELLED: run some turns in a throwaway context instead of the main one.

    The only lever that reaches BOTH halves of the pool: a subagent's own output
    and everything it reads stay in a context that is discarded. Only the
    summary is admitted.
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
        "measured accumulated read cost, re-priced with delegated turns excluded",
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


def effort_reduction(sessions, *, from_effort: str = "high",
                     to_effort: str = "medium", on: date | None = None) -> Estimate:
    """MODELLED: lower reasoning effort one notch.

    The only output-side lever that does NOT invalidate the prompt cache: same
    model, same prefix, fewer tokens. A model downgrade rebuilds the whole
    context at up to 12.5x the cached read price; an effort change costs nothing
    to apply mid-session.
    """
    mult = EFFORT_OUTPUT_MULT
    if from_effort not in mult or to_effort not in mult:
        raise ValueError(f"unknown effort level; known: {sorted(mult)}")
    frac = max(0.0, (mult[from_effort] - mult[to_effort]) / mult[from_effort])
    share = output_share_of_growth(sessions)
    _, _, accumulated = decompose_read_cost(sessions, on)

    gen = sum(
        t.out * frac * rate(t.model, on).out / M
        for s in sessions.values() for t in s.turns
    )
    reread = accumulated * share * frac
    return Estimate(
        f"Drop effort {from_effort} -> {to_effort} (~{frac:.0%} less output)",
        gen + reread,
        "MODELLED",
        "effort-to-output priors x measured output share x accumulated read cost",
        f"assumes {to_effort} effort produces ~{frac:.0%} less output than "
        f"{from_effort} AND does not reduce task success; the token ratio is a "
        "prior, not measured here",
        pool_fraction=frac * share,
        generation_saving=gen,
    )


def cache_discipline(sessions, on: date | None = None) -> Estimate:
    """MEASURED: cache rebuilds that a configuration change would have avoided.

    Attributed per-rebuild by cause in `router.cache`; only causes with an
    actual fix (a mid-session model switch, an idle gap a longer TTL covers)
    are counted. Rebuilds after compaction, or after gaps longer than the
    maximum 1h TTL, are excluded because nothing recovers them.
    """
    from .cache import analyse

    rep = analyse(sessions, on)
    return Estimate(
        f"Avoid recoverable cache rebuilds ({len(rep.misses):,} large rebuilds seen)",
        rep.recoverable,
        "MEASURED",
        "per-rebuild waste = (write_mult - 0.10) x input rate x rebuilt tokens",
        "",
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
        "cache-gated, context-limit-gated switch applied per recorded turn",
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
                               cache_write=t.cache_write, out=t.out, ttl=t.ttl, on=on)
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
    sessions = load_sessions(root, use_cache=True)
    total = sum(s.cost_on(on) for s in sessions.values())
    if not total:
        print(f"No priced turns found under {root}")
        return

    read_total, baseline, accumulated = decompose_read_cost(sessions, on)
    share = output_share_of_growth(sessions)

    pool = [
        terseness(sessions, on=on),
        tool_output_discipline(sessions, root, on=on),
        delegation(sessions, on=on),
        splitting(sessions, max_turns=max_turns, on=on),
        effort_reduction(sessions, on=on),
    ]
    separate = [model_routing(sessions, on), explore_on_haiku(sessions, on=on),
                cache_discipline(sessions, on)]

    print(f"\n  Measured spend ${total:,.0f} across {len(sessions)} sessions")
    print(f"  Root cause: ${accumulated:,.0f} of it is prior context being re-read")
    print(f"  (measured cache-read ${read_total:,.0f}; only ${baseline:,.0f} is irreducible baseline)")
    print(f"  That pool is ~{share:.0%} assistant output and ~{1-share:.0%} tool output "
          f"and user input.\n")

    print("  SUBSTITUTES - all attack the same pool; they do not add")
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
    print("\n  Every one of these trades tokens for something. Run `rt quality`")
    print("  before and after to check the agent did not get worse.")
    print()


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="router.savings")
    ap.add_argument("root", nargs="?", default=str(DEFAULT_ROOT))
    ap.add_argument("--max-turns", type=int, default=300)
    a = ap.parse_args(argv)
    report(a.root, max_turns=a.max_turns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
