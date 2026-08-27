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

from adder.core import settings as _settings
from adder.core.filters import root_of as _root_of
from adder.core.trace import DEFAULT_ROOT, load_sessions
from adder.measure.spend.debt import decompose_read_cost, output_share_of_growth, verbosity_saving
from adder.pricing.cost import EFFORT_OUTPUT_MULT, turn_cost
from adder.pricing.registry import rate

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


def duplicate_reads(sessions, root: Path | str = DEFAULT_ROOT, *,
                    on: date | None = None) -> Estimate:
    """ATTRIBUTED: stop admitting content the context already held.

    The one lever here that trades nothing. Every other substitute gives
    something up -- output that might have been needed, a grep narrow enough to
    miss, a subagent that may summarise away the point -- and this one declines
    calls whose results were already sitting in the prefix, being re-read on
    every turn, when the second copy landed.

    In the pool rather than beside it because it is a substitute for
    `tool_output_discipline`: piping the same command through `head` removes
    the duplicate too, so adding the two would count one saving twice.

    Priced from `reread`, which unions two views: the same call made twice, and
    the same *file* read again however the harness read it. The second is what
    was missing, and it is not a rounding error -- keyed on `Read`'s
    `file_path`, this lever measured $0.00 on any workload whose harness reads
    with `cat`, which is what `bypassPermissions` tells an agent to do.
    """
    from adder.measure.window.reread import _carry, _session_shape, recoverable, scan

    rep = scan(root)
    saving, n = recoverable(rep, _session_shape(sessions), _carry(sessions), on=on)
    _, _, acc = decompose_read_cost(sessions, on)
    return Estimate(
        f"Skip re-reads of content already in context ({n:,} results)",
        saving,
        "ATTRIBUTED",
        "per admission: identical results, plus files re-read after the "
        "session already held them, at the measured carry multiplier",
        "result sizes are estimated from characters, and an edit by a peer "
        "process or by `sed -i` is invisible, so this is an upper bound",
        pool_fraction=min(1.0, saving / acc) if acc else 0.0,
    )


def delegation(sessions, *, delegable_turns: float = 0.25,
               summary_ratio: float = 0.10, sub_model: str | None = None,
               on: date | None = None) -> Estimate:
    """MODELLED: run some turns in a throwaway context instead of the main one.

    The only lever that reaches BOTH halves of the pool: a subagent's own output
    and everything it reads stay in a context that is discarded. Only the
    summary is admitted.
    """
    sub_model = sub_model or _settings.sub_model()
    _, _, accumulated = decompose_read_cost(sessions, on)
    kept = accumulated * delegable_turns * (1 - summary_ratio)

    gen = 0.0
    for s in sessions.values():
        for t in s.turns:
            # `t.pricing_date(on)`, not `on`. `t.rates(on)` resolves the turn's
            # own date; pricing the counterfactual at `on` -- which is None,
            # meaning *today* -- compares a recorded turn against a rate that
            # was not in force when it ran. The two agree only while every rate
            # in the table is stable, which is the assumption `prices.py` exists
            # to refuse: Sonnet 5 reverts from $2/$10 to $3/$15 after
            # 2026-08-31, and this saving would move overnight with nothing in
            # the repository having changed.
            when = t.pricing_date(on)
            r_main, r_sub = t.rates(on), rate(sub_model, when)
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


def _positive(text: str) -> int:
    """An argparse type for a count that is used as a divisor."""
    import argparse

    try:
        n = int(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from e
    if n < 1:
        raise argparse.ArgumentTypeError(
            f"--max-turns must be at least 1, got {n}: sessions are split into "
            "chunks of this many turns, and a chunk of zero turns is not a split")
    return n


def splitting(sessions, *, max_turns: int = 300, on: date | None = None) -> Estimate:
    """MODELLED: cap session length so accumulation resets.

    Uses each session's own measured baseline as the restart floor -- justified
    because baseline context is only ~5-8% of median context (measured).
    """
    saving = 0.0
    for s in sessions.values():
        # Main chain only, and priced turn by turn. Splitting a session is a
        # decision about the main conversation: a subagent runs in its own
        # window and is unaffected by where the parent restarts. Counting its
        # turns put a 71%-sidechain session in here at 3.5x its real length,
        # and taking the whole session's rate from `turns[0]` priced two of the
        # 37 long sessions here at a subagent's model for their entire span.
        main = [t for t in s.turns if not t.sidechain]
        if max_turns < 1 or len(main) <= max_turns:
            # `i % max_turns` below. A chunk length of zero is not a smaller
            # split, it is a ZeroDivisionError out of a report.
            continue
        n = len(main)
        actual = sum(t.context * t.rates(on).cache_read for t in main) / M
        floor = s.base_context
        slope = max(0.0, (s.peak_context - floor) / max(1, n))
        # Each simulated turn keeps the rate of the turn it stands in for, so a
        # session that changed model mid-way is not re-priced at one of them.
        simulated = sum(
            (floor + slope * (i % max_turns)) * t.rates(on).cache_read / M
            for i, t in enumerate(main)
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
        t.out * frac * t.rates(on).out / M
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

    Attributed per-rebuild by cause in `adder.measure.window.cache`; only causes with an
    actual fix (a mid-session model switch, an idle gap a longer TTL covers)
    are counted. Rebuilds after compaction, or after gaps longer than the
    maximum 1h TTL, are excluded because nothing recovers them.
    """
    from adder.measure.window.cache import analyse

    rep = analyse(sessions, on)
    return Estimate(
        f"Avoid recoverable cache rebuilds ({len(rep.misses):,} large rebuilds seen)",
        rep.recoverable,
        "MEASURED",
        "per-rebuild waste = (write_mult - 0.10) x input rate x rebuilt tokens",
        "",
    )


def compaction_discipline(sessions, on: date | None = None) -> Estimate:
    """MODELLED: compact the sessions that ran full and never did.

    Competes with `splitting` for the same accumulated pool, which is why it
    belongs in the substitute list rather than beside it: a session that was
    split never reaches the ceiling, so it has no compaction to miss. The
    multiplicative composition in `combine` is what stops the two being added.
    """
    from adder.measure.window.compact import analyse

    rep = analyse(sessions)
    saving = rep.missed_total(on=on)
    _, _, acc = decompose_read_cost(sessions, on)
    return Estimate(
        "Compact sessions that ran full and never did",
        saving,
        "MODELLED",
        "one compaction per session that never compacted, simulated turn by "
        "turn against the real trajectory so the refill closes the gap",
        "assumes nothing dropped had to be re-derived; a compaction that loses "
        "a needed detail is paid for twice (`adder reread`)",
        pool_fraction=min(1.0, saving / acc) if acc else 0.0,
    )


def memory_trim(sessions, repo=None, on: date | None = None) -> Estimate:
    """ATTRIBUTED: delete resident text that is duplicated or over-long.

    Separate rather than pooled, and for a structural reason: every lever in
    the pool attacks `accumulated` read cost, and resident memory is not in it.
    It sits in the term `debt.decompose_read_cost` calls the **irreducible
    baseline** -- the floor carried on every turn. Part of that floor is a file
    on disk, so the name was wrong: it is reducible, it is just reducible by
    editing rather than by working differently.

    Only duplicated lines and over-long descriptions are counted. The rest of
    an instruction file is doing a job, and a lever that books deleting it as a
    saving is a lever that gets someone's `CLAUDE.md` deleted.
    """
    from adder.measure.window.memory import analyse

    rep = analyse(sessions, repo, on=on)
    saving = rep.pricing.window_cost(rep.recoverable_tokens, scope="project")
    return Estimate(
        "Delete duplicated and over-long resident memory",
        saving,
        "ATTRIBUTED",
        "measured resident tokens, priced at the measured re-read multiplier "
        "over this project's sessions",
        "retrospective: it is what the duplication has already cost, not a "
        "forecast",
    )


def model_routing(sessions, on: date | None = None) -> Estimate:
    """MODELLED: per-turn downgrade where the cache gate permits. The original ask."""
    from adder.pricing.cost import switch_is_profitable

    sub = _settings.sub_model()
    saving = 0.0
    eligible = 0
    for s in sessions.values():
        for t in s.turns:
            # The turn's OWN model is what a downgrade moves away from. Pricing
            # every non-Haiku turn as though it had run on Opus overstated the
            # saving by the Opus/Sonnet rate ratio on any workload that is not
            # pure Opus -- and it did so silently, because the number still
            # looked like a per-turn measurement.
            if t.model == sub or t.rates(on).out <= rate(sub, t.pricing_date(on)).out:
                continue
            d = switch_is_profitable(t.model, sub, t.context, t.out,
                                     on=t.pricing_date(on))
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


def explore_on_haiku(sessions, cheap: str | None = None,
                     on: date | None = None) -> Estimate:
    """MEASURED: rerun existing subagent turns at the cheap tier's rates.

    `cheap` defaults to the configured T0 rung rather than to Haiku. The old
    default named one vendor's model in the lever's own title, so a Codex or
    Gemini CLI workload was told to "run subagents on Haiku" -- advice for a
    model that machine does not dispatch to.
    """
    cheap = cheap or _settings.sub_model()
    actual = saved = 0.0
    n = 0
    for s in sessions.values():
        for t in s.turns:
            when = t.pricing_date(on)
            if not t.sidechain or t.rates(on).inp <= rate(cheap, when).inp:
                continue
            actual += t.cost(on)
            # Both legs on the turn's own date. `on=on` priced the recorded turn
            # at the day it ran and the counterfactual at today, so the
            # difference moved when a rate expired.
            saved += turn_cost(cheap, uncached_in=t.uncached_in, cache_read=t.cache_read,
                               cache_write=t.cache_write, out=t.out, ttl=t.ttl, on=when)
            n += 1
    return Estimate(
        f"Run subagents on {cheap} ({n:,} existing subagent turns)",
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


def levers(sessions, root: Path | str = DEFAULT_ROOT, *, max_turns: int = 300,
           on: date | None = None) -> tuple[list[Estimate], list[Estimate]]:
    """`(substitutes, separate)` — the two lists every caller needs.

    Split out of `report` so the JSON view and the text view cannot drift into
    quoting different levers, which is exactly what happens when a second
    caller reconstructs the list by hand.
    """
    pool = [
        terseness(sessions, on=on),
        tool_output_discipline(sessions, root, on=on),
        duplicate_reads(sessions, root, on=on),
        delegation(sessions, on=on),
        splitting(sessions, max_turns=max_turns, on=on),
        effort_reduction(sessions, on=on),
        compaction_discipline(sessions, on=on),
    ]
    separate = [model_routing(sessions, on), explore_on_haiku(sessions, on=on),
                cache_discipline(sessions, on), memory_trim(sessions, on=on)]
    return pool, separate


def report(root: Path | str = DEFAULT_ROOT, *, max_turns: int = 300,
           on: date | None = None) -> None:
    sessions = load_sessions(root, use_cache=True)
    total = sum(s.cost_on(on) for s in sessions.values())
    if not total:
        print(f"No priced turns found under {root}")
        return

    read_total, baseline, accumulated = decompose_read_cost(sessions, on)
    share = output_share_of_growth(sessions)

    pool, separate = levers(sessions, root, max_turns=max_turns, on=on)

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
    print("\n  COMBINED (substitutes compose multiplicatively on the residual):")
    print(f"    pool removed      ${pool_saving:>9,.0f} of ${accumulated:,.0f}")
    print(f"    generation + separate ${gen:>5,.0f}")
    print(f"    TOTAL             ${realistic:>9,.0f}   ({100*realistic/total:.0f}% of measured spend)")

    print("\n  Assumptions behind modelled figures:")
    for e in pool + separate:
        if e.confidence != "MEASURED" and e.assumptions:
            print(f"    - {e.lever.split('(')[0].strip()}: {e.assumptions}")
    print("\n  Every one of these trades tokens for something. Run `adder quality`")
    print("  before and after to check the agent did not get worse.")
    print()


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="adder savings")
    ap.add_argument("root", nargs="?", default=None,
                    help="transcript directory (default: the `root` setting)")
    ap.add_argument("--max-turns", type=_positive, default=300,
                    help="turns per chunk when modelling a split "
                         "(default: %(default)s)")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    a = ap.parse_args(argv)
    # `root_of`: the argument if one was given, else the `root`
    # setting. Resolved here so two commands cannot disagree
    # about which transcript directory `adder config` names.
    a.root = str(_root_of(a))

    if a.json:
        import json

        sessions = load_sessions(a.root, use_cache=True)
        total = sum(s.cost_on() for s in sessions.values())
        _, _, accumulated = decompose_read_cost(sessions)
        pool, separate = levers(sessions, a.root, max_turns=a.max_turns)
        pool_saving, gen = combine(accumulated, pool, separate)
        print(json.dumps({
            "total": round(total, 4),
            "addressable_pool": round(accumulated, 4),
            "substitutes": [
                {"lever": e.lever, "saving": round(e.saving, 4),
                 "confidence": e.confidence,
                 "share": round(e.saving / total, 5) if total else 0.0,
                 "assumptions": e.assumptions}
                for e in sorted(pool, key=lambda e: -e.saving)
            ],
            "separate": [
                {"lever": e.lever, "saving": round(e.saving, 4),
                 "confidence": e.confidence,
                 "share": round(e.saving / total, 5) if total else 0.0,
                 "assumptions": e.assumptions}
                for e in sorted(separate, key=lambda e: -e.saving)
            ],
            "combined": round(pool_saving + gen, 4),
            "combined_share": round((pool_saving + gen) / total, 5) if total else 0.0,
        }))
        return 0

    report(a.root, max_turns=a.max_turns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
