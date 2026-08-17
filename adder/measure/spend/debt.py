"""Context debt: the true lifetime cost of an output token.

The finding this module exists for
----------------------------------
Measured across this machine's transcripts, assistant output accounts for ~50%
of main-chain context growth -- the largest single source, and the only one the
model itself controls. Much of a long session's context is the model's own
previous words being re-read.

(An earlier version of this analysis reported ~105%. That figure came from
records that multi-counted every multi-block turn; see `trace.iter_file`. The
duplicates inflated output ~1.78x while leaving context deltas untouched. The
corrected share is ~50%, and `verbosity_saving` now scales by the measured
share rather than assuming all growth is output -- otherwise every terseness
claim is roughly double what it can deliver.)

That makes an output token a **liability, not an expense**. It is billed once at
generation, and then again as cached input on every remaining turn:

    true_cost(1 token, R remaining turns)
        = rate_out                       # generated once
        + rate_in * 0.10 * R             # re-read forever after

On Opus 5 that is $25/MTok + $0.50/MTok per remaining turn, so the multiple over
sticker price is `1 + R/50`:

    R=50    2.0x      R=340   7.8x  (measured median session)
    R=759  16.2x      R=1854 38.1x  (measured p90 and worst session)

Every cost tool in existence reports the generation cost and misses the rest.

The actionable consequence is unusual: **terseness compounds**. Halving output
does not save 8% of a bill that is 92% input -- it halves the largest single
contributor to the input. But it is bounded by the output share of growth: at
50%, cutting output by 30% removes at most 15% of the accumulated pool, not 30%.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from adder.core import settings as _settings
from adder.pricing.cost import Rates

M = 1_000_000.0


def debt_multiple(remaining_turns: int, model: str | None = None,
                  on: date | None = None) -> float:
    """How many times its sticker price an output token really costs.

    The re-read term is the *provider's* cache read rate, not a fixed 0.10x of
    input. That distinction decides the headline of this whole report: on a
    provider with no prompt cache, re-reads cost full input rate and the debt
    multiple is roughly ten times larger, so verbosity is ten times more
    expensive than an Anthropic-shaped estimate would say.
    """
    r = Rates.for_model(model or _settings.session_model(), on=on)
    if r.out <= 0:
        return 1.0
    return 1.0 + (r.cache_read * max(0, remaining_turns)) / r.out


def token_lifetime_cost(n_tokens: int, remaining_turns: int,
                        model: str | None = None, on: date | None = None) -> float:
    """USD for generating `n_tokens` AND re-reading them for the rest of the session."""
    r = Rates.for_model(model or _settings.session_model(), on=on)
    return n_tokens * (r.out + r.cache_read * max(0, remaining_turns)) / M


def breakeven_remaining_turns(model: str | None = None,
                              on: date | None = None) -> int:
    """Turns after which re-read cost exceeds generation cost. On Opus 5: 50.

    Returns 0 when the model has no cache read rate to speak of -- meaning the
    break-even is immediate, which is the honest answer rather than a divide by
    zero.
    """
    r = Rates.for_model(model or _settings.session_model(), on=on)
    if r.cache_read <= 0:
        return 0
    return round(r.out / r.cache_read)


@dataclass
class VerbosityImpact:
    current_out: int
    reduction: float
    generation_saved: float
    reread_saved: float

    @property
    def total(self) -> float:
        return self.generation_saved + self.reread_saved

    @property
    def leverage(self) -> float:
        """Downstream saving per dollar of generation saving."""
        return self.reread_saved / self.generation_saved if self.generation_saved else 0.0


def _measured_read_cost(sess, on: date | None = None) -> float:
    return sum(
        t.cache_read * t.rates(on).cache_read / M
        for t in sess.turns
    )


def decompose_read_cost(sessions, on: date | None = None) -> tuple[float, float, float]:
    """Split MEASURED cache-read spend into (total, irreducible baseline, accumulated).

    Bounded by what was actually spent. A naive forward projection -- assume every
    output token persists for all remaining turns -- overshoots the real bill by
    ~35% on this dataset, because compaction drops content. Measured decomposition
    is used instead so the saving can never be overstated.

    `accumulated` is the pool the three substitute levers compete over. Only the
    measured output share of it (~50%) is reachable by verbosity alone; the rest
    is tool results and user input. `verbosity_saving` applies that scaling.
    """
    total = baseline = 0.0
    for s in sessions.values():
        if not s.turns:
            continue
        measured = _measured_read_cost(s, on)
        total += measured
        # Each chain against its OWN floor and its OWN turn count. The main
        # chain's floor is the system prompt, tools and CLAUDE.md; a subagent
        # opens with its own, much smaller one. Multiplying the main-chain floor
        # by every turn in the session -- sidechain turns included -- charged
        # each delegated turn for a prefix it never carried, which OVERSTATES
        # the irreducible baseline and so understates `accumulated`. That is the
        # pool every substitute lever is scaled by, and `Session.base_context`
        # already spells out the same arithmetic in the other direction:
        # "Understating the floor overstates the pool."
        side = [t for t in s.turns if t.sidechain]
        base_cost = 0.0
        for chain in (s.main_turns, side):
            if not chain:
                continue
            floor = min(t.context for t in chain)
            base_cost += floor * len(chain) * chain[0].rates(on).cache_read / M
        baseline += min(base_cost, measured)      # can never exceed what was spent
    return total, baseline, max(0.0, total - baseline)


def output_share_of_growth(sessions) -> float:
    """Measured share of main-chain context growth that is assistant output.

    Delegates to `adder.measure.window.context` so there is one definition of this number.
    Returns 1.0 only when growth cannot be measured, which is the conservative
    direction for a *bound* but the optimistic one for a *claim* -- so callers
    that cannot measure it should say so.
    """
    from adder.measure.window.context import output_share_of_growth as _share

    try:
        share = _share(sessions)
    except Exception:
        return 1.0
    if share <= 0.0:
        return 1.0
    return min(1.0, share)


def verbosity_saving(sessions, *, reduction: float = 0.30,
                     output_share: float | None = None,
                     on: date | None = None) -> VerbosityImpact:
    """What a proportional cut in output verbosity is worth.

    Re-read saving is `reduction` x `output_share` x the MEASURED accumulated
    read cost -- not a forward projection, and not the whole pool.

    `output_share` is the fraction of context growth that is actually assistant
    output (~0.50 measured here). Assuming 1.0 is the single easiest way to
    over-claim this lever by 2x: the other half of the pool is tool results and
    user input, which no amount of terseness touches.
    """
    _, _, accumulated = decompose_read_cost(sessions, on)
    if output_share is None:
        output_share = output_share_of_growth(sessions)
    output_share = max(0.0, min(1.0, output_share))
    gen = 0.0
    total_out = 0
    for s in sessions.values():
        for t in s.turns:
            total_out += t.out
            gen += t.out * reduction * t.rates(on).out / M
    return VerbosityImpact(total_out, reduction,
                           gen, accumulated * reduction * output_share)


def report(sessions, on: date | None = None) -> str:
    """Human-readable context-debt analysis."""
    lines: list[str] = []
    be = breakeven_remaining_turns(_settings.session_model(), on)
    lines.append("  Context debt: what an output token really costs")
    lines.append("")
    lines.append(f"  {'remaining turns':>16}{'multiple of sticker price':>28}")
    for R in (0, be, 200, 340, 759, 1854):
        # On the same date as `be` above. `debt_multiple(R)` with no date
        # priced the table at *today* while the break-even beside it was
        # priced at `on`, so the row for `be` did not sit at the 2.0x the
        # break-even is defined as.
        lines.append(f"  {R:>16,}{debt_multiple(R, on=on):>27.1f}x")
    lines.append("")
    lines.append(f"  Past {be} remaining turns, re-reading an output token costs more")
    lines.append("  than generating it did.")

    total_out = sum(t.out for s in sessions.values() for t in s.turns)
    gen_cost = sum(
        t.out * t.rates(on).out / M for s in sessions.values() for t in s.turns
    )
    total_read, baseline, accumulated = decompose_read_cost(sessions, on)

    lines.append("")
    lines.append(f"  This dataset: {total_out:,} output tokens")
    lines.append(f"    generation cost           ${gen_cost:>9,.0f}   (what cost tools report)")
    lines.append(f"    measured cache-read       ${total_read:>9,.0f}")
    lines.append(f"      irreducible baseline    ${baseline:>9,.0f}   (system prompt, tools, CLAUDE.md)")
    lines.append(f"      accumulated prior output${accumulated:>9,.0f}   <- what verbosity controls")
    share = output_share_of_growth(sessions)
    if gen_cost:
        attributable = accumulated * share
        lines.append(f"    of which attributable to output ${attributable:>9,.0f}   "
                     f"({100*share:.0f}% measured output share)")
        lines.append(f"    true cost of output is {(gen_cost + attributable)/gen_cost:.1f}x "
                     f"the reported figure")

    lines.append("")
    lines.append(f"  Assistant output is {100*share:.0f}% of measured context growth, so a")
    lines.append("  verbosity cut can only reach that share of the accumulated pool.")
    lines.append("")
    lines.append("  Value of writing less:")
    lines.append(f"  {'cut':>6}{'generation':>13}{'re-read':>12}{'total':>11}{'leverage':>11}")
    for red in (0.10, 0.20, 0.30, 0.50):
        v = verbosity_saving(sessions, reduction=red, on=on)
        lines.append(f"  {red:>5.0%}${v.generation_saved:>12,.0f}${v.reread_saved:>11,.0f}"
                     f"${v.total:>10,.0f}{v.leverage:>10.1f}x")
    lines.append("")
    lines.append("  Leverage = downstream dollars saved per dollar of generation saved.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    from adder.core.filters import add_arguments as add_window
    from adder.core.filters import load as load_window

    ap = argparse.ArgumentParser(
        prog="adder debt",
        description="What an output token really costs once re-reads are counted.")
    add_window(ap)
    ap.add_argument("--model", default=_settings.session_model(),
                    help="model to price the debt table with")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    a = ap.parse_args(argv)

    sessions, window = load_window(a)

    if a.json:
        total, baseline, accumulated = decompose_read_cost(sessions)
        share = output_share_of_growth(sessions)
        print(json.dumps({
            "model": a.model,
            "breakeven_remaining_turns": breakeven_remaining_turns(a.model),
            "debt_multiple": {str(r): round(debt_multiple(r, a.model), 3)
                              for r in (0, 50, 200, 340, 759, 1854)},
            "measured_cache_read": round(total, 4),
            "irreducible_baseline": round(baseline, 4),
            "accumulated_pool": round(accumulated, 4),
            "output_share_of_growth": round(share, 5),
            "verbosity_saving": {
                f"{int(100 * r)}%": round(verbosity_saving(sessions, reduction=r).total, 4)
                for r in (0.10, 0.20, 0.30, 0.50)
            },
            "filter": window.describe(),
        }))
        return 0

    print()
    print(report(sessions))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
