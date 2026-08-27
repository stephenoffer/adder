"""What installing this tool is worth, and what it is worth only if you obey it.

Why this is not `adder plan`
----------------------------
`plan` asks the optimiser's question: what is the cheapest way this workload
could have been run? It answers with a regime it searched for, and on this
machine that regime reaches 10x. That is the right number for setting a target
and the wrong number to quote to somebody deciding whether to install anything,
because it silently assumes they will do everything the tool says.

The question this module asks instead is the one that comes first: **what
changes if I install adder and keep working exactly as I do now?** That is a
different and much smaller number, because only two things here can act without
being obeyed -- the PreToolUse read guard, which prices a read before it lands,
and the tier definitions in `.claude/agents/`, which decide what a delegated
step runs on. Everything else in this repo is a report. A report saves nothing
until somebody acts on it.

So the ladder below is split at that line, and the split is the finding:

    installed, nothing else changed        the guard's own defaults, on this workload
    + the threshold and cadence it solves  what the reports tell you to do

Measured on the author's transcripts the first is ~1.6x and the second ~6.7x.
Reporting only the second would be the more impressive number and a lie by
omission: it is not what installing the tool gets you, it is what restructuring
your work around the tool gets you, and the second requires the first as a
prerequisite rather than the other way round.

Where the line moved
--------------------
That gap was a standing indictment of the tool and it is the reason
`adder auto` exists. An enforcing guard does not advise the delegation
threshold, it *refuses* the calls above it, so the delegation half of the
bottom rung crossed the line and became something installing gets you. What did
not cross it is the restart cadence: no hook can restart a session, and pricing
it as if one could would put back exactly the lie this module was written to
avoid. So the two are now separate rungs, and only the cadence still carries
the asterisk.

The threshold the guard enforces is not the one the reports solve for -- 800
tokens against ~300 -- because below 800 the hook starts parsing a transcript
on half of all tool calls to find money that the dollar gate has already found.
That difference is visible in the ladder rather than averaged away.

The guard's threshold is derived, not chosen
--------------------------------------------
The guard fires on a **cost** -- $0.25 by default -- and not on a token count,
so "delegate reads over N tokens" is not a setting anybody typed. N falls out of
the horizon: admitting a token to a context that will be re-read E more times
costs `(w + m*E)` times the input rate, so the size at which that reaches $0.25
is one division. It also has a floor, `GUARD_MIN_TOKENS`, which exists so the
hook does not parse a transcript on every trivial read -- and on this workload
the floor is the binding constraint, which is worth knowing before tuning the
dollar gate that is not doing the work.

What is modelled, and how far it moves the answer
-------------------------------------------------
Three inputs decide the headline and none of them is recoverable from a
transcript: what a delegated read hands back (`summary_ratio`), how often a
delegated step has to be redone (`p_fail`), and how many tokens a restarted
session needs to be told (`handoff_tokens`). They are swept rather than
asserted, and the report prints the worst corner next to the nominal figure.
The pessimistic corner lands near 3.5x, so the honest statement of the result is
a range whose floor still clears "worth doing" and whose top requires believing
that a subagent can summarise a read to a tenth of its size without losing what
the session needed. `adder ab` is the only thing here that can test that.
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass, replace
from datetime import date
from itertools import product
from pathlib import Path

from adder.core.filters import root_of as _root_of
from adder.core.trace import DEFAULT_ROOT, load_sessions
from adder.evaluate.replay.plan import (
    DEFAULT_OUTPUT_SHARE,
    Regime,
    Result,
    dominant_model,
    prepare,
    recommended_cadence,
    recommended_threshold,
    replay,
)
from adder.measure.session.horizon import Horizon
from adder.measure.window.prefix import DEFAULT_HANDOFF, weighted_median_turns
from adder.pricing.cost import admitted_token_cost

# Mirrors MIN_TOKENS in .claude/hooks/pretooluse_read_guard.py. Duplicated
# rather than imported because the hook is not an importable module -- it is a
# standalone script the harness execs -- so `tests/test_bench.py` parses the
# hook and fails if these two ever drift.
GUARD_MIN_TOKENS = 2_000

# The corners the headline is quoted against. Nominal first in each tuple, so
# `corner_sweep` can report the nominal and the worst case from one grid.
SUMMARY_RATIOS = (0.10, 0.30)
P_FAILS = (0.15, 0.30)
HANDOFFS = (DEFAULT_HANDOFF, 20_000)


def guard_threshold(*, remaining_turns: int, model: str, min_cost: float,
                    min_tokens: int = GUARD_MIN_TOKENS, ttl: str = "5m",
                    on: date | None = None) -> int:
    """Read size, in tokens, at which the shipped guard actually speaks up.

    `admitted_token_cost` is linear in the token count, so the size at which a
    read reaches `min_cost` is one division rather than a search. The floor is
    applied afterwards because the guard applies it first, as an I/O guard: it
    returns before pricing anything below `min_tokens`, so a dollar gate that
    resolves below the floor cannot fire and quoting it would overstate what the
    hook does.
    """
    per_token = admitted_token_cost(1_000_000, model, remaining_turns,
                                    ttl=ttl, on=on) / 1_000_000
    if per_token <= 0:
        return min_tokens
    return max(min_tokens, math.ceil(min_cost / per_token))


def duplicate_admissions(root: Path | str = DEFAULT_ROOT,
                         *, window=None) -> dict[tuple[str, int], int]:
    """Tokens each turn admitted that its own context already held.

    One transcript scan, handed to `prepare` so every rung of the ladder is
    replayed against the same measured set. Never raises: a bench that cannot
    read the tree should report a ladder without this row, not a traceback.
    """
    from adder.measure.window.reread import avoidable_by_turn, scan

    try:
        return avoidable_by_turn(scan(root, window=window))
    except (OSError, ValueError):
        return {}


def expected_reads(sessions) -> int:
    """Re-reads a token admitted at a cost-typical point will have to pay for.

    Taken at the midpoint of the cost-weighted median session rather than at
    turn zero. A token admitted on the opening turn is re-read by the whole
    session and one admitted on the last turn by nothing; the midpoint is the
    position an average admitted token actually occupies, and it is what the
    guard would price against over the workload as a whole.
    """
    mid = max(0, weighted_median_turns(sessions) // 2)
    return max(1, int(Horizon.from_sessions(sessions).mean_remaining(mid)))


@dataclass(frozen=True)
class Config:
    """One rung, plus whether anything in the repo makes it happen by itself."""

    label: str
    regime: Regime
    enforced: bool          # True if a hook or an agent file does this unprompted
    note: str = ""


def ladder(sessions, *, min_cost: float, handoff_tokens: int = DEFAULT_HANDOFF,
           min_tokens: int | None = None, on: date | None = None,
           enforcing: bool | None = None, refuses_duplicates: bool | None = None,
           duplicates: bool = False) -> list[Config]:
    """The four rungs, each adding exactly one thing to the one above it.

    Rungs are cumulative because the levers are substitutes: they all attack the
    same pool of re-read context, so pricing them independently and adding the
    results counts the same dollars more than once.

    `min_tokens` is the guard's own floor, and it is a *setting*
    (`guard_min_tokens`). Pinning it to the shipped default made this report
    describe a guard the reader may not be running -- and the note below says
    which of the two constraints binds, so on a machine that raised the floor
    the report named the wrong one.
    """
    model = dominant_model(sessions)
    reads = expected_reads(sessions)
    floor = GUARD_MIN_TOKENS if min_tokens is None else max(0, int(min_tokens))
    thr = guard_threshold(remaining_turns=reads, model=model, min_cost=min_cost,
                          min_tokens=floor, on=on)
    gate = math.ceil(min_cost / (admitted_token_cost(1_000_000, model, reads, on=on)
                                 / 1_000_000))
    binding = "the token floor" if thr <= floor else f"the ${min_cost:.2f} gate"

    # Whether the guard on this machine refuses the calls it prices or merely
    # describes them. It decides which side of the enforced/advised line the
    # delegation rungs fall on, and it is a *setting*, so the report has to
    # read it rather than assume it -- the same mistake this module already
    # made once by pinning the guard's floor to its shipped default.
    enforcing = _enforcing() if enforcing is None else enforcing
    if refuses_duplicates is None:
        refuses_duplicates = _enforce_level() != "off" or enforcing
    verb = "refuse over" if enforcing else "delegate over"

    rungs = [Config("no adder -- as run", Regime(), enforced=True)]
    if duplicates:
        # First rung, and it belongs first: it is the only lever in this ladder
        # that costs nothing to be wrong about. Every other row trades
        # something -- a summary that may drop what was needed, a restart that
        # has to re-establish the thread -- and this one declines calls whose
        # results the context is already carrying. It is also the row that used
        # to be invisible: keyed on `Read`'s `file_path`, it measured zero on
        # any workload whose harness reads with `cat`.
        rungs.append(Config(
            f"+ the guard's duplicate {'refusal' if refuses_duplicates else 'advice'}",
            replace(rungs[-1].regime, label="duplicates", refuse_duplicates=True),
            enforced=True,
            note=("measured per call by `adder reread`: results whose content "
                  "the context already held, dropped turn by turn and the rest "
                  "of the session re-priced"
                  + (". `guard_enforce=certain` refuses these, so no summary "
                     "ratio and no uptake assumption stands behind this row"
                     if refuses_duplicates else
                     "; `guard_enforce` is off here, so the guard only says so "
                     "and only above its token floor -- `adder auto on` sets "
                     "`certain`, which refuses them"))))
    # Placement first with the model held fixed: this rung measures what moving
    # a read out of the context is worth on its own, before tier choice gets to
    # claim any of it.
    rungs.append(Config(
        f"+ the read guard ({verb} {thr:,} tok)",
        replace(rungs[-1].regime, label="guard", delegate_above=thr,
                right_size=False, sub_model=model),
        enforced=True,
        note=(f"${min_cost:.2f} at {reads:,} expected re-reads is {gate:,} tok; "
              f"the floor is {floor:,}, so {binding} binds"
              + ("" if enforcing else "; advisory, so this assumes it is heeded"))))
    rungs.append(Config(
        "+ the tier agents (.claude/agents)",
        replace(rungs[-1].regime, label="tiers", right_size=True),
        enforced=True,
        note="subagent tier chosen by expected cost, which is what route-t0/t1/t2 encode"))

    cadence, _opening, cadence_note = recommended_cadence(
        sessions, handoff_tokens=handoff_tokens, on=on)
    solved, solved_note = recommended_threshold(sessions, split_turns=cadence, on=on)
    if solved is not None:
        # Split from the cadence, because the two are enforceable by different
        # things. A hook can refuse a read; nothing here can restart a session.
        # Bundling them made the whole of the largest rung unenforced and hid
        # the fact that most of it no longer is.
        rungs.append(Config(
            f"+ the threshold it solves for (over {solved:,} tok)",
            replace(rungs[-1].regime, label="solved", delegate_above=solved),
            enforced=enforcing and solved >= thr,
            note=(solved_note + ("" if enforcing else
                                 "; the guard advises this, it does not enforce it — "
                                 "`adder auto on --full`"))))
        rungs.append(Config(
            f"+ restarting every {cadence} turns",
            replace(rungs[-1].regime, label="cadence", split_turns=cadence,
                    handoff_tokens=handoff_tokens),
            enforced=False,
            note=f"{cadence_note}. No hook can restart a session; this one is yours"))
    return rungs


def _enforce_level() -> str:
    """`off`, `certain` or `full` -- what the guard on this machine will refuse.

    Read at call time and never cached, for the reason `guard.Settings` gives:
    a constant resolved at import is one no test can change and no `.adder.json`
    can override.
    """
    try:
        from adder.decide.guard import Settings
        return Settings.resolve().enforce
    except Exception:
        return "off"


def _enforcing() -> bool:
    """Is the guard configured to refuse a read on *size*?

    `certain` refuses duplicates and nothing else, so it does not move the
    delegation threshold above the line. That distinction is the whole reason
    this is not a boolean any more: the duplicate refusal was landing on the
    unenforced side of a report about what installing the tool gets you, while
    being the one thing here that needs no uptake assumption at all.
    """
    return _enforce_level() == "full"


@dataclass
class Bench:
    """A finished run: the rungs, their results, and the corner sweep."""

    configs: list[Config]
    results: list[Result]
    measured: float
    sessions: int
    corners: list[tuple[tuple[float, float, int], float]]   # (inputs, multiple)

    @property
    def baseline(self) -> float:
        return self.results[0].total if self.results else 0.0

    def multiple(self, i: int) -> float:
        r = self.results[i]
        return self.baseline / r.total if r.total else float("inf")

    @property
    def installed(self) -> float:
        """Multiple from the last rung nothing has to be obeyed to reach.

        The *first* unenforced rung stops the count, not the last enforced one.
        Rungs are cumulative, so an unenforced row with an enforced row above
        it puts the unenforced saving into a number whose whole claim is that
        nothing has to be obeyed to get it. The ladder happens to be ordered so
        that cannot arise today; it took one inserted row to make it possible.
        """
        last = 0
        for i, c in enumerate(self.configs):
            if not c.enforced:
                break
            last = i
        return self.multiple(last)

    @property
    def followed(self) -> float:
        return self.multiple(len(self.results) - 1)

    @property
    def worst_corner(self) -> float:
        return min((m for _inputs, m in self.corners), default=self.followed)

    @property
    def residual(self) -> float:
        """How far the replay drifts from the measured bill. Nothing below a
        large residual is worth reading, so the caller has to be able to see it."""
        return (self.baseline - self.measured) / self.measured if self.measured else 0.0


def corner_sweep(prepared, regime: Regime, baseline: float, *,
                 output_share: float = DEFAULT_OUTPUT_SHARE,
                 on: date | None = None) -> list[tuple[tuple[float, float, int], float]]:
    """Re-price the headline regime at every corner of the three modelled inputs."""
    out = []
    for sr, pf, ho in product(SUMMARY_RATIOS, P_FAILS, HANDOFFS):
        res = replay(prepared, replace(regime, summary_ratio=sr, p_fail=pf,
                                       handoff_tokens=ho),
                     output_share=output_share, on=on)
        out.append(((sr, pf, ho), baseline / res.total if res.total else float("inf")))
    return out


def run(sessions, *, min_cost: float = 0.25, handoff_tokens: int = DEFAULT_HANDOFF,
        min_tokens: int | None = None, on: date | None = None,
        enforcing: bool | None = None, corners: bool = True,
        dups: dict[tuple[str, int], int] | None = None,
        refuses_duplicates: bool | None = None) -> Bench:
    """Price every rung of the ladder against the recorded turns.

    `corners=False` skips the eight-point sweep of the modelled inputs. The
    sweep is most of the work here -- eight more full replays on top of the
    ladder's five -- and a caller that only wants the enforced multiple, like
    the claim in `validate`, should not pay for a range it does not print.
    """
    from adder.measure.spend.debt import output_share_of_growth

    share = output_share_of_growth(sessions)
    prepared = prepare(sessions, on, dups)
    configs = ladder(sessions, min_cost=min_cost, handoff_tokens=handoff_tokens,
                     min_tokens=min_tokens, on=on, enforcing=enforcing,
                     refuses_duplicates=refuses_duplicates,
                     duplicates=bool(dups))
    results = [replay(prepared, c.regime, output_share=share, on=on) for c in configs]
    measured = sum(s.cost_on(on) for s in sessions.values())
    base = results[0].total
    swept = corner_sweep(prepared, configs[-1].regime, base,
                         output_share=share, on=on) if corners and len(configs) > 1 else []
    return Bench(configs, results, measured, len(sessions), swept)


def report(root: Path | str = DEFAULT_ROOT, *, min_cost: float = 0.25,
           handoff_tokens: int = DEFAULT_HANDOFF, min_tokens: int | None = None,
           on: date | None = None) -> int:
    sessions = load_sessions(root, use_cache=True)
    if not sessions:
        print(f"\n  No transcripts found under {root}\n")
        return 1
    b = run(sessions, min_cost=min_cost, handoff_tokens=handoff_tokens,
            min_tokens=min_tokens, on=on, dups=duplicate_admissions(root))
    if not b.measured:
        print(f"\n  No priced turns found under {root}\n")
        return 1

    print(f"\n  Measured spend            ${b.measured:>10,.0f}   "
          f"{b.results[0].turns:,} turns, {b.sessions} sessions")
    print(f"  Replay of the same turns  ${b.baseline:>10,.0f}   "
          f"residual {b.residual:+.1%}")
    if abs(b.residual) > 0.25:
        print("  ! the replay does not reproduce the measured total closely enough;")
        print("    treat the multiples below as illustrative, not as a quote")
    print()
    print("  Rows are cumulative. The levers are substitutes -- they attack the same")
    print("  pool of re-read context -- so adding them up separately double-counts it.\n")
    print(f"  {'configuration':<58}{'total':>10}{'vs no adder':>13}")
    print("  " + "-" * 81)
    for i, (cfg, res) in enumerate(zip(b.configs, b.results, strict=True)):
        mark = " " if cfg.enforced else "*"
        print(f" {mark}{cfg.label:<58}${res.total:>9,.0f}{b.multiple(i):>12.1f}x")
        for line in textwrap.wrap(cfg.note, 74):
            print(f"      {line}")
    print()
    print(f"  Installed and nothing else changed:  {b.installed:>5.1f}x")
    print("    Everything above the * line happens without you doing anything: the")
    print("    PreToolUse guard prices a read before it lands, and the tier files")
    print("    decide what the delegated step runs on.")
    if not _enforcing():
        print("    The guard is advisory here, so this rests on the advice being")
        print("    taken. `adder auto on --full` makes it refuse instead, which")
        print("    moves the threshold row above the line.")
    if any(not c.enforced for c in b.configs):
        print(f"  Following what the reports say:      {b.followed:>5.1f}x")
        print("    The * row is the restart cadence. No hook can restart a session,")
        print("    so the gap between the two numbers is the part that is on you.")

    if b.corners:
        print()
        print("  The * row rests on three numbers no transcript can settle. Swept:")
        print(f"  {'summary':>9}{'p_fail':>9}{'handoff':>10}{'vs no adder':>14}")
        for (sr, pf, ho), mult in b.corners:
            print(f"  {sr:>9.0%}{pf:>9.0%}{ho:>10,}{mult:>13.1f}x")
        print(f"\n  Nominal {b.followed:.1f}x, worst corner {b.worst_corner:.1f}x. "
              "The floor is set by the summary")
        print("  ratio: if a delegated read hands back 30% of what it read rather than")
        print("  10%, most of the content is back in the context and the carry it was")
        print("  supposed to avoid is only partly avoided.")
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    from adder.core import settings
    from adder.core.filters import Window
    from adder.core.filters import add_arguments as add_window

    ap = argparse.ArgumentParser(
        prog="adder bench",
        description="Benchmark this workload with and without adder, on the same turns.")
    add_window(ap)
    ap.add_argument("--guard-cost", type=float, default=None, metavar="USD",
                    help="read-guard trigger (default: the configured guard_min_cost)")
    ap.add_argument("--handoff", type=int, default=None, metavar="TOK",
                    help=f"tokens a restarted session is told (default: {DEFAULT_HANDOFF:,})")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    a = ap.parse_args(argv)
    # `root_of`: the argument if one was given, else the `root`
    # setting. Resolved here so two commands cannot disagree
    # about which transcript directory `adder config` names.
    a.root = str(_root_of(a))

    min_cost = a.guard_cost if a.guard_cost is not None else float(settings.get("guard_min_cost"))
    handoff = a.handoff if a.handoff is not None else int(settings.get("handoff_tokens"))
    # The guard's own floor is a setting too. Pinning it to the shipped default
    # made this report describe a guard the reader may not be running.
    try:
        min_floor = int(settings.get("guard_min_tokens"))
    except (KeyError, OSError, ValueError):
        min_floor = GUARD_MIN_TOKENS

    window = Window.from_args(a)
    if not window.active and not a.json:
        return report(a.root, min_cost=min_cost, handoff_tokens=handoff,
                      min_tokens=min_floor)

    sessions = load_sessions(a.root, use_cache=bool(settings.get("cache")))
    if window.active:
        sessions = window.apply(sessions)
    if not sessions:
        print(f"\n  No transcripts found under {a.root}\n")
        return 1
    b = run(sessions, min_cost=min_cost, handoff_tokens=handoff,
            min_tokens=min_floor)
    if not a.json:
        # A windowed run re-prices the same ladder; print it the same way.
        for i, (cfg, res) in enumerate(zip(b.configs, b.results, strict=True)):
            print(f"  {cfg.label:<58}${res.total:>9,.0f}{b.multiple(i):>12.1f}x")
        return 0
    print(json.dumps({
        "measured": round(b.measured, 2),
        "baseline": round(b.baseline, 2),
        "residual": round(b.residual, 4),
        "sessions": b.sessions,
        "turns": b.results[0].turns,
        "installed": round(b.installed, 3),
        "followed": round(b.followed, 3),
        "worst_corner": round(b.worst_corner, 3),
        "rungs": [
            {"label": c.label, "enforced": c.enforced,
             "total": round(r.total, 2), "multiple": round(b.multiple(i), 3),
             "delegated_share": round(r.delegated_share, 4)}
            for i, (c, r) in enumerate(zip(b.configs, b.results, strict=True))
        ],
        "corners": [
            {"summary_ratio": sr, "p_fail": pf, "handoff": ho, "multiple": round(m, 3)}
            for (sr, pf, ho), m in b.corners
        ],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
