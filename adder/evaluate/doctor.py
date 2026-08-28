"""One command, run once, that says what to do next — ranked by dollars.

The tool has twenty-odd reports. That is the right number of reports and the
wrong number of things to run on a Monday morning. Someone who has just been
handed a bill does not know whether their problem is cache rebuilds, tool
output, session length, or model choice, and the honest answer is that it is
usually one of those four and never all of them.

So this runs the checks, prices each finding, and sorts by the price. Nothing
here computes anything new: every check delegates to the module that owns that
measurement, which is what keeps `doctor` from becoming a second, disagreeing
implementation of the cost model.

Ranking rule
------------
Findings are ordered by **dollars at stake**, not by severity or by how alarming
they sound. A 12% cache hit rate on a $30 workload is worth less attention than
a mild delegation gap on a $6,000 one, and a report that shouts about the first
teaches people to stop reading it.

Exit code
---------
`0` when nothing failed, `1` under `--strict` when any check did. The intended
use is a pre-push hook or a scheduled job, so the code has to mean something
stable: a check *fails* only when it is both actionable and above its dollar
floor. A finding worth eleven cents is reported and does not fail the run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# A finding below this is noise on any workload: reporting it wastes the
# reader's attention, and failing a build over it wastes their afternoon.
MIN_MATERIAL = 1.0

# Share of total spend at which a finding is material regardless of its
# absolute size. A $2 problem on a $10 workload is 20% and worth naming.
MATERIAL_SHARE = 0.02


@dataclass
class Check:
    name: str
    ok: bool
    headline: str
    action: str = ""
    dollars: float = 0.0          # at stake, not promised
    detail: list[str] = field(default_factory=list)
    skipped: bool = False

    @property
    def status(self) -> str:
        if self.skipped:
            return "SKIP"
        return "OK" if self.ok else "FIX"


def _material(dollars: float, total: float) -> bool:
    return dollars >= MIN_MATERIAL or (total > 0 and dollars / total >= MATERIAL_SHARE)


def check_spend(sessions, total: float) -> Check:
    from adder.util.render import money
    from adder.util.stats import gini

    costs = [s.cost for s in sessions.values()]
    g = gini(costs)
    turns = sum(s.n_turns for s in sessions.values())
    per_turn = total / turns if turns else 0.0
    return Check(
        "spend", True,
        f"{len(sessions):,} sessions · {turns:,} turns · {money(total)} "
        f"({money(per_turn)}/turn)",
        detail=[f"concentration {g:.2f} — "
                + ("a few sessions hold most of the bill; fix those, not the average"
                   if g > 0.5 else "spend is spread evenly across sessions")],
    )


def check_cache(sessions, total: float) -> Check:
    from adder.measure.window.cache import analyse
    from adder.util.render import money

    rep = analyse(sessions)
    waste, recoverable = rep.waste, rep.recoverable
    ok = not _material(recoverable, total)
    causes = ", ".join(f"{c} {money(v)}" for c, (n, v) in list(rep.by_cause().items())[:3]
                       if v > 0)
    return Check(
        "cache", ok,
        f"hit rate {rep.hit_rate:.0%} · {money(waste)} spent rebuilding prefixes",
        action=("`adder cache` — set a 1h TTL where idle expiry is the cause, and "
                "stop switching model mid-session"),
        dollars=recoverable,
        detail=[f"{money(recoverable)} of that is recoverable", causes] if causes else [],
    )


def check_tools(root, sessions, total: float) -> Check:
    from adder.measure.window.tools import LEVERS, carried_cost, scan
    from adder.util.render import money, tokens

    rep = scan(root)
    if not rep.calls:
        return Check("tools", True, "no tool calls on record", skipped=True)
    costs = carried_cost(rep, sessions)
    worst = max(rep.by_tool.values(), key=lambda t: costs.get(t.name, 0.0))
    share = rep.share_of_growth(worst)
    at_stake = costs.get(worst.name, 0.0)
    return Check(
        "tools", share < 0.40 or not _material(at_stake, total),
        f"{worst.name} is {share:.0%} of context growth, carrying "
        f"{money(at_stake)}",
        action=f"`adder tools` — {LEVERS.get(worst.name, 'bound the output at the call site')}",
        dollars=at_stake,
        detail=[f"largest single result {tokens(worst.biggest)}; "
                f"p90 {tokens(worst.p90_result())}"],
    )


def check_memory(sessions, total: float, *, repo=None) -> Check:
    """The always-loaded prefix: what it costs, and what of it is waste.

    Delegates to `memory` rather than re-deriving anything. The dollars at
    stake are only the *recoverable* part -- duplicated lines and over-long
    descriptions -- because the rest of `CLAUDE.md` is doing a job, and a check
    that puts a repository's whole instruction file on the board as "at stake"
    is a check that gets ignored.
    """
    from adder.measure.window.memory import analyse
    from adder.util.render import money, tokens

    rep = analyse(sessions, repo)
    at_stake = rep.pricing.window_cost(rep.recoverable_tokens)
    detail = [f"{f.kind}: {f.detail}" for f in rep.findings[:3]]
    if rep.floor_tokens:
        detail.append(
            f"{rep.controllable_share:.0%} of the opening context is yours to edit; "
            f"the rest is system prompt and tool schemas")
    return Check(
        "memory", not _material(at_stake, total),
        f"{tokens(rep.resident)} resident in every prefix · "
        f"{money(rep.pricing.per_1k())} per 1,000 tokens per session",
        action="`adder memory` — delete what is duplicated, shorten what is stale",
        dollars=at_stake,
        detail=detail,
    )


def check_reread(root, sessions, total: float) -> Check:
    """Content admitted to the context that was already in it."""
    from adder.measure.window.reread import (
        _carry,
        _session_shape,
        recoverable,
        scan,
    )
    from adder.util.render import money

    rep = scan(root)
    if not rep.admissions:
        return Check("reread", True, "no tool results on record", skipped=True)
    carry, shape = _carry(sessions), _session_shape(sessions)
    repeats = rep.with_repeats()
    files = rep.with_path_repeats()
    if rep.unpriced_shell():
        # Not `ok`, and not a dollar figure. This corpus reads through the
        # shell in a shape the parser cannot name, so the lever is unmeasured
        # rather than clear -- and "0 identities · $0.00" is the sentence that
        # gets believed, which on one corpus was $13 of $53 of Bash carry
        # reported as nothing.
        return Check(
            "reread", True,
            f"not measurable here — {rep.shell_results:,} shell results, none "
            "naming a file this parser could follow",
            action="`adder reread` — the lever is real; this corpus is what "
                   "cannot be read, so treat the $0.00 as unknown",
            dollars=0.0, skipped=True,
        )
    waste, n = recoverable(rep, shape, carry)
    worst = max(files, key=lambda p: p.unchanged_tokens, default=None)
    detail = []
    if worst:
        detail.append(f"worst file: {worst.path[-60:]} "
                      f"({worst.calls} reads via {'+'.join(worst.tools)})")
    top_ident = max(repeats, key=lambda r: r.redundant_tokens, default=None)
    if top_ident:
        detail.append(f"worst identity: {top_ident.ident[:70]} "
                      f"({top_ident.calls} calls)")
    return Check(
        "reread", not _material(waste, total),
        f"{n} results the context already held · {money(waste)} spent "
        f"re-reading them · {len(repeats)} repeated by call, {len(files)} by file",
        action="`adder reread` — the second copy of a file buys nothing; the "
               "first one never left",
        dollars=waste,
        detail=detail,
    )


def check_compact(sessions, total: float) -> Check:
    """Sessions that carried a full context past the point of compacting it."""
    from adder.measure.window.compact import analyse, breakeven_remaining
    from adder.util.render import money

    rep = analyse(sessions)
    missed = rep.missed_total()
    need = breakeven_remaining(read_mult=rep.read_mult, kept=rep.mean_kept())
    return Check(
        "compact", not _material(missed, total),
        f"{rep.n} compactions on record · {len(rep.misses)} sessions carried a "
        f"near-full context and never compacted ({money(missed)})",
        action=f"`adder compact` — compact when more than ~{need} turns remain, "
               "not when the bar looks full",
        dollars=missed,
        detail=[f"median compaction kept {rep.mean_kept():.0%} of the context"],
    )


def check_delegation(sessions, total: float) -> Check:
    from adder.measure.session.horizon import Horizon
    from adder.measure.spend.agents import analyse
    from adder.util.render import money

    rep = analyse(sessions, horizon=Horizon.from_sessions(sessions))
    at_stake = sum(m.saving for m in rep.missed if m.saving > 0)
    down, _ = rep.downgradable()
    return Check(
        "delegation", not _material(at_stake + down, total),
        f"{rep.n_runs:,} subagent runs, {rep.share:.1%} of spend · "
        f"{len(rep.missed):,} large reads went inline",
        action="`adder agents` — delegate reads over 20K tokens; a subagent has no "
               "warm cache to lose",
        dollars=at_stake + down,
        detail=([f"{money(down)} recoverable by running subagents on a model that "
                 f"fits their context"] if down > 0.01 else []),
    )


def check_anomalies(sessions, total: float) -> Check:
    from adder.measure.spend.anomaly import scan
    from adder.util.render import money

    rep = scan(sessions)
    if not rep.turns:
        return Check("anomalies", True, "no turn is far above the median")
    top_cause = next(iter(rep.by_cause()))
    return Check(
        "anomalies", not _material(rep.excess, total),
        f"{len(rep.turns):,} unusual turns, {money(rep.excess)} above the median turn",
        action=f"`adder anomaly` — the dominant cause is {top_cause}",
        dollars=rep.excess,
        detail=[f"{cause}: {n} turns, {money(c)}"
                for cause, (n, c) in list(rep.by_cause().items())[:3]],
    )


def check_quality(root, sessions, total: float) -> Check:
    """The half of the thesis a cost report cannot see.

    Every lever in this repo trades tokens for something, and a degraded agent
    often looks *cheaper* per turn while taking more turns to finish. So a
    health check that only looked at money would be recommending exactly the
    changes it cannot evaluate.

    Only the tool error rate is gated, because it is the one proxy with a
    defensible absolute threshold: a failed call still costs a full turn, and
    its error text still enters the context and is re-read for the rest of the
    session. The others are reported without a verdict -- they only mean
    something compared against themselves before and after a change, which is
    what `adder quality --since DATE` is for.
    """
    from adder.measure.session.quality import scan

    q = scan(root)
    if not q.tool_calls:
        return Check("quality", True, "no tool calls to judge", skipped=True)

    turns = sum(s.n_turns for s in sessions.values())
    per_turn = total / turns if turns else 0.0
    # A failed call still bought a turn. This prices the turns, not the fix.
    wasted = q.tool_errors * per_turn

    detail = [
        f"correction rate {q.correction_rate:.1%} · "
        f"turns per prompt {q.turns_per_prompt:.1f} · "
        f"rework {q.rework_ratio:.2f} edits per file",
    ]
    if q.api_errors:
        detail.append(f"{q.api_errors:,} client-side API failures "
                      f"({q.api_error_rate:.2%} of turns)")
    detail.append("these only mean something before/after a change: "
                  "`adder quality --since DATE`")

    ok = q.tool_error_rate <= 0.10 or q.tool_calls < 50
    return Check(
        "quality", ok,
        f"tool error rate {q.tool_error_rate:.1%} "
        f"({q.tool_errors:,} of {q.tool_calls:,} calls)",
        action="`adder tools` names which tool is failing — a failed call costs "
               "a whole turn and leaves its error in the context",
        dollars=wasted if not ok else 0.0,
        detail=detail,
    )


# How much recorded history the machine-fitted parts of this tool need before
# they are describing this workload rather than the author's. Not a
# statistical threshold -- it is the point below which a week's work is one
# project and a fitted number is a fact about that project.
MIN_HISTORY_DAYS = 14
# How far a local p90 may be from the shipped prior before the prior is a
# fiction on this machine. Two-fold either way, the same band `adder guard`
# already draws its `Nx out` verdict at.
PRIOR_BAND = 2.0


def check_history(root, sessions) -> Check:
    """How much of what this tool tells you is fitted to you, and how much is shipped.

    Every number here that adapts -- the size model behind the guard, `p_fail`
    behind every escalation gate, the uptake term behind every advisory dollar,
    the trend `savings` reads as a trend -- needs weeks of transcripts to mean
    anything. On a fresh machine each of them silently falls back to a prior
    measured on one workload, and the report reads identically either way.
    That is the same failure this project calls its worst everywhere else: a
    confident number with nothing behind it.

    So this check says the quiet part. It does not fail the build for being new
    -- being new is not a defect -- but it names which features are currently
    describing somebody else's machine.
    """
    from adder.core.filters import span
    from adder.core.shapes import load_model
    from adder.decide.guard import Settings
    from adder.decide.guard import uptake as guard_uptake

    lo, hi = span(sessions)
    days = (hi - lo).days if (lo and hi) else 0
    shipped: list[str] = []

    sizes = load_model()
    if not sizes.calls:
        shipped.append("the guard's result-size model (`adder guard --learn`)")
    try:
        u = guard_uptake(root)
    except Exception:
        u = None
    cfg = Settings.resolve()
    if u is not None and not u.measured and not cfg.shadowing:
        shipped.append(f"the {cfg.advice_taken:.0%} uptake every advisory dollar "
                       "is multiplied by (`adder auto on --shadow` measures it)")
    try:
        from adder.decide.track.outcomes import load as load_outcomes

        if len(load_outcomes()) < 12:
            shipped.append("`p_fail`, behind every escalation gate "
                           "(`adder outcomes import --write`)")
    except Exception:
        pass

    if days >= MIN_HISTORY_DAYS and not shipped:
        return Check("history", True,
                     f"{days} days of transcripts; every fitted number is your own")
    if not shipped:
        return Check("history", True,
                     f"{days} days of transcripts — thin, but everything fitted "
                     "here is fitted to it", skipped=True)
    return Check(
        "history", False,
        f"{days} day{'s' if days != 1 else ''} of transcripts; "
        f"{len(shipped)} number{'s' if len(shipped) > 1 else ''} here "
        "come from the shipped prior, not from you",
        action="`adder guard --learn` and `adder outcomes import --write` fit "
               "what can be fitted today; the rest needs elapsed time",
        detail=shipped,
    )


def check_prior(root) -> Check:
    """Where the shipped prior is wrong about this machine, tool by tool.

    `adder guard` already prints this table, and printing it there was the
    right instinct in the wrong place: it is a finding, not a column. On the
    machine this was written for it showed `Agent` at 13.5x out -- a tool whose
    every prediction was off by more than an order of magnitude, in a report
    nobody opens unless they already suspect the guard.
    """
    from adder.core.shapes import PRIOR, load_model
    from adder.util.render import tokens

    sizes = load_model()
    if not sizes.calls:
        return Check("prior", True, "no local size model to compare against",
                     skipped=True)
    out = []
    for tool in sorted(sizes.tools):
        local = sizes.tools[tool]
        if local[2] < 3 or tool not in PRIOR:
            continue
        shipped, mine = PRIOR[tool][1], local[1]
        if not mine:
            continue
        ratio = shipped / mine
        if ratio > PRIOR_BAND or ratio < 1 / PRIOR_BAND:
            out.append(f"{tool}: prior {tokens(shipped)} p90, yours "
                       f"{tokens(mine)} over {local[2]:,} calls — {ratio:,.1f}x out")
    if not out:
        return Check("prior", True,
                     f"the shipped size prior is within {PRIOR_BAND:.0f}x of this "
                     "machine on every tool it covers")
    return Check(
        "prior", False,
        f"the shipped size prior is more than {PRIOR_BAND:.0f}x out on "
        f"{len(out)} tool{'s' if len(out) > 1 else ''}",
        action="`adder guard --learn` — until it runs, the guard is predicting "
               "these from one machine's workload, and it is not this one",
        detail=out,
    )


def check_horizon(sessions) -> Check:
    from adder.measure.session.horizon import Horizon

    h = Horizon.from_sessions(sessions)
    if len(h.lengths) < 5:
        return Check("horizon", True,
                     "not enough sessions to fit a remaining-turns estimate",
                     skipped=True)
    med, mean = h.remaining(100), h.mean_remaining(100)
    return Check(
        "horizon", True,
        f"at turn 100, ~{med:,} turns typically remain (mean {mean:,.0f})",
        detail=["the mean is what prices carry cost; the median is what to tell "
                "a person"],
    )


def check_prices(on: date | None = None) -> Check:
    """Intro rates expire. A threshold tuned under one is wrong the day it ends."""
    from adder.pricing.prices import expiring_soon

    soon = [f"{mid} reverts to ${base.inp}/${base.out} from "
            f"${intro.inp}/${intro.out} on {when}"
            for mid, when, intro, base in expiring_soon(30, on)]
    if not soon:
        return Check("prices", True, "no introductory rate expires in the next 30 days")
    return Check(
        "prices", False,
        f"{len(soon)} model price change{'s' if len(soon) > 1 else ''} within 30 days",
        action="re-run `adder plan` after the change; thresholds tuned against an "
               "intro rate move with it",
        detail=soon,
    )


def check_catalog() -> Check:
    from adder.pricing.catalog import load

    try:
        cat = load()
    except Exception:
        return Check("catalog", True, "catalog unavailable", skipped=True)
    age = cat.age_days()
    if age is None:
        return Check("catalog", True, f"{len(cat):,} models, age unknown", skipped=True)
    stale = cat.is_stale()
    return Check(
        "catalog", not stale,
        f"{len(cat):,} models, snapshot {age:.0f} days old",
        action="`adder models refresh` — the only networked command",
    )


def check_guard(root=None) -> Check:
    """Is the one mechanism that can prevent spend actually able to?

    Everything else in `doctor` reports on money already spent. This check is
    the only one about money that has not been spent yet, and it exists because
    the guard's failure mode is silence: an uninstalled hook, a size model that
    was never learned, and a correctly working guard all produce exactly the
    same experience.
    """
    from adder.core.shapes import load_model
    from adder.decide.guard import Settings, installed_in
    from adder.decide.guard import ledger as guard_ledger
    from adder.util.render import money

    cfg = Settings.resolve()
    sizes = load_model()
    if not installed_in():
        # Reported as a failure, because it is the only finding in `doctor`
        # about money that has not been spent yet. Everything else here is a
        # post-mortem.
        return Check(
            "guard", False,
            "not installed — nothing is preventing spend, only measuring it",
            action="`adder auto on` — installs the hooks and says what it will "
                   "change first; the hook is the only component that runs "
                   "while a read is still cancellable",
        )
    if not cfg.enforcing:
        # A separate finding from "not installed", because the two have
        # different fixes and very different values. Installed-but-advisory is
        # the configuration whose entire saving is multiplied by a guess about
        # whether advice gets taken; measured over 34,144 recorded calls,
        # enforcing moves the net from $93 to $513 and the share resting on
        # that guess from 100% to 4%.
        return Check(
            "guard", False,
            "installed but advisory — every dollar it claims is multiplied by "
            f"an assumed {cfg.advice_taken:.0%} chance the advice is taken",
            action="`adder auto on --full` — lets it refuse the calls that "
                   "admit nothing new, so the saving stops being an assumption",
        )
    if not sizes.calls:
        return Check(
            "guard", False,
            "no size model — every prediction falls back to the shipped prior",
            action="`adder guard --learn` — derive result sizes from your own "
                   "transcripts; the shipped prior is one machine's workload",
        )

    led = guard_ledger(cfg.state_path)
    if not led["fires"]:
        return Check("guard", True,
                     f"size model current ({sizes.calls:,} calls, "
                     f"{len(sizes.shapes):,} shapes); nothing flagged yet",
                     skipped=True)

    net = led["saving"] * cfg.advice_taken - led["overhead"]
    if net < 0:
        return Check(
            "guard", False,
            f"the guard has cost {money(led['overhead'])} in advice and is worth "
            f"{money(led['saving'] * cfg.advice_taken)} at {cfg.advice_taken:.0%} uptake",
            action="`adder guard` — raise guard_min_cost, or lower "
                   "guard_max_fires, until it clears its own overhead",
            dollars=-net,
        )
    from adder.decide.guard import uptake as guard_uptake

    # The root this report was pointed at, not the default. Every other check
    # in `run()` reads the corpus it was given; this one silently measured
    # uptake against `~/.claude/projects`, so `adder doctor <a-corpus>` mixed
    # two different workloads into one report.
    u = guard_uptake(root)
    basis = (f"a measured {u.rate:.0%} uptake" if u.measured
             else f"{cfg.advice_taken:.0%} assumed uptake")
    prevented = float(led.get("prevented") or 0.0)
    detail = [f"solvent at {basis}; net {money(net + prevented)}", u.describe()]
    if prevented:
        # Reported first and separately: it is the only figure in this check
        # that no assumption is applied to.
        detail.insert(0, f"{money(prevented)} of it is calls that did not "
                         f"happen, which no uptake term applies to")
    return Check(
        "guard", True,
        f"{led['fires']:,} findings worth {money(led['saving'])} promised for "
        f"{money(led['overhead'])} of advice",
        detail=detail,
    )


def check_ledger() -> Check:
    from adder.decide.track.ledger import current
    from adder.util.render import money

    try:
        led = current()
    except Exception:
        return Check("ledger", True, "no ledger", skipped=True)
    if not led.entries:
        return Check("ledger", True,
                     "no recommendations booked yet — `adder policy --record`",
                     skipped=True)
    return Check(
        "ledger", led.solvent,
        f"{money(led.delivered)} delivered against {money(led.spent)} of asking",
        action="`adder ledger` — the advice has not paid for the turns spent giving it",
    )


def check_evidence() -> Check:
    from adder.decide.track.outcomes import load

    try:
        rows = load()
    except Exception:
        return Check("evidence", True, "no outcome log", skipped=True)
    imported = sum(1 for r in rows if getattr(r, "source", "") == "transcript")
    if len(rows) < 12:
        return Check(
            "evidence", True,
            f"{len(rows)} dispatch outcomes recorded — too few to calibrate p_fail",
            action="`adder outcomes import --write` backfills this from your "
                   "transcripts; without it every escalation gate runs on a prior",
            skipped=True)
    note = f"{imported:,} of them imported from transcripts" if imported else ""
    return Check("evidence", True, f"{len(rows):,} dispatch outcomes recorded",
                 detail=[note] if note else [])


def check_budget(sessions) -> Check:
    from adder.core import settings
    from adder.measure.spend.budget import measure
    from adder.util.render import money

    limit = float(settings.get("budget"))
    if not limit:
        return Check("budget", True, "no budget set (`ADDER_BUDGET`)", skipped=True)
    b = measure(sessions, limit=limit)
    return Check(
        "budget", not b.over,
        f"{money(b.spent)} of {money(limit)} this month, projecting {money(b.projection)}",
        action="`adder budget` — the projection is over the limit",
        dollars=max(0.0, b.projection - limit),
    )


def run(root: Path | str, sessions=None, *, on: date | None = None) -> list[Check]:
    """Every check, in dollar order, with the informational ones last."""
    from adder.core.trace import load_sessions

    if sessions is None:
        sessions = load_sessions(Path(root).expanduser())
    total = sum(s.cost_on(on) for s in sessions.values())

    checks = [
        check_spend(sessions, total),
        check_cache(sessions, total),
        check_tools(root, sessions, total),
        check_memory(sessions, total),
        check_reread(root, sessions, total),
        check_compact(sessions, total),
        check_delegation(sessions, total),
        check_anomalies(sessions, total),
        check_quality(root, sessions, total),
        check_budget(sessions),
        check_prices(on),
        check_catalog(),
        check_guard(root),
        check_history(root, sessions),
        check_prior(root),
        check_ledger(),
        check_evidence(),
        check_horizon(sessions),
    ]
    # Actionable and expensive first; skipped last. A stable order matters
    # because this output gets diffed between runs.
    return sorted(checks, key=lambda c: (c.skipped, c.ok, -c.dollars, c.name))


def report(checks: list[Check]) -> str:
    from adder.util.render import money

    fixes = [c for c in checks if not c.ok and not c.skipped]
    at_stake = sum(c.dollars for c in fixes)
    lines = ["  adder doctor", ""]
    for c in checks:
        mark = {"OK": "  ok  ", "FIX": " FIX  ", "SKIP": " --   "}[c.status]
        amount = f"  {money(c.dollars)}" if c.dollars >= 0.01 else ""
        lines.append(f"  {mark}{c.name:<12}{c.headline}{amount}")
        for d in c.detail:
            if d:
                lines.append(f"                 {d}")
    lines.append("")
    if not fixes:
        lines.append("  Nothing material to fix. The expensive levers here are already")
        lines.append("  either pulled or not worth pulling on this workload.")
        return "\n".join(lines)

    lines.append(f"  {len(fixes)} finding{'s' if len(fixes) > 1 else ''}, "
                 f"{money(at_stake)} at stake, most expensive first:")
    lines.append("")
    for i, c in enumerate([c for c in fixes if c.action], 1):
        lines.append(f"  {i}. {c.name} — {money(c.dollars)}")
        lines.append(f"     {c.action}")
    lines.append("")
    lines.append("  `at stake` is what the measurement says is addressable, not a")
    lines.append("  promise. Levers overlap: fixing two does not save the sum of both.")
    lines.append("  `adder plan` prices a whole regime with the overlap removed.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from adder.core import settings
    from adder.core.filters import Window
    from adder.core.filters import add_arguments as add_window
    from adder.core.trace import load_sessions

    ap = argparse.ArgumentParser(
        prog="adder doctor",
        description="Run every check and rank the findings by dollars at stake.")
    add_window(ap)
    ap.add_argument("--json", action="store_true", help="machine-readable")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any check failed (for hooks and CI)")
    from adder.core.filters import root_of as _root_of

    a = ap.parse_args(argv)
    # `root_of`: the argument if one was given, else the `root`
    # setting. Resolved here so two commands cannot disagree
    # about which transcript directory `adder config` names.
    a.root = str(_root_of(a))

    root = Path(a.root).expanduser()
    sessions = load_sessions(root, use_cache=bool(settings.get("cache")))
    window = Window.from_args(a)
    if window.active:
        sessions = window.apply(sessions)
    if not sessions:
        # The first thing a new install prints, if it prints anything at all.
        # It used to be this sentence and nothing else, which reads as a broken
        # tool rather than an empty one: every check here measures a transcript
        # you have already paid for, and a machine with no history has none. The
        # part that needs no history is the part that prevents spend instead of
        # reporting it, so say so here rather than in the README only.
        print(f"No sessions under {root} yet.\n")
        if window.active:
            print(f"  Nothing matched {window.describe()}. Try it without the "
                  "window flags.\n")
        else:
            print("  Every check here reads a transcript you have already paid "
                  "for, so on a\n  fresh machine there is nothing to measure. "
                  "What does not need history is the\n  half that prevents "
                  "spend rather than reporting it:\n")
            print("      adder auto on --full\n")
            print("  That installs the hooks that price a call before its "
                  "result lands in your\n  context. Run this again after a "
                  "session or two and it will have numbers.\n")
            print(f"  Transcripts somewhere else? `adder doctor <dir>`, or set "
                  f"`root` in\n  {settings.PROJECT_FILE} (`adder config --init` "
                  "prints a template).\n")
        return 1

    checks = run(root, sessions)
    failed = [c for c in checks if not c.ok and not c.skipped]

    if a.json:
        print(json.dumps({
            "at_stake": round(sum(c.dollars for c in failed), 4),
            "failed": [c.name for c in failed],
            "checks": [
                {"name": c.name, "status": c.status, "ok": c.ok,
                 "skipped": c.skipped, "headline": c.headline,
                 "action": c.action, "dollars": round(c.dollars, 4),
                 "detail": c.detail}
                for c in checks
            ],
        }))
    else:
        print()
        print(report(checks))
        print()
    return 1 if (a.strict and failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
