"""The expensive turns, and what made each one expensive.

An average is the wrong tool for finding waste. Spend here has a Gini around
0.7 -- a few turns and a few sessions hold most of the bill -- so the mean turn
is not where the money is, and a report of averages points at nothing.

Robust statistics, for a reason
-------------------------------
The obvious detector is a z-score. It does not work on this data: a single $40
turn inflates the standard deviation enough to make itself look ordinary, which
is the definition of a detector that fails on the case it exists for. Every
threshold here uses the median and the median absolute deviation instead.
Neither moves when one observation is extreme, so the outlier stays an outlier.

Naming the cause, not just the row
----------------------------------
"Turn 412 cost $3.10" is a fact nobody can act on. Each flagged turn is
compared against the one before it, and the first mechanism that explains the
jump is reported:

    prefix rebuild   cache writes dominated: the cache was invalidated or expired
    context jump     something large was admitted -- a read, a tool result, a paste
    long output      the model wrote a great deal in one turn
    fast mode        billed at 2x for latency
    big context      no single event; the context is simply large now

The order matters. A rebuild after a compaction looks like a context jump too,
and reporting the jump would send someone looking for a file that was never
read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date

from adder.core.trace import Session, Turn

# Robust-z above which a turn is worth naming. 3.5 on a MAD scale is roughly
# the classic outlier threshold; on this data it flags well under 1% of turns.
TURN_Z = 3.5
SESSION_Z = 3.0

# A rebuild has to be large to matter; a 200-token write is noise.
REBUILD_TOKENS = 10_000

# Context growth over one turn that counts as an admission event.
JUMP_TOKENS = 20_000


@dataclass(frozen=True)
class Finding:
    kind: str            # "turn" or "session"
    key: str             # session id, or "session:index"
    project: str
    cost: float
    z: float
    cause: str
    detail: str
    when: str = ""

    @property
    def excess(self) -> float:
        """Not reported as a saving. See `Report.excess`."""
        return self.cost


def _cause(turn: Turn, prev: Turn | None,
           on: date | None = None) -> tuple[str, str]:
    from adder.util.render import money, tokens

    if turn.cache_write > turn.cache_read and turn.cache_write >= REBUILD_TOKENS:
        if prev is None:
            # The first turn of a chain writes its whole prefix because there
            # was nothing to have cached, not because a cache was invalidated.
            # Calling that a rebuild sends someone hunting an invalidation that
            # never happened -- the same failure the cause ordering below exists
            # to avoid -- and prices an unavoidable write as pure overhead.
            # `trace.Session.cache_misses` and `cache.analyse` both already skip
            # turn 0 for this reason.
            return ("opening write",
                    f"{tokens(turn.cache_write)} written to open the context; "
                    f"there was no prior cache to read, so this is the cost of "
                    f"starting, not of losing one")
        from adder.pricing.cost import cache_miss_cost

        # Priced on the turn's own date, like the cost beside it in the same
        # finding. `on=None` means *today*, so a turn recorded before a rate
        # change was reported with a waste figure it was never billed.
        when = turn.pricing_date(on)
        waste = cache_miss_cost(turn.cache_write, turn.model, turn.ttl, when)
        # The read multiplier is this provider's, not Anthropic's 0.10x. On an
        # endpoint with no prompt cache the alternative to rewriting is paying
        # full input rate, and quoting a discount that does not exist names a
        # saving nobody can take.
        r = turn.rates(on)
        mult = (r.cache_read / r.inp) if r.inp else 1.0
        return ("prefix rebuild",
                f"{tokens(turn.cache_write)} rewritten at {turn.ttl} rates "
                f"instead of read at {mult:.2f}x — {money(waste)} of pure overhead")
    if prev is not None and turn.context - prev.context >= JUMP_TOKENS:
        return ("context jump",
                f"context grew {tokens(turn.context - prev.context)} in one turn "
                f"({tokens(prev.context)} → {tokens(turn.context)})")
    if turn.speed == "fast":
        return ("fast mode", "billed at 2x standard rates for latency")
    if turn.out >= 8_000:
        return ("long output",
                f"{tokens(turn.out)} of output, of which "
                f"{tokens(turn.thinking)} thinking")
    return ("big context",
            f"no single event: the context is simply {tokens(turn.context)} now")


@dataclass
class Report:
    turns: list[Finding]
    sessions: list[Finding]
    total: float
    n_turns: int

    @property
    def flagged_cost(self) -> float:
        return sum(f.cost for f in self.turns)

    @property
    def excess(self) -> float:
        """Cost of flagged turns ABOVE the median turn.

        The part that is unusual, rather than the whole bill for those turns.
        Reporting the full cost as if it were avoidable would over-state the
        opportunity: a flagged turn still had to happen, it just cost more than
        it should have.
        """
        return sum(max(0.0, f.cost - self.median_turn) for f in self.turns)

    median_turn: float = 0.0

    def by_cause(self) -> dict[str, tuple[int, float]]:
        """(turns, excess USD) per cause. The rows decompose `excess`.

        Excess, not full cost, for the reason `excess` gives: a flagged turn
        still had to happen, so charging its whole bill to the cause overstates
        what is available. It also has to be the same quantity the headline
        reports, and it was not -- the breakdown summed full cost under a
        headline of excess, so on this corpus a single row read $412.13 beneath
        a total of $408.09. A part larger than its whole is the kind of number
        that makes a reader stop trusting the rest of the report.
        """
        out: dict[str, tuple[int, float]] = {}
        for f in self.turns:
            n, c = out.get(f.cause, (0, 0.0))
            out[f.cause] = (n + 1, c + max(0.0, f.cost - self.median_turn))
        return dict(sorted(out.items(), key=lambda kv: -kv[1][1]))


def scan(sessions: dict[str, Session], *, on: date | None = None,
         turn_z: float = TURN_Z, session_z: float = SESSION_Z) -> Report:
    from adder.util.stats import median, robust_z_series

    costs: list[float] = []
    rows: list[tuple[Session, int, Turn, Turn | None, float]] = []
    for s in sessions.values():
        # `prev` is the previous turn *in the same context*. A subagent turn
        # belongs to a different window, so using it as the predecessor labels
        # the following main-chain turn a "context jump" when nothing was
        # admitted. Each chain is compared against itself.
        prev_main: Turn | None = None
        prev_side: Turn | None = None
        for i, t in enumerate(s.turns):
            c = t.cost(on)
            costs.append(c)
            prev = prev_side if t.sidechain else prev_main
            rows.append((s, i, t, prev, c))
            if t.sidechain:
                prev_side = t
            else:
                prev_main = t

    total = sum(costs)
    med = median(costs)
    turn_findings: list[Finding] = []
    if costs:
        # Scored in one pass. Calling `robust_z(c, costs)` per turn re-sorts the
        # whole series each time, which turned a one-second report into a
        # ninety-second one on 24,000 turns.
        zs = robust_z_series(costs)
        for (s, i, t, prev, c), z in zip(rows, zs, strict=True):
            if z < turn_z:
                continue
            cause, detail = _cause(t, prev, on)
            turn_findings.append(Finding(
                kind="turn", key=f"{s.id[:8]}#{i}", project=s.project,
                cost=c, z=z, cause=cause, detail=detail,
                when=t.ts or ""))
    turn_findings.sort(key=lambda f: -f.cost)

    ordered = list(sessions.values())
    per_turn = [s.cost / max(1, s.n_turns) for s in ordered]
    session_findings: list[Finding] = []
    if per_turn:
        med_pt = median(per_turn)
        for s, pt, z in zip(ordered, per_turn, robust_z_series(per_turn), strict=True):
            if z < session_z:
                continue
            session_findings.append(Finding(
                kind="session", key=s.id[:8], project=s.project, cost=s.cost, z=z,
                cause="cost per turn",
                detail=f"${pt:.3f}/turn against a median of "
                       f"${med_pt:.3f}, peak context {s.peak_context:,}",
                when=s.started.isoformat() if s.started else ""))
    session_findings.sort(key=lambda f: -f.z)

    # Every finding is kept. Truncation happens at display time only: a
    # `by cause` tally computed over the top 6 rows describes the top 6 rows,
    # not the workload, and reads as though it described the workload.
    return Report(turns=turn_findings, sessions=session_findings,
                  total=total, n_turns=len(costs), median_turn=med)


def report(rep: Report, *, top: int = 20) -> str:
    from adder.util.render import money, table

    if not rep.n_turns:
        return "  Nothing to analyse."
    lines = [f"  {rep.n_turns:,} turns · median {money(rep.median_turn)}/turn", ""]

    if not rep.turns:
        lines.append("  No turn stands out: spend is spread evenly enough that no")
        lines.append("  single turn is more than 3.5 robust deviations above the median.")
    else:
        shown = rep.turns[:top]
        lines.append(f"  {len(rep.turns):,} unusual turns, "
                     f"{money(rep.excess)} of which is above the median turn"
                     f"{f' (showing {len(shown)})' if len(shown) < len(rep.turns) else ''}:")
        lines.append("")
        rows = [[f.key, f.when[:10], f.project[-24:], money(f.cost),
                 f"{f.z:.0f}x", f.cause] for f in shown]
        lines += table(rows, ["turn", "date", "project", "cost", "z", "cause"],
                       align="<<<>><")
        lines.append("")
        lines.append("  by cause:")
        for cause, (n, c) in rep.by_cause().items():
            lines.append(f"    {cause:<16}{n:>4}  {money(c)}")
        lines.append("")
        worst = rep.turns[0]
        lines.append(f"  worst: {worst.key} — {worst.detail}")

    if rep.sessions:
        lines.append("")
        lines.append("  Sessions whose cost per turn is out of line:")
        rows = [[f.key, f.when[:10], f.project[-28:], money(f.cost), f"{f.z:.0f}x"]
                for f in rep.sessions[:8]]
        lines += table(rows, ["session", "date", "project", "cost", "z"],
                       align="<<<>>")
        lines.append("    A high cost per turn is a context-size problem, not a")
        lines.append("    length problem: restarting sooner is the lever, not routing.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from adder.core.filters import add_arguments as add_window
    from adder.core.filters import load as load_window

    ap = argparse.ArgumentParser(
        prog="adder anomaly",
        description="Find the turns and sessions that cost far more than the rest.")
    add_window(ap)
    ap.add_argument("--z", type=float, default=TURN_Z, metavar="N",
                    help="robust-z threshold for a turn (default: %(default)s)")
    ap.add_argument("--top", type=int, default=20, metavar="N",
                    help="rows to show (default: %(default)s)")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    a = ap.parse_args(argv)

    sessions, _window = load_window(a)
    rep = scan(sessions, turn_z=a.z)

    if a.json:
        print(json.dumps({
            "turns_examined": rep.n_turns,
            "median_turn_cost": round(rep.median_turn, 6),
            "excess_above_median": round(rep.excess, 4),
            "by_cause": {k: {"turns": n, "cost": round(c, 4)}
                         for k, (n, c) in rep.by_cause().items()},
            "turns": [
                {"key": f.key, "project": f.project, "when": f.when,
                 "cost": round(f.cost, 6), "z": round(f.z, 2),
                 "cause": f.cause, "detail": f.detail}
                for f in rep.turns[: a.top]
            ],
            "sessions": [
                {"key": f.key, "project": f.project, "cost": round(f.cost, 4),
                 "z": round(f.z, 2), "detail": f.detail}
                for f in rep.sessions[: a.top]
            ],
        }))
        return 0

    print()
    print(report(rep, top=a.top))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
