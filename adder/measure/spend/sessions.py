"""The drill-down `trace` only hinted at: one row per session, sortable.

`trace` reports that the top quarter of sessions hold three quarters of the
spend and then prints three of them. That is the right headline and the wrong
granularity for doing anything about it: the question after "spend is
concentrated" is always "in which sessions, and what was different about them".

What each row is for
--------------------
* **$/turn** separates a session that was expensive because it was long from
  one that was expensive per turn. They have different fixes -- the first is a
  restart cadence problem, the second is a context-size problem.
* **peak context** is the feasibility number. A session that peaked at 900K was
  never a candidate for a 200K model, whatever its price.
* **compactions** is the count of times the context collapsed. Each one is a
  full cache rebuild at 1.25x, and a session with several of them is paying the
  restart cost repeatedly without getting the fresh prefix a restart gives.
* **rebuilds** is cache-write-dominant turns after the first: prefix thrown
  away and paid for again.

Sorting is the whole interface. `--sort cost` finds the bill, `--sort per-turn`
finds the pathology, `--sort rebuilds` finds the cache problem.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date

from adder.core.filters import day_of
from adder.core.filters import root_of as _root_of
from adder.core.trace import Session

SORTS = ("cost", "per-turn", "turns", "context", "rebuilds", "compactions",
         "recent", "duration", "output")


@dataclass(frozen=True)
class Row:
    session: Session

    @property
    def cost(self) -> float:
        return self.session.cost

    @property
    def per_turn(self) -> float:
        return self.cost / max(1, self.session.n_turns)

    @property
    def rebuilds(self) -> int:
        return len(self.session.cache_misses())

    @property
    def compactions(self) -> int:
        return self.session.compactions()

    @property
    def rebuild_cost(self) -> float:
        """USD spent rewriting a prefix that could have been read from cache.

        Only the excess over a cache read is waste; the read would have been
        paid either way.

        Each turn is priced on the day it ran, like `Row.cost` beside it.
        `on=None` means *today*, so before this the two columns of the same row
        were quoted at different rates whenever an introductory price had
        expired between the session and the report.
        """
        from adder.pricing.cost import cache_miss_cost

        return sum(cache_miss_cost(t.cache_write, t.model, t.ttl,
                                   t.pricing_date())
                   for t in self.session.cache_misses())

    @property
    def when(self):
        return self.session.started

    def sort_key(self, by: str) -> float:
        s = self.session
        return {
            "cost": self.cost,
            "per-turn": self.per_turn,
            "turns": float(s.n_turns),
            "context": float(s.peak_context),
            "rebuilds": float(self.rebuilds),
            "compactions": float(self.compactions),
            "recent": self.when.timestamp() if self.when else 0.0,
            "duration": s.wall_seconds,
            "output": float(s.out_tokens),
        }[by]


def rank(sessions: dict[str, Session], by: str = "cost") -> list[Row]:
    if by not in SORTS:
        raise ValueError(f"unknown sort {by!r}; known: {', '.join(SORTS)}")
    return sorted((Row(s) for s in sessions.values()),
                  key=lambda r: -r.sort_key(by))


def report(sessions: dict[str, Session], *, by: str = "cost", top: int = 20,
           on: date | None = None) -> str:
    from adder.util.render import duration, money, table, tokens
    from adder.util.stats import gini

    rows = rank(sessions, by)
    if not rows:
        return "  No sessions to rank."
    total = sum(r.cost for r in rows)

    body = []
    for i, r in enumerate(rows[:top], 1):
        s = r.session
        body.append([
            i,
            day_of(r.when).isoformat() if r.when else "—",
            s.project[-30:],
            f"{s.n_turns:,}",
            money(r.cost),
            money(r.per_turn),
            tokens(s.peak_context),
            tokens(s.out_tokens),
            r.compactions or "",
            r.rebuilds or "",
            duration(s.wall_seconds) if s.wall_seconds else "—",
        ])
    lines = [f"  {len(rows):,} sessions · {money(total)} · sorted by {by}", ""]
    lines += table(body, ["#", "date", "project", "turns", "cost", "$/turn",
                          "peak ctx", "out", "cmpct", "rblds", "wall"],
                   align="><<>>>>>>>>")
    if len(rows) > top:
        rest = sum(r.cost for r in rows[top:])
        lines.append(f"    … {len(rows) - top:,} more, {money(rest)} "
                     f"({100 * rest / total:.0f}% of the total)")
    lines.append("")
    lines.append(f"  concentration {gini([r.cost for r in rows]):.2f} "
                 f"(0 = every session costs the same, 1 = one session is everything)")

    worst = max(rows, key=lambda r: r.rebuild_cost, default=None)
    if worst is not None and worst.rebuild_cost > 0.01:
        lines.append(f"  most cache rebuild waste: {money(worst.rebuild_cost)} in "
                     f"{worst.session.id[:8]} ({worst.rebuilds} rebuilt prefixes)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from adder.core.filters import add_arguments as add_window
    from adder.core.filters import load as load_window

    ap = argparse.ArgumentParser(
        prog="adder sessions",
        description="Rank sessions by cost, cost per turn, or cache damage.")
    add_window(ap)
    ap.add_argument("--sort", choices=SORTS, default="cost",
                    help="ranking key (default: %(default)s)")
    ap.add_argument("--top", type=int, default=20, metavar="N",
                    help="rows to show (default: %(default)s)")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    a = ap.parse_args(argv)
    # `root_of`: the argument if one was given, else the `root`
    # setting. Resolved here so two commands cannot disagree
    # about which transcript directory `adder config` names.
    a.root = str(_root_of(a))

    sessions, window = load_window(a)
    if not sessions:
        print(f"No sessions under {a.root} matching {window.describe()}.")
        return 1

    rows = rank(sessions, a.sort)
    if a.json:
        print(json.dumps({
            "sessions": len(rows),
            "total": round(sum(r.cost for r in rows), 4),
            "sort": a.sort,
            "rows": [
                {"id": r.session.id, "project": r.session.project,
                 "started": r.when.isoformat() if r.when else None,
                 "turns": r.session.n_turns,
                 "cost": round(r.cost, 4), "cost_per_turn": round(r.per_turn, 6),
                 "peak_context": r.session.peak_context,
                 "out_tokens": r.session.out_tokens,
                 "compactions": r.compactions, "rebuilds": r.rebuilds,
                 "rebuild_cost": round(r.rebuild_cost, 4),
                 "wall_seconds": round(r.session.wall_seconds, 1),
                 "models": sorted(r.session.models)}
                for r in rows[: a.top]
            ],
        }))
        return 0

    print()
    print(report(sessions, by=a.sort, top=a.top))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
