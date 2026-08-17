"""Order a batch so the cache is warm when each task arrives.

The idea being borrowed
-----------------------
Offline inference engines get a large throughput win from **resource-aware
batching**: rather than running queued work in arrival order, group it so that
requests with similar resource profiles run together. Requests sharing a prompt
prefix are the clearest case -- run them consecutively and the prefix is
computed once; scatter them and it is recomputed every time the cache has
turned over in between.

The first pass over this idea rejected it, on the grounds that the engine here
belongs to somebody else so there is no utilisation to improve. That was the
wrong reason to stop. You do not control the batching, but you control the
**order you submit in**, and the prefix cache is charged to you by the token.
Ordering is free and the saving is real.

What this schedules
-------------------
A queue of deferrable tasks, each with a prefix it shares with some others -- a
repository, a document set, a system prompt, a long brief. Two orderings are
priced against each other:

* **arrival** -- run them as they were queued. Each task whose group has not been
  touched inside the TTL pays a full prefix write.
* **grouped** -- run all of a group's tasks consecutively. The first pays the
  write, the rest read it.

The difference is the ordering saving, and it is bounded by exactly one thing:
how much of the queue is shared prefix. A queue of unrelated one-off tasks has
nothing to gain and the report says so rather than inventing a small number.

Why the TTL is the whole story, and why the saving is not monotone in it
-------------------------------------------------------------------------
Grouping only helps inside a window, and the window has two edges:

* If the TTL is **shorter than the gap between consecutive tasks**, even a
  grouped run goes cold between its own members. Ordering changes nothing.
* If the TTL is **longer than the gap between scattered members of a group**,
  the prefix survives the interleaving anyway. Ordering changes nothing again.

The saving lives strictly between those two, and it is largest when the TTL sits
just above the grouped gap and just below the scattered one. So the ordering
saving is **not monotone in the TTL** -- it rises, peaks, and falls back to zero
-- which is the opposite of the intuition that a longer cache lifetime is always
worth more. The report therefore sweeps the TTL rather than quoting a single
number, because a single number is right only for one queue shape.

The practical reading: if the sweep is flat at zero, either your tasks are
further apart than the cache lives, or they are close enough together that the
cache survives without your help.

What this does not model
------------------------
Fairness or latency. Grouping delays whichever task happens to sort last, and
if that task had a deadline this is the wrong tool -- `adder deadline` is the one
that knows what a deadline is. This module answers a cost question and says
nothing about when any individual task finishes.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field

from adder.util import render
from adder.util.stats import share

M = 1_000_000.0

# Below this, a difference between two orderings is summation noise rather than
# money. A hundredth of a cent is already far below anything billable.
NOISE_USD = 1e-9


def _snap(x: float) -> float:
    return 0.0 if abs(x) < NOISE_USD else x

# Seconds a prefix survives unused. The shorter of the two published TTLs,
# because it is the default a workload gets without asking.
DEFAULT_TTL_S = 300.0

# Seconds one task occupies. Ordering only matters relative to the TTL, so this
# is the knob that decides whether a group can stay warm across its members.
DEFAULT_TASK_S = 60.0


@dataclass(frozen=True)
class Task:
    """One deferrable unit of work, and the prefix it shares."""

    key: str
    group: str              # tasks sharing a group share a prefix
    prefix_tokens: int      # the shared part
    own_tokens: int = 0     # the part unique to this task
    out_tokens: int = 500

    @property
    def total_in(self) -> int:
        return self.prefix_tokens + self.own_tokens


@dataclass
class Run:
    """What one ordering cost."""

    order: str
    prefix_writes: int = 0
    prefix_reads: int = 0
    cost: float = 0.0
    written_tokens: int = 0
    read_tokens: int = 0

    @property
    def reuse_rate(self) -> float:
        total = self.prefix_writes + self.prefix_reads
        return share(self.prefix_reads, total)


def order_arrival(tasks: list[Task]) -> list[Task]:
    """As queued. The baseline, and what happens if nobody thinks about it."""
    return list(tasks)


def order_grouped(tasks: list[Task]) -> list[Task]:
    """All of a group's tasks consecutively, largest prefix first.

    Largest first because a big shared prefix is where the saving is, and if
    the run is cut short -- a deadline, a cancelled batch -- the work that got
    done should be the work that was worth grouping.
    """
    by_group: dict[str, list[Task]] = {}
    for t in tasks:
        by_group.setdefault(t.group, []).append(t)
    groups = sorted(by_group.items(),
                    key=lambda kv: (-kv[1][0].prefix_tokens, kv[0]))
    return [t for _, members in groups for t in members]


def simulate(
    tasks: list[Task],
    *,
    ttl_s: float = DEFAULT_TTL_S,
    task_s: float = DEFAULT_TASK_S,
    in_rate: float,
    cache_read_rate: float,
    cache_write_rate: float,
    out_rate: float,
    order: str = "grouped",
) -> Run:
    """Price one ordering, tracking when each group's prefix was last touched.

    A group's prefix is readable only if some member of that group ran inside
    the TTL. That is the only state carried, because it is the only state that
    changes with the ordering -- the unique part of every task is paid for
    either way, and the output is paid for either way.
    """
    if order not in ("arrival", "grouped"):
        raise ValueError(f"unknown order {order!r}")
    sequence = order_arrival(tasks) if order == "arrival" else order_grouped(tasks)
    run = Run(order=order)
    last_seen: dict[str, float] = {}

    for i, t in enumerate(sequence):
        now = i * task_s
        warm = (t.group in last_seen) and (now - last_seen[t.group] <= ttl_s)
        if warm:
            run.prefix_reads += 1
            run.read_tokens += t.prefix_tokens
            run.cost += t.prefix_tokens * cache_read_rate / M
        else:
            run.prefix_writes += 1
            run.written_tokens += t.prefix_tokens
            run.cost += t.prefix_tokens * cache_write_rate / M
        # The unique part is never shared, so it is charged at the plain input
        # rate under either ordering. Including it keeps the totals comparable
        # to what a bill would say.
        run.cost += t.own_tokens * in_rate / M
        run.cost += t.out_tokens * out_rate / M
        last_seen[t.group] = now
    return run


@dataclass
class Report:
    arrival: Run | None = None
    grouped: Run | None = None
    tasks: int = 0
    groups: int = 0
    shared_tokens: int = 0
    total_in_tokens: int = 0
    ttl_s: float = DEFAULT_TTL_S
    task_s: float = DEFAULT_TASK_S
    sweep: list[tuple[float, float]] = field(default_factory=list)

    @property
    def saving(self) -> float:
        """Ordering saving, with float noise snapped to zero.

        The two orderings can be arithmetically identical and still differ by
        1e-16 through summation order. Reported raw that prints as `-0.0` and
        reads as "grouping made it worse", which is both false and the kind of
        detail that costs a reader their trust in the rest of the table.
        """
        if not (self.arrival and self.grouped):
            return 0.0
        return _snap(self.arrival.cost - self.grouped.cost)

    @property
    def saving_share(self) -> float:
        return share(self.saving, self.arrival.cost) if self.arrival else 0.0

    @property
    def shared_share(self) -> float:
        """How much of the queue's input is shared prefix. The ceiling."""
        return share(self.shared_tokens, self.total_in_tokens)

    @property
    def worth_ordering(self) -> bool:
        return self.saving > 0

    def to_json(self) -> dict:
        def row(r: Run | None) -> dict:
            if r is None:
                return {}
            return {"order": r.order, "cost_usd": r.cost,
                    "prefix_writes": r.prefix_writes,
                    "prefix_reads": r.prefix_reads, "reuse_rate": r.reuse_rate}

        return {
            "tasks": self.tasks,
            "groups": self.groups,
            "shared_share": self.shared_share,
            "ttl_s": self.ttl_s,
            "task_seconds": self.task_s,
            "arrival": row(self.arrival),
            "grouped": row(self.grouped),
            "saving_usd": self.saving,
            "saving_share": self.saving_share,
            "worth_ordering": self.worth_ordering,
            "ttl_sweep": [{"ttl_s": t, "saving_usd": v} for t, v in self.sweep],
        }


def analyse(
    tasks: list[Task],
    *,
    model: str = "claude-opus-5",
    ttl_s: float = DEFAULT_TTL_S,
    task_s: float = DEFAULT_TASK_S,
) -> Report:
    from adder.pricing.cost import Rates

    r = Rates.for_model(model)
    kw = {"in_rate": r.inp, "cache_read_rate": r.cache_read,
          "cache_write_rate": r.cache_write, "out_rate": r.out,
          "task_s": task_s}
    rep = Report(
        tasks=len(tasks),
        groups=len({t.group for t in tasks}),
        shared_tokens=sum(t.prefix_tokens for t in tasks),
        total_in_tokens=sum(t.total_in for t in tasks),
        ttl_s=ttl_s,
        task_s=task_s,
    )
    if not tasks:
        return rep
    rep.arrival = simulate(tasks, ttl_s=ttl_s, order="arrival", **kw)
    rep.grouped = simulate(tasks, ttl_s=ttl_s, order="grouped", **kw)
    for candidate in (60.0, 300.0, 1_800.0, 3_600.0):
        a = simulate(tasks, ttl_s=candidate, order="arrival", **kw)
        g = simulate(tasks, ttl_s=candidate, order="grouped", **kw)
        rep.sweep.append((candidate, _snap(a.cost - g.cost)))
    return rep


def load(path) -> list[Task]:
    """Read a queue from JSONL. Malformed lines are fatal, not skipped."""
    out: list[Task] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: not JSON ({exc.msg})") from exc
            if "group" not in d:
                raise ValueError(f"{path}:{lineno}: missing group")
            out.append(Task(
                key=str(d.get("key", lineno)),
                group=str(d["group"]),
                prefix_tokens=int(d.get("prefix_tokens", 0)),
                own_tokens=int(d.get("own_tokens", 0)),
                out_tokens=int(d.get("out_tokens", 500)),
            ))
    return out


def format_report(rep: Report) -> str:
    out: list[str] = []
    out += render.heading("blend — order the queue so the prefix stays warm",
                          rule="=")
    if not rep.tasks:
        out.append("  Empty queue. Nothing to order.")
        return "\n".join(out)

    out.append(render.kv("tasks", f"{rep.tasks:,} in {rep.groups:,} groups"))
    out.append(render.kv("shared prefix", render.pct(rep.shared_share) +
                         " of input tokens"))
    out.append(render.kv("cache TTL", f"{rep.ttl_s:,.0f}s, "
                                      f"{rep.task_s:,.0f}s per task"))
    out.append("")

    out += render.table(
        [[r.order, render.money(r.cost), f"{r.prefix_writes:,}",
          f"{r.prefix_reads:,}", render.pct(r.reuse_rate)]
         for r in (rep.arrival, rep.grouped) if r],
        ["order", "cost", "prefix writes", "prefix reads", "reuse"],
        align="<>>>>",
    )

    out.append("")
    if rep.worth_ordering:
        out += render.wrap(
            f"Grouping saves {render.money(rep.saving)} "
            f"({render.pct(rep.saving_share)}), entirely by not recomputing a "
            "prefix that was already paid for. Submission order is free to "
            "change and nothing else about the work moves.")
    elif rep.shared_share < 0.05:
        out += render.wrap(
            "Almost none of this queue is shared prefix, so no ordering can "
            "help. That is a property of the work, not a failure of the "
            "schedule.")
    else:
        out += render.wrap(
            "Ordering does not help here: at this TTL the prefix has expired "
            "before the next member of its group runs, so grouping buys nothing.")

    if rep.sweep:
        out.append("")
        out += render.heading("against the TTL")
        out += render.table(
            [[f"{int(t):,}s", render.money(v)] for t, v in rep.sweep],
            ["ttl", "ordering saves"], align="<>",
        )
        out += render.wrap(
            "The saving is not monotone in the TTL. Too short and even a grouped "
            "run goes cold between its own members; too long and the prefix "
            "survives the interleaving anyway. It peaks between the two, and a "
            "sweep that is flat at zero means this queue is on one side or the "
            "other of that window.")

    out.append("")
    out += render.wrap(
        "MODELLED, and it says nothing about latency: grouping delays whichever "
        "task sorts last. If any of this work has a deadline, price it with "
        "`adder deadline` before reordering anything.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    from pathlib import Path

    ap = argparse.ArgumentParser(
        prog="adder blend",
        description="Order a queue of deferrable work so shared prefixes stay "
                    "warm, and price what that ordering is worth.",
    )
    ap.add_argument("path", nargs="?", type=Path,
                    help="JSONL queue: group, prefix_tokens, own_tokens, out_tokens")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--ttl", type=float, default=DEFAULT_TTL_S,
                    help="seconds a prefix survives unused (default 300)")
    ap.add_argument("--task-seconds", type=float, default=DEFAULT_TASK_S,
                    help="seconds one task occupies (default 60)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    tasks: list[Task] = []
    if args.path is not None:
        if not args.path.exists():
            print(f"adder blend: no such file: {args.path}", file=sys.stderr)
            return 1
        try:
            tasks = load(args.path)
        except ValueError as exc:
            print(f"adder blend: {exc}", file=sys.stderr)
            return 2

    rep = analyse(tasks, model=args.model, ttl_s=max(0.0, args.ttl),
                  task_s=max(0.0, args.task_seconds))
    if args.json:
        print(json.dumps(rep.to_json(), indent=2, sort_keys=True))
    else:
        print(format_report(rep))
    return 0 if tasks else 1


if __name__ == "__main__":
    raise SystemExit(main())
