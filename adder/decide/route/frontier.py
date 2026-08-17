"""The cost-quality frontier, with the honesty that the intervals force.

Why a frontier and not a ranking
--------------------------------
`adder pick` answers "what is the cheapest model that clears my quality bar".
That is the right question once you know the bar. Nobody knows the bar. What
people actually want to see is the shape of the trade-off: what does the next
dollar buy, and where does it stop buying anything.

That shape is the Pareto frontier — the models for which nothing else is both
cheaper and better. Everything off the frontier is dominated and should never
be chosen: there is a model that beats it on both axes. Printing a catalog of
four hundred models sorted by price hides that; printing the frontier is the
whole decision on one screen.

The part everyone gets wrong
----------------------------
A frontier computed on point estimates is fiction. Model A rated 1247 and model
B rated 1241 are not two points, they are two intervals that overlap almost
entirely, and calling A "better" makes B look dominated when the data cannot
tell them apart. Do that across a catalog and the frontier becomes a list of
whichever models got lucky on their last thousand votes.

So domination here is **statistical**: a model is only allowed to sit above a
cheaper one when its rating interval clears that model's. If the intervals
overlap, the quality difference is not measurable, and the cheaper model wins
outright.

The consequence runs the opposite way to the intuition. Taking the intervals
seriously makes the frontier *narrower*, not wider. A model rated six points
higher for four times the price does not survive contact with its own
confidence interval — on point estimates it sits on the frontier and invites
you to pay for the difference; on intervals it is dominated and gone. Those are
exactly the models a price-and-rating table talks you into buying.

The third axis nobody prices
----------------------------
Cost here is not the list price per million tokens. It is what running *this
task in this session* costs, from `select.cost_of`, which includes the carry
term — tokens admitted to a context get re-read on every remaining turn — and
respects the fact that a model that cannot hold the session context is not an
option at any price. A frontier drawn on list prices puts a cheap 32K-window
model at the bottom left, which is exactly where it does not belong when the
session is already 190K tokens deep.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from itertools import pairwise

from adder.pricing.catalog import Catalog, Entry
from adder.util import render

# Which arena board to read quality from. The coding board correlates with what
# an agent session does; the text board rewards prose, and routing a coding
# agent on it picks the wrong model confidently.
DEFAULT_BOARD = "webdev"


@dataclass(frozen=True)
class Node:
    """One model as a point on the frontier, with its interval."""

    entry: Entry
    cost: float
    rating: float
    lo: float
    hi: float
    placement: str = ""

    @property
    def id(self) -> str:
        return self.entry.id

    @property
    def measured(self) -> bool:
        """Whether the catalog published an interval, or only a point.

        A point estimate is treated as a zero-width interval, which is what the
        arithmetic below already does. The flag exists so the report can mark
        the row: a zero-width interval is an overconfident claim, and a reader
        deciding on it deserves to know the catalog never published one.
        """
        return self.hi > self.lo

    def beats(self, other: Node) -> bool:
        """Measurably better quality: the intervals do not overlap."""
        return self.lo > other.hi

    def dominates(self, other: Node) -> bool:
        """Pareto domination, with quality compared through the intervals.

        `self` dominates `other` when it is no worse on either axis and better
        on at least one, where "no worse on quality" means *not measurably
        worse* and "better on quality" means *measurably better*.

        The consequence is the useful part, and it is the opposite of what you
        might expect: taking the intervals seriously makes the frontier
        **narrower**, not wider. Consider a cheap model rated 1241 and an
        expensive one rated 1247, intervals overlapping. On point estimates
        neither dominates -- the expensive one "wins on quality" -- so both sit
        on the frontier and the reader is invited to pay for the difference. On
        intervals, the quality difference is not measurable, the cheap model is
        cheaper, and the expensive one is dominated and disappears.

        That is the whole value of doing it this way: it removes the models
        whose lead is noise, which are exactly the ones a price-and-rating
        table talks you into buying.
        """
        if self.cost > other.cost:
            return False
        if other.beats(self):
            return False        # measurably worse: cannot dominate at any price
        return self.cost < other.cost or self.beats(other)


@dataclass
class Frontier:
    nodes: list[Node] = field(default_factory=list)
    dominated: list[Node] = field(default_factory=list)
    board: str = DEFAULT_BOARD
    considered: int = 0
    unmeasured: int = 0
    # Eligible and rated, but the catalog publishes no usable price. Counted
    # rather than dropped: without this the totals do not add up, and a reader
    # cannot tell "we looked at 40 models" from "we looked at 40 and silently
    # discarded 12 of them".
    unpriced: int = 0

    @property
    def cheapest(self) -> Node | None:
        return min(self.nodes, key=lambda n: n.cost) if self.nodes else None

    @property
    def best(self) -> Node | None:
        return max(self.nodes, key=lambda n: n.rating) if self.nodes else None

    @property
    def all_nodes(self) -> list[Node]:
        return self.nodes + self.dominated

    def equivalent_to(self, model: str) -> list[Node]:
        """Every model whose quality is indistinguishable from `model`'s.

        Searched over everything considered, not just the survivors. That is
        deliberate: the models indistinguishable from the cheapest one are
        precisely the ones it dominated off the frontier, so restricting this
        to frontier members would always return nothing.

        This is the free-substitution set. If the data cannot separate them,
        the only remaining difference is price.
        """
        pool = self.all_nodes
        target = next((n for n in pool if n.id == model), None)
        if target is None:
            return []
        return sorted(
            (n for n in pool
             if n.id != model and not (n.beats(target) or target.beats(n))),
            key=lambda n: n.cost,
        )

    def to_json(self) -> dict:
        def row(n: Node) -> dict:
            return {"id": n.id, "org": n.entry.org, "cost_usd": n.cost,
                    "rating": n.rating, "lo": n.lo, "hi": n.hi,
                    "measured": n.measured, "placement": n.placement}

        return {
            "board": self.board,
            "considered": self.considered,
            "unmeasured": self.unmeasured,
            "unpriced": self.unpriced,
            "frontier": [row(n) for n in self.nodes],
            "dominated": [row(n) for n in self.dominated],
        }


def _rating(e: Entry, board: str) -> tuple[float, float, float] | None:
    """`(rating, lo, hi)` on `board`, or None when the catalog has nothing."""
    r = e.elo.get(board)
    if r is None:
        return None
    lo = e.elo_lo.get(board, r)
    hi = e.elo_hi.get(board, r)
    return (float(r), float(min(lo, r)), float(max(hi, r)))


def build(
    cat: Catalog,
    need,
    *,
    board: str = DEFAULT_BOARD,
    session: Entry | None = None,
) -> Frontier:
    """Price every eligible model for this task, then keep the undominated ones.

    O(n^2) in the number of eligible models, which is a few hundred at worst and
    therefore milliseconds. A sort-and-sweep would be O(n log n) and is wrong
    here: with statistical domination the relation is not a total order --
    A can fail to dominate B and B fail to dominate A -- so a sweep that assumes
    transitivity drops models that belong on the frontier.

    A cost of zero is a price, not a missing one. Free endpoints exist -- the
    catalog carries sixteen `:free` rows -- and a rated model that costs nothing
    is the single most interesting point on a cost-quality frontier, since it
    dominates everything at its rating or below. Testing `cost <= 0` filed it
    under "rated but unpriced" and dropped it.
    """
    from adder.decide.route.select import _eligible, cost_of

    fr = Frontier(board=board)
    nodes: list[Node] = []
    for e in _eligible(cat, need):
        fr.considered += 1
        rated = _rating(e, board)
        if rated is None:
            fr.unmeasured += 1
            continue
        if not e.priced:
            # `_eligible` filters on this already; the guard is here because the
            # test below used to be `cost <= 0`, which is a different question.
            fr.unpriced += 1
            continue
        costed = cost_of(e, need, session=session)
        if not costed.usable:
            # Neither placement exists for this model under this harness, so it
            # is not a point on any frontier. `select.rank` applies the same
            # guard; without it an unplaceable model arrived carrying an
            # infinite cost and was printed at the far end of the axis as
            # though it were merely expensive.
            fr.unpriced += 1
            continue
        cost = costed.best
        if cost < 0:
            fr.unpriced += 1
            continue
        rating, lo, hi = rated
        nodes.append(Node(e, cost, rating, lo, hi, costed.placement))

    for n in nodes:
        if any(other.dominates(n) for other in nodes if other is not n):
            fr.dominated.append(n)
        else:
            fr.nodes.append(n)
    fr.nodes.sort(key=lambda n: n.cost)
    fr.dominated.sort(key=lambda n: n.cost)
    return fr


def marginal(fr: Frontier) -> list[tuple[Node, Node, float]]:
    """What each step up the frontier costs per rating point.

    The number that ends an argument. Two models forty points apart for three
    cents is a different decision from two models four points apart for eight
    dollars, and a table of prices next to a table of ratings will not tell you
    which one you are looking at.
    """
    out = []
    ordered = sorted(fr.nodes, key=lambda n: n.cost)
    for a, b in pairwise(ordered):
        gain = b.rating - a.rating
        spend = b.cost - a.cost
        if gain > 0 and spend > 0:
            out.append((a, b, spend / gain))
    return out


def report(fr: Frontier, *, top: int = 12) -> str:
    out: list[str] = []
    out += render.heading(f"cost-quality frontier — {fr.board} board", rule="=")
    if not fr.nodes:
        out.append("  No model in the catalog has both a price and a rating for "
                   "this task. Try `adder models refresh`.")
        return "\n".join(out)

    out.append(render.kv("models considered", str(fr.considered)))
    out.append(render.kv("on the frontier", str(len(fr.nodes))))
    out.append(render.kv("dominated", str(len(fr.dominated))))
    if fr.unmeasured:
        out.append(render.kv("unrated on this board", str(fr.unmeasured)))
    if fr.unpriced:
        out.append(render.kv("rated but unpriced", str(fr.unpriced)))
    out.append("")

    rows = []
    for n in fr.nodes[:top]:
        interval = f"[{n.lo:.0f}, {n.hi:.0f}]" if n.measured else "no interval"
        rows.append([n.id[:34], n.entry.org[:12], render.money(n.cost),
                     f"{n.rating:.0f}", interval, n.placement])
    out += render.table(
        rows, ["model", "org", "task cost", "rating", "95% CI", "placement"],
        align="<<>><<",
    )

    steps = marginal(fr)
    if steps:
        out.append("")
        out += render.heading("what the next step buys")
        out += render.table(
            [[f"{a.id[:22]} → {b.id[:22]}", render.money(b.cost - a.cost),
              f"{b.rating - a.rating:+.0f}", render.money(rate)]
             for a, b, rate in steps[:8]],
            ["step", "extra cost", "rating", "$ per point"],
            align="<>>>",
        )

    cheapest = fr.cheapest
    if cheapest is not None:
        free = fr.equivalent_to(cheapest.id)
        out.append("")
        if free:
            names = ", ".join(n.id for n in free[:4])
            out += render.wrap(
                f"Quality-indistinguishable from {cheapest.id} on this board: "
                f"{names}. The data cannot separate them, so the only real "
                "difference is price — pick on cost and stop arguing about it.")
        else:
            out += render.wrap(
                f"{cheapest.id} is the cheapest option and every other frontier "
                "model is measurably better, so there is a real trade-off here "
                "rather than a free substitution.")

    out.append("")
    out += render.wrap(
        "MODELLED: arena preference is a proxy for agentic tool use, not a "
        "measurement of it. A model is only kept above a cheaper one when its "
        "rating interval clears it, so this frontier is NARROWER than one drawn "
        "on point estimates — the models it drops are the ones whose lead is "
        "inside the noise.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    from adder.decide.route.classify import difficulty_of
    from adder.decide.route.select import Need
    from adder.pricing.catalog import load

    ap = argparse.ArgumentParser(
        prog="adder frontier",
        description="The cost-quality Pareto frontier for a task, with "
                    "domination decided by confidence intervals.",
    )
    ap.add_argument("task", nargs="?", default="",
                    help="the task, used to size the context and difficulty")
    ap.add_argument("--board", default=DEFAULT_BOARD,
                    help=f"arena board to read quality from (default {DEFAULT_BOARD})")
    ap.add_argument("--context", type=int, default=100_000,
                    help="current session context in tokens")
    ap.add_argument("--turns", type=int, default=100,
                    help="turns the session still has to run")
    ap.add_argument("--read", type=int, default=40_000,
                    help="tokens the task pulls in")
    ap.add_argument("--open-weights", action="store_true",
                    help="restrict to open-weight models")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    need = Need(
        context_tokens=max(0, args.context),
        remaining_turns=max(1, args.turns),
        est_read_tokens=max(0, args.read),
        open_weights_only=args.open_weights,
        # The task's DIFFICULTY, not the classifier's confidence in its
        # answer. Those are different quantities and, in the case that
        # matters, close to inverses: an abstention carries confidence 0.3 and
        # routes the task UP, and 0.3 read as a difficulty is the easiest
        # setting there is -- so the tasks the classifier understood least were
        # given the widest quality tolerance and the smallest modelled Elo gap.
        difficulty=difficulty_of(args.task) if args.task else 1.0,
    )
    fr = build(load(), need, board=args.board)

    if args.json:
        print(json.dumps(fr.to_json(), indent=2, sort_keys=True))
    else:
        print(report(fr, top=max(1, args.top)))
    return 0 if fr.nodes else 1


if __name__ == "__main__":
    raise SystemExit(main())
