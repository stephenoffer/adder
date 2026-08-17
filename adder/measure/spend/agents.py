"""Delegation, as it actually happened rather than as it was recommended.

`policy` decides whether to delegate. `plan` prices a world in which you always
do. Neither looks at what you *did*, and the gap between the two is where a
routing tool quietly stops being true: the advice says "delegate reads over 5K
tokens", the transcripts say delegation is 0.5% of spend, and nobody notices
because no report joins them.

This is that join. It reads the sidechain records -- the turns Claude Code
writes for subagent runs -- and answers three questions:

1. **How much delegation is actually happening?** Share of spend, number of
   runs, tokens kept out of the main context.
2. **Is each subagent running on the right model?** A subagent starts with an
   empty context, so it has no warm cache to lose. That is the one place in a
   session where a cheaper model is unambiguously cheaper -- the argument that
   kills mid-session downgrades does not apply. A subagent run on Opus whose
   peak context fits in Haiku is paying 5x for nothing.
3. **What was not delegated that should have been?** Large single-turn context
   admissions on the main chain, priced against what a subagent would have
   cost.

A run, and why it is not a session
----------------------------------
Sidechain turns share the parent's session id; a "run" is a contiguous block of
them. Two subagents dispatched back to back are indistinguishable in the
transcript from one subagent taking twice as long, so a run is a lower bound on
the number of dispatches and its cost is exact either way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from adder.core import settings as _settings
from adder.core.filters import root_of as _root_of
from adder.core.trace import Session, Turn
from adder.pricing.prices import MODELS, fits, resolve
from adder.pricing.registry import fits as _fits
from adder.pricing.registry import limit_str

# Main-chain context growth in one turn that would have been worth delegating.
# Below this, the routing turn's own overhead eats the saving -- the same gate
# `policy.decide` applies before it will emit a recommendation.
DELEGABLE_TOKENS = 20_000


@dataclass
class Run:
    """One contiguous block of subagent turns inside a session."""

    session: str
    project: str
    turns: list[Turn] = field(default_factory=list)

    @property
    def model(self) -> str:
        return self.turns[0].model if self.turns else ""

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    def cost(self, on: date | None = None) -> float:
        return sum(t.cost(on) for t in self.turns)

    @property
    def peak_context(self) -> int:
        return max((t.context for t in self.turns), default=0)

    @property
    def out_tokens(self) -> int:
        return sum(t.out for t in self.turns)

    @property
    def summary_tokens(self) -> int:
        """What the run handed back: its final assistant message.

        The only part of a subagent's work that is admitted to the parent
        context, and therefore the only part that carries a lifetime cost.
        """
        return self.turns[-1].out if self.turns else 0

    @property
    def when(self):
        return self.turns[0].when if self.turns else None


def runs(sessions: dict[str, Session]) -> list[Run]:
    """Every subagent run, split by the agent that produced it.

    Keyed on `agentId`, not on contiguity. Subagent records carry the *parent's*
    session id, so a session's sidechain turns sit together in one block however
    many separate agents wrote them -- and a workflow that fans out does not
    interleave a main-chain turn between them to break the block up. Grouping by
    adjacency merged 119 subagents into 4 runs here, one of them 509 turns long
    with the context collapsing four times inside it, each collapse a new agent
    starting on a fresh context.

    That is not a cosmetic miscount. `summary_tokens` is the last turn's output
    -- the part the parent actually admits and carries -- so a merged run
    reported one summary and hid the other hundred-and-eighteen, and
    `peak_context` became the maximum across every agent, which is the number
    the feasibility gate in `cheaper_model` reads.

    Records with no `agentId` fall back to the old adjacency grouping, so a
    transcript written before the field existed still yields runs.
    """
    out: list[Run] = []
    for s in sessions.values():
        by_agent: dict[str, Run] = {}
        cur: Run | None = None            # the no-agentId fallback
        for t in s.turns:
            if not t.sidechain:
                cur = None
                continue
            if t.agent_id:
                run = by_agent.get(t.agent_id)
                if run is None:
                    run = by_agent[t.agent_id] = Run(s.id, s.project)
                    out.append(run)
                run.turns.append(t)
                continue
            if cur is None:
                cur = Run(s.id, s.project)
                out.append(cur)
            cur.turns.append(t)
    return out


def cheaper_model(run: Run, *, on: date | None = None,
                  floor: str | None = None) -> tuple[str | None, float]:
    """The cheapest model that could have run this subagent, and what it saves.

    A subagent has no warm prefix, so the cache-invalidation argument that
    protects a main-session model does not apply here: the only gates are
    whether the context fits and whether the model is actually cheaper.

    Capability is NOT modelled. This is an upper bound on the saving and it is
    reported as one -- a cheaper model that cannot do the task costs a retry,
    which is exactly what `outcomes` measures and `cost.escalation_is_profitable`
    prices. Use this to find candidates, not to make the switch blind.
    """
    from adder.pricing.cost import run_cost

    ctx, out = run.peak_context, run.out_tokens
    if not ctx:
        return None, 0.0
    current = run_cost(run.model, ctx, out, on)
    best, best_cost = None, current
    for mid in MODELS:
        if not fits(mid, ctx):
            continue
        if floor and resolve(mid).base.inp < resolve(floor).base.inp:
            continue
        c = run_cost(mid, ctx, out, on)
        if c < best_cost - 1e-12:
            best, best_cost = mid, c
    return best, max(0.0, current - best_cost)


@dataclass
class Missed:
    """A main-chain turn that admitted a lot at once."""

    session: str
    project: str
    tokens: int
    when: str
    inline: float
    delegated: float

    @property
    def saving(self) -> float:
        return self.inline - self.delegated


def missed(sessions: dict[str, Session], *, horizon=None,
           threshold: int = DELEGABLE_TOKENS,
           sub_model: str | None = None,
           on: date | None = None) -> list[Missed]:
    """Large main-chain admissions, priced inline against delegating them.

    `remaining_turns` comes from the horizon estimator conditioned on the turn
    index, not from a constant: the same 40K-token read is worth delegating on
    turn 20 of a long session and not worth the routing overhead on turn 900 of
    a short one.

    The conditional **median** is used, not the mean, and that is a deliberate
    departure from `horizon`'s own rule that the mean is what prices carry cost.
    The rule is right for a forward-looking gate, where the expectation is the
    quantity being bet on. This is a retrospective number -- "here is what you
    left on the table" -- and session length is heavy-tailed, so the mean sits
    well above the median and would make every missed read look larger than the
    typical case justifies. Over-claiming a saving that already cannot be
    collected is the one direction with no upside, so the smaller estimator
    wins here.
    """
    sub_model = sub_model or _settings.sub_model()
    from adder.measure.session.horizon import Horizon
    from adder.pricing.cost import placement_cost

    h = horizon if horizon is not None else Horizon.default()
    out: list[Missed] = []
    for s in sessions.values():
        prev = None
        for i, t in enumerate(s.turns):
            if t.sidechain:
                # A subagent runs in its own context window. Carrying its size
                # forward as `prev` makes the next main-chain turn look like a
                # huge admission -- or like a collapse -- when nothing was
                # admitted at all. The main chain has to be compared against
                # the main chain.
                continue
            grew = t.context - prev.context if prev is not None else 0
            prev = t
            if grew < threshold:
                continue
            remaining = h.remaining(i)
            inline, sub, _ = placement_cost(
                tokens_read=grew,
                summary_tokens=max(200, grew // 10),
                remaining_turns=remaining,
                main_model=t.model,
                sub_model=sub_model,
                on=on,
            )
            # Through the registry: `sub_model` comes from the `ladder`
            # setting and is routinely not a Claude id, and `prices.fits`
            # raises `UnknownModelError` for anything outside its own table.
            if not _fits(sub_model, grew + max(200, grew // 10) + 400):
                continue
            out.append(Missed(s.id, s.project, grew, t.ts or "", inline, sub))
    return sorted(out, key=lambda m: -m.saving)


@dataclass
class AgentReport:
    runs: list[Run]
    total_cost: float
    sidechain_cost: float
    missed: list[Missed] = field(default_factory=list)

    @property
    def share(self) -> float:
        return self.sidechain_cost / self.total_cost if self.total_cost else 0.0

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    def by_model(self, on: date | None = None) -> dict[str, tuple[int, float]]:
        out: dict[str, tuple[int, float]] = {}
        for r in self.runs:
            n, c = out.get(r.model, (0, 0.0))
            out[r.model] = (n + 1, c + r.cost(on))
        return dict(sorted(out.items(), key=lambda kv: -kv[1][1]))

    def downgradable(self, on: date | None = None) -> tuple[float, dict[str, int]]:
        """Total upper-bound saving from running each subagent on the cheapest
        model its context fits in, and which model each would move to."""
        total = 0.0
        moves: dict[str, int] = {}
        for r in self.runs:
            model, saving = cheaper_model(r, on=on)
            if model and saving > 0:
                total += saving
                moves[model] = moves.get(model, 0) + 1
        return total, moves


def analyse(sessions: dict[str, Session], *, horizon=None,
            on: date | None = None) -> AgentReport:
    total = sum(s.cost_on(on) for s in sessions.values())
    side = sum(t.cost(on) for s in sessions.values() for t in s.turns if t.sidechain)
    return AgentReport(runs=runs(sessions), total_cost=total, sidechain_cost=side,
                       missed=missed(sessions, horizon=horizon, on=on))


def report(rep: AgentReport, *, top: int = 10, on: date | None = None) -> str:
    from adder.util.render import money, table, tokens

    lines = ["  Delegation, as measured", ""]
    lines.append(f"  {rep.n_runs:,} subagent runs · {money(rep.sidechain_cost)} "
                 f"({rep.share:.1%} of {money(rep.total_cost)} total)")

    if not rep.runs:
        lines.append("")
        lines.append("  No subagent runs on record. Every read landed in a main")
        lines.append("  context and is being re-read on every turn after it.")
    else:
        lines.append("")
        lines.append("  by subagent model:")
        rows = [[m, f"{n:,}", money(c), money(c / n)]
                for m, (n, c) in rep.by_model(on).items()]
        lines += table(rows, ["model", "runs", "cost", "$/run"], align="<>>>")

        saving, moves = rep.downgradable(on)
        if saving > 0.005:
            lines.append("")
            lines.append(f"  {money(saving)} of that is recoverable by model choice alone.")
            lines.append("  A subagent starts cold, so it has no warm cache to lose — the")
            lines.append("  argument against downgrading a live session does not apply here.")
            for m, n in sorted(moves.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {n:>4} runs fit in {m} "
                             f"({limit_str(m)} window)")
            lines.append("  Upper bound: capability is not modelled. A tier that fails")
            lines.append("  costs a retry — see `adder outcomes`.")

        biggest = sorted(rep.runs, key=lambda r: -r.cost(on))[:top]
        lines.append("")
        lines.append("  most expensive runs:")
        rows = [[r.session[:8], r.project[-24:], f"{r.n_turns:,}",
                 tokens(r.peak_context), tokens(r.summary_tokens), money(r.cost(on))]
                for r in biggest]
        lines += table(rows, ["session", "project", "turns", "peak ctx",
                              "returned", "cost"], align="<<>>>>")

    if rep.missed:
        total = sum(m.saving for m in rep.missed if m.saving > 0)
        lines.append("")
        lines.append(f"  {len(rep.missed):,} main-chain turns admitted more than "
                     f"{DELEGABLE_TOKENS:,} tokens at once.")
        lines.append(f"  Delegating those reads would have saved at least "
                     f"{money(total)}:")
        rows = [[m.session[:8], m.when[:10], tokens(m.tokens), money(m.inline),
                 money(m.delegated), money(m.saving)]
                for m in rep.missed[:top] if m.saving > 0]
        lines += table(rows, ["session", "date", "admitted", "inline",
                              "delegated", "saving"], align="<<>>>>")
        lines.append("  Priced with the horizon estimator at each turn's own index,")
        lines.append("  so a late-session read is not credited with 400 re-reads it")
        lines.append("  would never have had — and at the conditional median rather")
        lines.append("  than the mean, which understates a heavy-tailed distribution")
        lines.append("  on purpose. This is a floor, not a forecast.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from adder.core import settings
    from adder.core.filters import Window
    from adder.core.filters import add_arguments as add_window
    from adder.core.trace import load_sessions
    from adder.measure.session.horizon import Horizon

    ap = argparse.ArgumentParser(
        prog="adder agents",
        description="Measure delegation: what ran as a subagent, and what should have.")
    add_window(ap)
    ap.add_argument("--top", type=int, default=10, metavar="N",
                    help="rows to show (default: %(default)s)")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    a = ap.parse_args(argv)
    # `root_of`: the argument if one was given, else the `root`
    # setting. Resolved here so two commands cannot disagree
    # about which transcript directory `adder config` names.
    a.root = str(_root_of(a))

    sessions = load_sessions(Path(a.root).expanduser(),
                             use_cache=bool(settings.get("cache")))
    horizon = Horizon.from_sessions(sessions)
    window = Window.from_args(a)
    if window.active:
        sessions = window.apply(sessions)
    rep = analyse(sessions, horizon=horizon)

    if a.json:
        saving, moves = rep.downgradable()
        print(json.dumps({
            "runs": rep.n_runs,
            "sidechain_cost": round(rep.sidechain_cost, 4),
            "total_cost": round(rep.total_cost, 4),
            "share": round(rep.share, 5),
            "by_model": {m: {"runs": n, "cost": round(c, 4)}
                         for m, (n, c) in rep.by_model().items()},
            "downgradable_saving": round(saving, 4),
            "downgrade_targets": moves,
            "missed": [
                {"session": m.session, "when": m.when, "tokens": m.tokens,
                 "inline": round(m.inline, 4), "delegated": round(m.delegated, 4),
                 "saving": round(m.saving, 4)}
                for m in rep.missed[: a.top]
            ],
        }))
        return 0

    print()
    print(report(rep, top=a.top))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
