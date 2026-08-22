"""Delegations as the caller issued them, recovered from the transcripts.

The adaptive half of this tool has never run. `outcomes` needs a measured
`p_fail` per tier; the only way to write that log was `adder outcomes record`
after every dispatch, by hand; nobody does that; so the log is empty on every
machine, `p_fail` sits on its 0.5 prior forever, and the router is permanently
forbidden from preferring a cheaper tier. A feature that requires a discipline
nobody keeps is a feature that does not exist.

The evidence was on disk the whole time. A delegation is an `Agent` tool_use
block naming a `subagent_type`; its outcome is the `tool_result` that answers
it. This module recovers both and `outcomes import` turns them into log rows,
so the calibration accumulates from working normally rather than from
remembering to file a report.

Two views, deliberately separate
--------------------------------
`agents.py` reads **sidechain** records: what the subagent did, on which model,
at what cost. This reads the **caller's** side: what was asked for, of which
agent, and what came back. They are joined here only for enrichment (cost,
model), and the join is positional -- the Nth dispatch in a session matched to
the Nth sidechain run -- because the transcript carries no explicit link. The
escalation signal never depends on that match.

What this can and cannot see
----------------------------
Observable: an error result, and the `ESCALATE:` marker the tier agents are
instructed to return. Both are unambiguous.

Not observable: a subagent that returned a confident, wrong answer and was
believed. That failure looks exactly like a success from here -- and it looks
exactly like a success to a human filing a manual report too, so importing does
not make the estimate worse than hand-recording. It does mean the derived rate
is a **lower bound** on how often a tier is inadequate, which matters because
under-estimating `p_fail` is the expensive direction. `adder ab` is the only
thing in this repo that can measure answer quality directly.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from adder.core.trace import DEFAULT_ROOT, transcripts
from adder.util.records import mapping

# Tools that dispatch work to a fresh context. `Task` is the older name.
DISPATCH_TOOLS = ("Agent", "Task")

# The marker the tier agents in `.claude/agents/` are told to return when they
# cannot finish. Matched at the start of a line so a subagent *discussing*
# escalation in prose does not register as one.
_ESCALATE = re.compile(r"^\s*ESCALATE\b[:\s]", re.M)

# Agents this repo ships, mapped to the tier they represent. `Explore` is the
# read-only Haiku agent, which is a T0 by construction.
AGENT_TIERS: dict[str, str] = {
    "route-t0": "T0",
    "route-t1": "T1",
    "route-t2": "T2",
    "explore": "T0",
}


def tier_for_model(model: str) -> str:
    """The tier a model represents, from the ladder in `classify`.

    Reversed from `classify.LADDER` rather than hardcoded, so a project that
    rebinds a rung gets the tier it configured instead of the one that was true
    when this was written.

    Exact id first, then equal list price. A tier here is a *cost* tier -- that
    is the whole reason the escalation gate exists -- so a run on
    `claude-opus-4-8` belongs on the same rung as one on `claude-opus-5`: same
    $5/$25, same arithmetic, same decision. Matching ids alone left every run on
    a previous generation untiered and therefore uncounted, which is how a log
    stays empty while the evidence sits on disk.

    Base rates are compared, never dated ones. An introductory price makes a
    model temporarily cheaper without moving it to a different rung, and a tier
    map that reshuffles itself on an expiry date is worse than no tier map.
    """
    from adder.decide.route.classify import ladder as _ladder

    LADDER = _ladder()
    from adder.pricing.prices import is_known, resolve

    if not model or not is_known(model):
        return ""
    target = resolve(model)
    for tier, mid in LADDER.items():
        try:
            if resolve(mid).id == target.id:
                return tier
        except Exception:
            continue
    for tier, mid in LADDER.items():
        try:
            if resolve(mid).base == target.base:
                return tier
        except Exception:
            continue
    return ""


@dataclass
class Dispatch:
    """One delegation, from the request to whatever came back."""

    session: str
    project: str
    use_id: str
    agent_type: str
    description: str = ""
    ts: str = ""
    model: str = ""
    escalated: bool = False
    reason: str = ""
    resolved: bool = False        # did any result come back at all
    error: bool = False
    result_chars: int = 0
    cost: float = 0.0             # from the matched sidechain run, if any
    duration_s: float = 0.0
    turns: int = 0

    @property
    def tier(self) -> str:
        """T0..T3 from the agent name, falling back to the model that ran.

        Empty when neither is knowable, and an unknown tier is never written to
        the outcome log: a row filed under the wrong rung calibrates the wrong
        gate.
        """
        by_name = AGENT_TIERS.get(self.agent_type.strip().lower(), "")
        return by_name or tier_for_model(self.model)

    @property
    def task_hash(self) -> str:
        """Stable identity, so importing twice does not double-count a run."""
        raw = f"{self.session}:{self.use_id}".encode()
        return hashlib.sha1(raw, usedforsecurity=False).hexdigest()[:16]

    @property
    def epoch(self) -> float:
        try:
            return datetime.fromisoformat(
                self.ts.replace("Z", "+00:00")).timestamp()
        except (ValueError, AttributeError):
            return 0.0


@dataclass
class Scan:
    dispatches: list[Dispatch] = field(default_factory=list)
    unresolved: int = 0           # dispatched, no result on record
    untiered: int = 0             # neither agent name nor model placed it

    @property
    def usable(self) -> list[Dispatch]:
        """Dispatches that resolved and can be placed on a tier."""
        return [d for d in self.dispatches if d.resolved and d.tier]

    @property
    def escalations(self) -> int:
        return sum(1 for d in self.usable if d.escalated)

    def by_tier(self) -> dict[str, tuple[int, int]]:
        """`{tier: (runs, escalations)}`, tiers in ladder order."""
        out: dict[str, tuple[int, int]] = {}
        for d in self.usable:
            runs, esc = out.get(d.tier, (0, 0))
            out[d.tier] = (runs + 1, esc + int(d.escalated))
        return dict(sorted(out.items()))

    def by_agent(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for d in self.dispatches:
            out[d.agent_type] = out.get(d.agent_type, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _result_text(block: dict) -> str:
    from adder.util.text import flatten_text

    return flatten_text(block.get("content"))


def scan(root: Path | str = DEFAULT_ROOT, *, window=None) -> Scan:
    """Recover every delegation under `root`, with whatever came back.

    Dispatch blocks are deduplicated by `tool_use` id, not by message id:
    Claude Code writes one record per content block, so a turn that dispatched
    two subagents appears as two records sharing a message id, and collapsing
    them by message loses one of the dispatches.
    """
    out = Scan()
    seen: dict[str, Dispatch] = {}
    answered: set[str] = set()

    for path in transcripts(root):
        try:
            fh = path.open(encoding="utf-8", errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                if '"tool_use"' not in line and '"tool_result"' not in line:
                    continue
                try:
                    d = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if window is not None and not window.keeps_record(
                        d, path.parent.name):
                    continue
                msg = mapping(d, "message")
                blocks = msg.get("content")
                if not isinstance(blocks, list):
                    continue
                session = str(d.get("sessionId") or path.stem)
                ts = d.get("timestamp") or ""

                for b in blocks:
                    if not isinstance(b, dict):
                        continue
                    kind = b.get("type")
                    if kind == "tool_use" and b.get("name") in DISPATCH_TOOLS:
                        use_id = str(b.get("id") or "")
                        if not use_id or use_id in seen:
                            continue
                        inp = b.get("input") if isinstance(b.get("input"), dict) else {}
                        seen[use_id] = Dispatch(
                            session=session,
                            project=path.parent.name,
                            use_id=use_id,
                            agent_type=str(inp.get("subagent_type") or ""),
                            description=str(inp.get("description") or "")[:120],
                            ts=ts,
                            model=str(inp.get("model") or ""),
                        )
                    elif kind == "tool_result":
                        use_id = str(b.get("tool_use_id") or "")
                        disp = seen.get(use_id)
                        if disp is None or use_id in answered:
                            continue
                        answered.add(use_id)
                        text = _result_text(b)
                        disp.resolved = True
                        disp.error = bool(b.get("is_error"))
                        disp.result_chars = len(text)
                        if disp.error or _ESCALATE.search(text):
                            disp.escalated = True
                            disp.reason = ("tool error" if disp.error
                                           else _first_line(text))

    out.dispatches = sorted(seen.values(), key=lambda x: (x.session, x.ts))
    _enrich_from_sidechains(out, root, window=window)
    out.unresolved = sum(1 for x in out.dispatches if not x.resolved)
    out.untiered = sum(1 for x in out.dispatches if x.resolved and not x.tier)
    return out


def _first_line(text: str, limit: int = 120) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:limit]
    return ""


def _enrich_from_sidechains(out: Scan, root: Path | str, *, window=None) -> None:
    """Attach the model, cost, and duration of the run each dispatch caused.

    The match is positional within a session -- the Nth dispatch to the Nth
    sidechain run -- because nothing in the transcript links them. That is good
    enough for a cost figure and for naming the model when the caller did not,
    and it is deliberately not allowed to influence the escalation signal,
    which comes only from the result block.
    """
    from adder.core.trace import load_sessions
    from adder.measure.spend import agents as agents_mod

    try:
        sessions = load_sessions(root)
    except Exception:
        return
    if window is not None and getattr(window, "active", False):
        sessions = window.apply(sessions)

    runs_by_session: dict[str, list] = {}
    for run in agents_mod.runs(sessions):
        runs_by_session.setdefault(run.session, []).append(run)

    by_session: dict[str, list[Dispatch]] = {}
    for d in out.dispatches:
        by_session.setdefault(d.session, []).append(d)

    for sid, dispatches in by_session.items():
        runs = runs_by_session.get(sid, [])
        for d, run in zip(dispatches, runs, strict=False):
            d.cost = run.cost()
            d.turns = run.n_turns
            d.model = d.model or run.model
            first, last = run.turns[0].when, run.turns[-1].when
            if first and last:
                d.duration_s = max(0.0, (last - first).total_seconds())


def to_outcomes(scan_result: Scan, *, known_hashes: set[str] | None = None) -> list:
    """Convert usable dispatches into `outcomes.Outcome` rows.

    Skips anything already in the log by `task_hash`, so re-importing is a
    no-op rather than a way to double the evidence behind a gate.
    """
    from adder.decide.track.outcomes import Outcome
    from adder.decide.track.similar import sketch

    known = known_hashes or set()
    rows = []
    for d in scan_result.usable:
        if d.task_hash in known:
            continue
        rows.append(Outcome(
            tier=d.tier,
            model=d.model,
            project=d.project,
            escalated=d.escalated,
            cost=d.cost,
            task_hash=d.task_hash,
            reason=d.reason,
            duration_s=d.duration_s,
            ts=d.epoch or 0.0,
            source="transcript",
            # The description is all the task text a transcript keeps for a
            # dispatch, and it is the field the harness asks to be a summary,
            # so it is the most on-topic few words available. Empty for a
            # dispatch that had none, which leaves the row invisible to the
            # neighbour estimator rather than giving it a sketch of nothing.
            sketch=list(sketch(d.description)),
        ))
    return rows


def report(scan_result: Scan, *, new_rows: int | None = None) -> str:
    from adder.util.render import money, table

    s = scan_result
    lines = ["  Delegations recovered from transcripts", ""]
    lines.append(f"  {len(s.dispatches):,} dispatches · {len(s.usable):,} usable "
                 f"· {s.escalations:,} escalated")
    if s.unresolved:
        lines.append(f"  {s.unresolved:,} never came back (interrupted, or the "
                     f"session ended first) and are not counted")
    if s.untiered:
        lines.append(f"  {s.untiered:,} could not be placed on a tier — neither "
                     f"the agent name nor the model identified one")

    if not s.dispatches:
        lines.append("")
        lines.append("  Nothing to import. This machine has no recorded delegations,")
        lines.append("  so `p_fail` stays on its prior and the router cannot prefer")
        lines.append("  a cheaper tier. `adder agents` shows what could have been")
        lines.append("  delegated and was not.")
        return "\n".join(lines)

    lines.append("")
    lines.append("  by tier:")
    rows = [[t, f"{runs:,}", f"{esc:,}", f"{esc / runs:.0%}"]
            for t, (runs, esc) in s.by_tier().items()]
    lines += table(rows, ["tier", "runs", "escalated", "rate"], align="<>>>")

    lines.append("")
    lines.append("  by agent:")
    rows = [[name or "(unnamed)", f"{n:,}"] for name, n in s.by_agent().items()]
    lines += table(rows, ["subagent_type", "dispatches"], align="<>")

    cost = sum(d.cost for d in s.usable)
    if cost:
        lines.append("")
        lines.append(f"  matched sidechain spend: {money(cost)}")

    if new_rows is not None:
        lines.append("")
        if new_rows:
            lines.append(f"  {new_rows:,} new rows written to the outcome log.")
        else:
            lines.append("  Nothing new: every dispatch here is already in the log.")
    else:
        lines.append("")
        lines.append("  Dry run. Pass --write to append these to the outcome log.")

    lines.append("")
    lines.append("  An escalation is an error result or an `ESCALATE:` reply. A")
    lines.append("  subagent that returned a confident wrong answer is invisible")
    lines.append("  here — and to a hand-written report too — so this rate is a")
    lines.append("  LOWER bound on how often a tier is not up to the work.")
    return "\n".join(lines)
