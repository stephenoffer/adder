"""Outcome log and escalation calibration - the adaptive half of the router.

The escalation gate needs `p_fail`: how often a tier fails and forces a retry.
Guessing it makes the gate arbitrary, so it is measured here instead.

An append-only JSONL log is used rather than subagent `memory:` because it is
inspectable, testable, and diffable; a router that silently changes its own
behaviour from an opaque store is hard to trust or debug.

Estimates are smoothed with a Beta(1,1) prior so a single early failure does not
swing the gate, and are scoped per (project, tier) because task mix differs
sharply between codebases.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_LOG = Path(
    os.environ.get("ROUTER_LOG", Path.home() / ".claude" / "router-outcomes.jsonl")
)

# Beta(1,1) prior: with no evidence, p_fail = 0.5 (maximally cautious).
PRIOR_FAIL = 1.0
PRIOR_OK = 1.0


@dataclass
class Outcome:
    tier: str
    model: str
    project: str
    escalated: bool
    context_tokens: int = 0
    remaining_turns: int = 0
    cost: float = 0.0
    task_hash: str = ""
    reason: str = ""
    ts: float = field(default_factory=time.time)


def record(outcome: Outcome, log: Path | str = DEFAULT_LOG) -> None:
    """Append one outcome. Never raises: telemetry must not break routing."""
    try:
        p = Path(log)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as fh:
            fh.write(json.dumps(asdict(outcome)) + "\n")
    except OSError:
        pass


def load(log: Path | str = DEFAULT_LOG) -> list[Outcome]:
    p = Path(log)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            out.append(Outcome(**json.loads(line)))
        except (ValueError, TypeError):
            continue
    return out


def p_fail(
    tier: str,
    project: str | None = None,
    log: Path | str = DEFAULT_LOG,
    outcomes: list[Outcome] | None = None,
) -> float:
    """Smoothed escalation rate for a tier, scoped to a project when possible.

    Falls back to the global rate for that tier when a project has too little
    history, and to the 0.5 prior when there is none at all.
    """
    rows = outcomes if outcomes is not None else load(log)
    scoped = [o for o in rows if o.tier == tier and (project is None or o.project == project)]
    if len(scoped) < 5 and project is not None:
        scoped = [o for o in rows if o.tier == tier]
    fails = sum(1 for o in scoped if o.escalated)
    total = len(scoped)
    return (fails + PRIOR_FAIL) / (total + PRIOR_FAIL + PRIOR_OK)


def calibration(log: Path | str = DEFAULT_LOG) -> dict[str, dict[str, float | int]]:
    """Per-tier escalation stats, for `/route-doctor` and for spotting drift."""
    rows = load(log)
    by: dict[str, list[Outcome]] = defaultdict(list)
    for o in rows:
        by[o.tier].append(o)
    return {
        tier: {
            "n": len(v),
            "escalated": sum(1 for o in v if o.escalated),
            "p_fail": p_fail(tier, None, log, outcomes=rows),
            "cost": round(sum(o.cost for o in v), 4),
        }
        for tier, v in sorted(by.items())
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="router.outcomes")
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    a = ap.parse_args(argv)

    cal = calibration(a.log)
    if not cal:
        print(f"\n  No outcomes recorded yet ({a.log}).")
        print("  Until there is history, p_fail defaults to the 0.5 prior, which")
        print("  makes the escalation gate maximally cautious about cheap tiers.\n")
        return 0
    print(f"\n  {'tier':<8}{'runs':>7}{'escalated':>11}{'p_fail':>9}{'cost':>10}")
    for tier, s in cal.items():
        print(f"  {tier:<8}{s['n']:>7}{s['escalated']:>11}{s['p_fail']:>9.2f}${s['cost']:>9.3f}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
