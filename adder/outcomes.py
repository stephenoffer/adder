"""Outcome log and escalation calibration - the adaptive half of adder.

The escalation gate needs `p_fail`: how often a tier fails and forces a retry.
Guessing it makes the gate arbitrary, so it is measured here instead.

An append-only JSONL log is used rather than subagent `memory:` because it is
inspectable, testable, and diffable; a router that silently changes its own
behaviour from an opaque store is hard to trust or debug.

Estimates are smoothed with a Beta(1,1) prior so a single early failure does not
swing the gate, scoped per (project, tier) because task mix differs sharply
between codebases, and **recency-weighted** so a tier that used to fail but has
since improved -- or a codebase that has grown harder -- is tracked rather than
averaged over all history.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

DEFAULT_LOG = Path(
    os.environ.get("ADDER_LOG", Path.home() / ".claude" / "adder-outcomes.jsonl")
)

# Beta(1,1) prior: with no evidence, p_fail = 0.5 (maximally cautious).
PRIOR_FAIL = 1.0
PRIOR_OK = 1.0

# Below this many scoped observations, fall back to the global rate for the tier.
MIN_SCOPED = 5

# Older outcomes count for less. An outcome this many days old counts half.
HALF_LIFE_DAYS = 30.0

# Keep the log bounded; it is telemetry, not an audit trail.
MAX_ROWS = 20_000


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
    effort: str = ""
    duration_s: float = 0.0
    ts: float = field(default_factory=time.time)


_FIELDS = {f.name for f in fields(Outcome)}


def record(outcome: Outcome, log: Path | str = DEFAULT_LOG) -> None:
    """Append one outcome. Never raises: telemetry must not break routing.

    A single `write` of one short line to an O_APPEND file is atomic on POSIX,
    so concurrent sessions interleave lines rather than corrupting them.
    """
    try:
        p = Path(log)
        p.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(outcome), separators=(",", ":")) + "\n"
        with p.open("a") as fh:
            fh.write(line)
    except (OSError, TypeError, ValueError):
        pass


def load(log: Path | str = DEFAULT_LOG) -> list[Outcome]:
    """Read the log, skipping anything malformed and tolerating new fields."""
    p = Path(log)
    if not p.exists():
        return []
    out = []
    try:
        text = p.read_text(errors="replace")
    except OSError:
        return []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            if not isinstance(d, dict):
                continue
            # Ignore fields written by a newer version rather than crashing.
            out.append(Outcome(**{k: v for k, v in d.items() if k in _FIELDS}))
        except (ValueError, TypeError):
            continue
    return out


def prune(log: Path | str = DEFAULT_LOG, keep: int = MAX_ROWS) -> int:
    """Trim the log to its most recent `keep` rows. Returns rows dropped."""
    rows = load(log)
    if len(rows) <= keep:
        return 0
    rows.sort(key=lambda o: o.ts)
    kept = rows[-keep:]
    try:
        p = Path(log)
        tmp = p.with_suffix(".tmp")
        with tmp.open("w") as fh:
            for o in kept:
                fh.write(json.dumps(asdict(o), separators=(",", ":")) + "\n")
        tmp.replace(p)
    except OSError:
        return 0
    return len(rows) - len(kept)


def _weight(o: Outcome, now: float) -> float:
    """Exponential recency weight. Recent evidence should move the gate more."""
    age_days = max(0.0, (now - o.ts) / 86400.0)
    return 0.5 ** (age_days / HALF_LIFE_DAYS)


def p_fail(
    tier: str,
    project: str | None = None,
    log: Path | str = DEFAULT_LOG,
    outcomes: list[Outcome] | None = None,
    *,
    recency_weighted: bool = True,
) -> float:
    """Smoothed escalation rate for a tier, scoped to a project when possible.

    Falls back to the global rate for that tier when a project has too little
    history, and to the 0.5 prior when there is none at all.
    """
    rows = outcomes if outcomes is not None else load(log)
    scoped = [o for o in rows if o.tier == tier and (project is None or o.project == project)]
    if len(scoped) < MIN_SCOPED and project is not None:
        scoped = [o for o in rows if o.tier == tier]
    if not scoped:
        return PRIOR_FAIL / (PRIOR_FAIL + PRIOR_OK)

    if not recency_weighted:
        fails = sum(1 for o in scoped if o.escalated)
        return (fails + PRIOR_FAIL) / (len(scoped) + PRIOR_FAIL + PRIOR_OK)

    now = time.time()
    wf = sum(_weight(o, now) for o in scoped if o.escalated)
    wt = sum(_weight(o, now) for o in scoped)
    return (wf + PRIOR_FAIL) / (wt + PRIOR_FAIL + PRIOR_OK)


def calibration(log: Path | str = DEFAULT_LOG) -> dict[str, dict[str, float | int]]:
    """Per-tier escalation stats, for `/adder-doctor` and for spotting drift."""
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

    ap = argparse.ArgumentParser(prog="adder.outcomes")
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    ap.add_argument("--prune", action="store_true", help="trim the log and exit")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    if a.prune:
        print(f"  dropped {prune(a.log)} old rows")
        return 0

    cal = calibration(a.log)
    if a.json:
        print(json.dumps(cal))
        return 0
    if not cal:
        print(f"\n  No outcomes recorded yet ({a.log}).")
        print("  Until there is history, p_fail defaults to the 0.5 prior, which")
        print("  makes the escalation gate maximally cautious about cheap tiers.\n")
        return 0
    print(f"\n  {'tier':<8}{'runs':>7}{'escalated':>11}{'p_fail':>9}{'cost':>10}")
    for tier, s in cal.items():
        print(f"  {tier:<8}{s['n']:>7}{s['escalated']:>11}{s['p_fail']:>9.2f}${s['cost']:>9.3f}")
    print("\n  p_fail is recency-weighted (30-day half-life) and Beta(1,1)-smoothed.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
