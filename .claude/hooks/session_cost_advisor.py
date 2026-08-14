#!/usr/bin/env python3
"""UserPromptSubmit hook: warn when a session's context has become expensive.

Runs locally on every prompt and costs **zero model tokens**. Hooks cannot change
the model (there is no such output field), but they can inject `additionalContext`,
which is enough to surface the largest lever: session length.

Context cost grows with turns x context, so a long session is quadratic-ish in
turns. Splitting one into k sessions cuts that roughly k-fold -- on measured data
the single biggest available saving. Only Claude Code can act on that, and only
by suggesting it to the user, so this hook advises rather than acts.

It reads one transcript through a mtime-keyed parse cache, so it adds no
perceptible latency to a prompt.

Install (settings.json):
  {"hooks": {"UserPromptSubmit": [{"hooks": [
     {"type": "command", "command": "python3 /abs/path/session_cost_advisor.py"}]}]}}
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Advise once per threshold crossing, not on every prompt.
WARN_SPEND = float(os.environ.get("ADDER_WARN_SPEND", "15.0"))     # USD this session
WARN_CONTEXT = int(os.environ.get("ADDER_WARN_CONTEXT", "400000")) # tokens
STATE = Path(os.environ.get("ADDER_STATE", Path.home() / ".claude" / ".adder-advisor.json"))

# Keep the state file from growing without bound across many sessions.
MAX_STATE_ENTRIES = 500


def _seen(session: str, level: int) -> bool:
    try:
        state = json.loads(STATE.read_text())
        if not isinstance(state, dict):
            state = {}
    except (OSError, ValueError):
        state = {}
    if state.get(session, 0) >= level:
        return True
    state[session] = level
    if len(state) > MAX_STATE_ENTRIES:
        # Drop the oldest half; this is a dedup cache, not a record.
        state = dict(list(state.items())[-MAX_STATE_ENTRIES // 2:])
    try:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(STATE)
    except OSError:
        pass
    return False


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (ValueError, OSError):
        payload = {}

    try:
        from adder.cost import admitted_token_cost
        from adder.live import analyse, current_session
    except ImportError:
        return 0

    try:
        sess = current_session(payload.get("cwd"))
        if sess is None or sess.n_turns < 20:
            return 0
        r = analyse(sess)
    except Exception:
        return 0  # a hook must never break the turn

    level = 2 if (r.spent >= WARN_SPEND * 2 or r.context >= 800_000) else 1
    if r.spent < WARN_SPEND and r.context < WARN_CONTEXT:
        return 0
    if _seen(sess.id, level):
        return 0

    per10k = admitted_token_cost(10_000, r.model, r.projected_remaining)
    parts = [
        f"[session cost] {r.turns:,} turns, {r.context:,} tokens in context, "
        f"${r.spent:,.2f} spent (${r.per_turn:.3f}/turn). "
        f"One more turn costs ~${r.next_turn_cost:.3f}; every 10K tokens added now "
        f"costs ~${per10k:,.2f} over the rest of this session "
        f"(an output token written now costs {r.debt_multiple:.0f}x its sticker price)."
    ]
    if r.context_pressure >= 0.75:
        parts.append(
            f"Context is at {r.context_pressure:.0%} of the window — compaction is "
            f"imminent, and it rebuilds the cache at 1.25x instead of reading it at 0.10x."
        )
    parts.append(
        "Prefer delegating large reads to a subagent and bounding command output "
        "(head/wc/grep -n). If this work has reached a natural boundary, starting a "
        f"fresh session is the largest single saving — sessions this long typically "
        f"run ~{r.projected_remaining:,} more turns, not a handful."
    )
    json.dump({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                      "additionalContext": " ".join(parts)}}, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
