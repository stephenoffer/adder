#!/usr/bin/env python3
"""UserPromptSubmit hook: warn when a session's context has become expensive.

Runs locally on every prompt and costs **zero model tokens**. Hooks cannot change
the model (there is no such output field), but they can inject `additionalContext`,
which is enough to surface the largest lever: session length.

Context cost grows with turns x context, so a long session is quadratic-ish in
turns. Splitting one into k sessions cuts that roughly k-fold -- on measured data
the single biggest available saving. Only Claude Code can act on that, and only
by suggesting it to the user, so this hook advises rather than acts.

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
WARN_SPEND = float(os.environ.get("ROUTER_WARN_SPEND", "15.0"))     # USD this session
WARN_CONTEXT = int(os.environ.get("ROUTER_WARN_CONTEXT", "400000")) # tokens
STATE = Path(os.environ.get("ROUTER_STATE", Path.home() / ".claude" / ".router-advisor.json"))


def _seen(session: str, level: int) -> bool:
    try:
        state = json.loads(STATE.read_text())
    except (OSError, ValueError):
        state = {}
    if state.get(session, 0) >= level:
        return True
    state[session] = level
    try:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(state))
    except OSError:
        pass
    return False


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (ValueError, OSError):
        payload = {}

    try:
        from router.cost import admitted_token_cost
        from router.live import analyse, current_session
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
    msg = (
        f"[session cost] {r.turns:,} turns, {r.context:,} tokens in context, "
        f"${r.spent:,.2f} spent (${r.per_turn:.3f}/turn). "
        f"Every 10K tokens added now costs ~${per10k:,.2f} over the rest of this session. "
        f"Prefer delegating large reads to a subagent; if this work has reached a "
        f"natural boundary, starting a fresh session is the largest single saving."
    )
    json.dump({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                      "additionalContext": msg}}, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
