#!/usr/bin/env python3
"""PreToolUse hook: price a read BEFORE it lands in the context.

Every other tool in this repo measures spend after the fact. This one is the
only thing that can prevent it, because it runs while the decision is still
reversible.

The arithmetic it front-runs: a token admitted to a persistent context is billed
once as a cache write and again as a cache read on every remaining turn. At the
measured median session that is roughly 8x its sticker price, so a 50K-token
file read into a long session costs dollars, not cents -- and a subagent that
returns a 500-token summary of the same file costs a few cents.

The hook is ADVISORY by default: it injects the price and the alternative, and
lets the model decide. Set ROUTER_GUARD_BLOCK=1 to escalate to a confirmation
prompt above the hard threshold instead. It never blocks silently, and it never
fires on small reads.

Install (settings.json):
  {"hooks": {"PreToolUse": [{"matcher": "Read|Bash|Grep",
     "hooks": [{"type": "command",
                "command": "python3 /abs/path/pretooluse_read_guard.py"}]}]}}
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Advise above this; ask for confirmation above the hard threshold.
WARN_TOKENS = int(os.environ.get("ROUTER_GUARD_WARN", "15000"))
HARD_TOKENS = int(os.environ.get("ROUTER_GUARD_HARD", "60000"))
BLOCK = os.environ.get("ROUTER_GUARD_BLOCK", "") == "1"

CHARS_PER_TOKEN = 4

# Commands whose output is routinely enormous and routinely unnecessary in full.
_VERBOSE = ("cat ", "find ", "ls -R", "git log", "git diff", "npm ls",
            "pip list", "curl ", "grep -r", "rg ")
_BOUNDED = ("head", "tail", "wc", "| head", "| tail", "| wc", "-n ", "--max-count")


def _estimate_read_tokens(tool: str, inp: dict) -> tuple[int, str]:
    """Best-effort size of what this call will admit to context."""
    if tool == "Read":
        fp = inp.get("file_path")
        if not fp:
            return 0, ""
        try:
            size = Path(fp).stat().st_size
        except OSError:
            return 0, ""
        limit = inp.get("limit")
        if limit:                      # a bounded read is already the right shape
            return 0, ""
        return size // CHARS_PER_TOKEN, Path(fp).name
    if tool == "Bash":
        cmd = (inp.get("command") or "").lower()
        if any(b in cmd for b in _BOUNDED):
            return 0, ""               # already bounded; nothing to advise
        if any(v in cmd for v in _VERBOSE):
            # Unknown until it runs. Flag the shape, not a fabricated size.
            return WARN_TOKENS, "command output"
    return 0, ""


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (ValueError, OSError):
        return 0

    tool = payload.get("tool_name") or ""
    inp = payload.get("tool_input") or {}
    if tool not in ("Read", "Bash"):
        return 0

    try:
        tokens, what = _estimate_read_tokens(tool, inp)
    except Exception:
        return 0
    if tokens < WARN_TOKENS:
        return 0

    # Price it against the CURRENT session, not a global average.
    try:
        from router.cost import admitted_token_cost, placement_cost
        from router.live import analyse, current_session

        sess = current_session(payload.get("cwd"))
        if sess is None or sess.n_turns < 5:
            return 0
        r = analyse(sess)
        inline = admitted_token_cost(tokens, r.model, r.projected_remaining)
        _, sub, _d = placement_cost(
            tokens_read=tokens, summary_tokens=max(200, tokens // 10),
            remaining_turns=r.projected_remaining, main_model=r.model)
    except Exception:
        return 0                        # a hook must never break the turn

    if inline < 0.25:                   # not worth interrupting for
        return 0

    msg = (
        f"[read guard] This {tool} adds ~{tokens:,} tokens to a context that is "
        f"re-read ~{r.projected_remaining:,} more times: ~${inline:,.2f} over the "
        f"rest of this session, vs ~${sub:,.2f} if a subagent reads it and returns "
        f"a summary. If you need the whole file, read it. If you need a fact from "
        f"it, delegate the read or bound it (limit/offset, head, grep -n)."
    )

    if BLOCK and tokens >= HARD_TOKENS:
        json.dump({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "ask",
            "permissionDecisionReason": msg,
        }}, sys.stdout)
        return 0

    json.dump({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "additionalContext": msg,
    }}, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
