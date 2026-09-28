#!/usr/bin/env python3
"""PostToolUse hook: a read is in the context once it has landed, not before.

The read guard refuses a re-read on the grounds that the file is already in
this context, and it used to record that at PreToolUse -- before the permission
prompt, before any deny rule, before the tool ran. A Read the user rejected, one
a rule blocked, and a `cat` whose output the harness spilled to a file were all
remembered as read, and the next attempt was refused with "use the copy you
have" about a copy the model never got.

PostToolUse fires only for a call that ran and succeeded (a failed one goes to
PostToolUseFailure, a refused one to neither), and it carries the response, so
this is the one place the claim can be checked instead of predicted: a Read
reports the lines it returned out of how many, and a Bash result that spilled
names the file it spilled to. `guard.confirm` promotes the pending read only
when the whole file arrived.

It prints nothing and injects nothing. Like every hook here it fails open.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

WATCHED = frozenset({"Read", "Bash"})


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (ValueError, OSError):
        return 0
    if not isinstance(payload, dict) or payload.get("tool_name") not in WATCHED:
        return 0
    inp = payload.get("tool_input")
    if not isinstance(inp, dict):
        return 0
    try:
        from adder.decide import guard
        from adder.decide.hooks.pretooluse_read_guard import context_key

        session_id = context_key(payload)
        if not session_id:
            return 0
        state = guard.load_state(session_id)
        guard.confirm(payload["tool_name"], inp, payload.get("tool_response"), state,
                      cwd=str(payload.get("cwd") or "") or None)
        guard.save_state(session_id, state)
    except Exception as e:
        if os.environ.get("ADDER_GUARD_DEBUG") == "1":
            print("".join(traceback.format_exception(e)), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
