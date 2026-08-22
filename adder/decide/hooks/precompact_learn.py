#!/usr/bin/env python3
"""PreCompact hook: forget what the context holds, and re-learn result sizes.

Two jobs, and the first one is load-bearing for correctness rather than for
accuracy.

**Forgetting.** The guard refuses a re-read on the grounds that the content is
already in this context. Compaction is the event that makes that claim false:
the tokens it refers to are about to leave the window. A guard still holding
them would refuse a read of something the model no longer has, which is the one
way an enforcing guard can cost more than it saves. So the per-session memory
of what has been read and written is cleared here, before the compaction it is
invalidated by. The running spend totals survive -- they record money already
spent, and compaction does not refund it.

**Learning.** The rest of the original reason for this hook:

The guard predicts what a tool call will admit from what calls of that shape
have actually returned on this machine. That model has to be built by scanning
transcripts, which takes a couple of seconds -- affordable occasionally, and
never on the path of a tool call. So it is refreshed by `adder guard --learn`,
which means it is refreshed when somebody remembers to.

Compaction is the moment that fixes. It only happens in a long session, a long
session is exactly where a stale model costs the most, and the session is
already stopping to rebuild its context -- so a two-second scan is the cheapest
two seconds available anywhere in this tool.

**It prints nothing.** That is deliberate rather than minimal: this file does
work and emits no `hookSpecificOutput` at all, so it cannot inject tokens into
a context that is in the middle of being compacted, and it does not depend on
what a PreCompact hook is allowed to return. The only observable effect is that
`~/.claude/.adder-sizes.json` is newer afterwards.

It is also a no-op unless the model is actually stale: `shapes.refresh` checks
the cached model's age first and returns it untouched when it is current, so
compacting twice in an hour does not scan twice.

Install (settings.json):
  {"hooks": {"PreCompact": [{"hooks": [
     {"type": "command", "command": "python3 /abs/path/precompact_learn.py"}]}]}}

The same block works for `SessionEnd` if you would rather pay for it there.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys

# ROOT is the directory that holds the `adder` package, four levels up from this file
# (`<root>/adder/decide/hooks/`). The same arithmetic works from a checkout and
# from `site-packages`, which is the reason the hooks live inside the package at
# all: a hook that only exists in a git checkout is a hook a `pip install` user
# never gets. Inserted on `sys.path` rather than assumed importable, because the
# harness may invoke this with a different interpreter than the one adder was
# installed into.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)


def main() -> int:
    payload = {}
    with contextlib.suppress(ValueError, OSError):
        payload = json.load(sys.stdin) or {}

    # First, because it is the one that can be wrong in a way that costs money.
    # A failure to learn costs accuracy; a failure to forget costs a refusal
    # the model cannot satisfy.
    try:
        from adder.decide import guard

        session_id = str((payload or {}).get("session_id") or "")
        if session_id:
            state = guard.load_state(session_id)
            guard.save_state(session_id, state.forget_context())
    except Exception as e:
        if os.environ.get("ADDER_GUARD_DEBUG") == "1":
            import traceback

            print("".join(traceback.format_exception(e)), file=sys.stderr)

    try:
        from adder.core.shapes import refresh

        refresh()                     # a no-op unless the cached model is stale
    except Exception as e:
        # Fail open and silent, like every other hook here. A failure to learn
        # costs accuracy on the next prediction and nothing else.
        if os.environ.get("ADDER_GUARD_DEBUG") == "1":
            import traceback

            print("".join(traceback.format_exception(e)), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
