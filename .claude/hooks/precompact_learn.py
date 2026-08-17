#!/usr/bin/env python3
"""PreCompact hook: re-learn result sizes while the session is already paused.

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

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)


def main() -> int:
    with contextlib.suppress(ValueError, OSError):
        json.load(sys.stdin)          # drained, not used: nothing here is per-call

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
