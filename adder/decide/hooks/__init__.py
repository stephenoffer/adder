"""The three scripts the harness runs on its own, and the only code here that
is invoked by path rather than imported.

They used to live in `.claude/hooks/`, which was wrong in a way nothing caught
for four releases: `MANIFEST.in` prunes `.claude/`, so a wheel carried none of
them. `pip install adder-cli && adder auto on` therefore wrote three hook
entries pointing at files that did not exist, and the only measured
cost-*prevention* in the tool silently did nothing for every user who had not
cloned the repository. They are modules now, so the wheel ships them because
setuptools finds packages, not because somebody remembered a data glob.

What belongs here: I/O for one harness event -- read stdin, find the session,
print a decision. Nothing else. Every judgement these make lives in
`adder.decide.guard`, `adder.decide.handoff` and `adder.core.shapes`, where it
is unit-tested; the last time a decision lived inline, the one component whose
failure is silent was the one component with no tests behind it.

Each is also runnable directly (`python3 .../pretooluse_read_guard.py`) and
bootstraps its own `sys.path`, because the harness may run it with a different
interpreter than the one adder was installed into.
"""

from __future__ import annotations
