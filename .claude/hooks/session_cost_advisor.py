#!/usr/bin/env python3
"""Kept so a settings.json written before v0.2 keeps working.

The hook itself moved into the package -- `adder/decide/hooks/session_cost_advisor.py` -- because
`.claude/` is pruned from the wheel, so the version that lived here was one only
people with a git checkout ever ran. Anything installed by the old snippet, or by
an `adder auto on` from an earlier release, points at this path; this forwards it
rather than failing on every tool call. `adder auto off` removes either form.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adder.decide.hooks.session_cost_advisor import main

if __name__ == '__main__':
    raise SystemExit(main())
