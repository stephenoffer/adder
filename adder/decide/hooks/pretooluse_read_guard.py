#!/usr/bin/env python3
"""PreToolUse hook: price a tool result BEFORE it lands in the context.

Every other tool in this repo measures spend after the fact. This one is the
only thing that can prevent it, because it runs while the decision is still
reversible.

The arithmetic it front-runs: a token admitted to a persistent context is billed
once as a cache write and again as a cache read on every remaining turn. At the
measured median session that is roughly 8x its sticker price, so a 50K-token
file read into a long session costs dollars, not cents -- and a subagent that
returns a 500-token summary of the same file costs a few cents.

This file is deliberately thin. The judgement lives in `adder/decide/guard.py` and the
size prediction in `adder/core/shapes.py`, both of which are tested; what remains
here is the I/O a hook has to do -- read stdin, find the session, load and save
the small state file, print. The previous version kept the decision inline, and
the one component in this project whose failure is silent was the one component
with no unit tests behind it.

Three things it will not do, each learned from a measurement:

* **It does not guess a size from a pattern.** The old version assumed 15,000
  tokens for any command containing `cat ` or `git log`. Measured against 222
  local transcripts, the calls that matched produce a median of 143 result
  tokens -- 105x less -- while the eighteen largest results in the corpus
  matched none of its patterns. Sizes now come from what commands of that shape
  actually returned on this machine (`adder guard --learn`).
* **It does not speak for free.** Injected advice is admitted to the context
  and re-read on every remaining turn like any other token. The guard prices
  its own message and stays quiet unless the expected saving covers it.
* **It does not repeat itself.** Once per command shape, at most 15 times in a
  session.

The hook is ADVISORY by default: it injects the price and the alternative and
lets the model decide. Set ADDER_GUARD_BLOCK=1 to escalate to a confirmation
prompt above the hard threshold instead. It never blocks silently, and it never
fires on small reads.

Install (settings.json):
  {"hooks": {"PreToolUse": [{"matcher": "Read|Bash|Grep|Glob|WebFetch|WebSearch|Task|Agent|Write",
     "hooks": [{"type": "command",
                "command": "python3 /abs/path/pretooluse_read_guard.py"}]}]}}

Or run `adder guard --install` to write that block for you.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

# `os.path` rather than `pathlib`: importing pathlib costs about 10ms, and this
# runs once per tool call whether or not the guard has anything to say.
#
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

# A duplicate of `adder.decide.guard.OBSERVED`, on purpose and under test: the
# whole point is to answer "not my problem" without paying for the import.
WATCHED = frozenset({"Read", "Bash", "Grep", "Glob", "WebFetch", "WebSearch",
                     "Task", "Agent", "Write"})


def _swallow(exc: BaseException) -> int:
    """Fail open, and say so on stderr when asked.

    Every failure path here returns 0 so the tool call proceeds, which is the
    only acceptable behaviour for a hook -- but it also means a genuine bug
    reads exactly like "there was nothing to say". A stubbed report missing one
    attribute silently disabled this guard during its own development, and the
    only symptom was silence.

    `ADDER_GUARD_DEBUG=1` prints the traceback to stderr, where Claude Code
    shows it without it ever reaching the model's context.
    """
    if os.environ.get("ADDER_GUARD_DEBUG") == "1":
        print("".join(traceback.format_exception(exc)), file=sys.stderr)
    return 0


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (ValueError, OSError):
        return 0

    tool = payload.get("tool_name") or ""
    inp = payload.get("tool_input") or {}
    # The directory a relative path in a shell command is relative to. The
    # hook process usually inherits it, but "usually" is how a `cat
    # pyproject.toml` gets keyed against the wrong file; the payload says.
    cwd = str(payload.get("cwd") or "") or None

    # Checked against a literal before importing anything. Importing `adder`
    # costs 27ms on top of the interpreter's own 32ms, and a PreToolUse hook
    # runs on every single call -- including the `Edit`, `TodoWrite` and
    # `Task`-adjacent ones this guard has no opinion about. `test_hooks.py`
    # fails if this list drifts from `guard.OBSERVED`, which is the real one.
    if tool not in WATCHED:
        return 0

    try:
        from adder.core.shapes import load_model
        from adder.decide import guard
    except ImportError as e:
        return _swallow(e)              # not installed; a hook must stay silent

    if tool not in guard.OBSERVED:
        return 0

    # A `Write` is watched but never advised about: its content is already in
    # the context because the content is the tool's own input. Remembering it is
    # what lets the guard catch the *read back* of the same file later.
    if tool not in guard.GUARDED:
        try:
            session_id = str(payload.get("session_id") or "")
            state = guard.load_state(session_id)
            guard.observe(tool, inp, state, guard.Verdict(False, "watched only"), cwd=cwd)
            guard.save_state(session_id, state)
        except Exception as e:
            return _swallow(e)
        return 0

    try:
        session_id = str(payload.get("session_id") or "")
        state = guard.load_state(session_id)
        sizes = load_model()
        if not guard.needs_pricing(tool, inp, sizes=sizes, state=state, cwd=cwd):
            # Remembered even when there is nothing to say. A read has to be
            # recorded for the *second* one to be caught, and a small bounded
            # command has to be counted for the aggregate rule to work at all --
            # those are the calls that add up to most of it.
            if (tool == "Read" and inp.get("file_path")) or tool == "Bash":
                guard.observe(tool, inp, state, guard.Verdict(False, "below floor"),
                              sizes=sizes, cwd=cwd)
                guard.save_state(session_id, state)
            return 0
    except Exception as e:
        return _swallow(e)

    # Only now is it worth parsing a transcript.
    try:
        from adder.measure.session.live import analyse, current_session

        sess = current_session(cwd)
        if sess is None or sess.n_turns < 5:
            # Too early to price, but NOT too early to remember. Returning here
            # meant every read in a session's first five turns was forgotten --
            # and those are precisely the reads a later turn re-reads, which is
            # the one saving in this project that needs no model to justify it.
            # The same applies to the running per-shape total the aggregate
            # rule is built on: it only works if the small early calls are
            # counted.
            guard.observe(tool, inp, state, guard.Verdict(False, "session too short"),
                          sizes=sizes, cwd=cwd)
            guard.save_state(session_id, state)
            return 0
        r = analyse(sess)
        # The measured re-read multiplier for THIS session, not the assumed
        # 0.10x. `carry` documents at length that the assumption under-prices
        # the term that decides this gate, and `live.analyse` has already
        # computed the number from the session's own turns -- so using it costs
        # no additional I/O.
        from adder.measure.window.carry import Carry

        # `getattr`, because `_swallow` names this exact hazard: a report
        # missing one attribute silently disables the guard, and the only
        # symptom is silence.
        mult = getattr(r, "read_mult", 0.0) or 0.0
        fitted = (Carry(read_mult=mult, baseline_read_mult=mult,
                        sessions=1, source="measured") if mult > 0 else None)
        verdict = guard.decide(tool, inp, model=r.model,
                           remaining_turns=r.carry_turns,
                           sizes=sizes, state=state, carry=fitted,
                           context_tokens=getattr(r, "context", 0), cwd=cwd)
        guard.observe(tool, inp, state, verdict, sizes=sizes, cwd=cwd)
        guard.save_state(session_id, state)
        if verdict.fire:
            # Recorded so `adder guard` can later ask whether saying it changed
            # anything. It is the only way the 0.5 uptake assumption ever stops
            # being an assumption.
            guard.record_fire(session_id, tool, inp, verdict)
    except Exception as e:
        return _swallow(e)              # a hook must never break the turn

    out = verdict.payload()
    if out:
        json.dump(out, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
