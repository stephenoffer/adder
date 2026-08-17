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
#
# Resolved through `adder.core.settings`, not from the environment directly.
# `warn_spend` and `warn_context` are declared settings, so `adder config`
# reports whatever `.adder.json` says -- and reading the environment here meant
# the hook used a different number from the one the tool printed. A setting the
# tool reports and the code ignores is the invisible state `settings.py` exists
# to remove.
def _setting(name: str, cast, fallback):
    try:
        from adder.core.settings import get

        return cast(get(name))
    except Exception:
        return fallback


WARN_SPEND = _setting("warn_spend", float, 15.0)        # USD this session
WARN_CONTEXT = _setting("warn_context", int, 400_000)   # tokens
ADVISOR_STATE_NAME = ".adder-advisor.json"
# `Path("")` is `PosixPath('.')`, which is truthy -- so the env var has to be
# tested as a string, not as a Path.
_STATE_ENV = os.environ.get("ADDER_STATE", "").strip()
STATE = Path(_STATE_ENV) if _STATE_ENV else (
    Path(_setting("home", str, str(Path.home() / ".claude"))) / ADVISOR_STATE_NAME)

# Keep the state file from growing without bound across many sessions.
MAX_STATE_ENTRIES = 500

# Assumed share of this advice that is acted on, matching the read guard's
# `guard_advice_taken`. An assumption, and declared as one: nothing in a
# transcript says whether a person split a session because of a sentence.
ADVICE_TAKEN = _setting("guard_advice_taken", float, 0.5)


def _seen(session: str, level: int) -> bool:
    try:
        state = json.loads(STATE.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            state = {}
    except (OSError, ValueError):
        state = {}
    try:
        if float(state.get(session, 0) or 0) >= level:
            return True
    except (TypeError, ValueError):
        pass
    state.pop(session, None)          # re-insert, so the trim below is LRU
    state[session] = level
    if len(state) > MAX_STATE_ENTRIES:
        # Drop the oldest half; this is a dedup cache, not a record.
        state = dict(list(state.items())[-MAX_STATE_ENTRIES // 2:])
    try:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        # Unique per writer, like every other atomic write in this repo. A
        # fixed `.tmp` is a shared mutable path, and this is the most
        # concurrent thing in the product: it runs on every prompt of every
        # session on the machine. `trace._cache_store` measured 45% of writes
        # lost under three concurrent writers on exactly this pattern.
        tmp = STATE.with_name(f"{STATE.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(state), encoding="utf-8")
            tmp.replace(STATE)
        finally:
            tmp.unlink(missing_ok=True)
    except OSError:
        pass
    return False


def _context_verdict(r, sess, cwd) -> str:
    """The compact-or-restart call, priced, with what a brief would have to name.

    Fails open to the old generic sentence: this runs in front of a prompt, and
    an advisor that raises is an advisor that silently stops advising.
    """
    try:
        verdict, worth = r.context_verdict()
        if verdict == "carry on":
            return (f"Context is not yet worth resetting — sessions this long "
                    f"typically run ~{r.projected_remaining:,} more turns, so "
                    "keep admissions small.")
        from adder.decide.handoff import budget, items_from
        from adder.measure.session.live import current_transcript

        b = budget(sess, remaining=r.carry_turns, read_mult=r.read_mult)
        alt = r.compaction_net() if verdict == "restart" else r.restart_net()
        line = (f"[context] {verdict} now: worth ~${worth:,.2f} over the "
                f"~{r.carry_turns:,.0f} turns expected to remain "
                f"(the other option is worth ~${alt:,.2f}).")
        if verdict == "restart" and b.tokens:
            names = [i.name.rsplit("/", 1)[-1]
                     for i in items_from(current_transcript(cwd))[:5]]
            # A bound of a quarter-million tokens is not a brief budget, it is
            # "cost is not the constraint". Printing it as a budget invites
            # someone to fill it.
            size = (f" A brief of up to {b.tokens:,} tokens still pays"
                    if b.binding else " Cost is not the constraint on the brief")
            line += size + (f"; it has to name {', '.join(names)}." if names else ".")
        return line
    except Exception:
        return ("If this work has reached a natural boundary, starting a fresh "
                "session is the largest single saving.")


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (ValueError, OSError):
        payload = {}

    try:
        from adder.measure.session.live import analyse, current_session
        from adder.pricing.cost import admitted_token_cost
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

    # What this advice is arguing for, in dollars, so its own cost can be
    # weighed against it. `None` means the session could not be priced, and an
    # unpriceable saving is not netted against a real cost.
    try:
        worth = max(r.compaction_net(), r.restart_net())
    except Exception:
        worth = None

    per10k = admitted_token_cost(10_000, r.model, r.carry_turns)
    parts = [
        f"[session cost] {r.turns:,} turns, {r.context:,} tokens in context, "
        f"${r.spent:,.2f} spent (${r.per_turn:.3f}/turn). "
        f"One more turn costs ~${r.next_turn_cost:.3f}; every 10K tokens added now "
        f"costs ~${per10k:,.2f} over the rest of this session "
        f"(an output token written now costs {r.debt_multiple:.0f}x its sticker price)."
    ]
    if r.context_pressure >= 0.75:
        # This model's provider's multipliers, not Anthropic's. Under automatic
        # caching there is no write premium and this sentence named a penalty
        # the reader does not pay.
        try:
            from adder.pricing.cost import Rates

            rr = Rates.for_model(r.model, ttl=r.ttl)
            write_x = f"{rr.cache_write / rr.inp:.2f}x" if rr.inp else "full input rate"
            read_x = f"{rr.cache_read / rr.inp:.2f}x" if rr.inp else "full input rate"
        except Exception:
            write_x, read_x = "the write rate", "the cached rate"
        parts.append(
            f"Context is at {r.context_pressure:.0%} of the window — compaction is "
            f"imminent, and it rebuilds the cache at {write_x} instead of "
            f"reading it at {read_x}."
        )
    parts.append(
        "Prefer delegating large reads to a subagent and bounding command output "
        "(head/wc/grep -n)."
    )
    # The generic version of this line -- "starting fresh is the largest saving"
    # -- was true and unusable: it named no price, so it read as a slogan and
    # was ignored for hundreds of turns at a time. The verdict below is the same
    # advice with the two numbers that make it actionable, and it withdraws
    # itself when carrying on is actually cheaper.
    parts.append(_context_verdict(r, sess, payload.get("cwd")))
    message = " ".join(parts)

    # The advice is admitted to the context and re-read on every remaining
    # turn, exactly like a tool result. The read guard learned this the
    # expensive way -- it fired 903 times without ever charging for the
    # sentences it injected -- and the same test applies here: say nothing
    # unless what is being advocated is worth more than saying it.
    #
    # It clears easily and is expected to: this fires at most twice a session
    # and argues for a lever worth hundreds. That is the point of checking
    # rather than assuming, and if it ever stops clearing, the message is too
    # long or the threshold is too low.
    try:
        from adder.util.text import est_tokens

        overhead = admitted_token_cost(est_tokens(message), r.model, r.carry_turns)
        if worth is not None and worth * ADVICE_TAKEN <= overhead:
            return 0
    except Exception:
        pass

    json.dump({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                      "additionalContext": message}}, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
