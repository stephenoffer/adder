"""Rewrite a call to its priced bounded equal, instead of refusing it.

The gap this closes
-------------------
The guard's best sentence has always been "this admits ~15,000 tokens at ~$1.19
of carry, against ~$0.13 delegated — read 300 lines of it". In `full` mode it
then *refuses*, and the model re-issues a narrower call on the next turn. That
works, and it costs a turn: the refusal, the model reading the refusal, and the
second call. At the context sizes where the guard fires, that turn is not
rounding error, and it is spent agreeing with advice the guard had already
priced.

The harness has a seam that removes it. A `PreToolUse` hook may return
`updatedInput`, and the call executes with the arguments the hook substituted.
So the guard can stop asking for the bounded call and simply make the call
bounded — same information budget, one fewer turn, and no negotiation.

Verified against the shipped client rather than the docs, because the docs do
not state the three things that decide whether this is safe. In the strings of
Claude Code 2.1.238:

* `PreToolUse hook for <tool> returned updatedInput that failed schema
  validation:` — the field exists on this event and is checked against the
  tool's own input schema. So a rewrite must be a *complete, valid* input, not
  a patch.
* `updatedInput is missing or empty, falling back to original tool input` — an
  absent rewrite is a no-op rather than an error, which is what makes returning
  `None` from here the safe default.
* `Hook satisfied user interaction for <tool> via updatedInput, bypassing
  permission prompt` — and this is the dangerous one. A rewrite travels with an
  approval, so it can suppress a prompt the user would otherwise have seen.

That last line is why every rule below exists, and why the feature is off until
somebody turns it on.

What may be rewritten, and why so little
----------------------------------------
Two tools. `Read` gains a `limit`; `Grep` gains a `head_limit`. Both are
read-only, and in both cases the rewrite is a **strict subset** of what was
asked for: fewer lines of the same file, fewer hits of the same pattern. Nothing
else qualifies, and the near-misses are more instructive than the hits:

* **`Bash` is refused, not rewritten.** `_bounded_hint` suggests piping through
  `head -50`, which is right as advice and wrong as an edit. Appending a pipe to
  a command whose text this module did not write can change its exit status, cut
  a `&&` chain, or silently truncate the input to a command whose *output* was
  never the point. Rewriting somebody's shell is not a bounded operation.
* **`Grep` is not switched to `files_with_matches`.** That is the cheapest
  bounded form and it changes the *kind* of answer, not the amount: the caller
  asked what the matches say and would be handed a list of filenames. A
  truncation the model can see the edge of is recoverable; a different question
  answered silently is not.
* **`Glob` and `WebFetch` have no bounding parameter** to set, so there is
  nothing to substitute that would still validate.

The four rules
--------------
1. **Never widen.** A caller who already passed `limit=50` has bounded the call
   themselves and more tightly than the price floor would. Raising it to 300
   would be the guard spending money in the name of saving it.
2. **Never narrow past usefulness.** Below `MIN_LINES` the result is too small
   to answer anything and the call will simply be made again, so the turn the
   rewrite was meant to save is spent anyway plus the one it wasted.
3. **Only keys the tool already accepts.** The rewrite is schema-validated by
   the harness; an invented key fails the whole thing, and the failure mode is
   the hook silently doing nothing.
4. **Return `None` when there is nothing strictly narrower to say.** Every
   caller reads `None` as "leave this call alone", which is the behaviour the
   guard had before this module existed.

Why it is off by default
------------------------
Because of the third string above. On a call the user's settings would have
prompted for, a rewrite is a prompt that did not happen — and the fact that the
result is smaller than requested does not make it authorised. The harness
overrides an approval where a `deny` rule or an `ask` rule covers the call, so
the exposure is limited to calls that would have prompted by default; that is
still a decision belonging to the person whose files they are, not to a cost
tool. `guard_narrow` therefore starts `false`, and it only ever applies to a
call the guard was going to refuse outright — so turning it on relaxes a
refusal, and can never permit something the guard would have been silent about.
"""

from __future__ import annotations

# Fewer lines than this is not an answer, it is a round trip. The number is the
# same order as the smallest `limit` a caller writes by hand; below it the model
# reliably re-reads, which costs the turn this exists to save plus one.
MIN_LINES = 40

# The bounding parameter each rewritable tool accepts, and the keys that must
# survive the rewrite untouched. Written as a table because the schema belongs
# to the harness: when a tool grows a new argument, a rewrite that dropped it
# would be a silent change of meaning, and the fix belongs in one place.
BOUNDS: dict[str, str] = {
    'Read': 'limit',
    'Grep': 'head_limit',
}


def bounded_at(tool: str, tool_input: dict) -> int | None:
    """The bound already on this call, or None if it is unbounded.

    Read separately from `narrow` so the guard can tell "already bounded, leave
    it" apart from "not rewritable at all" without re-deriving either.
    """
    key = BOUNDS.get(tool)
    if key is None:
        return None
    raw = (tool_input or {}).get(key)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    n = int(raw)
    return n if n > 0 else None


def narrow(tool: str, tool_input: dict, *, lines: int) -> dict | None:
    """A strictly narrower, schema-valid replacement input, or None.

    `lines` is the affordable budget the guard already computed from the carry
    and the cost floor — this module does no pricing of its own, deliberately.
    Two places deciding what a read costs is how the message and the rewrite end
    up disagreeing about the same call.
    """
    key = BOUNDS.get(tool)
    if key is None:
        return None
    if not isinstance(tool_input, dict) or not tool_input:
        return None
    budget = int(lines)
    if budget < MIN_LINES:
        return None
    already = bounded_at(tool, tool_input)
    if already is not None and already <= budget:
        # Rule 1. The caller was stricter than the price floor; leave them alone.
        return None
    out = dict(tool_input)
    out[key] = budget
    # Rule 4, as an identity check rather than a promise: if nothing moved,
    # there is nothing to substitute and the caller should not be handed an
    # object that merely looks like a change.
    return out if out != tool_input else None


def describe(tool: str, tool_input: dict, narrowed: dict) -> str:
    """One clause saying exactly what was substituted, and how to undo it.

    A rewrite the model cannot see is a lie about what it read, and it will act
    on the truncated result as though it were the whole thing. So the clause is
    not optional decoration on the mechanism — it is the half that keeps the
    mechanism honest, and it names the parameter so the way back is a call the
    model already knows how to write.
    """
    key = BOUNDS.get(tool, '')
    if not key or key not in narrowed:
        return ''
    was = bounded_at(tool, tool_input)
    got = narrowed[key]
    subject = 'lines' if tool == 'Read' else 'matches'
    from_part = f'{was:,}' if was is not None else 'unbounded'
    return (f'bounded to {got:,} {subject} ({key}={got}, was {from_part}); '
            f're-issue with a larger {key} if you need the rest')
