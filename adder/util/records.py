"""Reading a field out of an untrusted JSON record without raising.

Nine modules scan `~/.claude/projects/**` line by line, and every one of them
reached into a record with the same idiom:

    content = (d.get("message") or {}).get("content")

which reads as defensive and is not. `or {}` only fires on a *falsy* value, so
it catches `None` and `{}` and misses every other wrong type. One record whose
`message` is a string -- a truncated write, a client-side error placeholder, a
hand-edited transcript -- makes that `.get` raise `AttributeError`, and none of
those scanners has a handler: `adder context`, `spec`, `quality`, `tools`,
`reread` and `doctor` each ended in a traceback on a single malformed line.

The tool reads this directory and does not write it, so its shape is not a
contract. The rule the rest of the package follows is that a record it cannot
make sense of costs one record, never the report.

Nothing here knows what a turn is or what anything costs; it is dict access
with the types checked.
"""

from __future__ import annotations

from typing import Any


def mapping(d: Any, *keys: str) -> dict:
    """The nested mapping at `keys`, or `{}` if any step is not one.

    `mapping(rec, "message", "usage")` is the safe spelling of
    `((rec.get("message") or {}).get("usage") or {})`, and unlike that one it
    is also correct when an intermediate value is a string or a number.
    """
    if not isinstance(d, dict):
        return {}
    for k in keys:
        d = d.get(k)
        if not isinstance(d, dict):
            return {}
    return d


def blocks(d: Any, *keys: str) -> list:
    """The content block list at `keys`, or `[]`.

    Message content is a list of blocks or a bare string; a caller iterating it
    wants the list form and nothing else. Returning `[]` for the string form is
    deliberate -- a scanner looking for `tool_use` blocks has nothing to do
    with a plain-text message.
    """
    if not isinstance(d, dict):
        return []
    for k in keys[:-1]:
        d = d.get(k)
        if not isinstance(d, dict):
            return []
    got = d.get(keys[-1]) if keys else d
    return got if isinstance(got, list) else []


def text(d: Any, *keys: str, default: str = "") -> str:
    """The string at `keys`, or `default`. Never a coerced dict or list."""
    if keys[:-1]:
        d = mapping(d, *keys[:-1])
    if not isinstance(d, dict):
        return default
    got = d.get(keys[-1]) if keys else d
    return got if isinstance(got, str) else default
