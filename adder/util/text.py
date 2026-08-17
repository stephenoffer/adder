"""Turning message content into a character count, and a character count into
an estimated token count.

These two functions are the most-copied thing in the repository -- five modules
had reached into `adder.measure.window.context` for them, which made a report
module a dependency of the transcript reader and of the PreToolUse guard. They
are here because they qualify for this package: neither one knows what a turn
is, what a session is, or what anything costs.

The estimate is deliberately crude and its bias is documented at the function.
"""

from __future__ import annotations

import json

CHARS_PER_TOKEN = 4.0


def est_tokens(text: str) -> int:
    """Rough token count from character length.

    Four characters per token is the usual English approximation and it is
    wrong for code, JSON, and non-Latin scripts in both directions. It is used
    only to apportion *shares* of a growth figure that was measured from billed
    token counts, never to produce a token count that is reported on its own --
    so a biased estimator moves the split between two sources, not the total.

    Anything that is not a string is coerced rather than raising. Every caller
    reads a `text` field off a transcript record this tool does not write, and
    the idiom they all use -- `est_tokens(b.get("text") or "")` -- only guards
    the falsy case: a block whose `text` is a number sails through it and
    raised `TypeError: object of type 'int' has no len()` out of `adder
    context`, which has no handler.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return int(len(text) / CHARS_PER_TOKEN)


def flatten_text(content) -> str:
    """Flatten a message `content` field to plain text for size estimation."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for b in content:
        if isinstance(b, str):
            parts.append(b)
        elif isinstance(b, dict):
            if isinstance(b.get("text"), str):
                parts.append(b["text"])
            elif isinstance(b.get("content"), (str, list)):
                parts.append(flatten_text(b["content"]))
            elif isinstance(b.get("input"), dict):
                parts.append(json.dumps(b["input"]))
    return "".join(parts)
