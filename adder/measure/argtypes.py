"""Argparse types for the numbers a report takes on its command line.

Each report declared `type=int` or `type=float` and trusted the value, so
`limits --hours 0`, `budget --limit -5`, `sessions --top -1`, `anomaly --z -1`
and `horizon --at -5` all ran. None of them failed loudly: a zero-hour window
reconstructs one block per turn, a negative `--top` slices off the tail of the
ranking and prints the rest as "the top", and a negative threshold flags every
turn as unusual. Each printed a well-formatted answer to a question nobody can
have meant, which for a cost tool is the failure mode rather than an edge case.

Rejected here, at parse time, so the error is argparse's usual usage line and
exit status 2 rather than a report built on the bad value. Non-finite floats
are refused too: `float("nan")` parses and then compares false with
everything, which silently disables whatever threshold it was given to.
"""

from __future__ import annotations

import argparse
import math


def _number(text: str, kind: type) -> float:
    try:
        v = kind(text)
    except (TypeError, ValueError):
        what = "a whole number" if kind is int else "a number"
        raise argparse.ArgumentTypeError(f"expected {what}, got {text!r}") from None
    if kind is float and not math.isfinite(v):
        raise argparse.ArgumentTypeError(f"expected a finite number, got {text!r}")
    return v


def positive_int(text: str) -> int:
    """An integer >= 1: a count of rows, turns, or anything else to show."""
    v = int(_number(text, int))
    if v < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {v}")
    return v


def nonneg_int(text: str) -> int:
    """An integer >= 0: an index, where turn 0 is a real place to stand."""
    v = int(_number(text, int))
    if v < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, got {v}")
    return v


def positive_float(text: str) -> float:
    """A finite number > 0: a duration or a threshold that zero would disable."""
    v = _number(text, float)
    if v <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {text}")
    return v


def nonneg_float(text: str) -> float:
    """A finite number >= 0: an amount where zero already means "none"."""
    v = _number(text, float)
    if v < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, got {text}")
    return v
