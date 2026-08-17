"""Formatting primitives shared by every report.

Fifteen modules were each hand-rolling `f"${x:>9,.2f}"` and column widths in
f-strings, and they had drifted: the same dollar figure printed as `$1,234.56`
in one report, `$1235` in another, and `1234.6` in a third. That is not a
cosmetic problem in a measurement tool -- a reader comparing two reports has to
know whether the difference is real or a format string.

Three rules this encodes, none of which are obvious in isolation:

* **Money below a cent is not `$0.00`.** A per-turn figure of `$0.004` rendered
  as `$0.00` reads as free. `money()` widens the precision instead of lying.
* **Colour is opt-out AND opt-in.** `NO_COLOR` disables it, a non-TTY stdout
  disables it, `ADDER_COLOR` forces it either way for a pager. Nothing here
  emits an escape sequence into a pipe by default.
* **Tables are computed, not typed.** Column widths come from the content, so
  adding a model with a longer id does not silently shift a column into its
  neighbour.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Sequence

# ANSI codes, deliberately few. A report that needs a fourth colour is a report
# that should be a table.
_CODES = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
}


def color_enabled(stream=None) -> bool:
    """Whether to emit ANSI codes at all.

    Checked at call time rather than import time: tests, pagers, and the prompt
    hook all change the answer, and an import-time constant makes it untestable.
    """
    # The `color` setting documents its vocabulary as "auto, always, or never"
    # and this only ever compared against "1" and "0" -- so `ADDER_COLOR=always`
    # and `ADDER_COLOR=never`, the two spellings the tool tells people to use,
    # both fell through to the TTY check and did nothing.
    want = os.environ.get("ADDER_COLOR", "").strip().lower()
    if want in ("1", "always", "true", "yes", "on"):
        return True
    # `NO_COLOR` per the convention at no-color.org: set AND non-empty. An
    # empty value is not a request for monochrome, and treating it as one
    # silences colour for anyone with `NO_COLOR=` in a stale profile.
    if os.environ.get("NO_COLOR", "").strip():
        return False
    if want in ("0", "never", "false", "no", "off"):
        return False
    stream = stream or sys.stdout
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def paint(text: str, style: str, *, stream=None) -> str:
    """`text` in `style`, or unchanged when colour is off or the style is unknown."""
    if not text or style not in _CODES or not color_enabled(stream):
        return text
    return f"{_CODES[style]}{text}{_CODES['reset']}"


def money(x: float, *, width: int = 0, sign: bool = False) -> str:
    """USD, with enough precision that a small number does not read as zero.

    Thresholds, not a single format: $1,204.55 wants two decimals and $0.0004
    wants four, and rendering the second as $0.00 is how a real per-turn cost
    becomes "free" in a report.
    """
    a = abs(x)
    if a >= 1:
        s = f"{x:,.2f}"
    elif a >= 0.01:
        s = f"{x:,.3f}"
    elif a >= 0.0001:
        s = f"{x:,.4f}"
    elif a > 0:
        s = f"{x:,.6f}"
    else:
        s = "0.00"
    out = f"${s}" if not s.startswith("-") else f"-${s[1:]}"
    # The sign goes outside the currency symbol, not between it and the digits.
    # `+$1,234.50` is what a delta column reads as; `$+1,234.50` is what the
    # obvious ordering produced, and it does not line up with the `-$1,234.50`
    # rendered directly beside it in the same column.
    if sign and x > 0:
        out = "+" + out
    return out.rjust(width) if width else out


def tokens(n: float, *, width: int = 0) -> str:
    """Token counts as humans read them: 544K, 1.2M, 900."""
    a = abs(n)
    if a >= 1_000_000:
        s = f"{n / 1_000_000:.1f}M"
    elif a >= 10_000:
        s = f"{n / 1_000:.0f}K"
    elif a >= 1_000:
        s = f"{n / 1_000:.1f}K"
    else:
        s = f"{n:,.0f}"
    return s.rjust(width) if width else s


def pct(x: float, *, digits: int = 0, width: int = 0, sign: bool = False) -> str:
    """A fraction in [0,1] as a percentage. Pass 0.23, not 23."""
    s = f"{x:+.{digits}%}" if sign else f"{x:.{digits}%}"
    return s.rjust(width) if width else s


def duration(seconds: float) -> str:
    """Seconds as the coarsest unit that stays readable."""
    s = abs(float(seconds))
    if s < 90:
        return f"{s:.0f}s"
    if s < 5400:
        return f"{s / 60:.0f}m"
    if s < 172_800:
        return f"{s / 3600:.1f}h"
    return f"{s / 86_400:.1f}d"


def bar(fraction: float, width: int = 20, *, fill: str = "█", empty: str = "·") -> str:
    """A proportion as a fixed-width bar. Clamped, so a >100% share cannot overflow."""
    f = max(0.0, min(1.0, fraction))
    n = round(f * width)
    return fill * n + empty * (width - n)


def table(
    rows: Iterable[Sequence[object]],
    headers: Sequence[str] | None = None,
    *,
    align: str = "",
    indent: str = "  ",
    gap: str = "  ",
) -> list[str]:
    """Render rows as aligned columns. Widths come from the content.

    `align` is one character per column: `<` left, `>` right, `^` centre.
    Short or missing alignment strings default to left for the first column and
    right for the rest, which is what every table in this repo wanted anyway.
    """
    body = [[("" if c is None else str(c)) for c in r] for r in rows]
    head = [str(h) for h in headers] if headers else []
    ncols = max((len(r) for r in body + ([head] if head else [])), default=0)
    if not ncols:
        return []
    for r in body:
        r.extend([""] * (ncols - len(r)))
    if head:
        head.extend([""] * (ncols - len(head)))
    widths = [
        max([len(r[i]) for r in body] + ([len(head[i])] if head else [0]))
        for i in range(ncols)
    ]
    aligns = [
        align[i] if i < len(align) else ("<" if i == 0 else ">")
        for i in range(ncols)
    ]
    out: list[str] = []
    if head:
        out.append(indent + gap.join(
            f"{head[i]:{aligns[i]}{widths[i]}}" for i in range(ncols)).rstrip())
    for r in body:
        out.append(indent + gap.join(
            f"{r[i]:{aligns[i]}{widths[i]}}" for i in range(ncols)).rstrip())
    return out


def heading(text: str, *, rule: str = "") -> list[str]:
    """A section title, optionally underlined. Returns lines, never prints."""
    lines = [f"  {text}"]
    if rule:
        lines.append("  " + rule * len(text))
    return lines


def kv(label: str, value: str, *, width: int = 22, indent: str = "  ") -> str:
    """`label ....... value`, aligned the way every summary block wants."""
    return f"{indent}{label:<{width}}{value}"


def warn(text: str) -> str:
    return paint(f"  ⚠ {text}", "yellow")


def bullet(text: str, *, indent: str = "    ") -> str:
    return f"{indent}- {text}"


def wrap(text: str, width: int = 78, indent: str = "  ") -> list[str]:
    """Greedy wrap. `textwrap` would do, but every caller wants the indent too."""
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width - len(indent):
            lines.append(indent + cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        lines.append(indent + cur)
    return lines
