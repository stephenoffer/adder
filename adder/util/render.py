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
* **A number that is not a number says so.** `money(nan)` printed `$0.00` --
  free, again -- and `money(inf)` printed `$inf`. Both render as `n/a`
  (`UNKNOWN`), as does `tokens` of either, and `bar(nan)` is empty, not full.
"""

from __future__ import annotations

import math
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

# What `money` and `tokens` print for NaN or an infinity. One spelling, so a
# reader learns it once; not `$0.00`, which reads as free.
UNKNOWN = "n/a"


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
    if not math.isfinite(x):
        return UNKNOWN.rjust(width) if width else UNKNOWN
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
    """Token counts as humans read them: 544K, 1.2M, 900.

    The thresholds are where the *rounded* figure crosses a unit, not the raw
    one: 999,950 is 999.95K, which `.0f` prints as `1000K`. It is `1.0M`.
    """
    if not math.isfinite(n):
        return UNKNOWN.rjust(width) if width else UNKNOWN
    a = abs(n)
    if a >= 999_500:
        s = f"{n / 1_000_000:.1f}M"
    elif a >= 9_950:
        s = f"{n / 1_000:.0f}K"
    elif a >= 1_000:
        s = f"{n / 1_000:.1f}K"
    else:
        s = f"{n:,.0f}"
    return s.rjust(width) if width else s


def pct(x: float, *, digits: int = 0, width: int = 0, sign: bool = False) -> str:
    """A fraction in [0,1] as a percentage. Pass 0.23, not 23."""
    if not math.isfinite(x):
        return UNKNOWN.rjust(width) if width else UNKNOWN          # not "nan%"
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


def clip_path(path: str, width: int = 60, *, home: str | None = None) -> str:
    """A path cut to `width` at a separator, with the file name always kept.

    `path[-60:]` cut wherever the count landed, so `wolfgang-v2/.claude/...`
    printed as `v2/.claude/...`, a directory that does not exist, and
    `ident[:70]` kept the tool and the home directory and dropped the file
    the finding was about. The head goes first, then whole leading segments,
    and a leading `…/` says something was removed. `home` abbreviates to `~`;
    it defaults to the current one, looked up when called.

    A `Tool:` prefix (`Read:/path`) is kept whole and the rest clipped.
    """
    if width <= 0:
        return ""
    tool, sep, rest = path.partition(":")
    # The prefix is kept only when it leaves room for at least `…` and one
    # character. It was kept unconditionally with `max(1, ...)` for the rest,
    # so a prefix as long as the width returned more than the width.
    if sep and rest.startswith(("/", "~")) and "/" not in tool and width - len(tool) - 1 >= 2:
        return tool + ":" + clip_path(rest, width - len(tool) - 1, home=home)
    home = os.path.expanduser("~") if home is None else home
    if home and home != "/" and (path == home or path.startswith(home + "/")):
        path = "~" + path[len(home):]
    if len(path) <= width:
        return path
    parts = path.split("/")
    kept = parts[-1]
    for seg in reversed(parts[:-1]):
        if len(seg) + 1 + len(kept) + 2 > width:
            break
        kept = seg + "/" + kept
    if len(kept) + 2 > width:
        # `kept[-0:]` is the whole string, so width 1 needs its own case.
        return "…" + (kept[-(width - 1):] if width > 1 else "")
    return "…/" + kept


def bar(fraction: float, width: int = 20, *, fill: str = "█", empty: str = "·") -> str:
    """A proportion as a fixed-width bar. Clamped, so a >100% share cannot overflow."""
    # NaN compares false with everything, so `min(1.0, nan)` is 1.0 and an
    # unknown share drew as a full bar. Unknown is drawn as nothing.
    f = 0.0 if math.isnan(fraction) else max(0.0, min(1.0, fraction))
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


def nothing_found(subject: str, root: str, *, window: str = "",
                  unread: int = 0, example: str = "") -> str:
    """The empty report: what was looked for, where, and the way forward.

    Six commands each printed their own bare sentence here, and a run that
    finds nothing is exactly when a bare sentence is least useful. "No priced
    turns found under /Users/x/.claude/projects" reads as a bug when the real
    answer is usually one of three mundane things: the window excluded every
    turn, the transcripts are somewhere this invocation did not look, or there
    are transcripts right there that the reader could not parse.

    That third case is the one worth separating, and it used to print as the
    first. Point adder at a directory of logs it does not understand and the
    answer was "nothing here has been priced yet ... this fills in after a
    session or two" -- advice to wait, given to somebody whose files are
    already on disk and whose real problem is a format. Anyone not running
    Claude Code met that message first.

    `window` is `Window.describe()` and `unread` the number of candidate files
    that produced no turn. Both are passed as plain values rather than a
    `Window` or a list of paths because `util` sits below `core` and may not
    import it.
    """
    filtered = bool(window) and window != "everything on disk"
    out = [f"No {subject} found under {root}" + (f" matching {window}." if filtered else ".")]
    if filtered:
        out += ["", *wrap("Every turn on disk was excluded by that filter. Widen it, or "
                          "drop --since/--project to see what is there.")]
    elif unread:
        out += ["", *wrap(f"{unread:,} file{'s' if unread != 1 else ''} "
                          f"{'are' if unread != 1 else 'is'} there, but none carried "
                          "a usage record this reader recognised, so there is nothing "
                          "to price. This is a format problem, not an empty history."),
                "",
                *wrap("Handled: Claude Code, Codex CLI, Gemini CLI and OpenCode "
                      "transcripts, the Anthropic and OpenAI APIs (chat and "
                      "responses), the Gemini API, OpenTelemetry "
                      "`gen_ai.usage.*` spans, and any record with plain "
                      "input/output token counts. A log with no token counts in "
                      "it cannot be priced by anything."),
                ""]
        if example:
            out += [f"      adder trace {example}", "",
                    *wrap("on a single file is the quickest way to see what it made "
                          "of one.")]
    else:
        out += ["", *wrap("Nothing here has been priced yet. Your agent writes a "
                          "transcript per session, so this fills in after a session "
                          "or two."),
                "",
                *wrap("Transcripts somewhere else? Name the agent (`claude`, "
                      "`codex`, `gemini`, `opencode`) or the directory as the "
                      "first argument, or set `root` in .adder.json "
                      "(`adder config --init` prints a template)."),
                "",
                *wrap("What does not need history is the half that prevents spend "
                      "rather than reporting it:"),
                "",
                "      adder auto on --full",
                "",
                *wrap("which prices a tool call before its result lands in your "
                      "context.")]
    return "\n".join(out)
