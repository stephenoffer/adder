"""One ISO-8601 reader, because `datetime.fromisoformat` depends on the Python.

Transcripts stamp time the way JavaScript and Go print it: `...T12:00:00.123Z`,
sometimes nine fractional digits, sometimes two, sometimes `+0000` with no
colon. Python 3.11 reads all of those. Python 3.10 -- which this package
supports -- reads none of them, and every caller had the same
`fromisoformat(ts.replace("Z", "+00:00"))` wrapped in `except ValueError:
return None`. So on 3.10 a turn stamped `2026-01-02T00:00:00.123456789Z` was
silently undated: dropped from every `--since` window and counted as
"undateable" in a report that looked like an answer.

`parse_iso` accepts the extended form both versions should have agreed on, and
reads it with a regular expression rather than by normalising and handing it
to `fromisoformat`, so a string it accepts on one version it accepts on the
other. 3.11's wider grammar (`20260102T000000`, week dates) is deliberately not
inherited; nothing in a transcript is written that way.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

_ISO = re.compile(
    r"""^\s*
    (?P<y>\d{4})-(?P<mo>\d{2})-(?P<d>\d{2})
    (?:[T ]
        (?P<h>\d{2}):(?P<mi>\d{2})
        (?::(?P<s>\d{2})(?:[.,](?P<f>\d+))?)?
        \s*
        (?P<tz>[Zz]|[+-]\d{2}(?::?\d{2}(?::?\d{2})?)?)?
    )?
    \s*$""",
    re.X,
)


def _offset(tz: str) -> timezone:
    if tz in ("Z", "z"):
        return timezone.utc
    sign = -1 if tz[0] == "-" else 1
    digits = tz[1:].replace(":", "")
    hours, minutes, seconds = int(digits[:2]), int(digits[2:4] or 0), int(digits[4:6] or 0)
    delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)
    if delta >= timedelta(hours=24):
        raise ValueError(f"offset out of range: {tz!r}")
    return timezone(sign * delta)


def parse_iso(text: str) -> datetime:
    """An ISO-8601 timestamp as a `datetime`, the same on Python 3.10 and 3.11.

    Accepts a date alone (midnight, naive), `T` or a space between date and
    time, seconds optional, any number of fractional digits (truncated to
    microseconds, never rounded, so `.9999999` cannot carry into the next
    second), and an offset of `Z`, `+HH`, `+HHMM` or `+HH:MM`. No offset gives
    a naive datetime, as `fromisoformat` does.

    Raises `ValueError` for anything else, and `TypeError` for a non-string,
    exactly as `fromisoformat` does -- so it drops into the `except (ValueError,
    TypeError)` every caller already has.
    """
    if not isinstance(text, str):
        raise TypeError(f"parse_iso() argument must be str, not {type(text).__name__}")
    m = _ISO.match(text)
    if m is None:
        raise ValueError(f"Invalid isoformat string: {text!r}")
    g = m.groupdict()
    frac = (g["f"] or "")[:6].ljust(6, "0")
    tz = _offset(g["tz"]) if g["tz"] else None
    return datetime(
        int(g["y"]), int(g["mo"]), int(g["d"]),
        int(g["h"] or 0), int(g["mi"] or 0), int(g["s"] or 0), int(frac),
        tzinfo=tz,
    )
