"""Get the priced turns out, so the analysis can continue somewhere else.

Everything else in this repo is an opinion about the data. This is the data.
Someone will want a pivot table, a notebook, a dashboard, or a join against
their own CI records, and the alternative to an export is re-implementing the
deduplication and the date-aware pricing in whatever tool they are using --
which is how a second, wrong set of numbers gets published.

Three formats, one shape
------------------------
`csv` for spreadsheets, `jsonl` for streaming into anything, `json` for a
single document. The columns are identical across all three, so a script can
switch format without changing its field names.

Privacy
-------
No message content is ever exported. Transcripts contain source code, file
paths, and prompts; a cost export needs none of it, and a file that quietly
carried it would be pasted into a ticket within the week. What leaves here is
token counts, prices, timestamps, model ids, and tool *names*.

Writing
-------
Output goes to stdout unless `-o` names a file, and `-o` refuses to overwrite
without `--force`. The tool does not delete or replace anything the user did
not ask it to.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from collections.abc import Iterator
from datetime import date
from pathlib import Path

from adder.core.filters import day_of
from adder.core.trace import Session

FORMATS = ("csv", "jsonl", "json")
GRAINS = ("turn", "session", "day")

TURN_COLUMNS = (
    "timestamp", "session", "project", "model", "speed", "ttl", "effort",
    "sidechain", "uncached_in", "cache_read", "cache_write", "context", "out",
    "thinking", "cost", "input_cost", "output_cost", "tools",
)

SESSION_COLUMNS = (
    "session", "project", "started", "ended", "wall_seconds", "turns", "cost",
    "cost_per_turn", "peak_context", "avg_context", "base_context",
    "out_tokens", "thinking_tokens", "compactions", "rebuilds", "models",
)

DAY_COLUMNS = ("day", "turns", "sessions", "cost", "out_tokens", "context_tokens")


def turn_rows(sessions: dict[str, Session], on: date | None = None) -> Iterator[dict]:
    for s in sorted(sessions.values(), key=lambda x: x.id):
        for t in s.turns:
            yield {
                "timestamp": t.ts or "",
                "session": t.session,
                "project": t.project,
                "model": t.model,
                "speed": t.speed,
                "ttl": t.ttl,
                "effort": t.effort,
                "sidechain": int(bool(t.sidechain)),
                "uncached_in": t.uncached_in,
                "cache_read": t.cache_read,
                "cache_write": t.cache_write,
                "context": t.context,
                "out": t.out,
                "thinking": t.thinking,
                "cost": round(t.cost(on), 8),
                "input_cost": round(t.input_cost(on), 8),
                "output_cost": round(t.output_cost(on), 8),
                "tools": " ".join(t.tools),
            }


def session_rows(sessions: dict[str, Session], on: date | None = None) -> Iterator[dict]:
    for s in sorted(sessions.values(), key=lambda x: -x.cost):
        cost = s.cost_on(on)
        yield {
            "session": s.id,
            "project": s.project,
            "started": s.started.isoformat() if s.started else "",
            "ended": s.ended.isoformat() if s.ended else "",
            "wall_seconds": round(s.wall_seconds, 1),
            "turns": s.n_turns,
            "cost": round(cost, 6),
            "cost_per_turn": round(cost / max(1, s.n_turns), 8),
            "peak_context": s.peak_context,
            "avg_context": s.avg_context,
            "base_context": s.base_context,
            "out_tokens": s.out_tokens,
            "thinking_tokens": s.thinking_tokens,
            "compactions": s.compactions(),
            "rebuilds": len(s.cache_misses()),
            "models": " ".join(sorted(s.models)),
        }


def day_rows(sessions: dict[str, Session], on: date | None = None) -> Iterator[dict]:
    """One row per calendar day. Undated turns are their own bucket, not dropped."""
    acc: dict[str, dict] = {}
    for s in sessions.values():
        for t in s.turns:
            w = t.when
            key = day_of(w).isoformat() if w else "undated"
            row = acc.setdefault(key, {"day": key, "turns": 0, "sessions": set(),
                                       "cost": 0.0, "out_tokens": 0,
                                       "context_tokens": 0})
            row["turns"] += 1
            row["sessions"].add(t.session)
            row["cost"] += t.cost(on)
            row["out_tokens"] += t.out
            row["context_tokens"] += t.context
    for key in sorted(acc):
        row = acc[key]
        row["sessions"] = len(row["sessions"])
        row["cost"] = round(row["cost"], 6)
        yield row


def rows_for(sessions: dict[str, Session], grain: str,
             on: date | None = None) -> tuple[tuple[str, ...], list[dict]]:
    if grain == "turn":
        return TURN_COLUMNS, list(turn_rows(sessions, on))
    if grain == "session":
        return SESSION_COLUMNS, list(session_rows(sessions, on))
    if grain == "day":
        return DAY_COLUMNS, list(day_rows(sessions, on))
    raise ValueError(f"unknown grain {grain!r}; known: {', '.join(GRAINS)}")


def render(columns: tuple[str, ...], rows: list[dict], fmt: str) -> str:
    if fmt == "csv":
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(columns), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
        return buf.getvalue()
    if fmt == "jsonl":
        return "".join(json.dumps(r, sort_keys=False) + "\n" for r in rows)
    if fmt == "json":
        return json.dumps({"columns": list(columns), "rows": rows}, indent=2) + "\n"
    raise ValueError(f"unknown format {fmt!r}; known: {', '.join(FORMATS)}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    from adder.core.filters import add_arguments as add_window
    from adder.core.filters import load as load_window

    ap = argparse.ArgumentParser(
        prog="adder export",
        description="Export priced turns, sessions, or days. No message content.")
    add_window(ap)
    ap.add_argument("--format", choices=FORMATS, default="csv",
                    help="output format (default: %(default)s)")
    ap.add_argument("--grain", choices=GRAINS, default="turn",
                    help="one row per what (default: %(default)s)")
    ap.add_argument("-o", "--out", metavar="PATH",
                    help="write here instead of stdout")
    ap.add_argument("--force", action="store_true",
                    help="allow -o to overwrite an existing file")
    a = ap.parse_args(argv)

    sessions, _window = load_window(a)

    columns, rows = rows_for(sessions, a.grain)
    text = render(columns, rows, a.format)

    if not a.out:
        sys.stdout.write(text)
        return 0

    dest = Path(a.out).expanduser()
    if dest.exists() and not a.force:
        print(f"adder export: {dest} exists; pass --force to replace it",
              file=sys.stderr)
        return 1
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
    except OSError as e:
        print(f"adder export: cannot write {dest}: {e}", file=sys.stderr)
        return 1
    print(f"wrote {len(rows):,} {a.grain} rows to {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
