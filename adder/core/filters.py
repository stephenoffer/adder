"""One definition of "which turns are we talking about", shared by every report.

`quality` grew `--since`. `verify` grew a cutover date. Nothing else could be
windowed at all, so the honest answer to "what did last week cost?" was "run
`trace` over your whole history and subtract". Worse, the two date filters that
did exist disagreed: one compared dates, the other compared datetimes, and a
turn at 23:59 UTC landed on different sides of the same boundary.

So the window is defined once, here, with the boundary rule written down:

    --since DATE   keeps turns on or after 00:00 of DATE   (inclusive)
    --until DATE   keeps turns strictly before 00:00 of DATE (exclusive)

Half-open, so `--since 2026-08-01 --until 2026-09-01` is exactly August and two
adjacent windows never double-count a turn.

Turns with no timestamp
-----------------------
A turn whose record carried no `timestamp` cannot be placed in time. When a
window is requested those turns are **dropped**, and the count of dropped turns
is reported rather than absorbed: a report that silently includes undateable
spend in every window is a report whose windows do not sum to the total.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

# Relative windows, because `--since 7d` is what people type.
_RELATIVE = re.compile(r"^(\d+)\s*([dwmy])$", re.I)
_UNIT_DAYS = {"d": 1, "w": 7, "m": 30, "y": 365}


def parse_date(text: str, *, today: date | None = None) -> date:
    """`2026-08-01`, `7d`, `2w`, `today`, or `yesterday`.

    A cost tool is used interactively and nobody wants to compute the date
    seven days ago in their head. Absolute dates still parse first, so nothing
    that already worked changes meaning.
    """
    s = (text or "").strip()
    if not s:
        raise ValueError("empty date")
    today = today or date.today()
    low = s.lower()
    if low == "today":
        return today
    if low == "yesterday":
        return today - timedelta(days=1)
    m = _RELATIVE.match(low)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        return today - timedelta(days=n * _UNIT_DAYS[unit])
    try:
        return date.fromisoformat(s)
    except ValueError as e:
        raise ValueError(
            f"cannot read {text!r} as a date; use YYYY-MM-DD, 7d, 2w, today, "
            "or yesterday"
        ) from e


def day_of(ts) -> date | None:
    """The **local** calendar day a transcript timestamp falls on, or None.

    Lives here rather than in the report that first needed it: a window filter
    cannot ask a report module what day a record is from without making the
    transcript reader depend on a report. Returns None for a missing or
    unparseable timestamp, which every caller must treat as "cannot be windowed"
    rather than as "today".

    Local, not UTC, and the difference is not small. Transcripts stamp UTC;
    `parse_date` builds `today`/`7d` from `date.today()`, which is local. Taking
    `.date()` off the UTC instant compared one against the other, so every
    evening turn west of Greenwich was filed under tomorrow: 6,778 of 28,191
    turns here -- 24% of them, carrying $1,484 of spend -- landed on the wrong
    day. That reaches `--since`, `--until`, the `--by day` breakdown, the daily
    budget burn-down and the `verify` cutover, all of which are answering a
    question the operator asked in their own timezone.

    A timestamp with no offset is taken as already local; there is nothing
    better to assume and it is what a naive `.date()` did anyway.
    """
    if not ts:
        return None
    try:
        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return (t if t.tzinfo is None else t.astimezone()).date()


def _date_arg(text: str) -> date:
    try:
        return parse_date(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


@dataclass
class Window:
    """A turn filter. Every field is optional and an empty Window keeps everything."""

    since: date | None = None
    until: date | None = None
    projects: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    sessions: tuple[str, ...] = ()
    sidechain: bool | None = None      # None = both, True = only, False = exclude
    min_turns: int = 0
    dropped_undated: int = field(default=0, compare=False)

    @property
    def dated(self) -> bool:
        return self.since is not None or self.until is not None

    @property
    def active(self) -> bool:
        return bool(self.dated or self.projects or self.models or self.sessions
                    or self.sidechain is not None or self.min_turns)

    # -- turn-level ------------------------------------------------------
    def keeps_turn(self, turn) -> bool:
        if self.models and not any(turn.model.startswith(m) for m in self.models):
            return False
        if self.projects and not any(p.lower() in turn.project.lower()
                                     for p in self.projects):
            return False
        if self.sessions and not any(turn.session.startswith(s) for s in self.sessions):
            return False
        if self.sidechain is not None and bool(turn.sidechain) != self.sidechain:
            return False
        if self.dated:
            when = turn.when
            if when is None:
                return False
            d = day_of(when)
            if self.since and d < self.since:
                return False
            if self.until and d >= self.until:
                return False
        return True

    def turns(self, turns: Iterable) -> list:
        """Filter a turn iterable, counting the undateable ones dropped."""
        kept = []
        for t in turns:
            if self.dated and t.when is None:
                self.dropped_undated += 1
                continue
            if self.keeps_turn(t):
                kept.append(t)
        return kept

    # -- raw-record level -------------------------------------------------
    def keeps_record(self, record: dict, project: str = "") -> bool:
        """Filter a raw transcript record, for the scanners that do not build Turns.

        `tools`, `context`, `quality`, and `dispatch` read message content
        rather than billed usage, so they never construct a `Turn` and cannot
        use `keeps_turn`. They were each applying only the date part of the
        window, which meant `adder tools --session abc` silently reported the
        whole corpus -- a filter that is accepted and ignored is worse than one
        that is rejected, because the number looks like an answer.

        `--model-filter` is the one field that genuinely cannot apply here: a
        `tool_result` block carries no model. `ignores_model` lets a caller say
        so out loud instead of quietly widening the report.
        """
        if self.sessions:
            sid = str(record.get("sessionId") or "")
            if not any(sid.startswith(s) for s in self.sessions):
                return False
        if self.projects and not any(p.lower() in project.lower()
                                     for p in self.projects):
            return False
        if (self.sidechain is not None
                and bool(record.get("isSidechain")) != self.sidechain):
            return False
        if self.dated:
            day = day_of(record.get("timestamp"))
            if day is None:
                return False
            if self.since and day < self.since:
                return False
            if self.until and day >= self.until:
                return False
        return True

    @property
    def ignores_model(self) -> bool:
        """True when a model filter was given that a raw-record scan cannot honour."""
        return bool(self.models)

    @property
    def filters_records(self) -> bool:
        """Whether `keeps_record` would reject anything at all."""
        return bool(self.dated or self.projects or self.sessions
                    or self.sidechain is not None)

    # -- session-level ---------------------------------------------------
    def apply(self, sessions: dict) -> dict:
        """A new session map holding only the turns that pass.

        Sessions are rebuilt rather than mutated: the caller may be holding the
        same objects from a cached parse, and quietly emptying them would make
        the second report on the same data disagree with the first.
        """
        from adder.core.trace import Session

        # Reset first: `apply` is idempotent from the caller's point of view,
        # and a counter that accumulates across calls makes the second report
        # on the same data claim twice as many undateable turns as the first.
        self.dropped_undated = 0
        out: dict = {}
        for sid, s in sessions.items():
            kept = self.turns(s.turns)
            if len(kept) < max(1, self.min_turns):
                continue
            fresh = Session(s.id, s.project)
            fresh.turns = kept
            out[sid] = fresh
        return out

    def describe(self) -> str:
        bits = []
        if self.since:
            bits.append(f"since {self.since}")
        if self.until:
            bits.append(f"before {self.until}")
        if self.projects:
            bits.append("project ~ " + "|".join(self.projects))
        if self.models:
            bits.append("model ~ " + "|".join(self.models))
        if self.sessions:
            bits.append("session ~ " + "|".join(self.sessions))
        if self.sidechain is True:
            bits.append("subagent turns only")
        if self.sidechain is False:
            bits.append("main-chain turns only")
        if self.min_turns:
            bits.append(f"sessions with >= {self.min_turns} turns")
        return ", ".join(bits) if bits else "everything on disk"

    @classmethod
    def from_args(cls, a: argparse.Namespace) -> Window:
        side = None
        if getattr(a, "only_subagents", False):
            side = True
        elif getattr(a, "no_subagents", False):
            side = False
        return cls(
            since=getattr(a, "since", None),
            until=getattr(a, "until", None),
            projects=tuple(getattr(a, "project", None) or ()),
            models=tuple(getattr(a, "model_filter", None) or ()),
            sessions=tuple(getattr(a, "session", None) or ()),
            sidechain=side,
            min_turns=getattr(a, "min_turns", 0) or 0,
        )


def add_arguments(ap: argparse.ArgumentParser, *, root: bool = True) -> argparse.ArgumentParser:
    """Attach the standard window flags. Same names and meanings everywhere.

    Kept in one function so a command cannot accidentally ship `--from/--to`
    while its neighbour ships `--since/--until`.
    """
    if root:
        ap.add_argument("root", nargs="?", default=None,
                        help="transcript directory (default: the `root` setting)")
    g = ap.add_argument_group("window")
    g.add_argument("--since", type=_date_arg, metavar="DATE",
                   help="keep turns on or after DATE (YYYY-MM-DD, 7d, 2w, today)")
    g.add_argument("--until", type=_date_arg, metavar="DATE",
                   help="keep turns strictly before DATE (half-open)")
    g.add_argument("--project", action="append", metavar="SUBSTR",
                   help="keep projects whose directory name contains SUBSTR (repeatable)")
    g.add_argument("--model-filter", action="append", metavar="PREFIX",
                   help="keep turns whose model starts with PREFIX (repeatable)")
    g.add_argument("--session", action="append", metavar="ID",
                   help="keep sessions whose id starts with ID (repeatable)")
    g.add_argument("--min-turns", type=int, default=0, metavar="N",
                   help="drop sessions shorter than N turns")
    x = g.add_mutually_exclusive_group()
    x.add_argument("--only-subagents", action="store_true",
                   help="keep only sidechain (subagent) turns")
    x.add_argument("--no-subagents", action="store_true",
                   help="drop sidechain (subagent) turns")
    return ap


def root_of(a: argparse.Namespace | None = None) -> Path:
    """The transcript directory in effect: the argument, or the `root` setting.

    `root` is the first setting in the table and is described as "transcript
    directory every report reads". It was read by nothing. Every parser
    declared `default=str(DEFAULT_ROOT)`, so `a.root` was always truthy and the
    `or settings.get("root")` below could never fire -- `adder config` printed
    the configured path and every report opened `~/.claude/projects`.

    Resolving here rather than in twenty `main()` functions so the answer
    cannot differ between two commands run against the same configuration.
    """
    from adder.core import settings
    from adder.core.trace import DEFAULT_ROOT

    given = getattr(a, "root", None) if a is not None else None
    if given:
        return Path(str(given)).expanduser()
    try:
        return Path(str(settings.get("root"))).expanduser()
    except (KeyError, OSError, ValueError):
        return Path(DEFAULT_ROOT)


def load(a: argparse.Namespace, *, use_cache: bool | None = None) -> tuple[dict, Window]:
    """Load sessions under `a.root` and apply the window from `a`.

    Returns `(sessions, window)` -- the window comes back so a report can print
    what it filtered to, and so the caller can see how many undateable turns
    were dropped.
    """
    from adder.core import settings
    from adder.core.trace import load_sessions

    if use_cache is None:
        use_cache = bool(settings.get("cache"))
    sessions = load_sessions(root_of(a), use_cache=use_cache)
    w = Window.from_args(a)
    return (w.apply(sessions) if w.active else sessions), w


def span(sessions: dict) -> tuple[datetime | None, datetime | None]:
    """First and last timestamp across every turn. `(None, None)` if undateable."""
    lo = hi = None
    for s in sessions.values():
        for t in s.turns:
            w = t.when
            if w is None:
                continue
            if lo is None or w < lo:
                lo = w
            if hi is None or w > hi:
                hi = w
    return lo, hi
