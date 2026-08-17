"""Content admitted to the context twice: the cache the agent already had.

The cheapest token is the one already in the window
---------------------------------------------------
Every other report here asks what a token costs. This one asks a narrower
question with a much better answer: **was it already there?**

An agent that reads `pyproject.toml` on turn 8 and reads it again on turn 140
has not paid twice for one file. It has paid once for the file, once for the
second copy, and then it re-reads *both* copies on every turn after that. The
second copy buys nothing -- the first was still sitting in the prefix, served
at the cache rate -- so its entire lifetime cost is waste.

Two kinds of repeat, and only one of them is a mistake
-----------------------------------------------------
    redundant   the result is byte-identical to a copy already in context. The
                agent re-derived something it was still holding. Recoverable in
                full: not making the call costs nothing.
    refresh     the result changed (a file was edited, a test re-run). The call
                was justified. But the superseded copy is *still resident* and
                is still being re-read every turn, so a refresh has a cost too --
                it is just not one you fix by skipping the call.

Reporting them together would be dishonest in the expensive direction: it would
tell someone that re-running their test suite is waste. It is not. Only the
first bucket is offered as a saving.

Recurring reads are a memory problem, not a context problem
-----------------------------------------------------------
The same identity read in forty *different* sessions is not a re-read at all --
each session started empty and had to learn it. That is the one case where the
fix is `CLAUDE.md` or a memory note, and it is a fix with a price: a resident
token is re-read on every turn of every session forever (`memory` prices it),
while a repeated read is paid only in the sessions that make it. This module
computes both sides and names the break-even, because "write it down" is
excellent advice right up to the point where the note costs more than the reads
it replaces.

What is measured, and what is estimated
---------------------------------------
Result sizes are estimated from characters (`util.text.est_tokens`); token
counts here are inputs to a *comparison* between two options, never reported as
billed quantities. The carry multiplier and the residency decay are the
measured ones from `carry`, so a re-read late in a session is priced lower than
one early in it -- which is the whole reason the ranking is by dollars and not
by count.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from adder.core import settings as _settings
from adder.core.filters import root_of as _root_of
from adder.core.trace import DEFAULT_ROOT, transcripts
from adder.util.records import mapping
from adder.util.text import est_tokens, flatten_text

# Results below this are not worth a line in a report: a 200-token `ls` re-run
# ten times is 2,000 tokens, and chasing it costs more attention than it saves.
MIN_RESULT_TOKENS = 400

# Commands whose whole purpose is to report changing state. Re-running these is
# not a re-read, it is the point, and a report that flags them teaches the
# reader to ignore the report.
VOLATILE_PREFIXES = (
    "git status", "git diff", "git log", "ls", "date", "ps ", "top",
    "pytest", "npm test", "make test", "cargo test", "go test", "tail -f",
    "curl", "docker ps", "kubectl get",
)

# Sessions in which an identity must appear before it is a memory candidate.
# Two is coincidence.
RECURRING_SESSIONS = 3


def digest(text: str) -> str:
    """Hash of the whitespace-normalised result.

    Normalised because a trailing newline or a re-wrapped line is not a
    different answer, and hashed because this module must never hold transcript
    content: the report is published, the content is not.
    """
    return hashlib.sha1(" ".join(text.split()).encode("utf-8"),
                        usedforsecurity=False).hexdigest()[:16]


def identity(tool: str, inp) -> str:
    """A stable name for "the same call again".

    Deliberately coarse for `Read`: a file read at two different offsets is one
    identity, because the question the report answers is "which *file* keeps
    coming back", and the digest comparison downstream is what separates a true
    duplicate from a different slice.
    """
    if not isinstance(inp, dict):
        return tool
    if tool == "Read":
        return f"Read:{inp.get('file_path') or inp.get('path') or '?'}"
    if tool in ("Grep", "Glob"):
        return f"{tool}:{inp.get('pattern', '?')}|{inp.get('path', '')}"
    if tool == "Bash":
        return "Bash:" + " ".join(str(inp.get("command", "?")).split())
    if tool in ("Edit", "Write", "NotebookEdit"):
        return f"{tool}:{inp.get('file_path', '?')}"
    # Everything else is identified by a **hash** of its input, never by the
    # input itself. A tool this module has no shape for can carry anything --
    # a prompt, a plan, a question written for the user -- and identities are
    # printed in reports and injected into contexts by the hooks. Hashing keeps
    # "was this the same call?" answerable without ever restating what it said.
    try:
        return f"{tool}#{digest(json.dumps(inp, sort_keys=True))[:8]}"
    except (TypeError, ValueError):
        return tool


def shorten(ident: str, width: int = 52) -> str:
    """Trim an identity to `width`, keeping the end.

    The informative half of `Read:/Users/me/very/long/path/to/thing.py` is the
    filename, and a left-anchored truncation prints five rows of the same
    directory prefix and no filenames at all.
    """
    if len(ident) <= width:
        return ident
    head, _, tail = ident.partition(":")
    keep = width - len(head) - 2
    return f"{head}:…{tail[-keep:]}" if keep > 4 else "…" + ident[-(width - 1):]


def is_volatile(ident: str) -> bool:
    """True when re-running this is expected to return something new."""
    if not ident.startswith("Bash:"):
        return False
    cmd = ident[5:].lstrip()
    return any(cmd.startswith(p) for p in VOLATILE_PREFIXES)


@dataclass(frozen=True)
class Admission:
    """One tool result entering one context."""

    session: str
    project: str
    ident: str
    tool: str
    tokens: int
    sha: str
    turn: int           # assistant turns seen in this session before it landed


@dataclass
class Repeat:
    """Every admission of one identity within one session."""

    ident: str
    tool: str
    session: str
    project: str
    admissions: list[Admission] = field(default_factory=list)

    @property
    def calls(self) -> int:
        return len(self.admissions)

    @property
    def redundant(self) -> list[Admission]:
        """Admissions whose content was already verbatim in the context."""
        seen: set[str] = set()
        out = []
        for a in self.admissions:
            if a.sha in seen:
                out.append(a)
            seen.add(a.sha)
        return out

    @property
    def refreshes(self) -> list[Admission]:
        """Admissions that brought something new, superseding an older copy."""
        seen: set[str] = set()
        out = []
        for a in self.admissions:
            if a.sha not in seen and seen:
                out.append(a)
            seen.add(a.sha)
        return out

    @property
    def redundant_tokens(self) -> int:
        return sum(a.tokens for a in self.redundant)

    @property
    def superseded_tokens(self) -> int:
        """Tokens of copies that a later refresh made obsolete but did not evict."""
        return sum(a.tokens for a in self.admissions[:-1]) if self.refreshes else 0


@dataclass
class Recurring:
    """One identity, across the sessions that each had to learn it."""

    ident: str
    tool: str
    sessions: set[str] = field(default_factory=set)
    tokens: int = 0          # summed over first admissions, one per session
    shas: set[str] = field(default_factory=set)

    @property
    def n_sessions(self) -> int:
        return len(self.sessions)

    @property
    def mean_tokens(self) -> float:
        return self.tokens / max(1, self.n_sessions)

    @property
    def stable(self) -> bool:
        """The answer was the same every time — the case a note can replace."""
        return len(self.shas) == 1


@dataclass
class RereadReport:
    repeats: dict[tuple[str, str], Repeat] = field(default_factory=dict)
    recurring: dict[str, Recurring] = field(default_factory=dict)
    files: int = 0
    admissions: int = 0
    total_tokens: int = 0

    def with_repeats(self, *, min_tokens: int = MIN_RESULT_TOKENS,
                     include_volatile: bool = False) -> list[Repeat]:
        out = [r for r in self.repeats.values()
               if r.calls > 1 and r.redundant_tokens >= min_tokens
               and (include_volatile or not is_volatile(r.ident))]
        return sorted(out, key=lambda r: -r.redundant_tokens)

    def memory_candidates(self, *, min_sessions: int = RECURRING_SESSIONS,
                          min_tokens: int = MIN_RESULT_TOKENS,
                          stable_only: bool = True) -> list[Recurring]:
        out = [r for r in self.recurring.values()
               if r.n_sessions >= min_sessions and r.mean_tokens >= min_tokens
               and not is_volatile(r.ident) and (r.stable or not stable_only)]
        return sorted(out, key=lambda r: -(r.tokens))

    @property
    def redundant_tokens(self) -> int:
        return sum(r.redundant_tokens for r in self.repeats.values())


def scan(root: Path | str = DEFAULT_ROOT, *, window=None,
         min_tokens: int = 1) -> RereadReport:
    """Every tool result on record, keyed by what was asked for and what came back.

    The `tool_use` -> `tool_result` mapping is kept across files for the same
    reason `tools.scan` keeps it: a sidechain answers a call issued in another
    transcript, and dropping those attributes the result to no tool at all.
    """
    rep = RereadReport()
    pending: dict[str, tuple[str, str]] = {}      # use_id -> (tool, identity)
    seen_use: set[str] = set()
    answered: set[str] = set()

    for path in transcripts(root):
        rep.files += 1
        try:
            fh = path.open(encoding="utf-8", errors="replace")
        except OSError:
            continue
        turns: dict[str, int] = defaultdict(int)
        seen_msg: set[str] = set()
        with fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if window is not None and not window.keeps_record(d, path.parent.name):
                    continue
                msg = mapping(d, "message")
                content = msg.get("content")
                blocks = content if isinstance(content, list) else []
                session = str(d.get("sessionId") or path.stem)
                project = path.parent.name

                if d.get("type") == "assistant":
                    mid = str(msg.get("id") or "")
                    if mid and mid not in seen_msg:
                        seen_msg.add(mid)
                        turns[session] += 1
                    for b in blocks:
                        if not isinstance(b, dict) or b.get("type") != "tool_use":
                            continue
                        use_id = str(b.get("id") or "")
                        if use_id:
                            if use_id in seen_use:
                                continue
                            seen_use.add(use_id)
                        name = str(b.get("name") or "?")
                        if use_id:
                            pending[use_id] = (name, identity(name, b.get("input")))

                elif d.get("type") == "user":
                    for b in blocks:
                        if not isinstance(b, dict) or b.get("type") != "tool_result":
                            continue
                        use_id = str(b.get("tool_use_id") or "")
                        if use_id and use_id in answered:
                            continue
                        if use_id:
                            answered.add(use_id)
                        tool, ident = pending.get(use_id, ("?", "?"))
                        if ident == "?":
                            continue
                        text = flatten_text(b.get("content"))
                        n = est_tokens(text)
                        rep.admissions += 1
                        rep.total_tokens += n
                        if n < min_tokens:
                            continue
                        a = Admission(session, project, ident, tool, n,
                                      digest(text), turns[session])
                        key = (session, ident)
                        r = rep.repeats.get(key)
                        if r is None:
                            r = rep.repeats[key] = Repeat(ident, tool, session, project)
                        r.admissions.append(a)
                        rc = rep.recurring.get(ident)
                        if rc is None:
                            rc = rep.recurring[ident] = Recurring(ident, tool)
                        if session not in rc.sessions:
                            rc.sessions.add(session)
                            rc.tokens += n
                            rc.shas.add(a.sha)
    return rep


def _carry(sessions):
    from adder.measure.window.carry import Carry

    return Carry.measure(sessions) if sessions else Carry.default()


def _session_shape(sessions) -> dict[str, tuple[str, int]]:
    """session id -> (model it mostly ran on, how many turns it ended up with)."""
    out: dict[str, tuple[str, int]] = {}
    for sid, s in sessions.items():
        by_model = s.cost_by_model()
        model = (max(by_model.items(), key=lambda kv: kv[1])[0] if by_model
                 else _settings.session_model())
        out[sid] = (model, s.n_turns)
    return out


def price_repeat(r: Repeat, shape, carry, *, on: date | None = None) -> float:
    """USD the redundant copies cost: one write each, plus their carry."""
    model, n_turns = shape.get(r.session, (_settings.session_model(), 0))
    total = 0.0
    for a in r.redundant:
        remaining = max(0, n_turns - a.turn)
        # `on` was accepted and dropped. A caller pricing a past window got
        # today's rates back with no sign that its argument had been ignored,
        # which is worse than not offering the parameter.
        total += carry.token_cost(a.tokens, model, remaining, on=on)
    return total


def price_recurring(rc: Recurring, shape, carry, *, on: date | None = None) -> float:
    """USD the first admission of this identity costs, summed over sessions."""
    per = rc.mean_tokens
    total = 0.0
    for sid in rc.sessions:
        model, n_turns = shape.get(sid, (_settings.session_model(), 0))
        # A recurring read is learned early; charge it against most of the
        # session rather than assuming it lands at the end, which would make
        # every note look profitable.
        total += carry.token_cost(int(per), model, max(0, n_turns - 1), on=on)
    return total


def note_cost(tokens: int, sessions, *, model: str | None = None) -> float:
    """USD to hold a `tokens`-long note resident in every session instead."""
    from adder.measure.window.memory import Pricing

    return Pricing.measure(sessions, model=model).window_cost(tokens)


def breakeven_note_tokens(rc: Recurring, shape, carry, sessions, *,
                          model: str | None = None) -> int:
    """The largest note that is still cheaper than re-reading this every time.

    Below this many resident tokens, writing it down wins. Above it, the note
    is more expensive than the reads it replaces -- which is the case nobody
    checks, and the reason "just put it in CLAUDE.md" is not free advice.
    """
    from adder.measure.window.memory import Pricing

    p = Pricing.measure(sessions, model=model)
    per_token = p.window_cost(1_000) / 1_000.0
    if per_token <= 0:
        return 0
    return int(price_recurring(rc, shape, carry, on=None) / per_token)


def report(rep: RereadReport, sessions, *, top: int = 10,
           min_tokens: int = MIN_RESULT_TOKENS,
           min_sessions: int = RECURRING_SESSIONS, on: date | None = None) -> str:
    from adder.util.render import money, table, tokens, warn

    carry = _carry(sessions)
    shape = _session_shape(sessions)
    lines = [f"  {rep.admissions:,} tool results across {rep.files:,} transcripts "
             f"· {tokens(rep.total_tokens)} admitted"]

    repeats = rep.with_repeats(min_tokens=min_tokens)
    priced = sorted(((r, price_repeat(r, shape, carry, on=on)) for r in repeats),
                    key=lambda kv: -kv[1])
    waste = sum(c for _, c in priced)

    lines.append("")
    if not priced:
        lines.append("  No content was admitted twice. Nothing to recover here.")
    else:
        lines.append(f"  Re-read: identical content admitted again — "
                     f"{money(waste)} across {len(priced)} identities")
        lines.append("")
        body = [[shorten(r.ident), r.calls, len(r.redundant),
                 tokens(r.redundant_tokens), money(c), r.session[:8]]
                for r, c in priced[:top]]
        lines += table(body, ["what was read again", "calls", "dup", "dup tok",
                              "cost", "session"], align="<>>>><")
        if len(priced) > top:
            rest = sum(c for _, c in priced[top:])
            lines.append(f"    … {len(priced) - top} more, {money(rest)}")

    cands = rep.memory_candidates(min_tokens=min_tokens,
                                  min_sessions=min_sessions)
    if cands:
        lines += ["", "  Recurring across sessions — each session paid to learn "
                      "the same thing:", ""]
        body = []
        for rc in cands[:top]:
            cost = price_recurring(rc, shape, carry, on=on)
            be = breakeven_note_tokens(rc, shape, carry, sessions)
            body.append([shorten(rc.ident, 46), rc.n_sessions, tokens(int(rc.mean_tokens)),
                         money(cost), f"{be:,} tok"])
        lines += table(body, ["what is re-learned", "sessions", "tok each",
                              "cost", "note budget"], align="<>>><")
        lines.append("")
        lines.append("    `note budget` is the largest resident note that still "
                     "beats re-reading it (`adder memory --what-if`).")

    stale = sum(r.superseded_tokens for r in rep.repeats.values())
    if stale:
        lines += ["", f"  {tokens(stale)} of superseded copies are still resident: "
                      "content a later call replaced but nothing evicted. Not "
                      "recoverable by skipping a call — only by compacting or "
                      "restarting."]
    if waste > 1.0:
        lines += ["", warn(f"  {money(waste)} was spent admitting content the "
                           "context already held.")]
    return "\n".join(lines)


def _json(rep: RereadReport, sessions, *, top: int, min_tokens: int,
          min_sessions: int = RECURRING_SESSIONS) -> str:
    carry = _carry(sessions)
    shape = _session_shape(sessions)
    repeats = rep.with_repeats(min_tokens=min_tokens)
    priced = sorted(((r, price_repeat(r, shape, carry)) for r in repeats),
                    key=lambda kv: -kv[1])
    cands = rep.memory_candidates(min_tokens=min_tokens,
                                  min_sessions=min_sessions)
    return json.dumps({
        "files": rep.files,
        "admissions": rep.admissions,
        "admitted_tokens": rep.total_tokens,
        "redundant_tokens": rep.redundant_tokens,
        "recoverable": round(sum(c for _, c in priced), 4),
        "repeats": [
            {"identity": r.ident, "tool": r.tool, "session": r.session,
             "project": r.project, "calls": r.calls,
             "duplicate_calls": len(r.redundant),
             "duplicate_tokens": r.redundant_tokens,
             "superseded_tokens": r.superseded_tokens,
             "cost": round(c, 4)}
            for r, c in priced[:top]
        ],
        "recurring": [
            {"identity": rc.ident, "tool": rc.tool, "sessions": rc.n_sessions,
             "tokens_each": int(rc.mean_tokens), "stable": rc.stable,
             "cost": round(price_recurring(rc, shape, carry), 4),
             "note_budget_tokens": breakeven_note_tokens(rc, shape, carry, sessions)}
            for rc in cands[:top]
        ],
    })


def main(argv: list[str] | None = None) -> int:
    import argparse

    from adder.core.filters import add_arguments as add_window
    from adder.core.filters import load as load_window

    ap = argparse.ArgumentParser(
        prog="adder.reread",
        description="Content the agent admitted to context more than once, and "
                    "the reads that recur across sessions a note could replace.")
    add_window(ap)
    ap.add_argument("--top", type=int, default=10, metavar="N",
                    help="rows per section (default: %(default)s)")
    ap.add_argument("--min-tokens", type=int, default=MIN_RESULT_TOKENS,
                    metavar="TOK",
                    help="ignore results smaller than this (default: %(default)s)")
    ap.add_argument("--min-sessions", type=int, default=RECURRING_SESSIONS,
                    metavar="N",
                    help="sessions an identity must appear in to count as "
                         "recurring (default: %(default)s)")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    a = ap.parse_args(argv)
    # `root_of`: the argument if one was given, else the `root`
    # setting. Resolved here so two commands cannot disagree
    # about which transcript directory `adder config` names.
    a.root = str(_root_of(a))

    sessions, window = load_window(a)
    rep = scan(a.root, window=window if window.active else None)
    # See `Window.ignores_model`: a `tool_result` block carries no model, so a
    # model filter cannot reach this scan. Said out loud rather than silently
    # widening the report.
    note = ("  note: --model-filter is not applied here — a tool result carries "
            "no model" if window.ignores_model else "")
    if a.json:
        print(_json(rep, sessions, top=a.top, min_tokens=a.min_tokens,
                    min_sessions=a.min_sessions))
        return 0
    print()
    if note:
        print(note)
    print(report(rep, sessions, top=a.top, min_tokens=a.min_tokens,
                 min_sessions=a.min_sessions))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
