"""Transcripts that other agent CLIs write to disk, read the way they were written.

Why this exists
---------------
`ingest` normalizes one *record* at a time: an API response or a proxy log line
carries its own model and its own usage, so each record can be priced on its
own. That works for anything that logs API calls, and on this repository's own
machine it read none of the Codex CLI transcripts it was pointed at -- five
files, zero turns. An agent CLI does not log API calls. It logs a session, and
the usage and the model sit in different records:

* Codex CLI writes `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`. The model is
  on a `turn_context` record; the usage is on a later `event_msg` whose payload
  is `token_count`. Neither record is priceable without the other.
* Gemini CLI writes one JSON document per session under
  `~/.gemini/tmp/<project-hash>/chats/`, with per-message `tokens`.
* OpenCode writes one JSON file per message under
  `~/.local/share/opencode/storage/message/<session-id>/`.

So these readers work per file, carrying state from one record to the next,
and `ingest.iter_turns` hands a file to them before it tries the per-record
adapters.

The trap each format sets
-------------------------
Codex writes every `token_count` event at least twice with identical totals --
on the transcript this module was written against, 68 events for 34 API calls.
Summing `last_token_usage` from each one doubles the bill. It is the same shape
of error as Claude Code's one-record-per-content-block (the 1.78x inflation
`trace.iter_file` exists to undo), and it is fixed the same way: a turn is
counted only when the cumulative `total_token_usage` moves.

Codex and Gemini both use the overlapping input convention -- the cached prefix
is *inside* the input count -- and are subtracted here, for the reason
`ingest`'s docstring gives. OpenCode already stores input net of the cache, so
it is taken as written.

What is not known, and is therefore labelled rather than guessed
----------------------------------------------------------------
OpenCode stores `tokens.reasoning` beside `tokens.output`. Whether `output`
already includes it depends on which provider SDK filled the record in, and the
file does not say. This reader bills `output` alone and keeps `reasoning` on
the side in `thinking`, so the error, where there is one, is an under-count --
the direction a cost tool can afford.

How the module is laid out
--------------------------
One `Agent` subclass per agent CLI, each carrying everything adder knows about
that agent: where it keeps transcripts, the names a person types for it, the
harness it implies, how to recognise one of its files from the first few
kilobytes, and how to read one into `ingest.Usage` -- the same normalized
record the per-record API adapters produce, so both paths share one
conversion into `Turn`. Adding an agent is one subclass and one entry in
`AGENTS`; nothing else in the tree lists agents.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

from adder.core.ingest import Usage, admit, count, iter_records
from adder.core.trace import Turn
from adder.pricing.registry import is_known

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _slug(cwd: Any) -> str:
    """A project label in Claude Code's own encoding, so `project_name` applies.

    Claude Code names a project directory by its path with `/` turned into
    `-`. Reusing that spelling means every report that groups or prints by
    project treats a Codex session in `~/app` and a Claude session in `~/app` as
    the same project, which they are.
    """
    if not isinstance(cwd, str) or not cwd:
        return ""
    return cwd.replace("\\", "/").rstrip("/").replace("/", "-")


def _iso_ms(v: Any) -> str | None:
    """Epoch milliseconds to ISO-8601, or None."""
    if not isinstance(v, (int, float)) or v <= 0:
        return None
    try:
        return datetime.fromtimestamp(float(v) / 1000.0, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _str(v: Any) -> str | None:
    return v if isinstance(v, str) and v else None


def _model(raw: Any, provider: Any = "") -> str:
    """A model id the registry can price, trying the provider-qualified form too."""
    if not isinstance(raw, str) or not raw:
        return ""
    if is_known(raw) or not isinstance(provider, str) or not provider:
        return raw
    qualified = f"{provider}/{raw}"
    return qualified if is_known(qualified) else raw


def _load_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# The base class
# ---------------------------------------------------------------------------


class Agent:
    """One agent CLI, described by where it writes and how to read what it wrote.

    Subclasses set the class attributes and implement `claims` and `usages`.
    The base supplies everything derived from those: the transcript directory
    under a home, the turns a file holds, and the deduplication every native
    format needs because every one of them can repeat a call.
    """

    name: ClassVar[str] = ""
    # The harness this agent runs as. Usually its own name; kept separate so an
    # agent and a harness can be renamed independently.
    harness: ClassVar[str] = ""
    # Transcript directory, relative to the home directory. Resolved when
    # asked for, never at import, so a test that moves HOME moves it too.
    root: ClassVar[tuple[str, ...]] = ()
    aliases: ClassVar[tuple[str, ...]] = ()

    def root_in(self, home: Path | None = None) -> Path:
        return (home or Path.home()).joinpath(*self.root)

    def claims(self, path: Path, head: str) -> bool:
        """Is `path`, whose first few kilobytes are `head`, one of this agent's files?"""
        return False

    def usages(self, path: Path) -> Iterator[tuple[Usage, str, str]]:
        """Every priced call in `path`, as (usage, session, project)."""
        return iter(())

    def turns(self, path: Path) -> Iterator[Turn]:
        """`usages` converted to turns, each call counted once.

        The id check is here rather than in each reader because the hazard is
        common to all of them: a resumed Codex rollout replays its events into
        a new file, and a Gemini CLI chat file is rewritten whole on every
        message.
        """
        seen: set[str] = set()
        for u, session, project in self.usages(Path(path)):
            t = u.to_turn(session=session, project=project)
            if t.msg_id and t.msg_id in seen:
                continue
            seen.add(t.msg_id)
            yield t


class ClaudeCode(Agent):
    """Read by `trace.iter_file`, which owns the per-content-block dedup.

    Listed so it is found, named and discovered like the others; it claims no
    file here, so a Claude Code transcript always falls through to `trace`.
    """

    name = harness = "claude-code"
    root = (".claude", "projects")
    aliases = ("claude", "cc", "claude-cli")


class Codex(Agent):
    """Codex CLI rollouts: `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`.

    A call is counted when `total_token_usage` changes, and priced from
    `last_token_usage`. `input_tokens` includes `cached_input_tokens` and
    `output_tokens` includes `reasoning_output_tokens` (the totals add up to
    `total_tokens` only that way), so the cache is subtracted from the input
    and reasoning is recorded without being added again.
    """

    name = harness = "codex"
    root = (".codex", "sessions")
    aliases = ("codex-cli", "openai-codex")
    _TOOL_ITEMS = ("function_call", "custom_tool_call", "local_shell_call",
                   "web_search_call")

    def claims(self, path: Path, head: str) -> bool:
        first = head.split("\n", 1)[0].replace(" ", "")
        return first.startswith("{") and (
            '"type":"session_meta"' in first or '"type":"turn_context"' in first)

    def usages(self, path: Path) -> Iterator[tuple[Usage, str, str]]:
        session, project, model, effort = path.stem, "", "", ""
        last_total: tuple[int, ...] | None = None
        pending: list[str] = []
        for d in iter_records(path):
            kind = d.get("type")
            p = d.get("payload") if isinstance(d.get("payload"), dict) else {}
            if kind == "session_meta":
                session = str(p.get("id") or session)
                project = _slug(p.get("cwd")) or project
            elif kind == "turn_context":
                model = str(p.get("model") or model)
                project = project or _slug(p.get("cwd"))
                mode = p.get("collaboration_mode")
                chosen = mode.get("settings") if isinstance(mode, dict) else None
                effort = str(p.get("effort") or (chosen or {}).get("reasoning_effort")
                             or effort)
            elif kind == "response_item" and p.get("type") in self._TOOL_ITEMS:
                pending.append(str(p.get("name") or p.get("type")))
            elif kind == "event_msg" and p.get("type") == "token_count":
                info = p.get("info") if isinstance(p.get("info"), dict) else {}
                total, last = info.get("total_token_usage"), info.get("last_token_usage")
                if not isinstance(total, dict) or not isinstance(last, dict):
                    continue
                key = tuple(count(total.get(k)) for k in (
                    "input_tokens", "cached_input_tokens", "output_tokens",
                    "total_tokens"))
                if key == last_total:
                    continue                  # the repeated event: same call, again
                last_total = key
                inp = count(last.get("input_tokens"))
                cached = min(count(last.get("cached_input_tokens")), inp)
                out = count(last.get("output_tokens"))
                tools, pending = tuple(pending), []
                if not model or not (inp or out):
                    continue
                yield Usage(
                    uncached_in=inp - cached, cache_read=cached, cache_write=0,
                    out=out, thinking=count(last.get("reasoning_output_tokens")),
                    model=model, ts=_str(d.get("timestamp")), effort=effort,
                    tools=tools,
                    # The cumulative total is unique per call within a session
                    # and survives a resumed rollout replaying the same events.
                    msg_id=f"codex:{session}:{key[-1]}",
                ), session, project


class GeminiCli(Agent):
    """Gemini CLI chats: `~/.gemini/tmp/<project-hash>/chats/session-*.json`.

    `tokens.input` is `promptTokenCount`, which includes `tokens.cached`.
    `tokens.thoughts` is billed at the output rate but reported outside
    `tokens.output`, so it is added -- the same conversion `ingest._from_gemini`
    makes for the raw API. `tokens.tool` is prompt spent on server-side tool
    use and bills as input.
    """

    name = harness = "gemini-cli"
    root = (".gemini", "tmp")
    aliases = ("gemini",)

    def claims(self, path: Path, head: str) -> bool:
        flat = head.replace(" ", "").replace("\n", "")
        return flat.startswith("{") and '"sessionId"' in flat and '"messages"' in flat

    def usages(self, path: Path) -> Iterator[tuple[Usage, str, str]]:
        doc = _load_json(path)
        if not isinstance(doc, dict):
            return
        session = str(doc.get("sessionId") or path.stem)
        project = str(doc.get("projectHash") or path.parent.parent.name)
        for i, m in enumerate(doc.get("messages") or ()):
            if not isinstance(m, dict) or m.get("type") != "gemini":
                continue
            tok = m.get("tokens")
            if not isinstance(tok, dict):
                continue
            prompt = count(tok.get("input"))
            cached = min(count(tok.get("cached")), prompt)
            think = count(tok.get("thoughts"))
            out = count(tok.get("output")) + think
            model = _model(m.get("model") or doc.get("model"))
            if not model or not (prompt or out):
                continue
            yield Usage(
                uncached_in=prompt - cached + count(tok.get("tool")),
                cache_read=cached, cache_write=0, out=out, thinking=think,
                model=model, ts=_str(m.get("timestamp")),
                tools=[str(c["name"]) for c in (m.get("toolCalls") or ())
                       if isinstance(c, dict) and c.get("name")],
                msg_id=f"gemini:{session}:{m.get('id') or i}",
            ), session, project


class OpenCode(Agent):
    """OpenCode messages: one JSON file each under `storage/message/<session>/`.

    Input is stored net of the cache, with reads and writes beside it, which is
    already the convention `Turn` uses. See the module docstring for why
    `reasoning` is not added to `output`.
    """

    name = harness = "opencode"
    root = (".local", "share", "opencode", "storage", "message")
    aliases = ("open-code",)

    def claims(self, path: Path, head: str) -> bool:
        flat = head.replace(" ", "").replace("\n", "")
        return (flat.startswith("{") and Path(path).name.startswith("msg_")
                and '"sessionID"' in flat and '"role"' in flat)

    def usages(self, path: Path) -> Iterator[tuple[Usage, str, str]]:
        m = _load_json(path)
        if not isinstance(m, dict) or m.get("role") != "assistant":
            return
        tok = m.get("tokens")
        if not isinstance(tok, dict):
            return
        cache = tok.get("cache") if isinstance(tok.get("cache"), dict) else {}
        model = _model(m.get("modelID"), m.get("providerID"))
        inp, out = count(tok.get("input")), count(tok.get("output"))
        read, write = count(cache.get("read")), count(cache.get("write"))
        if not model or not (inp or out or read or write):
            return
        where = m.get("path") if isinstance(m.get("path"), dict) else {}
        time = m.get("time") if isinstance(m.get("time"), dict) else {}
        yield Usage(
            uncached_in=inp, cache_read=read, cache_write=write, out=out,
            thinking=count(tok.get("reasoning")), model=model,
            ts=_iso_ms(time.get("created")),
            msg_id=f"opencode:{m.get('id') or path.stem}",
        ), str(m.get("sessionID") or path.parent.name), _slug(
            where.get("root") or where.get("cwd"))


# Claude Code first: when its directory has sessions it is the default root.
AGENTS: tuple[Agent, ...] = (ClaudeCode(), Codex(), GeminiCli(), OpenCode())
BY_NAME: dict[str, Agent] = {a.name: a for a in AGENTS}
_ALIASES: dict[str, Agent] = {k: a for a in AGENTS for k in (a.name, *a.aliases)}

CLAUDE_CODE, CODEX, GEMINI_CLI, OPENCODE = (a.name for a in AGENTS)


# ---------------------------------------------------------------------------
# Finding the files
# ---------------------------------------------------------------------------


def agents() -> tuple[str, ...]:
    """Every agent whose transcripts adder can find on its own."""
    return tuple(BY_NAME)


def get(name: str) -> Agent | None:
    """The agent a name or alias refers to, or None when it names no agent."""
    return _ALIASES.get(str(name).strip().lower())


def aliases() -> dict[str, str]:
    """Every spelling of every agent, mapped to the harness it runs as."""
    return {k: a.harness for k, a in _ALIASES.items()}


def canonical(name: str) -> str | None:
    a = get(name)
    return a.name if a else None


def root_for(name: str, home: Path | None = None) -> Path | None:
    """The transcript directory for an agent name, or None if it names none."""
    a = get(name)
    return a.root_in(home) if a else None


def discover(home: Path | None = None) -> dict[str, Path]:
    """Agents that have a transcript directory on this machine.

    Presence of the directory, not of parseable files: `adder doctor` is where
    "found it but could not read it" gets said, and it needs this list to say
    it about.
    """
    return {a.name: a.root_in(home) for a in AGENTS if a.root_in(home).is_dir()}


# Roots already seen to hold a session. Only the positive answer is kept: a
# directory that had a transcript still has one for the rest of the process,
# while an empty one may get its first at any moment and is cheap to re-walk.
_HAS_SESSIONS: set[str] = set()


def has_sessions(root: Path) -> bool:
    """Is there at least one transcript-shaped file under `root`? Stops at the first.

    Memoised, because the default `root` setting asks it and settings are
    resolved on every lookup: `adder plan` looks the ladder up per recorded
    turn, and an `rglob` over `~/.claude/projects` each time turned a
    67-session replay into one that did not finish in fifteen minutes.
    """
    key = str(Path(root).expanduser())
    if key in _HAS_SESSIONS:
        return True
    try:
        for pattern in ("*.jsonl", "*.json"):
            for _ in Path(root).rglob(pattern):
                _HAS_SESSIONS.add(key)
                return True
    except OSError:
        return False
    return False


def agent_of_root(root: Path | str) -> str | None:
    """Which agent's directory `root` is, or lies inside. None if no agent's.

    Compared on path components rather than against `Path.home()`, so a root
    copied off another machine (`/mnt/backup/jo/.codex/sessions`) is still
    recognised as Codex.
    """
    named = get(str(root))
    if named is not None and not Path(root).expanduser().exists():
        return named.name
    parts = Path(root).expanduser().parts
    for a in AGENTS:
        n = len(a.root)
        if any(tuple(parts[i:i + n]) == a.root for i in range(len(parts) - n + 1)):
            return a.name
    return None


# ---------------------------------------------------------------------------
# The model a machine runs
# ---------------------------------------------------------------------------

# Where a model id sits in each agent's records: `message.model` (Claude Code),
# `turn_context.payload.model` (Codex), `messages[].model` (Gemini CLI),
# `modelID` (OpenCode). A key match is enough, because the candidate has to
# resolve in the registry before it is believed.
_MODEL_KEY = re.compile(r'"(?:model|modelID|modelVersion)"\s*:\s*"([^"]{3,120})"')
_TAIL = 256 * 1024


def _models_near_end(path: Path) -> list[str]:
    try:
        with Path(path).open("rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - _TAIL))
            text = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    return _MODEL_KEY.findall(text)


@lru_cache(maxsize=32)
def _latest_model(root: str) -> str:
    from adder.core.trace import transcripts
    from adder.pricing.prices import is_synthetic

    try:
        paths = [p for p in transcripts(root) if "subagents" not in p.parts]
        paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return ""
    # The newest few, not just the newest: a session that has only just
    # started may not have reached its first model call yet.
    for p in paths[:5]:
        for m in reversed(_models_near_end(p)):
            if not is_synthetic(m) and is_known(m):
                return m
    return ""


def latest_model(root: Path | str) -> str:
    """The model the most recently written main session under `root` ran on.

    "" when nothing under `root` names a model adder can price. Only the tail
    of the newest few transcripts is read -- the last model named there is the
    one in use now, and a long transcript's head is the history of what it
    used to run. Memoised per root for the life of the process: it is asked
    for many times per command and the answer cannot change inside one.
    """
    return _latest_model(str(Path(root).expanduser()))


def forget_latest_model() -> None:
    """Drop the memoised answers. For tests, which write sessions mid-process."""
    _latest_model.cache_clear()
    _HAS_SESSIONS.clear()


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def owner(path: Path) -> Agent | None:
    """The agent that wrote this file, from its first bytes. None if none of these.

    Reads a few kilobytes, never the file: this runs once per transcript on
    every report, and most of those are Claude Code transcripts that must fall
    straight through to `trace`.
    """
    try:
        with Path(path).open("r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(4096)
    except OSError:
        return None
    return next((a for a in AGENTS if a.claims(Path(path), head)), None)


def detect(path: Path) -> str | None:
    """`owner(path)` by name."""
    a = owner(path)
    return a.name if a else None


def iter_turns(path: Path, *, agent: str | None = None, skip_unknown: bool = True,
               unknown: dict[str, int] | None = None) -> Iterator[Turn] | None:
    """Turns from a native agent transcript, or None if `path` is not one.

    None, rather than an empty iterator, so the caller can tell "not mine" from
    "mine, and empty" and fall through to the per-record adapters only in the
    first case.
    """
    a = get(agent) if agent else owner(path)
    if a is None or type(a).usages is Agent.usages:
        return None
    return (t for t in a.turns(Path(path))
            if admit(t, skip_unknown=skip_unknown, unknown=unknown))
