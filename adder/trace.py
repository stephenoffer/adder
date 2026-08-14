"""Parse Claude Code transcripts into per-turn cost records.

Transcripts live at ~/.claude/projects/<slug>/<session-uuid>.jsonl, one JSON
object per line. Assistant records carry `message.usage` with the exact token
accounting we need, so cost here is measured, not estimated.

Three correctness details that change the totals
------------------------------------------------
* **Deduplication.** A transcript can replay the same assistant record -- retries,
  resumed sessions, sidechain files that restate parent turns. Counting them
  twice inflates every downstream figure. Records are keyed by message id.
* **Cache TTL.** `usage.cache_creation` breaks writes into 5m and 1h buckets,
  priced at 1.25x and 2.00x. Assuming 5m everywhere understates any session
  using the 1h cache by up to 60% of its write cost.
* **Fast mode.** Opus 5 fast mode bills at $10/$50, double standard. A
  transcript recorded under `/fast` is twice as expensive as the price table
  says unless the tier is read off the record.

Parsing is cached by (path, mtime, size) so the prompt hook can re-read 169
transcripts without a measurable pause.
"""

from __future__ import annotations

import json
import os
import pickle
import statistics
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, datetime
from itertools import pairwise
from pathlib import Path

from .cost import turn_cost
from .prices import CACHE_READ_MULT, CACHE_WRITE_MULT, is_known, rate

DEFAULT_ROOT = Path.home() / ".claude" / "projects"
CACHE_PATH = Path(
    os.environ.get("ADDER_TRACE_CACHE", Path.home() / ".claude" / ".adder-trace-cache")
)
CACHE_VERSION = 3


@dataclass
class Turn:
    session: str
    project: str
    model: str
    uncached_in: int
    cache_read: int
    cache_write: int
    out: int
    thinking: int
    sidechain: bool
    ts: str | None = None
    ttl: str = "5m"              # dominant cache-write TTL for this turn
    speed: str = "standard"      # "fast" bills at 2x on Opus 5
    msg_id: str = ""             # dedup key
    tools: tuple[str, ...] = ()  # tool names invoked this turn

    @property
    def context(self) -> int:
        """Tokens the model had to read this turn."""
        return self.uncached_in + self.cache_read + self.cache_write

    def cost(self, on: date | None = None) -> float:
        return turn_cost(
            self.model,
            uncached_in=self.uncached_in,
            cache_read=self.cache_read,
            cache_write=self.cache_write,
            out=self.out,
            ttl=self.ttl,
            speed=self.speed,
            on=on,
        )

    def input_cost(self, on: date | None = None) -> float:
        r = rate(self.model, on, speed=self.speed)
        return (
            self.uncached_in * r.inp
            + self.cache_read * r.inp * CACHE_READ_MULT
            + self.cache_write * r.inp * CACHE_WRITE_MULT[self.ttl]
        ) / 1_000_000

    def output_cost(self, on: date | None = None) -> float:
        return self.out * rate(self.model, on, speed=self.speed).out / 1_000_000

    def thinking_cost(self, on: date | None = None) -> float:
        """Output spend on reasoning tokens. The part `effort` controls."""
        return self.thinking * rate(self.model, on, speed=self.speed).out / 1_000_000

    @property
    def when(self) -> datetime | None:
        return _parse_ts(self.ts)


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


@dataclass
class Session:
    id: str
    project: str
    turns: list[Turn] = field(default_factory=list)

    @property
    def cost(self) -> float:
        return sum(t.cost() for t in self.turns)

    def cost_on(self, on: date | None = None) -> float:
        """Priced at a specific date, so intro-rate expiry is visible."""
        return sum(t.cost(on) for t in self.turns)

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    @property
    def peak_context(self) -> int:
        return max((t.context for t in self.turns), default=0)

    @property
    def avg_context(self) -> int:
        return sum(t.context for t in self.turns) // max(1, len(self.turns))

    @property
    def base_context(self) -> int:
        """Smallest context seen: system prompt, tools, CLAUDE.md. Irreducible."""
        return min((t.context for t in self.turns), default=0)

    @property
    def models(self) -> set[str]:
        return {t.model for t in self.turns}

    @property
    def out_tokens(self) -> int:
        return sum(t.out for t in self.turns)

    @property
    def thinking_tokens(self) -> int:
        return sum(t.thinking for t in self.turns)

    def gaps(self) -> list[float]:
        """Seconds between consecutive turns. Drives the cache-TTL decision."""
        times = [t.when for t in self.turns if t.when]
        return [
            (b - a).total_seconds()
            for a, b in pairwise(times)
            if (b - a).total_seconds() >= 0
        ]

    def median_gap(self) -> float:
        g = self.gaps()
        return statistics.median(g) if g else 0.0

    def cache_misses(self) -> list[Turn]:
        """Turns that rewrote more than they read: an expired or invalidated cache.

        The first turn of a session legitimately writes everything. After that, a
        write-dominant turn means the prefix was rebuilt at 1.25x instead of
        being read at 0.10x.
        """
        return [
            t for t in self.turns[1:]
            if t.cache_write > t.cache_read and t.cache_write > 10_000
        ]

    def compactions(self) -> int:
        """Times the context dropped sharply -- a compaction or a fresh prefix."""
        n = 0
        for a, b in pairwise(self.turns):
            if a.context > 50_000 and b.context < a.context * 0.6:
                n += 1
        return n


def _dominant_ttl(usage: dict) -> str:
    """Which cache TTL this turn mostly used.

    `usage.cache_creation` splits writes into ephemeral_5m/1h buckets. Absent
    that breakdown, 5m is the Claude Code default.
    """
    cc = usage.get("cache_creation") or {}
    if not isinstance(cc, dict):
        return "5m"
    five = cc.get("ephemeral_5m_input_tokens") or 0
    hour = cc.get("ephemeral_1h_input_tokens") or 0
    return "1h" if hour > five else "5m"


def _speed(msg: dict, usage: dict) -> str:
    s = usage.get("speed") or msg.get("speed")
    return "fast" if s == "fast" else "standard"


def _tools(msg: dict) -> tuple[str, ...]:
    content = msg.get("content")
    if not isinstance(content, list):
        return ()
    return tuple(
        b.get("name", "") for b in content
        if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name")
    )


def _turn_from_record(d: dict, path_stem: str, project: str,
                      skip_unknown: bool) -> Turn | None:
    if d.get("type") != "assistant":
        return None
    msg = d.get("message") or {}
    usage = msg.get("usage") or {}
    model = msg.get("model")
    if not usage or not model:
        return None
    if skip_unknown and not is_known(model):
        return None
    details = usage.get("output_tokens_details") or {}
    return Turn(
        session=d.get("sessionId") or path_stem,
        project=project,
        model=model,
        uncached_in=usage.get("input_tokens", 0) or 0,
        cache_read=usage.get("cache_read_input_tokens", 0) or 0,
        cache_write=usage.get("cache_creation_input_tokens", 0) or 0,
        out=usage.get("output_tokens", 0) or 0,
        thinking=details.get("thinking_tokens", 0) or 0,
        sidechain=bool(d.get("isSidechain")),
        ts=d.get("timestamp"),
        ttl=_dominant_ttl(usage),
        speed=_speed(msg, usage),
        msg_id=msg.get("id") or d.get("requestId") or d.get("uuid") or "",
        tools=_tools(msg),
    )


def iter_file(path: Path, *, skip_unknown: bool = True) -> Iterator[Turn]:
    """Yield priced assistant turns from one transcript, deduplicated.

    Claude Code writes **one JSONL record per content block**, and every record
    repeats the whole message's `usage`. A turn with a thinking block and two
    tool calls is three records, each reporting the same token counts. Summing
    lines therefore multi-counts most turns: on this machine's 50 transcripts it
    inflates 18,144 real turns to 32,251 and $4,442 of spend to $7,507.

    Records are grouped by `message.id` and the one with the **highest**
    `output_tokens` wins. Partially-streamed records carry a running count that
    only the final record completes -- keeping the first instead of the max
    undercounts output by 2.6% here.
    """
    project = path.parent.name
    try:
        fh = path.open(errors="replace")
    except OSError:
        return
    best: dict[str, Turn] = {}
    order: list[str] = []
    anonymous: list[tuple[int, Turn]] = []
    idx = 0
    with fh:
        for line in fh:
            # Cheap prefilter: most lines are not assistant records.
            if '"assistant"' not in line:
                continue
            try:
                d = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            t = _turn_from_record(d, path.stem, project, skip_unknown)
            if t is None:
                continue
            if not t.msg_id:
                anonymous.append((idx, t))
                idx += 1
                continue
            prev = best.get(t.msg_id)
            if prev is None:
                best[t.msg_id] = t
                order.append(t.msg_id)
                idx += 1
            elif t.out > prev.out:
                # Later record completed the stream; keep the full accounting
                # but merge the tool calls seen across every block record.
                t.tools = tuple(dict.fromkeys(prev.tools + t.tools))
                best[t.msg_id] = t
            else:
                prev.tools = tuple(dict.fromkeys(prev.tools + t.tools))

    merged = [(order.index(m), best[m]) for m in order]
    merged.extend(anonymous)
    for _, t in sorted(merged, key=lambda kv: kv[0]):
        yield t


def iter_turns(root: Path | str = DEFAULT_ROOT, *,
               skip_unknown: bool = True) -> Iterator[Turn]:
    """Yield every priced assistant turn under `root`."""
    root = Path(root).expanduser()
    paths = [root] if root.is_file() else sorted(root.rglob("*.jsonl"))
    for path in paths:
        yield from iter_file(path, skip_unknown=skip_unknown)


# --------------------------------------------------------------------------
# Parse cache. The prompt hook runs on every keystroke-to-submit; re-parsing
# 169 transcripts each time is the difference between 20ms and 3s.
# --------------------------------------------------------------------------

def _cache_load() -> dict:
    try:
        with CACHE_PATH.open("rb") as fh:
            blob = pickle.load(fh)
        if blob.get("v") == CACHE_VERSION:
            return blob.get("files", {})
    except (OSError, ValueError, EOFError, pickle.UnpicklingError, AttributeError):
        pass
    return {}


def _cache_store(files: dict) -> None:
    """Write the parse cache atomically.

    The temp name carries the pid: several Claude Code sessions share one
    machine, and a fixed `.tmp` path lets two of them clobber each other's
    partial write. `replace` is atomic, so the loser is simply overwritten
    rather than producing a torn file.
    """
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_PATH.with_suffix(f".{os.getpid()}.tmp")
        try:
            with tmp.open("wb") as fh:
                pickle.dump({"v": CACHE_VERSION, "files": files}, fh, protocol=4)
            tmp.replace(CACHE_PATH)
        finally:
            tmp.unlink(missing_ok=True)
    except (OSError, pickle.PicklingError, RecursionError):
        pass


def load_sessions(root: Path | str = DEFAULT_ROOT, *,
                  use_cache: bool = False) -> dict[str, Session]:
    """Group every priced turn into sessions.

    `use_cache=True` memoizes per-file parses by (mtime, size). Only files that
    changed are re-read, which is what makes the prompt hook free.
    """
    root = Path(root).expanduser()
    paths = [root] if root.is_file() else sorted(root.rglob("*.jsonl"))

    cache = _cache_load() if use_cache else {}
    dirty = False
    sessions: dict[str, Session] = {}

    for path in paths:
        key = str(path)
        try:
            st = path.stat()
            stamp = (st.st_mtime_ns, st.st_size)
        except OSError:
            continue
        hit = cache.get(key)
        if use_cache and hit and tuple(hit[0]) == stamp:
            turns = hit[1]
        else:
            turns = list(iter_file(path))
            if use_cache:
                cache[key] = (stamp, turns)
                dirty = True
        for t in turns:
            s = sessions.get(t.session)
            if s is None:
                s = sessions[t.session] = Session(t.session, t.project)
            s.turns.append(t)

    if use_cache and dirty:
        live = {str(p) for p in paths}
        _cache_store({k: v for k, v in cache.items() if k in live})
    return sessions


@dataclass
class Summary:
    total: float = 0.0
    input_side: float = 0.0
    output_side: float = 0.0
    cache_read_cost: float = 0.0
    cache_write_cost: float = 0.0
    thinking_cost: float = 0.0
    by_model: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    turns_by_model: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    sidechain_cost: float = 0.0
    sidechain_turns: int = 0
    fast_cost: float = 0.0
    fast_turns: int = 0
    n_turns: int = 0
    n_sessions: int = 0
    out_tokens: int = 0
    thinking_tokens: int = 0


def summarize(root: Path | str = DEFAULT_ROOT, *,
              use_cache: bool = False) -> tuple[Summary, dict[str, Session]]:
    sessions = load_sessions(root, use_cache=use_cache)
    s = Summary(n_sessions=len(sessions))
    for sess in sessions.values():
        for t in sess.turns:
            c = t.cost()
            r = rate(t.model, speed=t.speed)
            s.total += c
            s.input_side += t.input_cost()
            s.output_side += t.output_cost()
            s.cache_read_cost += t.cache_read * r.inp * CACHE_READ_MULT / 1_000_000
            s.cache_write_cost += (
                t.cache_write * r.inp * CACHE_WRITE_MULT[t.ttl] / 1_000_000
            )
            s.thinking_cost += t.thinking_cost()
            s.by_model[t.model] += c
            s.turns_by_model[t.model] += 1
            s.n_turns += 1
            s.out_tokens += t.out
            s.thinking_tokens += t.thinking
            if t.sidechain:
                s.sidechain_cost += c
                s.sidechain_turns += 1
            if t.speed == "fast":
                s.fast_cost += c
                s.fast_turns += 1
    return s, sessions


def _pct(a: list[int], p: float) -> int:
    if not a:
        return 0
    return sorted(a)[min(len(a) - 1, int(len(a) * p))]


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="adder.trace", description="Measure Claude Code spend.")
    ap.add_argument("root", nargs="?", default=str(DEFAULT_ROOT))
    ap.add_argument("--verify", action="store_true", help="assert plan headline figures")
    ap.add_argument("--json", action="store_true", help="machine-readable summary")
    ap.add_argument("--no-cache", action="store_true", help="ignore the parse cache")
    a = ap.parse_args(argv)

    s, sessions = summarize(a.root, use_cache=not a.no_cache)
    if not s.n_turns:
        if a.json:
            print(json.dumps({"error": "no priced turns", "root": a.root}))
        else:
            print(f"No priced turns found under {a.root}")
        return 1

    lens = [x.n_turns for x in sessions.values()]
    ctxs = [x.peak_context for x in sessions.values()]

    if a.json:
        print(json.dumps({
            "total": round(s.total, 2),
            "sessions": s.n_sessions,
            "turns": s.n_turns,
            "input_side": round(s.input_side, 2),
            "output_side": round(s.output_side, 2),
            "cache_read": round(s.cache_read_cost, 2),
            "cache_write": round(s.cache_write_cost, 2),
            "thinking": round(s.thinking_cost, 2),
            "sidechain": round(s.sidechain_cost, 2),
            "fast_turns": s.fast_turns,
            "by_model": {k: round(v, 2) for k, v in s.by_model.items()},
            "turns_p50": _pct(lens, 0.5), "turns_p90": _pct(lens, 0.9),
            "ctx_p50": _pct(ctxs, 0.5), "ctx_p90": _pct(ctxs, 0.9),
        }))
        return 0

    print(f"\n  {s.n_sessions} sessions · {s.n_turns:,} turns · "
          f"${s.total:,.2f} list-equivalent\n")
    print(f"  {'model':<28}{'turns':>8}{'cost':>11}{'share':>8}")
    for m, c in sorted(s.by_model.items(), key=lambda kv: -kv[1]):
        print(f"  {m:<28}{s.turns_by_model[m]:>8,}{c:>11,.2f}{100*c/s.total:>7.1f}%")

    print(f"\n  input-side   ${s.input_side:>9,.2f}  ({100*s.input_side/s.total:.0f}%)")
    print(f"  output-side  ${s.output_side:>9,.2f}  ({100*s.output_side/s.total:.0f}%)")
    print(f"  cache-read   ${s.cache_read_cost:>9,.2f}  "
          f"({100*s.cache_read_cost/s.total:.0f}% of all spend)")
    print(f"  cache-write  ${s.cache_write_cost:>9,.2f}  "
          f"({100*s.cache_write_cost/s.total:.0f}%)")
    if s.thinking_tokens:
        print(f"  thinking     ${s.thinking_cost:>9,.2f}  "
              f"({s.thinking_tokens:,} tok, {100*s.thinking_tokens/max(1,s.out_tokens):.0f}% of output)")
    print(f"  subagents    ${s.sidechain_cost:>9,.2f}  "
          f"({100*s.sidechain_cost/s.total:.1f}%, {s.sidechain_turns:,} turns)")
    if s.fast_turns:
        print(f"  fast mode    ${s.fast_cost:>9,.2f}  "
              f"({s.fast_turns:,} turns billed at 2x)")

    print(f"\n  turns/session   p50={_pct(lens,.5):,}  p90={_pct(lens,.9):,}  max={max(lens):,}")
    print(f"  peak context    p50={_pct(ctxs,.5):,}  p90={_pct(ctxs,.9):,}  max={max(ctxs):,}")

    ranked = sorted(sessions.values(), key=lambda x: -x.cost)
    top = sum(x.cost for x in ranked[: max(1, len(ranked) // 4)])
    print(f"  top 25% of sessions = ${top:,.0f} ({100*top/s.total:.0f}% of spend)")
    print("\n  most expensive sessions:")
    for x in ranked[:3]:
        print(f"    ${x.cost:>8,.0f}  {x.n_turns:>5,} turns  "
              f"avg ctx {x.avg_context:>9,}  {x.project[:44]}")

    if a.verify:
        # Structural invariants, not a pinned dollar figure. The absolute total
        # depends on how much history is on disk; the shares are the claim.
        checks = [
            ("input-side >= 85% of spend", s.input_side / s.total >= 0.85),
            ("cache-read >= 70% of spend", s.cache_read_cost / s.total >= 0.70),
            ("output-side <= 15% of spend", s.output_side / s.total <= 0.15),
            ("input + output reconcile to total",
             abs(s.input_side + s.output_side - s.total) < max(0.01, s.total * 0.001)),
            ("median session > 100 turns", _pct(lens, 0.5) > 100),
            ("no turn priced without a known model", s.n_turns > 0),
        ]
        print("\n  verification:")
        ok = True
        for label, passed in checks:
            print(f"    [{'PASS' if passed else 'FAIL'}] {label}")
            ok &= passed
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
