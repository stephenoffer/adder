"""Parse Claude Code transcripts into per-turn cost records.

Transcripts live at ~/.claude/projects/<slug>/<session-uuid>.jsonl, one JSON
object per line. Assistant records carry `message.usage` with the exact token
accounting we need, so cost here is measured, not estimated.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterator

from .cost import turn_cost
from .prices import CACHE_READ_MULT, CACHE_WRITE_MULT, is_known, rate

DEFAULT_ROOT = Path.home() / ".claude" / "projects"


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
            on=on,
        )

    def input_cost(self, on: date | None = None) -> float:
        r = rate(self.model, on)
        return (
            self.uncached_in * r.inp
            + self.cache_read * r.inp * CACHE_READ_MULT
            + self.cache_write * r.inp * CACHE_WRITE_MULT["5m"]
        ) / 1_000_000

    def output_cost(self, on: date | None = None) -> float:
        return self.out * rate(self.model, on).out / 1_000_000


@dataclass
class Session:
    id: str
    project: str
    turns: list[Turn] = field(default_factory=list)

    @property
    def cost(self) -> float:
        return sum(t.cost() for t in self.turns)

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
    def models(self) -> set[str]:
        return {t.model for t in self.turns}


def iter_turns(root: Path | str = DEFAULT_ROOT, *, skip_unknown: bool = True) -> Iterator[Turn]:
    """Yield every priced assistant turn under `root`."""
    root = Path(root).expanduser()
    for path in root.rglob("*.jsonl"):
        project = path.parent.name
        try:
            fh = path.open(errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                # Cheap prefilter: most lines are not assistant records.
                if '"assistant"' not in line:
                    continue
                try:
                    d = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if d.get("type") != "assistant":
                    continue
                msg = d.get("message") or {}
                usage = msg.get("usage") or {}
                model = msg.get("model")
                if not usage or not model:
                    continue
                if skip_unknown and not is_known(model):
                    continue
                details = usage.get("output_tokens_details") or {}
                yield Turn(
                    session=d.get("sessionId") or path.stem,
                    project=project,
                    model=model,
                    uncached_in=usage.get("input_tokens", 0) or 0,
                    cache_read=usage.get("cache_read_input_tokens", 0) or 0,
                    cache_write=usage.get("cache_creation_input_tokens", 0) or 0,
                    out=usage.get("output_tokens", 0) or 0,
                    thinking=details.get("thinking_tokens", 0) or 0,
                    sidechain=bool(d.get("isSidechain")),
                    ts=d.get("timestamp"),
                )


def load_sessions(root: Path | str = DEFAULT_ROOT) -> dict[str, Session]:
    sessions: dict[str, Session] = {}
    for t in iter_turns(root):
        s = sessions.get(t.session)
        if s is None:
            s = sessions[t.session] = Session(t.session, t.project)
        s.turns.append(t)
    return sessions


@dataclass
class Summary:
    total: float = 0.0
    input_side: float = 0.0
    output_side: float = 0.0
    cache_read_cost: float = 0.0
    by_model: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    turns_by_model: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    sidechain_cost: float = 0.0
    sidechain_turns: int = 0
    n_turns: int = 0
    n_sessions: int = 0


def summarize(root: Path | str = DEFAULT_ROOT) -> tuple[Summary, dict[str, Session]]:
    sessions = load_sessions(root)
    s = Summary(n_sessions=len(sessions))
    for sess in sessions.values():
        for t in sess.turns:
            c = t.cost()
            s.total += c
            s.input_side += t.input_cost()
            s.output_side += t.output_cost()
            s.cache_read_cost += t.cache_read * rate(t.model).inp * CACHE_READ_MULT / 1_000_000
            s.by_model[t.model] += c
            s.turns_by_model[t.model] += 1
            s.n_turns += 1
            if t.sidechain:
                s.sidechain_cost += c
                s.sidechain_turns += 1
    return s, sessions


def _pct(a: list[int], p: float) -> int:
    if not a:
        return 0
    return sorted(a)[min(len(a) - 1, int(len(a) * p))]


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="router.trace", description="Measure Claude Code spend.")
    ap.add_argument("root", nargs="?", default=str(DEFAULT_ROOT))
    ap.add_argument("--verify", action="store_true", help="assert plan headline figures")
    a = ap.parse_args(argv)

    s, sessions = summarize(a.root)
    if not s.n_turns:
        print(f"No priced turns found under {a.root}")
        return 1

    print(f"\n  {s.n_sessions} sessions · {s.n_turns:,} turns · "
          f"${s.total:,.2f} list-equivalent\n")
    print(f"  {'model':<28}{'turns':>8}{'cost':>11}{'share':>8}")
    for m, c in sorted(s.by_model.items(), key=lambda kv: -kv[1]):
        print(f"  {m:<28}{s.turns_by_model[m]:>8,}{c:>11,.2f}{100*c/s.total:>7.1f}%")

    print(f"\n  input-side   ${s.input_side:>9,.2f}  ({100*s.input_side/s.total:.0f}%)")
    print(f"  output-side  ${s.output_side:>9,.2f}  ({100*s.output_side/s.total:.0f}%)")
    print(f"  cache-read   ${s.cache_read_cost:>9,.2f}  "
          f"({100*s.cache_read_cost/s.total:.0f}% of all spend)")
    print(f"  subagents    ${s.sidechain_cost:>9,.2f}  "
          f"({100*s.sidechain_cost/s.total:.1f}%, {s.sidechain_turns:,} turns)")

    lens = [x.n_turns for x in sessions.values()]
    ctxs = [x.peak_context for x in sessions.values()]
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
        checks = [
            ("total spend $6.5-7.5K", 6_500 <= s.total <= 7_500),
            ("input-side >= 85%", s.input_side / s.total >= 0.85),
            ("cache-read >= 70%", s.cache_read_cost / s.total >= 0.70),
            ("sessions >= 40", s.n_sessions >= 40),
            ("median session > 300 turns", _pct(lens, 0.5) > 300),
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
