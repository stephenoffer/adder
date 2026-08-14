"""Where context actually comes from.

The claim this module exists to test -- and falsified
-----------------------------------------------------
The original analysis concluded that assistant output is ~105% of context
growth, and therefore that verbosity is the root cause of the bill. That figure
was computed against records that multi-counted every turn with more than one
content block (see `trace.iter_file`). Output was inflated ~1.78x by the
duplicates while context deltas were not (duplicate records carry an identical
context, so the delta between them is zero) -- which is precisely how a real
52% became a reported 105%.

Re-derived on deduplicated records: assistant output is **52%** of main-chain
context growth. Still the largest single source, but no longer the whole story,
and no longer enough to justify terseness as the only lever.

Context grows for exactly four reasons, and only one of them is the model's
prose:

    assistant text      what the model wrote
    thinking            reasoning tokens, controlled by `effort`, not by style
    tool results        what the model read -- file contents, command output
    user messages       what the operator typed

Each has a different lever. Telling someone to "write less" when 70% of their
growth is `Bash` output that could have been piped through `head` is advice that
cannot work. This module measures the split instead of assuming it, by reading
the user/tool-result records that the cost parser skips.

Model-authored volume is taken from **billed** `output_tokens`, never estimated
from text. Two traps make estimation wrong here: on Opus 5 the thinking field is
returned empty (`display: "omitted"`) while still being billed, and a `tool_use`
block's JSON arguments are output tokens that appear in no `text` block. Reading
text lengths undercounts model output roughly sixtyfold.

Only *read* content -- tool results and user messages -- is estimated, at ~4
chars/token, and only ever to apportion shares of a measured total.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

CHARS_PER_TOKEN = 4.0

# Tool results above this are worth naming individually in the report.
BIG_RESULT_TOKENS = 5_000


def _est_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)


def _text_of(content) -> str:
    """Flatten a message `content` field to plain text for size estimation."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for b in content:
        if isinstance(b, str):
            parts.append(b)
        elif isinstance(b, dict):
            if isinstance(b.get("text"), str):
                parts.append(b["text"])
            elif isinstance(b.get("content"), (str, list)):
                parts.append(_text_of(b["content"]))
            elif isinstance(b.get("input"), dict):
                parts.append(json.dumps(b["input"]))
    return "".join(parts)


@dataclass
class Growth:
    assistant_output: int = 0      # BILLED output tokens (text + thinking + tool args)
    thinking: int = 0              # BILLED thinking tokens
    assistant_text: int = 0        # estimated, for reference only
    tool_results: int = 0
    user_messages: int = 0
    by_tool: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    tool_calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    big_results: list[tuple[str, int]] = field(default_factory=list)
    n_files: int = 0
    measured_growth: int = 0     # sum of positive context deltas, from usage

    @property
    def total(self) -> int:
        return self.assistant_output + self.tool_results + self.user_messages

    def shares(self) -> dict[str, float]:
        t = self.total or 1
        return {
            "assistant output": self.assistant_output / t,
            "tool results": self.tool_results / t,
            "user messages": self.user_messages / t,
        }

    @property
    def model_authored(self) -> int:
        """Tokens the model produced: what terseness and effort control."""
        return self.assistant_output


def scan(root: Path | str, *, limit: int | None = None) -> Growth:
    """Attribute context growth across every transcript under `root`."""
    root = Path(root).expanduser()
    paths = [root] if root.is_file() else sorted(root.rglob("*.jsonl"))
    if limit:
        paths = paths[:limit]
    g = Growth()
    pending: dict[str, str] = {}          # tool_use_id -> tool name

    for path in paths:
        g.n_files += 1
        try:
            fh = path.open(errors="replace")
        except OSError:
            continue
        seen_msg: set[str] = set()
        with fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                typ = d.get("type")
                msg = d.get("message") or {}
                content = msg.get("content")

                if typ == "assistant":
                    mid = msg.get("id") or ""
                    blocks = content if isinstance(content, list) else []
                    for b in blocks:
                        if not isinstance(b, dict):
                            continue
                        bt = b.get("type")
                        if bt == "tool_use":
                            name = b.get("name") or "?"
                            g.tool_calls[name] += 1
                            if b.get("id"):
                                pending[b["id"]] = name
                    # Text and thinking are per-message; count once per id.
                    if mid and mid in seen_msg:
                        continue
                    if mid:
                        seen_msg.add(mid)
                    for b in blocks:
                        if not isinstance(b, dict):
                            continue
                        if b.get("type") == "text":
                            # Reference only: billed output is the real figure.
                            g.assistant_text += _est_tokens(b.get("text") or "")

                elif typ == "user":
                    blocks = content if isinstance(content, list) else []
                    if isinstance(content, str):
                        g.user_messages += _est_tokens(content)
                        continue
                    for b in blocks:
                        if not isinstance(b, dict):
                            continue
                        if b.get("type") == "tool_result":
                            n = _est_tokens(_text_of(b.get("content")))
                            g.tool_results += n
                            name = pending.get(b.get("tool_use_id") or "", "?")
                            g.by_tool[name] += n
                            if n >= BIG_RESULT_TOKENS:
                                g.big_results.append((name, n))
                        elif b.get("type") == "text":
                            g.user_messages += _est_tokens(b.get("text") or "")
    return g


def measured_growth(sessions) -> int:
    """Sum of positive per-turn context increases, from billed token counts.

    This is the ground truth the estimated shares are apportioned against.
    """
    total = 0
    for s in sessions.values():
        for a, b in pairwise(s.turns):
            d = b.context - a.context
            if d > 0:
                total += d
    return total


def output_share_of_growth(sessions) -> float:
    """Measured: what fraction of main-chain context growth is assistant output.

    This is the number that scales every verbosity claim. Cutting output by x%
    cannot cut context by more than `x% * this`.
    """
    g = measured_growth(sessions)
    if not g:
        return 0.0
    out = sum(t.out for s in sessions.values() for t in s.turns if not t.sidechain)
    return out / g


def report(root: Path | str, sessions=None) -> str:
    g = scan(root)
    if sessions:
        g.assistant_output = sum(
            t.out for s in sessions.values() for t in s.turns if not t.sidechain)
        g.thinking = sum(
            t.thinking for s in sessions.values() for t in s.turns if not t.sidechain)
    if not g.total:
        return "  No message content found to attribute."

    lines = ["  Where context growth comes from", ""]
    lines.append(f"  {'source':<20}{'tokens':>14}{'share':>9}  basis      lever")
    rows = [
        ("assistant output", g.assistant_output, "billed", "terseness / effort"),
        ("tool results", g.tool_results, "est.", "delegation, narrower reads"),
        ("user messages", g.user_messages, "est.", "-"),
    ]
    for name, v, basis, lever in sorted(rows, key=lambda r: -r[1]):
        lines.append(f"  {name:<20}{v:>14,}{100*v/g.total:>8.1f}%  {basis:<10} {lever}")
    lines.append(f"  {'TOTAL':<20}{g.total:>14,}")
    if g.thinking:
        lines.append(f"    (thinking, billed as output: {g.thinking:,} tok, "
                     f"{100*g.thinking/max(1,g.assistant_output):.0f}% of output)")

    if sessions:
        mg = measured_growth(sessions)
        if mg:
            share = output_share_of_growth(sessions)
            lines.append("")
            lines.append(f"  Measured main-chain context growth: {mg:,} tok")
            lines.append(f"  Assistant output is {100*share:.0f}% of it.")
            lines.append(f"  Accounted for by the sources above: {100*g.total/mg:.0f}%")
            lines.append("  (the shortfall is tool-result estimation error, system")
            lines.append("   reminders, and attachments -- read content, not written)")

    if g.by_tool:
        lines.append("")
        lines.append("  Tool output admitted to context, by tool:")
        lines.append(f"  {'tool':<20}{'calls':>8}{'est. tokens':>14}{'per call':>11}")
        for name, n in sorted(g.by_tool.items(), key=lambda kv: -kv[1])[:8]:
            calls = g.tool_calls.get(name, 0) or 1
            lines.append(f"  {name:<20}{g.tool_calls.get(name,0):>8,}{n:>14,}{n//calls:>11,}")

    share = g.model_authored / g.total
    lines.append("")
    lines.append(f"  Model-authored: {100*share:.0f}% of attributed growth.")
    if share < 0.5:
        lines.append("  Most growth is READ, not WRITTEN. Terseness alone cannot")
        lines.append("  reach it -- delegation and narrower tool reads dominate.")
    else:
        lines.append("  Most growth is WRITTEN. Terseness and effort are the")
        lines.append("  primary levers, but they cap out at the share above.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .trace import DEFAULT_ROOT, load_sessions

    ap = argparse.ArgumentParser(prog="adder.context",
                                 description="Attribute context growth to its sources.")
    ap.add_argument("root", nargs="?", default=str(DEFAULT_ROOT))
    a = ap.parse_args(argv)
    print()
    print(report(a.root, load_sessions(a.root, use_cache=True)))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
