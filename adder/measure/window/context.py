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

from adder.core.filters import root_of as _root_of
from adder.util.records import mapping
from adder.util.text import est_tokens, flatten_text

# Tool results above this are worth naming individually in the report.
BIG_RESULT_TOKENS = 5_000


# Retained for the call sites inside this module; the definitions now live in
# `adder.util.text`, which the guard and the transcript reader also use.
_est_tokens = est_tokens
_text_of = flatten_text


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


def scan(root: Path | str, *, limit: int | None = None, window=None) -> Growth:
    """Attribute context growth across every transcript under `root`.

    `window` filters raw records the same way `filters.Window` filters turns.
    Without it, `adder context --since 7d` would report last week's *billed*
    totals beside an attribution computed over all history -- two numbers from
    different populations printed as though they described each other.
    """
    from adder.core.trace import transcripts

    paths = transcripts(root)
    if limit:
        paths = paths[:limit]
    g = Growth()
    pending: dict[str, str] = {}          # tool_use_id -> tool name
    # Scan-wide dedup, keyed by the id of the thing being counted. Two
    # mechanisms replay records: one JSONL record per content block repeating
    # the message envelope, and a resumed session writing a new transcript that
    # restates earlier turns. The shares this function produces feed
    # `output_share_of_growth`, which scales every terseness claim in the
    # repo -- and an inflation of exactly this kind is what put that number at
    # 105% once already (see `debt.py`).
    seen_msg: set[tuple[str, str]] = set()
    seen_use: set[str] = set()
    answered: set[str] = set()

    for path in paths:
        g.n_files += 1
        try:
            fh = path.open(encoding="utf-8", errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if window is not None and not window.keeps_record(d, path.parent.name):
                    continue
                typ = d.get("type")
                msg = mapping(d, "message")
                content = msg.get("content")
                session = str(d.get("sessionId") or path.stem)

                if typ == "assistant":
                    mid = str(msg.get("id") or "")
                    blocks = content if isinstance(content, list) else []
                    for b in blocks:
                        if not isinstance(b, dict):
                            continue
                        bt = b.get("type")
                        if bt == "tool_use":
                            use_id = str(b.get("id") or "")
                            if use_id:
                                if use_id in seen_use:
                                    continue
                                seen_use.add(use_id)
                            name = str(b.get("name") or "?")
                            g.tool_calls[name] += 1
                            if use_id:
                                pending[use_id] = name
                    # Text and thinking are per-message; count once per id, and
                    # once per *conversation* rather than once per file.
                    key = (session, mid)
                    if mid and key in seen_msg:
                        continue
                    if mid:
                        seen_msg.add(key)
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
                            use_id = str(b.get("tool_use_id") or "")
                            if use_id:
                                if use_id in answered:
                                    continue
                                answered.add(use_id)
                            n = _est_tokens(flatten_text(b.get("content")))
                            g.tool_results += n
                            name = pending.get(use_id, "?")
                            g.by_tool[name] += n
                            if n >= BIG_RESULT_TOKENS:
                                g.big_results.append((name, n))
                        elif b.get("type") == "text":
                            g.user_messages += _est_tokens(b.get("text") or "")
    return g


def measured_growth(sessions) -> int:
    """Sum of positive per-turn context increases along the MAIN chain.

    This is the ground truth the estimated shares are apportioned against, and
    the main-chain restriction is what makes it comparable with them.

    A subagent runs in its own short-lived context, and its turns are
    interleaved with the parent's in the same session. Pairing them together
    walks the context down to the subagent's few thousand tokens and then back
    up to the parent's several hundred thousand -- and that climb back is
    recorded as growth that never happened. One sidechain turn in a
    three-turn session inflated the total 5.75x here, which pushed the reported
    output share down by the same factor. The numerator in
    `output_share_of_growth` already excluded sidechains; only the denominator
    did not, so the two were measured over different populations.
    """
    total = 0
    for s in sessions.values():
        main = [t for t in s.turns if not t.sidechain]
        for a, b in pairwise(main):
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


def report(root: Path | str, sessions=None, *, window=None) -> str:
    g = scan(root, window=window)
    if sessions:
        g.assistant_output = sum(
            t.out for s in sessions.values() for t in s.turns if not t.sidechain)
        g.thinking = sum(
            t.thinking for s in sessions.values() for t in s.turns if not t.sidechain)
    if not g.total:
        return "  No message content found to attribute."

    lines = ["  Where context growth comes from", ""]
    # `--model-filter` cannot reach a raw-record scan: a `tool_result` block
    # carries no model. `Window.ignores_model` exists so a report can say so
    # rather than quietly widening itself, which is the failure it was written
    # for -- "a filter that is accepted and ignored is worse than one that is
    # rejected, because the number looks like an answer".
    if window is not None and getattr(window, "ignores_model", False):
        lines.append("  note: --model-filter is not applied here — a tool result "
                     "carries no model")
        lines.append("")
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

    from adder.core.filters import add_arguments as add_window
    from adder.core.filters import load as load_window

    ap = argparse.ArgumentParser(prog="adder context",
                                 description="Attribute context growth to its sources.")
    add_window(ap)
    ap.add_argument("--json", action="store_true", help="machine-readable")
    a = ap.parse_args(argv)
    # `root_of`: the argument if one was given, else the `root`
    # setting. Resolved here so two commands cannot disagree
    # about which transcript directory `adder config` names.
    a.root = str(_root_of(a))

    sessions, window = load_window(a)

    if a.json:
        g = scan(a.root, window=window)
        g.assistant_output = sum(t.out for s in sessions.values()
                                 for t in s.turns if not t.sidechain)
        g.thinking = sum(t.thinking for s in sessions.values()
                         for t in s.turns if not t.sidechain)
        print(json.dumps({
            "assistant_output": g.assistant_output,
            "thinking": g.thinking,
            "tool_results": g.tool_results,
            "user_messages": g.user_messages,
            "total": g.total,
            "shares": {k: round(v, 5) for k, v in g.shares().items()},
            "measured_growth": measured_growth(sessions),
            "output_share_of_growth": round(output_share_of_growth(sessions), 5),
            "by_tool": dict(sorted(g.by_tool.items(), key=lambda kv: -kv[1])),
            "tool_calls": dict(sorted(g.tool_calls.items(), key=lambda kv: -kv[1])),
            "filter": window.describe(),
        }))
        return 0

    print()
    print(report(a.root, sessions, window=window))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
