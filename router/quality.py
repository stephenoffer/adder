"""Performance proxies, so a cost cut can be caught degrading the agent.

Every lever in this repo trades tokens for something. Terseness can drop the
detail a later turn needed. A cheaper tier can fail and force a retry. Delegation
can lose context and produce a wrong answer. Splitting sessions can throw away
state the next session has to rediscover.

None of that shows up in a cost report -- a degraded agent often looks *cheaper*
per turn while taking more turns to finish. So cost numbers alone cannot tell
you whether an intervention worked, and a router that only reports savings is
measuring the wrong half.

These are proxies read straight out of the transcripts, not a benchmark:

    tool error rate     tool_result blocks flagged `is_error`
    interruption rate   turns the operator aborted
    correction rate     user messages that read as "no, that's wrong"
    turns per prompt    assistant turns spent per human instruction
    rework ratio        repeated edits to the same file in one session

Each is noisy alone. Together, and compared before/after a cutover, they are
enough to falsify "we cut cost and nothing got worse".
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

# Phrases that mark the operator correcting or redirecting the agent.
_CORRECTION = re.compile(
    r"\b(no,|nope\b|that'?s (wrong|not right|incorrect)|not what i|"
    r"you (broke|missed|forgot)|still (failing|broken|wrong)|"
    r"undo|revert|try again|doesn'?t work|didn'?t work)\b", re.I)

_INTERRUPT = re.compile(r"\[Request interrupted", re.I)

# Meta-messages Claude Code injects that are not human instructions.
_SYNTHETIC = re.compile(
    r"^\s*(<(command-name|command-message|local-command|system-reminder|"
    r"user-prompt-submit-hook)|Caveat: The messages below)", re.I)


@dataclass
class QualityStats:
    turns: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    user_prompts: int = 0
    corrections: int = 0
    interrupts: int = 0
    edits: int = 0
    edited_files: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    sessions: int = 0

    @property
    def tool_error_rate(self) -> float:
        return self.tool_errors / self.tool_calls if self.tool_calls else 0.0

    @property
    def correction_rate(self) -> float:
        return self.corrections / self.user_prompts if self.user_prompts else 0.0

    @property
    def interrupt_rate(self) -> float:
        return self.interrupts / self.user_prompts if self.user_prompts else 0.0

    @property
    def turns_per_prompt(self) -> float:
        return self.turns / self.user_prompts if self.user_prompts else 0.0

    @property
    def rework_ratio(self) -> float:
        """Edits per distinct file touched. 1.0 = every file edited once."""
        n = len(self.edited_files)
        return self.edits / n if n else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "tool_error_rate": self.tool_error_rate,
            "correction_rate": self.correction_rate,
            "interrupt_rate": self.interrupt_rate,
            "turns_per_prompt": self.turns_per_prompt,
            "rework_ratio": self.rework_ratio,
        }


# Which direction is bad for each metric. All of these are "higher is worse".
WORSE_IF_HIGHER = ("tool_error_rate", "correction_rate", "interrupt_rate",
                   "turns_per_prompt", "rework_ratio")


def _day(ts) -> date | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _flat_text(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for b in content:
        if isinstance(b, dict) and isinstance(b.get("text"), str):
            parts.append(b["text"])
    return "\n".join(parts)


def scan(root: Path | str, *, since: date | None = None,
         until: date | None = None) -> QualityStats:
    """Read quality proxies from transcripts, optionally windowed by date."""
    root = Path(root).expanduser()
    paths = [root] if root.is_file() else sorted(root.rglob("*.jsonl"))
    q = QualityStats()
    for path in paths:
        try:
            fh = path.open(errors="replace")
        except OSError:
            continue
        counted = False
        seen_msg: set[str] = set()
        with fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                day = _day(d.get("timestamp"))
                if since and (day is None or day < since):
                    continue
                if until and (day is None or day >= until):
                    continue
                if not counted:
                    q.sessions += 1
                    counted = True

                typ = d.get("type")
                msg = d.get("message") or {}
                content = msg.get("content")

                if typ == "assistant":
                    mid = msg.get("id") or ""
                    blocks = content if isinstance(content, list) else []
                    for b in blocks:
                        if isinstance(b, dict) and b.get("type") == "tool_use":
                            q.tool_calls += 1
                            name = b.get("name")
                            inp = b.get("input") or {}
                            if name in ("Edit", "Write", "NotebookEdit") and isinstance(inp, dict):
                                fp = inp.get("file_path") or inp.get("notebook_path")
                                if fp:
                                    q.edits += 1
                                    q.edited_files[str(fp)] += 1
                    if mid and mid in seen_msg:
                        continue
                    if mid:
                        seen_msg.add(mid)
                    q.turns += 1

                elif typ == "user":
                    blocks = content if isinstance(content, list) else []
                    is_tool_reply = any(
                        isinstance(b, dict) and b.get("type") == "tool_result"
                        for b in blocks
                    )
                    for b in blocks:
                        if (isinstance(b, dict) and b.get("type") == "tool_result"
                                and b.get("is_error")):
                            q.tool_errors += 1
                    text = _flat_text(content)
                    if _INTERRUPT.search(text):
                        q.interrupts += 1
                    # A real human instruction: not a tool reply, not injected meta.
                    if not is_tool_reply and text.strip() and not _SYNTHETIC.match(text):
                        q.user_prompts += 1
                        if _CORRECTION.search(text):
                            q.corrections += 1
    return q


def compare(root: Path | str, cutover: date) -> tuple[QualityStats, QualityStats]:
    return scan(root, until=cutover), scan(root, since=cutover)


def regressions(before: QualityStats, after: QualityStats, *,
                tolerance: float = 0.10) -> list[str]:
    """Metrics that got materially worse. Empty list means nothing regressed.

    `tolerance` is the relative move treated as noise. These proxies are noisy
    and the comparison is uncontrolled, so small moves mean nothing.
    """
    out = []
    b, a = before.as_dict(), after.as_dict()
    for k in WORSE_IF_HIGHER:
        if not b[k]:
            continue
        rel = (a[k] - b[k]) / b[k]
        if rel > tolerance:
            out.append(f"{k} rose {rel:+.0%} ({b[k]:.3f} -> {a[k]:.3f})")
    return out


def report(root: Path | str, cutover: date | None = None) -> str:
    if cutover is None:
        q = scan(root)
        lines = ["  Agent performance proxies", ""]
        lines.append(f"  {q.sessions:,} transcripts · {q.turns:,} turns · "
                     f"{q.user_prompts:,} human prompts")
        lines.append("")
        lines.append(f"  tool error rate    {q.tool_error_rate:>8.2%}   "
                     f"({q.tool_errors:,} of {q.tool_calls:,} tool calls)")
        lines.append(f"  correction rate    {q.correction_rate:>8.2%}   "
                     f"({q.corrections:,} prompts read as a correction)")
        lines.append(f"  interrupt rate     {q.interrupt_rate:>8.2%}   "
                     f"({q.interrupts:,} aborted turns)")
        lines.append(f"  turns per prompt   {q.turns_per_prompt:>8.1f}")
        lines.append(f"  rework ratio       {q.rework_ratio:>8.2f}   "
                     f"(edits per distinct file)")
        lines.append("")
        lines.append("  These are proxies, not a benchmark. Use them as a before/after")
        lines.append("  guard: `rt quality --since YYYY-MM-DD` after changing anything.")
        return "\n".join(lines)

    b, a = compare(root, cutover)
    if not b.turns or not a.turns:
        return (f"  Not enough data around {cutover} to compare "
                f"(before={b.turns:,} turns, after={a.turns:,} turns).")
    lines = [f"  Agent performance across {cutover}", ""]
    lines.append(f"  {'metric':<20}{'before':>10}{'after':>10}{'change':>10}")
    bd, ad = b.as_dict(), a.as_dict()
    for k in WORSE_IF_HIGHER:
        chg = "     n/a" if not bd[k] else f"{100*(ad[k]-bd[k])/bd[k]:+9.0f}%"
        fmt = "{:.3f}" if k != "turns_per_prompt" else "{:.1f}"
        lines.append(f"  {k:<20}{fmt.format(bd[k]):>10}{fmt.format(ad[k]):>10}{chg:>10}")
    regs = regressions(b, a)
    lines.append("")
    if regs:
        lines.append("  REGRESSED — the cost change may have cost you capability:")
        for r in regs:
            lines.append(f"    - {r}")
        lines.append("")
        lines.append("  Do not claim a clean saving. A cheaper agent that needs more")
        lines.append("  turns or more corrections is not cheaper.")
    else:
        lines.append("  No metric regressed beyond noise. The cost change looks clean.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .trace import DEFAULT_ROOT

    ap = argparse.ArgumentParser(
        prog="router.quality",
        description="Measure agent-performance proxies, before/after a change.")
    ap.add_argument("root", nargs="?", default=str(DEFAULT_ROOT))
    ap.add_argument("--since", help="cutover date, YYYY-MM-DD")
    a = ap.parse_args(argv)
    cut = date.fromisoformat(a.since) if a.since else None
    print()
    print(report(a.root, cut))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
