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
from datetime import date
from pathlib import Path

from adder.core.filters import day_of as _day
from adder.core.filters import root_of as _root_of
from adder.pricing.prices import is_synthetic
from adder.util.records import mapping

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
    api_errors: int = 0
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
    def api_error_rate(self) -> float:
        """Client-side failure records per assistant turn.

        Claude Code writes an assistant record with model `<synthetic>` when the
        *client* produced the message rather than the API: a dropped
        connection, a stream that ended early, a context that would not fit.
        Those are not billed, so no cost report sees them -- but they are a
        real quality signal, and the last of the three is caused by exactly the
        context growth this tool measures.

        Deliberately excluded from the before/after regression check: most of
        these are network flakiness, and failing a cost change because someone
        was on hotel wifi is a false positive that teaches people to ignore the
        check.
        """
        return self.api_errors / self.turns if self.turns else 0.0

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
         until: date | None = None, window=None) -> QualityStats:
    """Read quality proxies from transcripts, windowed by date or by `window`.

    `window` is a `filters.Window` and supersedes `since`/`until` when given,
    so a caller can scope these proxies to one project or one session the same
    way every other report does. The bare dates stay because `compare()` is
    built on them and a cutover is naturally two date-bounded scans.
    """
    from adder.core.trace import transcripts

    paths = transcripts(root)
    q = QualityStats()
    # Deduplication is scan-wide, not per file, and it is by the id of the
    # thing being counted. Two mechanisms replay records here: Claude Code
    # writes one record per content block repeating the message envelope on
    # each, and a resumed session writes a new `.jsonl` that restates earlier
    # turns. `trace.load_sessions`, `tools.scan`, `reread.scan` and
    # `shapes.iter_results` all dedup for this reason; this scan did not, so
    # every rate it reports -- tool errors, corrections, turns per prompt --
    # was computed over a population that counts a resumed session twice.
    seen_msg: set[tuple[str, str]] = set()
    seen_use: set[str] = set()
    answered: set[str] = set()
    seen_files: set[str] = set()
    for path in paths:
        try:
            fh = path.open(encoding="utf-8", errors="replace")
        except OSError:
            continue
        counted = False
        with fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if window is not None:
                    if not window.keeps_record(d, path.parent.name):
                        continue
                else:
                    day = _day(d.get("timestamp"))
                    if since and (day is None or day < since):
                        continue
                    if until and (day is None or day >= until):
                        continue
                typ = d.get("type")
                session = str(d.get("sessionId") or path.stem)
                if not counted:
                    # By conversation, not by file: a resumed session is two
                    # transcripts, and counting both made `sessions` disagree
                    # with every other report on the same corpus.
                    counted = True
                    if session not in seen_files:
                        seen_files.add(session)
                        q.sessions += 1
                msg = mapping(d, "message")
                content = msg.get("content")

                if typ == "assistant":
                    mid = str(msg.get("id") or "")
                    if is_synthetic(str(msg.get("model") or "")):
                        q.api_errors += 1
                        continue
                    blocks = content if isinstance(content, list) else []
                    for b in blocks:
                        if isinstance(b, dict) and b.get("type") == "tool_use":
                            use_id = str(b.get("id") or "")
                            if use_id:
                                if use_id in seen_use:
                                    continue
                                seen_use.add(use_id)
                            q.tool_calls += 1
                            name = b.get("name")
                            inp = b.get("input") or {}
                            if name in ("Edit", "Write", "NotebookEdit") and isinstance(inp, dict):
                                fp = inp.get("file_path") or inp.get("notebook_path")
                                if fp:
                                    q.edits += 1
                                    q.edited_files[str(fp)] += 1
                    key = (session, mid)
                    if mid and key in seen_msg:
                        continue
                    if mid:
                        seen_msg.add(key)
                    q.turns += 1

                elif typ == "user":
                    blocks = content if isinstance(content, list) else []
                    is_tool_reply = any(
                        isinstance(b, dict) and b.get("type") == "tool_result"
                        for b in blocks
                    )
                    for b in blocks:
                        if not (isinstance(b, dict) and b.get("type") == "tool_result"):
                            continue
                        use_id = str(b.get("tool_use_id") or "")
                        if use_id:
                            if use_id in answered:
                                continue
                            answered.add(use_id)
                        if b.get("is_error"):
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
            # A metric that was zero and is not any more. `continue` here meant
            # the check was blind to the clearest regression it could ever see:
            # a tool error rate going 0 -> 30%, a correction rate going 0 -> 50%,
            # and this function returning "nothing regressed". The relative move
            # is undefined against a zero baseline, not zero, and the two were
            # being treated as the same thing -- in a function whose whole job
            # is to falsify "we cut cost and nothing got worse".
            if a[k] > 0:
                out.append(f"{k} appeared: 0.000 -> {a[k]:.3f} "
                           f"(no baseline to compare against)")
            continue
        rel = (a[k] - b[k]) / b[k]
        if rel > tolerance:
            out.append(f"{k} rose {rel:+.0%} ({b[k]:.3f} -> {a[k]:.3f})")
    return out


def report(root: Path | str, cutover: date | None = None, *, window=None) -> str:
    if cutover is None:
        q = scan(root, window=window)
        lines = ["  Agent performance proxies", ""]
        # See `Window.ignores_model`: a raw-record scan cannot honour a model
        # filter, and accepting one silently widens every rate below.
        if window is not None and getattr(window, "ignores_model", False):
            lines.append("  note: --model-filter is not applied here — a tool "
                         "result carries no model")
            lines.append("")
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
        if q.api_errors:
            lines.append(f"  api error rate     {q.api_error_rate:>8.2%}   "
                         f"({q.api_errors:,} client-side failure records)")
        lines.append("")
        lines.append("  These are proxies, not a benchmark. Use them as a before/after")
        lines.append("  guard: `adder quality --since YYYY-MM-DD` after changing anything.")
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


    ap = argparse.ArgumentParser(
        prog="adder quality",
        description="Measure agent-performance proxies, before/after a change.")
    ap.add_argument("root", nargs="?", default=None,
                    help="transcript directory (default: the `root` setting)")
    ap.add_argument("--since", help="cutover date: YYYY-MM-DD, 7d, 2w, today")
    ap.add_argument("--until", help="upper bound, exclusive (same formats)")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    a = ap.parse_args(argv)
    # `root_of`: the argument if one was given, else the `root`
    # setting. Resolved here so two commands cannot disagree
    # about which transcript directory `adder config` names.
    a.root = str(_root_of(a))

    from adder.core.filters import parse_date

    cut = parse_date(a.since) if a.since else None
    until = parse_date(a.until) if a.until else None

    if a.json:
        if cut and not until:
            b, aft = compare(a.root, cut)
            print(json.dumps({
                "cutover": cut.isoformat(),
                "before": {**b.as_dict(), "turns": b.turns, "prompts": b.user_prompts},
                "after": {**aft.as_dict(), "turns": aft.turns,
                          "prompts": aft.user_prompts},
                "regressions": regressions(b, aft),
            }))
            return 0
        q = scan(a.root, since=cut, until=until)
        print(json.dumps({
            **q.as_dict(),
            "turns": q.turns, "sessions": q.sessions,
            "api_errors": q.api_errors,
            "api_error_rate": q.api_error_rate,
            "tool_calls": q.tool_calls, "tool_errors": q.tool_errors,
            "user_prompts": q.user_prompts, "corrections": q.corrections,
            "interrupts": q.interrupts, "edits": q.edits,
            "distinct_files_edited": len(q.edited_files),
        }))
        return 0

    print()
    print(report(a.root, cut))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
