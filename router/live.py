"""Current-session cost analysis.

Answers the question that actually changes behaviour mid-session: "what is this
conversation costing me per turn right now, and what will the next big read cost?"
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .cost import admitted_token_cost, placement_cost
from .trace import DEFAULT_ROOT, Session, Turn, iter_turns

# Session-length priors measured from this machine's transcripts.
MEDIAN_SESSION_TURNS = 607
P90_SESSION_TURNS = 1159


def slug_for(cwd: Path | str | None = None) -> str:
    """Claude Code's project directory name for a working directory.

    Non-alphanumeric characters all collapse to '-', not just path separators:
    /Users/stephen.offer/Desktop/x -> -Users-stephen-offer-Desktop-x
    (the dot in a username becomes a dash too).
    """
    p = Path(cwd or os.getcwd()).resolve()
    return re.sub(r"[^A-Za-z0-9]", "-", str(p))


def find_project_dir(cwd: Path | str | None = None, root: Path | str = DEFAULT_ROOT) -> Path | None:
    """Locate the transcript directory, falling back to a case-insensitive match."""
    root = Path(root).expanduser()
    exact = root / slug_for(cwd)
    if exact.is_dir():
        return exact
    want = slug_for(cwd).lower()
    for d in root.iterdir() if root.is_dir() else []:
        if d.is_dir() and d.name.lower() == want:
            return d
    return None


def current_session(cwd: Path | str | None = None, root: Path | str = DEFAULT_ROOT) -> Session | None:
    """Most recently modified transcript for this working directory."""
    d = find_project_dir(cwd, root)
    if d is None:
        return None
    files = sorted(d.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        return None
    newest = files[0]
    turns = [t for t in iter_turns(newest.parent) if t.session == newest.stem]
    if not turns:
        turns = [t for t in iter_turns(newest.parent)]
    if not turns:
        return None
    s = Session(newest.stem, d.name)
    s.turns = turns
    return s


@dataclass
class LiveReport:
    turns: int
    context: int
    spent: float
    per_turn: float
    projected_remaining: int
    projected_total: float
    model: str

    def read_cost(self, tokens: int) -> tuple[float, float, str]:
        """What reading `tokens` inline costs vs delegating it, from here."""
        inline, sub, d = placement_cost(
            tokens_read=tokens,
            summary_tokens=max(200, tokens // 10),
            remaining_turns=self.projected_remaining,
            main_model=self.model,
        )
        return inline, sub, ("delegate" if d else "inline")


def analyse(sess: Session, *, horizon: int = MEDIAN_SESSION_TURNS) -> LiveReport:
    spent = sess.cost
    n = sess.n_turns
    last = sess.turns[-1]
    remaining = max(0, horizon - n)
    per_turn = spent / max(1, n)
    return LiveReport(
        turns=n,
        context=last.context,
        spent=spent,
        per_turn=per_turn,
        projected_remaining=remaining,
        projected_total=spent + per_turn * remaining,
        model=last.model,
    )


def render(sess: Session | None) -> str:
    if sess is None:
        return "  No transcript found for this directory yet."
    r = analyse(sess)
    out = [
        f"  This session: {r.turns:,} turns · {r.context:,} tokens in context · "
        f"${r.spent:,.2f} spent (${r.per_turn:.3f}/turn)",
    ]
    if r.projected_remaining:
        out.append(
            f"  At the median session length ({MEDIAN_SESSION_TURNS} turns), "
            f"~{r.projected_remaining:,} turns left → ~${r.projected_total:,.2f} total"
        )
    out.append("")
    out.append("  Cost of reading a file into THIS context from here:")
    out.append(f"    {'file size':>12}  {'inline':>10}  {'delegated':>10}   verdict")
    for tokens in (5_000, 20_000, 50_000, 150_000):
        inline, sub, verdict = r.read_cost(tokens)
        out.append(f"    {tokens:>10,} tok  ${inline:>9,.3f}  ${sub:>9,.3f}   {verdict}")
    cost10k = admitted_token_cost(10_000, r.model, r.projected_remaining)
    out.append("")
    out.append(
        f"  Every 10K tokens added to this context now costs ~${cost10k:,.2f} "
        f"over the rest of the session."
    )
    return "\n".join(out)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(prog="router.live")
    ap.add_argument("--cwd", default=None)
    a = ap.parse_args()
    print()
    print(render(current_session(a.cwd)))
    print()
