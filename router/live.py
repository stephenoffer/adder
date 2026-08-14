"""Current-session cost analysis.

Answers the question that actually changes behaviour mid-session: "what is this
conversation costing me per turn right now, and what will the next big read cost?"

Everything here is priced from the session's own measured turns, so the advice
adapts to the model, cache TTL, and context actually in play rather than to a
global average.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .cost import admitted_token_cost, marginal_turn_cost, placement_cost
from .debt import debt_multiple
from .horizon import Horizon
from .horizon import load as load_horizon
from .prices import context_limit
from .trace import DEFAULT_ROOT, Session, iter_file

# Descriptive session-length stats, measured on deduplicated transcripts. These
# are for reporting only -- remaining turns comes from `horizon`, because a
# countdown from a median is badly wrong on a heavy-tailed length distribution.
# (The pre-deduplication figures were 607/1159; every multi-block turn was being
# counted once per content block. See `trace.iter_file`.)
MEDIAN_SESSION_TURNS = 340
P90_SESSION_TURNS = 759

# Above this share of the model's window, compaction is imminent and the next
# turns are the most expensive of the session.
CONTEXT_PRESSURE = 0.75


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
    """Most recently modified transcript for this working directory.

    Reads only that one file. An earlier version fell back to parsing every
    transcript in the directory and treating the union as one session, which
    reported the sum of unrelated conversations as "this session".
    """
    d = find_project_dir(cwd, root)
    if d is None:
        return None
    files = sorted(d.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
    for newest in files[:3]:          # skip empty/unpriced files, don't merge them
        turns = list(iter_file(newest))
        if turns:
            s = Session(newest.stem, d.name)
            s.turns = turns
            return s
    return None


@dataclass
class LiveReport:
    turns: int
    context: int
    spent: float
    per_turn: float
    projected_remaining: int
    projected_total: float
    model: str
    out_per_turn: int = 0
    median_gap: float = 0.0
    ttl: str = "5m"

    @property
    def context_pressure(self) -> float:
        """How full the model's window is. Past ~0.75 compaction is imminent."""
        return self.context / max(1, context_limit(self.model))

    @property
    def next_turn_cost(self) -> float:
        return marginal_turn_cost(self.context, self.out_per_turn, self.model)

    @property
    def debt_multiple(self) -> float:
        """What a token written now really costs, vs its sticker price."""
        return debt_multiple(self.projected_remaining, self.model)

    def read_cost(self, tokens: int) -> tuple[float, float, str]:
        """What reading `tokens` inline costs vs delegating it, from here."""
        inline, sub, d = placement_cost(
            tokens_read=tokens,
            summary_tokens=max(200, tokens // 10),
            remaining_turns=self.projected_remaining,
            main_model=self.model,
        )
        return inline, sub, ("delegate" if d else "inline")


def analyse(sess: Session, *, horizon: Horizon | None = None) -> LiveReport:
    """Price the session so far and project the rest of it.

    `remaining` comes from the empirical survivor function, not a countdown from
    a median length. Session length is heavy-tailed and close to memoryless, so
    reaching turn 600 is evidence of being in a LONG session, not of being near
    its end -- a countdown says 7 turns left where the data says ~350.
    """
    spent = sess.cost
    n = sess.n_turns
    last = sess.turns[-1]
    h = horizon if horizon is not None else load_horizon()
    remaining = h.remaining(n)
    per_turn = spent / max(1, n)
    return LiveReport(
        turns=n,
        context=last.context,
        spent=spent,
        per_turn=per_turn,
        projected_remaining=remaining,
        projected_total=spent + per_turn * remaining,
        model=last.model,
        out_per_turn=sess.out_tokens // max(1, n),
        median_gap=sess.median_gap(),
        ttl=last.ttl,
    )


def render(sess: Session | None) -> str:
    if sess is None:
        return "  No transcript found for this directory yet."
    r = analyse(sess)
    out = [
        f"  This session: {r.turns:,} turns · {r.context:,} tokens in context · "
        f"${r.spent:,.2f} spent (${r.per_turn:.3f}/turn)",
        f"  Model {r.model} · cache TTL {r.ttl} · "
        f"context {r.context_pressure:.0%} of the {context_limit(r.model):,}-token window",
    ]
    if r.projected_remaining:
        out.append(
            f"  Sessions that reach turn {r.turns:,} typically run ~{r.projected_remaining:,} "
            f"more turns → ~${r.projected_total:,.2f} total"
        )
    out.append(f"  One more turn at this context costs ~${r.next_turn_cost:.3f}.")

    if r.context_pressure >= CONTEXT_PRESSURE:
        out.append("")
        out.append("  ⚠ Context is near the window limit. The next turns are the most")
        out.append("    expensive of the session, and compaction is imminent — which")
        out.append("    rebuilds the cache at 1.25x instead of reading it at 0.10x.")

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
    out.append(
        f"  An output token written now costs {r.debt_multiple:.1f}x its sticker price, "
        f"once downstream re-reads are counted."
    )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="router.live")
    ap.add_argument("--cwd", default=None)
    a = ap.parse_args(argv)
    print()
    print(render(current_session(a.cwd)))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
