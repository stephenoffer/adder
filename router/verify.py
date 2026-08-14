"""Did an intervention actually work?

Every saving in this repo is a projection until it shows up in the transcripts.
This compares measured output-per-turn and cost-per-turn before and after a
cutover date, so a change (a terseness block in CLAUDE.md, Explore on Haiku,
shorter sessions) can be confirmed or falsified against real data.

Cost per turn is the headline metric because it captures both halves: writing
less reduces generation cost now and context-read cost on every later turn.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from .trace import DEFAULT_ROOT, load_sessions


def _day(ts: str | None) -> date | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
    except ValueError:
        return None


@dataclass
class Window:
    label: str
    turns: int = 0
    sessions: int = 0
    out: int = 0
    cost: float = 0.0
    ctx: list[int] | None = None

    @property
    def out_per_turn(self) -> float:
        return self.out / self.turns if self.turns else 0.0

    @property
    def cost_per_turn(self) -> float:
        return self.cost / self.turns if self.turns else 0.0

    @property
    def median_ctx(self) -> float:
        return statistics.median(self.ctx) if self.ctx else 0.0


def compare(cutover: date, root: Path | str = DEFAULT_ROOT) -> tuple[Window, Window]:
    before, after = Window("before"), Window("after")
    seen: dict[str, set[str]] = {"before": set(), "after": set()}
    for s in load_sessions(root).values():
        for t in s.turns:
            d = _day(t.ts)
            if d is None:
                continue
            w = before if d < cutover else after
            key = "before" if w is before else "after"
            if w.ctx is None:
                w.ctx = []
            w.turns += 1
            w.out += t.out
            w.cost += t.cost()
            w.ctx.append(t.context)
            if s.id not in seen[key]:
                seen[key].add(s.id)
                w.sessions += 1
    return before, after


def _delta(a: float, b: float) -> str:
    if not a:
        return "     n/a"
    pct = 100 * (b - a) / a
    return f"{pct:+7.1f}%"


def report(cutover: date, root: Path | str = DEFAULT_ROOT) -> str:
    b, a = compare(cutover, root)
    if not b.turns or not a.turns:
        missing = "before" if not b.turns else "after"
        return (f"  Not enough data {missing} {cutover} to compare "
                f"(before={b.turns:,} turns, after={a.turns:,} turns).")

    rows = [
        ("output tokens / turn", b.out_per_turn, a.out_per_turn, "{:,.0f}"),
        ("cost / turn", b.cost_per_turn, a.cost_per_turn, "${:,.4f}"),
        ("median context", b.median_ctx, a.median_ctx, "{:,.0f}"),
        ("turns / session", b.turns / max(1, b.sessions), a.turns / max(1, a.sessions), "{:,.0f}"),
    ]
    out = [f"  Cutover {cutover}   before: {b.turns:,} turns / {b.sessions} sessions"
           f"   after: {a.turns:,} turns / {a.sessions} sessions", ""]
    out.append(f"  {'metric':<24}{'before':>14}{'after':>14}{'change':>12}")
    for label, x, y, fmt in rows:
        out.append(f"  {label:<24}{fmt.format(x):>14}{fmt.format(y):>14}{_delta(x, y):>12}")

    # Decompose the change. Context at a typical turn scales with CUMULATIVE
    # output, i.e. output-per-turn x session-length. The two factors are
    # multiplicative, which is why cutting verbosity alone can be cancelled out
    # entirely by longer sessions.
    v_ratio = (a.out_per_turn / b.out_per_turn) if b.out_per_turn else 1.0
    l_ratio = ((a.turns / max(1, a.sessions)) / (b.turns / max(1, b.sessions))
               if b.sessions and b.turns else 1.0)
    predicted = v_ratio * l_ratio
    actual = (a.median_ctx / b.median_ctx) if b.median_ctx else 1.0
    out.append("")
    out.append("  Why context moved (cost/turn tracks context almost exactly):")
    out.append(f"    verbosity effect      x{v_ratio:>6.3f}   ({_delta(1.0, v_ratio).strip()})")
    out.append(f"    session-length effect x{l_ratio:>6.3f}   ({_delta(1.0, l_ratio).strip()})")
    out.append(f"    predicted context     x{predicted:>6.3f}   (product of the two)")
    out.append(f"    actual context        x{actual:>6.3f}")
    if v_ratio < 0.95 and l_ratio > 1.05 and predicted > 1.0:
        out.append("")
        out.append("    Writing less was CANCELLED OUT by longer sessions. Terseness only")
        out.append("    pays if session length is held constant - the two multiply.")

    saved_per_turn = b.cost_per_turn - a.cost_per_turn
    out.append("")
    if saved_per_turn > 0:
        out.append(f"  Cost per turn fell ${saved_per_turn:.4f}. Over the {a.turns:,} turns")
        out.append(f"  since cutover that is ${saved_per_turn * a.turns:,.2f} saved.")
    else:
        out.append(f"  Cost per turn ROSE ${-saved_per_turn:.4f}. The intervention did not land,")
        out.append("  or something else changed. Do not claim a saving.")
    out.append("")
    out.append("  Caveat: this is an uncontrolled before/after, not an A/B. Task mix")
    out.append("  changes between periods too. Treat a small move as noise.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="router.verify",
                                 description="Measure whether an intervention landed.")
    ap.add_argument("--since", required=True, help="cutover date, YYYY-MM-DD")
    ap.add_argument("root", nargs="?", default=str(DEFAULT_ROOT))
    a = ap.parse_args(argv)
    print()
    print(report(date.fromisoformat(a.since), a.root))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
