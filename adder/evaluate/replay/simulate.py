"""Re-simulate a session's cost under interventions, to test the composition model.

The headline savings figure assumes the three substitute levers -- terseness,
delegation, splitting -- compose multiplicatively on the residual pool:

    combined = pool * (1 - prod(1 - f_i))

That is an approximation, and it was never checked. This replays each real
session's context trajectory under each intervention and measures the actual
combined saving, so the approximation can be validated or replaced.

Simulation model (each step grounded in a measured claim):
  * context_i = baseline + cumulative admitted content       (measured: growth is
    ~all prior assistant output, ratio 1.02 on non-compacting sessions)
  * terseness t   -> every turn admits (1-t) as much
  * delegation d  -> a fraction d of admitted content is replaced by a summary
  * splitting M   -> context resets to baseline every M turns
  * cost = sum(context_i) * rate_in * 0.10
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

M_TOK = 1_000_000.0


@dataclass(frozen=True)
class Intervention:
    terseness: float = 0.0        # fraction of output not written
    delegation: float = 0.0       # fraction of admissions moved to a subagent
    summary_ratio: float = 0.10   # what a delegated chunk returns
    split_turns: int | None = None

    @property
    def label(self) -> str:
        parts = []
        if self.terseness:
            parts.append(f"terse {self.terseness:.0%}")
        if self.delegation:
            parts.append(f"delegate {self.delegation:.0%}")
        if self.split_turns:
            parts.append(f"split@{self.split_turns}")
        return " + ".join(parts) or "baseline"

    @property
    def pool_fraction(self) -> float:
        """The multiplicative prediction for this intervention."""
        residual = (1 - self.terseness) * (1 - self.delegation * (1 - self.summary_ratio))
        return 1 - residual


def admissions(sess) -> tuple[int, list[float]]:
    """(baseline_context, per-turn admitted tokens) from the real trajectory."""
    if not sess.turns:
        return 0, []
    # Main chain only: the climb back out of a delegated run is not an
    # admission to this context. See `plan._admissions`.
    ctxs = [t.context for t in sess.main_turns]
    baseline = min(ctxs)
    adm: list[float] = []
    prev = ctxs[0]
    for i, c in enumerate(ctxs):
        adm.append(0.0 if i == 0 else float(max(0, c - prev)))
        prev = c
    return baseline, adm


def simulate(sess, iv: Intervention, on: date | None = None) -> float:
    """Total context-read cost for this session under `iv`."""
    if not sess.turns:
        return 0.0
    baseline, adm = admissions(sess)
    r = sess.main_turns[0].rates(on).cache_read

    keep = (1 - iv.terseness)
    keep *= (1 - iv.delegation * (1 - iv.summary_ratio))

    total = 0.0
    cum = 0.0
    for i, a in enumerate(adm):
        if iv.split_turns and i and i % iv.split_turns == 0:
            cum = 0.0                       # fresh session: context back to baseline
        cum += a * keep
        total += (baseline + cum) * r / M_TOK
    return total


def evaluate(sessions, interventions: list[Intervention],
             on: date | None = None) -> list[tuple[Intervention, float, float]]:
    """(intervention, simulated_saving, multiplicative_prediction) per intervention."""
    base = sum(simulate(s, Intervention(), on) for s in sessions.values())
    out = []
    for iv in interventions:
        sim = base - sum(simulate(s, iv, on) for s in sessions.values())
        # Multiplicative prediction, applied to the same addressable pool.
        from adder.measure.spend.debt import decompose_read_cost
        _, _, pool = decompose_read_cost(sessions, on)
        pred = pool * iv.pool_fraction
        if iv.split_turns:
            split_only = base - sum(
                simulate(s, Intervention(split_turns=iv.split_turns), on)
                for s in sessions.values())
            f_split = min(1.0, split_only / pool) if pool else 0.0
            pred = pool * (1 - (1 - iv.pool_fraction) * (1 - f_split))
        out.append((iv, sim, pred))
    return out


def report(sessions, on: date | None = None) -> str:
    ivs = [
        Intervention(terseness=0.30),
        Intervention(delegation=0.25),
        Intervention(split_turns=300),
        Intervention(terseness=0.30, delegation=0.25),
        Intervention(terseness=0.30, split_turns=300),
        Intervention(terseness=0.30, delegation=0.25, split_turns=300),
    ]
    rows = evaluate(sessions, ivs, on)
    base = sum(simulate(s, Intervention(), on) for s in sessions.values())
    lines = [f"  Simulated context-read cost, baseline ${base:,.0f}", "",
             f"  {'intervention':<38}{'simulated':>12}{'predicted':>12}{'error':>9}"]
    for iv, sim, pred in rows:
        err = (pred - sim) / sim if sim else 0.0
        lines.append(f"  {iv.label:<38}${sim:>11,.0f}${pred:>11,.0f}{err:>8.0%}")
    worst = max((abs((p - s) / s) if s else 0) for _, s, p in rows)
    lines += ["",
              f"  Largest error in the multiplicative approximation: {worst:.0%}"]
    lines.append("  Under-prediction is the safe direction; over-prediction inflates"
                 if worst else "")
    lines.append("  the headline figure and must be corrected.")
    return "\n".join(ln for ln in lines if ln is not None)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from adder.core.trace import load_sessions

    ap = argparse.ArgumentParser(prog="adder simulate")
    ap.add_argument("root", nargs="?", default=None,
                    help="transcript directory (default: the `root` setting)")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    from adder.core.filters import root_of as _root_of

    a = ap.parse_args(argv)
    # `root_of`: the argument if one was given, else the `root`
    # setting. Resolved here so two commands cannot disagree
    # about which transcript directory `adder config` names.
    a.root = str(_root_of(a))
    sessions = load_sessions(a.root)
    if a.json:
        import json

        ivs = [Intervention(terseness=0.30), Intervention(delegation=0.25),
               Intervention(split_turns=300),
               Intervention(terseness=0.30, delegation=0.25),
               Intervention(terseness=0.30, split_turns=300),
               Intervention(terseness=0.30, delegation=0.25, split_turns=300)]
        base = sum(simulate(s, Intervention()) for s in sessions.values())
        rows = evaluate(sessions, ivs)
        print(json.dumps({
            "baseline": round(base, 4),
            "interventions": [
                {"label": iv.label, "simulated": round(sim, 4),
                 "predicted": round(pred, 4),
                 "error": round((pred - sim) / sim, 5) if sim else 0.0}
                for iv, sim, pred in rows
            ],
        }))
        return 0
    print()
    print(report(sessions))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
