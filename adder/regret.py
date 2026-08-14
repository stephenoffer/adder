"""Evaluate the remaining-turns estimator by dollar regret, not prediction error.

Prediction accuracy is the wrong metric here, and measurably so. Leave-one-out
cross-validation on real session lengths gives:

    estimator            median rel-err   mean rel-err
    empirical survivor            0.52           1.80
    countdown                     0.64           1.15

The countdown looks better on mean error -- yet it costs far more, because the
two error directions have wildly different prices:

    under-predict -> skip a profitable delegation  (lose the whole saving)
    over-predict  -> delegate marginally           (waste routing overhead ~$0.26)

So an estimator should be judged on the regret of the decisions it drives:

    regret = value(optimal decision under true R) - value(decision under estimate)

Measured that way the ranking inverts. The countdown's failure is specific and
silent: it collapses to zero late in a session and disables delegation entirely
in exactly the sessions where debt compounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .cost import placement_cost
from .horizon import DEFAULT_REMAINING, Horizon
from .policy import routing_overhead

PROBE_TURNS = (50, 100, 200, 400, 600, 1000)

# (read_tokens, summary_tokens, context_tokens) - spans marginal to obvious cases.
SCENARIOS = (
    (1_000, 500, 200_000),
    (3_000, 1_000, 200_000),
    (5_000, 2_500, 500_000),
    (10_000, 5_000, 500_000),
    (20_000, 2_000, 500_000),
    (50_000, 5_000, 900_000),
)


@dataclass
class Result:
    scenario: tuple[int, int, int]
    regret: dict[str, float] = field(default_factory=dict)

    @property
    def spread(self) -> float:
        return max(self.regret.values()) - min(self.regret.values()) if self.regret else 0.0

    @property
    def best(self) -> str:
        return min(self.regret, key=self.regret.get) if self.regret else ""


def _value(read: int, summ: int, remaining: int, model: str) -> float:
    _, _, d = placement_cost(tokens_read=read, summary_tokens=summ,
                             remaining_turns=remaining, main_model=model)
    return d.saving


def evaluate(lengths: list[int], *, model: str = "claude-opus-5",
             scenarios=SCENARIOS, probes=PROBE_TURNS) -> list[Result]:
    """Leave-one-out regret for each estimator, per scenario."""
    out: list[Result] = []
    for read, summ, ctx in scenarios:
        overhead = routing_overhead(ctx, model)
        reg = {"empirical": 0.0, "countdown": 0.0, "flat": 0.0}
        for i, held in enumerate(lengths):
            train = Horizon(sorted(lengths[:i] + lengths[i + 1:]))
            for n in probes:
                if held <= n:
                    continue
                true_val = _value(read, summ, held - n, model)
                best = max(0.0, true_val - overhead)
                for name, est in (("empirical", train.remaining(n)),
                                  ("countdown", train.countdown(n)),
                                  ("flat", DEFAULT_REMAINING)):
                    acted = _value(read, summ, est, model) > overhead
                    reg[name] += best - ((true_val - overhead) if acted else 0.0)
        out.append(Result((read, summ, ctx), reg))
    return out


def report(lengths: list[int], model: str = "claude-opus-5") -> str:
    if len(lengths) < 10:
        return f"  Need >=10 sessions to cross-validate; have {len(lengths)}."
    rows = evaluate(lengths, model=model)
    lines = [f"  Leave-one-out decision regret over {len(lengths)} sessions "
             f"(lower is better)", ""]
    lines.append(f"  {'read':>8}{'summary':>9}{'context':>10}"
                 f"{'empirical':>12}{'countdown':>12}{'flat':>10}   best")
    for r in rows:
        read, summ, ctx = r.scenario
        lines.append(f"  {read:>8,}{summ:>9,}{ctx:>10,}"
                     f"${r.regret['empirical']:>11.2f}${r.regret['countdown']:>11.2f}"
                     f"${r.regret['flat']:>9.2f}   {r.best}")
    tot_e = sum(r.regret["empirical"] for r in rows)
    tot_c = sum(r.regret["countdown"] for r in rows)
    lines += ["",
              f"  total regret: empirical ${tot_e:,.2f}  countdown ${tot_c:,.2f}"]
    if tot_c > 0:
        lines.append(f"  empirical cuts regret {100 * (tot_c - tot_e) / tot_c:.0f}%")
    lines += ["",
              "  Caveat: for large reads in big contexts, delegation wins under any",
              "  non-zero horizon, so the estimator hardly matters there. It earns its",
              "  keep on marginal decisions, and by never collapsing to zero."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .trace import DEFAULT_ROOT, load_sessions

    ap = argparse.ArgumentParser(prog="adder.regret")
    ap.add_argument("root", nargs="?", default=str(DEFAULT_ROOT))
    a = ap.parse_args(argv)
    sessions = load_sessions(a.root)
    lengths = sorted(len(s.turns) for s in sessions.values() if len(s.turns) >= 5)
    print()
    print(report(lengths))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
