"""Is `p_fail` a probability, or just a number the gate happens to accept?

The claim this module tests
---------------------------
`adder outcomes` estimates `p_fail`: how often a tier fails and forces an
escalation. Every routing gate in this package multiplies by it. A gate fed a
probability that is systematically 15 points too low escalates too rarely, and
the resulting failures are attributed to the model rather than to the estimate
that sent the work there.

The estimator has never been scored. It has been *inspected* -- `outcomes
calibration` prints the escalation rate per tier -- but printing the rate you
fitted next to the data you fitted it on is not a test, it is a tautology. This
module scores it properly.

Prequential, because a retrospective score is worthless here
------------------------------------------------------------
Scoring a predictor on the data used to fit it measures memorisation. The fix
is not a train/test split: the estimator is *online* and recency-weighted, so a
random split would let it learn from the future, and a chronological split
would waste most of the log.

So this runs the log **prequentially**: walk it in timestamp order, and at each
row ask the estimator -- given only the rows before this one -- for its
probability, then reveal the outcome. Every prediction is genuinely
out-of-sample, every row is used, and the score is exactly what the deployed
gate would have experienced. This is the standard way to evaluate an online
predictor and it is the only one that is honest about a recency-weighted fit.

What gets reported, and why each one
-------------------------------------
* **Brier score** -- mean squared error of the probabilities. Proper: it cannot
  be improved by shading predictions toward the middle.
* **Brier skill score** against the base rate. This is the one that matters. A
  Brier of 0.18 sounds fine until you learn that always predicting the overall
  escalation rate scores 0.17, at which point the estimator has added nothing
  and its per-project scoping is decoration.
* **Calibration error** -- do things predicted at 30% happen 30% of the time?
  A gate multiplies by the number, so a bias here is a bias in dollars.
* **Reliability by bin** -- where the miscalibration lives. Over-confidence at
  the top of the range and at the bottom have opposite fixes.
* **Dollar regret** -- the only one anybody acts on. For each row, would the
  escalation gate have decided differently under the predicted probability
  than under the realised one, and what did that decision cost? Miscalibration
  is only a problem where it flips a decision, and this counts the flips and
  prices them.

A note on what a low score would mean
-------------------------------------
Not that the tool is broken. `p_fail` is estimated from a few hundred rows,
scoped per project, over a task mix that shifts weekly. A skill score near zero
would mean the scoping is not paying for itself and the global rate is as good,
which is a finding worth having rather than a failure to hide. The test exists
to be able to say that.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from adder.util import render
from adder.util.stats import bootstrap_ci, mean, share, wilson_interval

# Reliability bins. Ten is the convention; the bins are equal-width rather than
# equal-count so a bin's position on the axis means what it says, and an empty
# bin is reported as empty instead of being silently merged away.
N_BINS = 10

# Below this many observations a bin's observed rate is noise. It is still
# printed -- hiding it would misrepresent coverage -- but it is marked, and it
# does not contribute to the headline calibration error.
MIN_BIN = 5


@dataclass(frozen=True)
class Prediction:
    """One prequential prediction and the outcome that followed it."""

    key: str
    predicted: float
    happened: bool
    cost: float = 0.0
    tier: str = ""
    project: str = ""


@dataclass
class Bin:
    lo: float
    hi: float
    n: int = 0
    predicted_sum: float = 0.0
    happened: int = 0

    @property
    def predicted(self) -> float:
        return share(self.predicted_sum, self.n)

    @property
    def observed(self) -> float:
        return share(self.happened, self.n)

    @property
    def gap(self) -> float:
        return self.observed - self.predicted

    @property
    def thin(self) -> bool:
        return self.n < MIN_BIN

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.happened, self.n)

    @property
    def consistent(self) -> bool:
        """True when the prediction lands inside the observed rate's interval."""
        lo, hi = self.interval
        return lo <= self.predicted <= hi


@dataclass
class Report:
    n: int = 0
    base_rate: float = 0.0
    brier: float = 0.0
    brier_base: float = 0.0
    log_loss: float = 0.0
    ece: float = 0.0
    mce: float = 0.0
    bias: float = 0.0
    resolution: float = 0.0
    bins: list[Bin] = field(default_factory=list)
    brier_ci: tuple[float, float] = (0.0, 0.0)
    flips: int = 0
    flip_cost: float = 0.0

    @property
    def skill(self) -> float:
        """Brier skill score against always predicting the base rate.

        1.0 is perfect, 0.0 is "no better than the base rate", and negative
        means the estimator is actively worse than a constant -- which is a
        thing that happens and which no per-tier table would ever reveal.
        """
        if self.brier_base <= 0:
            return 0.0
        return 1.0 - (self.brier / self.brier_base)

    @property
    def beats_base_rate(self) -> bool:
        """Only true when the interval clears the constant predictor."""
        return self.brier_ci[1] < self.brier_base

    @property
    def calibrated(self) -> bool:
        """Every well-populated bin's prediction sits inside its own interval."""
        solid = [b for b in self.bins if not b.thin]
        return bool(solid) and all(b.consistent for b in solid)

    def to_json(self) -> dict:
        return {
            "n": self.n,
            "base_rate": self.base_rate,
            "brier": self.brier,
            "brier_base_rate": self.brier_base,
            "brier_ci95": list(self.brier_ci),
            "skill": self.skill,
            "beats_base_rate": self.beats_base_rate,
            "log_loss": self.log_loss,
            "ece": self.ece,
            "mce": self.mce,
            "bias": self.bias,
            "resolution": self.resolution,
            "calibrated": self.calibrated,
            "decision_flips": self.flips,
            "flip_cost_usd": self.flip_cost,
            "bins": [
                {"lo": b.lo, "hi": b.hi, "n": b.n, "predicted": b.predicted,
                 "observed": b.observed, "thin": b.thin,
                 "consistent": b.consistent}
                for b in self.bins
            ],
        }


def brier(preds: list[Prediction]) -> float:
    """Mean squared error of the probabilities. Lower is better; 0.25 is a coin."""
    if not preds:
        return 0.0
    return mean([(p.predicted - float(p.happened)) ** 2 for p in preds])


def log_loss(preds: list[Prediction], *, eps: float = 1e-6) -> float:
    """Cross-entropy. Reported next to Brier because it punishes confidence.

    A predictor that says 0.99 and is wrong once is barely dented in Brier and
    heavily penalised here. Since the gate acts most decisively on extreme
    probabilities, that is the failure worth being sensitive to.
    """
    import math

    if not preds:
        return 0.0
    total = 0.0
    for p in preds:
        q = min(max(p.predicted, eps), 1.0 - eps)
        total += -(math.log(q) if p.happened else math.log(1.0 - q))
    return total / len(preds)


def bins_of(preds: list[Prediction], n_bins: int = N_BINS) -> list[Bin]:
    """Group predictions into equal-width probability bins."""
    out = [Bin(i / n_bins, (i + 1) / n_bins) for i in range(n_bins)]
    for p in preds:
        idx = min(n_bins - 1, max(0, int(p.predicted * n_bins)))
        b = out[idx]
        b.n += 1
        b.predicted_sum += p.predicted
        b.happened += int(p.happened)
    return out


def calibration_error(bins: list[Bin], total: int) -> tuple[float, float]:
    """`(expected, maximum)` calibration error over the well-populated bins.

    Thin bins are excluded from both. A bin holding two observations can show a
    50-point gap from pure noise, and letting it set the headline number turns
    the metric into a random variable.
    """
    solid = [b for b in bins if not b.thin]
    if not solid or total <= 0:
        return (0.0, 0.0)
    weight = sum(b.n for b in solid)
    if weight <= 0:
        return (0.0, 0.0)
    ece = sum(b.n * abs(b.gap) for b in solid) / weight
    mce = max(abs(b.gap) for b in solid)
    return (ece, mce)


def decomposition(preds: list[Prediction], bins: list[Bin]) -> tuple[float, float]:
    """`(bias, resolution)`: is it shifted, and does it discriminate at all?

    Bias is the plain difference between mean prediction and base rate -- a
    constant offset, and the easiest thing to fix. Resolution is how far the
    bins' observed rates spread away from the base rate; a predictor with zero
    resolution is a constant wearing a probability's clothes, no matter how
    good its calibration looks.
    """
    if not preds:
        return (0.0, 0.0)
    base = mean([float(p.happened) for p in preds])
    bias = mean([p.predicted for p in preds]) - base
    n = len(preds)
    resolution = sum(b.n * (b.observed - base) ** 2 for b in bins if b.n) / n
    return (bias, resolution)


def prequential(
    rows: list,
    *,
    scoped: bool = True,
    warmup: int = 5,
) -> list[Prediction]:
    """Replay the outcome log in order, predicting each row from its past only.

    `warmup` rows are consumed before scoring begins. With no history the
    estimator returns its Beta(1,1) prior of 0.5 for everything, and scoring
    those rows measures the prior rather than the estimator.
    """
    from adder.decide.track.outcomes import p_fail

    ordered = sorted(rows, key=lambda o: o.ts)
    out: list[Prediction] = []
    for i, row in enumerate(ordered):
        if i < warmup:
            continue
        history = ordered[:i]
        # Decay toward this row's own moment, not the wall clock: the question
        # is what the gate would have said when this row arrived.
        predicted = p_fail(row.tier, row.project if scoped else None,
                           outcomes=history, now=float(row.ts))
        out.append(Prediction(
            key=getattr(row, "task_hash", "") or f"row{i}",
            predicted=float(predicted),
            happened=bool(row.escalated),
            cost=float(getattr(row, "cost", 0.0) or 0.0),
            tier=str(row.tier),
            project=str(row.project),
        ))
    return out


def decision_flips(preds: list[Prediction], *, threshold: float = 0.5) -> tuple[int, float]:
    """How often the predicted probability sits on the wrong side of the gate.

    Miscalibration only costs money where it changes a decision. A prediction
    of 0.30 against an outcome that failed is inaccurate; it is only *expensive*
    if 0.30 was below the escalation threshold and the failure that followed had
    to be paid for twice -- once for the tier that failed, once for the retry.

    The cost attributed is the recorded cost of the row, which is what was
    really spent on the arm that failed. It is a floor: the retry cost more.
    """
    flips = 0
    cost = 0.0
    for p in preds:
        predicted_escalate = p.predicted >= threshold
        if predicted_escalate != p.happened:
            flips += 1
            if p.happened and not predicted_escalate:
                # The expensive direction: told not to escalate, then failed.
                cost += p.cost
    return (flips, cost)


def evaluate(preds: list[Prediction], *, n_bins: int = N_BINS) -> Report:
    rep = Report(n=len(preds))
    if not preds:
        return rep
    rep.base_rate = mean([float(p.happened) for p in preds])
    rep.brier = brier(preds)
    rep.brier_base = mean([(rep.base_rate - float(p.happened)) ** 2 for p in preds])
    rep.log_loss = log_loss(preds)
    rep.bins = bins_of(preds, n_bins)
    rep.ece, rep.mce = calibration_error(rep.bins, len(preds))
    rep.bias, rep.resolution = decomposition(preds, rep.bins)
    rep.brier_ci = bootstrap_ci(
        [(p.predicted - float(p.happened)) ** 2 for p in preds])
    rep.flips, rep.flip_cost = decision_flips(preds)
    return rep


# --- report ----------------------------------------------------------------

def format_report(rep: Report) -> str:
    out: list[str] = []
    out += render.heading("p_fail calibration — scored out of sample", rule="=")
    if rep.n == 0:
        out.append("  Not enough recorded outcomes to score. "
                   "Record escalations with `adder outcomes record`.")
        return "\n".join(out)

    out.append(render.kv("predictions scored", f"{rep.n:,} (prequential)"))
    out.append(render.kv("base rate", render.pct(rep.base_rate)))
    lo, hi = rep.brier_ci
    out.append(render.kv("Brier", f"{rep.brier:.4f}  [{lo:.4f}, {hi:.4f}]"))
    out.append(render.kv("Brier, base rate only", f"{rep.brier_base:.4f}"))
    out.append(render.kv("skill vs base rate", f"{rep.skill:+.3f}"))
    out.append(render.kv("log loss", f"{rep.log_loss:.4f}"))
    out.append(render.kv("calibration error", f"{rep.ece:.3f} mean, {rep.mce:.3f} worst"))
    out.append(render.kv("bias", f"{rep.bias:+.3f}"))
    out.append(render.kv("resolution", f"{rep.resolution:.4f}"))

    out.append("")
    out += render.heading("reliability")
    rows = []
    for b in rep.bins:
        if not b.n:
            continue
        blo, bhi = b.interval
        rows.append([
            f"{b.lo:.1f}-{b.hi:.1f}",
            f"{b.n:,}",
            f"{b.predicted:.3f}",
            f"{b.observed:.3f}",
            f"[{blo:.2f}, {bhi:.2f}]",
            "thin" if b.thin else ("ok" if b.consistent else "OFF"),
        ])
    out += render.table(
        rows, ["bin", "n", "predicted", "observed", "95% CI", ""],
        align="<>>>><",
    )

    out.append("")
    out += render.heading("what it costs")
    out.append(render.kv("decisions flipped", f"{rep.flips:,} of {rep.n:,}"))
    out.append(render.kv("cost of the wrong side",
                         render.money(rep.flip_cost) + "  (floor: excludes the retry)"))

    out.append("")
    if rep.beats_base_rate:
        out += render.wrap(
            "The estimator beats a constant base-rate predictor with the interval "
            "clear of it, so the per-project scoping is earning its complexity.")
    else:
        out += render.wrap(
            f"The estimator does NOT beat always predicting {render.pct(rep.base_rate)}: "
            f"its Brier interval [{lo:.4f}, {hi:.4f}] does not clear "
            f"{rep.brier_base:.4f}. On this much data the scoping is not paying "
            "for itself, and the global rate would serve the gate as well.")

    if not rep.calibrated:
        worst = max((b for b in rep.bins if not b.thin and b.n),
                    key=lambda b: abs(b.gap), default=None)
        if worst is not None:
            direction = "under" if worst.gap > 0 else "over"
            out += render.wrap(
                f"Worst bin {worst.lo:.1f}-{worst.hi:.1f}: predicted "
                f"{worst.predicted:.2f}, observed {worst.observed:.2f} — "
                f"{direction}-confident there by {abs(worst.gap):.2f}. A gate "
                "multiplying by this number inherits that error.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    from adder.decide.track.outcomes import load

    ap = argparse.ArgumentParser(
        prog="adder calib",
        description="Score p_fail out of sample: Brier, skill, reliability, "
                    "and the dollars miscalibration moved.",
    )
    ap.add_argument("--log", type=Path, default=None,
                    help="outcome log to score (default $ADDER_LOG)")
    ap.add_argument("--global-rate", action="store_true",
                    help="score the unscoped estimator instead of the per-project one")
    ap.add_argument("--warmup", type=int, default=5,
                    help="rows consumed before scoring starts (default 5)")
    ap.add_argument("--bins", type=int, default=N_BINS)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    rows = load(args.log)
    preds = prequential(rows, scoped=not args.global_rate, warmup=max(0, args.warmup))
    rep = evaluate(preds, n_bins=max(2, args.bins))

    if args.json:
        print(json.dumps(rep.to_json(), indent=2, sort_keys=True))
    else:
        print(format_report(rep))
    return 0 if rep.n else 1


if __name__ == "__main__":
    raise SystemExit(main())
