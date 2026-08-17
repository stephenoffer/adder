"""Re-fit the effort→output-volume priors from what the model actually wrote.

`cost.EFFORT_OUTPUT_MULT` maps a reasoning-effort level to relative output
volume: low 0.35, medium 0.60, high 1.00, xhigh 1.50, max 2.20. Those are
**priors**, not measurements. They were written down because the effort lever
had to be priced before anyone had data, and they have been quietly scaling
every effort-related saving in this repo ever since.

`cost.py`'s own docstring has claimed since then that "`adder.measure.session.effort` re-fits
them from a transcript when there is enough per-effort history to do so". This
module is that claim, made true. CLAUDE.md's rule is that every claim is
testable or it is not made; a docstring referring to a module that does not
exist is the worst version of that -- it reads as evidence.

What is fitted, and what is refused
-----------------------------------
Transcripts record the effort level per record. Grouping billed output tokens
by level gives an empirical ratio to `high`. Two things stop that being naive:

* **Confounding.** Effort is not assigned at random. A person raises effort for
  hard work, and hard work produces more output for reasons that have nothing
  to do with the setting. The measured ratio is therefore an upper bound on
  effort's causal effect, and it is reported as one. `adder ab` is where a
  causal answer would come from.
* **Thin data.** A level seen fewer than `MIN_TURNS` times is not fitted at
  all; its prior is kept. Fitting a multiplier from nine turns and then
  multiplying a dollar figure by it is how a cost tool invents money.

If the local history has only ever run at one effort level -- which is the
common case, and is the case on the machine this was written on -- the honest
output is "no fit available, priors retained", and that is what it prints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date

from adder.core import settings as _settings
from adder.core.trace import Session
from adder.pricing.cost import EFFORT_OUTPUT_MULT

# Turns needed at a level before its ratio is trusted.
MIN_TURNS = 50

# The level everything is indexed to. Changing this changes the meaning of
# every multiplier in `cost.EFFORT_OUTPUT_MULT`.
BASE_LEVEL = "high"


@dataclass
class Level:
    name: str
    turns: int = 0
    out_tokens: int = 0
    thinking_tokens: int = 0
    cost: float = 0.0
    outs: list[int] = field(default_factory=list)

    @property
    def mean_out(self) -> float:
        return self.out_tokens / self.turns if self.turns else 0.0

    @property
    def median_out(self) -> float:
        from adder.util.stats import median

        return median(self.outs)

    @property
    def mean_thinking(self) -> float:
        return self.thinking_tokens / self.turns if self.turns else 0.0

    @property
    def thinking_share(self) -> float:
        return self.thinking_tokens / self.out_tokens if self.out_tokens else 0.0

    @property
    def enough(self) -> bool:
        return self.turns >= MIN_TURNS


@dataclass
class Fit:
    levels: dict[str, Level] = field(default_factory=dict)
    unlabelled: int = 0

    @property
    def base(self) -> Level | None:
        b = self.levels.get(BASE_LEVEL)
        return b if b and b.enough else None

    @property
    def fittable(self) -> list[str]:
        """Levels with enough data, excluding the base itself."""
        if self.base is None:
            return []
        return [n for n, lv in self.levels.items()
                if lv.enough and n != BASE_LEVEL and lv.mean_out > 0]

    @property
    def measured(self) -> bool:
        return bool(self.fittable)

    def multipliers(self) -> dict[str, float]:
        """Fitted multipliers where the data allows, priors everywhere else.

        Always returns a complete table, so a caller can pass the result
        straight to `cost.effort_saving` without checking which entries moved.
        """
        out = dict(EFFORT_OUTPUT_MULT)
        base = self.base
        if base is None or base.mean_out <= 0:
            return out
        out[BASE_LEVEL] = 1.0
        for name in self.fittable:
            out[name] = self.levels[name].mean_out / base.mean_out
        return out

    def drift(self) -> dict[str, tuple[float, float]]:
        """`{level: (prior, measured)}` for every level that was actually fitted."""
        fitted = self.multipliers()
        return {n: (EFFORT_OUTPUT_MULT[n], fitted[n])
                for n in self.fittable if n in EFFORT_OUTPUT_MULT}


def fit(sessions: dict[str, Session], on: date | None = None) -> Fit:
    """Fit the multipliers from main-chain turns.

    Main chain only, because that is the population the multipliers are applied
    to: `cost.effort_saving` prices the effort lever for the *session* model.
    A subagent runs at its own effort on its own -- usually cheaper, usually
    terser -- model, so including those turns fits an effort ratio partly out
    of a model ratio, and the module's own worry about confounding is exactly
    this kind of thing. Sidechain turns are counted nowhere rather than counted
    wrongly.
    """
    f = Fit()
    for s in sessions.values():
        for t in s.main_turns:
            if not t.effort:
                f.unlabelled += 1
                continue
            lv = f.levels.get(t.effort)
            if lv is None:
                lv = f.levels[t.effort] = Level(t.effort)
            lv.turns += 1
            lv.out_tokens += t.out
            lv.thinking_tokens += t.thinking
            lv.cost += t.cost(on)
            lv.outs.append(t.out)
    return f


def report(f: Fit, *, sessions: dict[str, Session] | None = None,
           model: str | None = None, on: date | None = None) -> str:
    from adder.util.render import money, table, tokens

    model = model or _settings.session_model()
    labelled = sum(lv.turns for lv in f.levels.values())
    lines = ["  Reasoning effort: priors vs what was measured", ""]
    lines.append(f"  {labelled:,} turns carry an effort label; "
                 f"{f.unlabelled:,} do not.")
    if not f.levels:
        lines.append("")
        lines.append("  No transcript here records an effort level, so the priors in")
        lines.append("  cost.EFFORT_OUTPUT_MULT stand unmeasured. They are labelled")
        lines.append("  MODELLED wherever they are used.")
        return "\n".join(lines)

    lines.append("")
    order = [n for n in EFFORT_OUTPUT_MULT if n in f.levels]
    order += [n for n in sorted(f.levels) if n not in EFFORT_OUTPUT_MULT]
    rows = []
    fitted = f.multipliers()
    for name in order:
        lv = f.levels[name]
        prior = EFFORT_OUTPUT_MULT.get(name)
        moved = name in f.fittable or (name == BASE_LEVEL and f.base is not None)
        rows.append([
            name,
            f"{lv.turns:,}",
            tokens(lv.mean_out),
            tokens(lv.median_out),
            f"{lv.thinking_share:.0%}",
            money(lv.cost),
            f"{prior:.2f}" if prior is not None else "—",
            f"{fitted.get(name, 0):.2f}" if moved else "—",
        ])
    lines += table(rows, ["effort", "turns", "mean out", "median", "think",
                          "spend", "prior", "measured"], align="<>>>>>>>")

    lines.append("")
    if not f.measured:
        why = ("only one effort level appears in this history"
               if len(f.levels) < 2 else
               f"no second level reaches {MIN_TURNS} turns")
        lines.append(f"  No re-fit: {why}.")
        lines.append("  The priors are kept, and every figure derived from them stays")
        lines.append("  labelled MODELLED rather than being dressed up as measured.")
    else:
        lines.append("  Fitted multipliers, relative to `high`:")
        for name, (prior, meas) in f.drift().items():
            direction = "above" if meas > prior else "below"
            lines.append(f"    {name:<8}prior {prior:.2f}  measured {meas:.2f}  "
                         f"({abs(meas - prior) / prior:.0%} {direction} the prior)")
        lines.append("")
        lines.append("  Effort is not assigned at random — it is raised for hard work,")
        lines.append("  and hard work is verbose for its own reasons. Treat these as an")
        lines.append("  upper bound on effort's causal effect, and `adder ab` for a")
        lines.append("  controlled answer.")

    if sessions:
        from adder.measure.session.horizon import Horizon
        from adder.pricing.cost import effort_saving

        h = Horizon.from_sessions(sessions)
        out_per_turn = int(sum(lv.mean_out * lv.turns for lv in f.levels.values())
                           / max(1, labelled)) or 1
        remaining = int(h.mean_remaining(0))
        lines.append("")
        lines.append(f"  What one step down is worth per turn on {model}, "
                     f"with {remaining:,} turns of re-reads ahead:")
        rows = []
        table_mult = f.multipliers()
        steps = [("max", "xhigh"), ("xhigh", "high"), ("high", "medium"),
                 ("medium", "low")]
        for hi, lo in steps:
            total, _ = effort_saving(out_per_turn, model, from_effort=hi,
                                     to_effort=lo, remaining_turns=remaining,
                                     mult=table_mult, on=on)
            rows.append([f"{hi} → {lo}", money(total)])
        lines += table(rows, ["step", "$/turn"], align="<>")
        lines.append(f"  At {tokens(out_per_turn)} output per turn, the measured mean.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from adder.core import settings
    from adder.core.filters import add_arguments as add_window
    from adder.core.filters import load as load_window

    ap = argparse.ArgumentParser(
        prog="adder effort",
        description="Re-fit the effort→output priors from local transcripts.")
    add_window(ap)
    ap.add_argument("--model", default=None,
                    help="model to price the step-down table with")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    a = ap.parse_args(argv)

    sessions, _window = load_window(a)
    f = fit(sessions)

    if a.json:
        print(json.dumps({
            "measured": f.measured,
            "unlabelled_turns": f.unlabelled,
            "min_turns": MIN_TURNS,
            "levels": {
                n: {"turns": lv.turns, "mean_out": round(lv.mean_out, 1),
                    "median_out": round(lv.median_out, 1),
                    "thinking_share": round(lv.thinking_share, 4),
                    "cost": round(lv.cost, 4), "enough": lv.enough}
                for n, lv in f.levels.items()
            },
            "priors": EFFORT_OUTPUT_MULT,
            "multipliers": {k: round(v, 4) for k, v in f.multipliers().items()},
        }))
        return 0

    print()
    print(report(f, sessions=sessions, model=a.model or str(settings.get("model"))))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
