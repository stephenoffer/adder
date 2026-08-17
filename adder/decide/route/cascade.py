"""Cascades: try the cheap model, check the answer, escalate only if it is bad.

The case for a cascade, and the case against
--------------------------------------------
A router decides from the prompt. That is a hard prediction problem: the
question "will the cheap model get this right" is close to "what is the answer",
and a classifier reading forty words of task description is guessing.

A cascade sidesteps the prediction. Run the cheap model, look at what came
back, and escalate if it is wrong. The signal is enormously better -- an answer
is far easier to judge than a prompt -- and it needs no trained router at all.

The cost of that is that you sometimes pay twice, plus whatever the check
costs. So a cascade wins on a narrow set of conditions and loses badly outside
it, and the entire value of this module is drawing that boundary rather than
asserting one side of it.

The term the batch-inference version of this analysis leaves out
----------------------------------------------------------------
Published cascade economics come from batch settings: you run the small model,
score the output, discard it if it fails, and run the big one. The failed
attempt costs what it cost, and then it is gone.

**In an agent session it is not gone.** A failed attempt that ran inline leaves
its tokens in the context, and every remaining turn re-reads them at the cached
input rate. On a session with 200 turns left, a 4,000-token dead end is not a
4,000-token event -- it is 4,000 tokens re-read 200 times. That term is
routinely larger than the failed attempt's own generation cost, and leaving it
out is why cascades look better on paper than they behave in a session.

This module prices it, and it is the reason the recommendation here often comes
out as "cascade, but in a subagent" rather than "cascade": running the cheap
attempt in a throwaway context means only its summary is carried, which
collapses the term that was killing it.

The verifier is not free and not perfect
----------------------------------------
Two error rates, and they cost different things:

* **False negative** (the check passes a bad answer). The failure ships. This
  is a quality cost, not a dollar cost, and it is the one that makes a cheap
  cascade look cheap.
* **False positive** (the check rejects a good answer). You pay for the strong
  model for nothing. A pure dollar cost.

A verifier with a 30% false-negative rate does not turn a 20%-failure tier into
a reliable one; it turns it into a 6%-failure tier that also costs more. Both
rates are parameters here because both are measurable and neither is usually
measured.

Everything below is closed-form arithmetic over prices that come from the same
date-aware table as the rest of the tool. The probabilities are the modelled
part, and `p_fail` should come from the outcome log (`adder calib` scores
whether it can be trusted) rather than from a guess.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field

from adder.core import settings as _settings
from adder.pricing.cost import Rates, fits, run_cost
from adder.util import render

M = 1_000_000.0

# What a verification step costs, as a share of the attempt it checks. A check
# that re-reads the answer and says pass/fail is much cheaper than producing the
# answer; a check that re-derives the answer is not a check, it is a second
# attempt. 0.15 is the shape of the former.
DEFAULT_VERIFY_SHARE = 0.15


@dataclass(frozen=True)
class Setup:
    """Everything the arithmetic needs, with the modelled parts named."""

    weak_model: str
    strong_model: str
    ctx_tokens: int = 100_000
    est_out_tokens: int = 1_200
    # Probability the weak model's answer is unusable. MEASURED, from the
    # outcome log, if you have one.
    p_fail: float = 0.2
    # Verifier error rates. MODELLED unless you have run the check against
    # labelled outcomes.
    false_negative: float = 0.0     # bad answer passes the check
    false_positive: float = 0.0     # good answer is rejected
    verify_share: float = DEFAULT_VERIFY_SHARE
    # Session shape, which is what makes the carry term real.
    remaining_turns: int = 100
    session_model: str = field(default_factory=_settings.session_model)
    # Tokens the failed attempt leaves in the main context. Inline, that is the
    # whole attempt; delegated, only its summary comes back.
    inline: bool = True
    summary_tokens: int = 400
    # What it costs the main session to notice the failure and dispatch again.
    retry_overhead: float = 0.0

    def __post_init__(self) -> None:
        for name in ("p_fail", "false_negative", "false_positive"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0,1], got {v}")
        if self.verify_share < 0.0:
            raise ValueError(f"verify_share must be >= 0, got {self.verify_share}")

    @property
    def p_escalate(self) -> float:
        """How often the check sends the task on to the strong model.

        Both error paths land here: genuine failures the check caught, plus
        good answers it rejected.
        """
        return (self.p_fail * (1.0 - self.false_negative)
                + (1.0 - self.p_fail) * self.false_positive)

    @property
    def p_ships_broken(self) -> float:
        """How often a bad answer survives the check and ships.

        The quality cost of the cascade, and the number a dollar comparison
        will happily ignore.
        """
        return self.p_fail * self.false_negative


@dataclass(frozen=True)
class Strategy:
    """One way of doing the task, priced, with the quality it delivers."""

    name: str
    cost: float
    p_broken: float
    detail: str = ""

    @property
    def usable(self) -> float:
        return 1.0 - self.p_broken


def carry_cost(tokens: int, setup: Setup) -> float:
    """What leaving `tokens` in the main context costs over the rest of the run.

    The term that decides most cascade questions in an agent session, and the
    one no batch-derived analysis contains. Priced at the session model's cached
    input rate, because that is what re-reading a prefix costs.

    That rate comes from the provider, not from Anthropic's 0.10x multiplier
    applied to whatever model was named. `session_model` is free text here --
    `fits` and `run_cost` both resolve any model in the catalog -- and on a
    provider with no prompt cache a re-read costs the full input rate, so
    assuming the discount understates this term tenfold. It is the term that
    decides the question.
    """
    if tokens <= 0 or setup.remaining_turns <= 0:
        return 0.0
    read = Rates.for_model(setup.session_model).cache_read
    return tokens * setup.remaining_turns * read / M


def strategies(setup: Setup) -> list[Strategy]:
    """Price every strategy on the same task, so they can be compared at all."""
    weak = run_cost(setup.weak_model, setup.ctx_tokens, setup.est_out_tokens)
    strong = run_cost(setup.strong_model, setup.ctx_tokens, setup.est_out_tokens)
    verify = weak * setup.verify_share

    out = [
        Strategy("always strong", strong, 0.0,
                 f"one run of {setup.strong_model}"),
        Strategy("always weak", weak, setup.p_fail,
                 f"one run of {setup.weak_model}, no check"),
    ]

    if not fits(setup.weak_model, setup.ctx_tokens):
        # Feasibility gates profitability. A model that cannot hold the context
        # is not a cheap option, it is not an option.
        out.append(Strategy(
            "cascade", float("inf"), 0.0,
            f"{setup.weak_model} cannot hold {setup.ctx_tokens:,} tokens"))
        return out

    # The dead end is only carried if it happened inline. In a subagent the
    # context is thrown away and only the summary comes back.
    dead_end_tokens = (setup.ctx_tokens + setup.est_out_tokens
                       if setup.inline else setup.summary_tokens)
    dead_end = carry_cost(dead_end_tokens, setup) * setup.p_escalate

    cascade = (weak + verify
               + setup.p_escalate * (strong + setup.retry_overhead)
               + dead_end)
    where = "inline" if setup.inline else "in a subagent"
    out.append(Strategy(
        "cascade", cascade, setup.p_ships_broken,
        f"{setup.weak_model} {where}, checked, escalating "
        f"{setup.p_escalate:.0%} of the time"))

    if setup.inline:
        # The same cascade, delegated. Almost always the interesting row,
        # because it is the one that deletes the carry term.
        from dataclasses import replace

        delegated = replace(setup, inline=False)
        d_weak = run_cost(delegated.weak_model, delegated.ctx_tokens,
                          delegated.est_out_tokens)
        d_carry = carry_cost(delegated.summary_tokens, delegated) * delegated.p_escalate
        out.append(Strategy(
            "cascade (delegated)",
            d_weak + d_weak * delegated.verify_share
            + delegated.p_escalate * (strong + delegated.retry_overhead) + d_carry,
            delegated.p_ships_broken,
            f"{setup.weak_model} in a subagent; only a "
            f"{setup.summary_tokens:,}-token summary is carried"))
    return out


def best(setup: Setup) -> Strategy:
    """Cheapest strategy that does not ship more broken answers than the strong one.

    Not simply `min(cost)`. "Always weak" is the cheapest row on almost every
    task and it is only the right answer if you are willing to accept its
    failure rate; a comparison that ranks on dollars alone recommends it every
    time. Strategies are therefore only compared against each other at equal or
    better usability than the safe baseline, and "always weak" qualifies only
    when its own failure rate is negligible.
    """
    rows = strategies(setup)
    safe = next(s for s in rows if s.name == "always strong")
    eligible = [s for s in rows if s.p_broken <= 0.01 and s.cost < float("inf")]
    return min(eligible or [safe], key=lambda s: s.cost)


def breakeven_p_fail(setup: Setup, *, tol: float = 1e-4) -> float:
    """The failure rate above which the cascade stops being worth it.

    Solved by bisection rather than algebraically because the carry term makes
    the expression a mess and the bisection is six lines that cannot be got
    subtly wrong. Monotone in `p_fail`, which is what makes bisection valid:
    every term that depends on it only increases the cascade's cost.

    Returns 0.0 when the cascade never wins, and 1.0 when it always does.
    """
    from dataclasses import replace

    def cascade_saving(p: float) -> float:
        rows = strategies(replace(setup, p_fail=p))
        strong = next(s for s in rows if s.name == "always strong")
        casc = min((s for s in rows if s.name.startswith("cascade")),
                   key=lambda s: s.cost)
        return strong.cost - casc.cost

    if cascade_saving(0.0) <= 0:
        return 0.0
    if cascade_saving(1.0) > 0:
        return 1.0
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if cascade_saving(mid) > 0:
            lo = mid
        else:
            hi = mid
    return lo


def max_false_negative(setup: Setup, *, budget: float = 0.05) -> float:
    """How blind the check may be before the cascade ships too many failures.

    `budget` is the share of tasks you are willing to see fail silently. The
    answer is `budget / p_fail`, capped at 1: a check on a tier that fails 20%
    of the time may miss at most a quarter of those failures to stay inside a
    5% budget.
    """
    if setup.p_fail <= 0:
        return 1.0
    return min(1.0, budget / setup.p_fail)


def report(setup: Setup) -> str:
    rows = strategies(setup)
    pick = best(setup)
    out: list[str] = []
    out += render.heading("cascade — try cheap, check, escalate", rule="=")
    out.append(render.kv("task", f"{setup.ctx_tokens:,} tok context, "
                                 f"{setup.est_out_tokens:,} tok answer"))
    out.append(render.kv("weak / strong", f"{setup.weak_model} / {setup.strong_model}"))
    out.append(render.kv("p_fail (weak)", f"{setup.p_fail:.0%}"))
    out.append(render.kv("check misses / fires",
                         f"{setup.false_negative:.0%} / {setup.false_positive:.0%}"))
    out.append(render.kv("session", f"{setup.remaining_turns} turns remaining"))
    out.append("")

    out += render.table(
        [[s.name,
          "—" if s.cost == float("inf") else render.money(s.cost),
          f"{s.p_broken:.1%}",
          s.detail[:52]] for s in rows],
        ["strategy", "cost", "ships broken", "what it is"],
        align="<>><",
    )

    out.append("")
    out += render.heading("where the boundary is")
    be = breakeven_p_fail(setup)
    out.append(render.kv("cascade wins below", f"p_fail = {be:.0%}"))
    out.append(render.kv("measured p_fail", f"{setup.p_fail:.0%}"))
    out.append(render.kv("check may miss", f"{max_false_negative(setup):.0%} of failures"))

    inline_row = next((s for s in rows if s.name == "cascade"), None)
    deleg_row = next((s for s in rows if s.name == "cascade (delegated)"), None)
    if inline_row and deleg_row and inline_row.cost < float("inf"):
        gap = inline_row.cost - deleg_row.cost
        if gap > 0:
            out.append("")
            out += render.wrap(
                f"Running the cheap attempt inline costs {render.money(gap)} more "
                f"than running it in a subagent, and none of that is generation — "
                "it is the failed attempt sitting in the context being re-read for "
                f"the remaining {setup.remaining_turns} turns. If you cascade, "
                "cascade in a subagent.")

    out.append("")
    out += render.wrap(f"Cheapest strategy that is no less reliable than always "
                       f"using {setup.strong_model}: **{pick.name}**.")
    out += render.wrap(
        "MODELLED: the verifier error rates and the failure rate are inputs, not "
        "measurements. `adder calib` scores whether this machine's p_fail can be "
        "trusted; the verifier rates need a labelled sample and most people do "
        "not have one.")
    return "\n".join(out)


def to_json(setup: Setup) -> dict:
    return {
        "setup": {
            "weak": setup.weak_model, "strong": setup.strong_model,
            "ctx_tokens": setup.ctx_tokens, "p_fail": setup.p_fail,
            "false_negative": setup.false_negative,
            "false_positive": setup.false_positive,
            "remaining_turns": setup.remaining_turns,
            "p_escalate": setup.p_escalate,
            "p_ships_broken": setup.p_ships_broken,
        },
        "strategies": [
            {"name": s.name,
             "cost_usd": None if s.cost == float("inf") else s.cost,
             "ships_broken": s.p_broken, "detail": s.detail}
            for s in strategies(setup)
        ],
        "best": best(setup).name,
        "breakeven_p_fail": breakeven_p_fail(setup),
        "max_false_negative": max_false_negative(setup),
        "modelled": True,
    }


def main(argv: list[str] | None = None) -> int:
    from adder.decide.route.classify import Tier

    ap = argparse.ArgumentParser(
        prog="adder cascade",
        description="Is it cheaper to try the small model and check, or to go "
                    "straight to the big one?",
    )
    ap.add_argument("--weak", default=Tier.T0.model, help="the cheap model to try first")
    ap.add_argument("--strong", default=Tier.T2.model, help="the escalation target")
    ap.add_argument("--context", type=int, default=100_000)
    ap.add_argument("--out", type=int, default=1_200, help="expected answer tokens")
    ap.add_argument("--p-fail", type=float, default=None,
                    help="failure rate of the weak model (default: from the outcome log)")
    ap.add_argument("--miss", type=float, default=0.0,
                    help="verifier false-negative rate (bad answers it passes)")
    ap.add_argument("--over-fire", type=float, default=0.0,
                    help="verifier false-positive rate (good answers it rejects)")
    ap.add_argument("--turns", type=int, default=100, help="turns remaining in the session")
    ap.add_argument("--delegated", action="store_true",
                    help="price the cheap attempt as a subagent run")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    p_fail = args.p_fail
    if p_fail is None:
        from adder.decide.track.outcomes import p_fail as measured

        p_fail = measured("T0")

    try:
        setup = Setup(
            weak_model=args.weak,
            strong_model=args.strong,
            ctx_tokens=max(0, args.context),
            est_out_tokens=max(0, args.out),
            p_fail=float(p_fail),
            false_negative=args.miss,
            false_positive=args.over_fire,
            remaining_turns=max(0, args.turns),
            inline=not args.delegated,
        )
    except ValueError as exc:
        print(f"adder cascade: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(to_json(setup), indent=2, sort_keys=True))
    else:
        print(report(setup))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
