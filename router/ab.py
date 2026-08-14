"""Controlled A/B: does routing to a cheaper model cost quality?

Why this exists
---------------
Every saving figure in this repo is cost-only. Transcripts cannot answer the
quality question: they contain only what the expensive model produced, so there
is no counterfactual. An observational attempt (keyword-matching failure language
in subagent replies) turned out to measure text length, not capability -- the
"failure rate" ranked models in exact order of their average reply length.

So quality needs a controlled experiment: identical prompts, identical context,
objective pass criteria, different models.

Scope, stated honestly
----------------------
This tests **comprehension over supplied source**, which is what tier T0 (Haiku,
read-only lookups) actually does. It does NOT test multi-step agentic work, tool
use, or long-horizon coherence. A pass here licenses routing lookups to Haiku.
It licenses nothing about T1 or T2.

Running it costs money and needs credentials, so it is opt-in: `--run`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parent.parent


@dataclass
class Task:
    id: str
    source: str                 # repo-relative file supplied as context
    prompt: str
    check: Callable[[str], bool]
    tier: str = "T0"

    def context(self) -> str:
        p = REPO / self.source
        return p.read_text() if p.exists() else ""


def _has(*needles: str) -> Callable[[str], bool]:
    """Answer must contain all needles (case-insensitive, digit-separator tolerant)."""
    def f(text: str) -> bool:
        t = re.sub(r"[,_]", "", text.lower())
        return all(re.sub(r"[,_]", "", n.lower()) in t for n in needles)
    return f


def _none_of(*needles: str) -> Callable[[str], bool]:
    def f(text: str) -> bool:
        t = text.lower()
        return not any(n.lower() in t for n in needles)
    return f


def _all(*checks: Callable[[str], bool]) -> Callable[[str], bool]:
    return lambda text: all(c(text) for c in checks)


TASKS: list[Task] = [
    Task("breakeven", "router/debt.py",
         "What value does breakeven_remaining_turns return for claude-opus-5? "
         "Answer with the number only.",
         _has("50")),
    Task("cache-read-mult", "router/prices.py",
         "What is the numeric value of CACHE_READ_MULT? Answer with the number only.",
         _has("0.1")),
    Task("write-mult-1h", "router/prices.py",
         "What is the cache write multiplier for a 1h TTL? Answer with the number only.",
         _has("2")),
    Task("haiku-rate", "router/prices.py",
         "What are the input and output rates for claude-haiku-4-5, in dollars "
         "per million tokens?",
         _all(_has("1"), _has("5"))),
    Task("sonnet-intro-expiry", "router/prices.py",
         "On what date does the Claude Sonnet 5 introductory price stop applying? "
         "Answer with the date.",
         _has("2026", "08", "31")),
    Task("default-remaining", "router/horizon.py",
         "What is the numeric value of DEFAULT_REMAINING? Answer with the number only.",
         _has("450")),
    Task("min-samples", "router/horizon.py",
         "What is the numeric value of MIN_SAMPLES? Answer with the number only.",
         _has("5")),
    Task("gate-falsy", "router/cost.py",
         "When switch_is_profitable decides the switch loses money, is the returned "
         "Decision truthy or falsy? Answer 'truthy' or 'falsy'.",
         _all(_has("falsy"), _none_of("truthy"))),
    Task("routing-overhead-tokens", "router/policy.py",
         "How many output tokens does the routing-overhead estimate assume? "
         "Answer with the number only.",
         _has("400")),
    Task("t0-can-write", "router/classify.py",
         "According to the Tier enum, which model does Tier.T0 map to? "
         "Answer with the model id.",
         _has("haiku")),
    Task("t3-effort", "router/classify.py",
         "What effort level does Tier.T3 use? Answer with one word.",
         _has("xhigh")),
    Task("abstain-direction", "router/classify.py",
         "When the classifier abstains because it has no high-precision signal, "
         "does it route UP to a stronger model or DOWN to a cheaper one? "
         "Answer 'up' or 'down'.",
         _all(_has("up"), _none_of("down to", "downward"))),
]


@dataclass
class Outcome:
    task: str
    model: str
    passed: bool
    in_tokens: int = 0
    out_tokens: int = 0
    cost: float = 0.0
    reply: str = ""
    error: str = ""


@dataclass
class ArmResult:
    model: str
    outcomes: list[Outcome] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for o in self.outcomes if o.passed)

    @property
    def n(self) -> int:
        return len(self.outcomes)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.n if self.n else 0.0

    @property
    def cost(self) -> float:
        return sum(o.cost for o in self.outcomes)


def run_arm(model: str, tasks: list[Task], *, max_tokens: int = 200) -> ArmResult:
    """Run every task against one model. Requires the anthropic SDK + credentials."""
    import anthropic

    from .prices import rate

    client = anthropic.Anthropic()
    r = rate(model)
    arm = ArmResult(model)
    for t in tasks:
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=("Answer strictly from the supplied file. Be terse: give the "
                        "answer and nothing else."),
                messages=[{"role": "user",
                           "content": f"<file path=\"{t.source}\">\n{t.context()}\n</file>\n\n{t.prompt}"}],
            )
            text = " ".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            cost = (msg.usage.input_tokens * r.inp + msg.usage.output_tokens * r.out) / 1e6
            arm.outcomes.append(Outcome(t.id, model, t.check(text),
                                        msg.usage.input_tokens, msg.usage.output_tokens,
                                        cost, text.strip()[:120]))
        except Exception as exc:                    # network, auth, rate limit
            arm.outcomes.append(Outcome(t.id, model, False, error=str(exc)[:120]))
    return arm


def wilson_lower_bound(passed: int, n: int, z: float = 1.96) -> float:
    """Lower bound of a 95% CI on the pass rate. Small n must not look conclusive."""
    if n == 0:
        return 0.0
    p = passed / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return max(0.0, (centre - margin) / d)


def report(arms: list[ArmResult]) -> str:
    lines = [f"  {'model':<24}{'passed':>10}{'pass rate':>12}"
             f"{'95% CI low':>12}{'cost':>10}", "  " + "-" * 70]
    for a in arms:
        lines.append(f"  {a.model:<24}{a.passed:>4}/{a.n:<5}{a.pass_rate:>11.0%}"
                     f"{wilson_lower_bound(a.passed, a.n):>12.0%}${a.cost:>9.4f}")
    if len(arms) >= 2:
        cheap, strong = arms[0], arms[-1]
        lines.append("")
        gap = strong.pass_rate - cheap.pass_rate
        if strong.cost > 0:
            lines.append(f"  {cheap.model} costs {100 * cheap.cost / strong.cost:.0f}% "
                         f"of {strong.model}")
        lines.append(f"  pass-rate gap: {gap:+.0%}")
        if gap <= 0:
            lines.append("  => no measured quality loss on this task class")
        else:
            lines.append("  => measured quality loss; do not route this task class down")
        lines.append("")
        lines.append(f"  With n={cheap.n} per arm the confidence interval is wide. Treat")
        lines.append("  this as a smoke test, not proof. Scope: comprehension over supplied")
        lines.append("  source (tier T0). Says nothing about agentic or multi-step work.")
    return "\n".join(lines)


def preflight() -> list[str]:
    """Report exactly what is missing, rather than failing mid-run."""
    import os
    import shutil

    out = []
    try:
        import anthropic  # noqa: F401
        out.append("[ok]      anthropic SDK installed")
    except ImportError:
        out.append("[MISSING] anthropic SDK        -> pip install anthropic")

    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        out.append("[ok]      credentials in environment")
    elif shutil.which("ant"):
        out.append("[maybe]   ant CLI present; run `ant auth status` to confirm a profile")
    else:
        out.append("[MISSING] credentials          -> export ANTHROPIC_API_KEY, "
                   "or install the ant CLI and run `ant auth login`")
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="router.ab",
                                 description="Controlled quality A/B across models.")
    ap.add_argument("--run", action="store_true",
                    help="actually call the API (costs money, needs credentials)")
    ap.add_argument("--models", default="claude-haiku-4-5,claude-opus-5")
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeat the task set N times to tighten the interval")
    a = ap.parse_args(argv)

    models = [m.strip() for m in a.models.split(",") if m.strip()]
    tasks = TASKS * max(1, a.repeats)

    if not a.run:
        print(f"\n  DRY RUN. {len(tasks)} tasks x {len(models)} models "
              f"= {len(tasks) * len(models)} API calls.\n")
        for line in preflight():
            print(f"    {line}")
        print()
        for t in TASKS:
            ctx = len(t.context())
            print(f"    [{t.tier}] {t.id:<24} {t.source:<22} "
                  f"{'context ' + str(ctx) + ' chars' if ctx else 'SOURCE MISSING'}")
        missing = [t.id for t in TASKS if not t.context()]
        print(f"\n  {len(TASKS) - len(missing)}/{len(TASKS)} task sources resolve.")
        if missing:
            print(f"  MISSING: {', '.join(missing)}")
        blocked = [l for l in preflight() if l.startswith("[MISSING]")]
        if blocked:
            print("  Cannot run yet: the prerequisites above are unmet. Nothing else\n"
                  "  in this repo needs them - only this experiment calls the API.\n")
            return 1
        print("\n  Re-run with --run to execute. Roughly a cent per arm.\n")
        return 1 if missing else 0

    arms = [run_arm(m, tasks) for m in models]
    print()
    print(report(arms))
    print()
    fails = [o for a_ in arms for o in a_.outcomes if not o.passed]
    if fails:
        print("  failures:")
        for o in fails[:12]:
            detail = o.error or o.reply
            print(f"    {o.model:<22}{o.task:<24}{detail[:60]}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
