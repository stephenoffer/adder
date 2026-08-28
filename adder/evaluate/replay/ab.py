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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from adder.evaluate.replay.seeded import Recall


def _repo_root() -> Path:
    """Walk up to the directory holding `pyproject.toml`.

    This was a fixed `parents[N]`, and N was silently wrong the moment the
    module moved one package deeper: every task's context file resolved to a
    path that did not exist, `context()` returned the empty string, and the
    A/B harness went on to grade answers against no context at all rather than
    failing. A count that has to be edited whenever a file moves is a count
    that will be wrong; a marker that identifies the root cannot be.
    """
    here = Path(__file__).resolve()
    for d in here.parents:
        if (d / "pyproject.toml").is_file():
            return d
    return here.parents[-1]


REPO = _repo_root()


@dataclass
class Task:
    id: str
    source: str                 # repo-relative file supplied as context
    prompt: str
    check: Callable[[str], bool]
    tier: str = "T0"

    def context(self) -> str:
        p = REPO / self.source
        return p.read_text(encoding="utf-8") if p.exists() else ""


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
    Task("breakeven", "adder/measure/spend/debt.py",
         "What value does breakeven_remaining_turns return for claude-opus-5? "
         "Answer with the number only.",
         _has("50")),
    Task("cache-read-mult", "adder/pricing/prices.py",
         "What is the numeric value of CACHE_READ_MULT? Answer with the number only.",
         _has("0.1")),
    Task("write-mult-1h", "adder/pricing/prices.py",
         "What is the cache write multiplier for a 1h TTL? Answer with the number only.",
         _has("2")),
    Task("haiku-rate", "adder/pricing/prices.py",
         "What are the input and output rates for claude-haiku-4-5, in dollars "
         "per million tokens?",
         _all(_has("1"), _has("5"))),
    Task("sonnet-intro-expiry", "adder/pricing/prices.py",
         "On what date does the Claude Sonnet 5 introductory price stop applying? "
         "Answer with the date.",
         _has("2026", "08", "31")),
    Task("default-remaining", "adder/measure/session/horizon.py",
         "What is the numeric value of DEFAULT_REMAINING? Answer with the number only.",
         _has("450")),
    Task("min-samples", "adder/measure/session/horizon.py",
         "What is the numeric value of MIN_SAMPLES? Answer with the number only.",
         _has("5")),
    Task("gate-falsy", "adder/pricing/cost.py",
         "When switch_is_profitable decides the switch loses money, is the returned "
         "Decision truthy or falsy? Answer 'truthy' or 'falsy'.",
         _all(_has("falsy"), _none_of("truthy"))),
    Task("routing-overhead-tokens", "adder/decide/route/policy.py",
         "How many output tokens does the routing-overhead estimate assume? "
         "Answer with the number only.",
         _has("400")),
    Task("t0-can-write", "adder/decide/route/classify.py",
         "According to the Tier enum, which model does Tier.T0 map to? "
         "Answer with the model id.",
         _has("haiku")),
    Task("t3-effort", "adder/decide/route/classify.py",
         "What effort level does Tier.T3 use? Answer with one word.",
         _has("xhigh")),
    Task("abstain-direction", "adder/decide/route/classify.py",
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

    from adder.pricing.prices import rate

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


def run_arm_cli(model: str, tasks: list[Task], *, timeout: int = 180) -> ArmResult:
    """Run tasks through the local `claude` CLI in headless mode.

    Uses the credentials Claude Code already holds, so no separate API key is
    needed. Note the CLI loads its own system prompt on every call (~25K tokens
    of cached context), so the reported cost is Claude-Code-harness cost, not raw
    API cost. Both arms pay it equally, so the QUALITY comparison is unaffected;
    for cost modelling use adder.pricing.cost instead of these figures.
    """
    import json
    import subprocess

    arm = ArmResult(model)
    for t in tasks:
        prompt = (f'<file path="{t.source}">\n{t.context()}\n</file>\n\n'
                  f"{t.prompt}\n\n"
                  "Answer strictly from the file above. Do not use any tools. "
                  "Reply with the answer only.")
        try:
            proc = subprocess.run(
                ["claude", "-p", prompt, "--model", model,
                 "--output-format", "json", "--max-turns", "1"],
                capture_output=True, text=True, timeout=timeout, cwd="/tmp",
            )
            if proc.returncode != 0:
                arm.outcomes.append(Outcome(t.id, model, False,
                                            error=(proc.stderr or "nonzero exit")[:120]))
                continue
            data = json.loads(proc.stdout)
            text = data.get("result") or ""
            usage = data.get("usage") or {}
            arm.outcomes.append(Outcome(
                t.id, model, t.check(text),
                in_tokens=usage.get("input_tokens", 0) or 0,
                out_tokens=usage.get("output_tokens", 0) or 0,
                cost=float(data.get("total_cost_usd") or 0.0),
                reply=text.strip().replace("\n", " ")[:120],
            ))
        except subprocess.TimeoutExpired:
            arm.outcomes.append(Outcome(t.id, model, False, error="timeout"))
        except (json.JSONDecodeError, ValueError) as exc:
            arm.outcomes.append(Outcome(t.id, model, False, error=f"bad json: {exc}"[:120]))
    return arm


def run_recall_cli(model: str, *, timeout: int = 300) -> Recall:
    """Ask one model to find the planted defects, and score what came back.

    Through the same CLI backend as `run_arm_cli`, for the same reason: it uses
    the credentials Claude Code already holds. What is different is what comes
    out of it -- a set of ids matched against a known list, computed by
    `seeded.score`, which touches nothing in the cost model. That is the whole
    point of the mode. Every other quality signal in this repo is derived from
    the same parse and the same prices as the saving it is checking.
    """
    import json
    import subprocess

    from adder.evaluate.replay import seeded

    prompt = (f'<file path="cache_helpers.py">\n{seeded.SOURCE}\n</file>\n\n'
              f"{seeded.PROMPT}\n\nAnswer strictly from the file above. "
              "Do not use any tools.")
    try:
        proc = subprocess.run(
            ["claude", "-p", prompt, "--model", model,
             "--output-format", "json", "--max-turns", "1"],
            capture_output=True, text=True, timeout=timeout, cwd="/tmp",
        )
        if proc.returncode != 0:
            return seeded.Recall(model, error=(proc.stderr or "nonzero exit")[:160])
        data = json.loads(proc.stdout)
        text = data.get("result") or ""
        return seeded.Recall(model, seeded.score(text), reply=text,
                             cost=float(data.get("total_cost_usd") or 0.0))
    except subprocess.TimeoutExpired:
        return seeded.Recall(model, error="timeout")
    except (json.JSONDecodeError, ValueError) as exc:
        return seeded.Recall(model, error=f"bad json: {exc}"[:160])


def run_recall_sdk(model: str, *, max_tokens: int = 2_000) -> Recall:
    """The same, through the anthropic SDK. Needs credentials of its own."""
    import anthropic

    from adder.evaluate.replay import seeded
    from adder.pricing.prices import rate

    try:
        client = anthropic.Anthropic()
        r = rate(model)
        msg = client.messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user",
                       "content": f'<file path="cache_helpers.py">\n'
                                  f'{seeded.SOURCE}\n</file>\n\n{seeded.PROMPT}'}],
        )
        text = " ".join(b.text for b in msg.content
                        if getattr(b, "type", "") == "text")
        cost = (msg.usage.input_tokens * r.inp + msg.usage.output_tokens * r.out) / 1e6
        return seeded.Recall(model, seeded.score(text), reply=text, cost=cost)
    except Exception as exc:                        # network, auth, rate limit
        return seeded.Recall(model, error=str(exc)[:160])


def wilson_lower_bound(passed: int, n: int, *, alpha: float = 0.05) -> float:
    """Lower bound of a 95% CI on the pass rate. Small n must not look conclusive.

    Delegates to `stats.wilson_interval` rather than carrying a private copy of
    the algebra. That copy was the eighth transcription of a shared estimator
    in this repo -- the thing `util.stats` opens by saying it exists to stop --
    and it had already drifted: it did not clamp an out-of-range count (so a
    `passed` above `n` raised a math domain error), and it did not snap an
    exact endpoint, so a clean run reported a lower bound of 2.8e-17 where
    every other report in the tool prints 0.
    """
    from adder.util.stats import wilson_interval

    return wilson_interval(passed, n, alpha=alpha)[0]


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
    if shutil.which("claude"):
        out.append("[ok]      claude CLI backend (uses existing Claude Code credentials)")
        return out
    try:
        import anthropic  # noqa: F401
        out.append("[ok]      anthropic SDK installed")
    except ImportError:
        out.append("[MISSING] backend              -> install the claude CLI, "
                   "or pip install anthropic")
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        out.append("[ok]      credentials in environment")
    else:
        out.append("[MISSING] credentials          -> export ANTHROPIC_API_KEY")
    return out


def _recall(a, models: list[str]) -> int:
    """The `--recall` arm of `main`. Same dry-run contract as everything else here."""
    from adder.evaluate.replay import seeded

    if not a.run:
        print(f"\n  DRY RUN. {seeded.K} planted defects x {len(models)} models "
              f"= {len(models)} API calls.\n")
        for line in preflight():
            print(f"    {line}")
        print()
        for s in seeded.SEEDS:
            print(f"    {s.symbol:<18}{s.what}")
        print("\n  `--print-seed` prints the fixture and the prompt. Re-run with "
              "--run to score\n  a model against it.\n")
        return 1 if any(ln.startswith("[MISSING]") for ln in preflight()) else 0

    runner = run_recall_cli if a.backend == "cli" else run_recall_sdk
    results = [runner(m) for m in models]
    print()
    print(seeded.report(results))
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="adder ab",
                                 description="Controlled quality A/B across models.")
    ap.add_argument("--run", action="store_true",
                    help="actually call the API (costs money, needs credentials)")
    ap.add_argument("--models", default="claude-haiku-4-5,claude-opus-5")
    ap.add_argument("--backend", choices=("cli", "sdk"), default="cli",
                    help="cli uses the local claude binary and existing credentials")
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeat the task set N times to tighten the interval")
    ap.add_argument("--recall", action="store_true",
                    help="score each model on a fixture with a known number of "
                         "planted defects, instead of the comprehension tasks. "
                         "Shares no code with the cost model")
    ap.add_argument("--print-seed", action="store_true",
                    help="print the seeded fixture and the prompt, so the same "
                         "measurement can be run by hand or by another tool")
    a = ap.parse_args(argv)

    if a.print_seed:
        from adder.evaluate.replay import seeded

        print(seeded.SOURCE)
        print(seeded.PROMPT)
        print()
        print(f"  {seeded.K} defects are planted. `adder ab --recall --run` "
              "scores a reply against them.")
        return 0

    if a.recall:
        return _recall(a, models=[m.strip() for m in a.models.split(",") if m.strip()])

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
        blocked = [ln for ln in preflight() if ln.startswith("[MISSING]")]
        if blocked:
            print("  Cannot run yet: the prerequisites above are unmet. Nothing else\n"
                  "  in this repo needs them - only this experiment calls the API.\n")
            return 1
        print("\n  Re-run with --run to execute. Via the claude CLI each call also\n"
              "  loads Claude Code's own system prompt (~25K cached tokens), so budget\n"
              "  roughly $0.02 per call -- about $0.50 for the full matrix.\n")
        return 1 if missing else 0

    runner = run_arm_cli if a.backend == 'cli' else run_arm
    arms = [runner(m, tasks) for m in models]
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
