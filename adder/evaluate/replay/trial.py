"""`adder trial`: run real coding tasks with and without adder, and check the work.

Every other multiple in this repo is a replay: recorded turns re-priced under a
counterfactual. A replay can say what a regime would have cost. It cannot say
whether the work would still have been done, because the transcript only holds
what one model did once. `plan` makes that explicit -- its session-model row is
priced with a *modelled* rework rate, and that rate decides whether 10x is
reachable at all.

This runs the experiment instead. Each task in `trial_tasks` is a small
repository with failing tests; each arm is a real headless Claude Code session
configured one way; the grader is a set of hidden tests copied in after the
agent finishes, so a pass cannot be had by editing the tests it was shown. Cost
is the session's own `total_cost_usd`, at list price.

The arms are what adder actually ships, and nothing it does not:

- `baseline`: the model most of the measured history ran on, at its default
  effort, with no hooks, and every part of a multi-part task in one session,
  so context accumulates across them the way a long session's does.
- `adder-<model>`: the session model `plan` recommends, effort medium, the
  hooks `auto on --full` installs, the tier agents, and a fresh session per
  part with a short handoff -- the restart `plan` says is the largest lever.

The terseness and tool-discipline rows of `plan` have no arm: nothing adder
ships implements them, so an arm claiming them would be measuring a wish.

Running spends money, so `--run` is required and every session has a hard
`--max-budget-usd` cap; the run stops before starting a session the remaining
budget cannot cover. Sessions are not persisted, so nothing lands in
`~/.claude/projects` for the other reports to count as your own work, and
results are written only to the `--out` path you name.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from adder.evaluate.replay.trial_tasks import BY_ID, TASKS, Task

BASELINE_MODEL = "claude-opus-5"
# The ceiling per session. A run that hits it is a failure for that arm, which
# is the honest reading: the work did not get done for that money.
SESSION_CAP_USD = 4.0
TIMEOUT_S = 1_200


@dataclass(frozen=True)
class Arm:
    name: str
    model: str
    effort: str | None = None
    adder: bool = False           # hooks and tier agents installed
    restart: bool = False         # a fresh session per part, with a handoff
    # Appended to the system prompt. Only the candidate arms carry one: it is
    # how a lever `plan` models but adder does not ship yet gets measured,
    # quality included, before anything installs it.
    system: str = ""


ARMS: dict[str, Arm] = {
    "baseline": Arm("baseline", BASELINE_MODEL),
    "adder-sonnet": Arm("adder-sonnet", "claude-sonnet-5", "medium", True, True),
    "adder-opus55": Arm("adder-opus55", "claude-opus-5-5", "medium", True, True),
}
# The terseness `plan` prices at 30% of output, written as an instruction. Not
# part of any shipped configuration; `--arms adder-sonnet-terse` measures it.
TERSE = ("Work tersely. Do not narrate what you are about to do or summarise what "
         "you did; do not restate code or file contents; read only the parts of "
         "files you need. Your final message is one or two sentences.")
CANDIDATES: dict[str, Arm] = {
    "adder-sonnet-terse": Arm("adder-sonnet-terse", "claude-sonnet-5", "medium",
                              True, True, TERSE),
}
ALL_ARMS: dict[str, Arm] = {**ARMS, **CANDIDATES}


@dataclass
class Result:
    task: str
    arm: str
    repeat: int
    passed: bool
    cost: float
    sessions: int
    turns: int
    seconds: float
    detail: str = ""


def _preamble() -> str:
    return (f"You are working in a Python repository in the current directory. "
            f"Run the tests with `{sys.executable} -m pytest -q`. Finish with the "
            f"tests passing.\n\n")


def prompts(task: Task, arm: Arm) -> list[str]:
    """What each session of this arm is asked. One prompt per session."""
    if not task.multi:
        return [_preamble() + task.parts[0]]
    if not arm.restart:
        body = "\n".join(f"{i}. {p}" for i, p in enumerate(task.parts, 1))
        return [_preamble() + "Do these tasks in order, finishing each, with its "
                "tests passing, before starting the next:\n\n" + body]
    out = []
    for i, part in enumerate(task.parts):
        done = "".join(f"- {p}\n" for p in task.parts[:i])
        handoff = (f"Earlier sessions in this repository already did:\n{done}\n"
                   if done else "")
        out.append(_preamble() + handoff + "Your task:\n" + part)
    return out


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


def settings_for(arm: Arm, state: Path) -> dict:
    """The settings file this arm's sessions run with."""
    if not arm.adder:
        return {}
    from adder.decide import auto

    blob, _ = auto.merge({})
    return blob


def _env(arm: Arm, state: Path) -> dict[str, str]:
    import os

    env = dict(os.environ)
    if arm.adder:
        env.update({"ADDER_GUARD_ENFORCE": "full",
                    "ADDER_GUARD_STATE": str(state / "guard.json"),
                    "ADDER_TRACE_CACHE": str(state / "trace-cache"),
                    "ADDER_STATE": str(state / "advisor.json"),
                    "ADDER_LEDGER": str(state / "ledger.jsonl"),
                    "ADDER_LOG": str(state / "outcomes.jsonl")})
    return env


def command(arm: Arm, prompt: str, settings: Path, cap: float) -> list[str]:
    py = sys.executable
    cmd = ["claude", "-p", prompt, "--model", arm.model, "--output-format", "json",
           "--no-session-persistence", "--strict-mcp-config",
           "--setting-sources", "project", "--settings", str(settings),
           "--max-budget-usd", f"{cap:.2f}", "--permission-mode", "acceptEdits",
           "--allowedTools", "Read", "Edit", "Write", "Glob", "Grep",
           f"Bash({py}:*)", "Bash(python3:*)", "Bash(ls:*)", "Bash(cat:*)"]
    if arm.effort:
        cmd += ["--effort", arm.effort]
    if arm.system:
        cmd += ["--append-system-prompt", arm.system]
    return cmd


def grade(work: Path, task: Task) -> tuple[bool, str]:
    """Hidden tests in, then the visible and hidden suites together."""
    _write(work, task.hidden)
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "tests", "tests_hidden"], cwd=work, capture_output=True,
                       text=True, timeout=300)
    last = (r.stdout.strip().splitlines() or [r.stderr.strip()[-200:]])[-1]
    return r.returncode == 0, last


def run_one(task: Task, arm: Arm, repeat: int, *, cap: float = SESSION_CAP_USD,
            spend_left: float = float("inf")) -> Result:
    """Run `task` under `arm` in a throwaway directory. Spends money."""
    start = time.monotonic()
    # A fixed directory per task and arm, not a fresh temp name. Claude Code
    # keys a project entry on the working directory even when nothing is
    # persisted, and a random name per run left one empty entry in
    # `~/.claude/projects` for every run; a fixed one leaves one per pair.
    tmp = Path(tempfile.gettempdir()) / "adder-trial" / f"{task.id}-{arm.name}"
    shutil.rmtree(tmp, ignore_errors=True)
    work, state = tmp / "repo", tmp / "state"
    work.mkdir(parents=True)
    state.mkdir()
    _write(work, task.files)
    if arm.adder:
        from adder.decide.auto import AGENTS, agents_dir

        dst = work / ".claude" / "agents"
        dst.mkdir(parents=True)
        for name in AGENTS:
            if (agents_dir() / name).is_file():
                shutil.copy(agents_dir() / name, dst / name)
    settings = state / "settings.json"
    settings.write_text(json.dumps(settings_for(arm, state)), encoding="utf-8")
    cost, turns, sessions, note = 0.0, 0, 0, ""
    for prompt in prompts(task, arm):
        session_cap = min(cap, spend_left - cost)
        if session_cap <= 0.05:
            note = "budget exhausted mid-task"
            break
        try:
            done = subprocess.run(command(arm, prompt, settings, session_cap),
                                  cwd=work, env=_env(arm, state), stdin=subprocess.DEVNULL,
                                  capture_output=True, text=True, timeout=TIMEOUT_S)
            out = json.loads(done.stdout or "{}")
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            note = f"session failed: {type(e).__name__}"
            break
        sessions += 1
        cost += float(out.get("total_cost_usd") or 0.0)
        turns += int(out.get("num_turns") or 0)
        if out.get("is_error"):
            note = f"session error: {str(out.get('subtype') or out.get('result'))[:80]}"
    passed, last = grade(work, task)
    return Result(task.id, arm.name, repeat, passed and not note.startswith("budget"),
                  round(cost, 6), sessions, turns, round(time.monotonic() - start, 1),
                  note or last)


def load(path: Path) -> list[Result]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(Result(**json.loads(line)))
    return out


def report(results: list[Result]) -> str:
    """Pass rate with its interval, and cost against the baseline on the same tasks."""
    from adder.util.render import money
    from adder.util.stats import proportion_diff_ci, wilson_interval

    by: dict[str, list[Result]] = {}
    for r in results:
        by.setdefault(r.arm, []).append(r)
    if not results:
        return "  No results."
    lines = [f"  {'arm':<16}{'runs':>6}{'passed':>9}{'95% CI':>15}{'total':>11}"
             f"{'per run':>10}", "  " + "-" * 67]
    for name, rs in by.items():
        k, n = sum(r.passed for r in rs), len(rs)
        lo, hi = wilson_interval(k, n)
        total = sum(r.cost for r in rs)
        lines.append(f"  {name:<16}{n:>6}{k:>5}/{n:<3}{lo:>7.0%}-{hi:<6.0%}"
                     f"{money(total, width=11)}{money(total / n, width=10)}")
    base = by.get("baseline", [])
    if base:
        lines.append("")
        bk, bn = sum(r.passed for r in base), len(base)
        for name, rs in by.items():
            if name == "baseline":
                continue
            # Cost on the same tasks only, so a missing run cannot move the ratio.
            tasks = {r.task for r in rs} & {r.task for r in base}
            bc = sum(r.cost for r in base if r.task in tasks)
            ac = sum(r.cost for r in rs if r.task in tasks)
            k, n = sum(r.passed for r in rs), len(rs)
            lo, hi = proportion_diff_ci(bk, bn, k, n)
            ratio = f"{bc / ac:.2f}x cheaper" if ac else "n/a"
            lines.append(f"  {name}: {ratio} than baseline on the same tasks; pass-rate "
                         f"change {k / n - bk / bn:+.0%} (95% CI {lo:+.0%} to {hi:+.0%})")
        lines.append("")
        lines.append("  Equivalent output means the pass-rate interval sits at or above")
        lines.append("  zero. With a dozen runs per arm it is wide: a cost multiple here is")
        lines.append("  measured, the equivalence is only as strong as that interval.")
    return "\n".join(lines)


def _transcripts_listing() -> set[str]:
    """Every transcript under `~/.claude/projects`: what a leak would add to."""
    root = Path.home() / ".claude" / "projects"
    return {str(p) for p in root.rglob("*.jsonl")} if root.is_dir() else set()


def main(argv: list[str] | None = None) -> int:
    import argparse

    from adder.measure.argtypes import positive_float, positive_int

    ap = argparse.ArgumentParser(
        prog="adder trial",
        description="Run real coding tasks with and without adder, grade them with "
                    "hidden tests, and compare cost and pass rate.")
    ap.add_argument("--run", action="store_true",
                    help="actually run sessions (costs money); without it, a dry run")
    ap.add_argument("--tasks", default=",".join(t.id for t in TASKS),
                    help="comma-separated task ids (default: all)")
    ap.add_argument("--arms", default=",".join(ARMS), help="comma-separated arms")
    ap.add_argument("--repeats", type=positive_int, default=1)
    ap.add_argument("--budget", type=positive_float, default=25.0,
                    help="stop before a session the remaining budget cannot cover (USD)")
    ap.add_argument("--out", type=Path, help="results JSONL to append to (required with --run)")
    ap.add_argument("--report", type=Path, metavar="JSONL",
                    help="print the report for an existing results file and exit")
    a = ap.parse_args(argv)

    if a.report:
        print(report(load(a.report)))
        return 0
    try:
        tasks = [BY_ID[t] for t in a.tasks.split(",") if t]
        arms = [ALL_ARMS[x] for x in a.arms.split(",") if x]
    except KeyError as e:
        print(f"adder trial: unknown task or arm {e}", file=sys.stderr)
        return 2
    plan_ = [(t, arm, r) for r in range(a.repeats) for t in tasks for arm in arms]
    sessions = sum(len(prompts(t, arm)) for t, arm, _ in plan_)
    print(f"\n  {len(plan_)} runs, {sessions} sessions, capped at "
          f"${SESSION_CAP_USD:.2f} a session and ${a.budget:.2f} in all.\n")
    if not a.run:
        for t, arm, _ in plan_[:len(tasks) * len(arms)]:
            print(f"    {t.id:<12}{arm.name:<16}{arm.model:<18}"
                  f"{len(prompts(t, arm))} session(s)")
        print("\n  DRY RUN. Add --run --out results.jsonl to spend money.\n")
        return 0
    if a.out is None:
        print("adder trial: --run needs --out, the one file this writes", file=sys.stderr)
        return 2
    if not shutil.which("claude"):
        print("adder trial: the claude CLI is not on PATH", file=sys.stderr)
        return 1
    before = _transcripts_listing()
    spent = sum(r.cost for r in load(a.out)) if a.out.is_file() else 0.0
    for t, arm, rep in plan_:
        left = a.budget - spent
        if left < 0.5:
            print(f"  budget reached (${spent:.2f} of ${a.budget:.2f}); stopping")
            break
        res = run_one(t, arm, rep, spend_left=left)
        spent += res.cost
        with a.out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(res)) + "\n")
        print(f"  {t.id:<12}{arm.name:<16}{'PASS' if res.passed else 'FAIL':<6}"
              f"${res.cost:>7.3f}  {res.turns:>3} turns  {res.detail[:60]}")
    leaked = {p for p in _transcripts_listing() - before if "adder-trial" in p}
    if leaked:
        print(f"  WARNING: {len(leaked)} trial transcripts were persisted despite "
              f"--no-session-persistence: {sorted(leaked)[:3]}")
    print()
    print(report(load(a.out)))
    return 0
