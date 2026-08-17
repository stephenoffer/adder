"""Outcome log and escalation calibration - the adaptive half of adder.

The escalation gate needs `p_fail`: how often a tier fails and forces a retry.
Guessing it makes the gate arbitrary, so it is measured here instead.

An append-only JSONL log is used rather than subagent `memory:` because it is
inspectable, testable, and diffable; a router that silently changes its own
behaviour from an opaque store is hard to trust or debug.

Estimates are smoothed with a Beta(1,1) prior so a single early failure does not
swing the gate, scoped per (project, tier) because task mix differs sharply
between codebases, and **recency-weighted** so a tier that used to fail but has
since improved -- or a codebase that has grown harder -- is tracked rather than
averaged over all history.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path

# The built-in location. `log_path()` is what callers should use: it lets an
# explicitly-configured `log` setting win, which this constant alone cannot,
# because it is read once at import.
DEFAULT_LOG = Path(
    os.environ.get("ADDER_LOG", Path.home() / ".claude" / "adder-outcomes.jsonl")
)


def log_path(log: Path | str | None = None) -> Path:
    """The outcome log in effect: the caller's, the `log` setting, or the default."""
    if log is not None:
        return Path(log)
    from adder.core.settings import configured_path

    return configured_path("log", DEFAULT_LOG)

# Beta(1,1) prior: with no evidence, p_fail = 0.5 (maximally cautious).
PRIOR_FAIL = 1.0
PRIOR_OK = 1.0

# Below this many scoped observations, fall back to the global rate for the tier.
MIN_SCOPED = 5

# Recency-weighted observation mass required before the log is allowed to
# *contradict* the classifier rather than merely inform a gate it already
# agreed with. Escalating on thin evidence is cheap -- you end up on the model
# you would have used anyway. De-escalating on thin evidence is not: it routes
# real work to a model the text classifier did not think could do it. So the
# two directions get different burdens of proof, and this is the harder one.
MIN_EVIDENCE = 12.0

# Older outcomes count for less. An outcome this many days old counts half.
HALF_LIFE_DAYS = 30.0

# Keep the log bounded; it is telemetry, not an audit trail.
MAX_ROWS = 20_000


@dataclass
class Outcome:
    tier: str
    model: str
    project: str
    escalated: bool
    context_tokens: int = 0
    remaining_turns: int = 0
    cost: float = 0.0
    task_hash: str = ""
    reason: str = ""
    effort: str = ""
    duration_s: float = 0.0
    ts: float = field(default_factory=time.time)
    # Where the row came from: "recorded" by hand, or "transcript" by
    # `adder outcomes import`. Provenance only -- both are weighed the same,
    # because both are blind to the same thing. A subagent that returned a
    # confident wrong answer and was believed looks like a success in the
    # transcript AND to the person filing the report afterwards.
    source: str = "recorded"


_FIELDS = {f.name for f in fields(Outcome)}


def _coerce_ts(v: object) -> float | None:
    """Epoch seconds from whatever a log line actually carries.

    `Outcome.ts` is epoch seconds, but every other timestamp in this repo is an
    ISO string (`Turn.ts`, the transcript files themselves), so a caller
    writing one here is a matter of time. Left alone the row loads fine and
    then raises inside the recency weighting -- where both callers swallow the
    exception, so the symptom is not an error but the outcome log silently
    ceasing to influence any routing decision. Coerce what is recoverable,
    drop what is not.
    """
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def record(outcome: Outcome, log: Path | str | None = None) -> None:
    """Append one outcome. Never raises: telemetry must not break routing.

    A single `write` of one short line to an O_APPEND file is atomic on POSIX,
    so concurrent sessions interleave lines rather than corrupting them.
    """
    try:
        p = log_path(log)
        p.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(outcome), separators=(",", ":")) + "\n"
        with p.open("a", encoding="utf-8") as fh:
            fh.write(line)
    except (OSError, TypeError, ValueError):
        pass


def load(log: Path | str | None = None) -> list[Outcome]:
    """Read the log, skipping anything malformed and tolerating new fields."""
    p = log_path(log)
    if not p.exists():
        return []
    out = []
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            if not isinstance(d, dict):
                continue
            # Ignore fields written by a newer version rather than crashing.
            kw = {k: v for k, v in d.items() if k in _FIELDS}
            if "ts" in kw:
                ts = _coerce_ts(kw["ts"])
                if ts is None:
                    continue
                kw["ts"] = ts
            out.append(Outcome(**kw))
        except (ValueError, TypeError):
            continue
    return out


def prune(log: Path | str | None = None, keep: int = MAX_ROWS) -> int:
    """Trim the log to its most recent `keep` rows. Returns rows dropped."""
    rows = load(log)
    if len(rows) <= keep:
        return 0
    rows.sort(key=lambda o: o.ts)
    kept = rows[-keep:]
    try:
        p = log_path(log)
        # Unique per writer. Several Claude Code sessions share one machine
        # and run this from a hook, so a fixed `.tmp` name is a shared
        # mutable path: one writer's `replace` moves the file out from
        # under another's, and the loser raises FileNotFoundError into an
        # `except OSError` that drops it. Measured at 45% of writes lost
        # under three concurrent writers. `trace._cache_store` already
        # carries the pid for exactly this reason.
        tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                for o in kept:
                    fh.write(json.dumps(asdict(o), separators=(",", ":")) + "\n")
            tmp.replace(p)
        finally:
            tmp.unlink(missing_ok=True)
    except OSError:
        return 0
    return len(rows) - len(kept)


def _weight(o: Outcome, now: float) -> float:
    """Exponential recency weight. Recent evidence should move the gate more."""
    age_days = max(0.0, (now - o.ts) / 86400.0)
    return 0.5 ** (age_days / HALF_LIFE_DAYS)


@dataclass(frozen=True)
class Evidence:
    """An escalation rate together with how much data stands behind it.

    A bare float cannot distinguish "0.5 measured over 200 runs" from "0.5
    because nothing has ever been recorded". Those are the same number and
    opposite instructions: the first is a calibrated coin, the second is an
    admission of ignorance that happens to be encoded as one. Callers that are
    about to override a classifier need to know which one they have, so the
    scope travels with the estimate.
    """

    p_fail: float
    n: int                # raw observations in scope
    weight: float         # recency-weighted observation mass
    scope: str            # "project" | "global" | "prior"
    fails: float = 0.0    # recency-weighted mass that escalated

    @property
    def informative(self) -> bool:
        """Enough real, recent history to act against a prior on."""
        return self.scope != "prior" and self.weight >= MIN_EVIDENCE

    def bounds(self, alpha: float = 0.10):
        """Credible interval for the rate, not just its mean.

        The mean cannot tell 20% over four runs from 20% over four hundred, and
        those two carry opposite instructions for a gate that is about to bet a
        redo on the number. The posterior is the same Beta the mean already
        comes from -- Beta(1,1) prior, recency-weighted mass as the likelihood --
        so this adds no assumption, only the width that was being discarded.
        """
        from adder.util.risk import beta_bounds

        return beta_bounds(self.fails, self.weight, alpha=alpha,
                           prior_fail=PRIOR_FAIL, prior_ok=PRIOR_OK)

    def upper(self, alpha: float = 0.10) -> float:
        """The rate a gate should charge the failure branch at.

        Pessimism here is not conservatism for its own sake. The two errors are
        priced differently: over-estimating `p_fail` sends work to a model that
        would have finished it anyway, and the loss is the rate difference.
        Under-estimating it routes work to a tier that fails, and the loss is
        the failed run plus the redo plus the turn that noticed -- several times
        larger. When the losses are asymmetric the estimate should be too.
        """
        return self.bounds(alpha).hi

    def quantiles(self, strata: int = 8) -> list[float]:
        """Equally-weighted ladder from the posterior, for `risk.p_cheaper`."""
        from adder.util.risk import beta_quantiles

        return beta_quantiles(self.fails, self.weight, strata=strata,
                              prior_fail=PRIOR_FAIL, prior_ok=PRIOR_OK)

    def describe(self) -> str:
        if self.scope == "prior":
            return "no recorded outcomes; using the 0.5 prior"
        b = self.bounds()
        return (f"{self.p_fail:.0%} over {self.n} {self.scope} runs "
                f"(recency-weighted mass {self.weight:.1f}, "
                f"90% credible {b.lo:.0%}-{b.hi:.0%})")


def evidence(
    tier: str,
    project: str | None = None,
    log: Path | str | None = None,
    outcomes: list[Outcome] | None = None,
    *,
    recency_weighted: bool = True,
    now: float | None = None,
) -> Evidence:
    """Smoothed escalation rate for a tier, plus the evidence behind it.

    Falls back to the global rate for that tier when a project has too little
    history, and to the 0.5 prior when there is none at all.

    `now` is the instant the recency weights decay toward, and it defaults to
    the wall clock. It is a parameter rather than a hidden call to `time.time()`
    for two reasons, both of which bit:

    * **Replay.** Asking "what would the gate have said when this row arrived"
      requires decaying toward *that* moment. Against the wall clock instead,
      every row in a log older than a few half-lives weighs essentially nothing,
      the evidence mass collapses, and the estimator silently returns the 0.5
      prior for everything -- which looks like a calibrated coin and is really an
      admission that no data was used. `adder calib` found exactly that.
    * **Determinism.** A function whose output depends on the wall clock cannot
      be tested to a fixed value, and the tests around it end up asserting only
      that it returned something between 0 and 1.
    """
    rows = outcomes if outcomes is not None else load(log)
    scope = "project" if project is not None else "global"
    scoped = [o for o in rows if o.tier == tier and (project is None or o.project == project)]
    if len(scoped) < MIN_SCOPED and project is not None:
        scoped = [o for o in rows if o.tier == tier]
        scope = "global"
    if not scoped:
        return Evidence(PRIOR_FAIL / (PRIOR_FAIL + PRIOR_OK), 0, 0.0, "prior", 0.0)

    if not recency_weighted:
        fails = sum(1 for o in scoped if o.escalated)
        p = (fails + PRIOR_FAIL) / (len(scoped) + PRIOR_FAIL + PRIOR_OK)
        return Evidence(p, len(scoped), float(len(scoped)), scope, float(fails))

    at = time.time() if now is None else now
    wf = sum(_weight(o, at) for o in scoped if o.escalated)
    wt = sum(_weight(o, at) for o in scoped)
    p = (wf + PRIOR_FAIL) / (wt + PRIOR_FAIL + PRIOR_OK)
    return Evidence(p, len(scoped), wt, scope, wf)


def p_fail(
    tier: str,
    project: str | None = None,
    log: Path | str | None = None,
    outcomes: list[Outcome] | None = None,
    *,
    recency_weighted: bool = True,
    now: float | None = None,
) -> float:
    """The escalation rate on its own, for callers that only gate on the number."""
    return evidence(tier, project, log, outcomes,
                    recency_weighted=recency_weighted, now=now).p_fail


def calibration(log: Path | str | None = None) -> dict[str, dict[str, float | int]]:
    """Per-tier escalation stats, for `/adder-doctor` and for spotting drift."""
    rows = load(log)
    by: dict[str, list[Outcome]] = defaultdict(list)
    for o in rows:
        by[o.tier].append(o)
    return {
        tier: {
            "n": len(v),
            "escalated": sum(1 for o in v if o.escalated),
            "p_fail": p_fail(tier, None, log, outcomes=rows),
            "cost": round(sum(o.cost for o in v), 4),
        }
        for tier, v in sorted(by.items())
    }


def readiness(
    log: Path | str | None = None,
    *,
    project: str | None = None,
    context_tokens: int = 300_000,
    confidence: float = 0.30,
    on=None,
) -> list[dict]:
    """Per tier: what the log says now, and what it would take to act on it.

    This exists because the most common state of this file is empty, and an
    empty file produces a router that always answers Opus without ever saying
    why. "p_fail 0.50" is not an explanation -- it is the prior, and the prior
    is what you get for having no data. Naming the gap turns a silent default
    into a to-do list: this many more runs at this tier, and the failure rate
    has to come in under this, before the router is permitted to prefer it.

    `confidence` is the classifier confidence to assume for the prior, and
    defaults to the abstaining case -- the one where a downgrade is even on the
    table. The break-even moves with context size, so `context_tokens` is an
    input rather than a constant.
    """
    from adder.decide.route.classify import Tier
    from adder.decide.route.policy import prior_p_fail, routing_overhead
    from adder.pricing.cost import max_tolerable_p_fail

    rows = load(log)
    overhead = routing_overhead(context_tokens, Tier.T2.model, on)
    need_by_tier = {Tier.T0: 8_000, Tier.T1: 20_000}
    out = []
    for tier in (Tier.T0, Tier.T1):
        ev = evidence(tier.name, project, log, rows)
        need = need_by_tier[tier]
        cap = max_tolerable_p_fail(tier.model, Tier.T2.model, ctx_tokens=need,
                                   est_out_tokens=800, retry_overhead=overhead, on=on)
        p = ev.p_fail if ev.informative else prior_p_fail(confidence)
        if not ev.informative:
            verdict = "needs history"
        elif p >= cap:
            verdict = "fails its own break-even"
        else:
            verdict = "usable"
        out.append({
            "tier": tier.name, "model": tier.model, "n": ev.n,
            "weight": ev.weight, "scope": ev.scope, "p_fail": p,
            "is_prior": not ev.informative, "break_even": cap,
            "shortfall": max(0.0, MIN_EVIDENCE - ev.weight), "verdict": verdict,
        })
    return out


def known_hashes(log: Path | str | None = None) -> set[str]:
    """Every `task_hash` already on record, so an import can be idempotent."""
    return {o.task_hash for o in load(log) if o.task_hash}


def cmd_import(a) -> int:
    """Backfill the outcome log from transcripts.

    Dry by default. Writing to a file that calibrates routing is not something
    a command should do because someone was curious what it would find.
    """
    from adder.decide.track.dispatch import report as dispatch_report
    from adder.decide.track.dispatch import scan, to_outcomes

    found = scan(a.root)
    rows = to_outcomes(found, known_hashes=known_hashes(a.log))
    written = None
    if a.write:
        for row in rows:
            record(row, a.log)
        written = len(rows)

    if a.json:
        print(json.dumps({
            "dispatches": len(found.dispatches),
            "usable": len(found.usable),
            "escalations": found.escalations,
            "unresolved": found.unresolved,
            "untiered": found.untiered,
            "new_rows": len(rows),
            "written": bool(a.write),
            "by_tier": {t: {"runs": n, "escalated": e}
                        for t, (n, e) in found.by_tier().items()},
        }))
        return 0

    print()
    print(dispatch_report(found, new_rows=written))
    if a.write and rows:
        print()
        for tier in sorted({r.tier for r in rows}):
            print(f"  {tier} is now at {evidence(tier, None, a.log).describe()}")
    print()
    return 0


def cmd_record(a) -> int:
    """Append one outcome from the command line.

    The only way to write this log used to be a Python snippet pasted out of a
    skill file. That is the whole adaptive half of the tool sitting behind a
    step nobody performs, and the empty log on every machine is the evidence:
    `p_fail` never leaves its prior, so the router never learns that a cheaper
    tier works here and never stops sending the work to Opus.
    """
    record(Outcome(
        tier=a.tier, model=a.model, project=a.project, escalated=a.escalated,
        context_tokens=a.context, remaining_turns=a.remaining, cost=a.cost,
        task_hash=a.task_hash, reason=a.reason, effort=a.effort,
        duration_s=a.duration,
    ), a.log)
    ev = evidence(a.tier, a.project, a.log)
    print(f"  recorded: {a.tier} {'escalated' if a.escalated else 'completed'} "
          f"on {a.model or 'an unnamed model'}")
    print(f"  {a.tier} is now at {ev.describe()}"
          f"{'' if ev.informative else ' -- still short of actionable'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="adder outcomes",
        description="escalation calibration: each tier's failure rate, and what "
                    "it would take to act on it")
    sub = ap.add_subparsers(dest="cmd")
    ap.add_argument("--log", default=None)
    ap.add_argument("--prune", action="store_true", help="trim the log and exit")
    ap.add_argument("--project", default=None, help="scope the readiness table")
    ap.add_argument("--context", type=int, default=300_000,
                    help="context size the break-even is quoted at")
    ap.add_argument("--json", action="store_true")

    p_imp = sub.add_parser(
        "import", help="backfill the log from transcripts (dry run by default)")
    p_imp.add_argument("root", nargs="?", default=None,
                       help="transcript directory (default: the `root` setting)")
    p_imp.add_argument("--write", action="store_true",
                       help="actually append the derived rows")
    p_imp.add_argument("--log", default=None)
    p_imp.add_argument("--json", action="store_true")

    p_rec = sub.add_parser("record", help="append one outcome")
    p_rec.add_argument("--tier", required=True, help="T0 | T1 | T2 | T3")
    p_rec.add_argument("--model", default="", help="the model that actually ran")
    p_rec.add_argument("--project", default="", help="project the task belonged to")
    p_rec.add_argument("--escalated", action="store_true",
                       help="the tier could not finish and the work moved up")
    p_rec.add_argument("--cost", type=float, default=0.0)
    p_rec.add_argument("--context", type=int, default=0)
    p_rec.add_argument("--remaining", type=int, default=0)
    p_rec.add_argument("--effort", default="")
    p_rec.add_argument("--reason", default="")
    p_rec.add_argument("--task-hash", default="")
    p_rec.add_argument("--duration", type=float, default=0.0)
    p_rec.add_argument("--log", default=None)

    from adder.core.filters import root_of as _root_of

    a = ap.parse_args(argv)
    # `root_of`: the argument if one was given, else the `root`
    # setting. Resolved here so two commands cannot disagree
    # about which transcript directory `adder config` names.
    a.root = str(_root_of(a))
    if a.cmd == "record":
        return cmd_record(a)
    if a.cmd == "import":
        if a.root is None:
            from adder.core import settings

            a.root = settings.get("root")
        return cmd_import(a)

    if a.prune:
        print(f"  dropped {prune(a.log)} old rows")
        return 0

    cal = calibration(a.log)
    ready = readiness(a.log, project=a.project, context_tokens=a.context)
    if a.json:
        print(json.dumps({"calibration": cal, "readiness": ready}))
        return 0

    print()
    if not cal:
        print(f"  No outcomes recorded yet ({log_path(a.log)}).")
        print("  Until there is history, p_fail falls back to a prior read off the")
        print("  classifier's confidence, and the router may not route below the")
        print("  tier the classifier asked for. Record one with:")
        print("    adder outcomes record --tier T1 --model claude-sonnet-5 "
              "--project <name>")
    else:
        print(f"  {'tier':<8}{'runs':>7}{'escalated':>11}{'p_fail':>9}{'cost':>10}")
        for tier, st in cal.items():
            print(f"  {tier:<8}{st['n']:>7}{st['escalated']:>11}"
                  f"{st['p_fail']:>9.2f}   ${st['cost']:>8.3f}")
        print()
        print("  p_fail is recency-weighted (30-day half-life) and Beta(1,1)-smoothed.")

    print()
    print(f"  Before the router may prefer a cheaper tier "
          f"(at {a.context:,} tok of context):")
    print(f"  {'tier':<6}{'evidence now':<28}{'still needed':>14}"
          f"{'p_fail':>9} {'break-even':>10}   verdict")
    for r in ready:
        have = (f"{r['n']} {r['scope']} runs (mass {r['weight']:.1f})"
                if r["n"] else "none")
        short = f"{r['shortfall']:.1f} runs" if r["shortfall"] else "-"
        star = "*" if r["is_prior"] else " "
        print(f"  {r['tier']:<6}{have:<28}{short:>14}"
              f"{r['p_fail']:>8.2f}{star}{r['break_even']:>10.0%}   {r['verdict']}")
    print("  * a prior, not a measurement. A prior never buys a downgrade.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
