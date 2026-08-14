"""Routing decisions: combine classification, session state, and the cost model.

The most important behaviour here is *declining to route*.

A routing step is not free. If `/route` costs an extra turn, that turn re-reads
the whole context: at 500K tokens on Opus that is ~$0.25 before the router has
done anything useful. So a recommendation is only emitted when the modelled
saving clears that overhead. Otherwise the honest answer is "just do it".

Four gates, in the order they can each veto
-------------------------------------------
1. **Feasibility.** A tier whose context window cannot hold the task is not an
   option, however cheap it looks.
2. **Placement.** Delegating keeps the read out of a context that is re-read on
   every remaining turn. This is the largest lever and is checked first.
3. **Escalation risk.** A cheap tier that fails costs its own run *plus* the
   expensive one. `p_fail` comes from the measured outcome log, not a guess, so
   a tier that keeps failing on this project stops being recommended.
4. **Overhead.** If none of the above clears the cost of the routing turn
   itself, the plan is "inline".

Effort is chosen alongside the model because it is the one lever that reduces
output volume *without* invalidating the cache -- a downgrade rebuilds the
prefix, a lower effort level does not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from .classify import Tier, Verdict, classify
from .cost import (
    Decision,
    effort_saving,
    escalation_is_profitable,
    placement_cost,
    switch_is_profitable,
)
from .prices import CACHE_READ_MULT, context_limit, fits, rate, supports_effort

M = 1_000_000.0

# Cost of the routing turn itself: it re-reads context and emits a dispatch.
ROUTING_TURN_OUTPUT_TOKENS = 400

# Literal placeholders that mean argument substitution did not happen.
_PLACEHOLDER = re.compile(r"\$?\{?ARGUMENTS\}?|\$[0-9]", re.I)


@dataclass
class Plan:
    action: str                 # "inline" | "delegate" | "downgrade"
    tier: Tier
    model: str
    effort: str
    agent: str | None
    saving: float
    overhead: float
    confidence: float
    reasons: list[str]
    p_fail: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def worth_it(self) -> bool:
        return self.action != "inline" and self.saving > self.overhead

    def render(self) -> str:
        head = f"{self.action.upper()}"
        if self.action != "inline":
            head += f" -> {self.agent} ({self.model}, effort={self.effort})"
        lines = [
            head,
            f"  modelled saving ${self.saving:,.3f}  routing overhead ${self.overhead:,.3f}"
            f"  confidence {self.confidence:.2f}",
        ]
        lines += [f"  - {r}" for r in self.reasons]
        for w in self.warnings:
            lines.append(f"  ! {w}")
        if self.action != "inline" and not self.worth_it:
            lines.append("  - saving does not clear routing overhead; do it inline instead")
        return "\n".join(lines)


def routing_overhead(context_tokens: int, session_model: str, on: date | None = None) -> float:
    """Marginal cost of spending one extra turn deciding how to route."""
    r = rate(session_model, on)
    return (
        context_tokens * r.inp * CACHE_READ_MULT
        + ROUTING_TURN_OUTPUT_TOKENS * r.out
    ) / M


def choose_effort(tier: Tier, model: str) -> str:
    """Lowest effort level that suits the tier, if the model accepts one.

    Effort is free to lower mid-session: unlike a model switch it does not
    invalidate the prompt cache, so it is the safe way to cut output volume.
    """
    want = tier.effort
    if not supports_effort(model, want):
        # Haiku 4.5 rejects `effort` entirely; fall back to the model default.
        return "default"
    return want


def decide(
    task: str,
    *,
    context_tokens: int,
    remaining_turns: int,
    session_model: str = "claude-opus-5",
    est_read_tokens: int | None = None,
    est_out_tokens: int = 800,
    compression: float = 10.0,
    project: str | None = None,
    p_fail: float | None = None,
    on: date | None = None,
) -> Plan:
    """Recommend where and on what model to run `task`."""
    overhead = routing_overhead(context_tokens, session_model, on)
    warnings: list[str] = []

    # Guard: an empty or unsubstituted task must never produce a confident
    # dispatch. If `$ARGUMENTS` fails to substitute in the skill, the string
    # arrives literally -- refuse rather than delegate an imaginary task.
    stripped = (task or "").strip()
    if not stripped or _PLACEHOLDER.fullmatch(stripped):
        return Plan(
            action="inline", tier=Tier.T2, model=session_model, effort="high",
            agent=None, saving=0.0, overhead=overhead, confidence=0.0,
            reasons=["no task text received; refusing to route "
                     "(check that the task was passed through)"],
        )

    v: Verdict = classify(task)
    reasons = list(v.reasons)

    # Estimate how much the task will pull into context if run inline.
    if est_read_tokens is None:
        est_read_tokens = {Tier.T0: 8_000, Tier.T1: 20_000,
                           Tier.T2: 60_000, Tier.T3: 120_000}[v.tier]

    # --- Gate 1: feasibility. A tier that cannot hold the task is not an option.
    tier, model = v.tier, v.tier.model
    need = est_read_tokens + max(200, int(est_read_tokens / compression)) + 400
    while not fits(model, need) and tier < Tier.T2:
        warnings.append(
            f"{model} holds {context_limit(model):,} tok but the task needs "
            f"~{need:,}; escalating a tier for feasibility, not capability")
        tier = Tier(int(tier) + 1)
        model = tier.model

    # --- Gate 2: escalation risk, from the measured outcome log.
    if p_fail is None:
        try:
            from .outcomes import p_fail as measured_p_fail

            p_fail = measured_p_fail(tier.name, project)
        except Exception:
            p_fail = 0.5
    if tier < Tier.T2:
        esc = escalation_is_profitable(
            model, Tier.T2.model, ctx_tokens=min(need, context_limit(model)),
            est_out_tokens=est_out_tokens, p_fail=p_fail, on=on)
        if not esc:
            reasons.append(esc.reason)
            tier, model = Tier.T2, Tier.T2.model
        else:
            reasons.append(esc.reason)

    effort = choose_effort(tier, model)

    inline_cost, sub_cost, place = placement_cost(
        tokens_read=est_read_tokens,
        summary_tokens=max(200, int(est_read_tokens / compression)),
        remaining_turns=remaining_turns,
        main_model=session_model,
        sub_model=model,
        on=on,
    )

    if place:
        reasons.append(place.reason)
        return Plan(
            action="delegate", tier=tier, model=model, effort=effort,
            agent=tier.agent, saving=place.saving, overhead=overhead,
            confidence=v.confidence, reasons=reasons, p_fail=p_fail,
            warnings=warnings,
        )

    # Delegation not worth it. Would an in-session downgrade help? Usually not.
    if tier < Tier.T2:
        sw: Decision = switch_is_profitable(
            session_model, model, context_tokens, est_out_tokens, on=on
        )
        reasons.append(sw.reason)
        if sw:
            return Plan(
                action="downgrade", tier=tier, model=model, effort=effort,
                agent=None, saving=sw.saving, overhead=overhead,
                confidence=v.confidence, reasons=reasons, p_fail=p_fail,
                warnings=warnings,
            )
    else:
        reasons.append("classified as needing the strong model; no downgrade considered")

    # Last resort that keeps the cache warm: lower the effort level instead of
    # the model. Same prefix, same cache, fewer output tokens.
    inline_effort = "high"
    if supports_effort(session_model, "medium") and v.tier <= Tier.T1:
        saved, ed = effort_saving(
            est_out_tokens, session_model, from_effort="high", to_effort="medium",
            remaining_turns=remaining_turns, on=on)
        if ed and saved > 0:
            reasons.append(ed.reason)
            inline_effort = "medium"

    reasons.append(place.reason)
    return Plan(
        action="inline", tier=Tier.T2, model=session_model, effort=inline_effort,
        agent=None, saving=0.0, overhead=overhead,
        confidence=v.confidence, reasons=reasons, p_fail=p_fail,
        warnings=warnings,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    from .horizon import load as load_horizon
    from .live import analyse, current_session

    ap = argparse.ArgumentParser(prog="router.policy")
    ap.add_argument("task", nargs="*")
    ap.add_argument("--context", type=int, default=None)
    ap.add_argument("--remaining", type=int, default=None)
    ap.add_argument("--read-tokens", type=int, default=None)
    ap.add_argument("--project", default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    ctx, rem, model, project = a.context, a.remaining, "claude-opus-5", a.project
    if ctx is None or rem is None:
        s = current_session()
        if s is not None:
            r = analyse(s)
            ctx = ctx if ctx is not None else r.context
            rem = rem if rem is not None else r.projected_remaining
            model = r.model
            project = project or s.project
    ctx = ctx if ctx is not None else 100_000
    rem = rem if rem is not None else load_horizon().remaining(0)

    p = decide(" ".join(a.task), context_tokens=ctx, remaining_turns=rem,
               session_model=model, est_read_tokens=a.read_tokens, project=project)
    if a.json:
        print(json.dumps({
            "action": p.action, "model": p.model, "effort": p.effort,
            "agent": p.agent, "saving": round(p.saving, 4),
            "overhead": round(p.overhead, 4), "worth_it": p.worth_it,
            "confidence": p.confidence, "p_fail": round(p.p_fail, 3),
            "reasons": p.reasons, "warnings": p.warnings,
            "context_tokens": ctx, "remaining_turns": rem,
        }))
    else:
        print(p.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
