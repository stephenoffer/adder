"""Routing decisions: combine classification, session state, and the cost model.

The most important behaviour here is *declining to route*.

A routing step is not free. If `/route` costs an extra turn, that turn re-reads
the whole context: at 500K tokens on Opus that is ~$0.25 before the router has
done anything useful. So a recommendation is only emitted when the modelled
saving clears that overhead. Otherwise the honest answer is "just do it".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from .classify import Tier, Verdict, classify
from .cost import Decision, admitted_token_cost, placement_cost, switch_is_profitable
from .prices import CACHE_READ_MULT, rate

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


def decide(
    task: str,
    *,
    context_tokens: int,
    remaining_turns: int,
    session_model: str = "claude-opus-5",
    est_read_tokens: int | None = None,
    est_out_tokens: int = 800,
    compression: float = 10.0,
    on: date | None = None,
) -> Plan:
    """Recommend where and on what model to run `task`."""
    overhead = routing_overhead(context_tokens, session_model, on)

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

    inline_cost, sub_cost, place = placement_cost(
        tokens_read=est_read_tokens,
        summary_tokens=max(200, int(est_read_tokens / compression)),
        remaining_turns=remaining_turns,
        main_model=session_model,
        sub_model=v.tier.model,
        on=on,
    )

    if place:
        reasons.append(place.reason)
        return Plan(
            action="delegate", tier=v.tier, model=v.tier.model, effort=v.tier.effort,
            agent=v.tier.agent, saving=place.saving, overhead=overhead,
            confidence=v.confidence, reasons=reasons,
        )

    # Delegation not worth it. Would an in-session downgrade help? Usually not.
    if v.tier < Tier.T2:
        sw: Decision = switch_is_profitable(
            session_model, v.tier.model, context_tokens, est_out_tokens, on=on
        )
        reasons.append(sw.reason)
        if sw:
            return Plan(
                action="downgrade", tier=v.tier, model=v.tier.model, effort=v.tier.effort,
                agent=None, saving=sw.saving, overhead=overhead,
                confidence=v.confidence, reasons=reasons,
            )
    else:
        reasons.append("classified as needing the strong model; no downgrade considered")

    reasons.append(place.reason)
    return Plan(
        action="inline", tier=Tier.T2, model=session_model, effort="high",
        agent=None, saving=0.0, overhead=overhead,
        confidence=v.confidence, reasons=reasons,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    from .horizon import DEFAULT_REMAINING, load as load_horizon
    from .live import analyse, current_session

    ap = argparse.ArgumentParser(prog="router.policy")
    ap.add_argument("task", nargs="*")
    ap.add_argument("--context", type=int, default=None)
    ap.add_argument("--remaining", type=int, default=None)
    ap.add_argument("--read-tokens", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    ctx, rem, model = a.context, a.remaining, "claude-opus-5"
    if ctx is None or rem is None:
        s = current_session()
        if s is not None:
            r = analyse(s)
            ctx = ctx if ctx is not None else r.context
            rem = rem if rem is not None else r.projected_remaining
            model = r.model
    ctx = ctx if ctx is not None else 100_000
    rem = rem if rem is not None else load_horizon().remaining(0)

    p = decide(" ".join(a.task), context_tokens=ctx, remaining_turns=rem,
               session_model=model, est_read_tokens=a.read_tokens)
    if a.json:
        print(json.dumps({
            "action": p.action, "model": p.model, "effort": p.effort,
            "agent": p.agent, "saving": round(p.saving, 4),
            "overhead": round(p.overhead, 4), "worth_it": p.worth_it,
            "confidence": p.confidence, "reasons": p.reasons,
            "context_tokens": ctx, "remaining_turns": rem,
        }))
    else:
        print(p.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
