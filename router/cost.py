"""Cache- and context-aware cost model.

The term every other LLM router omits
-------------------------------------
Routers built for stateless APIs price a request as `in*rate_in + out*rate_out`.
In a persistent agent session that is wrong, because the conversation prefix is
re-sent on every turn. A token admitted to the main context is billed once as a
cache write and then again, as a cache read, on **every remaining turn**:

    admitted_token_cost(n, model, remaining_turns)
      = n * rate_in * write_mult                    # once
      + n * rate_in * 0.10 * remaining_turns        # forever after

On measured data (median session: 607 turns, 544K context) this term is ~76% of
total spend. It dominates model selection by roughly an order of magnitude, and
it is why downgrading a *warm* conversation to a cheaper model loses money: the
cheap model must re-read the whole context uncached.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .prices import CACHE_READ_MULT, CACHE_WRITE_MULT, rate

M = 1_000_000.0


def turn_cost(
    model: str,
    *,
    uncached_in: int = 0,
    cache_read: int = 0,
    cache_write: int = 0,
    out: int = 0,
    ttl: str = "5m",
    on: date | None = None,
) -> float:
    """USD for a single turn, given its token accounting."""
    r = rate(model, on)
    wm = CACHE_WRITE_MULT[ttl]
    return (
        uncached_in * r.inp
        + cache_read * r.inp * CACHE_READ_MULT
        + cache_write * r.inp * wm
        + out * r.out
    ) / M


def admitted_token_cost(
    n_tokens: int,
    model: str,
    remaining_turns: int,
    *,
    ttl: str = "5m",
    on: date | None = None,
) -> float:
    """Full lifetime cost of putting `n_tokens` into a persistent context.

    This is the number that should drive placement decisions, not the one-off
    read cost. 10K tokens in a 1000-turn session costs ~$5.06 on Opus 5.
    """
    r = rate(model, on)
    write = n_tokens * r.inp * CACHE_WRITE_MULT[ttl] / M
    reads = n_tokens * r.inp * CACHE_READ_MULT * max(0, remaining_turns) / M
    return write + reads


@dataclass(frozen=True)
class Decision:
    """A gate result that can explain itself. Never returns a bare bool."""

    ok: bool
    saving: float          # USD saved by taking the action (negative = loses money)
    reason: str

    def __bool__(self) -> bool:  # allows `if gate(...)`
        return self.ok


def switch_is_profitable(
    from_model: str,
    to_model: str,
    ctx_tokens: int,
    est_out_tokens: int,
    *,
    switch_in_mult: float = 1.0,
    on: date | None = None,
) -> Decision:
    """Is switching model for ONE turn of a warm conversation worth it?

    Staying keeps a warm cache (context read at 0.1x). Switching invalidates it —
    the cache is model-scoped, with no escape hatch — so the new model re-reads
    the entire context at `switch_in_mult` x its input rate.

    `switch_in_mult=1.0` (uncached) is the optimistic case and yields the
    headline break-even `out > ctx/40` for Opus->Haiku. Real Claude Code caching
    writes at 1.25x, which tightens it to ~ctx/26.7. If the optimistic case
    already fails, the switch is definitely wrong.
    """
    rf, rt = rate(from_model, on), rate(to_model, on)
    stay = (ctx_tokens * rf.inp * CACHE_READ_MULT + est_out_tokens * rf.out) / M
    switch = (ctx_tokens * rt.inp * switch_in_mult + est_out_tokens * rt.out) / M
    saving = stay - switch
    if saving > 0:
        return Decision(True, saving, f"saves ${saving:.4f}: output win beats cache loss")
    breakeven = _breakeven_out(ctx_tokens, rf, rt, switch_in_mult)
    return Decision(
        False,
        saving,
        f"loses ${-saving:.4f}: re-reading {ctx_tokens:,} tok uncached on "
        f"{to_model} costs more than the output saving "
        f"(needs >{breakeven:,.0f} output tok, est {est_out_tokens:,})",
    )


def _breakeven_out(ctx: int, rf, rt, mult: float) -> float:
    """Output tokens at which switching breaks even. inf if it never does."""
    denom = rf.out - rt.out
    if denom <= 0:
        return float("inf")
    return ctx * (rt.inp * mult - rf.inp * CACHE_READ_MULT) / denom


def placement_cost(
    *,
    tokens_read: int,
    summary_tokens: int,
    remaining_turns: int,
    main_model: str,
    sub_model: str = "claude-haiku-4-5",
    brief_tokens: int = 400,
    on: date | None = None,
) -> tuple[float, float, Decision]:
    """Inline vs. subagent for a read-heavy task. Returns (inline, sub, decision).

    Inline: everything read lands in the main context and is amortized.
    Subagent: reads happen once in a throwaway context on a cheap model; only
    the summary is amortized. This is the dominant lever in long sessions.
    """
    inline = admitted_token_cost(tokens_read, main_model, remaining_turns, on=on)

    rs = rate(sub_model, on)
    # Subagent reads fresh (uncached) in its own short-lived context.
    sub_side = (brief_tokens * rs.inp + tokens_read * rs.inp + summary_tokens * rs.out) / M
    # Only the returned summary is admitted to the main context.
    sub = sub_side + admitted_token_cost(summary_tokens, main_model, remaining_turns, on=on)

    saving = inline - sub
    if saving > 0:
        return inline, sub, Decision(
            True, saving,
            f"delegate: saves ${saving:.4f} "
            f"({tokens_read:,} tok stay out of a context re-read {remaining_turns} more times)",
        )
    return inline, sub, Decision(
        False, saving,
        f"keep inline: delegating costs ${-saving:.4f} more "
        f"(only {remaining_turns} turns left to amortize over)",
    )


def escalation_is_profitable(
    cheap_model: str,
    expensive_model: str,
    *,
    ctx_tokens: int,
    est_out_tokens: int,
    p_fail: float,
    on: date | None = None,
) -> Decision:
    """Should we ATTEMPT the cheap tier, given it may fail and force a retry?

    A failed cheap attempt is not free: you pay for it AND for the expensive run.
    Try cheap only when  cheap + p_fail*(cheap + expensive) < expensive.
    `p_fail` must come from the outcome log, not a guess.
    """
    if not 0.0 <= p_fail <= 1.0:
        raise ValueError(f"p_fail must be in [0,1], got {p_fail}")
    rc, re_ = rate(cheap_model, on), rate(expensive_model, on)
    cheap = (ctx_tokens * rc.inp + est_out_tokens * rc.out) / M
    exp = (ctx_tokens * re_.inp + est_out_tokens * re_.out) / M
    expected = cheap + p_fail * (cheap + exp)
    saving = exp - expected
    if saving > 0:
        max_p = (exp - cheap) / (cheap + exp) if (cheap + exp) else 0.0
        return Decision(True, saving, f"try cheap: expected saving ${saving:.4f} (p_fail {p_fail:.0%} < {max_p:.0%})")
    return Decision(
        False, saving,
        f"go straight to {expensive_model}: at p_fail={p_fail:.0%} the retry risk "
        f"costs ${-saving:.4f} more than just doing it once",
    )
