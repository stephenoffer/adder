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

Four failure modes this module gates on, which a price-only model cannot see
-----------------------------------------------------------------------------
* **Feasibility.** Haiku holds 200K; the median context here is 544K. A
  "saving" that names a model the context does not fit in is a 400 error.
* **Cache minimums.** A prefix under the model's minimum silently does not
  cache, so a delegation whose brief is below it pays full input rate twice.
* **TTL.** A 5m cache that expires between turns is re-written at 1.25x rather
  than read at 0.10x -- a 12.5x swing. Idle gaps, not token counts, decide it.
* **Fan-out.** N parallel calls sharing a prefix all miss: a cache entry is only
  readable once the first response starts streaming. Serialising one call first
  turns N cache writes into 1 write + (N-1) reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .prices import (
    BATCH_MULT,
    CACHE_READ_MULT,
    CACHE_WRITE_MULT,
    TTL_SECONDS,
    cache_min,
    caches,
    context_limit,
    fits,
    rate,
)

M = 1_000_000.0

# Relative output volume by effort level, indexed to `high` = 1.0.
#
# These are PRIORS, not measurements: effort controls thinking depth, and the
# mapping from level to token volume is workload-specific. Anything using them
# is labelled MODELLED, and `router.effort` re-fits them from a transcript when
# there is enough per-effort history to do so.
EFFORT_OUTPUT_MULT = {"low": 0.35, "medium": 0.60, "high": 1.00,
                      "xhigh": 1.50, "max": 2.20}


def turn_cost(
    model: str,
    *,
    uncached_in: int = 0,
    cache_read: int = 0,
    cache_write: int = 0,
    out: int = 0,
    ttl: str = "5m",
    speed: str = "standard",
    batch: bool = False,
    on: date | None = None,
) -> float:
    """USD for a single turn, given its token accounting."""
    r = rate(model, on, speed=speed)
    wm = CACHE_WRITE_MULT[ttl]
    total = (
        uncached_in * r.inp
        + cache_read * r.inp * CACHE_READ_MULT
        + cache_write * r.inp * wm
        + out * r.out
    ) / M
    return total * (BATCH_MULT if batch else 1.0)


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
    check_context: bool = True,
    on: date | None = None,
) -> Decision:
    """Is switching model for ONE turn of a warm conversation worth it?

    Staying keeps a warm cache (context read at 0.1x). Switching invalidates it --
    the cache is model-scoped, with no escape hatch -- so the new model re-reads
    the entire context at `switch_in_mult` x its input rate.

    `switch_in_mult=1.0` (uncached) is the optimistic case and yields the
    headline break-even `out > ctx/40` for Opus->Haiku. Real Claude Code caching
    writes at 1.25x, which tightens it to ~ctx/26.7. If the optimistic case
    already fails, the switch is definitely wrong.

    `check_context=True` additionally refuses switches the target model cannot
    physically hold. Set it False to probe the pure break-even arithmetic.
    """
    rf, rt = rate(from_model, on), rate(to_model, on)
    stay = (ctx_tokens * rf.inp * CACHE_READ_MULT + est_out_tokens * rf.out) / M
    switch = (ctx_tokens * rt.inp * switch_in_mult + est_out_tokens * rt.out) / M
    saving = stay - switch
    breakeven = _breakeven_out(ctx_tokens, rf, rt, switch_in_mult)

    if check_context and not fits(to_model, ctx_tokens):
        limit = context_limit(to_model)
        return Decision(
            False, min(saving, 0.0) if saving < 0 else 0.0,
            f"impossible: {ctx_tokens:,} tok exceeds {to_model}'s "
            f"{limit:,}-token context limit. Even if it fit, the switch "
            f"needs >{breakeven:,.0f} output tok (est {est_out_tokens:,})",
        )

    if saving > 0:
        return Decision(True, saving, f"saves ${saving:.4f}: output win beats cache loss")
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

    Refuses to recommend a subagent whose context window cannot hold the read.
    """
    inline = admitted_token_cost(tokens_read, main_model, remaining_turns, on=on)

    rs = rate(sub_model, on)
    # Subagent reads fresh (uncached) in its own short-lived context.
    sub_side = (brief_tokens * rs.inp + tokens_read * rs.inp + summary_tokens * rs.out) / M
    # Only the returned summary is admitted to the main context.
    sub = sub_side + admitted_token_cost(summary_tokens, main_model, remaining_turns, on=on)
    saving = inline - sub

    need = brief_tokens + tokens_read + summary_tokens
    if not fits(sub_model, need):
        return inline, sub, Decision(
            False, 0.0,
            f"cannot delegate to {sub_model}: the task needs ~{need:,} tok of "
            f"context, over its {context_limit(sub_model):,} limit. Use a "
            f"larger subagent model or split the read",
        )

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
    if not fits(cheap_model, ctx_tokens):
        return Decision(
            False, 0.0,
            f"go straight to {expensive_model}: {ctx_tokens:,} tok exceeds "
            f"{cheap_model}'s {context_limit(cheap_model):,}-token limit",
        )
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


# --------------------------------------------------------------------------
# Cache TTL, expiry, and fan-out: three levers a price-only model cannot see.
# --------------------------------------------------------------------------

def cache_write_cost(n_tokens: int, model: str, ttl: str = "5m",
                     on: date | None = None) -> float:
    return n_tokens * rate(model, on).inp * CACHE_WRITE_MULT[ttl] / M


def cache_read_cost(n_tokens: int, model: str, on: date | None = None) -> float:
    return n_tokens * rate(model, on).inp * CACHE_READ_MULT / M


def cache_miss_cost(n_tokens: int, model: str, ttl: str = "5m",
                    on: date | None = None) -> float:
    """Extra USD paid when a cache entry expired and had to be rewritten.

    Re-writing costs `write_mult` x input rate; reading would have cost 0.10x.
    On Opus 5 with a 5m TTL that is 1.15x input rate of pure waste -- 11.5x
    what the read would have been.
    """
    return cache_write_cost(n_tokens, model, ttl, on) - cache_read_cost(n_tokens, model, on)


def choose_ttl(
    n_tokens: int,
    model: str,
    *,
    turns: int,
    gap_seconds: float,
    on: date | None = None,
) -> tuple[str, float, str]:
    """Pick the cheaper cache TTL for a session with a known think-time gap.

    5m costs 1.25x to write but expires after 300s of idle; 1h costs 2.00x and
    survives an hour. A session whose turns are more than 5 minutes apart pays
    the 5m write premium on EVERY turn instead of once.

    Returns (ttl, saving_vs_other, reason).
    """
    r = rate(model, on).inp
    turns = max(1, turns)
    costs = {}
    for ttl, secs in TTL_SECONDS.items():
        # A write happens on the first turn and again after every expiry.
        expiries = turns - 1 if gap_seconds > secs else 0
        writes = 1 + expiries
        reads = turns - writes
        costs[ttl] = (
            writes * n_tokens * r * CACHE_WRITE_MULT[ttl]
            + max(0, reads) * n_tokens * r * CACHE_READ_MULT
        ) / M
    best = min(costs, key=lambda t: costs[t])
    other = "1h" if best == "5m" else "5m"
    saving = costs[other] - costs[best]
    if gap_seconds > TTL_SECONDS["5m"] and best == "1h":
        why = (f"turns are {gap_seconds:.0f}s apart, past the 5m TTL: a 5m cache "
               f"expires every turn and is rewritten at 1.25x instead of read at 0.10x")
    elif best == "5m":
        why = f"turns are {gap_seconds:.0f}s apart, inside the 5m TTL; the 2x write premium of 1h never pays back"
    else:
        why = "1h keeps the prefix warm across the measured gap"
    return best, saving, why


def fanout_cost(
    n_agents: int,
    shared_prefix_tokens: int,
    model: str,
    *,
    serialise_first: bool = True,
    ttl: str = "5m",
    on: date | None = None,
) -> tuple[float, float, Decision]:
    """Cost of N parallel calls over a shared prefix, staggered vs. all-at-once.

    A cache entry only becomes readable once the first response starts
    streaming. Fire N identical-prefix requests simultaneously and all N pay the
    write premium. Await the first token of one, then fan out the rest, and it
    is 1 write + (N-1) reads.

    Returns (parallel_cost, staggered_cost, decision).
    """
    n = max(1, n_agents)
    w = cache_write_cost(shared_prefix_tokens, model, ttl, on)
    rd = cache_read_cost(shared_prefix_tokens, model, on)
    parallel = n * w
    staggered = w + (n - 1) * rd
    saving = parallel - staggered
    if not caches(model, shared_prefix_tokens):
        return parallel, staggered, Decision(
            False, 0.0,
            f"prefix is {shared_prefix_tokens:,} tok, under {model}'s "
            f"{cache_min(model):,}-token cache minimum: nothing caches either way",
        )
    if n < 2 or saving <= 0:
        return parallel, staggered, Decision(
            False, saving, "single call; nothing to stagger")
    if not serialise_first:
        return parallel, staggered, Decision(
            True, saving,
            f"stagger the fan-out: {n} simultaneous calls each rewrite the "
            f"{shared_prefix_tokens:,}-token prefix (${parallel:.4f}); waiting for "
            f"the first to start streaming makes the rest cache reads (${staggered:.4f})",
        )
    return parallel, staggered, Decision(
        True, saving, f"staggered fan-out saves ${saving:.4f} over {n} cold writes")


def effort_saving(
    out_tokens: int,
    model: str,
    *,
    from_effort: str = "high",
    to_effort: str = "medium",
    remaining_turns: int = 0,
    mult: dict[str, float] | None = None,
    on: date | None = None,
) -> tuple[float, Decision]:
    """USD saved by lowering reasoning effort, including downstream re-reads.

    Effort is the only lever that reduces output volume without changing the
    model, so it keeps the cache warm -- unlike a downgrade. And because output
    is re-read on every later turn, the saving carries the same debt multiple as
    terseness does.
    """
    mult = mult or EFFORT_OUTPUT_MULT
    if from_effort not in mult or to_effort not in mult:
        raise ValueError(f"unknown effort level; known: {sorted(mult)}")
    r = rate(model, on)
    delta = out_tokens * (mult[from_effort] - mult[to_effort])
    gen = delta * r.out / M
    reread = delta * r.inp * CACHE_READ_MULT * max(0, remaining_turns) / M
    total = gen + reread
    if delta <= 0:
        return total, Decision(
            False, total, f"{to_effort} is not cheaper than {from_effort}")
    return total, Decision(
        True, total,
        f"effort {from_effort}->{to_effort} cuts ~{delta:,.0f} output tok: "
        f"${gen:.4f} generation + ${reread:.4f} downstream re-reads, "
        f"with no cache invalidation (same model)",
    )


def batch_saving(cost: float) -> float:
    """USD saved by moving a workload to the Batch API (50% off, async)."""
    return cost * (1.0 - BATCH_MULT)


def marginal_turn_cost(context_tokens: int, out_tokens: int, model: str,
                       on: date | None = None) -> float:
    """What one more turn costs at the current context size. The number that
    should be on screen when deciding whether to keep going."""
    r = rate(model, on)
    return (context_tokens * r.inp * CACHE_READ_MULT + out_tokens * r.out) / M
