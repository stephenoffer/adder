"""Quantify what each lever is worth, with assumptions stated per estimate.

Confidence tiers, so a reader can tell measurement from modelling:
  MEASURED  - recomputed from recorded token counts; no assumptions.
  ATTRIBUTED- exact arithmetic on recorded data, attributing cost to a cause.
  MODELLED  - depends on a stated counterfactual assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .cost import turn_cost
from .prices import CACHE_READ_MULT, rate
from .trace import DEFAULT_ROOT, Session, load_sessions

M = 1_000_000.0


@dataclass
class Estimate:
    lever: str
    saving: float
    confidence: str          # MEASURED | ATTRIBUTED | MODELLED
    basis: str
    assumptions: str = ""

    def line(self, total: float) -> str:
        pct = f"{100 * self.saving / total:.1f}%" if total else "-"
        return f"${self.saving:>9,.2f}  {pct:>6}  [{self.confidence:<10}] {self.lever}"


def explore_savings(sessions: dict[str, Session], cheap: str = "claude-haiku-4-5") -> Estimate:
    """MEASURED: rerun existing subagent turns at a cheap model's rates.

    Subagents run in fresh contexts, so there is no cache to lose. This is the
    one lever with no counterfactual: same tokens, different rate card.
    """
    actual = saved = 0.0
    n = 0
    for s in sessions.values():
        for t in s.turns:
            if not t.sidechain or rate(t.model).inp <= rate(cheap).inp:
                continue
            actual += t.cost()
            saved += turn_cost(
                cheap,
                uncached_in=t.uncached_in,
                cache_read=t.cache_read,
                cache_write=t.cache_write,
                out=t.out,
            )
            n += 1
    return Estimate(
        f"Run subagents/Explore on Haiku ({n:,} existing subagent turns)",
        actual - saved,
        "MEASURED",
        f"recomputed {n:,} recorded subagent turns at Haiku rates",
        "assumes Haiku handles the same read-only work (it is the documented default for Explore)",
    )


def _session_read_cost(sess: Session) -> float:
    """The actual cache-read dollars this session spent. Measured, not modelled."""
    return sum(
        t.cache_read * rate(t.model).inp * CACHE_READ_MULT / M for t in sess.turns
    )


def _admissions(sess: Session) -> list[tuple[float, int]]:
    """(tokens_admitted, remaining_turns) per turn, from context growth.

    NOT from `cache_write`: Claude Code refreshes cache segments, so summing
    cache_creation_input_tokens overcounts admitted content ~5x (measured).

    Growth alone is also not enough. These sessions hit the 1M ceiling, so
    compaction shrinks the context and re-growth after a compaction is not new
    admission. Reconstructed totals therefore overshoot the real bill (measured:
    $7.8K reconstructed vs $5.6K actual). So this returns raw *weights* only;
    callers normalise them against the session's measured read cost.
    """
    out: list[tuple[float, int]] = []
    n = len(sess.turns)
    prev = 0
    for i, t in enumerate(sess.turns):
        admitted = float(t.context) if i == 0 else float(max(0, t.context - prev))
        prev = t.context
        out.append((admitted, n - i - 1))
    return out


def _attributed(sess: Session) -> list[tuple[float, float, int]]:
    """(dollars, tokens_admitted, remaining_turns), summing exactly to the
    session's measured cache-read spend.

    Each admission's share is proportional to size x longevity -- the actual
    causal driver -- and the total is pinned to what was really spent, so
    attribution can never exceed reality.
    """
    adm = _admissions(sess)
    weights = [a * r for a, r in adm]
    total_w = sum(weights)
    actual = _session_read_cost(sess)
    if total_w <= 0:
        return [(0.0, a, r) for a, r in adm]
    return [(actual * w / total_w, a, r) for w, (a, r) in zip(weights, adm)]


def amortization_profile(sessions: dict[str, Session]) -> tuple[Estimate, list[tuple[str, float, int]]]:
    """ATTRIBUTED: the measured cache-read bill, decomposed by what admitted it."""
    total = 0.0
    worst: list[tuple[str, float, int]] = []
    for s in sessions.values():
        c = sum(d for d, _, _ in _attributed(s))
        total += c
        worst.append((s.project, c, s.n_turns))
    worst.sort(key=lambda x: -x[1])

    actual = sum(_session_read_cost(s) for s in sessions.values())
    if actual and total > actual * 1.001:
        raise AssertionError(
            f"attribution ${total:,.2f} exceeds measured cache-read spend ${actual:,.2f}"
        )
    return (
        Estimate(
            "Cache-read spend, attributed to the content that caused it",
            total,
            "ATTRIBUTED",
            "measured cache-read dollars, split by admission size x remaining turns",
            "this is the pool the placement lever draws from, not a saving by itself",
        ),
        worst[:5],
    )


def delegation_savings(
    sessions: dict[str, Session],
    *,
    delegable_fraction: float = 0.30,
    compression: float = 10.0,
    sub_model: str = "claude-haiku-4-5",
) -> Estimate:
    """MODELLED: route a fraction of admitted content through a cheap subagent.

    Bounded by the measured bill: a delegated admission still costs 1/compression
    of its amortised share, plus one cheap fresh read. Conservative defaults --
    30% delegable at 10:1 -- because both are guesses, not measurements.
    """
    saving = 0.0
    for s in sessions.values():
        r_sub = rate(sub_model)
        for dollars, tokens, _ in _attributed(s):
            moved_cost = dollars * delegable_fraction
            if moved_cost <= 0:
                continue
            kept = moved_cost / compression                       # summary still amortised
            read_once = tokens * delegable_fraction * r_sub.inp / M
            saving += max(0.0, moved_cost - kept - read_once)
    return Estimate(
        f"Delegate reads to subagents ({delegable_fraction:.0%} of admitted content, "
        f"{compression:.0f}:1 compression)",
        saving,
        "MODELLED",
        "measured cache-read spend re-priced as delegated reads",
        f"assumes {delegable_fraction:.0%} of admitted content is delegable and "
        f"compresses {compression:.0f}:1; both are estimates, not measurements",
    )


def split_savings(sessions: dict[str, Session], *, max_turns: int = 300) -> Estimate:
    """MODELLED: cap session length, restarting context at a floor.

    Context cost over a session is roughly the sum of per-turn context. Capping
    length resets that growth. Assumes a restarted session re-establishes a
    baseline context equal to the session's own minimum observed context.
    """
    saving = 0.0
    for s in sessions.values():
        if s.n_turns <= max_turns:
            continue
        r = rate(s.turns[0].model).inp * CACHE_READ_MULT
        actual = sum(t.context for t in s.turns) * r / M
        floor = min(t.context for t in s.turns)
        # After each cap, context restarts at `floor` and regrows at the observed slope.
        slope = max(0.0, (s.peak_context - floor) / max(1, s.n_turns))
        simulated = 0.0
        for i in range(s.n_turns):
            simulated += (floor + slope * (i % max_turns)) * r / M
        saving += max(0.0, actual - simulated)
    return Estimate(
        f"Split sessions longer than {max_turns} turns",
        saving,
        "MODELLED",
        "re-simulates context growth with periodic resets",
        "assumes work is separable at turn boundaries and a restart re-reads only "
        "the session's minimum observed context; real handoffs cost more",
    )


def model_routing_savings(sessions: dict[str, Session]) -> Estimate:
    """MODELLED: per-turn downgrade, but only where the cache gate allows it.

    This is the original ask. It is included honestly: the gate refuses on warm
    contexts, so the reachable saving is small.
    """
    from .cost import switch_is_profitable

    saving = 0.0
    eligible = 0
    for s in sessions.values():
        for t in s.turns:
            if t.model.startswith("claude-haiku"):
                continue
            d = switch_is_profitable("claude-opus-5", "claude-haiku-4-5", t.context, t.out)
            if d:
                saving += d.saving
                eligible += 1
    return Estimate(
        f"Per-turn model downgrade where the cache gate permits ({eligible:,} turns)",
        saving,
        "MODELLED",
        "cache-gated switch applied per recorded turn",
        "assumes a cheaper model would have produced acceptable output on those turns "
        "(unverifiable from transcripts alone)",
    )


def report(root: Path | str = DEFAULT_ROOT, *, max_turns: int = 300) -> None:
    sessions = load_sessions(root)
    total = sum(s.cost for s in sessions.values())
    if not total:
        print(f"No priced turns found under {root}")
        return

    pool, worst = amortization_profile(sessions)
    estimates = [
        explore_savings(sessions),
        delegation_savings(sessions),
        split_savings(sessions, max_turns=max_turns),
        model_routing_savings(sessions),
    ]

    print(f"\n  Measured spend: ${total:,.2f} across {len(sessions)} sessions\n")
    print(f"  {'saving':>10} {'of tot':>6}  {'confidence':<12} lever")
    print("  " + "-" * 78)
    for e in sorted(estimates, key=lambda e: -e.saving):
        print("  " + e.line(total))

    overlapping = [e for e in estimates if e.lever.startswith(("Split", "Delegate"))]
    if len(overlapping) == 2:
        naive = sum(e.saving for e in overlapping)
        print(f"\n  NOT ADDITIVE: Split (${overlapping[0].saving:,.0f}) and Delegate "
              f"(${overlapping[1].saving:,.0f}) both draw from the same "
              f"${pool.saving:,.0f} cache-read pool.")
        print(f"  Naive sum ${naive:,.0f} would double-count; combined realistic ceiling "
              f"is the pool itself, ${pool.saving:,.0f}.")

    print(f"\n  Context pool: {pool.line(total)}")
    print(f"    {pool.basis}")
    print("\n  Sessions whose admitted content cost the most:")
    for proj, cost, n in worst:
        print(f"    ${cost:>8,.2f}  {n:>5,} turns  {proj[:52]}")

    print("\n  Assumptions behind the modelled figures:")
    for e in estimates:
        if e.confidence == "MODELLED" and e.assumptions:
            print(f"    - {e.lever.split('(')[0].strip()}: {e.assumptions}")
    print()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(prog="router.savings")
    ap.add_argument("root", nargs="?", default=str(DEFAULT_ROOT))
    ap.add_argument("--max-turns", type=int, default=300)
    a = ap.parse_args()
    report(a.root, max_turns=a.max_turns)
