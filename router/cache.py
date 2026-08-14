"""Cache efficiency: the measured waste nobody itemises.

A cached prefix read costs 0.10x the input rate. Rebuilding it costs 1.25x (5m
TTL) or 2.00x (1h). So every avoidable cache invalidation is a **12.5x** price
increase on the whole context for that turn -- and because contexts here run to
half a million tokens, a single miss can cost more than a hundred ordinary turns
of output.

This module finds those turns in the transcripts and attributes each one to a
cause, because the causes have different fixes and different feasibility:

    model switch      a mid-session model change; caches are model-scoped, so
                      the entire prefix is rebuilt. Avoidable: delegate to a
                      subagent instead of switching the main loop.
    idle expiry       the gap to the previous turn exceeded the TTL. Avoidable
                      ONLY when a longer TTL would have covered the gap: a 5m
                      cache idle for 20 minutes is fixable by the 1h TTL, one
                      idle for two hours is not, because 1h is the longest TTL
                      offered. The two are counted separately.
    post-compaction   the prefix genuinely changed. Not avoidable, and not a bug.
    growth            a large legitimate addition to context. Not a miss.

Only the first two are recoverable, so only those are reported as a saving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .prices import CACHE_READ_MULT, CACHE_WRITE_MULT, TTL_SECONDS, rate

M = 1_000_000.0

# A rebuild smaller than this is noise, not a lost cache.
MIN_MISS_TOKENS = 10_000

EXPIRY_FIXABLE = "idle expiry (1h TTL would cover)"
EXPIRY_UNFIXABLE = "idle expiry (beyond any TTL)"
RECOVERABLE = ("model switch", EXPIRY_FIXABLE)

CAUSES = ("model switch", EXPIRY_FIXABLE, EXPIRY_UNFIXABLE, "post-compaction", "growth")


@dataclass
class Miss:
    session: str
    model: str
    tokens: int
    ttl: str
    cause: str
    waste: float          # USD paid over what a cache read would have cost
    gap: float = 0.0


@dataclass
class CacheReport:
    read_cost: float = 0.0
    write_cost: float = 0.0
    total_cost: float = 0.0
    misses: list[Miss] = field(default_factory=list)
    n_turns: int = 0
    ttl_tokens: dict[str, int] = field(default_factory=lambda: {"5m": 0, "1h": 0})

    @property
    def waste(self) -> float:
        return sum(m.waste for m in self.misses)

    @property
    def recoverable(self) -> float:
        """Waste from causes that a configuration change can actually remove."""
        return sum(m.waste for m in self.misses if m.cause in RECOVERABLE)

    def by_cause(self) -> dict[str, tuple[int, float]]:
        out: dict[str, tuple[int, float]] = {}
        for m in self.misses:
            n, w = out.get(m.cause, (0, 0.0))
            out[m.cause] = (n + 1, w + m.waste)
        return dict(sorted(out.items(), key=lambda kv: -kv[1][1]))

    @property
    def hit_rate(self) -> float:
        """Share of cacheable input tokens actually served from cache."""
        tot = self.read_tokens + self.write_tokens
        return self.read_tokens / tot if tot else 0.0

    read_tokens: int = 0
    write_tokens: int = 0


def _classify(prev, turn, gap: float) -> str:
    if prev is not None and prev.model != turn.model:
        return "model switch"
    if prev is not None and turn.context < prev.context * 0.6:
        return "post-compaction"
    if gap > TTL_SECONDS.get(turn.ttl, 300):
        # Recoverable only if a longer TTL exists that would have spanned the gap.
        return EXPIRY_FIXABLE if gap <= TTL_SECONDS["1h"] else EXPIRY_UNFIXABLE
    return "growth"


def analyse(sessions, on: date | None = None) -> CacheReport:
    """Measure cache spend and attribute every large rebuild to a cause."""
    rep = CacheReport()
    for s in sessions.values():
        prev = None
        for i, t in enumerate(s.turns):
            r = rate(t.model, on, speed=t.speed).inp
            rep.n_turns += 1
            rep.read_tokens += t.cache_read
            rep.write_tokens += t.cache_write
            rep.read_cost += t.cache_read * r * CACHE_READ_MULT / M
            rep.write_cost += t.cache_write * r * CACHE_WRITE_MULT[t.ttl] / M
            rep.ttl_tokens[t.ttl] = rep.ttl_tokens.get(t.ttl, 0) + t.cache_write

            if i and t.cache_write > t.cache_read and t.cache_write >= MIN_MISS_TOKENS:
                gap = 0.0
                if prev is not None and prev.when and t.when:
                    gap = (t.when - prev.when).total_seconds()
                cause = _classify(prev, t, gap)
                waste = t.cache_write * r * (CACHE_WRITE_MULT[t.ttl] - CACHE_READ_MULT) / M
                rep.misses.append(
                    Miss(s.id, t.model, t.cache_write, t.ttl, cause, waste, gap))
            prev = t
    rep.total_cost = rep.read_cost + rep.write_cost
    return rep


def ttl_recommendation(sessions, on: date | None = None) -> tuple[str, float, str]:
    """Should this workload use the 1h cache TTL? Answers from measured gaps.

    Mix-aware: a workload already writing at 1h cannot save by "switching to
    1h", and gaps longer than an hour are not fixable by any TTL. Only tokens
    currently written at 5m, whose gap a 1h cache would have covered, count.
    """
    gaps = [g for s in sessions.values() for g in s.gaps()]
    if not gaps:
        return "5m", 0.0, "no timestamps to measure idle gaps from"

    rep = analyse(sessions, on)
    already_1h = rep.ttl_tokens.get("1h", 0)
    tokens_5m = rep.ttl_tokens.get("5m", 0)
    total_written = already_1h + tokens_5m
    share_1h = already_1h / total_written if total_written else 0.0

    # Only 5m-written rebuilds that a 1h cache would have covered are fixable.
    fixable = sum(m.waste for m in rep.misses
                  if m.cause == EXPIRY_FIXABLE and m.ttl == "5m")
    beyond = sum(m.waste for m in rep.misses if m.cause == EXPIRY_UNFIXABLE)
    # Cost of moving the remaining 5m writes to the 1h premium.
    r_avg = (rep.write_cost / total_written) if total_written else 0.0
    extra_write = (
        tokens_5m * r_avg * (CACHE_WRITE_MULT["1h"] / CACHE_WRITE_MULT["5m"] - 1.0)
    )
    net = fixable - extra_write

    if share_1h > 0.9:
        return "1h", 0.0, (
            f"{share_1h:.0%} of cache writes already use the 1h TTL. The remaining "
            f"${beyond:,.0f} of expiry waste comes from gaps longer than an hour, "
            f"which no TTL setting covers -- that is a session-boundary problem, "
            f"not a cache-configuration one")
    if net > 0:
        return "1h", net, (
            f"{tokens_5m:,} tok are written at the 5m TTL and ${fixable:,.0f} of "
            f"rebuilds would have been covered by a 1h cache; the write premium "
            f"adds ${extra_write:,.0f}, for a net ${net:,.0f}")
    return "5m", -net, (
        f"only ${fixable:,.0f} of rebuilds would be covered by a 1h cache, while "
        f"its write premium would add ${extra_write:,.0f} -- keep the 5m default")


def report(sessions, on: date | None = None) -> str:
    rep = analyse(sessions, on)
    if not rep.n_turns:
        return "  No priced turns to analyse."
    lines = ["  Cache efficiency", ""]
    lines.append(f"  cache reads   ${rep.read_cost:>9,.0f}   {rep.read_tokens:>14,} tok  @0.10x")
    lines.append(f"  cache writes  ${rep.write_cost:>9,.0f}   {rep.write_tokens:>14,} tok  @1.25x/2.00x")
    lines.append(f"  hit rate      {rep.hit_rate:>9.1%}   of cacheable input tokens served from cache")
    mix = rep.ttl_tokens
    tot = sum(mix.values()) or 1
    lines.append(f"  TTL mix       {100*mix.get('5m',0)/tot:>8.0f}% 5m   {100*mix.get('1h',0)/tot:.0f}% 1h")

    lines.append("")
    if not rep.misses:
        lines.append("  No large cache rebuilds detected.")
        return "\n".join(lines)

    lines.append(f"  {len(rep.misses):,} large rebuilds cost ${rep.waste:,.0f} over what a cache read would have:")
    lines.append(f"  {'cause':<34}{'turns':>7}{'wasted':>11}{'':>3}recoverable")
    for cause, (n, w) in rep.by_cause().items():
        rec = "yes" if cause in RECOVERABLE else "no"
        lines.append(f"  {cause:<34}{n:>7,}{w:>10,.0f}{'':>4}{rec}")
    lines.append("")
    lines.append(f"  Recoverable: ${rep.recoverable:,.0f}")

    ttl, _saving, why = ttl_recommendation(sessions, on)
    lines.append("")
    lines.append(f"  TTL recommendation: {ttl}")
    lines.append(f"    {why}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .trace import DEFAULT_ROOT, load_sessions

    ap = argparse.ArgumentParser(prog="router.cache",
                                 description="Measure cache efficiency and rebuild waste.")
    ap.add_argument("root", nargs="?", default=str(DEFAULT_ROOT))
    a = ap.parse_args(argv)
    print()
    print(report(load_sessions(a.root, use_cache=True)))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
