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

from adder.core.trace import _ordered
from adder.pricing.registry import (
    UnknownModelError,
    UnpricedModelError,
    provider_for,
    resolve,
)

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
    # Write spend split the same way. The TTL recommendation has to price
    # the 5m population on its own; a blended $/token drawn from both
    # buckets charges 5m tokens partly at the 1h premium they are not
    # paying, which biases the answer against the switch precisely on the
    # workloads already using some 1h.
    ttl_write_cost: dict[str, float] = field(
        default_factory=lambda: {"5m": 0.0, "1h": 0.0})

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
    prov = provider_for(turn.model)
    lived = prov.ttl_for(turn.ttl)
    if lived is not None and gap > lived:
        # Recoverable only if this provider offers a longer TTL that would have
        # spanned the gap. On a provider with a single, non-selectable TTL --
        # which is every automatic-caching provider -- there is no longer one to
        # move to, so the expiry is unfixable by configuration no matter how
        # short the gap was. Calling it fixable there would recommend a setting
        # that does not exist.
        longest = max((prov.ttl_seconds or {}).values(), default=lived)
        return EXPIRY_FIXABLE if gap <= longest and longest > lived else EXPIRY_UNFIXABLE
    return "growth"


def analyse(sessions, on: date | None = None) -> CacheReport:
    """Measure cache spend and attribute every large rebuild to a cause."""
    rep = CacheReport()
    for s in sessions.values():
        # Spend is summed over every turn; *misses* are walked per chain.
        for t in s.turns:
            r = t.rates(on)
            rep.n_turns += 1
            rep.read_tokens += t.cache_read
            rep.write_tokens += t.cache_write
            rep.read_cost += t.cache_read * r.cache_read / M
            rep.write_cost += t.cache_write * r.cache_write / M
            rep.ttl_tokens[t.ttl] = rep.ttl_tokens.get(t.ttl, 0) + t.cache_write
            rep.ttl_write_cost[t.ttl] = (rep.ttl_write_cost.get(t.ttl, 0.0)
                                         + t.cache_write * r.cache_write / M)

        # Main chain and sidechain walked separately, and each skipped past its
        # OWN first turn. Walking the combined list made the boundary between
        # the two look like a cache event it is not: a subagent opens a fresh
        # context on a cheaper model, so the turn after a main-chain turn is a
        # different model writing its whole prefix -- which `_classify` read as
        # a `model switch`, one of the two RECOVERABLE causes, and whose stated
        # fix is *"delegate to a subagent instead of switching the main loop"*.
        # The report was charging the reader for having taken its own advice.
        # `Session.cache_misses` and `measured_read_mult` both split the chains
        # for the same reason.
        for chain in (s.main_turns, [t for t in s.turns if t.sidechain]):
            prev = None
            for i, t in enumerate(chain):
                if i and t.cache_write > t.cache_read and t.cache_write >= MIN_MISS_TOKENS:
                    gap = 0.0
                    if prev is not None and prev.when and t.when:
                        gap = (_ordered(t.when) - _ordered(prev.when)).total_seconds()
                    cause = _classify(prev, t, gap)
                    # Waste is the premium over what a hit would have cost.
                    # Under automatic caching a rebuild is billed as plain
                    # input, so the waste is real but smaller: input minus the
                    # read rate, not a 1.25x premium minus it.
                    r = t.rates(on)
                    waste = t.cache_write * max(0.0, r.cache_write - r.cache_read) / M
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

    # TTL is only a lever where the provider sells more than one. Under
    # automatic caching there is exactly one lifetime and it is not selectable,
    # so the honest answer is "this setting does not exist for you" rather than
    # a dollar figure for a change nobody can make.
    choosable = _models_with_selectable_ttl(sessions)
    if not choosable:
        prov = _dominant_provider(sessions)
        return "", 0.0, (
            f"{prov.name} caching is {prov.cache_style}: the TTL is not "
            f"selectable, so idle gaps are a session-boundary problem rather "
            f"than a cache-configuration one")

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
    # The rate the 5m tokens actually paid, not the blend across both TTLs.
    r_avg = (rep.ttl_write_cost.get("5m", 0.0) / tokens_5m) if tokens_5m else 0.0
    ref = resolve(choosable[0])
    inp = ref.rate(on).inp or 1.0
    premium = (ref.cache_write_rate("1h", on) / inp) / max(
        1e-12, ref.cache_write_rate("5m", on) / inp)
    extra_write = tokens_5m * r_avg * (premium - 1.0)
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


def _models_in(sessions) -> list[str]:
    seen: dict[str, None] = {}
    for s in sessions.values():
        for t in s.turns:
            seen.setdefault(t.model, None)
    return list(seen)


def _models_with_selectable_ttl(sessions) -> list[str]:
    """Models in this workload whose provider actually sells a TTL choice."""
    out = []
    for m in _models_in(sessions):
        prov = provider_for(m)
        if prov.caches and len(prov.ttl_seconds or {}) > 1:
            out.append(m)
    return out


def _dominant_provider(sessions):
    """The provider carrying the most turns. Reports name one, not a set."""
    tally: dict[str, int] = {}
    for s in sessions.values():
        for t in s.turns:
            tally[t.model] = tally.get(t.model, 0) + 1
    if not tally:
        return provider_for("")
    return provider_for(max(tally, key=lambda m: tally[m]))


def _read_label(sessions, on: date | None = None) -> str:
    """`0.10x`, or `full input rate (no cache)`.

    The multiplier used to be printed as a constant, which made every report
    claim Anthropic's economics regardless of what produced the transcript.
    """
    mults = set()
    for m in _models_in(sessions):
        try:
            spec = resolve(m)
            inp = spec.rate(on).inp
        except (UnknownModelError, UnpricedModelError):
            continue
        if not inp:
            continue
        mults.add(round(spec.cache_read_rate(on) / inp, 2))
    if not mults:
        return "?"
    if mults == {1.0}:
        return "full input rate (no cache)"
    return "/".join(f"{x:.2f}x" for x in sorted(mults))


def _write_label(sessions, on: date | None = None) -> str:
    mults = set()
    for m in _models_in(sessions):
        try:
            spec = resolve(m)
            inp = spec.rate(on).inp
        except (UnknownModelError, UnpricedModelError):
            continue
        if not inp:
            continue
        for ttl in (spec.provider.ttl_seconds or {"": None}):
            mults.add(round(spec.cache_write_rate(ttl or None, on) / inp, 2))
    if not mults:
        return "?"
    if mults == {1.0}:
        return "input rate (no write premium)"
    return "/".join(f"{x:.2f}x" for x in sorted(mults))


def report(sessions, on: date | None = None) -> str:
    rep = analyse(sessions, on)
    if not rep.n_turns:
        return "  No priced turns to analyse."
    lines = ["  Cache efficiency", ""]
    lines.append(f"  cache reads   ${rep.read_cost:>9,.0f}   {rep.read_tokens:>14,} tok  "
                 f"@{_read_label(sessions, on)}")
    lines.append(f"  cache writes  ${rep.write_cost:>9,.0f}   {rep.write_tokens:>14,} tok  "
                 f"@{_write_label(sessions, on)}")
    lines.append(f"  hit rate      {rep.hit_rate:>9.1%}   of cacheable input tokens served from cache")
    # Labels come from the data, not from Anthropic's menu. A workload with no
    # cache writes at all, or one on a provider whose single TTL is called
    # something else, used to render as "0% 5m 0% 1h" -- two numbers about
    # settings that do not exist, and no mention of the label that does.
    mix = {k: v for k, v in rep.ttl_tokens.items() if v}
    tot = sum(mix.values())
    if not tot:
        lines.append("  TTL mix       {:>8}  no cache writes to attribute".format("--"))
    else:
        parts = "   ".join(f"{100 * v / tot:.0f}% {k}"
                           for k, v in sorted(mix.items(), key=lambda kv: -kv[1]))
        lines.append(f"  TTL mix       {parts:>8}")

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
    import json

    from adder.core.filters import add_arguments as add_window
    from adder.core.filters import load as load_window

    ap = argparse.ArgumentParser(prog="adder cache",
                                 description="Measure cache efficiency and rebuild waste.")
    add_window(ap)
    ap.add_argument("--json", action="store_true", help="machine-readable")
    a = ap.parse_args(argv)

    sessions, window = load_window(a)

    if a.json:
        rep = analyse(sessions)
        ttl, saving, why = ttl_recommendation(sessions)
        print(json.dumps({
            "hit_rate": round(rep.hit_rate, 5),
            "waste": round(rep.waste, 4),
            "recoverable": round(rep.recoverable, 4),
            "misses": len(rep.misses),
            "by_cause": {c: {"misses": n, "cost": round(v, 4)}
                         for c, (n, v) in rep.by_cause().items()},
            "recommended_ttl": ttl,
            "ttl_saving": round(saving, 4),
            "ttl_reason": why,
            "filter": window.describe(),
        }))
        return 0

    print()
    print(report(sessions))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
