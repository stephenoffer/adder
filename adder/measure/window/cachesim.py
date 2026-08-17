"""Simulate the prefix cache instead of only observing the bill it produced.

What `adder cache` cannot answer
--------------------------------
`adder cache` reads the billing fields and reports what the cache did: a hit
rate, and the rebuilds that were avoidable. That is a measurement and it is the
right one, but it is stuck describing the cache you were given. It cannot
answer the questions that decide how much a workload *could* cost:

* If every session in a project shared one prefix cache instead of each holding
  its own copy of the same system prompt, tool schemas and CLAUDE.md, how much
  of the input volume would be served from cache?
* That sharing needs memory. How much? What does the hit rate do as capacity
  falls -- is it a cliff or a slope?
* Blocks are matched whole. What does the block size cost at the boundaries?
* Entries expire. How much of the miss volume is capacity pressure and how much
  is just idle time?

Those are cache-design questions, and the only way to answer them from a
transcript is to replay the workload against a simulated cache. That is what
this module does, and calling it a simulation rather than a measurement is the
whole of its honesty.

How the cache is modelled
-------------------------
A radix tree over fixed-size blocks, which is what production serving stacks
converge on: split the token sequence into blocks, key each block by the
content of every block up to and including it, and reuse the longest prefix
already resident. Matching is prefix-anchored -- block k is only usable if
blocks 0..k-1 matched -- because a transformer's key/value state for a position
depends on everything before it. A cache that matched blocks out of order would
report a wonderful hit rate and be wrong.

Eviction is LRU over blocks, bounded by a capacity, plus an idle TTL. Both are
knobs because both are real: capacity is what you rent, TTL is what the
provider gives you.

The one assumption, stated plainly
----------------------------------
Transcripts record *how many* tokens were in context, never *which* ones. So
token identity has to be modelled, and the model is this: sessions in the same
project share an identical opening prefix, sized at the smallest base context
observed in that project, and everything above that line is unique to its
session.

That is an approximation in both directions. It is optimistic in that two
sessions' opening prompts are not byte-identical -- a timestamp or a working
directory differs, and in a real block-hashed cache one differing byte in block
0 discards the entire shared prefix. It is pessimistic in that real sessions
share far more than their opening: the same files get read into two sessions
constantly, and none of that shows up as shared here.

The honest reading is therefore: **this is the value of prefix sharing if the
opening prefix is byte-stable, and it is a lower bound on total sharing.** Every
number this module prints is labelled SIMULATED for that reason, and none of
them are wired into a saving that another report claims as measured.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime

from adder.pricing.registry import provider_for
from adder.util import render
from adder.util.stats import share

M = 1_000_000.0

# Production stacks block at 16 to 256 tokens. 16 is the common default and the
# one that wastes least at a boundary; the sweep exists because the answer is
# workload-dependent and this workload is not the one those defaults were
# tuned on.
DEFAULT_BLOCK = 16
BLOCK_SWEEP: tuple[int, ...] = (16, 64, 256, 1024)

# Capacities to sweep, in tokens. The top of the range is "everything fits",
# which is the number the sharing argument is usually made with and almost
# never the number anyone would pay for.
CAPACITY_SWEEP: tuple[int, ...] = (
    1_000_000, 4_000_000, 16_000_000, 64_000_000, 256_000_000)

# Blocks idle longer than this are gone. Matches the shorter of the two TTLs
# the provider offers, because that is the default a workload gets.
DEFAULT_TTL_S = 300.0


@dataclass(frozen=True)
class Request:
    """One turn's demand on the cache: a prefix of `tokens`, at a moment."""

    session: str
    project: str
    model: str
    tokens: int
    shared_prefix: int
    when: float
    cost_read: float          # USD per token, cached read
    cost_write: float         # USD per token, cache write
    # Which subagent run this turn belongs to, empty on the main chain. A
    # sidechain turn carries the PARENT's session id, so without this a
    # subagent's 20K brief and the parent's 700K context share one namespace
    # and the subagent's prefix scores as hits against blocks it never wrote --
    # an optimistic error, in the one module whose whole job is to avoid them.
    agent: str = ""


def _spans(req: Request, block_size: int) -> tuple[tuple[str, int], tuple[str, int]]:
    """This request's prefix as two contiguous block ranges: shared, then private.

    Blocks below `shared_prefix` are namespaced to the project, so two sessions
    in the same project produce the same ids and can hit each other's entries.
    Above that line they are namespaced to the session and can never collide,
    which is the conservative half of the assumption.

    The boundary block is the interesting one: when `shared_prefix` is not a
    multiple of the block size, the block straddling it contains bytes from
    both regions, so it cannot be shared. Flooring assigns it to the session,
    which is what a content hash would do.

    The trailing partial block is **not** stored. Real caches key a block by the
    content of every token in it, so a half-filled block has no stable key and
    is recomputed next time. Rounding it up instead -- which this module did at
    first -- credits the next request with up to a full block of tokens it never
    cached, and the error scales with the block size. It showed up as a 1024-token
    block reporting a *higher* hit rate than a 16-token one, which is backwards:
    bigger blocks match less often and waste more at the boundary. Floor, not
    ceiling, and the remainder is always a miss.

    Returning ranges rather than a list of block ids is not a micro-optimisation.
    A 500K-token context at a 16-token block is 31,250 blocks, and a real
    workload is tens of thousands of turns; enumerating every block per request
    is billions of operations for a report that has to finish in a second. It is
    also unnecessary, because a request always touches a *prefix* -- the blocks
    it wants are contiguous by construction, so two `(namespace, count)` pairs
    describe them exactly.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    n_blocks = req.tokens // block_size
    shared_blocks = min(n_blocks, req.shared_prefix // block_size)
    return (
        (f"p:{req.project}:{req.model}", shared_blocks),
        (f"s:{req.session}:{req.agent}:{req.model}", n_blocks - shared_blocks),
    )


@dataclass
class SimResult:
    """What the simulated cache did, in tokens and in dollars."""

    block_size: int
    capacity_tokens: int
    hit_tokens: int = 0
    miss_tokens: int = 0
    write_tokens: int = 0
    evictions: int = 0
    expiries: int = 0
    capacity_misses: int = 0
    cold_misses: int = 0
    cost: float = 0.0
    peak_resident: int = 0
    requests: int = 0

    @property
    def hit_rate(self) -> float:
        return share(self.hit_tokens, self.hit_tokens + self.miss_tokens)

    @property
    def total_tokens(self) -> int:
        return self.hit_tokens + self.miss_tokens

    def to_json(self) -> dict:
        return {
            "block_size": self.block_size,
            "capacity_tokens": self.capacity_tokens,
            "requests": self.requests,
            "hit_rate": self.hit_rate,
            "hit_tokens": self.hit_tokens,
            "miss_tokens": self.miss_tokens,
            "cost_usd": self.cost,
            "evictions": self.evictions,
            "expiries": self.expiries,
            "capacity_misses": self.capacity_misses,
            "cold_misses": self.cold_misses,
            "peak_resident_tokens": self.peak_resident,
        }


def simulate(
    requests: list[Request],
    *,
    block_size: int = DEFAULT_BLOCK,
    capacity_tokens: int = CAPACITY_SWEEP[-1],
    ttl_s: float = DEFAULT_TTL_S,
) -> SimResult:
    """Replay `requests` against an LRU-bounded, TTL-expiring prefix cache.

    Prefix-anchored matching: the walk stops at the first block that is not
    resident, and every block after it is a miss even if an identical block
    happens to be in the cache from some other sequence. That is the rule the
    hardware imposes, and relaxing it is the single easiest way to write a
    cache simulator that reports a number nobody can reproduce.

    Misses are separated into cold (never seen) and capacity (seen, then
    evicted or expired), because they have different fixes: cold misses are the
    price of new work, capacity misses are a purchasing decision.
    """
    res = SimResult(block_size=block_size, capacity_tokens=capacity_tokens)
    if capacity_tokens <= 0:
        raise ValueError(f"capacity must be positive, got {capacity_tokens}")
    capacity_blocks = max(1, capacity_tokens // block_size)

    # namespace -> (blocks resident, last touched). Every block in a namespace
    # was touched by the same request at the same instant, so one timestamp per
    # namespace is exact rather than an approximation. OrderedDict keeps them
    # in LRU order, and eviction drops whole namespaces: for a prefix workload
    # "this session's context is resident or it is not" is the real granularity,
    # and evicting the middle of a prefix would leave a tail that can never be
    # matched anyway.
    resident: OrderedDict[str, tuple[int, float]] = OrderedDict()
    high_water: dict[str, int] = {}
    live_blocks = 0

    def _evict_to_fit() -> None:
        nonlocal live_blocks
        while live_blocks > capacity_blocks and resident:
            _, (blocks, _) = resident.popitem(last=False)
            live_blocks -= blocks
            res.evictions += 1

    for req in sorted(requests, key=lambda r: (r.when, r.session)):
        res.requests += 1
        (shared_ns, shared_blocks), (own_ns, own_blocks) = _spans(req, block_size)
        if req.tokens <= 0:
            continue

        # Expire before serving: an entry idle past the TTL is not there, and
        # counting it as a hit is the bug that makes a simulator optimistic.
        if ttl_s > 0:
            dead = [ns for ns, (_, seen) in resident.items() if req.when - seen > ttl_s]
            for ns in dead:
                blocks, _ = resident.pop(ns)
                live_blocks -= blocks
                res.expiries += 1

        # Prefix-anchored: the private range is only reachable if the whole
        # shared range matched first.
        matched = min(shared_blocks, resident.get(shared_ns, (0, 0.0))[0])
        if matched == shared_blocks:
            matched += min(own_blocks, resident.get(own_ns, (0, 0.0))[0])

        # Only whole blocks can hit; the trailing remainder is always a miss,
        # because a partial block was never stored.
        hit_tokens = min(matched * block_size, req.tokens)
        miss_tokens = req.tokens - hit_tokens
        res.hit_tokens += hit_tokens
        res.miss_tokens += miss_tokens
        res.write_tokens += miss_tokens
        res.cost += hit_tokens * req.cost_read + miss_tokens * req.cost_write

        # A missed block that was resident before is a capacity (or TTL) miss;
        # one never seen is the price of new work. The two have different
        # fixes, so they are counted apart.
        missed_blocks = (shared_blocks + own_blocks) - matched
        seen_before = max(0, min(missed_blocks,
                                 high_water.get(shared_ns, 0) + high_water.get(own_ns, 0)
                                 - matched))
        res.capacity_misses += seen_before
        res.cold_misses += missed_blocks - seen_before

        for ns, want in ((shared_ns, shared_blocks), (own_ns, own_blocks)):
            if want <= 0:
                continue
            held, _ = resident.get(ns, (0, 0.0))
            live_blocks += max(0, want - held)
            resident[ns] = (max(held, want), req.when)
            resident.move_to_end(ns)
            high_water[ns] = max(high_water.get(ns, 0), want)
        _evict_to_fit()
        res.peak_resident = max(res.peak_resident, live_blocks * block_size)
    return res


# --- turning sessions into requests -----------------------------------------

def _epoch(ts) -> float:
    if isinstance(ts, datetime):
        return ts.timestamp()
    return 0.0


def _ttl_label(model: str, ttl_s: float) -> str | None:
    """The provider TTL label closest to the simulated lifetime.

    The simulation sweeps a cache lifetime in seconds; pricing needs a label.
    Picking the longest label the simulated lifetime reaches keeps the two in
    step, and on a provider with one non-selectable TTL it always returns that
    one rather than a 5m/1h choice that provider does not offer.

    Getting this wrong has a signature failure: writes are not one price, so
    simulating an hour-long TTL while pricing writes at the five-minute rate
    shows a longer TTL as free. That is the single most tempting wrong
    conclusion this module can produce.
    """
    table = provider_for(model).ttl_seconds or {}
    if not table:
        return None
    reached = [k for k, secs in table.items() if ttl_s >= secs]
    if not reached:
        return min(table, key=lambda k: table[k])
    return max(reached, key=lambda k: table[k])


def requests_from(sessions: dict, *, on=None, ttl_s: float = DEFAULT_TTL_S) -> list[Request]:
    """One request per turn, with the project's shared prefix line attached.

    The shared line is the smallest base context observed across the project's
    sessions: the part every session in that project started with, and
    therefore the part that sharing could serve from one copy. Taking the
    minimum rather than the mean is deliberate -- it is the largest prefix that
    is certainly common to all of them.
    """
    by_project: dict[str, list] = {}
    for s in sessions.values():
        if s.turns:
            by_project.setdefault(s.project, []).append(s)

    shared: dict[str, int] = {}
    for project, group in by_project.items():
        shared[project] = min(s.base_context for s in group)

    out: list[Request] = []
    for s in sessions.values():
        base = shared.get(s.project, 0)
        for t in s.turns:
            ttl = _ttl_label(t.model, ttl_s)
            r = t.rates(on, ttl=ttl)
            ctx = t.context
            if ctx <= 0:
                continue
            # `base` is the MAIN chain's floor. A subagent's whole context is
            # routinely smaller than it, and `min(base, ctx)` then claimed the
            # subagent's entire prefix was identical to the project's shared
            # instruction floor -- 100% shared, by arithmetic rather than by
            # evidence. Nothing measured says what a subagent's prefix has in
            # common with the main chain's, so it is credited with none of it.
            shared_line = 0 if t.sidechain else min(base, ctx)
            out.append(Request(
                session=s.id,
                project=s.project,
                model=t.model,
                tokens=ctx,
                shared_prefix=shared_line,
                agent=t.agent_id,
                when=_epoch(t.when),
                cost_read=r.cache_read / M,
                cost_write=r.cache_write / M,
            ))
    return out


def measured_baseline(sessions: dict, *, on=None) -> tuple[float, float]:
    """`(hit_rate, cost)` as actually billed, to sit next to the simulation.

    Without this the simulated numbers have nothing to be read against, and a
    simulated 94% hit rate looks like an achievement next to a measured 99%.
    """
    hit = miss = 0
    cost = 0.0
    for s in sessions.values():
        for t in s.turns:
            r = t.rates(on)
            hit += t.cache_read
            miss += t.cache_write + t.uncached_in
            cost += (t.cache_read * r.cache_read
                     + t.cache_write * r.cache_write
                     + t.uncached_in * r.inp) / M
    return (share(hit, hit + miss), cost)


@dataclass
class Sweep:
    """The capacity/hit-rate curve, which is the actual deliverable."""

    by_capacity: list[SimResult] = field(default_factory=list)
    by_block: list[SimResult] = field(default_factory=list)
    measured_hit_rate: float = 0.0
    measured_cost: float = 0.0

    @property
    def knee(self) -> SimResult | None:
        """Smallest capacity within a point of the best hit rate it can reach.

        The number worth quoting: buying past the knee buys nothing, and every
        "just cache everything" argument is really a claim that the knee is
        further right than it is.
        """
        if not self.by_capacity:
            return None
        best = max(r.hit_rate for r in self.by_capacity)
        for r in sorted(self.by_capacity, key=lambda r: r.capacity_tokens):
            if r.hit_rate >= best - 0.01:
                return r
        return self.by_capacity[-1]

    def to_json(self) -> dict:
        return {
            "simulated": True,
            "measured": {"hit_rate": self.measured_hit_rate,
                         "cost_usd": self.measured_cost},
            "by_capacity": [r.to_json() for r in self.by_capacity],
            "by_block": [r.to_json() for r in self.by_block],
            "knee": self.knee.to_json() if self.knee else None,
        }


def sweep(
    sessions: dict,
    *,
    capacities: tuple[int, ...] = CAPACITY_SWEEP,
    blocks: tuple[int, ...] = BLOCK_SWEEP,
    ttl_s: float = DEFAULT_TTL_S,
    on=None,
) -> Sweep:
    reqs = requests_from(sessions, on=on, ttl_s=ttl_s)
    hit_rate, cost = measured_baseline(sessions, on=on)
    out = Sweep(measured_hit_rate=hit_rate, measured_cost=cost)
    if not reqs:
        return out
    for cap in capacities:
        out.by_capacity.append(
            simulate(reqs, block_size=DEFAULT_BLOCK, capacity_tokens=cap, ttl_s=ttl_s))
    biggest = max(capacities)
    for block in blocks:
        out.by_block.append(
            simulate(reqs, block_size=block, capacity_tokens=biggest, ttl_s=ttl_s))
    return out


# --- report ----------------------------------------------------------------

def report(sessions: dict, *, ttl_s: float = DEFAULT_TTL_S, on=None) -> str:
    sw = sweep(sessions, ttl_s=ttl_s, on=on)
    out: list[str] = []
    out += render.heading("prefix cache, simulated — what sharing would be worth",
                          rule="=")
    if not sw.by_capacity:
        out.append("  No turns with a context to replay.")
        return "\n".join(out)

    out.append(render.kv("measured hit rate", render.pct(sw.measured_hit_rate)))
    out.append(render.kv("measured input cost", render.money(sw.measured_cost)))
    out.append("")
    out += render.wrap(
        "SIMULATED below this line. Sessions in a project are assumed to share "
        "an identical opening prefix and nothing else, so these are the value "
        "of prefix sharing alone and a lower bound on total reuse.")

    out.append("")
    out += render.heading(f"capacity (block size {DEFAULT_BLOCK})")
    out += render.table(
        [[render.tokens(r.capacity_tokens), render.pct(r.hit_rate),
          render.money(r.cost), f"{r.capacity_misses:,}", f"{r.evictions:,}"]
         for r in sw.by_capacity],
        ["capacity", "hit rate", "input cost", "capacity misses", "evictions"],
        align="<>>>>",
    )

    knee = sw.knee
    if knee is not None:
        out.append("")
        out += render.wrap(
            f"The curve flattens at {render.tokens(knee.capacity_tokens)} of cache: "
            f"{render.pct(knee.hit_rate)} hit rate, and the largest capacity "
            "simulated buys no more than a point beyond it. Renting past the knee "
            "buys nothing.")

    out.append("")
    out += render.heading("block size")
    out += render.table(
        [[f"{r.block_size}", render.pct(r.hit_rate), render.money(r.cost)]
         for r in sw.by_block],
        ["block", "hit rate", "input cost"],
        align="<>>",
    )
    out += render.wrap(
        "Bigger blocks waste more at the boundary and match less often; smaller "
        "blocks cost more bookkeeping per token. The difference here is the "
        "size of the prize for tuning it.")

    best = min(sw.by_capacity, key=lambda r: r.cost)
    delta = sw.measured_cost - best.cost
    out.append("")
    if delta > 0:
        out += render.wrap(
            f"Against the measured input bill of {render.money(sw.measured_cost)}, "
            f"the best simulated configuration spends {render.money(best.cost)} — "
            f"a difference of {render.money(delta)}. That gap is what prefix "
            "sharing across sessions is worth if the opening prefix is "
            "byte-stable. It is SIMULATED and no other report treats it as saved.")
    else:
        out += render.wrap(
            "The simulated cache does not beat the measured bill, which means the "
            "provider's cache is already doing at least this well and cross-session "
            "sharing is not the lever here.")
    return "\n".join(out)


def _ttl_arg(text: str) -> float:
    """An argparse type for a cache lifetime.

    A negative value silently meant "never expires" -- `simulate` gates the
    expiry pass on `ttl_s > 0` -- which is the opposite of what somebody typing
    `--ttl -1` is asking for, and it produced the most optimistic possible hit
    rate from an input that reads as the most pessimistic.
    """
    import argparse

    try:
        v = float(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number of seconds") from e
    if v < 0:
        raise argparse.ArgumentTypeError(
            f"--ttl is a lifetime in seconds and cannot be negative, got {v:g}; "
            "pass 0 for a cache that never expires")
    return v


def main(argv: list[str] | None = None) -> int:
    from adder.core import filters

    ap = argparse.ArgumentParser(
        prog="adder cachesim",
        description="Replay the workload against a simulated prefix cache: "
                    "hit rate versus capacity, block size, and TTL.",
    )
    ap.add_argument("--ttl", type=_ttl_arg, default=DEFAULT_TTL_S,
                    help="idle seconds before a block expires; 0 means never "
                         "(default: %(default)s)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    filters.add_arguments(ap)
    args = ap.parse_args(argv)

    sessions, _window = filters.load(args, use_cache=True)
    if not sessions:
        msg = {"simulated": True, "requests": 0} if args.json else \
            "  No sessions found to replay."
        print(json.dumps(msg, indent=2) if args.json else msg)
        return 1

    if args.json:
        print(json.dumps(sweep(sessions, ttl_s=args.ttl).to_json(),
                         indent=2, sort_keys=True))
        return 0
    print(report(sessions, ttl_s=args.ttl))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
