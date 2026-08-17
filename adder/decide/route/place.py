"""Where a warm session should run, when somewhere else is cheaper.

Locality has a price, and it is not zero
-----------------------------------------
A session that has been running for two hours has something valuable that does
not appear on any price list: a resident prefix. Its context is cached with one
provider, at one model, and every turn reads it at a tenth of the input rate.

Move that session to a cheaper model and the prefix does not come with it. The
cache is scoped to the model, so the new one starts cold: the whole context is
re-read at full price, and then written again to establish a new prefix. On a
200K-token context that single move costs more than a hundred ordinary turns.

This is the same trade a load balancer makes when it decides whether to send a
follow-up request to the region that already holds its cached state or to the
region with cheaper capacity. The answer is the same shape too: **affinity wins
on short remaining horizons, price wins on long ones, and the crossover is a
number you can compute rather than argue about.**

    breakeven_turns = migration_cost / (cost_per_turn_here - cost_per_turn_there)

Above that many remaining turns, move. Below it, stay, however much cheaper the
other model looks per token. A recommendation that omits this is not
conservative, it is wrong: it will confidently move a session that had thirty
turns left onto a model whose migration alone costs a hundred turns of savings.

What this adds over the existing switch gate
---------------------------------------------
`cost.switch_is_profitable` already answers this for one named pair, and its
arithmetic is the same. Three things are different here:

* it sweeps the **whole catalog** rather than a pair you already suspected, so
  the answer can be a provider you were not considering;
* it prices the **affinity you would be discarding** as a number in its own
  right, because that is the quantity people do not believe until they see it;
* it handles providers whose cache economics differ from the one this tool grew
  up on. A provider with no prompt cache at all does not re-read your prefix at
  0.10x, it re-reads it at 1.00x, and a per-token price that looks 40% cheaper
  is then 5x more expensive in a long session. Feasibility and cache behaviour
  gate the comparison before price is consulted at all.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

from adder.pricing.catalog import Catalog, Entry
from adder.util import render


@dataclass(frozen=True)
class Placement:
    """One candidate home for this session, priced against staying put."""

    entry: Entry
    per_turn: float
    migration: float
    feasible: bool = True
    blocked: str = ""
    assumed_cache: bool = False

    @property
    def id(self) -> str:
        return self.entry.id

    def saving_per_turn(self, incumbent: float) -> float:
        return incumbent - self.per_turn

    def breakeven_turns(self, incumbent: float) -> float:
        """Turns of remaining session before the move repays its own cost.

        `inf` when the candidate is not cheaper per turn: no horizon repays a
        migration to a model that costs more every turn, and returning a large
        finite number there would invite a caller to compare it.
        """
        gain = self.saving_per_turn(incumbent)
        if gain <= 0:
            return float("inf")
        return self.migration / gain

    def worth_it(self, incumbent: float, remaining_turns: int,
                 *, margin: float = 1.5) -> bool:
        """Move only with room to spare.

        The margin is not timidity. `remaining_turns` is an estimate with a wide
        interval, and the cost of being wrong is asymmetric: staying on a
        slightly dearer model wastes a little on every turn, while moving a
        session that ends early wastes the entire migration at once.
        """
        return self.feasible and remaining_turns > self.breakeven_turns(incumbent) * margin


def affinity_value(ctx_tokens: int, model: str, remaining_turns: int) -> float:
    """What the resident prefix is worth: the discount it earns before the end.

    The number that makes the argument concrete. It is not the cost of the
    context, it is the difference between reading it warm and reading it cold,
    for as long as the session has left.
    """
    from adder.pricing.cost import Rates

    if ctx_tokens <= 0 or remaining_turns <= 0:
        return 0.0
    r = Rates.for_model(model)
    cold = ctx_tokens * r.inp * remaining_turns / 1_000_000.0
    warm = ctx_tokens * r.cache_read * remaining_turns / 1_000_000.0
    return max(0.0, cold - warm)


def migration_cost(ctx_tokens: int, target: str) -> float:
    """One cold read of the context plus one prefix write, on the target model."""
    from adder.pricing.cost import Rates

    if ctx_tokens <= 0:
        return 0.0
    r = Rates.for_model(target)
    return ctx_tokens * (r.inp + r.cache_write) / 1_000_000.0


def per_turn_cost(model: str, ctx_tokens: int, out_tokens: int) -> float:
    """A steady-state warm turn: read the prefix cached, write the answer."""
    from adder.pricing.cost import Rates

    r = Rates.for_model(model)
    return (ctx_tokens * r.cache_read + out_tokens * r.out) / 1_000_000.0


@dataclass(frozen=True)
class _EntryRates:
    """Rates taken straight off a catalog entry, for models the registry lacks.

    The registry knows the models this tool prices first-hand. A catalog can
    contain more than that -- a project override, a provider added yesterday,
    an entry someone pinned by hand -- and the first version of this module
    skipped every one of them, silently, so a user-supplied catalog produced an
    empty field and no explanation.

    The fallback is deliberately pessimistic where the catalog is silent. A
    provider that publishes no cache-read rate is assumed to have **no prompt
    cache**, so the prefix is re-read at full input rate rather than at a tenth
    of it. Assuming the discount instead would make an uncacheable provider look
    like the cheapest place to put a long session, which is precisely backwards.
    """

    inp: float
    out: float
    cache_read: float
    cache_write: float


def _rates_for(e: Entry):
    """Registry rates when it knows the model, the entry's own otherwise."""
    from adder.pricing.cost import Rates
    from adder.pricing.registry import UnknownModelError, UnpricedModelError

    try:
        return Rates.for_model(e.id)
    except (UnknownModelError, UnpricedModelError):
        if e.inp is None or e.out is None:
            return None
        return _EntryRates(
            inp=e.inp,
            out=e.out,
            cache_read=e.cache_read if e.cache_read is not None else e.inp,
            cache_write=e.cache_write if e.cache_write is not None else e.inp,
        )


@dataclass
class Options:
    incumbent: str = ""
    incumbent_per_turn: float = 0.0
    ctx_tokens: int = 0
    remaining_turns: int = 0
    out_tokens: int = 0
    affinity: float = 0.0
    places: list[Placement] = field(default_factory=list)
    considered: int = 0
    infeasible: int = 0

    @property
    def movable(self) -> list[Placement]:
        return [p for p in self.places
                if p.worth_it(self.incumbent_per_turn, self.remaining_turns)]

    @property
    def best(self) -> Placement | None:
        movable = self.movable
        return min(movable, key=lambda p: p.per_turn) if movable else None

    def to_json(self) -> dict:
        def row(p: Placement) -> dict:
            be = p.breakeven_turns(self.incumbent_per_turn)
            return {
                "id": p.id, "org": p.entry.org,
                "per_turn_usd": p.per_turn,
                "migration_usd": p.migration,
                "saving_per_turn_usd": p.saving_per_turn(self.incumbent_per_turn),
                "breakeven_turns": None if be == float("inf") else be,
                "worth_it": p.worth_it(self.incumbent_per_turn, self.remaining_turns),
                "assumed_cache": p.assumed_cache,
            }

        return {
            "incumbent": self.incumbent,
            "incumbent_per_turn_usd": self.incumbent_per_turn,
            "context_tokens": self.ctx_tokens,
            "remaining_turns": self.remaining_turns,
            "affinity_value_usd": self.affinity,
            "considered": self.considered,
            "infeasible": self.infeasible,
            "best": self.best.id if self.best else None,
            "candidates": [row(p) for p in self.places],
        }


def evaluate(
    cat: Catalog,
    *,
    incumbent: str,
    ctx_tokens: int,
    remaining_turns: int,
    out_tokens: int = 1_200,
    board: str = "webdev",
    require_rating: bool = False,
) -> Options:
    """Price every model in the catalog as a home for this session."""
    from adder.pricing.registry import UnknownModelError, UnpricedModelError, fits

    opts = Options(
        incumbent=incumbent,
        incumbent_per_turn=per_turn_cost(incumbent, ctx_tokens, out_tokens),
        ctx_tokens=ctx_tokens,
        remaining_turns=remaining_turns,
        out_tokens=out_tokens,
        affinity=affinity_value(ctx_tokens, incumbent, remaining_turns),
    )

    for e in cat:
        if e.id == incumbent or e.inp is None or e.out is None:
            continue
        if require_rating and board not in e.elo:
            continue
        opts.considered += 1
        # Feasibility gates profitability: a window that cannot hold the
        # context is not a cheap home, it is a 400.
        if e.context is not None and e.context < ctx_tokens:
            opts.infeasible += 1
            continue
        try:
            registry_knows = fits(e.id, ctx_tokens)
        except (UnknownModelError, UnpricedModelError):
            # Not a first-party model. The catalog's own context limit was
            # already checked above, so fall through and price from the entry.
            registry_knows = True
        if not registry_knows:
            opts.infeasible += 1
            continue

        r = _rates_for(e)
        if r is None:
            continue
        per_turn = (ctx_tokens * r.cache_read + out_tokens * r.out) / 1_000_000.0
        move = ctx_tokens * (r.inp + r.cache_write) / 1_000_000.0
        opts.places.append(Placement(
            entry=e, per_turn=per_turn, migration=move,
            assumed_cache=e.cache_read is None,
        ))
    opts.places.sort(key=lambda p: p.per_turn)
    return opts


def _breakeven_str(be: float) -> str:
    """Format a break-even horizon. Sub-turn values read as `<1`, not `0`.

    A candidate cheap enough to repay its migration inside a single turn
    formats to "0" at zero decimal places, which reads as "no break-even
    computed" rather than "repays immediately".
    """
    if be == float("inf"):
        return "never"
    return "<1" if be < 1.0 else f"{be:,.0f}"


def report(opts: Options, *, top: int = 10) -> str:
    out: list[str] = []
    out += render.heading("placement — stay warm, or move somewhere cheaper",
                          rule="=")
    out.append(render.kv("running on", opts.incumbent))
    out.append(render.kv("context", f"{opts.ctx_tokens:,} tok"))
    out.append(render.kv("turns left", str(opts.remaining_turns)))
    out.append(render.kv("cost per turn", render.money(opts.incumbent_per_turn)))
    out.append(render.kv("resident prefix worth", render.money(opts.affinity)))
    out.append("")
    out += render.wrap(
        f"That last number is what the cache under this session saves you over "
        f"its remaining {opts.remaining_turns} turns, against reading the same "
        "context cold. Moving discards it.")

    if not opts.places:
        out.append("")
        out.append("  No other model in the catalog can hold this context and "
                   "carries a price.")
        return "\n".join(out)

    out.append("")
    rows = []
    for p in opts.places[:top]:
        be = p.breakeven_turns(opts.incumbent_per_turn)
        rows.append([
            p.id[:30], p.entry.org[:12],
            render.money(p.per_turn), render.money(p.migration),
            _breakeven_str(be),
            "move" if p.worth_it(opts.incumbent_per_turn, opts.remaining_turns) else "stay",
        ])
    out += render.table(
        rows, ["model", "org", "$/turn", "migration", "breakeven", ""],
        align="<<>>><",
    )

    best = opts.best
    out.append("")
    if best is None:
        cheapest = opts.places[0]
        be = cheapest.breakeven_turns(opts.incumbent_per_turn)
        if be == float("inf"):
            out += render.wrap(
                "Nothing in the catalog is cheaper per turn than staying put. "
                "The resident prefix is the cheapest thing you own.")
        else:
            out += render.wrap(
                f"Stay. The cheapest alternative, {cheapest.id}, needs "
                f"{be:,.0f} turns to repay its migration and this session has "
                f"{opts.remaining_turns}. Moving now spends "
                f"{render.money(cheapest.migration)} to save "
                f"{render.money(cheapest.saving_per_turn(opts.incumbent_per_turn) * opts.remaining_turns)}.")
    else:
        be = best.breakeven_turns(opts.incumbent_per_turn)
        net = (best.saving_per_turn(opts.incumbent_per_turn) * opts.remaining_turns
               - best.migration)
        out += render.wrap(
            f"Move to {best.id}: it repays the migration in {be:,.0f} turns and "
            f"this session has {opts.remaining_turns} left, for a net "
            f"{render.money(net)}.")

    if any(p.assumed_cache for p in opts.places[:top]):
        out.append("")
        out += render.wrap(
            "Rows marked with an assumed cache publish no cache rate. A provider "
            "with no prompt cache re-reads your prefix at full input rate, not a "
            "tenth of it, so a per-token price that looks cheaper can be several "
            "times dearer in a long session.")

    out.append("")
    out += render.wrap(
        "MODELLED: prices come from the catalog and a move is priced as one cold "
        "read plus one prefix write. Quality is not considered here at all — use "
        "`adder frontier` for that, and never move on price alone.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    from adder.pricing.catalog import load

    ap = argparse.ArgumentParser(
        prog="adder place",
        description="Should this warm session move to a cheaper model? "
                    "Prices the prefix you would be throwing away.",
    )
    ap.add_argument("--model", default="claude-opus-5",
                    help="the model the session is running on now")
    ap.add_argument("--context", type=int, default=190_000,
                    help="current context size in tokens")
    ap.add_argument("--turns", type=int, default=100,
                    help="turns the session still has to run")
    ap.add_argument("--out", type=int, default=1_200, help="tokens per answer")
    ap.add_argument("--rated-only", action="store_true",
                    help="only consider models with a rating on the coding board")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    opts = evaluate(
        load(), incumbent=args.model,
        ctx_tokens=max(0, args.context),
        remaining_turns=max(0, args.turns),
        out_tokens=max(0, args.out),
        require_rating=args.rated_only,
    )
    if args.json:
        print(json.dumps(opts.to_json(), indent=2, sort_keys=True))
    else:
        print(report(opts, top=max(1, args.top)))
    return 0 if opts.places else 1


if __name__ == "__main__":
    raise SystemExit(main())
