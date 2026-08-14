"""Browse the catalog, refresh it, and check the hardcoded ladder against it.

`adder models` is the answer to "what can this thing route to today". Four views:

    adder models                what is in the catalog, by value
    adder models refresh        pull the public sources (the only networked command)
    adder models show <name>    everything known about one model
    adder models ladder         the hardcoded tier ladder vs what the data says

`ladder` is the one that matters over time. adder ships a three-rung
ladder -- Haiku, Sonnet, Opus -- written into `classify.py` as constants. Those
constants are correct on the day they are written and quietly wrong afterwards:
a new model lands, prices move, an intro rate expires, and adder keeps
dispatching to last quarter's tier. `ladder` re-derives each rung from the
catalog and prints the drift, so the staleness is visible instead of silent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .catalog import Catalog, Entry, load, user_cache
from .select import Need, cost_of

BANDS = (
    # (rung, ceiling on input $/Mtok, what the rung is for)
    ("T0", 1.5, "lookups, searches, read-only triage"),
    ("T1", 3.5, "scoped edits, mechanical refactors, tests"),
    ("T2", 99.0, "multi-file, ambiguous, long-horizon"),
)


def _fmt_price(v: float | None) -> str:
    return "-" if v is None else f"{v:>6.2f}"


def _row(e: Entry, rating: float | None) -> str:
    lic = "open" if e.open_weights else ("prop" if e.license else "?")
    ctx = f"{e.context // 1000}K" if e.context else "-"
    return (f"{e.id[:36]:<36} {_fmt_price(e.inp)} {_fmt_price(e.out)} "
            f"{ctx:>7} {lic:<5} {(f'{rating:,.0f}' if rating else '-'):>6}")


def cmd_list(cat: Catalog, a: argparse.Namespace) -> int:
    entries = cat.find(
        org=a.org,
        open_weights=True if a.open_weights else None,
        needs_tools=a.tools,
        min_context=a.min_context,
        priced_only=not a.include_unpriced,
        rated_only=not a.include_unrated,
    )
    if a.json:
        print(json.dumps([e.to_json() for e in entries], indent=1))
        return 0

    rated = [(e, e.rating()) for e in entries]
    rated.sort(key=lambda t: -(t[1] or 0))
    age = cat.age_days()
    src = cat.provenance.get("sources") or []
    when = cat.provenance.get("refreshed_at", "never")
    print(f"{len(cat)} models in the catalog  |  refreshed {when}"
          + (f"  ({age:.0f}d ago)" if age is not None else ""))
    if src:
        print("  sources: " + ", ".join(
            f"{s['name']}{'' if s.get('ok') else ' (FAILED)'}" for s in src))
    if cat.is_stale():
        print("  ! stale; run `adder models refresh`")
    print()
    print(f"{'model':<36} {'$in':>6} {'$out':>6} {'ctx':>7} {'lic':<5} {'elo':>6}")
    for e, r in rated[:a.limit]:
        print(_row(e, r))
    if len(rated) > a.limit:
        print(f"... {len(rated) - a.limit} more (--limit)")
    return 0


def cmd_show(cat: Catalog, a: argparse.Namespace) -> int:
    e = cat.get(a.name)
    if e is None:
        near = [x.key for x in cat if a.name.lower().replace("/", "-") in x.key][:8]
        print(f"no model matching {a.name!r} in the catalog")
        if near:
            print("  close: " + ", ".join(near))
        return 1
    if a.json:
        print(json.dumps(e.to_json(), indent=1))
        return 0
    print(f"{e.id}   ({e.org or 'unknown org'}, {e.license or 'license unknown'})")
    print(f"  price      ${e.inp}/M in, ${e.out}/M out"
          + (f", ${e.cache_read}/M cached read" if e.cache_read is not None
             else ", cache rates not published"))
    print(f"  context    {e.context:,} tok" if e.context else "  context    unknown")
    if e.max_output:
        print(f"  max out    {e.max_output:,} tok")
    if e.elo:
        for board, v in sorted(e.elo.items(), key=lambda kv: -kv[1]):
            print(f"  elo        {v:,.0f}  ({board}, {e.votes:,} votes)")
    for label, v in (("intelligence", e.intelligence), ("coding", e.coding),
                     ("agentic", e.agentic)):
        if v is not None:
            print(f"  {label:<10} {v}")
    if e.modalities:
        print(f"  inputs     {', '.join(e.modalities)}")
    if e.params:
        print(f"  supports   {', '.join(sorted(e.params)[:12])}")
    print(f"  released   {e.released or 'unknown'}")
    print(f"  sources    {', '.join(e.sources) or 'none'}"
          f"{'  (verified)' if e.verified else '  (unverified prices)'}")
    return 0


def cmd_ladder(cat: Catalog, a: argparse.Namespace) -> int:
    """Re-derive each rung from the catalog and show the drift."""
    from .classify import LADDER

    need = Need(context_tokens=a.context, remaining_turns=a.remaining,
                est_read_tokens=a.read_tokens, harness=a.harness)
    ref = cat.get("claude-opus-5")

    rows = []
    for rung, ceiling, purpose in BANDS:
        current = LADDER.get(rung, "")
        cands = [
            e for e in cat.find(needs_tools=True, priced_only=True, rated_only=True,
                                min_context=a.read_tokens + 8_000)
            if e.inp is not None and e.inp <= ceiling
            and (a.harness == "any" or e.org.lower() == "anthropic")
        ]
        # Within a price band the rung should hold the strongest model, not the
        # cheapest: the band already bought the saving.
        cands.sort(key=lambda e: -(e.rating() or 0))
        best = cands[0] if cands else None
        rows.append((rung, purpose, ceiling, current, best))

    if a.json:
        print(json.dumps([{
            "rung": r, "purpose": p, "max_input_price": c, "current": cur,
            "suggested": (b.id if b else None),
            "suggested_elo": (b.rating() if b else None),
            "drift": bool(b and b.id != cur),
        } for r, p, c, cur, b in rows], indent=1))
        return 0

    print(f"harness={a.harness}  |  band ceiling is input $/Mtok\n")
    print(f"{'rung':<5} {'hardcoded':<22} {'catalog says':<22} {'elo':>6}  purpose")
    drift = 0
    for rung, purpose, _ceiling, current, best in rows:
        sug = best.id if best else "-"
        elo = f"{best.rating():,.0f}" if best and best.rating() else "-"
        mark = " " if (best and best.id == current) or best is None else "*"
        if mark == "*":
            drift += 1
        print(f"{rung:<5} {current:<22} {mark}{sug:<21} {elo:>6}  {purpose}")
    print()
    if drift:
        print(f"* {drift} rung(s) differ from the constants in classify.py.")
        print("  Drift is not automatically an upgrade: a rung is only worth")
        print("  changing if the new model also holds the context and its price")
        print("  survives `adder savings` on your own history.")
    else:
        print("Ladder matches the catalog.")

    if ref is not None and a.harness == "any":
        c = cost_of(ref, need)
        print(f"\nreference {ref.id}: ${c.best:,.3f} for this shape of task")
    return 0


def cmd_refresh(a: argparse.Namespace) -> int:
    """Pull the public sources. The one command in the tool that uses a socket.

    Imported here rather than at module scope so that `adder models list` -- and
    every other report -- cannot reach the network even by accident.
    """
    from .sources import refresh

    # `--if-stale` exists so this can be wired into a cron job or a session
    # hook without hammering two public endpoints on every invocation. The
    # check is local: it reads the catalog already on disk and returns before
    # any socket is opened.
    if a.if_stale:
        current = load()
        age = current.age_days()
        if not current.is_stale(max_age_days=a.max_age):
            print(f"catalog is {age:.0f}d old (limit {a.max_age:.0f}d); "
                  "nothing to do")
            return 0
        print(f"catalog is {'missing' if age is None else f'{age:.0f}d old'}; "
              "refreshing")

    offline = {}
    for spec in a.from_files:
        name, _, path = spec.partition("=")
        if not path:
            print(f"bad --from {spec!r}; expected NAME=PATH")
            return 2
        offline[name] = Path(path)

    cat, results = refresh(offline_files=offline, timeout=a.timeout)
    if not any(r.ok for r in results):
        for r in results:
            print(f"  ! {r.name}: {r.error}")
        print("no source succeeded; catalog left unchanged")
        return 1

    out = a.out or user_cache()
    cat.save(out)
    if a.json:
        print(json.dumps({"path": str(out), "models": len(cat),
                          "sources": cat.provenance["sources"]}, indent=1))
        return 0
    for r in results:
        print(f"  {'ok ' if r.ok else 'FAIL'} {r.name:<12} "
              f"{f'{r.count} models  {r.origin}' if r.ok else r.error}")
    both = len([e for e in cat if e.priced and e.rating() is not None])
    print(f"\n{len(cat)} models -> {out}")
    print(f"  {len([e for e in cat if e.priced])} priced, "
          f"{len([e for e in cat if e.rating() is not None])} rated, {both} both")
    print("  prices from an aggregator are unverified; first-party Claude rates "
          "still come from prices.py")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="adder.models",
                                 description="browse and refresh the model catalog")
    sub = ap.add_subparsers(dest="cmd")

    p_list = sub.add_parser("list", help="what is in the catalog (default)")
    for p in (p_list, ap):
        p.add_argument("--org", default=None, help="filter by organisation")
        p.add_argument("--open-weights", action="store_true")
        p.add_argument("--tools", action="store_true", help="tool-use capable only")
        p.add_argument("--min-context", type=int, default=0)
        p.add_argument("--include-unrated", action="store_true")
        p.add_argument("--include-unpriced", action="store_true")
        p.add_argument("--limit", type=int, default=25)
        p.add_argument("--json", action="store_true")

    p_show = sub.add_parser("show", help="everything known about one model")
    p_show.add_argument("name")
    p_show.add_argument("--json", action="store_true")

    p_lad = sub.add_parser("ladder", help="hardcoded tier ladder vs the catalog")
    p_lad.add_argument("--harness", default="claude-code", choices=("claude-code", "any"))
    p_lad.add_argument("--context", type=int, default=200_000)
    p_lad.add_argument("--remaining", type=int, default=100)
    p_lad.add_argument("--read-tokens", type=int, default=40_000)
    p_lad.add_argument("--json", action="store_true")

    p_ref = sub.add_parser("refresh",
                           help="pull public sources (the only networked command)")
    p_ref.add_argument("--from", dest="from_files", action="append", default=[],
                       metavar="NAME=PATH",
                       help="replay a saved capture instead of fetching "
                            "(lmarena=page.html, openrouter=models.json)")
    p_ref.add_argument("--out", type=Path, default=None)
    p_ref.add_argument("--timeout", type=int, default=30)
    p_ref.add_argument("--json", action="store_true")
    p_ref.add_argument("--if-stale", action="store_true",
                       help="only fetch when the catalog is older than --max-age")
    p_ref.add_argument("--max-age", type=float, default=21.0,
                       metavar="DAYS", help="staleness threshold (default 21)")

    a = ap.parse_args(argv)
    if a.cmd == "refresh":
        return cmd_refresh(a)
    cat = load()
    if a.cmd == "show":
        return cmd_show(cat, a)
    if a.cmd == "ladder":
        return cmd_ladder(cat, a)
    return cmd_list(cat, a)


if __name__ == "__main__":
    raise SystemExit(main())
