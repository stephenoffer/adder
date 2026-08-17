"""`adder trace`: the whole bill, and the four ways it breaks down.

The reading and deduplication this report stands on live in
`adder.core.trace`, not here. That separation exists because the reader is
imported by every other report in the package, and a foundation that drags an
argparse parser and a printing routine behind it is a foundation that cannot be
imported cheaply -- the prompt hook pays for this on every submit.

So: `core.trace` answers "what turns are on disk and what did they cost".
This module answers "show me", and owns nothing else.
"""

from __future__ import annotations

import json

from adder.core.filters import root_of as _root_of
from adder.core.trace import (
    GROUPINGS,
    group_by,
    load_sessions,
    summarize_sessions,
)


def _pct(a, p: float) -> int:
    """Interpolated percentile, as an integer count.

    Kept as a thin wrapper over `stats.quantile` so the three call sites below
    read the same as they always did. The estimator underneath changed: this
    used to index a sorted list at `int(len*p)`, which reports the maximum as
    the p90 on any sample of ten.
    """
    from adder.util.stats import quantile

    return round(quantile(a, p))


def main(argv: list[str] | None = None) -> int:
    import argparse

    from adder.core import settings
    from adder.core.filters import Window
    from adder.core.filters import add_arguments as add_window
    from adder.util.render import bar, money, table, tokens
    from adder.util.stats import gini, share

    ap = argparse.ArgumentParser(prog="adder trace",
                                 description="Measure Claude Code spend.")
    add_window(ap)
    ap.add_argument("--verify", action="store_true", help="assert plan headline figures")
    ap.add_argument("--json", action="store_true", help="machine-readable summary")
    ap.add_argument("--no-cache", action="store_true", help="ignore the parse cache")
    ap.add_argument("--by", choices=GROUPINGS, default=None,
                    help="break the total down by one dimension")
    ap.add_argument("--top", type=int, default=3, metavar="N",
                    help="how many rows to show in each ranking (default: %(default)s)")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any turn used a model with no known price")
    a = ap.parse_args(argv)
    # `root_of`: the argument if one was given, else the `root`
    # setting. Resolved here so two commands cannot disagree
    # about which transcript directory `adder config` names.
    a.root = str(_root_of(a))

    use_cache = bool(settings.get("cache")) and not a.no_cache
    unknown: dict[str, int] = {}
    sessions = load_sessions(a.root, use_cache=use_cache, unknown=unknown)
    window = Window.from_args(a)
    if window.active:
        sessions = window.apply(sessions)
    s = summarize_sessions(sessions, unknown=unknown)

    if not s.n_turns:
        detail = f" matching {window.describe()}" if window.active else ""
        # The unknown-model tally is carried into the empty case too, because
        # it is the whole explanation for it: a corpus of turns nobody can
        # price reports "no priced turns" and, without this, no reason. That is
        # the "quietly too small total" this module's docstring says the tally
        # exists to prevent, in its most extreme form -- a total of zero.
        if a.json:
            print(json.dumps({"error": "no priced turns", "root": a.root,
                              "filter": window.describe(),
                              "unknown_models": s.unknown_models,
                              "unknown_turns": s.unknown_turns,
                              "synthetic_turns": s.synthetic_turns}))
        else:
            print(f"No priced turns found under {a.root}{detail}")
            if s.unknown_models:
                names = ", ".join(sorted(s.unknown_models)[:5])
                print(f"  ⚠ {s.unknown_turns:,} turns used a model with no price "
                      f"in this build: {names}")
                print("    Run `adder models refresh`, or add it to "
                      ".adder/catalog.json.")
            if s.synthetic_turns:
                print(f"  {s.synthetic_turns:,} client-side placeholder turns "
                      "were skipped; nothing was billed for them.")
        return 1 if not (a.strict and s.unknown_models) else 2

    lens = [x.n_turns for x in sessions.values()]
    ctxs = [x.peak_context for x in sessions.values()]
    costs = [x.cost for x in sessions.values()]
    groups = group_by(sessions, a.by) if a.by else []

    if a.json:
        payload = {
            "total": round(s.total, 2),
            "sessions": s.n_sessions,
            "turns": s.n_turns,
            "cost_per_turn": round(s.cost_per_turn, 6),
            "input_side": round(s.input_side, 2),
            "output_side": round(s.output_side, 2),
            "cache_read": round(s.cache_read_cost, 2),
            "cache_write": round(s.cache_write_cost, 2),
            "thinking": round(s.thinking_cost, 2),
            "sidechain": round(s.sidechain_cost, 2),
            "fast_turns": s.fast_turns,
            "by_model": {k: round(v, 2) for k, v in s.by_model.items()},
            "turns_p50": _pct(lens, 0.5), "turns_p90": _pct(lens, 0.9),
            "ctx_p50": _pct(ctxs, 0.5), "ctx_p90": _pct(ctxs, 0.9),
            "concentration": round(gini(costs), 3),
            "filter": window.describe(),
            "unknown_models": s.unknown_models,
            "synthetic_turns": s.synthetic_turns,
        }
        if groups:
            payload["by_" + a.by] = [
                {"key": g.key, "cost": round(g.cost, 2), "turns": g.turns}
                for g in groups[: a.top] if a.top > 0
            ] or [{"key": g.key, "cost": round(g.cost, 2), "turns": g.turns}
                  for g in groups]
        print(json.dumps(payload))
        return 1 if (a.strict and s.unknown_models) else 0

    if window.active:
        print(f"\n  filter: {window.describe()}")
        if window.dropped_undated:
            print(f"  {window.dropped_undated:,} turns had no timestamp and were dropped")
    print(f"\n  {s.n_sessions} sessions · {s.n_turns:,} turns · "
          f"${s.total:,.2f} list-equivalent · {money(s.cost_per_turn)}/turn\n")
    print(f"  {'model':<28}{'turns':>8}{'cost':>11}{'share':>8}")
    for m, c in sorted(s.by_model.items(), key=lambda kv: -kv[1]):
        print(f"  {m:<28}{s.turns_by_model[m]:>8,}{c:>11,.2f}{100*share(c, s.total):>7.1f}%")

    print(f"\n  input-side   ${s.input_side:>9,.2f}  ({100*share(s.input_side, s.total):.0f}%)")
    print(f"  output-side  ${s.output_side:>9,.2f}  ({100*share(s.output_side, s.total):.0f}%)")
    print(f"  cache-read   ${s.cache_read_cost:>9,.2f}  "
          f"({100*share(s.cache_read_cost, s.total):.0f}% of all spend)")
    print(f"  cache-write  ${s.cache_write_cost:>9,.2f}  "
          f"({100*share(s.cache_write_cost, s.total):.0f}%)")
    if s.thinking_tokens:
        print(f"  thinking     ${s.thinking_cost:>9,.2f}  "
              f"({s.thinking_tokens:,} tok, {100*s.thinking_tokens/max(1,s.out_tokens):.0f}% of output)")
    print(f"  subagents    ${s.sidechain_cost:>9,.2f}  "
          f"({100*share(s.sidechain_cost, s.total):.1f}%, {s.sidechain_turns:,} turns)")
    if s.fast_turns:
        print(f"  fast mode    ${s.fast_cost:>9,.2f}  "
              f"({s.fast_turns:,} turns billed at 2x)")

    print(f"\n  turns/session   p50={_pct(lens,.5):,}  p90={_pct(lens,.9):,}  max={max(lens):,}")
    print(f"  peak context    p50={_pct(ctxs,.5):,}  p90={_pct(ctxs,.9):,}  max={max(ctxs):,}")

    ranked = sorted(sessions.values(), key=lambda x: -x.cost)
    top = sum(x.cost for x in ranked[: max(1, len(ranked) // 4)])
    print(f"  top 25% of sessions = ${top:,.0f} ({100*share(top, s.total):.0f}% of spend)"
          f"  ·  concentration {gini(costs):.2f}")

    if groups:
        # Days read chronologically or they read as nothing. Every other
        # grouping is a ranking, where most-expensive-first is the answer; a
        # date axis sorted by cost is a bar chart with the x-axis shuffled.
        if a.by == "day":
            groups = sorted(groups, key=lambda g: g.key)
            shown = groups[-a.top:] if a.top > 0 else groups
        else:
            shown = groups[: a.top] if a.top > 0 else groups
        print(f"\n  by {a.by}:")
        peak = max((g.cost for g in shown), default=0.0) or 1.0
        rows = []
        for g in shown:
            row = [g.key[:44], f"{g.turns:,}", money(g.cost),
                   f"{100 * share(g.cost, s.total):.1f}%", money(g.cost_per_turn),
                   tokens(g.out_tokens)]
            if a.by == "day":
                row.append(bar(g.cost / peak, 20))
            rows.append(row)
        header = [a.by, "turns", "cost", "share", "$/turn", "out"]
        if a.by == "day":
            header.append("")
        for line in table(rows, header, align="<>>>>><"):
            print(line)
        if a.by == "tool":
            print("    a turn calling N tools is counted under each, so shares "
                  "sum above 100%")
        if len(groups) > len(shown):
            seen = {id(g) for g in shown}
            rest = sum(g.cost for g in groups if id(g) not in seen)
            print(f"    … {len(groups) - len(shown):,} more, ${rest:,.2f}")

    if a.top > 0:
        print("\n  most expensive sessions:")
        for x in ranked[: a.top]:
            print(f"    ${x.cost:>8,.0f}  {x.n_turns:>5,} turns  "
                  f"avg ctx {x.avg_context:>9,}  {x.project[:44]}")

    if s.synthetic_turns:
        print(f"\n  {s.synthetic_turns:,} client-side placeholder records "
              f"(API errors, interrupted streams) were not billed and are "
              f"not counted as turns.")

    if s.unknown_models:
        print()
        print(f"  ⚠ {s.unknown_turns:,} turns used a model with no price in "
              f"prices.py and are NOT in the totals above:")
        for m, n in sorted(s.unknown_models.items(), key=lambda kv: -kv[1])[:5]:
            print(f"      {n:>7,}  {m}")
        print("    Every figure in this report is therefore a lower bound.")

    if a.verify:
        # Structural invariants, not a pinned dollar figure. The absolute total
        # depends on how much history is on disk; the shares are the claim.
        checks = [
            ("input-side >= 85% of spend", share(s.input_side, s.total) >= 0.85),
            ("cache-read >= 70% of spend", share(s.cache_read_cost, s.total) >= 0.70),
            ("output-side <= 15% of spend", share(s.output_side, s.total) <= 0.15),
            ("input + output reconcile to total",
             abs(s.input_side + s.output_side - s.total) < max(0.01, s.total * 0.001)),
            ("median session > 100 turns", _pct(lens, 0.5) > 100),
            ("no turn priced without a known model", s.n_turns > 0),
            ("every turn on disk had a known price", not s.unknown_models),
        ]
        print("\n  verification:")
        ok = True
        for label, passed in checks:
            print(f"    [{'PASS' if passed else 'FAIL'}] {label}")
            ok &= passed
        return 0 if ok else 1
    return 1 if (a.strict and s.unknown_models) else 0


if __name__ == "__main__":
    raise SystemExit(main())
