"""A comparison must be priced on one date, or it moves when a rate expires.

`Turn.pricing_date` exists because "a recorded turn was billed on the day it
ran, so that is the day it has to be priced on" -- and because Sonnet 5's
introductory $2/$10 reverts to $3/$15 after 2026-08-31, so a report that
priced one leg at the turn's date and the other at *today* changed overnight
with nothing in the repository having changed.

This is a repository-level check rather than a unit test: a function that
accepts an `on` date and then calls a price-aware helper without passing it is
the shape of that bug, and it is easier to spot statically than to catch in
each report.
"""

from __future__ import annotations

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[2] / "adder"

# Helpers whose answer depends on the date they are asked about.
PRICED = {
    "turn_cost", "run_cost", "admitted_token_cost", "placement_cost",
    "switch_is_profitable", "cache_miss_cost", "cache_write_cost",
    "cache_read_cost", "cache_storage_cost", "marginal_turn_cost",
    "effort_saving", "max_tolerable_p_fail", "escalation_is_profitable",
    "choose_ttl", "fanout_cost", "token_cost", "debt_multiple",
    "token_lifetime_cost", "breakeven_remaining_turns",
}

# Ways of naming a date that are correct: the caller's, or the turn's own.
_OK_NAMES = {"on", "when", "today"}


def _passes_a_date(call: ast.Call) -> bool:
    for kw in call.keywords:
        if kw.arg == "on":
            return True
    for node in call.args + [kw.value for kw in call.keywords]:
        if isinstance(node, ast.Name) and node.id in _OK_NAMES:
            return True
        # `t.pricing_date(on)` / `st.on`
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "pricing_date":
            return True
        if isinstance(node, ast.Attribute) and node.attr == "on":
            return True
    return False


def _offenders() -> list[str]:
    out = []
    for path in sorted(PKG.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            args = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
            if "on" not in args:
                continue
            for call in [n for n in ast.walk(fn) if isinstance(n, ast.Call)]:
                name = (call.func.attr if isinstance(call.func, ast.Attribute)
                        else getattr(call.func, "id", ""))
                if name in PRICED and not _passes_a_date(call):
                    out.append(
                        f"{path.relative_to(PKG.parent)}:{call.lineno} "
                        f"{fn.name}() calls {name}() without a date")
    return out


def test_a_dated_function_prices_everything_on_that_date():
    bad = _offenders()
    assert not bad, (
        "these take an `on` date and then price something at today instead:\n  "
        + "\n  ".join(bad))
