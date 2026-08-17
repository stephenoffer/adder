"""A recorded turn is priced on the day it ran, not on the day you ask.

`prices.py` exists because rates move: Sonnet 5's introductory $2/$10 reverts to
$3/$15 after 2026-08-31. Every rate lookup takes an `on` date and resolves
`None` to *today*, which is right for a forecast and wrong for a measurement.
`Turn.cost()` took that default, so on 1 September every Sonnet turn already on
disk would have started reporting 1.5x what it was actually billed -- a
retroactive change to a measurement of the past, with nothing in the repo having
changed. A confident wrong number is the failure mode this tool exists to avoid.

`cost_on` still prices a whole history at one named date, because "what would
last month cost at today's rates" is a real and different question.
"""
from __future__ import annotations

from datetime import date

import pytest

from adder.core.trace import Session, Turn

INTRO = date(2026, 8, 10)      # inside Sonnet 5's introductory window
AFTER = date(2026, 9, 1)       # after it reverts


def _turn(ts="2026-08-10T12:00:00Z", model="claude-sonnet-5"):
    return Turn("s", "p", model, uncached_in=0, cache_read=1_000_000,
                cache_write=0, out=100_000, thinking=0, sidechain=False, ts=ts)


class TestATurnIsPricedOnItsOwnDate:
    def test_the_default_matches_the_date_it_ran(self):
        t = _turn()
        assert t.cost() == pytest.approx(t.cost(INTRO))

    def test_and_does_not_drift_to_the_post_intro_rate(self):
        t = _turn()
        assert t.cost() < t.cost(AFTER)

    def test_the_input_side_is_dated_too(self):
        t = _turn()
        assert t.input_cost() == pytest.approx(t.input_cost(INTRO))

    def test_the_output_side_is_dated_too(self):
        t = _turn()
        assert t.output_cost() == pytest.approx(t.output_cost(INTRO))

    def test_an_explicit_date_still_wins(self):
        """`cost_on` has to keep working: repricing history is a real question."""
        t = _turn()
        assert t.cost(AFTER) == pytest.approx(t.cost(AFTER))
        assert t.cost(AFTER) > t.cost(INTRO)

    def test_an_undated_turn_falls_back_rather_than_failing(self):
        assert _turn(ts=None).cost() > 0

    def test_a_model_with_no_intro_rate_is_unaffected(self):
        t = _turn(model="claude-opus-5")
        assert t.cost() == pytest.approx(t.cost(AFTER))


class TestSessionTotals:
    def test_a_session_sums_each_turn_at_its_own_date(self):
        s = Session("s", "p")
        s.turns = [_turn(), _turn()]
        assert s.cost == pytest.approx(2 * _turn().cost(INTRO))

    def test_cost_on_reprices_the_whole_history_at_one_date(self):
        s = Session("s", "p")
        s.turns = [_turn(), _turn()]
        assert s.cost_on(AFTER) > s.cost


class TestOnePricingPathForARecordedTurn:
    """Every module prices a recorded turn on the same date: the turn's own.

    `Turn.cost()` resolves the turn's date, but a dozen modules priced turns by
    calling `Rates.for_model(t.model, ...)` with `on` left at None -- which
    silently means *today*. While every rate in the table is stable the two
    agree; the day an introductory rate expires they diverge by the size of the
    change, and `validate.replay_reproduces_measured_spend` compares one against
    the other. `Turn.rates()` is the single accessor they now share.
    """

    def test_the_accessor_is_dated_like_cost(self):
        t = _turn()
        assert t.rates().inp == t.rates(INTRO).inp
        assert t.rates().inp != t.rates(AFTER).inp

    def test_it_carries_the_turns_own_ttl(self):
        t = _turn()
        assert t.rates().cache_write == t.rates(ttl=t.ttl).cache_write

    def test_the_ttl_override_does_not_move_the_date(self):
        """The simulator asks a different question of the same turn."""
        t = _turn()
        assert t.rates(ttl="1h").inp == t.rates().inp
        assert t.rates(ttl="1h").cache_write > t.rates(ttl="5m").cache_write

    def test_input_cost_agrees_with_the_accessor(self):
        t = _turn()
        expected = (t.uncached_in * t.rates().inp
                    + t.cache_read * t.rates().cache_read
                    + t.cache_write * t.rates().cache_write) / 1_000_000
        assert t.input_cost() == pytest.approx(expected)

    def test_no_module_prices_a_turn_off_the_wall_clock(self):
        """The rule, enforced: `Rates.for_model(t.model, ...)` is not the way."""
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parents[2] / "adder"
        bad = []
        # Scanned over the whole file, not line by line: the call is routinely
        # wrapped after the open paren, and a line-anchored pattern walked
        # straight past `Rates.for_model(\n    t.model, ...)` in `plan.py`.
        pat = re.compile(r"(?:Rates\.for_model|\brate)\(\s*t\.model")
        for p in root.rglob("*.py"):
            if "__pycache__" in str(p):
                continue
            src = p.read_text()
            for m in pat.finditer(src):
                line = src[: m.start()].count("\n") + 1
                # Prose mentions the call in backticks; only code counts.
                if "`" in src.splitlines()[line - 1]:
                    continue
                bad.append(f"{p.relative_to(root.parent)}:{line}")
        assert not bad, f"these price a turn without its date; use t.rates(): {bad}"
