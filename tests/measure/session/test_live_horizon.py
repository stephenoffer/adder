"""Which horizon `live` hands the cost model: the mean, not the median.

`horizon.py` says it outright -- "`mean_remaining()` exists and is what the cost
model should call" -- and `analyse` called `remaining()`, the conditional
median. Carry cost is linear in remaining turns, so its expectation is set by
E[R]; session length is heavy-tailed here, so the median sits far below the mean
and every dollar figure downstream of `live` was scaled by the smaller number.
That under-prices admission and under-recommends delegation in precisely the
long sessions that hold the spend.

The median is still reported, because a report that quotes a mean session length
is quoting a number no session has. The two are now separate fields.
"""
from __future__ import annotations

import pytest

from adder.core.trace import Session, Turn
from adder.measure.session.horizon import Horizon
from adder.measure.session.live import LiveReport, analyse


def _heavy_tail() -> Horizon:
    """Many short sessions, a few very long ones: the measured shape."""
    return Horizon(sorted([20] * 40 + [80] * 20 + [400] * 8 + [1800] * 4))


class TestTheTwoHorizonsAreKeptApart:
    def test_the_mean_is_reported_alongside_the_median(self, make_session):
        h = _heavy_tail()
        r = analyse(make_session(10), horizon=h)
        assert r.projected_remaining == h.remaining(10)
        assert r.expected_remaining == pytest.approx(h.mean_remaining(10))

    def test_on_a_heavy_tail_they_are_not_the_same_number(self, make_session):
        r = analyse(make_session(10), horizon=_heavy_tail())
        assert r.expected_remaining > r.projected_remaining


class TestPricingUsesTheMean:
    def test_carry_turns_is_the_mean(self, make_session):
        r = analyse(make_session(10), horizon=_heavy_tail())
        assert r.carry_turns == pytest.approx(r.expected_remaining)

    def test_the_debt_multiple_is_priced_off_the_mean(self, make_session):
        r = analyse(make_session(10), horizon=_heavy_tail())
        median_based = LiveReport(
            turns=r.turns, context=r.context, spent=r.spent, per_turn=r.per_turn,
            projected_remaining=r.projected_remaining, projected_total=0.0,
            model=r.model)
        assert r.debt_multiple > median_based.debt_multiple

    def test_reading_a_file_is_priced_off_the_mean(self, make_session):
        """`read_cost` is the number the guard hook gates on."""
        r = analyse(make_session(10), horizon=_heavy_tail())
        inline_mean, _, _ = r.read_cost(50_000)
        median_based = LiveReport(
            turns=r.turns, context=r.context, spent=r.spent, per_turn=r.per_turn,
            projected_remaining=r.projected_remaining, projected_total=0.0,
            model=r.model)
        inline_median, _, _ = median_based.read_cost(50_000)
        assert inline_mean > inline_median


class TestTheFallback:
    def test_a_report_built_without_a_mean_still_prices(self):
        """A hand-built report -- a hook stub, a test double -- must not price at zero."""
        r = LiveReport(turns=10, context=100_000, spent=1.0, per_turn=0.1,
                       projected_remaining=400, projected_total=2.0,
                       model="claude-opus-5")
        assert r.carry_turns == 400.0


class TestLiveReadsTheConversationNotTheSubagent:
    """"This session" is the main chain, whatever the last record happens to be.

    `analyse` took the context, model and TTL off `turns[-1]`. A session whose
    final record is a subagent turn would report that subagent's few-thousand
    token context on its cheap model as the state of the conversation -- and
    that report is what `adder live` prints, what `policy.decide` takes its
    context and model from, and what the PreToolUse guard prices every read
    against.
    """

    @staticmethod
    def _session(trailing_subagent: bool):
        s = Session("s", "p")
        for i in range(30):
            s.turns.append(Turn("s", "p", "claude-opus-5", uncached_in=0,
                                cache_read=500_000 + i, cache_write=0, out=300,
                                thinking=0, sidechain=False,
                                ts=f"2026-08-10T12:{i:02d}:00Z"))
        if trailing_subagent:
            s.turns.append(Turn("s", "p", "claude-haiku-4-5", uncached_in=0,
                                cache_read=3_000, cache_write=0, out=50,
                                thinking=0, sidechain=True,
                                ts="2026-08-10T12:31:00Z"))
        return s

    def _analyse(self, trailing):
        return analyse(self._session(trailing), horizon=Horizon([100] * 10))

    def test_the_context_is_the_main_chains(self):
        assert self._analyse(True).context > 400_000

    def test_the_model_is_the_main_chains(self):
        assert self._analyse(True).model == "claude-opus-5"

    def test_a_trailing_subagent_turn_changes_nothing(self):
        with_sub, without = self._analyse(True), self._analyse(False)
        assert (with_sub.context, with_sub.model) == (without.context, without.model)

    def test_the_next_turn_is_priced_on_the_real_context(self):
        """Off the subagent's 3K on Haiku this read as pennies."""
        assert self._analyse(True).next_turn_cost > 0.1
