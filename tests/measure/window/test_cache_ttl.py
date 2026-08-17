"""Cache TTL, fan-out, and rebuild waste — levers a price-only model cannot see."""
from __future__ import annotations

import pytest

from adder.core.trace import Session, Turn
from adder.measure.window.cache import EXPIRY_FIXABLE, EXPIRY_UNFIXABLE, analyse, ttl_recommendation
from adder.pricing.cost import cache_miss_cost, choose_ttl, effort_saving, fanout_cost

OPUS = "claude-opus-5"


def _sess(turns, gap_s=10, ttl="5m", model=OPUS, ctx=200_000):
    s = Session("s", "p")
    for i in range(turns):
        s.turns.append(Turn(
            "s", "p", model, uncached_in=0, cache_read=ctx, cache_write=0,
            out=500, thinking=0, sidechain=False,
            ts=f"2026-08-14T{10 + (i*gap_s)//3600:02d}:{((i*gap_s)//60)%60:02d}:{(i*gap_s)%60:02d}Z",
            ttl=ttl))
    return s


class TestMissCost:
    def test_rebuild_costs_more_than_a_read(self):
        assert cache_miss_cost(100_000, OPUS) > 0

    def test_one_hour_rebuild_costs_more_than_five_minute(self):
        assert cache_miss_cost(100_000, OPUS, "1h") > cache_miss_cost(100_000, OPUS, "5m")


class TestChooseTTL:
    def test_short_gaps_keep_the_cheap_ttl(self):
        ttl, _, _ = choose_ttl(500_000, OPUS, turns=100, gap_seconds=10)
        assert ttl == "5m"

    def test_long_gaps_justify_the_one_hour_ttl(self):
        ttl, saving, why = choose_ttl(500_000, OPUS, turns=100, gap_seconds=600)
        assert ttl == "1h" and saving > 0 and "5m TTL" in why

    def test_a_single_turn_never_needs_the_premium(self):
        ttl, _, _ = choose_ttl(500_000, OPUS, turns=1, gap_seconds=99_999)
        assert ttl == "5m"


class TestFanout:
    def test_parallel_calls_each_pay_the_write(self):
        par, stag, d = fanout_cost(5, 50_000, OPUS)
        assert par > stag and d and d.saving > 0

    def test_single_call_has_nothing_to_stagger(self):
        _, _, d = fanout_cost(1, 50_000, OPUS)
        assert not d

    def test_prefix_below_the_cache_minimum_does_not_cache(self):
        _, _, d = fanout_cost(5, 100, OPUS)
        assert not d and "cache minimum" in d.reason


class TestEffortSaving:
    def test_lower_effort_saves_generation_and_rereads(self):
        saved, d = effort_saving(1000, OPUS, remaining_turns=300)
        assert d and saved > 0

    def test_saving_grows_with_remaining_turns(self):
        a, _ = effort_saving(1000, OPUS, remaining_turns=10)
        b, _ = effort_saving(1000, OPUS, remaining_turns=1000)
        assert b > a

    def test_raising_effort_is_not_a_saving(self):
        _, d = effort_saving(1000, OPUS, from_effort="low", to_effort="max")
        assert not d

    def test_unknown_level_raises(self):
        with pytest.raises(ValueError):
            effort_saving(1000, OPUS, to_effort="turbo")


class TestMissAttribution:
    def test_model_switch_is_recoverable(self):
        s = Session("s", "p")
        for i, m in enumerate([OPUS, OPUS, "claude-haiku-4-5"]):
            s.turns.append(Turn("s", "p", m, 0, 100_000 if i < 2 else 0,
                                0 if i < 2 else 200_000, 500, 0, False,
                                ts=f"2026-08-14T10:00:{i:02d}Z"))
        rep = analyse({"s": s})
        assert any(m.cause == "model switch" for m in rep.misses)
        assert rep.recoverable > 0

    def test_gap_beyond_an_hour_is_not_recoverable(self):
        s = Session("s", "p")
        s.turns.append(Turn("s", "p", OPUS, 0, 100_000, 0, 500, 0, False,
                            ts="2026-08-14T10:00:00Z", ttl="1h"))
        s.turns.append(Turn("s", "p", OPUS, 0, 0, 200_000, 500, 0, False,
                            ts="2026-08-14T14:00:00Z", ttl="1h"))
        rep = analyse({"s": s})
        assert any(m.cause == EXPIRY_UNFIXABLE for m in rep.misses)
        assert rep.recoverable == 0

    def test_five_minute_gap_within_an_hour_is_fixable(self):
        s = Session("s", "p")
        s.turns.append(Turn("s", "p", OPUS, 0, 100_000, 0, 500, 0, False,
                            ts="2026-08-14T10:00:00Z", ttl="5m"))
        s.turns.append(Turn("s", "p", OPUS, 0, 0, 200_000, 500, 0, False,
                            ts="2026-08-14T10:20:00Z", ttl="5m"))
        rep = analyse({"s": s})
        assert any(m.cause == EXPIRY_FIXABLE for m in rep.misses)

    def test_hit_rate_is_a_share_of_cacheable_tokens(self):
        rep = analyse({"s": _sess(10)})
        assert rep.hit_rate == pytest.approx(1.0)

    def test_already_on_one_hour_is_not_told_to_switch(self):
        s = _sess(5, ttl="1h")
        _ttl, saving, why = ttl_recommendation({"s": s})
        assert "already use the 1h TTL" in why or saving >= 0


class TestTheTTLReasonIsTrue:
    """The recommendation was right and its explanation was not.

    With a single turn the short TTL wins -- there is no second turn, so no
    cache read to lose, and only the cheaper write is left. The reason printed
    for that was "turns are 600s apart, inside the 5m TTL", which contradicts
    itself inside one sentence: 600s is not inside a 300s TTL. A `Decision` that
    explains itself falsely is worse than one that says nothing, because the
    reader takes a fact away from it.
    """

    def test_a_gap_past_the_ttl_is_never_described_as_inside_it(self):
        _, _, why = choose_ttl(100_000, OPUS, turns=1, gap_seconds=600)
        assert "inside" not in why

    def test_it_says_why_the_short_ttl_still_won(self):
        _, _, why = choose_ttl(100_000, OPUS, turns=1, gap_seconds=600)
        assert "past the 5m TTL" in why
        assert "never re-reads" in why

    def test_the_recommendation_itself_is_unchanged(self):
        ttl, _, _ = choose_ttl(100_000, OPUS, turns=1, gap_seconds=600)
        assert ttl == "5m"

    def test_a_gap_inside_the_ttl_still_says_so(self):
        _, _, why = choose_ttl(100_000, OPUS, turns=10, gap_seconds=60)
        assert "inside the 5m TTL" in why

    def test_a_long_session_past_the_ttl_still_recommends_1h(self):
        ttl, _, why = choose_ttl(100_000, OPUS, turns=10, gap_seconds=600)
        assert ttl == "1h"
        assert "expires every turn" in why
