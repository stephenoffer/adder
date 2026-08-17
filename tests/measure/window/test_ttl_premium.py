"""The cost of moving 5m writes to 1h is priced off the 5m writes.

`ttl_recommendation` charged the switch at `write_cost / total_written` -- a
blend across both TTL buckets. The 1h tokens in that blend already pay the 2.00x
premium, so the blended rate is above what the 5m tokens actually paid, and the
premium of switching them came out too high. The bias grows with the share
already on 1h, which is backwards: those are the workloads with the least left
to convert and the least reason to be discouraged from converting it.

Priced off `ttl_write_cost["5m"]` instead, which is what that population paid.
"""
from __future__ import annotations

import pytest

from adder.core.trace import Session, Turn
from adder.measure.window.cache import analyse

OPUS = "claude-opus-5"


def _turn(ttl, i, *, write=200_000, read=10_000):
    return Turn("s", "p", OPUS, uncached_in=0, cache_read=read, cache_write=write,
                out=100, thinking=0, sidechain=False, ttl=ttl,
                ts=f"2026-08-10T{12 + i // 4:02d}:{(i * 10) % 60:02d}:00Z")


def _mixed_session(n_1h=8, n_5m=4):
    s = Session("s", "p")
    s.turns = ([_turn("1h", i) for i in range(n_1h)]
               + [_turn("5m", i) for i in range(n_1h, n_1h + n_5m)])
    return {"s": s}


class TestWriteSpendIsSplitByTtl:
    def test_both_buckets_are_recorded(self):
        rep = analyse(_mixed_session())
        assert rep.ttl_write_cost["1h"] > 0
        assert rep.ttl_write_cost["5m"] > 0

    def test_the_split_reconciles_with_the_total(self):
        rep = analyse(_mixed_session())
        assert sum(rep.ttl_write_cost.values()) == pytest.approx(rep.write_cost)

    def test_a_1h_token_costs_more_than_a_5m_token(self):
        """The premium the recommendation is about: 2.00x against 1.25x."""
        rep = analyse(_mixed_session())
        per_1h = rep.ttl_write_cost["1h"] / rep.ttl_tokens["1h"]
        per_5m = rep.ttl_write_cost["5m"] / rep.ttl_tokens["5m"]
        assert per_1h == pytest.approx(per_5m * 2.00 / 1.25)


class TestTheBlendedRateWasTheBug:
    def test_the_blend_overstates_what_5m_writes_paid(self):
        rep = analyse(_mixed_session())
        blended = rep.write_cost / sum(rep.ttl_tokens.values())
        actual_5m = rep.ttl_write_cost["5m"] / rep.ttl_tokens["5m"]
        assert blended > actual_5m

    def test_the_overstatement_grows_with_the_1h_share(self):
        """Which is the wrong way round, and why this mattered."""
        def bias(n_1h):
            rep = analyse(_mixed_session(n_1h=n_1h, n_5m=4))
            blended = rep.write_cost / sum(rep.ttl_tokens.values())
            return blended / (rep.ttl_write_cost["5m"] / rep.ttl_tokens["5m"])
        assert bias(16) > bias(8) > bias(2)

    def test_a_pure_5m_workload_has_no_blend_to_get_wrong(self):
        rep = analyse(_mixed_session(n_1h=0, n_5m=6))
        blended = rep.write_cost / sum(rep.ttl_tokens.values())
        actual_5m = rep.ttl_write_cost["5m"] / rep.ttl_tokens["5m"]
        assert blended == pytest.approx(actual_5m)
