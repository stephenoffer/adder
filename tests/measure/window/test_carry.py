"""The carry model, and the restart price it used to assume.

Scoped to the closed forms and to `restart_cost`: `optimal_split` assumed every
restart rebuilt the whole prefix, which `adder prefix` measures it does not.
An assumed `W` that is 3x too high is a `k*` that is 1.7x too long, and this is
where that is held down.
"""
from __future__ import annotations

import pytest

from adder.core.trace import Session, Turn
from adder.measure.window.carry import Carry, delegate_threshold, measured_read_mult, optimal_split
from adder.measure.window.prefix import Opening, measure
from adder.pricing.prices import CACHE_READ_MULT, CACHE_WRITE_MULT

OPUS, HAIKU = "claude-opus-5", "claude-haiku-4-5"
FLOOR = 28_000


def _warm_session(sid, minute, n_turns=10, read=21_000, write=7_000):
    s = Session(sid, "proj")
    s.turns.append(Turn(sid, "proj", OPUS, 0, read, write, 300, 0, False,
                        ts=f"2026-08-14T10:{minute:02d}:00Z", ttl="1h"))
    ctx = read + write
    for i in range(1, n_turns):
        ctx += 1_000
        s.turns.append(Turn(sid, "proj", OPUS, 0, ctx, 0, 300, 0, False,
                            ts=f"2026-08-14T10:{minute:02d}:{i:02d}Z", ttl="1h"))
    return s


def _workload(n=8):
    return {f"s{i}": _warm_session(f"s{i}", i) for i in range(n)}


class TestRestartCost:
    def _k(self, *, growth=960, read_mult=0.115, **kw):
        k, _saving, _why = optimal_split(model=OPUS, floor_tokens=FLOOR,
                                         growth=growth, read_mult=read_mult, **kw)
        return k

    def test_the_default_is_still_the_cold_rebuild(self):
        """No measurement means the pessimistic assumption, unchanged."""
        r, w = 5.0, CACHE_WRITE_MULT["5m"]
        cold = (FLOOR + 2_000) * w * r / 1e6
        assert self._k() == self._k(restart_cost=cold)

    def test_a_measured_restart_shortens_the_cycle(self):
        op = measure(_workload())
        assert self._k(restart_cost=op.cost(OPUS, handoff_tokens=2_000)) < self._k()

    def test_the_cycle_moves_as_the_square_root_of_the_price(self):
        base = 0.30
        assert self._k(restart_cost=base) / self._k(restart_cost=base / 4) == \
            pytest.approx(2.0, rel=0.05)

    def test_a_free_restart_does_not_divide_by_zero(self):
        assert self._k(restart_cost=0.0) >= 1

    def test_the_explanation_says_which_one_it_used(self):
        _, _, assumed = optimal_split(model=OPUS, floor_tokens=FLOOR)
        _, _, given = optimal_split(model=OPUS, floor_tokens=FLOOR, restart_cost=0.1)
        assert "rewrites" in assumed
        assert "re-opening costs" in given

    def test_growth_and_multiplier_can_be_overridden_together(self):
        """A caller holding a measurement should not have to build a Carry to use it."""
        assert self._k(growth=10_000) < self._k(growth=100), (
            "a context filling faster is worth restarting sooner")


class TestCarryModel:
    def test_the_realized_multiplier_is_read_off_the_turns(self):
        """A fully warm workload realizes the cache-read multiplier and no more."""
        sessions = _workload()
        assert measured_read_mult(sessions, min_turns=2) == pytest.approx(
            CACHE_READ_MULT, abs=0.02)

    def test_a_workload_that_keeps_missing_realizes_more_than_the_read_rate(self):
        s = Session("s", "proj")
        for _ in range(30):
            s.turns.append(Turn("s", "proj", OPUS, 0, 10_000, 90_000, 300, 0, False))
        assert measured_read_mult({"s": s}, min_turns=2) > CACHE_READ_MULT

    def test_too_little_data_falls_back_to_the_prior(self):
        assert not Carry.measure({"s": _warm_session("s", 0)}).measured

    def test_expected_reads_without_compaction_is_the_horizon(self):
        assert Carry.default().expected_reads(400) == 400

    def test_compaction_takes_reads_off_the_horizon(self):
        c = Carry(compact_every=50, growth=1_000, survival=0.3, source="measured")
        assert c.expected_reads(400) < 400


class TestDelegateThreshold:
    def test_a_longer_horizon_lowers_the_threshold(self):
        near, _ = delegate_threshold(main_model=OPUS, sub_model=HAIKU,
                                     remaining_turns=10)
        far, _ = delegate_threshold(main_model=OPUS, sub_model=HAIKU,
                                    remaining_turns=1_000)
        assert far < near, "more re-reads to avoid means delegating sooner"

    def test_delegating_to_the_same_model_stops_paying_at_a_short_horizon(self):
        """With no re-reads to avoid, the summary and the brief are all that is left."""
        x, why = delegate_threshold(main_model=OPUS, sub_model=OPUS,
                                    remaining_turns=0)
        assert x == float("inf")
        assert "never pays" in why

    def test_a_cheaper_subagent_still_pays_even_with_no_re_reads(self):
        """Admitting at 1.25x beats reading on Haiku at 1.0x before any carry."""
        x, _ = delegate_threshold(main_model=OPUS, sub_model=HAIKU,
                                  remaining_turns=0)
        assert 0 < x < float("inf")


class TestPrefixInterop:
    def test_the_prior_opening_reproduces_the_old_assumption(self):
        """`Opening.default()` is the cold rebuild, so it must not change `k*`."""
        prior = Opening.default()
        k_assumed, _, _ = optimal_split(model=OPUS, floor_tokens=prior.floor_tokens,
                                        handoff_tokens=0, ttl="5m")
        k_prior, _, _ = optimal_split(
            model=OPUS, floor_tokens=prior.floor_tokens, handoff_tokens=0, ttl="5m",
            restart_cost=prior.cost(OPUS, ttl="5m"))
        assert k_assumed == k_prior
