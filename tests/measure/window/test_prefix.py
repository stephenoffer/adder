"""A restart is priced off what an opening actually cost, or it is priced wrong.

The two failure modes this file exists to catch, because both were shipped:
a restart that costs nothing (an optimiser will take infinitely many), and a
restart that costs a full prefix rebuild (an optimiser will refuse to take one).
"""
from __future__ import annotations

import pytest

from adder.core.trace import Session, Turn
from adder.measure.window.prefix import (
    DEFAULT_HANDOFF,
    MIN_OPENINGS,
    Opening,
    cadence,
    measure,
    warmth_by_gap,
    weighted_median_turns,
)
from adder.pricing.prices import CACHE_WRITE_MULT

OPUS = "claude-opus-5"


def _session(sid, *, start_min, n_turns=10, floor=28_000, warm=21_000,
             write=7_000, model=OPUS, growth=1_000):
    """One session opening at `start_min` minutes past the hour, warm by default."""
    s = Session(sid, "proj")
    s.turns.append(Turn(sid, "proj", model, 0, warm, write, 300, 0, False,
                        ts=f"2026-08-14T10:{start_min:02d}:00Z", ttl="1h"))
    ctx = floor
    for i in range(1, n_turns):
        ctx += growth
        s.turns.append(Turn(sid, "proj", model, 0, ctx, 0, 300, 0, False,
                            ts=f"2026-08-14T10:{start_min:02d}:{i:02d}Z", ttl="1h"))
    return s


def _workload(n=8, **kw):
    """Sessions one minute apart, so every opening but the first follows a turn."""
    return {f"s{i}": _session(f"s{i}", start_min=i, **kw) for i in range(n)}


class TestMeasurement:
    def test_an_opening_is_measured_as_read_plus_written(self):
        op = measure(_workload())
        assert op.measured
        assert op.floor_tokens == 28_000
        assert op.read_tokens == pytest.approx(21_000, abs=2)
        assert op.write_tokens == pytest.approx(7_000, abs=2)
        assert op.warm_share == pytest.approx(0.75, abs=0.01)

    def test_the_split_adds_back_to_the_opening_context(self):
        """Three independent medians need not sum to one; a split that does not add
        up prices a restart against a session that never existed."""
        op = measure(_workload())
        assert op.read_tokens + op.write_tokens + op.uncached_tokens == op.floor_tokens

    def test_a_cold_workload_is_not_reported_as_warm(self):
        op = measure(_workload(warm=0, write=28_000))
        assert op.warm_share == 0.0
        assert op.warm_openings == 0

    def test_too_few_openings_falls_back_to_the_prior(self):
        assert not measure(_workload(n=MIN_OPENINGS - 1)).measured

    def test_the_prior_is_the_pessimistic_one(self):
        """With no data the answer must be 'restarting is expensive', not 'cheap'."""
        prior = Opening.default()
        assert not prior.measured
        assert prior.cost(OPUS) == pytest.approx(prior.rebuild_cost(OPUS))
        assert prior.discount(OPUS) == pytest.approx(1.0)

    def test_an_opening_after_a_long_gap_is_not_used_as_evidence(self):
        """Warmth after an hour has no explanation, so it is not what this counts."""
        near = _workload(n=6)
        far = dict(near)
        far["late"] = _session("late", start_min=59, n_turns=10)
        # The late session opens 50 minutes after anything else ran, so it is
        # excluded and the measurement is unchanged.
        assert measure(far).openings == measure(near).openings

    def test_a_session_with_one_turn_is_not_a_restart(self):
        assert measure(_workload(n_turns=1)).openings == 0


class TestPricing:
    def test_a_warm_restart_is_cheaper_than_a_rebuild(self):
        op = measure(_workload())
        assert op.cost(OPUS) < op.rebuild_cost(OPUS)
        assert op.discount(OPUS) > 1.5

    def test_the_handoff_is_charged_at_the_write_rate(self):
        op = measure(_workload())
        extra = op.cost(OPUS, handoff_tokens=10_000) - op.cost(OPUS)
        assert extra == pytest.approx(10_000 * 5 * CACHE_WRITE_MULT["1h"] / 1e6)

    def test_a_negative_handoff_cannot_discount_a_restart(self):
        op = measure(_workload())
        assert op.cost(OPUS, handoff_tokens=-99_999) == pytest.approx(op.cost(OPUS))

    def test_the_ttl_moves_the_written_part_only(self):
        op = measure(_workload())
        assert op.cost(OPUS, ttl="1h") > op.cost(OPUS, ttl="5m")


class TestCadence:
    def _k(self, warm, handoff=DEFAULT_HANDOFF):
        op = measure(_workload())
        k, _at, _never = cadence(op, model=OPUS, growth=960, read_mult=0.115,
                                 handoff_tokens=handoff, warm=warm)
        return k

    def test_a_cheaper_restart_shortens_the_cycle(self):
        assert self._k(warm=True) < self._k(warm=False)

    def test_it_shortens_it_as_a_square_root_not_linearly(self):
        """k* = sqrt(2W/(m*r*g)), so a 3x cheaper restart is a 1.7x shorter cycle."""
        op = measure(_workload())
        ratio_w = op.rebuild_cost(OPUS, handoff_tokens=DEFAULT_HANDOFF) / op.cost(
            OPUS, handoff_tokens=DEFAULT_HANDOFF)
        assert self._k(warm=False) / self._k(warm=True) == pytest.approx(
            ratio_w ** 0.5, rel=0.1)

    def test_a_bigger_handoff_means_restarting_less_often(self):
        assert self._k(warm=True, handoff=50_000) > self._k(warm=True)

    def test_restarting_at_the_optimum_beats_a_long_session(self):
        op = measure(_workload())
        k, at_k, never = cadence(op, model=OPUS, growth=960, read_mult=0.115,
                                 observed_turns=536)
        assert 0 < at_k < never
        assert k < 536

    def test_the_comparison_horizon_is_the_one_it_is_given(self):
        """Quoting against a horizon nobody reaches would flatter the result."""
        op = measure(_workload())
        _, _, short = cadence(op, model=OPUS, growth=960, read_mult=0.115,
                              observed_turns=50)
        _, _, long_ = cadence(op, model=OPUS, growth=960, read_mult=0.115,
                              observed_turns=1_000)
        assert long_ > short


class TestEvidence:
    def test_warmth_is_bucketed_by_how_long_before_the_opening(self):
        rows = {b: (n, share) for b, n, share in warmth_by_gap(_workload())}
        assert rows["nothing before"][0] == 1, "the first session has nothing before it"
        assert rows["<5m"][0] == 7
        assert rows["<5m"][1] == pytest.approx(0.75, abs=0.01)

    def test_the_horizon_is_cost_weighted(self):
        """Short sessions are numerous and nearly free; they must not set the horizon."""
        sessions = {f"s{i}": _session(f"s{i}", start_min=i, n_turns=3)
                    for i in range(20)}
        sessions["big"] = _session("big", start_min=30, n_turns=400, growth=50_000)
        assert weighted_median_turns(sessions) == 400

    def test_no_sessions_is_not_a_horizon(self):
        assert weighted_median_turns({}) == 0


class TestTheOpeningIsTheConversationsOpening:
    """A session that begins with a delegated turn opened with the parent's prefix.

    Every measurement here keys off `turns[0]`, and this module's whole output
    is what re-opening a conversation costs. When the first record is a subagent
    turn -- 4 of 105 sessions on this corpus -- it measured that subagent's
    small, cold context instead: a floor no main-chain turn ever had, on
    whatever cheap model the subagent ran.
    """

    @staticmethod
    def _turn(ctx, *, side=False, read=0, write=0, model="claude-opus-5", minute=0):
        return Turn("s", "p", model, uncached_in=0, cache_read=read or ctx,
                    cache_write=write, out=10, thinking=0, sidechain=side,
                    ts=f"2026-08-10T12:{minute:02d}:00Z")

    def _sess(self, turns):
        s = Session("s", "p")
        s.turns = turns
        return s

    def test_a_leading_subagent_turn_does_not_set_the_floor(self):
        withsub = self._sess([
            self._turn(4_000, side=True, model="claude-haiku-4-5", minute=0),
            self._turn(80_000, minute=1),
            self._turn(90_000, minute=2)])
        plain = self._sess([self._turn(80_000, minute=1),
                            self._turn(90_000, minute=2)])
        assert Opening.from_session(withsub).floor_tokens == \
            Opening.from_session(plain).floor_tokens

    def test_the_floor_is_the_first_main_chain_context(self):
        s = self._sess([self._turn(4_000, side=True, minute=0),
                        self._turn(80_000, minute=1)])
        assert Opening.from_session(s).floor_tokens == 80_000

    def test_a_session_with_no_delegation_is_unchanged(self):
        s = self._sess([self._turn(80_000, minute=0), self._turn(90_000, minute=1)])
        assert Opening.from_session(s).floor_tokens == 80_000

    def test_an_empty_session_still_returns_the_default(self):
        assert Opening.from_session(self._sess([])).source != "measured"


class TestTheWeightedMedianIsIndexedOnTheMainChain:
    """The number goes straight into a horizon built from main-chain lengths.

    `Horizon.from_sessions` counts `main_turns` for a measured reason it
    states: a subagent turn does not re-read the main context, so counting it
    asks where a session sits "using a ruler it was not measured with" -- 716
    records for a 207-turn conversation, on the corpus that was measured.
    `bench.expected_reads` feeds this straight into `mean_remaining`, and
    `memory.Pricing.turns` needs it for the same reason: a resident token is
    re-read once per turn of the conversation, not once per subagent step.
    """

    def _sessions(self, make_turn):
        from adder.core.trace import Session

        s = Session("s", "p")
        s.turns = [make_turn(session="s", minutes=i) for i in range(20)]
        s.turns += [make_turn(session="s", sidechain=True, minutes=100 + i)
                    for i in range(60)]
        return {"s": s}

    def test_subagent_turns_do_not_lengthen_the_session(self, make_turn):
        from adder.measure.window.prefix import weighted_median_turns

        assert weighted_median_turns(self._sessions(make_turn)) == 20

    def test_it_agrees_with_the_horizon_it_indexes(self, make_turn):
        from adder.measure.session.horizon import Horizon
        from adder.measure.window.prefix import weighted_median_turns

        sessions = self._sessions(make_turn)
        h = Horizon.from_sessions(sessions, min_turns=1)
        assert weighted_median_turns(sessions) in h.lengths

    def test_a_session_with_no_delegation_is_unchanged(self, make_sessions):
        from adder.measure.window.prefix import weighted_median_turns

        assert weighted_median_turns(make_sessions(3, 40)) == 40
