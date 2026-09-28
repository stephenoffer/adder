"""Trajectory simulation: validates the multiplicative composition approximation.

The headline savings figure depends on that approximation, so the property that
matters is DIRECTION of error: under-predicting is safe, over-predicting inflates
the claim.
"""
from __future__ import annotations

import pytest

from adder.core.trace import Session, Turn
from adder.evaluate.replay.simulate import Intervention, admissions, evaluate, simulate

OPUS = "claude-opus-5"


def _sess(n: int, step: int = 5_000, base: int = 25_000, out: int = 500) -> Session:
    s = Session("s", "p")
    for i in range(n):
        s.turns.append(Turn("s", "p", OPUS, uncached_in=0, cache_read=base + step * i,
                            cache_write=0, out=out, thinking=0, sidechain=False))
    return s


class TestAdmissions:
    def test_first_turn_admits_nothing_incremental(self):
        _, adm = admissions(_sess(5))
        assert adm[0] == 0.0

    def test_baseline_is_the_minimum_context(self):
        base, _ = admissions(_sess(10, base=30_000))
        assert base == 30_000

    def test_steady_growth_is_recovered(self):
        _, adm = admissions(_sess(5, step=5_000))
        assert all(a == 5_000 for a in adm[1:])

    def test_empty_session_is_safe(self):
        assert admissions(Session("s", "p")) == (0, [])


class TestSimulate:
    def test_baseline_matches_actual_read_cost(self):
        s = _sess(100)
        expected = sum(t.context for t in s.turns) * 5 * 0.1 / 1e6
        assert simulate(s, Intervention()) == pytest.approx(expected, rel=1e-6)

    def test_terseness_reduces_cost(self):
        s = _sess(200)
        assert simulate(s, Intervention(terseness=0.5)) < simulate(s, Intervention())

    def test_total_terseness_leaves_only_baseline(self):
        s = _sess(50)
        floor = 25_000 * 50 * 5 * 0.1 / 1e6
        assert simulate(s, Intervention(terseness=1.0)) == pytest.approx(floor, rel=1e-6)

    def test_splitting_reduces_cost(self):
        s = _sess(600)
        assert simulate(s, Intervention(split_turns=100)) < simulate(s, Intervention())

    def test_shorter_splits_save_more(self):
        s = _sess(600)
        assert (simulate(s, Intervention(split_turns=50))
                < simulate(s, Intervention(split_turns=300)))

    def test_no_split_on_short_session(self):
        s = _sess(50)
        assert simulate(s, Intervention(split_turns=300)) == pytest.approx(
            simulate(s, Intervention()))

    def test_levers_stack(self):
        s = _sess(600)
        both = simulate(s, Intervention(terseness=0.3, split_turns=300))
        assert both < simulate(s, Intervention(terseness=0.3))
        assert both < simulate(s, Intervention(split_turns=300))

    def test_delegation_with_no_compression_saves_nothing(self):
        s = _sess(300)
        assert simulate(s, Intervention(delegation=0.5, summary_ratio=1.0)) == \
               pytest.approx(simulate(s, Intervention()))


class TestCompositionApproximation:
    """The property the headline number depends on."""

    def test_approximation_does_not_overstate(self):
        sessions = {f"s{i}": _sess(400 + 100 * i) for i in range(4)}
        rows = evaluate(sessions, [
            Intervention(terseness=0.30),
            Intervention(terseness=0.30, delegation=0.25),
            Intervention(terseness=0.30, delegation=0.25, split_turns=300),
        ])
        for iv, sim, pred in rows:
            assert pred <= sim * 1.05, (
                f"{iv.label}: prediction ${pred:,.0f} overstates simulated ${sim:,.0f}")

    def test_pool_fraction_is_bounded(self):
        for iv in (Intervention(terseness=1.0, delegation=1.0),
                   Intervention(), Intervention(terseness=0.5)):
            assert 0.0 <= iv.pool_fraction <= 1.0

    def test_labels_describe_the_intervention(self):
        assert Intervention().label == "baseline"
        assert "terse" in Intervention(terseness=0.3).label
        assert "split@300" in Intervention(split_turns=300).label


class TestTersenessOnlyTouchesOutput:
    """`savings` prices terseness at t x the measured output share of growth.
    This simulated t x all of it, so the claim that the composition model
    never overstates was checking a lever `savings` does not use."""

    def test_the_prediction_scales_with_the_share(self):
        from adder.evaluate.replay.simulate import Intervention

        assert Intervention(terseness=0.3, output_share=0.5).pool_fraction == \
            pytest.approx(0.15)

    def test_the_simulation_uses_the_measured_share(self, make_sessions, monkeypatch):
        from adder.evaluate.replay import simulate as sim

        sessions = make_sessions(3, 60)
        iv = sim.Intervention(terseness=0.3)
        monkeypatch.setattr("adder.measure.spend.debt.output_share_of_growth",
                            lambda s: 1.0)
        (_, all_output, _), = sim.evaluate(sessions, [iv])
        monkeypatch.setattr("adder.measure.spend.debt.output_share_of_growth",
                            lambda s: 0.5)
        (got, half, _), = sim.evaluate(sessions, [iv])
        assert got.output_share == 0.5
        assert half == pytest.approx(all_output / 2, rel=1e-6)
