"""A benchmark that flatters the tool is worse than no benchmark.

The thing being guarded here is not the arithmetic -- `test_plan.py` covers the
replay -- it is the *separation*. The headline is only honest if the rungs
nothing enforces stay labelled as such, and if the guard's threshold is the one
the shipped hook would actually use rather than a number picked to look good.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from adder.core.trace import Session, Turn
from adder.evaluate.replay.bench import (
    GUARD_MIN_TOKENS,
    corner_sweep,
    expected_reads,
    guard_threshold,
    ladder,
    run,
)
from adder.evaluate.replay.plan import Regime, prepare, replay
from adder.pricing.cost import admitted_token_cost

OPUS = "claude-opus-5"
REPO = Path(__file__).resolve().parents[2]


def _sessions(n=3, n_turns=120, admit=6_000, out=600, base=20_000, model=OPUS):
    out_sessions = {}
    for k in range(n):
        s = Session(f"s{k}", "proj")
        ctx = base
        for i in range(n_turns):
            if i:
                ctx += admit
            s.turns.append(Turn(f"s{k}", "proj", model, 0, ctx, 0, out, 0, False,
                                ts=f"2026-08-14T10:{i % 60:02d}:00Z"))
        out_sessions[f"s{k}"] = s
    return out_sessions


class TestGuardThreshold:
    def test_the_token_floor_is_the_hooks_floor(self):
        """The benchmark must price the guard people actually run.

        This used to parse the hook's source for a `MIN_TOKENS` assignment,
        because the hook was exec'd rather than imported and the constant was
        genuinely duplicated. It is not any more: the threshold lives in
        `adder.decide.guard.Settings`, which the hook resolves at run time, so
        the drift check is a direct comparison against the same default the
        guard uses.
        """
        from adder.decide.guard import Settings

        assert Settings().min_tokens == GUARD_MIN_TOKENS, (
            "bench.GUARD_MIN_TOKENS has drifted from the guard's floor; "
            "the benchmark would be pricing a guard nobody is running")

    def test_the_dollar_gate_resolves_to_the_size_that_costs_that_much(self):
        thr = guard_threshold(remaining_turns=900, model=OPUS, min_cost=0.25,
                              min_tokens=0)
        assert admitted_token_cost(thr, OPUS, 900) == pytest.approx(0.25, rel=0.01)

    def test_the_floor_wins_when_the_gate_falls_below_it(self):
        """A dollar gate that resolves under the floor cannot fire; quoting it lies."""
        assert guard_threshold(remaining_turns=10_000, model=OPUS,
                               min_cost=0.25) == GUARD_MIN_TOKENS

    def test_a_shorter_horizon_raises_the_threshold(self):
        """Fewer re-reads to avoid means a read has to be bigger to be worth moving."""
        near = guard_threshold(remaining_turns=50, model=OPUS, min_cost=5.0, min_tokens=0)
        far = guard_threshold(remaining_turns=500, model=OPUS, min_cost=5.0, min_tokens=0)
        assert near > far

    def test_with_no_re_reads_left_the_write_alone_still_sets_a_threshold(self):
        """A token admitted on the last turn is never re-read, but it is still
        written. The threshold there is the cache write's break-even, not infinity."""
        thr = guard_threshold(remaining_turns=0, model=OPUS, min_cost=0.25, min_tokens=0)
        assert math.isfinite(thr)
        assert admitted_token_cost(thr, OPUS, 0) == pytest.approx(0.25, rel=0.01)

    def test_a_zero_priced_model_falls_back_to_the_floor(self, monkeypatch):
        """Division by a free model's rate must not raise inside a report."""
        monkeypatch.setattr("adder.evaluate.replay.bench.admitted_token_cost", lambda *a, **k: 0.0)
        assert guard_threshold(remaining_turns=100, model=OPUS, min_cost=0.25,
                               min_tokens=777) == 777


class TestExpectedReads:
    def test_it_is_positive_on_any_workload_with_turns(self):
        assert expected_reads(_sessions()) >= 1

    def test_an_empty_workload_falls_back_rather_than_failing(self):
        assert expected_reads({"s": Session("s", "p")}) >= 1


class TestLadder:
    def test_every_rung_is_at_least_as_cheap_as_the_one_above(self):
        """Cumulative rungs that get more expensive mean a lever is mispriced."""
        sess = _sessions()
        prep = prepare(sess)
        totals = [replay(prep, c.regime).total for c in ladder(sess, min_cost=0.25)]
        assert totals == sorted(totals, reverse=True), totals

    def test_the_first_rung_is_the_workload_as_run(self):
        sess = _sessions()
        base = ladder(sess, min_cost=0.25)[0]
        measured = sum(s.cost_on(None) for s in sess.values())
        assert base.enforced and base.regime.delegate_above is None
        assert replay(sess, base.regime).total == pytest.approx(measured, rel=1e-9)

    def test_the_advice_rung_is_not_marked_enforced(self):
        """The whole point of the split. If this flips, the headline overstates."""
        rungs = ladder(_sessions(), min_cost=0.25)
        assert not rungs[-1].enforced
        assert all(c.enforced for c in rungs[:-1])

    def test_the_enforced_rungs_use_the_guard_threshold(self):
        sess = _sessions()
        rungs = ladder(sess, min_cost=0.25)
        thr = guard_threshold(remaining_turns=expected_reads(sess), model=OPUS,
                              min_cost=0.25)
        assert rungs[1].regime.delegate_above == thr

    def test_only_the_advice_rung_restarts_sessions(self):
        """Nothing in this repo makes a session restart. If a rung marked enforced
        splits, the 'installed and changed nothing' number is fiction."""
        for cfg in ladder(_sessions(), min_cost=0.25):
            if cfg.enforced:
                assert cfg.regime.split_turns is None


class TestCornerSweep:
    def test_the_nominal_case_is_never_worse_than_the_worst_corner(self):
        sess = _sessions()
        prep = prepare(sess)
        reg = Regime(delegate_above=1_000, right_size=True, split_turns=50)
        base = replay(prep, Regime()).total
        sweep = corner_sweep(prep, reg, base)
        nominal = base / replay(prep, reg).total
        assert min(m for _i, m in sweep) <= nominal + 1e-9

    def test_a_worse_summary_ratio_never_helps(self):
        """More of the read handed back means less carry avoided, always."""
        sess = _sessions()
        prep = prepare(sess)
        base = replay(prep, Regime()).total
        by_ratio = {}
        for (sr, pf, ho), mult in corner_sweep(
                prep, Regime(delegate_above=1_000, right_size=True, split_turns=50), base):
            if (pf, ho) == (0.15, 2_000):
                by_ratio[sr] = mult
        assert by_ratio[0.30] < by_ratio[0.10]


class TestRun:
    def test_the_replay_reproduces_the_measured_bill(self):
        b = run(_sessions())
        assert b.residual == pytest.approx(0.0, abs=1e-9)

    def test_installed_is_read_off_the_last_enforced_rung(self):
        b = run(_sessions())
        last = max(i for i, c in enumerate(b.configs) if c.enforced)
        assert b.installed == pytest.approx(b.multiple(last))

    def test_following_the_advice_beats_only_installing_it(self):
        b = run(_sessions())
        assert b.followed >= b.installed

    def test_the_worst_corner_never_exceeds_the_nominal(self):
        b = run(_sessions())
        assert b.worst_corner <= b.followed + 1e-9

    def test_multiples_are_finite_and_at_least_one(self):
        b = run(_sessions())
        for i in range(len(b.results)):
            assert math.isfinite(b.multiple(i)) and b.multiple(i) >= 1.0 - 1e-9
