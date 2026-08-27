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

    def test_the_cadence_rung_is_never_marked_enforced(self):
        """The whole point of the split. If this flips, the headline overstates.

        The bottom rung used to bundle the delegation threshold with the
        restart cadence, and both were advice. An enforcing guard refuses the
        reads, so the threshold can cross the line -- the cadence cannot, ever,
        because nothing in this repository can restart a session."""
        rungs = ladder(_sessions(), min_cost=0.25)
        assert not rungs[-1].enforced
        assert rungs[-1].regime.split_turns is not None

    def test_the_advisory_guard_does_not_enforce_the_solved_threshold(self,
                                                                      monkeypatch):
        """The default is still advisory, and the report must say so."""
        monkeypatch.setenv("ADDER_GUARD_ENFORCE", "off")
        rungs = ladder(_sessions(), min_cost=0.25)
        solved = [c for c in rungs if c.regime.label == "solved"]
        assert solved and not solved[0].enforced

    def test_enforcement_cannot_claim_a_threshold_the_guard_would_not_refuse_at(
            self, monkeypatch):
        """The guard refuses at its own floor, which is above the one the
        reports solve for. Marking that rung enforced would credit activation
        with money it does not collect."""
        monkeypatch.setenv("ADDER_GUARD_ENFORCE", "full")
        monkeypatch.setenv("ADDER_GUARD_MIN_TOKENS", "100000")
        rungs = ladder(_sessions(), min_cost=0.25, min_tokens=100_000)
        solved = [c for c in rungs if c.regime.label == "solved"]
        assert solved and not solved[0].enforced

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


class TestTheDuplicateRung:
    """What `guard_enforce=certain` is worth, on the report someone reads
    before installing anything.

    The duplicate refusal is the only rung here with no modelled input behind
    it: no summary ratio, no p_fail, no handoff. The call does not run.
    """

    def _corpus(self, tmp_path, results):
        import json

        d = tmp_path / "proj"
        d.mkdir(parents=True, exist_ok=True)
        recs, ctx = [], 20_000
        for i, (cmd, out) in enumerate(results):
            ctx += len(out) // 4 + 300
            recs.append({
                "type": "assistant", "sessionId": "s", "cwd": "/repo",
                "timestamp": f"2026-08-01T10:{i:02d}:00Z",
                "message": {"id": f"m{i}", "model": "claude-opus-5",
                            "usage": {"input_tokens": 1, "cache_read_input_tokens": ctx,
                                      "cache_creation_input_tokens": 2_000,
                                      "output_tokens": 300},
                            "content": [{"type": "tool_use", "id": f"u{i}",
                                         "name": "Bash", "input": {"command": cmd}}]}})
            recs.append({
                "type": "user", "sessionId": "s", "cwd": "/repo",
                "timestamp": f"2026-08-01T10:{i:02d}:30Z",
                "message": {"content": [{"type": "tool_result",
                                         "tool_use_id": f"u{i}", "content": out}]}})
        (d / "s.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
        return tmp_path

    def test_the_shell_re_reads_are_dropped_and_the_rest_re_priced(self, tmp_path):
        from adder.core.trace import load_sessions
        from adder.evaluate.replay.bench import duplicate_admissions, run

        root = self._corpus(tmp_path, [("cat /a.py", "x" * 40_000)] * 4)
        sessions = load_sessions(root)
        dups = duplicate_admissions(root)
        assert sum(dups.values()) > 0

        b = run(sessions, dups=dups, corners=False)
        assert b.configs[1].regime.refuse_duplicates
        assert b.results[1].refused_tokens > 0
        assert b.results[1].total < b.results[0].total

    def test_without_a_measurement_the_rung_is_absent(self, tmp_path):
        """No transcript to measure, no row -- rather than a row of zeroes that
        reads like a lever worth nothing."""
        from adder.core.trace import load_sessions
        from adder.evaluate.replay.bench import run

        root = self._corpus(tmp_path, [(f"cat /{i}.py", "x" * 40_000) for i in range(4)])
        b = run(load_sessions(root), corners=False)
        assert not any(c.regime.refuse_duplicates for c in b.configs)

    def test_certain_counts_as_enforced_and_full_is_not_required(self, monkeypatch):
        """`_enforcing` asked whether the level was `full`, so `certain` -- what
        `adder auto on` installs -- landed on the unenforced side of a report
        about what installing gets you."""
        import adder.evaluate.replay.bench as mod

        monkeypatch.setattr(mod, "_enforce_level", lambda: "certain")
        assert not mod._enforcing()

        monkeypatch.setattr(mod, "_enforce_level", lambda: "full")
        assert mod._enforcing()

    def test_installed_stops_at_the_first_unenforced_rung(self):
        """Rungs are cumulative, so `max(enforced)` would let an unenforced row
        below an enforced one into a number whose whole claim is that nothing
        has to be obeyed."""
        from adder.evaluate.replay.bench import Bench, Config
        from adder.evaluate.replay.plan import Regime, Result

        cfgs = [Config("a", Regime(), enforced=True),
                Config("b", Regime(), enforced=False),
                Config("c", Regime(), enforced=True)]
        results = [Result(regime=Regime(), main_input=100.0),
                   Result(regime=Regime(), main_input=50.0),
                   Result(regime=Regime(), main_input=25.0)]
        b = Bench(cfgs, results, measured=100.0, sessions=1, corners=[])
        assert b.installed == 1.0
