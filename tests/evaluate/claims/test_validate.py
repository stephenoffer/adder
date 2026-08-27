"""The claims everything else rests on, re-tested on synthetic sessions.

`adder validate` runs these against real transcripts, where the answer depends
on the machine. These fixtures pin the *arithmetic*, so a regression in a claim
function is caught without anyone's history.
"""
from __future__ import annotations

from adder.core.trace import Session, Turn
from adder.evaluate.claims.validate import (
    a_prior_never_buys_a_downgrade,
    model_routing_is_marginal,
    replay_reproduces_measured_spend,
    run,
    session_model_choice_is_not_marginal,
    the_target_reduction_is_reachable,
)

OPUS = "claude-opus-5"


def _sessions(n=300, admit=1_500, out=700, base=30_000):
    """A 300-turn session ending near 480K context -- inside every 1M window.

    The admission rate matters: at 6K/turn this fixture ends past 1,000,000
    tokens, where Sonnet no longer fits, and the session-model claim quietly
    measures feasibility instead of price.
    """
    s = Session("s", "proj")
    ctx = base
    for i in range(n):
        if i:
            ctx += admit
        s.turns.append(Turn("s", "proj", OPUS, 0, ctx, 0, out, 0, False,
                            ts=f"2026-08-14T10:{i % 60:02d}:00Z"))
    return {"s": s}


class TestReplayFidelity:
    def test_the_null_replay_is_the_measured_bill(self):
        c = replay_reproduces_measured_spend(_sessions())
        assert c.ok, c.measured

    def test_no_data_is_neither_a_pass_nor_a_failure(self):
        """It used to be a FAIL, which is the one thing it is not.

        A failure from `adder validate` means a number this repo quotes has
        stopped holding on your data. "Your data contains none of the event"
        borrows that weight and spends it on nothing -- so the claim reports
        `untestable`, is excluded from the tally and from the exit code, and
        the run says how many were.
        """
        c = replay_reproduces_measured_spend({})
        assert c.untestable and c.status == "N/A "
        assert "FAIL" not in c.line()


class TestSessionModelClaim:
    def test_starting_cheap_is_a_large_lever(self):
        c = session_model_choice_is_not_marginal(_sessions())
        assert c.ok
        assert "modelled" in c.note, "the capability cost must not read as measured"

    def test_switching_mid_session_is_a_small_one(self):
        """The two claims are only worth stating together; they must disagree."""
        sess = _sessions()
        assert model_routing_is_marginal(sess).ok
        assert session_model_choice_is_not_marginal(sess).ok

    def test_no_data_is_neither_a_pass_nor_a_failure(self):
        c = session_model_choice_is_not_marginal({})
        assert c.untestable and "FAIL" not in c.line()


class TestSuite:
    def test_every_claim_returns_a_claim_and_never_raises(self, tmp_path):
        """An empty root must report failing claims, not crash the command."""
        for c in run(tmp_path):
            assert isinstance(c.ok, bool) and c.name and c.expected

    def test_claim_lines_are_renderable(self):
        c = replay_reproduces_measured_spend(_sessions())
        assert "PASS" in c.line() and c.name in c.line()


class TestSafetyClaims:
    """The claims that make "cheaper than not using it" checkable rather than
    asserted. These run the decision rule, so they hold on any machine."""

    def test_no_emitted_advice_costs_more_than_it_saves(self):
        """The sweep found three real counterexamples the day it was written:
        downgrade recommendations of $0.011 emitted by a turn costing $0.015."""
        from adder.evaluate.claims.validate import emitted_advice_clears_its_own_overhead

        c = emitted_advice_clears_its_own_overhead({})
        assert c.ok, c.note

    def test_the_sweep_would_notice_a_regression(self, monkeypatch):
        """A claim that cannot fail is not a check.

        Disabling the confidence gate is the mutation to make: raising the
        overhead instead would send every plan inline and the sweep would pass
        vacuously, which is exactly the kind of check that looks green forever.
        With the gate forced open, a delegation saving a billionth of a cent is
        emitted against a real routing turn, and the claim has to notice.
        """
        from adder.evaluate.claims.validate import emitted_advice_clears_its_own_overhead
        from adder.pricing.cost import Decision
        from adder.util.risk import Guarantee

        monkeypatch.setattr(Guarantee, "safe", property(lambda self: True))
        monkeypatch.setattr("adder.decide.route.policy.placement_cost",
                            lambda **kw: (1.0, 0.5, Decision(True, 1e-9, "forced")))
        assert not emitted_advice_clears_its_own_overhead({}).ok

    def test_an_empty_ledger_is_solvent(self, tmp_path, monkeypatch):
        from adder.evaluate.claims.validate import the_tool_has_paid_for_itself

        monkeypatch.setenv("ADDER_LEDGER", str(tmp_path / "none.jsonl"))
        assert the_tool_has_paid_for_itself({}).ok

    def test_carry_claim_needs_data(self):
        from adder.evaluate.claims.validate import carry_multiplier_is_above_the_assumption

        c = carry_multiplier_is_above_the_assumption({})
        assert c.untestable and c.measured == "no data"

    def test_horizon_claim_needs_data(self):
        from adder.evaluate.claims.validate import horizon_mean_exceeds_median

        assert horizon_mean_exceeds_median({}).untestable


class TestPriorNeverBuysADowngrade:
    """The safety property of right-sizing, and whether the check has teeth."""

    @staticmethod
    def _log(tmp_path, monkeypatch, rows):
        import json
        import time

        path = tmp_path / "outcomes.jsonl"
        now = time.time()
        with path.open("w") as fh:
            for tier, n, fails in rows:
                for i in range(n):
                    fh.write(json.dumps({
                        "tier": tier, "model": "m", "project": "p",
                        "escalated": i < fails, "ts": now - i * 3600,
                    }) + "\n")
        monkeypatch.setattr("adder.decide.track.outcomes.DEFAULT_LOG", path)
        return path

    def test_an_empty_log_produces_no_downgrades_at_all(self, tmp_path, monkeypatch):
        self._log(tmp_path, monkeypatch, [])
        c = a_prior_never_buys_a_downgrade({})
        assert c.ok and c.measured.startswith("0/")
        assert "0 of" in c.note, "it must say the branch never ran, not imply it passed"

    def test_an_informative_log_does_produce_them(self, tmp_path, monkeypatch):
        """Otherwise the claim is vacuous: it can only ever pass."""
        self._log(tmp_path, monkeypatch, [("T0", 60, 3), ("T1", 60, 3), ("T2", 60, 3)])
        c = a_prior_never_buys_a_downgrade({})
        assert c.ok
        descended = int(c.note.split()[0])
        assert descended > 0, "the sweep never exercised the branch it is guarding"

    def test_it_fails_when_the_permission_gate_is_removed(self, tmp_path, monkeypatch):
        """The point of the check. Break the gate; the claim must notice."""
        import adder.decide.route.policy as pol

        self._log(tmp_path, monkeypatch, [])
        monkeypatch.setattr(pol, "_may_descend", lambda *a, **k: (True, ""))
        c = a_prior_never_buys_a_downgrade({})
        assert not c.ok, "a router that downgrades on a prior must be caught"
        assert "first bad" in c.note


class TestTargetIsReachable:
    def test_a_long_session_workload_reaches_the_target(self):
        c = the_target_reduction_is_reachable(_sessions(n=400, admit=2_000))
        assert c.ok, c.measured

    def test_a_short_session_workload_honestly_does_not(self):
        """Little carry to remove means no 10x, and saying so is the right answer."""
        c = the_target_reduction_is_reachable(_sessions(n=6, admit=500))
        assert not c.ok

    def test_the_bound_is_the_frontier_not_a_search(self):
        from adder.evaluate.replay.plan import Regime, frontier, replay

        sess = _sessions(n=400, admit=2_000)
        base = replay(sess, Regime()).total
        edge = replay(sess, frontier()).total
        c = the_target_reduction_is_reachable(sess)
        assert c.measured == f"{base / edge:.1f}x"

    def test_no_data_is_neither_a_pass_nor_a_failure(self):
        assert the_target_reduction_is_reachable({}).untestable

    def test_the_target_is_a_parameter_not_a_constant(self):
        sess = _sessions(n=400, admit=2_000)
        assert the_target_reduction_is_reachable(sess, target=2.0).ok
        assert not the_target_reduction_is_reachable(sess, target=1_000.0).ok


class TestTheThirdState:
    """Absence of evidence is the one thing this suite must never report as
    evidence against, and it was reporting exactly that."""

    def test_an_untestable_claim_is_not_counted_as_passing(self):
        from adder.evaluate.claims.validate import untestable

        c = untestable("x", "no events", "<=35% kept")
        assert c.untestable and c.status == "N/A "
        # `ok` stays True so that every existing `not c.ok` reader treats it as
        # non-failing; `untestable` is what stops it being counted as a pass.
        assert c.ok

    def test_a_vacuous_truth_is_still_a_pass(self, tmp_path, monkeypatch):
        """The distinction the third state has to keep.

        Spending nothing really does mean the tool paid for itself: the claim
        is settled, trivially. Recording no compactions does not settle whether
        a compaction keeps under 35%.
        """
        from adder.evaluate.claims.validate import the_tool_has_paid_for_itself

        monkeypatch.setenv("ADDER_LEDGER", str(tmp_path / "none.jsonl"))
        c = the_tool_has_paid_for_itself({})
        assert c.ok and not c.untestable

    def test_the_run_reports_how_many_it_could_not_test(self, tmp_path, capsys):
        from adder.evaluate.claims.validate import main

        (tmp_path / "proj").mkdir()
        main([str(tmp_path)])
        out = capsys.readouterr().out
        assert "nothing here to test against" in out
        assert "[N/A ] compaction keeps less than assumed" in out

    def test_a_claim_about_this_machine_still_fails(self, tmp_path, capsys):
        """Not everything with no data is untestable.

        "there is enough local history to estimate a horizon" is *falsified* by
        an empty corpus, not left open by it -- the tool really does fall back
        to a prior, and that is a fact about this machine worth a FAIL. The
        third state is for claims a corpus leaves unsettled, not for every
        claim a corpus is small enough to disappoint.
        """
        from adder.evaluate.claims.validate import horizon_is_calibrated

        c = horizon_is_calibrated({})
        assert not c.ok and not c.untestable
