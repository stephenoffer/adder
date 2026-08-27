"""A regime has to be priced on both sides, or the total is fiction."""
from __future__ import annotations

from datetime import date

import pytest

from adder.core.trace import Session, Turn
from adder.decide.route.classify import Tier
from adder.evaluate.replay.plan import (
    BRIEF_TOKENS,
    GRID,
    Regime,
    cheapest_tier,
    frontier,
    ladder,
    prepare,
    recommended_cadence,
    recommended_threshold,
    replay,
    solve,
)

OPUS, SONNET, HAIKU = "claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"
ON = None


def _sessions(n_turns=200, admit=4_000, out=800, base=20_000, model=OPUS):
    """One synthetic session with a linear context trajectory."""
    s = Session("s", "proj")
    ctx = base
    for i in range(n_turns):
        if i:
            ctx += admit
        # Everything after the first turn is a cache read; that is what a real
        # warm session looks like and it keeps the fixture's arithmetic legible.
        s.turns.append(Turn("s", "proj", model, 0, ctx, 0, out, 0, False,
                            ts=f"2026-08-14T10:{i % 60:02d}:00Z"))
    return {"s": s}


class TestBaselineFidelity:
    def test_the_null_regime_reproduces_the_measured_cost(self):
        """If the replay cannot reproduce what happened, no multiple off it means anything."""
        sess = _sessions()
        measured = sum(s.cost_on(ON) for s in sess.values())
        assert replay(sess, Regime()).total == pytest.approx(measured, rel=1e-9)

    def test_a_compaction_does_not_run_the_replay_away_from_reality(self):
        """Context that drops must drop in the replay too, or later turns are fiction."""
        s = Session("s", "proj")
        for ctx in [50_000, 90_000, 130_000, 40_000, 70_000, 100_000]:
            s.turns.append(Turn("s", "proj", OPUS, 0, ctx, 0, 500, 0, False))
        sess = {"s": s}
        measured = sum(x.cost_on(ON) for x in sess.values())
        assert replay(sess, Regime()).total == pytest.approx(measured, rel=1e-9)

    def test_an_empty_session_is_priced_at_zero(self):
        assert replay({"s": Session("s", "p")}, Regime()).total == 0.0


class TestLevers:
    def test_delegation_is_charged_on_both_sides(self):
        """`simulate` charges nothing for the subagent. A total that does that is wrong."""
        sess = _sessions(admit=40_000)
        res = replay(sess, Regime(delegate_above=5_000))
        assert res.delegated > 0
        assert res.sub_run > 0, "somebody still had to read those tokens"
        assert res.sub_escalation > 0, "and some of those runs get redone"

    def test_delegation_below_the_threshold_never_fires(self):
        res = replay(_sessions(admit=1_000), Regime(delegate_above=5_000))
        assert res.delegated == 0 and res.sub_run == 0

    def test_delegated_share_counts_tokens_not_turns(self):
        """Four percent of turns can be half the tokens; turns are the wrong unit."""
        s = Session("s", "proj")
        ctx = 10_000
        for i in range(100):
            ctx += 100_000 if i == 50 else 500
            s.turns.append(Turn("s", "proj", OPUS, 0, ctx, 0, 400, 0, False))
        res = replay({"s": s}, Regime(delegate_above=5_000))
        assert res.delegated == 1
        assert res.delegated_share > 0.6, "one turn, most of the tokens"

    def test_splitting_beats_not_splitting_on_a_long_session(self):
        sess = _sessions(n_turns=600)
        assert replay(sess, Regime(split_turns=100)).total < replay(sess, Regime()).total

    def test_effort_only_touches_output(self):
        sess = _sessions()
        base, low = replay(sess, Regime()), replay(sess, Regime(effort="low"))
        assert low.main_out < base.main_out
        assert low.main_input < base.main_input, "less output means less to re-read"

    def test_every_lever_moves_the_total_down(self):
        sess = _sessions(n_turns=400, admit=8_000)
        base = replay(sess, Regime()).total
        for r in (Regime(delegate_above=5_000), Regime(split_turns=100),
                  Regime(effort="low"), Regime(terseness=0.3),
                  Regime(tool_discipline=0.4), Regime(session_model=SONNET)):
            assert replay(sess, r).total < base, r


class TestSessionModel:
    def test_a_cheaper_session_model_is_priced_with_its_rework(self):
        sess = _sessions()
        free = replay(sess, Regime(session_model=SONNET, session_rework=0.0))
        paid = replay(sess, Regime(session_model=SONNET, session_rework=0.20))
        assert paid.total > free.total
        assert paid.session_rework > 0

    def test_it_is_not_a_cache_rebuild(self):
        """A session that starts on Sonnet never had an Opus prefix to lose."""
        sess = _sessions()
        base = replay(sess, Regime())
        cheap = replay(sess, Regime(session_model=SONNET, session_rework=0.0))
        assert cheap.main_input < base.main_input, (
            "starting cheap is a rate substitution, not a switch"
        )

    def test_a_model_that_cannot_hold_the_context_is_not_used(self):
        sess = _sessions(n_turns=100, admit=10_000, base=400_000)   # past Haiku's 200K
        res = replay(sess, Regime(session_model=HAIKU))
        assert res.reprised == 0
        assert res.total == pytest.approx(replay(sess, Regime()).total)

    def test_enough_rework_makes_it_a_loss(self):
        sess = _sessions()
        assert (replay(sess, Regime(session_model=SONNET, session_rework=1.5)).total
                > replay(sess, Regime()).total)


class TestSubagentRightSizing:
    def test_a_read_that_fits_goes_to_the_cheap_tier(self):
        assert cheapest_tier(50_000, 5_000, p_fail=0.1, overhead=0.01) == Tier.T0

    def test_a_read_too_big_for_haiku_does_not(self):
        assert cheapest_tier(400_000, 40_000, p_fail=0.1, overhead=0.01) > Tier.T0

    def test_the_brief_counts_against_the_window(self):
        limit = 200_000
        assert cheapest_tier(limit - BRIEF_TOKENS, 1, p_fail=0.0, overhead=0.0) > Tier.T0

    def test_a_high_failure_rate_pushes_the_tier_up(self):
        cheap = cheapest_tier(50_000, 5_000, p_fail=0.0, overhead=10.0)
        risky = cheapest_tier(50_000, 5_000, p_fail=0.95, overhead=10.0)
        assert risky >= cheap

    def test_right_sizing_never_costs_more_than_delegating_to_opus(self):
        sess = _sessions(n_turns=300, admit=20_000)
        blind = replay(sess, Regime(delegate_above=5_000, right_size=False,
                                    sub_model=OPUS))
        sized = replay(sess, Regime(delegate_above=5_000, right_size=True))
        assert sized.total <= blind.total
        assert sized.by_tier and blind.by_tier == {}


class TestLadderAndSolve:
    def test_the_ladder_is_cumulative_and_monotone(self):
        sess = _sessions(n_turns=500, admit=8_000)
        totals = [replay(sess, r).total for r in ladder()]
        assert totals == sorted(totals, reverse=True), "each row adds a lever"

    def test_the_ladder_starts_from_the_untouched_baseline(self):
        assert ladder()[0] == Regime()

    def test_placement_is_measured_before_price(self):
        """The first delegating row holds the model fixed, so the next row can claim it."""
        first = ladder()[1]
        assert first.delegate_above and not first.right_size
        assert ladder()[2].right_size

    def test_solve_finds_a_regime_that_meets_the_target(self):
        sess = _sessions(n_turns=600, admit=8_000)
        base = replay(sess, Regime()).total
        reg, res, severity = solve(sess, target=3.0, baseline=base, output_share=0.5)
        assert reg is not None and res.total <= base / 3.0
        assert severity > 0

    def test_solve_returns_nothing_rather_than_a_regime_that_misses(self):
        sess = _sessions(n_turns=50, admit=500)
        base = replay(sess, Regime()).total
        reg, res, _ = solve(sess, target=1_000.0, baseline=base, output_share=0.5)
        assert reg is None and res is None

    def test_solve_prefers_the_milder_of_two_sufficient_regimes(self):
        sess = _sessions(n_turns=600, admit=8_000)
        base = replay(sess, Regime()).total
        _, _, easy = solve(sess, target=1.5, baseline=base, output_share=0.5)
        _, _, hard = solve(sess, target=4.0, baseline=base, output_share=0.5)
        assert easy <= hard, "a gentler target must not need a harsher regime"

    def test_the_frontier_bounds_the_grid(self):
        sess = _sessions(n_turns=600, admit=8_000)
        edge = replay(sess, frontier()).total
        for r in ladder():
            assert replay(sess, r).total >= edge

    def test_every_grid_axis_has_a_do_nothing_setting_scored_zero(self):
        """Severity has to start at 'change nothing', or the scale means nothing."""
        for axis, options in GRID.items():
            assert options[0][1] == 0, axis
            assert min(sev for _, sev in options) == 0, axis


class TestRestartsArePriced:
    """A lever with no cost term gets pushed to the end of its range for free.

    Splitting used to be exactly that: the context reset and nothing was
    charged. These lock the price in, in both directions -- a restart that costs
    nothing and a restart that costs a full rebuild are both wrong.
    """

    def _with_writes(self, n_turns=200, admit=4_000, write=2_000, base=20_000):
        s = Session("s", "proj")
        ctx = base
        for i in range(n_turns):
            if i:
                ctx += admit
            s.turns.append(Turn("s", "proj", OPUS, 0, ctx - write, write, 500, 0,
                                False, ttl="1h"))
        return {"s": s}

    def test_a_restart_is_not_free(self):
        res = replay(_sessions(n_turns=200), Regime(split_turns=50))
        assert res.restarts == 3, "turns 50, 100 and 150"
        assert res.restart > 0

    def test_it_is_priced_off_what_the_opening_cost(self):
        sess = _sessions(n_turns=200)
        opening = prepare(sess)[0][2][0].in_cost
        res = replay(sess, Regime(split_turns=50, handoff_tokens=0))
        assert res.restart == pytest.approx(3 * opening)

    def test_the_handoff_is_written_in_on_top(self):
        sess = _sessions(n_turns=200)
        free = replay(sess, Regime(split_turns=50, handoff_tokens=0)).restart
        paid = replay(sess, Regime(split_turns=50, handoff_tokens=10_000)).restart
        assert paid > free

    def test_a_handoff_big_enough_makes_splitting_a_loss(self):
        """The guard against the old bug: splitting must be able to lose money."""
        sess = _sessions(n_turns=300)
        base = replay(sess, Regime()).total
        assert replay(sess, Regime(split_turns=10, handoff_tokens=2_000)).total < base
        assert replay(sess, Regime(split_turns=10, handoff_tokens=900_000)).total > base

    def test_a_restart_cannot_open_below_an_opening(self):
        """A session that compacted has a floor under its opening; a restart does not."""
        s = Session("s", "proj")
        for ctx in [80_000, 120_000, 30_000, 60_000, 90_000, 120_000, 150_000]:
            s.turns.append(Turn("s", "proj", OPUS, 0, ctx, 0, 400, 0, False))
        deep = replay({"s": s}, Regime(split_turns=2, handoff_tokens=0))
        assert deep.main_input > 0
        # Restarting to the 30,000 low would have priced turns against a session
        # that never opened that small.
        assert deep.main_input >= 80_000 * 5 * 0.10 / 1e6 * 2

    def test_writes_do_not_shrink_just_because_the_context_did(self):
        """Splitting stops content being re-read. It does not stop it being written."""
        sess = self._with_writes()
        floor = sum(t.cache_write * 5 * 2.00 / 1e6 for t in sess["s"].turns)
        hard = replay(sess, Regime(split_turns=5, handoff_tokens=0))
        assert hard.main_input >= floor * 0.999
        assert hard.main_input < replay(sess, Regime()).main_input

    def test_terseness_does_shrink_them(self):
        """Writing less is the lever that reaches the write side; splitting is not."""
        sess = self._with_writes()
        base = replay(sess, Regime())
        terse = replay(sess, Regime(terseness=0.5, tool_discipline=0.5))
        assert terse.main_input < base.main_input


class TestSolvedCadence:
    def test_the_cadence_is_solved_from_the_workload(self):
        sess = _sessions(n_turns=300)
        k, _opening, why = recommended_cadence(sess)
        assert k >= 1 and str(k) in why

    def test_with_nothing_to_measure_it_says_so_and_stays_pessimistic(self):
        """No openings on record means the cold rebuild, and a note saying why."""
        k, opening, why = recommended_cadence({"s": Session("s", "p")})
        assert not opening.measured
        assert "prior" in why
        assert k >= 1

    def test_a_bigger_handoff_lengthens_it(self):
        sess = _sessions(n_turns=300)
        small, _, _ = recommended_cadence(sess, handoff_tokens=2_000)
        large, _, _ = recommended_cadence(sess, handoff_tokens=100_000)
        assert large > small


class TestDelegatedWorkIsStillPaidFor:
    """Delegation moves work; it does not delete it.

    Charging nothing for a delegated step's own output made delegation a free
    way to remove the session's generation cost, and the solved threshold duly
    collapsed to ~300 tokens and delegated 99% of admitted tokens with output
    falling to 1% of the bill. Same failure as the free restart, one lever over.
    """

    def test_a_delegated_turn_still_pays_for_its_output(self):
        sess = _sessions(n_turns=100, admit=40_000, out=5_000)
        res = replay(sess, Regime(delegate_above=5_000))
        assert res.delegated > 0
        assert res.main_out > 0, "the work still happened, on somebody's model"

    def test_it_is_charged_at_the_subagent_rate_not_the_session_rate(self):
        sess = _sessions(n_turns=100, admit=40_000, out=5_000)
        base = replay(sess, Regime())
        deleg = replay(sess, Regime(delegate_above=5_000))
        assert deleg.main_out < base.main_out, "a cheaper model wrote it"
        # Haiku output is 5/25 of Opus, and every turn but the first delegates.
        assert deleg.main_out > base.main_out * 0.1

    def test_delegating_everything_cannot_zero_the_output(self):
        sess = _sessions(n_turns=100, admit=40_000, out=5_000)
        assert replay(sess, Regime(delegate_above=1)).main_out > 0


class TestSolvedThreshold:
    def test_the_threshold_is_solved_from_the_workload(self):
        sess = _sessions(n_turns=300)
        tok, why = recommended_threshold(sess, split_turns=19)
        assert tok is not None and tok >= 100
        assert "delegate reads over" in why

    def test_a_shorter_cycle_raises_it(self):
        """Fewer re-reads to avoid means a read has to be bigger to be worth moving."""
        sess = _sessions(n_turns=600)
        short, _ = recommended_threshold(sess, split_turns=10)
        long_, _ = recommended_threshold(sess, split_turns=400)
        assert short > long_

    def test_it_is_rounded_to_something_a_hook_can_apply(self):
        sess = _sessions(n_turns=300)
        tok, _ = recommended_threshold(sess, split_turns=19)
        assert tok % 100 == 0

    def test_a_subagent_with_no_rate_advantage_has_to_read_more_to_pay(self):
        sess = _sessions(n_turns=300)
        cheap, _ = recommended_threshold(sess, split_turns=19, sub_model=HAIKU)
        same, _ = recommended_threshold(sess, split_turns=19, sub_model=OPUS)
        assert same > cheap


class TestCounterfactualsArePricedOnTheTurnsDate:
    """A replay compares a recorded past against a hypothetical past, not a present.

    Every baseline figure comes from `Turn.input_cost`/`output_cost`, which
    resolve the turn's own date. The counterfactuals did not: `cheap_rate` was
    looked up once outside the loop, and the subagent run, the escalation redo
    and the restart write all passed the caller's `on` (None, meaning today).
    While the rate table is stable nothing moves; the day Sonnet 5's
    introductory $2/$10 reverts to $3/$15 the two halves are in different
    calendars -- and `ladder()` names Sonnet as the cheap session model, so it
    is that comparison, not an incidental one.
    """

    @staticmethod
    def _sessions(ts="2026-08-10T12:00:00Z", n=40):
        s = Session("s", "p")
        for i in range(n):
            s.turns.append(Turn("s", "p", "claude-opus-5", uncached_in=0,
                                cache_read=50_000 + 500 * i, cache_write=1_000,
                                out=300, thinking=0, sidechain=False, ts=ts))
        return {"s": s}

    def _cheap(self, sessions, on=None):
        return replay(sessions, Regime(session_model="claude-sonnet-5",
                                       session_rework=0.0), on=on)

    def test_the_intro_rate_applies_to_turns_inside_the_window(self):
        d = self._sessions()
        assert self._cheap(d).total == pytest.approx(
            self._cheap(d, on=date(2026, 8, 10)).total)

    def test_and_the_reverted_rate_applies_after_it(self):
        d = self._sessions()
        assert self._cheap(d, on=date(2026, 9, 1)).total > self._cheap(d).total

    def test_the_step_carries_the_date(self):
        (_, _, steps), = prepare(self._sessions(n=2))
        assert steps[0].on == date(2026, 8, 10)

    def test_a_model_with_no_intro_rate_is_unaffected(self):
        d = self._sessions()
        a = replay(d, Regime(session_model="claude-haiku-4-5", session_rework=0.0))
        b = replay(d, Regime(session_model="claude-haiku-4-5", session_rework=0.0),
                   on=date(2026, 9, 1))
        assert a.total == pytest.approx(b.total)

    def test_the_baseline_is_dated_the_same_way(self):
        """Guards the premise: both halves must move together or neither."""
        d = self._sessions()
        assert replay(d, Regime()).total == pytest.approx(
            replay(d, Regime(), on=date(2026, 8, 10)).total)


class TestRefusingDuplicates:
    """A refusal deletes an admission; it does not compress one.

    `tool_discipline` scales what a turn admits and stands on somebody keeping
    a habit. This removes a measured set of calls a hook declines, which is why
    it is subtracted before the delegation gate rather than folded into that
    fraction.
    """

    def _sessions(self, n=6, admit=4_000):
        from adder.core.trace import Session, Turn

        s = Session("s", "p")
        ctx = 20_000
        for _ in range(n):
            ctx += admit
            s.turns.append(Turn("s", "p", "claude-opus-5", uncached_in=0,
                                cache_read=ctx, cache_write=admit, out=300,
                                thinking=0, sidechain=False))
        return {"s": s}

    def test_a_refused_admission_is_never_carried(self):
        from adder.evaluate.replay.plan import Regime, prepare, replay

        sess = self._sessions()
        dups = {("s", i): 2_000 for i in range(1, 6)}
        prepared = prepare(sess, None, dups)
        plain = replay(prepared, Regime())
        refusing = replay(prepared, Regime(refuse_duplicates=True))
        assert refusing.refused_tokens == 10_000
        assert refusing.admitted_tokens < plain.admitted_tokens
        assert refusing.total < plain.total

    def test_it_cannot_subtract_more_than_the_turn_admitted(self):
        """The two quantities are counted by different methods -- context
        growth here, estimated result sizes there -- so the subtraction is
        clamped rather than trusted."""
        from adder.evaluate.replay.plan import Regime, prepare, replay

        sess = self._sessions(admit=1_000)
        prepared = prepare(sess, None, {("s", i): 10_000_000 for i in range(6)})
        r = replay(prepared, Regime(refuse_duplicates=True))
        assert r.admitted_tokens == 0
        assert r.refused_tokens == 5_000

    def test_no_map_means_no_change(self):
        from adder.evaluate.replay.plan import Regime, prepare, replay

        prepared = prepare(self._sessions(), None)
        assert (replay(prepared, Regime(refuse_duplicates=True)).total
                == replay(prepared, Regime()).total)
