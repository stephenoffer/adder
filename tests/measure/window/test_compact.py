"""Compaction economics: what it cost, what it bought, and when it pays.

The number this file exists to keep honest is the missed-compaction saving.
Priced naively -- freed tokens re-read once per remaining turn -- it invents
money, because a compacted context refills while the un-compacted one is pinned
against the ceiling. The regrowth simulation is asserted directly, because a
report that over-states this lever is a report that tells people to compact
sessions where compacting loses.
"""

from __future__ import annotations

import json

import pytest

from adder.core.trace import Session
from adder.measure.window import compact

OPUS = "claude-opus-5"


@pytest.fixture
def session(make_turn):
    """A session builder taking explicit per-turn contexts."""
    def _make(contexts, *, sid="s", model=OPUS, writes=None, project="proj"):
        s = Session(sid, project)
        for i, ctx in enumerate(contexts):
            w = 0 if writes is None else writes[i]
            s.turns.append(make_turn(model=model, session=sid, project=project,
                                     read=max(0, ctx - w), write=w, minutes=i * 2))
        return s
    return _make


class TestEvent:
    def event(self, **kw):
        base = {"session": "s", "project": "p", "model": OPUS, "turn": 10,
                "before": 800_000, "after": 40_000, "rebuild_tokens": 40_000,
                "remaining": 200}
        base.update(kw)
        return compact.Event(**base)

    def test_freed_and_kept(self):
        e = self.event()
        assert e.freed == 760_000
        assert e.kept == pytest.approx(0.05)

    def test_rebuild_is_priced_at_the_write_multiplier(self):
        cheap = self.event(ttl="5m").rebuild_cost()
        dear = self.event(ttl="1h").rebuild_cost()
        assert dear > cheap > 0

    def test_carry_saved_scales_with_the_turns_that_followed(self):
        short = self.event(remaining=10).carry_saved(0.1)
        long = self.event(remaining=200).carry_saved(0.1)
        assert long == pytest.approx(20 * short)

    def test_net_is_saving_minus_rebuild(self):
        e = self.event()
        assert e.net(0.1) == pytest.approx(e.carry_saved(0.1) - e.rebuild_cost())

    def test_breakeven_is_where_net_crosses_zero(self):
        e = self.event()
        need = e.breakeven_turns(0.1)
        assert self.event(remaining=need).net(0.1) > 0
        assert self.event(remaining=max(0, need - 2)).net(0.1) <= 0

    def test_a_compaction_with_no_turns_left_is_too_late(self):
        assert self.event(remaining=0).verdict(0.1) == "too late"

    def test_a_compaction_with_a_long_tail_paid_off(self):
        assert self.event(remaining=500).verdict(0.1) == "paid off"

    def test_freeing_nothing_has_no_breakeven(self):
        assert self.event(before=100, after=100).breakeven_turns(0.1) == 0


class TestFindEvents:
    def test_a_near_ceiling_collapse_is_a_compaction(self, session):
        s = session([900_000] * 3 + [40_000] * 3, writes=[0, 0, 0, 40_000, 0, 0])
        events = compact.find_events({"s": s})
        assert len(events) == 1
        assert events[0].before == 900_000
        assert events[0].rebuild_tokens == 40_000

    def test_a_small_dip_is_not_a_compaction(self, session):
        assert compact.find_events({"s": session([30_000, 20_000, 30_000])}) == []

    def test_a_shallow_drop_near_the_ceiling_is_not_a_compaction(self, session):
        # 20% off is a branch wobble, not a compaction: it keeps most of itself.
        assert compact.find_events({"s": session([900_000, 720_000])}) == []

    def test_remaining_counts_the_turns_after_the_event(self, session):
        s = session([900_000] * 2 + [40_000] * 5)
        assert compact.find_events({"s": s})[0].remaining == 4

    def test_rebuild_falls_back_to_the_context_when_nothing_was_written(self, session):
        s = session([900_000, 40_000])
        assert compact.find_events({"s": s})[0].rebuild_tokens == 40_000


class TestFindMisses:
    def test_a_long_run_at_the_ceiling_with_no_compaction(self, session):
        s = session([700_000] * 60)
        misses = compact.find_misses({"s": s}, min_turns=40)
        assert len(misses) == 1
        assert misses[0].turns_above == 60

    def test_a_short_run_is_not_a_miss(self, session):
        assert compact.find_misses({"s": session([700_000] * 10), }, min_turns=40) == []

    def test_a_session_that_did_compact_is_not_a_miss(self, session):
        s = session([700_000] * 50 + [40_000] * 10)
        assert compact.find_misses({"s": s}, min_turns=40) == []

    def test_a_low_context_session_is_not_a_miss(self, session):
        assert compact.find_misses({"s": session([30_000] * 100)}, min_turns=40) == []


class TestMissPricing:
    def miss(self, *, n=100, ctx=700_000, growth=0.0):
        return compact.Miss(session="s", project="p", model=OPUS,
                            contexts=tuple([ctx] * n), growth=growth, remaining=n)

    def test_regrowth_reduces_the_saving(self):
        still = self.miss(growth=0.0).saving(0.1)
        refills = self.miss(growth=20_000).saving(0.1)
        assert 0 < refills < still

    def test_fast_regrowth_closes_the_gap_entirely(self):
        assert self.miss(growth=10_000_000).saving(0.1) == 0.0

    def test_a_bigger_context_is_worth_more(self):
        assert self.miss(ctx=900_000).saving(0.1) > self.miss(ctx=300_000).saving(0.1)

    def test_the_rebuild_is_subtracted(self):
        # One turn above the trigger: the rebuild dominates and nothing is owed.
        assert self.miss(n=1).saving(0.1) == 0.0

    def test_a_saving_never_goes_negative(self):
        assert self.miss(n=2, ctx=60_000).saving(0.1) >= 0.0

    def test_the_saving_cannot_exceed_what_was_carried(self):
        m = self.miss()
        r = 15.0 / 1_000_000          # opus input rate, USD per token
        carried = sum(m.contexts) * r * 0.1
        assert m.saving(0.1) <= carried


class TestBreakeven:
    def test_a_higher_reread_multiplier_needs_fewer_turns(self):
        assert (compact.breakeven_remaining(read_mult=0.2)
                < compact.breakeven_remaining(read_mult=0.05))

    def test_keeping_more_needs_more_turns(self):
        assert (compact.breakeven_remaining(kept=0.6)
                > compact.breakeven_remaining(kept=0.1))

    def test_the_one_hour_ttl_raises_the_bar(self):
        assert (compact.breakeven_remaining(ttl="1h")
                > compact.breakeven_remaining(ttl="5m"))

    def test_context_threshold_is_unreachable_with_no_horizon(self):
        assert compact.breakeven_context(OPUS, 0) == 0

    def test_context_threshold_clears_with_a_long_horizon(self):
        assert compact.breakeven_context(OPUS, 10_000) == compact.MIN_CONTEXT


class TestVersusRestart:
    def test_a_large_context_favours_a_restart(self, make_sessions):
        choice, gap, why = compact.versus_restart(
            make_sessions(3, 40), model=OPUS, context_tokens=800_000)
        assert choice == "restart"
        assert gap > 0
        assert "handoff" in why

    def test_a_small_context_favours_compacting(self, make_sessions):
        choice, _, _ = compact.versus_restart(
            make_sessions(3, 40), model=OPUS, context_tokens=5_000)
        assert choice == "compact"

    def test_a_bigger_handoff_makes_a_restart_less_attractive(self, make_sessions):
        s = make_sessions(3, 40)
        _, small, _ = compact.versus_restart(s, model=OPUS, context_tokens=800_000,
                                             handoff_tokens=1_000)
        _, large, _ = compact.versus_restart(s, model=OPUS, context_tokens=800_000,
                                             handoff_tokens=50_000)
        assert small > large


class TestOutput:
    def test_report_states_the_rule(self, session, make_sessions):
        rep = compact.analyse({"s": session([900_000] * 2 + [40_000] * 50)})
        text = compact.report(rep, {})
        assert "pays for itself" in text
        assert "Not priced" in text

    def test_report_says_so_when_nothing_compacted(self, make_sessions):
        text = compact.report(compact.analyse(make_sessions(2, 20)), {})
        assert "No compactions on record" in text

    def test_json_is_one_document(self, write_jsonl, tmp_path, capsys):
        recs = []
        for i in range(6):
            ctx = 900_000 if i < 3 else 40_000
            recs.append({"type": "assistant", "sessionId": "s",
                         "timestamp": f"2026-08-01T10:{i:02d}:00Z",
                         "message": {"id": f"m{i}", "model": OPUS,
                                     "usage": {"input_tokens": 0,
                                               "cache_read_input_tokens": ctx,
                                               "cache_creation_input_tokens": 0,
                                               "output_tokens": 100}}})
        d = write_jsonl(recs, into=tmp_path / "projects" / "proj")
        assert compact.main([str(d.parent), "--json"]) == 0
        doc = json.loads(capsys.readouterr().out)
        assert doc["compactions"] == 1
        assert doc["events"][0]["verdict"] in ("paid off", "marginal", "too late")

    def test_no_sessions_is_a_nonzero_exit(self, tmp_path, capsys):
        assert compact.main([str(tmp_path)]) == 1

    def test_no_sessions_still_emits_json(self, tmp_path, capsys):
        assert compact.main([str(tmp_path), "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["sessions"] == 0


class TestCompactionsAreMainChainOnly:
    """A subagent opening its own window is not a compaction of the parent.

    `is_compaction` asks for a near-ceiling context that loses most of itself,
    and the step down into a delegated run satisfies both clauses exactly.
    `Session.compactions` walks `main_turns` for this reason and says so;
    `find_events` and `find_misses` walked the combined list, so `adder compact`
    and `Session.compactions` reported different numbers for the same session --
    and the invented event carried the subagent's model, the subagent's context
    as `after`, and the subagent's legitimate opening write as its "rebuild".
    """

    def _turn(self, i, ctx, *, side=False, model="claude-opus-5", write=0):
        from datetime import datetime, timedelta, timezone

        from adder.core.trace import Turn

        start = datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc)
        return Turn("s", "p", model, 0, ctx, write, 100, 0, side,
                    ts=(start + timedelta(minutes=i)).isoformat())

    def _with_a_subagent(self):
        from adder.core.trace import Session

        s = Session("s", "p")
        s.turns = [self._turn(i, 700_000) for i in range(10)]
        s.turns += [self._turn(10, 20_000, side=True,
                               model="claude-haiku-4-5", write=20_000)]
        s.turns += [self._turn(11, 705_000)]
        return {"s": s}

    def test_a_delegation_is_not_an_event(self):
        from adder.measure.window.compact import find_events

        assert find_events(self._with_a_subagent()) == []

    def test_it_agrees_with_session_compactions(self):
        from adder.measure.window.compact import find_events

        sessions = self._with_a_subagent()
        assert len(find_events(sessions)) == sessions["s"].compactions()

    def test_a_real_compaction_is_still_found(self):
        from adder.core.trace import Session
        from adder.measure.window.compact import find_events

        s = Session("s2", "p")
        s.turns = [self._turn(i, 700_000) for i in range(10)]
        s.turns += [self._turn(10, 30_000, write=30_000)]
        events = find_events({"s2": s})
        assert len(events) == 1 == s.compactions()
        assert events[0].model == "claude-opus-5"
        assert events[0].after == 60_000       # 30K read back + 30K written

    def test_a_miss_is_not_disqualified_by_a_delegation(self):
        from adder.core.trace import Session
        from adder.measure.window.compact import find_misses

        s = Session("s3", "p")
        # 60 main-chain turns pinned at the ceiling, with one subagent turn in
        # the middle. That delegation used to read as a compaction and take the
        # whole session out of the report.
        s.turns = [self._turn(i, 700_000) for i in range(30)]
        s.turns += [self._turn(30, 20_000, side=True, model="claude-haiku-4-5")]
        s.turns += [self._turn(i, 700_000) for i in range(31, 61)]
        assert [m.session for m in find_misses({"s3": s})] == ["s3"]
