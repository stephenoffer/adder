"""Window reconstruction, and the three things it must refuse to assert.

The boundary rule belongs to somebody else and is not published, so the tests
that matter are not "does it agree with Anthropic" -- unanswerable -- but:

* the reconstruction is the documented rule, applied consistently;
* the window is per account, so concurrent sessions share one meter;
* nothing here claims a capacity. The envelope is the heaviest window observed
  and must be described as a floor, because that is all a survived window
  proves.

The within-block slope is the one number asserted without qualification, so it
gets the arithmetic tests: it must be absent rather than 1.0 when unmeasurable,
above 1 when the context grows, and below 1 when a restart shrinks it.
"""
from __future__ import annotations

import json
from datetime import timedelta

from adder.core.trace import Session
from adder.measure.spend.limits import (
    MIN_SLOPE_TURNS,
    WINDOW_HOURS,
    Block,
    blocks,
    build,
    main,
    render,
)
from tests.conftest import START


def sess(turns, sid="s", project="proj"):
    s = Session(sid, project)
    s.turns.extend(turns)
    return s


class TestReconstruction:
    def test_turns_inside_the_window_form_one_block(self, make_session):
        # 20 turns two minutes apart is 40 minutes: one window.
        got = blocks({"s": make_session(20, minutes_apart=2)})
        assert len(got) == 1
        assert len(got[0].turns) == 20

    def test_a_gap_of_a_full_window_opens_a_new_one(self, make_turn):
        s = sess([make_turn(minutes=0), make_turn(minutes=1),
                  make_turn(minutes=int(WINDOW_HOURS * 60) + 30)])
        got = blocks({"s": s})
        assert len(got) == 2
        assert [len(b.turns) for b in got] == [2, 1]

    def test_a_gap_just_under_a_window_does_not(self, make_turn):
        s = sess([make_turn(minutes=0),
                  make_turn(minutes=int(WINDOW_HOURS * 60) - 10)])
        # Same block by the gap rule, but the window has not elapsed either.
        assert len(blocks({"s": s})) == 1

    def test_the_window_elapsing_ends_a_block_even_without_a_gap(self, make_turn):
        # Turns every 30 minutes for eight hours: no gap ever reaches five
        # hours, so only the elapsed-window rule can close the first block.
        step = 30
        s = sess([make_turn(minutes=i * step) for i in range(17)])
        got = blocks({"s": s})
        assert len(got) > 1, "a block ran past the window length"
        assert all((b.last - b.start) < timedelta(hours=WINDOW_HOURS) for b in got)

    def test_blocks_open_on_the_hour(self, make_turn):
        s = sess([make_turn(minutes=17)])   # START is 09:00, so 09:17
        assert blocks({"s": s})[0].start.minute == 0

    def test_the_meter_is_shared_across_concurrent_sessions(self, make_session):
        a = make_session(10, sid="a", minutes_apart=2)
        b = make_session(10, sid="b", minutes_apart=2)
        got = blocks({"a": a, "b": b})
        assert len(got) == 1, "two sessions in the same hours were metered separately"
        assert len(got[0].turns) == 20

    def test_undateable_turns_are_dropped_not_defaulted(self, make_turn):
        s = sess([make_turn(minutes=0), make_turn(ts=None)])
        assert sum(len(b.turns) for b in blocks({"s": s})) == 1

    def test_no_dateable_turns_is_no_blocks(self, make_turn):
        assert blocks({"s": sess([make_turn(ts=None)])}) == []

    def test_hours_is_a_parameter_not_a_constant(self, make_turn):
        s = sess([make_turn(minutes=0), make_turn(minutes=90)])
        assert len(blocks({"s": s}, hours=1.0)) == 2
        assert len(blocks({"s": s}, hours=5.0)) == 1


class TestBlockArithmetic:
    def test_read_and_new_are_different_numbers(self, make_turn):
        # One turn re-reading 100k of cache and writing 400 out.
        b = Block(start=START, turns=[make_turn(read=100_000, out=400)])
        assert b.tokens == 100_400
        assert b.new_tokens == 400
        assert b.carry_tokens == 100_000

    def test_carry_share_counts_cache_writes_as_new(self, make_turn):
        # A cache write is text arriving for the first time, so it is work.
        b = Block(start=START, turns=[make_turn(read=0, write=1_000, uncached=0)])
        assert b.carry_share == 0.0
        assert b.new_tokens >= 1_000

    def test_an_empty_block_does_not_divide_by_zero(self):
        b = Block(start=START)
        assert b.carry_share == 0.0
        assert b.tokens == 0
        assert b.burn == 0.0

    def test_burn_is_measured_on_elapsed_clock_not_active_time(self, make_turn):
        # Two turns an hour apart: the window drained for an hour even though
        # only two turns happened.
        b = Block(start=START, turns=[make_turn(read=60_000, out=0, minutes=0),
                                      make_turn(read=60_000, out=0, minutes=60)])
        assert b.burn == 120_000 / 60


class TestSlope:
    def test_absent_rather_than_flat_when_too_short(self, make_session):
        b = Block(start=START, turns=make_session(MIN_SLOPE_TURNS - 1).turns)
        assert b.slope() is None, "a two-sample ratio was reported as a measurement"

    def test_above_one_when_the_context_grows(self, make_session):
        b = Block(start=START, turns=make_session(40, base=20_000, growth=5_000).turns)
        s = b.slope()
        assert s is not None and s[0] > 1.5
        assert s[2] > s[1]

    def test_below_one_when_a_restart_shrinks_the_context(self, make_turn):
        # Twenty large turns, then twenty small ones: a restart inside the window.
        turns = [make_turn(read=300_000, minutes=i) for i in range(20)]
        turns += [make_turn(read=10_000, minutes=20 + i) for i in range(20)]
        s = Block(start=START, turns=turns).slope()
        assert s is not None and s[0] < 1.0

    def test_order_is_by_timestamp_not_list_order(self, make_turn):
        turns = [make_turn(read=10_000 * (i + 1), minutes=i) for i in range(20)]
        forward = Block(start=START, turns=list(turns)).slope()
        shuffled = Block(start=START, turns=list(reversed(turns))).slope()
        assert forward == shuffled


class TestReport:
    def test_envelope_is_the_heaviest_window(self, make_turn):
        light = [make_turn(read=1_000, minutes=0)]
        heavy = [make_turn(read=900_000, minutes=int(WINDOW_HOURS * 60) * 2)]
        rep = build({"s": sess(light + heavy)})
        assert rep.envelope is not None
        assert rep.envelope.tokens == max(b.tokens for b in rep.blocks)

    def test_no_blocks_means_no_envelope_and_no_crash(self):
        rep = build({})
        assert rep.envelope is None and rep.active is None
        assert rep.projected() is None
        assert rep.carry_share == 0.0

    def test_active_only_while_the_window_is_open(self, make_session):
        s = make_session(10, minutes_apart=2)
        rep = build({"s": s}, now=START + timedelta(hours=1))
        assert rep.active is not None
        closed = build({"s": s}, now=START + timedelta(hours=WINDOW_HOURS + 1))
        assert closed.active is None

    def test_active_requires_a_now(self, make_session):
        assert build({"s": make_session(10)}).active is None

    def test_projection_extrapolates_the_observed_rate(self, make_turn):
        # 10 turns of 60k over 60 minutes = 10k/min; two hours in, three left.
        turns = [make_turn(read=60_000, out=0, minutes=i * 6) for i in range(11)]
        rep = build({"s": sess(turns)}, now=START + timedelta(hours=2))
        proj = rep.projected()
        assert proj is not None
        total, share = proj
        assert total > rep.blocks[-1].tokens, "the projection did not look forward"
        # Not capped at 1.0, and it must not be: the envelope is the heaviest
        # window *so far*, so a share above 1 is the report saying this window
        # is heading past anything on record. Clamping it would hide the only
        # case where the comparison is worth printing.
        assert share > 1.0

    def test_median_slope_skips_unmeasurable_blocks(self, make_turn):
        # One long block with a real slope, one two-turn block with none. The
        # short one must not be counted as flat and drag the median to 1.
        long = [make_turn(read=10_000 * (i + 1), minutes=i) for i in range(20)]
        short = [make_turn(read=5_000, minutes=int(WINDOW_HOURS * 60) * 2 + i)
                 for i in range(2)]
        rep = build({"s": sess(long + short)})
        assert len(rep.blocks) == 2
        assert rep.median_slope() == rep.blocks[0].slope()[0]

    def test_median_slope_is_none_when_nothing_is_measurable(self, make_turn):
        assert build({"s": sess([make_turn(minutes=0)])}).median_slope() is None


class TestHonesty:
    def test_the_envelope_is_called_a_floor_not_a_limit(self, make_session):
        out = render(build({"s": make_session(20)}))
        assert "floor" in out
        assert "not the limit" in out

    def test_the_boundaries_are_called_a_reconstruction(self, make_session):
        assert "reconstructed" in render(build({"s": make_session(20)}))

    def test_the_projection_is_labelled_a_floor_too(self, make_session):
        rep = build({"s": make_session(20, minutes_apart=2)},
                    now=START + timedelta(hours=1))
        assert rep.projected() is not None
        assert "read it as a floor" in render(rep)

    def test_json_carries_the_caveats_rather_than_bare_numbers(self, make_session):
        rep = build({"s": make_session(20, minutes_apart=2)},
                    now=START + timedelta(hours=1))
        d = rep.to_json()
        assert "lower bound on capacity" in d["envelope"]["note"]
        assert "under-estimate" in d["projected"]["note"]

    def test_empty_input_says_so(self):
        assert "Nothing to place in a window" in render(build({}))


class TestCli:
    def test_json_is_machine_readable(self, tmp_path, capsys, write_jsonl,
                                      isolated_home, monkeypatch):
        monkeypatch.setenv("ADDER_ROOT", str(tmp_path / "none"))
        assert main(["--json"]) == 0
        d = json.loads(capsys.readouterr().out)
        assert d["window_hours"] == WINDOW_HOURS
        assert d["blocks"] == 0

    def test_hours_flag_reaches_the_reconstruction(self, tmp_path, capsys,
                                                  isolated_home, monkeypatch):
        monkeypatch.setenv("ADDER_ROOT", str(tmp_path / "none"))
        assert main(["--json", "--hours", "1"]) == 0
        assert json.loads(capsys.readouterr().out)["window_hours"] == 1.0

class TestWeek:
    """The weekly cap, and why the span slides instead of being anchored."""

    def spread(self, make_turn, per_day, days):
        turns = []
        for d in range(days):
            for i in range(per_day):
                turns.append(make_turn(read=100_000,
                                       minutes=d * 24 * 60 + i * 10))
        return sess(turns)

    def test_the_peak_is_anchor_independent(self, make_turn):
        # A heavy burst on days 3-5 of a 14-day span. A calendar-week bucket
        # would split it or not depending on where the week starts; the sliding
        # window finds it either way, which is the whole reason it slides.
        turns = [make_turn(read=10_000, minutes=d * 24 * 60) for d in range(14)]
        turns += [make_turn(read=900_000, minutes=(3 + d) * 24 * 60 + 30)
                  for d in range(3)]
        rep = build({"s": sess(turns)})
        peak = rep.rolling(days=7.0)
        assert peak is not None
        tokens, start = peak
        assert tokens > sum(b.tokens for b in rep.blocks) * 0.5, (
            "the sliding window missed the burst it exists to find")
        assert START <= start <= START + timedelta(days=6)

    def test_a_shorter_span_cannot_hold_more(self, make_turn):
        rep = build({"s": self.spread(make_turn, 3, 10)})
        assert rep.rolling(days=3.0)[0] <= rep.rolling(days=7.0)[0]

    def test_trailing_needs_a_now(self, make_turn):
        rep = build({"s": self.spread(make_turn, 2, 10)})
        assert rep.trailing() is None

    def test_trailing_counts_only_the_recent_span(self, make_turn):
        s = self.spread(make_turn, 2, 14)
        rep = build({"s": s}, now=START + timedelta(days=14))
        trail = rep.trailing(days=7.0)
        assert trail is not None
        assert 0 < trail < sum(b.tokens for b in rep.blocks), (
            "the trailing window counted the whole history")

    def test_no_blocks_means_no_week(self):
        rep = build({})
        assert rep.rolling() is None and rep.trailing() is None

    def test_the_json_carries_the_two_caveats(self, make_turn):
        d = build({"s": self.spread(make_turn, 2, 9)}).to_json()
        assert "not a calendar week" in d["week"]["note"]
        assert "compute hours" in d["week"]["note"], (
            "the proxy is not labelled as a proxy")

    def test_the_report_calls_it_a_proxy(self, make_turn):
        out = render(build({"s": self.spread(make_turn, 2, 9)}))
        assert "sliding span, not a calendar week" in out
        assert "proxy" in out
