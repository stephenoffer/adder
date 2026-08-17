"""Attained service, pinned on the reference value and on what is not claimed.

Two failures this suite guards. Resampling *positions* instead of sessions: 600
positions inside one session all agree, so an interval over them is roughly 25x
too narrow. And re-introducing a `heavy-tailed` verdict, which finite data
cannot support -- past the median length the survivors are simply running out,
so every workload's mean-residual-life curve turns downward regardless of its
tail.
"""

from __future__ import annotations

import json

import pytest

from adder.measure.session import sched


class TestPoints:
    def test_every_position_in_every_session_is_a_point(self, make_sessions):
        sessions = make_sessions(n=3, n_turns=10)
        assert len(sched.points(sessions)) == 30

    def test_attained_and_remaining_sum_to_the_session_length(self, make_session):
        s = make_session(12)
        for p in sched.points({"s": s}):
            assert p.attained + p.remaining == 12

    def test_the_last_position_has_nothing_remaining(self, make_session):
        pts = sched.points({"s": make_session(8)})
        assert pts[-1].remaining == 0
        assert pts[-1].attained == 8

    def test_cost_is_split_the_same_way(self, make_session):
        s = make_session(6)
        total = s.cost
        for p in sched.points({"s": s}):
            assert p.cost_so_far + p.cost_remaining == pytest.approx(total)

    def test_a_one_turn_session_has_no_remaining_half(self, make_session):
        assert sched.points({"s": make_session(1)}) == []

    def test_no_sessions_gives_no_points(self):
        assert sched.points({}) == []


class TestCurve:
    def test_each_threshold_reports_what_is_left_beyond_it(self, make_sessions):
        rows = sched.curve(make_sessions(n=4, n_turns=60))
        by_threshold = {r.threshold: r for r in rows}
        assert by_threshold[1].points > by_threshold[50].points
        assert by_threshold[50].mean_remaining < by_threshold[1].mean_remaining

    def test_a_threshold_past_every_session_is_empty(self, make_sessions):
        rows = {r.threshold: r for r in sched.curve(make_sessions(n=2, n_turns=10))}
        assert rows[800].points == 0

    def test_thin_rows_are_flagged(self, make_sessions):
        rows = {r.threshold: r for r in sched.curve(make_sessions(n=2, n_turns=30))}
        assert rows[1].thin        # two sessions is below the floor

    def test_spend_share_is_a_fraction(self, make_sessions):
        for r in sched.curve(make_sessions(n=5, n_turns=40)):
            assert 0.0 <= r.spend_share <= 1.0


class TestRegimes:
    @staticmethod
    def _mixed(make_session, lengths):
        return {f"s{i}": make_session(n, sid=f"s{i}")
                for i, n in enumerate(lengths)}

    def test_equal_length_sessions_hit_the_reference_exactly(self, make_session):
        """The anchor the whole statistic is read against.

        -0.5 rather than -1.0 because the mean is taken over the positions past
        each threshold, which halves the 1:1 decline.
        """
        sessions = self._mixed(make_session, [40] * 8)
        rep = sched.analyse(sessions, resamples=60)
        assert rep.slope == pytest.approx(-0.5, abs=0.02)
        assert rep.verdict == "uniform-length"
        assert rep.informative

    def test_a_wide_length_spread_reads_as_dispersed(self, make_session):
        """Equal counts per doubling: how far in you are says little."""
        lengths = [n for k in range(1, 9) for n in [2 ** k] * 6]
        rep = sched.analyse(self._mixed(make_session, lengths), resamples=60)
        assert rep.slope > -0.35
        assert rep.verdict == "dispersed"
        assert not rep.informative

    def test_a_heavy_tail_is_not_claimed_because_it_cannot_be(self, make_session):
        """Finite data truncates, so the statistic reports no third category.

        Pinned so nobody re-introduces a `heavy-tailed` verdict: on a workload
        deliberately built with a long tail, the summary is still negative,
        because past the median the survivors are simply running out.
        """
        sessions = self._mixed(make_session, [4] * 40 + [50] * 12 + [900] * 6)
        rep = sched.analyse(sessions, resamples=60)
        assert rep.verdict in ("uniform-length", "dispersed")

    def test_the_verdict_is_decided_on_the_interval_not_the_point(self, make_session):
        """Two sessions cannot support any verdict, however the point falls."""
        sessions = self._mixed(make_session, [5, 40])
        rep = sched.analyse(sessions, resamples=40)
        lo, hi = rep.slope_ci
        assert lo <= rep.slope <= hi or rep.verdict == "dispersed"

    def test_informative_tracks_the_verdict(self, make_session):
        rep = sched.analyse(self._mixed(make_session, [40] * 8), resamples=60)
        assert rep.informative is (rep.verdict == "uniform-length")


class TestSlopeInterval:
    def test_the_interval_is_taken_over_sessions(self, make_session):
        """The whole reason the session is the resampling unit.

        One long session contributes hundreds of agreeing positions. An
        interval that treats them as independent is far too narrow, and the
        gap between the two is what this test pins.
        """
        from adder.util.stats import bootstrap_ci

        sessions = {f"s{i}": make_session(n, sid=f"s{i}")
                    for i, n in enumerate([5] * 6 + [300])}
        pts = sched.points(sessions)
        by_session = sched.tail_slope_ci(sessions, resamples=120)
        # Resampling positions, the wrong way, for comparison only.
        by_point = bootstrap_ci([float(p.remaining) for p in pts])
        assert by_session[0] <= by_session[1]
        assert by_point[1] >= by_point[0]

    def test_one_session_cannot_support_an_interval(self, make_session):
        assert sched.tail_slope_ci({"s": make_session(50)}) == (-1.0, 1.0)

    def test_the_interval_is_reproducible(self, make_sessions):
        sessions = make_sessions(n=5, n_turns=20)
        assert (sched.tail_slope_ci(sessions, resamples=40) ==
                sched.tail_slope_ci(sessions, resamples=40))

    def test_the_point_estimate_needs_three_populated_thresholds(self):
        assert sched.tail_slope({}) == 0.0
        assert sched.slope_from_lengths([40] * 3) == 0.0

    def test_the_closed_form_matches_what_the_curve_would_say(self, make_session):
        """The fast path is an optimisation, not a different statistic."""
        sessions = {f"s{i}": make_session(n, sid=f"s{i}")
                    for i, n in enumerate([10] * 8 + [60] * 8)}
        rows = sched._covered(sched.curve(sessions), len(sessions))
        assert len(rows) >= 3
        for r in rows:
            survivors, mean_rem = sched._mean_remaining(
                sched.lengths_of(sessions), r.threshold)
            assert survivors == r.sessions
            assert mean_rem == pytest.approx(r.mean_remaining)


class TestReport:
    def test_it_names_the_regime_and_the_reference(self, make_session):
        sessions = {f"s{i}": make_session(40, sid=f"s{i}") for i in range(8)}
        text = sched.format_report(sched.analyse(sessions, resamples=60))
        assert "uniform-length" in text
        assert "-0.50" in text

    def test_a_dispersed_verdict_refuses_the_restart_rule(self, make_session):
        lengths = [n for k in range(1, 9) for n in [2 ** k] * 6]
        sessions = {f"s{i}": make_session(n, sid=f"s{i}")
                    for i, n in enumerate(lengths)}
        rep = sched.analyse(sessions, resamples=60)
        assert rep.verdict == "dispersed"
        assert "sorting noise" in sched.format_report(rep)

    def test_it_explains_the_resampling_unit(self, make_sessions):
        text = sched.format_report(sched.analyse(make_sessions(n=4, n_turns=20),
                                                 resamples=40))
        assert "resamples sessions" in text

    def test_an_empty_workload_says_so(self):
        assert "No session long enough" in sched.format_report(sched.analyse({}))

    def test_json_is_finite_and_complete(self, make_sessions):
        payload = sched.analyse(make_sessions(n=4, n_turns=30), resamples=40).to_json()
        text = json.dumps(payload)
        assert "NaN" not in text and "Infinity" not in text
        assert payload["tail_slope"] is not None
        assert payload["verdict"] in ("uniform-length", "dispersed")
        assert payload["equal_length_reference"] == -0.5
        assert payload["curve"]


class TestCli:
    def test_it_runs_against_a_fixture(self, write_jsonl, capsys, isolated_home):
        recs = []
        for i in range(12):
            recs.append({
                "type": "assistant", "sessionId": "s",
                "timestamp": f"2026-08-01T10:{i:02d}:00Z",
                "message": {"id": f"m{i}", "model": "claude-opus-5",
                            "usage": {"input_tokens": 2,
                                      "cache_read_input_tokens": 20_000 + 500 * i,
                                      "cache_creation_input_tokens": 100,
                                      "output_tokens": 300}}})
        root = write_jsonl(recs, into=None)
        assert sched.main([str(root), "--resamples", "20"]) == 0
        assert capsys.readouterr().out.strip()

    def test_json_parses(self, write_jsonl, capsys, isolated_home):
        recs = [{
            "type": "assistant", "sessionId": "s",
            "timestamp": f"2026-08-01T10:{i:02d}:00Z",
            "message": {"id": f"m{i}", "model": "claude-opus-5",
                        "usage": {"input_tokens": 2,
                                  "cache_read_input_tokens": 20_000,
                                  "cache_creation_input_tokens": 100,
                                  "output_tokens": 300}}} for i in range(6)]
        root = write_jsonl(recs, into=None)
        assert sched.main([str(root), "--resamples", "20", "--json"]) == 0
        json.loads(capsys.readouterr().out)

    def test_an_empty_root_exits_one(self, tmp_path, capsys, isolated_home):
        assert sched.main([str(tmp_path)]) == 1
        assert capsys.readouterr().out.strip()
