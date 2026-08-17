"""Outlier detection, and the reason it uses a median rather than a mean.

The load-bearing test is `test_one_outlier_does_not_hide_itself`: with a
standard z-score, a single very expensive turn inflates sigma enough to score
below the threshold, so the detector misses the exact case it exists for.
"""

from __future__ import annotations

import json

import pytest

from adder.measure.spend.anomaly import JUMP_TOKENS, Report, _cause, scan


class TestDetection:
    def test_a_uniform_workload_flags_nothing(self, make_session):
        s = make_session(60, growth=0)
        assert scan({"s": s}).turns == []

    def test_one_outlier_does_not_hide_itself(self, make_session, make_turn):
        """The failure mode of a mean-and-sigma detector, as an assertion."""
        s = make_session(200, growth=0, base=10_000)
        s.turns.append(make_turn(read=5_000_000, minutes=999))
        found = scan({"s": s}).turns
        assert found, "the one expensive turn was not flagged"
        assert found[0].cost == max(t.cost() for t in s.turns)

    def test_threshold_is_respected(self, make_session, make_turn):
        s = make_session(100, growth=0, base=10_000)
        s.turns.append(make_turn(read=200_000, minutes=999))
        assert scan({"s": s}, turn_z=1.0).turns
        assert not scan({"s": s}, turn_z=1e9).turns

    def test_findings_are_sorted_by_cost(self, make_session, make_turn):
        s = make_session(100, growth=0, base=10_000)
        s.turns.append(make_turn(read=200_000, minutes=998))
        s.turns.append(make_turn(read=900_000, minutes=999))
        found = scan({"s": s}).turns
        assert found == sorted(found, key=lambda f: -f.cost)

    def test_empty_input(self):
        rep = scan({})
        assert rep.n_turns == 0
        assert rep.turns == []


class TestCause:
    def test_a_write_dominant_turn_is_a_rebuild(self, make_turn):
        """With a predecessor, a write-dominant turn really did lose a cache.

        This used to pass `None` for the predecessor, which is the one case
        where it is NOT a rebuild -- see the opening-write test below. The claim
        being made here is about write-dominance, so the turn now has something
        to have been rebuilt from.
        """
        prev = make_turn(read=500_000)
        cause, detail = _cause(make_turn(read=0, write=500_000), prev)
        assert cause == "prefix rebuild"
        assert "0.10x" in detail

    def test_the_first_turn_of_a_chain_is_an_opening_write(self, make_turn):
        """Nothing was invalidated: there was no prior cache to invalidate.

        Labelling it a rebuild sends someone hunting a cache loss that never
        happened, and prices the unavoidable cost of starting a context as
        "pure overhead". `trace.Session.cache_misses` and `cache.analyse` both
        already skip turn 0 for the same reason.
        """
        cause, detail = _cause(make_turn(read=0, write=500_000), None)
        assert cause == "opening write"
        assert "no prior cache" in detail

    def test_the_opening_write_is_not_called_overhead(self, make_turn):
        _, detail = _cause(make_turn(read=0, write=500_000), None)
        assert "overhead" not in detail

    def test_a_large_admission_is_a_context_jump(self, make_turn):
        prev = make_turn(read=10_000)
        cur = make_turn(read=10_000 + JUMP_TOKENS + 1)
        assert _cause(cur, prev)[0] == "context jump"

    def test_rebuild_wins_over_jump(self, make_turn):
        """A rebuild also grows the context; reporting the jump misdirects."""
        prev = make_turn(read=10_000)
        cur = make_turn(read=0, write=500_000)
        assert _cause(cur, prev)[0] == "prefix rebuild"

    def test_fast_mode_is_named(self, make_turn):
        assert _cause(make_turn(speed="fast"), make_turn())[0] == "fast mode"

    def test_a_long_answer_is_named(self, make_turn):
        assert _cause(make_turn(out=20_000), make_turn())[0] == "long output"

    def test_otherwise_it_is_just_a_big_context(self, make_turn):
        cause, detail = _cause(make_turn(read=800_000), make_turn(read=800_000))
        assert cause == "big context"
        assert "no single event" in detail


class TestReportShape:
    def test_excess_is_measured_above_the_median_not_the_whole_bill(self, make_session,
                                                                    make_turn):
        s = make_session(100, growth=0, base=10_000)
        s.turns.append(make_turn(read=2_000_000, minutes=999))
        rep = scan({"s": s})
        assert 0 < rep.excess < rep.flagged_cost

    def test_by_cause_covers_every_finding_not_just_the_shown_ones(self, make_session,
                                                                   make_turn):
        s = make_session(100, growth=0, base=10_000)
        for i in range(6):
            s.turns.append(make_turn(read=0, write=800_000, minutes=900 + i))
        rep = scan({"s": s})
        counted = sum(n for n, _ in rep.by_cause().values())
        assert counted == len(rep.turns)

    def test_session_outliers_are_found(self, make_session):
        sessions = {f"s{i}": make_session(30, sid=f"s{i}", growth=0, base=10_000)
                    for i in range(12)}
        sessions["hot"] = make_session(30, sid="hot", growth=0, base=900_000)
        assert any(f.key == "hot"[:8] for f in scan(sessions).sessions)

    def test_median_turn_is_reported(self, make_session):
        rep = scan({"s": make_session(20, growth=0)})
        assert rep.median_turn > 0

    def test_empty_report_text(self):
        assert "Nothing to analyse" in __import__(
            "adder.measure.spend.anomaly", fromlist=["report"]).report(
                Report(turns=[], sessions=[], total=0.0, n_turns=0))


class TestCli:
    def test_json(self, make_session, tmp_path, capsys, write_jsonl):
        from adder.measure.spend.anomaly import main

        records = []
        for i in range(40):
            records.append({
                "type": "assistant", "sessionId": "s",
                "timestamp": f"2026-08-01T10:{i:02d}:00Z",
                "message": {"id": f"m{i}", "model": "claude-opus-5",
                            "usage": {"input_tokens": 1,
                                      "cache_read_input_tokens": 10_000,
                                      "output_tokens": 100}, "content": []}})
        records.append({
            "type": "assistant", "sessionId": "s",
            "timestamp": "2026-08-01T11:00:00Z",
            "message": {"id": "big", "model": "claude-opus-5",
                        "usage": {"input_tokens": 1,
                                  "cache_creation_input_tokens": 900_000,
                                  "output_tokens": 100}, "content": []}})
        write_jsonl(records)
        assert main([str(tmp_path), "--json"]) == 0
        d = json.loads(capsys.readouterr().out)
        assert d["turns"], "the expensive turn was not reported"
        assert d["turns"][0]["cause"] == "prefix rebuild"

    def test_text_runs(self, tmp_path, write_jsonl, capsys):
        from adder.measure.spend.anomaly import main

        write_jsonl([{"type": "assistant", "sessionId": "s",
                      "timestamp": "2026-08-01T10:00:00Z",
                      "message": {"id": "m1", "model": "claude-opus-5",
                                  "usage": {"input_tokens": 1,
                                            "cache_read_input_tokens": 900,
                                            "output_tokens": 10}, "content": []}}])
        assert main([str(tmp_path)]) == 0
        assert "median" in capsys.readouterr().out


class TestTheBreakdownDecomposesTheHeadline:
    """A part cannot be larger than its whole.

    `Report.excess` is the headline -- cost above the median turn, because "a
    flagged turn still had to happen, it just cost more than it should have".
    `by_cause` summed the *full* cost of each flagged turn instead, so the rows
    were measuring a different quantity than the total above them. On the
    author's corpus a single row read $412.13 under a headline of $408.09.
    """

    def _report(self, make_session):
        s = make_session(60, base=20_000, growth=1_000)
        s.turns[30].cache_write = 900_000
        s.turns[30].cache_read = 0
        s.turns[45].out = 60_000
        return scan({"s": s})

    def test_the_causes_sum_to_the_headline(self, make_session):
        rep = self._report(make_session)
        assert sum(c for _, c in rep.by_cause().values()) == pytest.approx(rep.excess)

    def test_no_single_cause_exceeds_the_headline(self, make_session):
        rep = self._report(make_session)
        assert all(c <= rep.excess + 1e-9 for _, c in rep.by_cause().values())

    def test_the_turn_counts_still_cover_every_finding(self, make_session):
        rep = self._report(make_session)
        assert sum(n for n, _ in rep.by_cause().values()) == len(rep.turns)

    def test_a_cause_is_never_negative(self, make_session):
        """A flagged turn below the median contributes zero, not a credit."""
        rep = self._report(make_session)
        assert all(c >= 0.0 for _, c in rep.by_cause().values())
