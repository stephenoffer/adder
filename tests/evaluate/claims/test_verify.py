"""Before/after verification must be able to report FAILURE, not just success."""
from __future__ import annotations

from datetime import date

import pytest

from adder.core.trace import Session, Turn
from adder.evaluate.claims.verify import Window, _day, compare, report

OPUS = "claude-opus-5"


def _turn(ts: str, out: int, ctx: int, sid: str = "s") -> Turn:
    return Turn(sid, "p", OPUS, uncached_in=0, cache_read=ctx, cache_write=0,
                out=out, thinking=0, sidechain=False, ts=ts)


class TestDayParsing:
    def test_parses_iso_with_z(self):
        assert _day("2026-08-14T10:00:00Z") == date(2026, 8, 14)

    @pytest.mark.parametrize("bad", [None, "", "not-a-date"])
    def test_bad_timestamps_are_skipped(self, bad):
        assert _day(bad) is None


class TestWindow:
    def test_empty_window_has_no_divide_by_zero(self):
        w = Window("x")
        assert w.out_per_turn == 0.0 and w.cost_per_turn == 0.0 and w.median_ctx == 0.0


class TestReport:
    def _sessions(self, before_out, after_out, before_ctx, after_ctx):
        b, a = Session("b", "p"), Session("a", "p")
        b.turns = [_turn("2026-07-01T00:00:00Z", before_out, before_ctx, "b") for _ in range(50)]
        a.turns = [_turn("2026-09-01T00:00:00Z", after_out, after_ctx, "a") for _ in range(50)]
        return {"b": b, "a": a}

    def test_reports_a_real_improvement(self, monkeypatch):
        sess = self._sessions(2000, 500, 400_000, 100_000)
        monkeypatch.setattr("adder.evaluate.claims.verify.load_sessions", lambda root: sess)
        text = report(date(2026, 8, 1))
        assert "saved" in text and "ROSE" not in text

    def test_compare_splits_on_the_cutover(self, monkeypatch):
        sess = self._sessions(2000, 500, 400_000, 100_000)
        monkeypatch.setattr("adder.evaluate.claims.verify.load_sessions", lambda root: sess)
        b, a = compare(date(2026, 8, 1))
        assert b.turns == 50 and a.turns == 50
        assert b.out_per_turn == 2000 and a.out_per_turn == 500

    def test_refuses_to_claim_saving_when_cost_rose(self, monkeypatch):
        sess = self._sessions(2000, 500, 100_000, 400_000)   # terser but bigger context
        monkeypatch.setattr("adder.evaluate.claims.verify.load_sessions", lambda root: sess)
        text = report(date(2026, 8, 1))
        assert "ROSE" in text and "Do not claim a saving" in text

    def test_detects_terseness_cancelled_by_longer_sessions(self, monkeypatch):
        """The measured real-world case: output/turn down, cost/turn up."""
        s1 = Session("a", "p")
        s1.turns = [_turn("2026-07-01T00:00:00Z", 1400, 290_000, "a") for _ in range(40)]
        s2 = Session("b", "p")
        s2.turns = [_turn("2026-09-01T00:00:00Z", 700, 360_000, "b") for _ in range(160)]
        monkeypatch.setattr("adder.evaluate.claims.verify.load_sessions", lambda root: {"a": s1, "b": s2})
        text = report(date(2026, 8, 1))
        assert "CANCELLED OUT" in text
        assert "verbosity effect" in text and "session-length effect" in text

    def test_handles_one_sided_data(self, monkeypatch):
        s = Session("s", "p")
        s.turns = [_turn("2026-07-01T00:00:00Z", 500, 100_000) for _ in range(10)]
        monkeypatch.setattr("adder.evaluate.claims.verify.load_sessions", lambda root: {"s": s})
        assert "Not enough data" in report(date(2026, 8, 1))

    def test_always_states_the_uncontrolled_caveat(self, monkeypatch):
        sess = self._sessions(2000, 500, 400_000, 100_000)
        monkeypatch.setattr("adder.evaluate.claims.verify.load_sessions", lambda root: sess)
        assert "not an A/B" in report(date(2026, 8, 1))


class TestASessionIsNotSplitAtTheCutover:
    """Two identical sessions, each running past midnight, straddling the cutover.

    Split per turn, the late large-context turns of the first one landed after
    the cutover: turns/session -39% and cost/turn +17% with nothing changed.
    """

    def _pair(self):
        out = {}
        for sid, day in (("x", "2026-08-01"), ("y", "2026-08-02")):
            s = Session(sid, "p")
            s.turns = [_turn(f"{day}T23:{i // 2:02d}:00Z" if i < 100 else
                             f"{day[:-2]}{int(day[-2:]) + 1:02d}T00:{(i - 100) // 2:02d}:00Z",
                             400, 10_000 + 1_000 * i, sid) for i in range(150)]
            out[sid] = s
        return out

    def test_identical_sessions_compare_equal(self, monkeypatch, tz):
        tz("UTC")
        sess = self._pair()
        monkeypatch.setattr("adder.evaluate.claims.verify.load_sessions", lambda root: sess)
        b, a = compare(date(2026, 8, 2))
        assert (b.sessions, a.sessions) == (1, 1)
        assert b.turns == a.turns == 150
        assert b.cost_per_turn == pytest.approx(a.cost_per_turn)

    def test_no_change_is_not_reported_as_a_rise(self, monkeypatch, tz):
        tz("UTC")
        sess = self._pair()
        monkeypatch.setattr("adder.evaluate.claims.verify.load_sessions", lambda root: sess)
        text = report(date(2026, 8, 2), root="/nonexistent")
        assert "ROSE" not in text and "did not move" in text
