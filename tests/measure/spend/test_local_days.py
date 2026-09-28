"""Every spend report files a turn under the same, local, day.

`sessions` and `export` go through `filters.day_of`, which is local. `anomaly`
and `agents` sliced `when[:10]` off the UTC string and `limits` formatted the
UTC datetime, so one evening turn west of Greenwich was on two different days
depending on which report was asked. The timestamp used throughout is 03:30 UTC
on 2 August, which is still 1 August in New York.
"""
from __future__ import annotations

from datetime import datetime, timezone

LATE = "2026-08-02T03:30:00+00:00"


class TestAnomaly:
    def _report(self):
        from adder.measure.spend.anomaly import Finding, Report

        turn = Finding(kind="turn", key="s:1", project="p", cost=3.0, z=7.25,
                       cause="growth", detail="d", when=LATE)
        sess = Finding(kind="session", key="s", project="p", cost=9.0, z=4.5,
                       cause="", detail="", when=LATE)
        return Report(turns=[turn], sessions=[sess], total=12.0, n_turns=10,
                      median_turn=0.1)

    def test_dates_are_local(self, tz):
        from adder.measure.spend.anomaly import report

        tz("America/New_York")
        out = report(self._report())
        assert "2026-08-01" in out
        assert "2026-08-02" not in out

    def test_robust_z_prints_as_z_not_as_a_multiple(self, tz):
        """"7x" reads as seven times the median; it is 7.25 MADs above it."""
        from adder.measure.spend.anomaly import report

        tz("UTC")
        out = report(self._report())
        assert "7.2" in out and "4.5" in out
        assert "7x" not in out and "4x" not in out and "5x" not in out


class TestAgents:
    def test_missed_delegation_dates_are_local(self, tz):
        from adder.measure.spend.agents import AgentReport, Missed, report

        tz("America/New_York")
        rep = AgentReport(runs=[], total_cost=10.0, sidechain_cost=0.0,
                          missed=[Missed(session="abcdef123", project="p",
                                         tokens=80_000, when=LATE, inline=2.0,
                                         delegated=0.5)])
        out = report(rep)
        assert "2026-08-01" in out
        assert "2026-08-02" not in out


class TestLimits:
    def test_window_times_are_local(self, tz, make_turn):
        from adder.core.trace import Session
        from adder.measure.spend.limits import build, render

        tz("America/New_York")
        s = Session("s", "p")
        s.turns.append(make_turn(ts=LATE))
        rep = build({"s": s}, now=datetime(2026, 8, 2, 4, 0, tzinfo=timezone.utc))
        out = render(rep)
        # The block opens on the UTC hour, 03:00Z, which is 23:00 the day before.
        assert "2026-08-01 23:00" in out
        assert "2026-08-02 03:00" not in out
