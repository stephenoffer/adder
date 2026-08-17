"""Budget periods, projections, and the guarantee that none of it reads a clock.

`today` is a parameter throughout so these tests are deterministic. A budget
report that depends on when it ran cannot be tested, and this one has an exit
code CI might use.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from adder.measure.spend.budget import PERIODS, Burn, measure, period_bounds, period_length, report


class TestPeriodBounds:
    def test_month_is_half_open(self):
        start, end = period_bounds("month", date(2026, 8, 15))
        assert start == date(2026, 8, 1)
        assert end == date(2026, 9, 1)

    def test_december_rolls_the_year(self):
        start, end = period_bounds("month", date(2026, 12, 20))
        assert (start, end) == (date(2026, 12, 1), date(2027, 1, 1))

    def test_week_starts_on_monday(self):
        start, end = period_bounds("week", date(2026, 8, 15))   # a Saturday
        assert start.weekday() == 0
        assert (end - start).days == 7

    def test_day(self):
        start, end = period_bounds("day", date(2026, 8, 15))
        assert (end - start).days == 1

    def test_all_covers_everything(self):
        start, end = period_bounds("all", date(2026, 8, 15))
        assert start < date(1900, 1, 1) < end

    def test_unknown_period_is_an_error(self):
        with pytest.raises(ValueError):
            period_bounds("fortnight", date(2026, 8, 15))

    @pytest.mark.parametrize("p", PERIODS)
    def test_every_declared_period_resolves(self, p):
        assert period_bounds(p, date(2026, 8, 15))

    def test_period_length_knows_month_lengths(self):
        assert period_length("month", date(2026, 2, 10)) == 28
        assert period_length("month", date(2026, 8, 10)) == 31
        assert period_length("week", date(2026, 8, 10)) == 7


class TestMeasure:
    def test_only_counts_turns_inside_the_period(self, make_session, make_turn):
        s = make_session(5)                      # August 1st
        s.turns.append(make_turn(ts="2026-07-01T10:00:00Z"))
        b = measure({"s": s}, period="month", today=date(2026, 8, 15))
        assert b.turns == 5

    def test_undated_turns_are_excluded_rather_than_assumed(self, make_session,
                                                            make_turn):
        s = make_session(3)
        s.turns.append(make_turn(ts=None))
        b = measure({"s": s}, period="month", today=date(2026, 8, 15))
        assert b.turns == 3

    def test_spend_is_split_by_day_and_project(self, make_session):
        s = make_session(48, minutes_apart=60)   # spans two days
        b = measure({"s": s}, period="month", today=date(2026, 8, 15))
        assert len(b.by_day) >= 2
        assert sum(b.by_day.values()) == pytest.approx(b.spent)
        assert sum(b.by_project.values()) == pytest.approx(b.spent)


class TestProjection:
    def _burn(self, **kw):
        b = Burn(period="month", start=date(2026, 8, 1), end=date(2026, 9, 1),
                 today=date(2026, 8, 10), limit=kw.pop("limit", 100.0))
        for k, v in kw.items():
            setattr(b, k, v)
        return b

    def test_days_elapsed_includes_today(self):
        assert self._burn().days_elapsed == 10

    def test_days_left_is_the_rest_of_the_period(self):
        b = self._burn()
        assert b.days_total == 31
        assert b.days_left == 21

    def test_days_elapsed_is_never_zero(self):
        b = Burn("day", date(2026, 8, 10), date(2026, 8, 11), date(2026, 8, 10), 0.0)
        assert b.days_elapsed == 1

    def test_run_rate_is_the_mean_of_elapsed_days(self):
        assert self._burn(spent=50.0).run_rate == pytest.approx(5.0)

    def test_recent_rate_falls_back_without_enough_active_days(self):
        b = self._burn(spent=50.0, by_day={"2026-08-09": 50.0})
        assert b.recent_rate == pytest.approx(b.run_rate)

    def test_recent_rate_is_a_median_of_active_days(self):
        b = self._burn(spent=60.0, by_day={"2026-08-07": 10.0, "2026-08-08": 20.0,
                                           "2026-08-09": 30.0})
        assert b.recent_rate == pytest.approx(20.0)

    def test_recent_rate_ignores_zero_days(self):
        b = self._burn(spent=60.0, by_day={"2026-08-06": 0.0, "2026-08-07": 10.0,
                                           "2026-08-08": 20.0, "2026-08-09": 30.0})
        assert b.recent_rate == pytest.approx(20.0)

    def test_projection_takes_the_higher_of_the_two(self):
        b = self._burn(spent=60.0, by_day={"2026-08-07": 10.0, "2026-08-08": 20.0,
                                           "2026-08-09": 30.0})
        assert b.projection == pytest.approx(max(b.projected(b.run_rate),
                                                 b.projected(b.recent_rate)))
        assert b.projection >= b.spent

    def test_over_needs_a_limit(self):
        assert self._burn(limit=0.0, spent=9e9).over is False

    def test_pace_is_a_multiple_of_the_limit(self):
        b = self._burn(limit=100.0, spent=100.0)
        assert b.pace >= 1.0

    def test_sustainable_daily_lands_on_the_limit(self):
        b = self._burn(limit=100.0, spent=37.0)
        assert b.sustainable_daily == pytest.approx((100.0 - 37.0) / 21)

    def test_sustainable_daily_is_zero_when_already_over(self):
        assert self._burn(limit=10.0, spent=50.0).sustainable_daily == 0.0

    def test_exhausted_on_is_none_without_a_limit(self):
        assert self._burn(limit=0.0).exhausted_on is None

    def test_exhausted_on_is_today_when_the_limit_is_gone(self):
        assert self._burn(limit=10.0, spent=50.0).exhausted_on == date(2026, 8, 10)

    def test_exhausted_on_is_none_when_it_falls_outside_the_period(self):
        assert self._burn(limit=1e6, spent=1.0).exhausted_on is None


class TestReport:
    def test_no_limit_says_so(self, make_sessions):
        b = measure(make_sessions(1, 5), today=date(2026, 8, 15))
        assert "No budget set" in report(b)

    def test_over_budget_names_the_excess(self, make_sessions):
        b = measure(make_sessions(2, 60), today=date(2026, 8, 15), limit=0.01)
        assert "past the limit" in report(b)

    def test_inside_budget_reports_headroom(self, make_sessions):
        b = measure(make_sessions(1, 3), today=date(2026, 8, 15), limit=1e6)
        assert "headroom" in report(b)


class TestCli:
    def _root(self, write_jsonl):
        return write_jsonl([
            {"type": "assistant", "sessionId": "s",
             "timestamp": "2026-08-01T10:00:00Z",
             "message": {"id": "m1", "model": "claude-opus-5",
                         "usage": {"input_tokens": 1,
                                   "cache_read_input_tokens": 500_000,
                                   "output_tokens": 1000}, "content": []}}])

    def test_json(self, write_jsonl, capsys):
        from adder.measure.spend.budget import main

        root = self._root(write_jsonl)
        assert main([str(root), "--json", "--period", "all"]) == 0
        d = json.loads(capsys.readouterr().out)
        assert d["turns"] == 1
        assert d["period"] == "all"

    def test_strict_exits_one_when_over(self, write_jsonl, capsys):
        from adder.measure.spend.budget import main

        root = self._root(write_jsonl)
        assert main([str(root), "--limit", "0.0001", "--strict", "--period", "all"]) == 1

    def test_strict_exits_zero_when_inside(self, write_jsonl, capsys):
        from adder.measure.spend.budget import main

        root = self._root(write_jsonl)
        assert main([str(root), "--limit", "100000", "--strict", "--period", "all"]) == 0


class TestThePeriodWithNoDeadline:
    """`all` has no start, no end, and nothing to project into.

    Its bounds are `date.min`/`date.max`, so `days_elapsed` counted 739,000 days
    and `days_left` 2.9 million. A three-day, $22 workload then reported a run
    rate of $0.000030/day, a projection of $108.81 against a $100 budget it is
    nowhere near, and a budget that "runs out on 9190-05-01".
    """

    def _burn(self, make_sessions, limit=100.0):
        from datetime import date

        from adder.measure.spend.budget import measure

        return measure(make_sessions(3, 40), period="all",
                       today=date(2026, 8, 5), limit=limit)

    def test_elapsed_days_are_the_days_on_record(self, make_sessions):
        b = self._burn(make_sessions)
        assert 1 <= b.days_elapsed <= 5

    def test_nothing_is_left_to_project_into(self, make_sessions):
        b = self._burn(make_sessions)
        assert b.days_left == 0

    def test_the_projection_is_what_was_spent(self, make_sessions):
        b = self._burn(make_sessions)
        assert b.projection == pytest.approx(b.spent)

    def test_the_run_rate_is_spend_over_the_observed_span(self, make_sessions):
        b = self._burn(make_sessions)
        assert b.run_rate == pytest.approx(b.spent / b.days_elapsed)

    def test_no_exhaustion_date_is_invented(self, make_sessions):
        assert self._burn(make_sessions).exhausted_on is None

    def test_a_dated_period_is_unaffected(self, make_sessions):
        from datetime import date

        from adder.measure.spend.budget import measure

        b = measure(make_sessions(3, 40), period="month",
                    today=date(2026, 8, 5), limit=100.0)
        assert b.days_elapsed == 5 and b.days_total == 31 and b.days_left == 26


def test_a_by_day_key_that_is_not_a_date_does_not_raise():
    from datetime import date

    from adder.measure.spend.budget import Burn

    b = Burn(period="month", start=date(2026, 8, 1), end=date(2026, 9, 1),
             today=date(2026, 8, 5), limit=0.0, spent=1.0,
             by_day={"undated": 1.0})
    assert b.recent_rate >= 0.0
