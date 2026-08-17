"""Am I going to blow the budget, and by when?

Every other report here is retrospective. This one is the only one that answers
a question with a deadline attached, and it answers it in the only way that is
honest: by projecting the *measured* rate forward, with the projection's
assumption stated on screen rather than buried.

Why a naive projection is wrong, and what is done instead
--------------------------------------------------------
`spent / days_elapsed * days_in_period` treats a workload as a constant drip.
Agent spend is not a drip -- it is bursty, concentrated in a few long sessions,
and heavily weekday-shaped. A projection anchored to the mean of a bursty
series swings wildly early in the period, when `days_elapsed` is small and one
expensive day dominates.

So two projections are reported, and the gap between them is the honest
uncertainty:

* **run-rate**, from the mean daily spend over the period so far;
* **recent**, from the median of the last `RECENT_DAYS` *active* days, which is
  robust to a single outlier day and to the weekend gaps that would otherwise
  drag the mean down.

Days with no spend are excluded from the recent median on purpose. Including
them makes a Friday-afternoon check-in project a weekend of zero spend onto
Monday, which is the wrong direction to be wrong in when the question is
"will I go over".

Determinism
-----------
Nothing here reads the clock. `today` is a parameter everywhere, and only
`main()` supplies `date.today()`. A report that silently depends on when it ran
cannot be tested, and this one has an exit code that CI might use.
"""

from __future__ import annotations

import json
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import date, timedelta

from adder.core.filters import day_of

# Days of history the robust projection looks back over.
RECENT_DAYS = 14

PERIODS = ("month", "week", "day", "all")


def period_bounds(period: str, today: date) -> tuple[date, date]:
    """Half-open `[start, end)` for a named period containing `today`.

    `end` is the first day *after* the period, matching the `filters.Window`
    convention so the two never disagree about a boundary day.
    """
    if period == "month":
        start = today.replace(day=1)
        end = date(today.year + (today.month == 12),
                   1 if today.month == 12 else today.month + 1, 1)
        return start, end
    if period == "week":                      # ISO week, Monday-based
        start = today - timedelta(days=today.weekday())
        return start, start + timedelta(days=7)
    if period == "day":
        return today, today + timedelta(days=1)
    if period == "all":
        return date.min, date.max
    raise ValueError(f"unknown period {period!r}; known: {', '.join(PERIODS)}")


def _as_date(key: str) -> date | None:
    """A `by_day` key as a date, or None. The keys are built by `day_of`, but a
    `Burn` can be assembled by hand and a report must not raise on one."""
    try:
        return date.fromisoformat(key)
    except (TypeError, ValueError):
        return None


def period_length(period: str, today: date) -> int:
    if period == "month":
        return monthrange(today.year, today.month)[1]
    if period == "week":
        return 7
    if period == "day":
        return 1
    return 0


@dataclass
class Burn:
    """What has been spent in a period, and what the rest of it looks like."""

    period: str
    start: date
    end: date
    today: date
    limit: float
    spent: float = 0.0
    turns: int = 0
    by_day: dict[str, float] = field(default_factory=dict)
    by_project: dict[str, float] = field(default_factory=dict)

    @property
    def days_elapsed(self) -> int:
        """Days of the period that have happened, including today. Never zero.

        `all` is not a period with a start: its bounds are `date.min`/`date.max`,
        so this counted **739,000 elapsed days** and left 2.9 million of them
        remaining. Every number built on it was then nonsense in the confident
        direction -- a $22 workload spanning three days reported a run rate of
        $0.000030/day, a projection of $108.81 against a $100 budget, and a
        budget that "runs out on 9190-05-01". For `all` the elapsed span is the
        span of the data, which is the only reading of "so far" that exists.
        """
        if self.period == "all":
            return max(1, self._observed_days)
        return max(1, (min(self.today, self.end - timedelta(days=1)) - self.start).days + 1)

    @property
    def _observed_days(self) -> int:
        """Calendar days between the first and last day that carried spend."""
        days = sorted(d for d in (_as_date(k) for k in self.by_day) if d is not None)
        if not days:
            return 1
        return (days[-1] - days[0]).days + 1

    @property
    def days_total(self) -> int:
        # `all` has no remainder to project into: it is every day on record and
        # not one of them is in the future.
        if self.period == "all":
            return self.days_elapsed
        return max(self.days_elapsed, (self.end - self.start).days)

    @property
    def days_left(self) -> int:
        return max(0, self.days_total - self.days_elapsed)

    @property
    def run_rate(self) -> float:
        """Mean USD per elapsed day, counting days with no spend."""
        return self.spent / self.days_elapsed

    @property
    def recent_rate(self) -> float:
        """Median USD per ACTIVE day over the recent window.

        Robust to one expensive day and to weekends. Falls back to the run rate
        when there is not enough history to take a median of.
        """
        from adder.util.stats import median

        cutoff = self.today - timedelta(days=RECENT_DAYS)
        active = [v for k, v in self.by_day.items()
                  if v > 0 and _as_date(k) is not None and _as_date(k) > cutoff]
        return median(active) if len(active) >= 3 else self.run_rate

    def projected(self, rate: float) -> float:
        return self.spent + rate * self.days_left

    @property
    def projection(self) -> float:
        """The number to act on: the higher of the two projections.

        Being wrong about a budget is asymmetric. Under-projecting means
        finding out at the end of the month; over-projecting means an
        unnecessary look at the spend. The expensive error is the first one.
        """
        return max(self.projected(self.run_rate), self.projected(self.recent_rate))

    @property
    def over(self) -> bool:
        return bool(self.limit) and self.projection > self.limit

    @property
    def remaining(self) -> float:
        return self.limit - self.spent if self.limit else 0.0

    @property
    def sustainable_daily(self) -> float:
        """Daily spend that lands exactly on the limit. 0.0 when already over."""
        if not self.limit or not self.days_left:
            return 0.0
        return max(0.0, self.remaining / self.days_left)

    @property
    def pace(self) -> float:
        """Projection as a multiple of the limit. 1.0 is exactly on budget."""
        return self.projection / self.limit if self.limit else 0.0

    @property
    def exhausted_on(self) -> date | None:
        """Day the limit is reached at the current rate, if inside this period."""
        if not self.limit:
            return None
        if self.remaining <= 0:
            return self.today          # already gone; the day is today
        rate = max(self.run_rate, self.recent_rate)
        if rate <= 0:
            return None
        # Clamped before the arithmetic: a tiny rate against a large budget
        # produces a day count that overflows `date`, and the answer in that
        # case is "not in this period" rather than a traceback.
        days = self.remaining / rate
        if days > self.days_left:
            return None
        return self.today + timedelta(days=int(days))


def measure(sessions, *, period: str = "month", today: date | None = None,
            limit: float = 0.0, on: date | None = None) -> Burn:
    """Sum spend inside the period. Undated turns are excluded, not assumed.

    A turn with no timestamp cannot be placed in a month, and quietly counting
    it against the current one would make a budget report disagree with the
    `trace` window that produced it.

    Days are the **UTC** days the transcript records, not local days, which is
    the same convention `filters.Window` uses. They agree with each other and
    can disagree with the reader's calendar by a few hours near midnight; that
    is the correct trade, because a report whose buckets shift with the
    machine's timezone cannot be compared against the same report run
    elsewhere.
    """
    today = today or date.today()
    start, end = period_bounds(period, today)
    b = Burn(period=period, start=start, end=end, today=today, limit=limit)
    for s in sessions.values():
        for t in s.turns:
            w = t.when
            if w is None:
                continue
            d = day_of(w)
            if not (start <= d < end):
                continue
            c = t.cost(on)
            b.spent += c
            b.turns += 1
            key = d.isoformat()
            b.by_day[key] = b.by_day.get(key, 0.0) + c
            b.by_project[t.project] = b.by_project.get(t.project, 0.0) + c
    return b


def report(b: Burn, *, top_projects: int = 5) -> str:
    from adder.util.render import bar, money, table

    lines = [f"  Spend this {b.period}: {money(b.spent)} over {b.turns:,} turns", ""]
    if b.period != "all":
        lines.append(f"  day {b.days_elapsed} of {b.days_total} · "
                     f"{b.days_left} left")
    else:
        lines.append(f"  {b.days_elapsed} days on record · nothing left to "
                     f"project into")
    lines.append(f"  run rate      {money(b.run_rate)}/day  (mean of every elapsed day)")
    lines.append(f"  recent rate   {money(b.recent_rate)}/day  "
                 f"(median of active days in the last {RECENT_DAYS})")

    if b.limit:
        lines.append("")
        width = 28
        lines.append(f"  budget        {money(b.limit)}")
        lines.append(f"  spent         {money(b.spent):<12}{bar(b.spent / b.limit, width)} "
                     f"{b.spent / b.limit:.0%}")
        lines.append(f"  projected     {money(b.projection):<12}"
                     f"{bar(b.pace, width)} {b.pace:.0%}")
        lines.append("")
        if b.spent >= b.limit:
            lines.append(f"  Already {money(b.spent - b.limit)} past the limit, with "
                         f"{b.days_left} days of the {b.period} left.")
        elif b.over:
            excess = b.projection - b.limit
            lines.append(f"  OVER by {money(excess)} at the current pace.")
            if b.days_left:
                lines.append(f"  Landing on budget needs {money(b.sustainable_daily)}/day "
                             f"for the remaining {b.days_left} days, against a recent "
                             f"{money(b.recent_rate)}/day.")
            when = b.exhausted_on
            if when:
                lines.append(f"  The budget runs out on {when.isoformat()}.")
        else:
            lines.append(f"  Inside budget: {money(b.limit - b.projection)} of headroom "
                         f"at the projected pace.")
    else:
        lines.append("")
        lines.append("  No budget set. `adder config --init` or ADDER_BUDGET=<usd> "
                     "turns this into a burn-down.")

    if b.by_project:
        lines.append("")
        lines.append("  by project:")
        ranked = sorted(b.by_project.items(), key=lambda kv: -kv[1])[:top_projects]
        rows = [[k[-38:], money(v), f"{v / b.spent:.0%}" if b.spent else ""]
                for k, v in ranked]
        lines += table(rows, ["project", "cost", "share"], align="<>>")

    if b.by_day:
        lines.append("")
        lines.append("  by day:")
        peak = max(b.by_day.values()) or 1.0
        for day in sorted(b.by_day)[-RECENT_DAYS:]:
            v = b.by_day[day]
            lines.append(f"    {day}  {bar(v / peak, 24)} {money(v)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from adder.core import settings
    from adder.core.filters import add_arguments as add_window
    from adder.core.filters import load as load_window

    ap = argparse.ArgumentParser(
        prog="adder budget",
        description="Burn-down against a spend target, with a robust projection.")
    add_window(ap)
    ap.add_argument("--limit", type=float, default=None, metavar="USD",
                    help="budget for the period (default: the `budget` setting)")
    ap.add_argument("--period", choices=PERIODS, default="month",
                    help="budget period (default: %(default)s)")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when the projection is over the limit")
    a = ap.parse_args(argv)

    limit = a.limit if a.limit is not None else float(settings.get("budget"))
    sessions, _window = load_window(a)

    b = measure(sessions, period=a.period, limit=limit)

    if a.json:
        print(json.dumps({
            "period": b.period, "start": b.start.isoformat(),
            "end": b.end.isoformat(),
            "spent": round(b.spent, 4), "turns": b.turns, "limit": b.limit,
            "days_elapsed": b.days_elapsed, "days_left": b.days_left,
            "run_rate": round(b.run_rate, 4),
            "recent_rate": round(b.recent_rate, 4),
            "projected": round(b.projection, 4),
            "pace": round(b.pace, 4),
            "over": b.over,
            "sustainable_daily": round(b.sustainable_daily, 4),
            "exhausted_on": b.exhausted_on.isoformat() if b.exhausted_on else None,
            "by_day": {k: round(v, 4) for k, v in sorted(b.by_day.items())},
        }))
    else:
        print()
        print(report(b))
        print()
    return 1 if (a.strict and b.over) else 0


if __name__ == "__main__":
    raise SystemExit(main())
