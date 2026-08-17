"""The shared window, and the boundary rule two commands used to disagree on.

`--since`/`--until` is half-open on purpose. The test that matters is that two
adjacent windows partition the data exactly: no turn in both, no turn in
neither. Anything else makes "August" plus "September" not equal to "August and
September", which is how a cost report loses money that was really spent.
"""

from __future__ import annotations

import argparse
from datetime import date

import pytest

from adder.core.filters import Window, add_arguments, day_of, parse_date, span
from adder.core.trace import Session, Turn

OPUS, HAIKU = "claude-opus-5", "claude-haiku-4-5"


def _turn(day: str | None, *, model: str = OPUS, project: str = "proj",
          session: str = "s1", sidechain: bool = False) -> Turn:
    return Turn(session, project, model, uncached_in=0, cache_read=1000,
                cache_write=0, out=100, thinking=0, sidechain=sidechain,
                ts=(f"{day}T12:00:00Z" if day else None))


def _sessions(*turns: Turn) -> dict[str, Session]:
    out: dict[str, Session] = {}
    for t in turns:
        s = out.setdefault(t.session, Session(t.session, t.project))
        s.turns.append(t)
    return out


class TestParseDate:
    def test_absolute(self):
        assert parse_date("2026-08-01") == date(2026, 8, 1)

    @pytest.mark.parametrize("text,days", [("7d", 7), ("2w", 14), ("1m", 30), ("1y", 365)])
    def test_relative(self, text, days):
        today = date(2026, 8, 15)
        assert (today - parse_date(text, today=today)).days == days

    def test_today_and_yesterday(self):
        today = date(2026, 8, 15)
        assert parse_date("today", today=today) == today
        assert parse_date("yesterday", today=today) == date(2026, 8, 14)

    @pytest.mark.parametrize("bad", ["", "  ", "not-a-date", "2026-13-40", "5x"])
    def test_rejects_nonsense_with_a_useful_message(self, bad):
        with pytest.raises(ValueError):
            parse_date(bad)


class TestWindowBoundaries:
    def test_since_is_inclusive_and_until_is_exclusive(self):
        turns = [_turn("2026-08-01"), _turn("2026-08-31"), _turn("2026-09-01")]
        w = Window(since=date(2026, 8, 1), until=date(2026, 9, 1))
        kept = w.turns(turns)
        assert len(kept) == 2
        assert all(t.when.date().month == 8 for t in kept)

    def test_adjacent_windows_partition_exactly(self):
        turns = [_turn(f"2026-08-{d:02d}") for d in range(1, 29)]
        aug = Window(until=date(2026, 8, 15)).turns(turns)
        rest = Window(since=date(2026, 8, 15)).turns(turns)
        assert len(aug) + len(rest) == len(turns)
        assert not ({id(t) for t in aug} & {id(t) for t in rest})

    def test_undated_turns_are_dropped_and_counted(self):
        w = Window(since=date(2026, 1, 1))
        kept = w.turns([_turn("2026-08-01"), _turn(None)])
        assert len(kept) == 1
        assert w.dropped_undated == 1

    def test_undated_turns_survive_when_no_window_is_set(self):
        w = Window(models=(OPUS,))
        assert len(w.turns([_turn(None)])) == 1
        assert w.dropped_undated == 0


class TestWindowPredicates:
    def test_model_is_a_prefix_match(self):
        turns = [_turn("2026-08-01", model="claude-opus-5"),
                 _turn("2026-08-01", model="claude-haiku-4-5-20251001")]
        assert len(Window(models=("claude-haiku",)).turns(turns)) == 1

    def test_project_is_a_case_insensitive_substring(self):
        turns = [_turn("2026-08-01", project="-Users-me-Desktop-Adder"),
                 _turn("2026-08-01", project="-Users-me-other")]
        assert len(Window(projects=("adder",)).turns(turns)) == 1

    def test_session_is_a_prefix_match(self):
        turns = [_turn("2026-08-01", session="abc123"), _turn("2026-08-01", session="zzz")]
        assert len(Window(sessions=("abc",)).turns(turns)) == 1

    def test_sidechain_selects_both_ways(self):
        turns = [_turn("2026-08-01"), _turn("2026-08-01", sidechain=True)]
        assert len(Window(sidechain=True).turns(turns)) == 1
        assert len(Window(sidechain=False).turns(turns)) == 1
        assert len(Window().turns(turns)) == 2

    def test_predicates_compose_as_and(self):
        turns = [_turn("2026-08-01", model=HAIKU, project="a"),
                 _turn("2026-08-01", model=HAIKU, project="b")]
        assert len(Window(models=(HAIKU,), projects=("a",)).turns(turns)) == 1

    def test_empty_window_is_inactive(self):
        assert Window().active is False
        assert Window(min_turns=1).active is True


class TestApply:
    def test_sessions_are_rebuilt_not_mutated(self):
        sessions = _sessions(_turn("2026-08-01"), _turn("2026-09-01"))
        before = len(sessions["s1"].turns)
        out = Window(until=date(2026, 8, 15)).apply(sessions)
        assert len(sessions["s1"].turns) == before, "input map was mutated"
        assert len(out["s1"].turns) == 1

    def test_sessions_emptied_by_the_filter_disappear(self):
        sessions = _sessions(_turn("2026-01-01"))
        assert Window(since=date(2026, 8, 1)).apply(sessions) == {}

    def test_min_turns_drops_short_sessions(self):
        sessions = _sessions(_turn("2026-08-01"), _turn("2026-08-02", session="s2"),
                             _turn("2026-08-03", session="s2"))
        out = Window(min_turns=2).apply(sessions)
        assert set(out) == {"s2"}

    def test_project_and_session_identity_are_preserved(self):
        sessions = _sessions(_turn("2026-08-01"))
        out = Window(min_turns=1).apply(sessions)
        assert out["s1"].project == "proj"
        assert out["s1"].id == "s1"


class TestArgparseIntegration:
    def _parse(self, argv):
        ap = argparse.ArgumentParser()
        add_arguments(ap, root=False)
        return ap.parse_args(argv)

    def test_flags_build_a_window(self):
        a = self._parse(["--since", "2026-08-01", "--project", "adder",
                         "--model-filter", "claude-opus", "--min-turns", "3"])
        w = Window.from_args(a)
        assert w.since == date(2026, 8, 1)
        assert w.projects == ("adder",)
        assert w.models == ("claude-opus",)
        assert w.min_turns == 3

    def test_repeatable_flags_accumulate(self):
        a = self._parse(["--project", "a", "--project", "b"])
        assert Window.from_args(a).projects == ("a", "b")

    def test_subagent_flags_are_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            self._parse(["--only-subagents", "--no-subagents"])

    def test_subagent_flags_map_to_the_tristate(self):
        assert Window.from_args(self._parse(["--only-subagents"])).sidechain is True
        assert Window.from_args(self._parse(["--no-subagents"])).sidechain is False
        assert Window.from_args(self._parse([])).sidechain is None

    def test_bad_date_is_a_usage_error_not_a_traceback(self):
        with pytest.raises(SystemExit):
            self._parse(["--since", "nonsense"])

    def test_root_argument_is_optional_and_defaults(self):
        """Optional, and resolved through the settings layer rather than the parser.

        The parser used to bake `str(DEFAULT_ROOT)` in as the default, which
        made `a.root` always truthy -- so `load`'s fallback to the `root`
        setting was unreachable and the first entry in the settings table did
        nothing at all. The default now lives in `root_of`.
        """
        from adder.core.filters import root_of

        ap = argparse.ArgumentParser()
        add_arguments(ap)
        assert ap.parse_args([]).root is None
        assert root_of(ap.parse_args([]))
        assert str(root_of(ap.parse_args(["/tmp/somewhere"]))) == "/tmp/somewhere"

    def test_the_root_setting_is_what_an_absent_argument_resolves_to(
            self, tmp_path, monkeypatch):
        import json

        from adder.core.filters import root_of

        (tmp_path / ".adder.json").write_text(
            json.dumps({"root": str(tmp_path / "elsewhere")}), encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("ADDER_ROOT", raising=False)
        ap = argparse.ArgumentParser()
        add_arguments(ap)
        assert str(root_of(ap.parse_args([]))) == str(tmp_path / "elsewhere")

    def test_describe_names_the_filters(self):
        w = Window.from_args(self._parse(["--since", "2026-08-01", "--no-subagents"]))
        text = w.describe()
        assert "2026-08-01" in text and "main-chain" in text

    def test_describe_of_nothing_says_so(self):
        assert "everything" in Window().describe()


class TestSpan:
    def test_returns_first_and_last(self):
        sessions = _sessions(_turn("2026-08-01"), _turn("2026-08-09"))
        lo, hi = span(sessions)
        assert lo.date() == date(2026, 8, 1)
        assert hi.date() == date(2026, 8, 9)

    def test_undated_only_is_none(self):
        assert span(_sessions(_turn(None))) == (None, None)


class TestDaysAreLocalDays:
    """A window built from `date.today()` has to be compared against a local day.

    Transcripts stamp UTC. `parse_date("today"/"7d")` builds its boundary from
    `date.today()`, which is local. `day_of` took `.date()` off the UTC instant,
    so the two sides of every comparison were in different calendars and every
    evening turn west of Greenwich was filed under tomorrow: 6,778 of 28,191
    turns on the author's corpus -- 24% of them, carrying $1,484 of spend.

    It reaches `--since`, `--until`, `--by day`, the daily budget burn-down and
    the `verify` cutover, all of which answer a question the operator asked in
    their own timezone.
    """

    def test_an_evening_turn_stays_on_its_own_day(self, tz):
        tz("America/Denver")
        assert day_of("2026-08-02T02:00:00Z") == date(2026, 8, 1)   # 20:00 local

    def test_a_morning_turn_east_of_greenwich_moves_forward(self, tz):
        tz("Asia/Tokyo")
        assert day_of("2026-08-01T23:00:00Z") == date(2026, 8, 2)   # 08:00 local

    def test_utc_is_unchanged(self, tz):
        tz("UTC")
        assert day_of("2026-08-02T02:00:00Z") == date(2026, 8, 2)

    def test_a_naive_timestamp_is_taken_as_local(self, tz):
        tz("America/Denver")
        assert day_of("2026-08-02T02:00:00") == date(2026, 8, 2)

    def test_junk_is_still_none(self):
        assert day_of(None) is None
        assert day_of("not a date") is None
        assert day_of(12345) is None

    def test_the_window_filters_on_the_same_calendar(self, tz, make_turn):
        """`keeps_turn` and `day_of` must not disagree about which day it is."""
        tz("America/Denver")
        t = make_turn(ts="2026-08-02T02:00:00Z")
        assert Window(since=date(2026, 8, 1), until=date(2026, 8, 2)).keeps_turn(t)
        assert not Window(since=date(2026, 8, 2)).keeps_turn(t)
