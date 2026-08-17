"""Attribution, cross-file deduplication, and the turns that are not turns.

Three separate ways a total went wrong before this file existed:

* a resumed session replays earlier turns into a second transcript, and both
  copies were counted;
* a record whose message carried no id sorted next to an unrelated turn,
  because two different position counters were being compared;
* a model missing from `prices.py` was dropped in silence, so the report was a
  lower bound that did not say so.
"""

from __future__ import annotations

import json

import pytest

from adder.core.trace import (
    GROUPINGS,
    Session,
    Turn,
    group_by,
    is_compaction,
    iter_file,
    load_sessions,
    summarize_sessions,
    transcripts,
)

OPUS, HAIKU = "claude-opus-5", "claude-haiku-4-5"


def _rec(mid, *, out=100, model=OPUS, session="s", ts="2026-08-14T10:00:00Z",
         tools=(), read=1000):
    blocks = [{"type": "tool_use", "name": t} for t in tools] or [
        {"type": "text", "text": "hi"}]
    u = {"input_tokens": 2, "cache_read_input_tokens": read,
         "cache_creation_input_tokens": 0, "output_tokens": out}
    return {"type": "assistant", "timestamp": ts, "sessionId": session,
            "message": {"id": mid, "model": model, "usage": u, "content": blocks}}


def _write(tmp_path, records, name="s.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in records))
    return p


def _turn(model=OPUS, *, project="p", session="s", out=100, read=1000,
          tools=(), sidechain=False, ts="2026-08-14T10:00:00Z") -> Turn:
    return Turn(session, project, model, uncached_in=0, cache_read=read,
                cache_write=0, out=out, thinking=0, sidechain=sidechain,
                ts=ts, tools=tuple(tools))


def _sessions(*turns: Turn) -> dict[str, Session]:
    out: dict[str, Session] = {}
    for t in turns:
        s = out.setdefault(t.session, Session(t.session, t.project))
        s.turns.append(t)
    return out


class TestOrdering:
    def test_anonymous_records_keep_their_position(self, tmp_path):
        """The bug: an id-less record sorted by a counter the others did not use."""
        anon = _rec("m2", out=2)
        del anon["message"]["id"]
        p = _write(tmp_path, [_rec("m1", out=1), anon, _rec("m3", out=3)])
        assert [t.out for t in iter_file(p)] == [1, 2, 3]

    def test_many_turns_stay_ordered(self, tmp_path):
        """Also the regression guard for the quadratic `order.index` it replaced."""
        p = _write(tmp_path, [_rec(f"m{i}", out=i) for i in range(300)])
        assert [t.out for t in iter_file(p)] == list(range(300))


class TestCrossFileDedup:
    def test_a_replayed_session_is_not_counted_twice(self, tmp_path):
        _write(tmp_path, [_rec("m1"), _rec("m2")], name="a.jsonl")
        _write(tmp_path, [_rec("m1"), _rec("m2"), _rec("m3")], name="b.jsonl")
        sessions = load_sessions(tmp_path)
        assert sessions["s"].n_turns == 3

    def test_same_id_in_different_sessions_is_kept(self, tmp_path):
        """Message ids are unique per conversation, not globally."""
        _write(tmp_path, [_rec("m1", session="s1")], name="a.jsonl")
        _write(tmp_path, [_rec("m1", session="s2")], name="b.jsonl")
        sessions = load_sessions(tmp_path)
        assert sorted(sessions) == ["s1", "s2"]

    def test_dedup_survives_the_parse_cache(self, tmp_path, monkeypatch):
        import adder.core.trace as trace

        monkeypatch.setattr(trace, "CACHE_PATH", tmp_path / "cache.pkl")
        _write(tmp_path, [_rec("m1")], name="a.jsonl")
        _write(tmp_path, [_rec("m1")], name="b.jsonl")
        first = load_sessions(tmp_path, use_cache=True)["s"].n_turns
        second = load_sessions(tmp_path, use_cache=True)["s"].n_turns
        assert first == second == 1


class TestUnknownAndSynthetic:
    def test_unknown_models_are_counted_not_just_dropped(self, tmp_path):
        p = _write(tmp_path, [_rec("m1"), _rec("m2", model="gpt-9-turbo")])
        seen: dict[str, int] = {}
        turns = list(iter_file(p, unknown=seen))
        assert len(turns) == 1
        assert seen == {"gpt-9-turbo": 1}

    def test_summary_separates_synthetic_from_unknown(self, tmp_path):
        p = _write(tmp_path, [_rec("m1"), _rec("m2", model="<synthetic>"),
                              _rec("m3", model="gpt-9-turbo")])
        seen: dict[str, int] = {}
        sessions = {"s": Session("s", "p")}
        sessions["s"].turns = list(iter_file(p, unknown=seen))
        s = summarize_sessions(sessions, unknown=seen)
        assert s.synthetic_turns == 1
        assert s.unknown_models == {"gpt-9-turbo": 1}
        assert s.unknown_turns == 1

    def test_synthetic_records_are_not_turns(self, tmp_path):
        p = _write(tmp_path, [_rec("m1"), _rec("m2", model="<synthetic>")])
        assert len(list(iter_file(p))) == 1

    def test_unknown_tally_survives_the_parse_cache(self, tmp_path, monkeypatch):
        import adder.core.trace as trace

        monkeypatch.setattr(trace, "CACHE_PATH", tmp_path / "cache.pkl")
        _write(tmp_path, [_rec("m1"), _rec("m2", model="gpt-9-turbo")])
        a: dict[str, int] = {}
        load_sessions(tmp_path, use_cache=True, unknown=a)
        b: dict[str, int] = {}
        load_sessions(tmp_path, use_cache=True, unknown=b)
        assert a == b == {"gpt-9-turbo": 1}


class TestGrouping:
    def test_every_grouping_runs(self):
        sessions = _sessions(_turn(tools=("Read",)), _turn(model=HAIKU))
        for by in GROUPINGS:
            assert group_by(sessions, by), f"{by} produced nothing"

    def test_unknown_grouping_is_an_error(self):
        with pytest.raises(ValueError):
            group_by(_sessions(_turn()), "phase-of-the-moon")

    def test_groups_are_sorted_most_expensive_first(self):
        sessions = _sessions(_turn(model=HAIKU, read=10), _turn(model=OPUS, read=100_000))
        assert group_by(sessions, "model")[0].key == OPUS

    def test_model_shares_sum_to_the_total(self):
        sessions = _sessions(_turn(), _turn(model=HAIKU), _turn())
        total = summarize_sessions(sessions).total
        assert sum(g.cost for g in group_by(sessions, "model")) == pytest.approx(total)

    def test_tool_grouping_counts_a_turn_under_each_tool(self):
        sessions = _sessions(_turn(tools=("Read", "Bash")))
        groups = {g.key: g for g in group_by(sessions, "tool")}
        assert set(groups) == {"Read", "Bash"}
        assert groups["Read"].turns == groups["Bash"].turns == 1

    def test_turns_with_no_tool_call_are_their_own_bucket(self):
        groups = {g.key: g for g in group_by(_sessions(_turn()), "tool")}
        assert "(no tool call)" in groups

    def test_day_grouping_uses_the_date(self, tz):
        """Two turns either side of local midnight land on different days.

        The timezone is pinned rather than inherited. Days are local here --
        `--since today` is built from `date.today()`, so filing a UTC instant
        under its UTC day filed every evening turn west of Greenwich under
        tomorrow -- and a test that reads the machine's own zone would pass in
        one place and fail in another.
        """
        tz("UTC")
        sessions = _sessions(_turn(ts="2026-08-01T23:59:00Z"),
                             _turn(ts="2026-08-02T00:01:00Z"))
        assert {g.key for g in group_by(sessions, "day")} == {"2026-08-01", "2026-08-02"}

    def test_a_turn_is_filed_under_its_local_day(self, tz):
        """Evening in Denver is already tomorrow in UTC. The operator's day wins."""
        tz("America/Denver")
        sessions = _sessions(_turn(ts="2026-08-02T02:00:00Z"))   # 20:00 on the 1st
        assert {g.key for g in group_by(sessions, "day")} == {"2026-08-01"}

    def test_the_same_instant_files_differently_east_of_greenwich(self, tz):
        tz("Asia/Tokyo")
        sessions = _sessions(_turn(ts="2026-08-01T23:00:00Z"))   # 08:00 on the 2nd
        assert {g.key for g in group_by(sessions, "day")} == {"2026-08-02"}

    def test_undated_turns_group_together(self):
        assert group_by(_sessions(_turn(ts=None)), "day")[0].key == "undated"

    def test_group_tracks_distinct_sessions(self):
        sessions = _sessions(_turn(session="a"), _turn(session="b"))
        assert group_by(sessions, "model")[0].sessions == {"a", "b"}

    def test_cost_per_turn_of_an_empty_group_is_zero(self):
        from adder.core.trace import Group

        assert Group("x").cost_per_turn == 0.0


class TestConveniences:
    def test_total_tokens_includes_both_directions(self):
        t = _turn(out=50, read=1000)
        assert t.total_tokens == t.context + 50

    def test_cache_hit_rate(self):
        assert _turn(read=900).cache_hit_rate == pytest.approx(1.0)
        assert Turn("s", "p", OPUS, 0, 0, 0, 0, 0, False).cache_hit_rate == 0.0

    def test_session_wall_clock(self):
        s = Session("s", "p")
        s.turns = [_turn(ts="2026-08-14T10:00:00Z"), _turn(ts="2026-08-14T11:00:00Z")]
        assert s.wall_seconds == pytest.approx(3600)
        assert s.started < s.ended

    def test_wall_clock_without_timestamps_is_zero(self):
        s = Session("s", "p")
        s.turns = [_turn(ts=None)]
        assert s.wall_seconds == 0.0
        assert s.started is None

    def test_cost_by_model_sums_to_session_cost(self):
        s = Session("s", "p")
        s.turns = [_turn(), _turn(model=HAIKU)]
        assert sum(s.cost_by_model().values()) == pytest.approx(s.cost)


class TestTranscripts:
    def test_a_single_file_root_is_itself(self, tmp_path):
        p = _write(tmp_path, [_rec("m1")])
        assert transcripts(p) == [p]

    def test_a_directory_is_walked_recursively(self, tmp_path):
        (tmp_path / "sub").mkdir()
        _write(tmp_path / "sub", [_rec("m1")], name="a.jsonl")
        assert len(transcripts(tmp_path)) == 1

    def test_a_missing_directory_is_empty_not_an_error(self, tmp_path):
        assert transcripts(tmp_path / "nope") == []


class TestTraceCli:
    """The command surface. Every one of these passes an explicit root: a test
    that reads the developer's own `~/.claude` is a test that fails in CI."""

    def _root(self, tmp_path):
        _write(tmp_path, [_rec("m1", tools=("Read",)), _rec("m2", model=HAIKU),
                          _rec("m3", ts="2026-09-01T10:00:00Z")])
        return str(tmp_path)

    def test_plain_run(self, tmp_path, capsys):
        from adder.measure.spend.trace import main

        assert main([self._root(tmp_path), "--no-cache"]) == 0
        assert "sessions" in capsys.readouterr().out

    def test_json_is_parseable(self, tmp_path, capsys):
        from adder.measure.spend.trace import main

        assert main([self._root(tmp_path), "--json", "--no-cache"]) == 0
        d = json.loads(capsys.readouterr().out)
        assert d["turns"] == 3
        assert "concentration" in d

    def test_by_grouping_appears_in_json(self, tmp_path, capsys):
        from adder.measure.spend.trace import main

        assert main([self._root(tmp_path), "--json", "--by", "model", "--no-cache"]) == 0
        d = json.loads(capsys.readouterr().out)
        assert d["by_model"]

    def test_window_filters_the_total(self, tmp_path, capsys):
        from adder.measure.spend.trace import main

        root = self._root(tmp_path)
        assert main([root, "--json", "--since", "2026-08-20", "--no-cache"]) == 0
        assert json.loads(capsys.readouterr().out)["turns"] == 1

    def test_a_window_that_matches_nothing_exits_one(self, tmp_path, capsys):
        from adder.measure.spend.trace import main

        assert main([self._root(tmp_path), "--since", "2030-01-01", "--no-cache"]) == 1
        assert "No priced turns" in capsys.readouterr().out

    def test_strict_fails_on_an_unpriced_model(self, tmp_path, capsys):
        from adder.measure.spend.trace import main

        _write(tmp_path, [_rec("m1"), _rec("m9", model="gpt-9-turbo")], name="x.jsonl")
        assert main([str(tmp_path), "--strict", "--no-cache"]) == 1
        assert "lower bound" in capsys.readouterr().out

    def test_strict_passes_when_every_model_is_priced(self, tmp_path):
        from adder.measure.spend.trace import main

        assert main([self._root(tmp_path), "--strict", "--no-cache"]) == 0

    def test_day_grouping_is_chronological_not_ranked(self, tmp_path, capsys):
        """A date axis sorted by cost is a bar chart with the x-axis shuffled."""
        from adder.measure.spend.trace import main

        records = []
        # The cheapest day is in the middle, so cost order and date order differ.
        for i, (day, read) in enumerate([("01", 900_000), ("02", 1_000),
                                         ("03", 500_000)]):
            records.append(_rec(f"m{i}", ts=f"2026-08-{day}T10:00:00Z", read=read))
        _write(tmp_path, records, name="d.jsonl")
        assert main([str(tmp_path), "--by", "day", "--no-cache"]) == 0
        out = capsys.readouterr().out
        days = [ln.split()[0] for ln in out.splitlines()
                if ln.strip().startswith("2026-08-")]
        assert days == sorted(days)

    def test_synthetic_records_are_reported_as_placeholders(self, tmp_path, capsys):
        from adder.measure.spend.trace import main

        _write(tmp_path, [_rec("m1"), _rec("m2", model="<synthetic>")], name="x.jsonl")
        assert main([str(tmp_path), "--strict", "--no-cache"]) == 0
        assert "placeholder" in capsys.readouterr().out


class TestCompactionsUseTheOneDetector:
    """`Session.compactions` had a private rule the repo had already replaced.

    `carry.is_compaction` exists because "the context dropped by 40%" counts
    branch-resumption dips, which outnumber real auto-compactions 15 to 1 --
    122 context drops on this corpus, 7 of them real. `Session.compactions`
    kept the naive rule anyway, and `adder sessions` printed it as a per-session
    column: 13 events on this machine where the canonical detector finds 9, one
    of them nothing but the step from a parent turn into a subagent's context.

    The detector now lives in `core.trace`, because `Session` needs it and
    `core` may not import `measure`. `carry` and `compact` import it from here.
    """

    @staticmethod
    def _sess(turns):
        s = Session("s", "p")
        s.turns = turns
        return s

    @staticmethod
    def _turn(ctx, *, side=False, model="claude-opus-5"):
        return Turn("s", "p", model, uncached_in=0, cache_read=ctx, cache_write=0,
                    out=10, thinking=0, sidechain=side)

    def test_a_real_auto_compaction_is_counted(self):
        """Near the 1M ceiling, dropping to 5% -- the measured shape."""
        s = self._sess([self._turn(950_000), self._turn(50_000)])
        assert s.compactions() == 1

    def test_a_branch_resumption_dip_is_not(self):
        """60K -> 30K is a 50% drop and nowhere near any ceiling."""
        s = self._sess([self._turn(60_000), self._turn(30_000)])
        assert s.compactions() == 0

    def test_a_step_into_a_subagent_is_not(self):
        """A subagent's context is not this session's context."""
        s = self._sess([self._turn(900_000), self._turn(5_000, side=True)])
        assert s.compactions() == 0

    def test_it_is_the_same_function_carry_uses(self):
        from adder.measure.window.carry import is_compaction as carry_version

        assert carry_version is is_compaction

    def test_and_the_same_one_compact_uses(self):
        from adder.measure.window.compact import is_compaction_pair

        turns = [self._turn(950_000), self._turn(50_000)]
        assert is_compaction_pair(turns, 1)
        assert not is_compaction_pair([self._turn(60_000), self._turn(30_000)], 1)


class TestTheIrreducibleFloorIsMainChain:
    """A subagent's opening prompt is not this conversation's floor.

    `base_context` is documented as "system prompt, tools, CLAUDE.md.
    Irreducible" -- a property of the main context. Taken as `min()` over every
    turn, any session with delegation reported the subagent's much smaller
    prompt instead: a number no main-chain turn ever had. Four of 105 sessions
    here, halving the figure in each.

    It is load-bearing. `debt.decompose_read_cost` multiplies this by the turn
    count to get the irreducible baseline, and whatever is not irreducible
    becomes the "addressable pool" that every verbosity saving is scaled by, so
    understating the floor overstates the pool.
    """

    @staticmethod
    def _turn(ctx, *, side=False):
        return Turn("s", "p", "claude-opus-5", uncached_in=0, cache_read=ctx,
                    cache_write=0, out=10, thinking=0, sidechain=side)

    def _sess(self, turns):
        s = Session("s", "p")
        s.turns = turns
        return s

    def test_a_subagent_turn_does_not_set_the_floor(self):
        s = self._sess([self._turn(60_000), self._turn(5_000, side=True),
                        self._turn(90_000)])
        assert s.base_context == 60_000

    def test_the_floor_is_unchanged_without_delegation(self):
        s = self._sess([self._turn(60_000), self._turn(90_000)])
        assert s.base_context == 60_000

    def test_an_all_sidechain_session_falls_back_to_its_own_turns(self):
        s = self._sess([self._turn(5_000, side=True), self._turn(9_000, side=True)])
        assert s.base_context == 5_000

    def test_an_empty_session_is_zero(self):
        assert self._sess([]).base_context == 0

    def test_the_debt_baseline_uses_the_property(self):
        """Two modules had inlined the same `min()`; they now share one answer."""
        from adder.measure.spend.debt import decompose_read_cost

        s = self._sess([self._turn(60_000), self._turn(5_000, side=True),
                        self._turn(90_000)])
        _, base, _ = decompose_read_cost({"s": s})
        assert base > 0
