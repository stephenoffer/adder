"""Speculation metrics, pinned on the cases that would quietly invert them.

The two that matter most:

* a `tool_result` arrives as a `user` record, so a naive "did a human speak"
  test fires between every probe and makes steerability meaningless;
* `trace.iter_file` deduplicates by `message.id` because it prices turns, and
  doing that here would erase parallel fan-out -- the thing this report is for.
"""

from __future__ import annotations

import json

import pytest

from adder.measure.session import speculation as spec


def _assistant(turn, tools, *, session="s", sidechain=False):
    """One assistant record whose message fires `tools` in parallel."""
    return {
        "type": "assistant", "sessionId": session, "isSidechain": sidechain,
        "timestamp": f"2026-08-01T10:{turn:02d}:00Z",
        "message": {
            "id": f"m{turn}", "model": "claude-opus-5",
            "usage": {"input_tokens": 2, "cache_read_input_tokens": 20_000,
                      "cache_creation_input_tokens": 0, "output_tokens": 300},
            "content": [
                {"type": "tool_use", "id": f"u{turn}-{i}", "name": name,
                 "input": inp}
                for i, (name, inp) in enumerate(tools)
            ],
        },
    }


def _results(turn, tools, *, session="s", size=400):
    return {
        "type": "user", "sessionId": session,
        "timestamp": f"2026-08-01T10:{turn:02d}:30Z",
        "message": {"content": [
            {"type": "tool_result", "tool_use_id": f"u{turn}-{i}",
             "content": "x" * size}
            for i in range(len(tools))
        ]},
    }


def _human(turn, text="try the parser instead", session="s"):
    return {
        "type": "user", "sessionId": session,
        "timestamp": f"2026-08-01T10:{turn:02d}:45Z",
        "message": {"content": [{"type": "text", "text": text}]},
    }


class TestPhase:
    def test_reads_and_searches_are_exploration(self):
        assert spec.phase_of("Read", "a.py") == spec.EXPLORE
        assert spec.phase_of("Grep", "def foo") == spec.EXPLORE

    def test_edits_are_formulation(self):
        assert spec.phase_of("Edit", "a.py") == spec.FORMULATE
        assert spec.phase_of("Write", "b.py") == spec.FORMULATE

    def test_bash_is_classified_by_what_it_runs(self):
        assert spec.phase_of("Bash", "python3 -m pytest -q") == spec.VALIDATE
        assert spec.phase_of("Bash", "ruff check .") == spec.VALIDATE
        assert spec.phase_of("Bash", "ls -la adder/") == spec.EXPLORE
        assert spec.phase_of("Bash", "git status") == spec.EXPLORE

    def test_an_unknown_tool_is_exploration_not_a_crash(self):
        assert spec.phase_of("SomeNewTool", "x") == spec.EXPLORE


class TestTarget:
    def test_a_file_probe_is_identified_by_its_path(self):
        assert spec.target_of("Read", {"file_path": "/a/b.py"}) == "/a/b.py"

    def test_offsets_do_not_make_it_a_different_probe(self):
        a = spec.target_of("Read", {"file_path": "/a/b.py", "offset": 1})
        b = spec.target_of("Read", {"file_path": "/a/b.py", "offset": 900})
        assert a == b

    def test_two_different_commands_are_two_probes(self):
        a = spec.target_of("Bash", {"command": "pytest tests/a.py"})
        b = spec.target_of("Bash", {"command": "pytest tests/b.py"})
        assert a != b

    def test_whitespace_is_normalised(self):
        assert spec.target_of("Bash", {"command": "ls   -la\n"}) == "ls -la"

    def test_a_long_target_is_bounded(self):
        assert len(spec.target_of("Bash", {"command": "x" * 5000})) == 200

    def test_a_probe_with_no_recognisable_target(self):
        assert spec.target_of("Read", {}) == ""
        assert spec.target_of("Read", None) == ""


class TestScan:
    def test_parallel_probes_in_one_message_are_not_collapsed(self, write_jsonl):
        """Dedup by message.id is right for pricing and wrong for fan-out."""
        tools = [("Read", {"file_path": "a.py"}),
                 ("Read", {"file_path": "b.py"}),
                 ("Grep", {"pattern": "def"})]
        root = write_jsonl([_assistant(1, tools), _results(1, tools)])
        sc = spec.scan(root)
        assert sc.n == 3

    def test_result_sizes_are_attached_to_their_probe(self, write_jsonl):
        tools = [("Read", {"file_path": "a.py"})]
        root = write_jsonl([_assistant(1, tools), _results(1, tools, size=4_000)])
        sc = spec.scan(root)
        assert sc.probes[0].result_tokens > 500

    def test_a_probe_with_no_result_still_counts(self, write_jsonl):
        tools = [("Read", {"file_path": "a.py"})]
        root = write_jsonl([_assistant(1, tools)])
        sc = spec.scan(root)
        assert sc.n == 1
        assert sc.probes[0].result_tokens == 0

    def test_tool_results_are_not_mistaken_for_human_turns(self, write_jsonl):
        tools = [("Read", {"file_path": "a.py"})]
        root = write_jsonl([_assistant(i, tools) for i in range(1, 6)] +
                           [_results(i, tools) for i in range(1, 6)])
        sc = spec.scan(root)
        assert sc.steers == {} or sc.steers.get("s", []) == []

    def test_a_real_human_turn_is_a_steer(self, write_jsonl):
        tools = [("Read", {"file_path": "a.py"})]
        root = write_jsonl([_assistant(1, tools), _results(1, tools), _human(2)])
        sc = spec.scan(root)
        assert sc.steers["s"] == [1]

    def test_a_string_message_body_counts_as_human(self, write_jsonl):
        rec = {"type": "user", "sessionId": "s",
               "message": {"content": "do the thing"}}
        root = write_jsonl([_assistant(1, [("Read", {"file_path": "a.py"})]), rec])
        assert spec.scan(root).steers["s"] == [1]

    def test_malformed_lines_are_skipped_not_fatal(self, write_jsonl, tmp_path):
        d = tmp_path / "p"
        d.mkdir()
        (d / "s.jsonl").write_text('not json\n{"type":"assistant"}\n')
        assert spec.scan(d).n == 0

    def test_an_empty_root_scans_to_nothing(self, tmp_path):
        sc = spec.scan(tmp_path)
        assert sc.n == 0 and sc.files == 0


class TestRedundancy:
    def test_a_repeated_probe_is_counted_once_as_a_repeat(self, write_jsonl):
        tools = [("Read", {"file_path": "a.py"})]
        root = write_jsonl([r for i in (1, 2, 3)
                            for r in (_assistant(i, tools), _results(i, tools))])
        rep = spec.redundancy(spec.scan(root))
        assert rep.probes == 3
        assert rep.distinct == 1
        assert rep.repeats == 2
        assert rep.rate == pytest.approx(2 / 3)

    def test_distinct_probes_are_not_repeats(self, write_jsonl):
        recs = []
        for i in (1, 2, 3):
            tools = [("Read", {"file_path": f"{i}.py"})]
            recs += [_assistant(i, tools), _results(i, tools)]
        rep = spec.redundancy(spec.scan(write_jsonl(recs)))
        assert rep.repeats == 0
        assert rep.rate == 0.0

    def test_the_same_file_in_two_sessions_is_not_a_repeat(self, write_jsonl):
        """A second session has its own context and genuinely has to look."""
        tools = [("Read", {"file_path": "a.py"})]
        recs = [_assistant(1, tools, session="s1"), _results(1, tools, session="s1"),
                _assistant(1, tools, session="s2"), _results(1, tools, session="s2")]
        rep = spec.redundancy(spec.scan(write_jsonl(recs)))
        assert rep.repeats == 0

    def test_the_worst_repeats_are_ranked_by_tokens_not_count(self, write_jsonl):
        recs = []
        # Three cheap repeats of a.py, two expensive repeats of big.log.
        for i in (1, 2, 3):
            t = [("Read", {"file_path": "a.py"})]
            recs += [_assistant(i, t), _results(i, t, size=100)]
        for i in (4, 5):
            t = [("Read", {"file_path": "big.log"})]
            recs += [_assistant(i, t), _results(i, t, size=40_000)]
        rep = spec.redundancy(spec.scan(write_jsonl(recs)))
        assert "big.log" in rep.worst[0][0]

    def test_probes_with_no_target_are_not_counted_as_repeats(self, write_jsonl):
        tools = [("Read", {})]
        recs = [r for i in (1, 2) for r in (_assistant(i, tools), _results(i, tools))]
        assert spec.redundancy(spec.scan(write_jsonl(recs))).repeats == 0

    def test_an_empty_scan_does_not_divide_by_zero(self):
        rep = spec.redundancy(spec.Scan())
        assert rep.rate == 0.0 and rep.token_rate == 0.0


class TestMix:
    def test_the_mix_sums_to_one(self, write_jsonl):
        recs = []
        for i, t in enumerate([("Read", {"file_path": "a"}),
                               ("Edit", {"file_path": "a"}),
                               ("Bash", {"command": "pytest"})], start=1):
            recs += [_assistant(i, [t]), _results(i, [t])]
        mix = spec.phase_mix(spec.scan(write_jsonl(recs)))
        assert sum(mix.values()) == pytest.approx(1.0)
        assert mix[spec.EXPLORE] == pytest.approx(1 / 3)

    def test_an_empty_scan_has_no_mix(self):
        assert spec.phase_mix(spec.Scan()) == {}

    def test_a_sequential_session_scores_low_interleaving(self, write_jsonl):
        recs = []
        seq = ([("Read", {"file_path": "a"})] * 4 +
               [("Edit", {"file_path": "a"})] * 4)
        for i, t in enumerate(seq, start=1):
            recs += [_assistant(i, [t]), _results(i, [t])]
        assert spec.phase_interleaving(spec.scan(write_jsonl(recs))) < 0.2

    def test_an_interleaved_session_scores_high(self, write_jsonl):
        recs = []
        seq = [("Read", {"file_path": "a"}), ("Edit", {"file_path": "a"})] * 4
        for i, t in enumerate(seq, start=1):
            recs += [_assistant(i, [t]), _results(i, [t])]
        assert spec.phase_interleaving(spec.scan(write_jsonl(recs))) > 0.8


class TestFanOut:
    def test_probes_per_turn_and_the_widest_turn(self, write_jsonl):
        wide = [("Read", {"file_path": f"{i}.py"}) for i in range(6)]
        one = [("Read", {"file_path": "z.py"})]
        recs = [_assistant(1, wide), _results(1, wide),
                _assistant(2, one), _results(2, one)]
        fan = spec.fan_out(spec.scan(write_jsonl(recs)))
        assert fan["max_in_one_turn"] == 6
        assert fan["per_turn_median"] == pytest.approx(3.5)

    def test_subagent_share_is_reported(self, write_jsonl):
        t = [("Read", {"file_path": "a.py"})]
        recs = [_assistant(1, t, sidechain=True), _results(1, t),
                _assistant(2, t), _results(2, t)]
        assert spec.fan_out(spec.scan(write_jsonl(recs)))["sidechain_share"] == 0.5

    def test_fan_out_on_nothing(self):
        fan = spec.fan_out(spec.Scan())
        assert fan["max_in_one_turn"] == 0.0
        assert fan["sessions"] == 0.0


class TestSteerability:
    def test_a_steer_that_collapses_the_search_is_measured(self, write_jsonl):
        recs = []
        turn = 1
        for _ in range(6):
            # Busy before: four probes a turn for three turns, then a steer,
            # then one probe a turn.
            for _ in range(3):
                wide = [("Read", {"file_path": f"{turn}-{k}.py"}) for k in range(4)]
                recs += [_assistant(turn, wide), _results(turn, wide)]
                turn += 1
            recs.append(_human(turn))
            for _ in range(3):
                one = [("Read", {"file_path": f"{turn}.py"})]
                recs += [_assistant(turn, one), _results(turn, one)]
                turn += 1
        st = spec.steerability(spec.scan(write_jsonl(recs)))
        assert st.measured
        assert st.before > st.after
        assert st.ci[0] > 0            # the interval excludes "no change"

    def test_too_few_steers_is_reported_as_unmeasured(self, write_jsonl):
        t = [("Read", {"file_path": "a.py"})]
        recs = [_assistant(1, t), _results(1, t), _human(2)]
        assert not spec.steerability(spec.scan(write_jsonl(recs))).measured

    def test_no_steers_at_all(self):
        st = spec.steerability(spec.Scan())
        assert st.n == 0 and st.reduction == 0.0


class TestReport:
    def test_it_renders_against_a_real_fixture(self, write_jsonl, isolated_home):
        recs = []
        for i in range(1, 8):
            t = [("Read", {"file_path": "a.py"}), ("Bash", {"command": "pytest -q"})]
            recs += [_assistant(i, t), _results(i, t)]
        root = write_jsonl(recs, into=None)
        text = spec.report(root)
        assert "agentic speculation" in text
        assert "redundancy" in text
        assert "OBSERVATIONAL" in text

    def test_an_empty_root_says_so(self, tmp_path):
        assert "No tool calls" in spec.report(tmp_path)

    def test_json_is_finite_and_complete(self, write_jsonl, isolated_home):
        t = [("Read", {"file_path": "a.py"})]
        root = write_jsonl([_assistant(1, t), _results(1, t)])
        payload = spec.to_json(root)
        text = json.dumps(payload)
        assert "NaN" not in text and "Infinity" not in text
        assert payload["steerability"]["observational"] is True
        assert set(payload) >= {"probes", "scale", "heterogeneity", "redundancy"}

    def test_the_cli_runs_and_prints(self, write_jsonl, capsys, isolated_home):
        t = [("Read", {"file_path": "a.py"})]
        root = write_jsonl([_assistant(1, t), _results(1, t)])
        assert spec.main([str(root)]) == 0
        assert capsys.readouterr().out.strip()

    def test_the_cli_json_parses(self, write_jsonl, capsys, isolated_home):
        t = [("Read", {"file_path": "a.py"})]
        root = write_jsonl([_assistant(1, t), _results(1, t)])
        assert spec.main([str(root), "--json"]) == 0
        json.loads(capsys.readouterr().out)


class TestCostAttribution:
    def test_redundancy_cost_is_zero_without_sessions(self):
        assert spec.redundancy_cost(spec.Redundancy(repeat_tokens=100), None) == 0.0

    def test_redundancy_cost_is_bounded_by_the_measured_pool(self, write_jsonl,
                                                             isolated_home):
        from adder.core.trace import load_sessions
        from adder.measure.spend.debt import decompose_read_cost

        recs = []
        for i in range(1, 10):
            t = [("Read", {"file_path": "a.py"})]
            recs += [_assistant(i, t), _results(i, t, size=8_000)]
        root = write_jsonl(recs)
        sessions = load_sessions(root)
        rep = spec.redundancy(spec.scan(root))
        _, _, accumulated = decompose_read_cost(sessions)
        assert 0.0 <= spec.redundancy_cost(rep, sessions) <= accumulated + 1e-9


class TestSubagentProbes:
    def test_a_subagent_repeat_still_counts_as_a_repeat(self, write_jsonl):
        """A subagent shares the session id, so its re-reads are re-reads."""
        tools = [("Read", {"file_path": "a.py"})]
        recs = [_assistant(1, tools), _results(1, tools),
                _assistant(2, tools, sidechain=True), _results(2, tools)]
        rep = spec.redundancy(spec.scan(write_jsonl(recs)))
        assert rep.repeats == 1

    def test_phase_mix_covers_every_probe(self, write_jsonl):
        recs = []
        for i, t in enumerate([("Read", {"file_path": "a"}),
                               ("SomeFutureTool", {"query": "x"})], start=1):
            recs += [_assistant(i, [t]), _results(i, [t])]
        sc = spec.scan(write_jsonl(recs))
        assert sum(1 for p in sc.probes if p.phase in spec.PHASES) == sc.n


class TestATurnIsAMessageNotARecord:
    """The 1.78x inflation, reproduced in the denominator of this report.

    Claude Code writes one record per content block. `scan` counted a turn per
    *record*, so `assistant_turns` came to 50,145 where there were 28,784 real
    turns -- 1.74x, almost exactly the figure `trace.iter_file` exists to undo.
    `fan_out` divides probes by that number, so probes-per-turn read 0.582 when
    it is 1.033.

    The ordinal was per-file too, so two transcripts sharing a session id
    collided in `Counter((session, turn_index))` and `max_in_one_turn` reported
    103 parallel probes that were never in the same turn. It is 12.

    Probes themselves are still never deduplicated -- two tool_use blocks in one
    message are two parallel probes, which is the thing being measured.
    """

    @staticmethod
    def _records():
        """One message, three records, three tool calls -- one turn."""
        def rec(block):
            return {"type": "assistant", "sessionId": "s",
                    "timestamp": "2026-08-10T12:00:00Z",
                    "message": {"id": "msg_1", "content": [block]}}
        return [
            rec({"type": "thinking", "thinking": "..."}),
            rec({"type": "tool_use", "id": "u1", "name": "Read",
                 "input": {"file_path": "a.py"}}),
            rec({"type": "tool_use", "id": "u2", "name": "Read",
                 "input": {"file_path": "b.py"}}),
            rec({"type": "tool_use", "id": "u3", "name": "Grep",
                 "input": {"pattern": "x"}}),
        ]

    def test_one_message_is_one_turn(self, write_jsonl):
        sc = spec.scan(write_jsonl(self._records()))
        assert sum(sc.assistant_turns.values()) == 1

    def test_but_every_probe_is_still_kept(self, write_jsonl):
        sc = spec.scan(write_jsonl(self._records()))
        assert sc.n == 3

    def test_they_share_one_turn_ordinal(self, write_jsonl):
        sc = spec.scan(write_jsonl(self._records()))
        assert len({p.turn_index for p in sc.probes}) == 1

    def test_so_fan_out_sees_them_as_parallel(self, write_jsonl):
        sc = spec.scan(write_jsonl(self._records()))
        assert spec.fan_out(sc)["max_in_one_turn"] == 3.0

    def test_and_probes_per_turn_is_not_diluted(self, write_jsonl):
        sc = spec.scan(write_jsonl(self._records()))
        assert spec.fan_out(sc)["per_turn_median"] == pytest.approx(3.0)

    def test_two_messages_are_two_turns(self, write_jsonl):
        recs = self._records()
        second = [{**r, "message": {**r["message"], "id": "msg_2"}}
                  for r in recs[1:2]]
        sc = spec.scan(write_jsonl(recs + second))
        assert sum(sc.assistant_turns.values()) == 2


class TestAResumedSessionIsNotMoreProbes:
    """A resumed session writes a NEW transcript that restates earlier turns.

    Probes are deliberately not deduplicated by message id -- two content
    blocks in one message are two probes fired in parallel, and collapsing
    them erases the fan-out this report exists to measure. But a *block id* is
    unique per probe, so deduplicating on that keeps the fan-out and drops the
    replay. Without it a resumed session doubled the probe count, the
    probes-per-turn, and the repeat rate.
    """

    def _pair(self, uid, mid="m1"):
        use = {"type": "assistant", "sessionId": "s",
               "timestamp": "2026-08-01T10:00:00Z",
               "message": {"id": mid, "model": "claude-opus-5", "content": [
                   {"type": "tool_use", "id": uid, "name": "Read",
                    "input": {"file_path": "/a.py"}}]}}
        res = {"type": "user", "sessionId": "s",
               "timestamp": "2026-08-01T10:01:00Z",
               "message": {"content": [
                   {"type": "tool_result", "tool_use_id": uid, "content": "x" * 400}]}}
        return [use, res]

    def test_a_replayed_probe_is_counted_once(self, write_jsonl):
        from adder.measure.session.speculation import scan

        rows = self._pair("t1")
        d = write_jsonl(rows, name="a.jsonl")
        write_jsonl(rows, name="b.jsonl", into=d)      # the resumed session
        sc = scan(d)
        assert sc.n == 1
        assert sc.assistant_turns["s"] == 1

    def test_parallel_probes_in_one_message_are_still_two(self, write_jsonl):
        from adder.measure.session.speculation import scan

        rec = {"type": "assistant", "sessionId": "s",
               "timestamp": "2026-08-01T10:00:00Z",
               "message": {"id": "m1", "model": "claude-opus-5", "content": [
                   {"type": "tool_use", "id": "t1", "name": "Read",
                    "input": {"file_path": "/a.py"}},
                   {"type": "tool_use", "id": "t2", "name": "Read",
                    "input": {"file_path": "/b.py"}}]}}
        sc = scan(write_jsonl([rec]))
        assert sc.n == 2 and sc.assistant_turns["s"] == 1

    def test_a_result_in_a_later_file_still_sizes_its_probe(self, write_jsonl):
        from adder.measure.session.speculation import scan

        use, res = self._pair("t1")
        d = write_jsonl([use], name="a.jsonl")
        write_jsonl([res], name="b.jsonl", into=d)
        assert scan(d).result_tokens == 100
