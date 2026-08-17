"""Tool attribution, and the block-level deduplication it depends on.

The bug this file was written after: deduplicating assistant records by
*message* id throws away every `tool_use` block after the first, because Claude
Code writes one record per content block. The tool results then come back
referencing ids the scan never saw, and 56% of all context growth is attributed
to a tool called `?`. Dedup is by block id for exactly this reason.
"""

from __future__ import annotations

import json

import pytest

from adder.core.trace import Session, Turn
from adder.measure.window.tools import LEVERS, ToolStat, carried_cost, report, scan

OPUS = "claude-opus-5"


def _assistant(mid, blocks, ts="2026-08-14T10:00:00Z"):
    return {"type": "assistant", "timestamp": ts, "sessionId": "s",
            "message": {"id": mid, "model": OPUS,
                        "usage": {"input_tokens": 1, "output_tokens": 10},
                        "content": blocks}}


def _use(use_id, name):
    return {"type": "tool_use", "id": use_id, "name": name, "input": {}}


def _result(use_id, text, *, error=False, ts="2026-08-14T10:00:01Z"):
    block = {"type": "tool_result", "tool_use_id": use_id, "content": text}
    if error:
        block["is_error"] = True
    return {"type": "user", "timestamp": ts, "sessionId": "s",
            "message": {"content": [block]}}


def _write(tmp_path, records, name="s.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in records))
    return p


class TestBlockDedup:
    def test_multi_tool_turns_are_split_across_records(self, tmp_path):
        """One message, three records, three distinct tool_use blocks."""
        _write(tmp_path, [
            _assistant("m1", [{"type": "text", "text": "hi"}]),
            _assistant("m1", [_use("u1", "Bash")]),
            _assistant("m1", [_use("u2", "Read")]),
            _result("u1", "x" * 400),
            _result("u2", "y" * 800),
        ])
        rep = scan(tmp_path)
        assert set(rep.by_tool) == {"Bash", "Read"}
        assert rep.by_tool["Bash"].calls == 1
        assert "?" not in rep.by_tool, "results were orphaned from their tool"

    def test_a_repeated_block_record_counts_once(self, tmp_path):
        _write(tmp_path, [_assistant("m1", [_use("u1", "Bash")]),
                          _assistant("m1", [_use("u1", "Bash")])])
        assert scan(tmp_path).by_tool["Bash"].calls == 1

    def test_a_replayed_transcript_does_not_double_count_results(self, tmp_path):
        records = [_assistant("m1", [_use("u1", "Bash")]), _result("u1", "x" * 4000)]
        _write(tmp_path, records, name="a.jsonl")
        _write(tmp_path, records, name="b.jsonl")
        rep = scan(tmp_path)
        assert rep.by_tool["Bash"].calls == 1
        assert rep.by_tool["Bash"].result_tokens == pytest.approx(1000, rel=0.01)

    def test_blocks_with_no_id_fall_back_to_message_dedup(self, tmp_path):
        block = {"type": "tool_use", "name": "Bash", "input": {}}
        _write(tmp_path, [_assistant("m1", [block]), _assistant("m1", [block])])
        assert scan(tmp_path).by_tool["Bash"].calls == 1


class TestAttribution:
    def test_result_size_lands_on_the_calling_tool(self, tmp_path):
        _write(tmp_path, [_assistant("m1", [_use("u1", "Bash")]),
                          _result("u1", "x" * 4000)])
        st = scan(tmp_path).by_tool["Bash"]
        assert st.result_tokens == pytest.approx(1000, rel=0.01)
        assert st.biggest == st.result_tokens

    def test_an_orphan_result_is_named_rather_than_dropped(self, tmp_path):
        _write(tmp_path, [_result("nope", "x" * 400)])
        assert "?" in scan(tmp_path).by_tool

    def test_errors_are_counted_per_tool(self, tmp_path):
        _write(tmp_path, [
            _assistant("m1", [_use("u1", "Bash")]), _result("u1", "boom", error=True),
            _assistant("m2", [_use("u2", "Bash")]), _result("u2", "fine"),
        ])
        st = scan(tmp_path).by_tool["Bash"]
        assert st.errors == 1
        assert st.error_rate == pytest.approx(0.5)

    def test_shares_of_growth_never_exceed_one(self, tmp_path):
        _write(tmp_path, [_assistant("m1", [_use("u1", "Bash")]),
                          _result("u1", "x" * 8000)])
        rep = scan(tmp_path)
        assert sum(rep.share_of_growth(t) for t in rep.by_tool.values()) <= 1.0

    def test_sessions_are_tracked(self, tmp_path):
        _write(tmp_path, [_assistant("m1", [_use("u1", "Bash")])])
        assert scan(tmp_path).by_tool["Bash"].sessions == {"s"}

    def test_empty_root(self, tmp_path):
        rep = scan(tmp_path)
        assert rep.calls == 0
        assert report(tmp_path).strip().startswith("No tool calls")


class TestWindow:
    def test_date_window_excludes_older_records(self, tmp_path):
        from datetime import date

        from adder.core.filters import Window

        _write(tmp_path, [
            _assistant("m1", [_use("u1", "Bash")], ts="2026-01-01T10:00:00Z"),
            _result("u1", "x" * 400, ts="2026-01-01T10:00:01Z"),
            _assistant("m2", [_use("u2", "Read")], ts="2026-08-14T10:00:00Z"),
            _result("u2", "y" * 400, ts="2026-08-14T10:00:01Z"),
        ])
        rep = scan(tmp_path, window=Window(since=date(2026, 6, 1)))
        assert set(rep.by_tool) == {"Read"}

    def test_an_inactive_window_keeps_everything(self, tmp_path):
        from adder.core.filters import Window

        _write(tmp_path, [_assistant("m1", [_use("u1", "Bash")])])
        assert scan(tmp_path, window=Window()).calls == 1


class TestGrowthDenominator:
    """The arithmetic error this module shipped with, pinned so it cannot return.

    Assistant output is the largest of the three growth sources. Leaving it out
    of the denominator inflated every tool's share -- and every dollar
    apportioned by that share -- roughly threefold, which made `adder doctor`
    rank a $1,000 finding as a $3,000 one.
    """

    def _root(self, write_jsonl):
        return write_jsonl([
            _assistant("m1", [_use("u1", "Bash")]),
            _result("u1", "x" * 4_000),                     # 1,000 tok of result
            {"type": "assistant", "sessionId": "s",
             "timestamp": "2026-08-14T10:01:00Z",
             "message": {"id": "m2", "model": OPUS,
                         "usage": {"input_tokens": 1, "output_tokens": 10},
                         "content": [{"type": "text", "text": "y" * 12_000}]}},
        ])

    def test_assistant_output_is_in_the_denominator(self, write_jsonl):
        rep = scan(self._root(write_jsonl))
        assert rep.assistant_tokens > 0
        assert rep.growth() == (rep.total_result_tokens + rep.user_tokens
                                + rep.assistant_tokens)

    def test_the_share_falls_once_output_is_counted(self, write_jsonl):
        rep = scan(self._root(write_jsonl))
        bash = rep.by_tool["Bash"]
        naive = bash.result_tokens / (rep.total_result_tokens + rep.user_tokens)
        assert rep.share_of_growth(bash) < naive

    def test_billed_output_overrides_the_estimate(self, write_jsonl):
        from adder.measure.window.tools import billed_output

        rep = scan(self._root(write_jsonl))
        sessions = self._sessions()
        assert rep.growth(billed_output(sessions)) != rep.growth()

    def test_sidechain_output_is_not_in_the_denominator(self, make_turn):
        """A subagent's tokens never entered the main context."""
        from adder.core.trace import Session
        from adder.measure.window.tools import billed_output

        s = Session("s", "p")
        s.turns = [make_turn(out=100), make_turn(out=900, sidechain=True)]
        assert billed_output({"s": s}) == 100

    def test_it_agrees_with_the_context_report(self, write_jsonl):
        """Two modules, one decomposition. They must not drift."""
        from adder.measure.window.context import scan as context_scan

        root = self._root(write_jsonl)
        tools_rep = scan(root)
        growth = context_scan(root)
        assert tools_rep.total_result_tokens == growth.tool_results
        assert tools_rep.user_tokens == growth.user_messages

    def _sessions(self):
        from adder.core.trace import Session, Turn

        s = Session("s", "p")
        for i in range(50):
            s.turns.append(Turn("s", "p", OPUS, uncached_in=0,
                                cache_read=20_000 + 1_000 * i, cache_write=0,
                                out=200, thinking=0, sidechain=False))
        return {"s": s}


class TestCarriedCost:
    def _sessions(self):
        s = Session("s", "p")
        for i in range(50):
            s.turns.append(Turn("s", "p", OPUS, uncached_in=0,
                                cache_read=20_000 + 1_000 * i, cache_write=0,
                                out=200, thinking=0, sidechain=False))
        return {"s": s}

    def test_cost_is_bounded_by_measured_spend(self, tmp_path):
        from adder.measure.spend.debt import decompose_read_cost

        _write(tmp_path, [_assistant("m1", [_use("u1", "Bash")]),
                          _result("u1", "x" * 40_000)])
        sessions = self._sessions()
        _, _, accumulated = decompose_read_cost(sessions)
        total = sum(carried_cost(scan(tmp_path), sessions).values())
        assert 0 < total <= accumulated + 1e-9

    def test_no_growth_means_no_cost_rather_than_a_guess(self, tmp_path):
        _write(tmp_path, [_assistant("m1", [_use("u1", "Bash")])])
        assert set(carried_cost(scan(tmp_path), self._sessions()).values()) == {0.0}


class TestStat:
    def test_rates_of_an_unused_tool_are_zero(self):
        st = ToolStat("X")
        assert st.error_rate == 0.0
        assert st.mean_result == 0.0
        assert st.p90_result() == 0.0

    def test_results_seen_differs_from_calls_when_one_went_unanswered(self, tmp_path):
        _write(tmp_path, [_assistant("m1", [_use("u1", "Bash")]),
                          _assistant("m2", [_use("u2", "Bash")]),
                          _result("u1", "x" * 400)])
        st = scan(tmp_path).by_tool["Bash"]
        assert st.calls == 2 and st.results_seen == 1


class TestReport:
    def test_names_the_worst_tool_and_its_lever(self, tmp_path):
        _write(tmp_path, [_assistant("m1", [_use("u1", "Bash")]),
                          _result("u1", "x" * 40_000)])
        text = report(tmp_path)
        assert "Bash" in text
        assert LEVERS["Bash"].split(";")[0] in text

    def test_flags_a_failing_tool(self, tmp_path):
        recs = []
        for i in range(30):
            recs.append(_assistant(f"m{i}", [_use(f"u{i}", "WebFetch")]))
            recs.append(_result(f"u{i}", "boom", error=i % 2 == 0))
        _write(tmp_path, recs)
        assert "failing more than 10%" in report(tmp_path)


class TestCli:
    def test_json_output(self, tmp_path, capsys):
        from adder.measure.window.tools import main

        _write(tmp_path, [_assistant("m1", [_use("u1", "Bash")]),
                          _result("u1", "x" * 4000)])
        assert main([str(tmp_path), "--json"]) == 0
        d = json.loads(capsys.readouterr().out)
        assert d["tools"][0]["name"] == "Bash"
        assert d["calls"] == 1

    def test_text_output(self, tmp_path, capsys):
        from adder.measure.window.tools import main

        _write(tmp_path, [_assistant("m1", [_use("u1", "Bash")]),
                          _result("u1", "x" * 4000)])
        assert main([str(tmp_path)]) == 0
        assert "context" in capsys.readouterr().out


class TestWindowOnRawRecords:
    """A filter that is accepted and ignored is worse than one that is rejected:
    the number looks like an answer. These pin which fields apply here."""

    def _two_sessions(self, tmp_path, write_jsonl):
        write_jsonl([
            _assistant("m1", [_use("u1", "Bash")]),
            _result("u1", "x" * 4000),
        ], name="a.jsonl")
        recs = [
            {"type": "assistant", "sessionId": "other",
             "timestamp": "2026-08-14T10:00:00Z",
             "message": {"id": "m2", "model": OPUS,
                         "usage": {"input_tokens": 1, "output_tokens": 10},
                         "content": [_use("u2", "Read")]}},
            {"type": "user", "sessionId": "other",
             "timestamp": "2026-08-14T10:00:01Z",
             "message": {"content": [{"type": "tool_result",
                                      "tool_use_id": "u2",
                                      "content": "y" * 4000}]}},
        ]
        write_jsonl(recs, name="b.jsonl")
        return tmp_path

    def test_session_filter_is_honoured(self, tmp_path, write_jsonl):
        from adder.core.filters import Window

        root = self._two_sessions(tmp_path, write_jsonl)
        rep = scan(root, window=Window(sessions=("other",)))
        assert set(rep.by_tool) == {"Read"}

    def test_project_filter_is_honoured(self, tmp_path, write_jsonl):
        from adder.core.filters import Window

        write_jsonl([_assistant("m1", [_use("u1", "Bash")]), _result("u1", "x" * 400)],
                    into=tmp_path / "-Users-me-alpha")
        write_jsonl([_assistant("m2", [_use("u2", "Read")]), _result("u2", "y" * 400)],
                    into=tmp_path / "-Users-me-beta")
        rep = scan(tmp_path, window=Window(projects=("beta",)))
        assert set(rep.by_tool) == {"Read"}

    def test_subagent_filter_is_honoured(self, tmp_path, write_jsonl):
        from adder.core.filters import Window

        main = _assistant("m1", [_use("u1", "Bash")])
        side = _assistant("m2", [_use("u2", "Read")])
        side["isSidechain"] = True
        write_jsonl([main, side, _result("u1", "x" * 400), _result("u2", "y" * 400)])
        rep = scan(tmp_path, window=Window(sidechain=True))
        assert "Bash" not in rep.by_tool

    def test_a_model_filter_is_reported_rather_than_silently_ignored(
            self, tmp_path, write_jsonl):
        from adder.core.filters import Window

        write_jsonl([_assistant("m1", [_use("u1", "Bash")]), _result("u1", "x" * 400)])
        w = Window(models=("claude-opus",))
        assert w.ignores_model is True
        assert "--model-filter is not applied" in report(tmp_path, window=w)


class TestTheGrowthDenominatorIsOnePopulation:
    """A subagent's tokens never entered the main context, so none of them count.

    `billed_output` already excluded subagent output from the assistant term of
    the denominator, and said why. The tool-result and user-message terms did
    not, so the three were measured over two different populations: on the
    author's corpus, 403,581 subagent result tokens and 160,057 subagent user
    tokens were counted as main-context growth against an output term that
    deliberately left the matching subagent output out.

    It moves real money. `WebSearch` had 199,811 result tokens of which only
    41,703 reached the main context; the rest were inside subagents, which is
    exactly what delegating them bought.
    """

    @staticmethod
    def _records(side_tokens=4_000, main_tokens=1_000):
        """One main-chain tool call and one inside a subagent."""
        def pair(uid, n, side):
            return [
                {"type": "assistant", "isSidechain": side, "sessionId": "s",
                 "message": {"id": f"m{uid}", "content": [
                     {"type": "tool_use", "id": uid, "name": "Bash",
                      "input": {"command": "ls"}}]}},
                {"type": "user", "isSidechain": side, "sessionId": "s",
                 "message": {"content": [
                     {"type": "tool_result", "tool_use_id": uid,
                      "content": "x" * (n * 4)}]}},
            ]
        return pair("u1", main_tokens, False) + pair("u2", side_tokens, True)

    def test_sidechain_results_are_tracked_separately(self, write_jsonl):
        root = write_jsonl(self._records())
        rep = scan(root)
        st = rep.by_tool["Bash"]
        assert st.sidechain_result_tokens > 0
        assert st.main_result_tokens == st.result_tokens - st.sidechain_result_tokens

    def test_the_denominator_excludes_them(self, write_jsonl):
        root = write_jsonl(self._records())
        rep = scan(root)
        assert rep.growth(0) == rep.by_tool["Bash"].main_result_tokens + (
            rep.user_tokens - rep.sidechain_user_tokens)

    def test_a_tool_used_only_in_subagents_is_apportioned_nothing(self, write_jsonl):
        root = write_jsonl(self._records())
        rep = scan(root)
        st = rep.by_tool["Bash"]
        # everything the subagent returned is excluded from the main-chain share
        assert rep.share_of_growth(st, 10_000) < st.result_tokens / 10_000

    def test_a_corpus_with_no_subagents_is_unchanged(self, write_jsonl):
        """The fix must be a no-op where there is nothing to exclude."""
        root = write_jsonl(self._records(side_tokens=0)[:2])
        rep = scan(root)
        assert rep.sidechain_result_tokens == 0
        assert rep.growth(500) == (rep.total_result_tokens + rep.user_tokens + 500)
