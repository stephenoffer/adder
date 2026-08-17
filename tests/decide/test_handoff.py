"""The brief budget: how much may cross a restart before the restart stops paying.

Two failures this guards. A budget that comes out positive for a session with
no future tells someone to throw away a working context for nothing. And a
budget reported as a *target* rather than a bound invites a 300,000-token
"brief", which is not a brief, it is the context with extra steps.
"""

from __future__ import annotations

import json

import pytest

from adder.core.trace import Session, Turn
from adder.decide import handoff
from adder.pricing.cost import Rates


class TestMaxHandoff:
    def test_a_long_horizon_buys_a_bigger_brief(self):
        short = handoff.max_handoff(context=400_000, remaining=20,
                                    model="claude-opus-5")
        long = handoff.max_handoff(context=400_000, remaining=400,
                                   model="claude-opus-5")
        assert long > short

    def test_it_never_exceeds_the_context(self):
        b = handoff.max_handoff(context=10_000, remaining=100_000,
                                model="claude-opus-5")
        assert b <= 10_000

    def test_no_horizon_means_no_restart(self):
        assert handoff.max_handoff(context=400_000, remaining=0,
                                   model="claude-opus-5") == 0

    def test_a_small_context_cannot_fund_an_opening(self):
        assert handoff.max_handoff(context=1_000, remaining=10,
                                   model="claude-opus-5") == 0

    def test_a_worse_cache_buys_a_bigger_brief(self):
        warm = handoff.max_handoff(context=400_000, remaining=200,
                                   model="claude-opus-5", read_mult=0.10)
        cold = handoff.max_handoff(context=400_000, remaining=200,
                                   model="claude-opus-5", read_mult=0.40)
        assert cold > warm

    def test_a_dearer_opening_shrinks_the_brief(self):
        from adder.measure.window.prefix import Opening

        cheap = Opening(floor_tokens=20_000, read_tokens=19_000,
                        write_tokens=1_000, openings=1, source="measured")
        dear = Opening(floor_tokens=200_000, read_tokens=0,
                       write_tokens=200_000, openings=1, source="measured")
        assert (handoff.max_handoff(context=400_000, remaining=200,
                                    model="claude-opus-5", opening=cheap)
                > handoff.max_handoff(context=400_000, remaining=200,
                                      model="claude-opus-5", opening=dear))


class TestBudget:
    def _budget(self, tokens, context):
        return handoff.Budget(tokens=tokens, context=context, remaining=200,
                              model="claude-opus-5", read_mult=0.1,
                              opening_floor=28_000, warm_share=0.7)

    def test_a_tiny_budget_is_not_viable(self):
        assert not self._budget(50, 400_000).viable

    def test_a_budget_near_the_whole_context_is_not_binding(self):
        b = self._budget(380_000, 400_000)
        assert b.viable
        assert not b.binding
        assert "not the constraint" in b.describe()

    def test_a_real_ceiling_is_binding(self):
        b = self._budget(20_000, 400_000)
        assert b.binding
        assert "stay under" in b.describe()

    def test_an_unviable_budget_says_do_not_restart(self):
        assert "no brief is worth writing" in self._budget(10, 400_000).describe()

    def test_an_empty_session_has_no_budget(self):
        assert handoff.budget(Session("s", "p"), remaining=100).tokens == 0

    def test_a_live_session_is_priced_off_its_own_opening(self, make_session):
        b = handoff.budget(make_session(80, base=100_000, growth=5_000),
                           remaining=300)
        assert b.context > 0
        assert b.opening_floor > 0


class TestItems:
    def _transcript(self, tmp_path, calls):
        recs = []
        for i, (tool, inp, result) in enumerate(calls):
            recs.append({"type": "assistant", "sessionId": "s",
                         "timestamp": f"2026-08-01T10:{i:02d}:00Z",
                         "message": {"id": f"m{i}", "model": "claude-opus-5",
                                     "usage": {"input_tokens": 1,
                                               "cache_read_input_tokens": 20_000,
                                               "cache_creation_input_tokens": 0,
                                               "output_tokens": 100},
                                     "content": [{"type": "tool_use", "id": f"u{i}",
                                                  "name": tool, "input": inp}]}})
            recs.append({"type": "user", "sessionId": "s",
                         "message": {"content": [{"type": "tool_result",
                                                  "tool_use_id": f"u{i}",
                                                  "content": result}]}})
        p = tmp_path / "s.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in recs))
        return p

    def test_edits_outrank_a_large_one_off_read(self, tmp_path):
        path = self._transcript(tmp_path, [
            ("Read", {"file_path": "/huge.py"}, "x" * 200_000),
            ("Edit", {"file_path": "/small.py"}, "ok"),
        ])
        items = handoff.items_from(path)
        assert items[0].kind == "edited"

    def test_a_command_run_once_is_history_not_state(self, tmp_path):
        path = self._transcript(tmp_path, [("Bash", {"command": "ls"}, "a b c")])
        assert handoff.items_from(path) == []
        assert handoff.items_from(path, keep_single_commands=True)

    def test_a_repeated_command_is_named(self, tmp_path):
        path = self._transcript(tmp_path, [("Bash", {"command": "pytest -q"}, "ok")] * 3)
        items = handoff.items_from(path)
        assert items[0].kind == "ran"
        assert items[0].calls == 3

    def test_reads_are_ranked_by_what_they_cost_to_re_establish(self, tmp_path):
        path = self._transcript(tmp_path, [
            ("Read", {"file_path": "/small.py"}, "x" * 400),
            ("Read", {"file_path": "/big.py"}, "x" * 40_000),
        ])
        reads = [i for i in handoff.items_from(path) if i.kind == "read"]
        assert reads[0].name == "/big.py"

    def test_tools_with_prose_inputs_are_never_named(self, tmp_path):
        # An unknown tool's input can be a question written for a human. Those
        # identities are hashes, and a hash in a brief is noise at best and a
        # leak at worst, so they do not appear at all.
        path = self._transcript(tmp_path, [
            ("AskUserQuestion", {"question": "which approach do you want?"},
             "answered")] * 3)
        assert handoff.items_from(path) == []

    def test_no_message_text_is_emitted(self, tmp_path):
        path = self._transcript(tmp_path, [
            ("Edit", {"file_path": "/a.py"}, "SECRET CONTENT HERE")] * 2)
        assert all("SECRET" not in i.name for i in handoff.items_from(path))


class TestMeasuredHandoffs:
    def test_openings_above_the_shared_floor(self, make_session):
        sessions = {f"s{i}": make_session(10, sid=f"s{i}", base=20_000 + 5_000 * i)
                    for i in range(5)}
        m = handoff.measured_handoffs(sessions)
        assert m.n == 5
        assert m.floor > 0
        assert m.p90() >= m.median() >= 0

    def test_no_sessions_is_survivable(self):
        m = handoff.measured_handoffs({})
        assert m.n == 0 and m.median() == 0


class TestOutput:
    def test_report_tells_a_doomed_restart_to_finish(self):
        b = handoff.Budget(tokens=0, context=400_000, remaining=1,
                           model="claude-opus-5", read_mult=0.1,
                           opening_floor=28_000, warm_share=0.7)
        assert "Finish the session" in handoff.report(b, [])

    def test_report_prices_writing_the_brief(self):
        b = handoff.Budget(tokens=5_000, context=400_000, remaining=300,
                           model="claude-opus-5", read_mult=0.1,
                           opening_floor=28_000, warm_share=0.7)
        assert "Writing the brief costs" in handoff.report(b, [])

    def test_hypothetical_context_mode_is_json(self, capsys):
        assert handoff.main(["--context", "500000", "--remaining", "300",
                             "--json"]) == 0
        doc = json.loads(capsys.readouterr().out)
        assert doc["budget_tokens"] > 0

    def test_no_session_is_a_nonzero_exit(self, capsys):
        assert handoff.main(["--cwd", "/nonexistent-for-adder-tests"]) == 1

    def test_no_session_still_emits_json(self, capsys):
        assert handoff.main(["--cwd", "/nonexistent-for-adder-tests",
                             "--json"]) == 1
        assert "error" in json.loads(capsys.readouterr().out)


class TestTheCarryRateComesFromTheProvider:
    """Whether to restart turns on what carrying context costs. That is a rate.

    `max_handoff` defaulted `read_mult` to Anthropic's 0.10x and took the write
    premium from `CACHE_WRITE_MULT`, on whatever model the session happened to
    be running. On a provider with no prompt cache a re-read costs the full
    input rate, so the default told those sessions carrying was ten times
    cheaper than it is -- and that is the side of the equation that argues
    against restarting at all.
    """

    def test_claude_is_unchanged(self):
        """Anthropic really does charge 0.10x, so the old default was right here."""
        r = Rates.for_model("claude-opus-5", ttl="1h")
        assert r.cache_read / r.inp == pytest.approx(0.10)

    def test_a_provider_with_no_cache_permits_a_bigger_brief(self):
        """Carrying is dearer there, so a restart pays sooner and can say more."""
        claude = handoff.max_handoff(context=400_000, remaining=200,
                                     model="claude-opus-5")
        uncached = handoff.max_handoff(context=400_000, remaining=200,
                                       model="cohere/command-a")
        assert uncached > claude

    def test_an_explicit_read_mult_still_wins(self):
        """A fitted multiplier from `carry` is a measurement; it overrides."""
        a = handoff.max_handoff(context=400_000, remaining=200,
                                model="claude-opus-5", read_mult=0.10)
        b = handoff.max_handoff(context=400_000, remaining=200,
                                model="claude-opus-5", read_mult=1.00)
        assert b > a

    def test_the_default_matches_an_explicit_claude_multiplier(self):
        assert handoff.max_handoff(context=400_000, remaining=200,
                                   model="claude-opus-5") == \
            handoff.max_handoff(context=400_000, remaining=200,
                                model="claude-opus-5", read_mult=0.10)


class TestTheBudgetReadsTheConversation:
    def test_a_trailing_subagent_turn_does_not_set_the_context(self, make_session):
        s = make_session(40, base=200_000, growth=1_000)
        s.turns.append(Turn("s", "p", "claude-haiku-4-5", uncached_in=0,
                            cache_read=3_000, cache_write=0, out=10, thinking=0,
                            sidechain=True))
        assert handoff.budget(s, remaining=200).context > 100_000
        assert handoff.budget(s, remaining=200).model == "claude-opus-5"
