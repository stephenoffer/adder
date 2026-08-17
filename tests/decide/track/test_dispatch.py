"""Recovering delegations from transcripts, and the rules for counting them.

The point of this module is that the outcome log is empty on every machine
because filling it required a manual step. So the tests that matter are the
ones about *not* over-claiming: an unresolved dispatch is not a success, an
untiered dispatch is not evidence about any tier, and importing twice does not
double the evidence behind a gate.
"""

from __future__ import annotations

import json

import pytest

from adder.decide.track.dispatch import AGENT_TIERS, Dispatch, scan, tier_for_model, to_outcomes

OPUS, HAIKU, SONNET = "claude-opus-5", "claude-haiku-4-5", "claude-sonnet-5"


def _assistant(mid, blocks, ts="2026-08-01T10:00:00Z", session="s"):
    return {"type": "assistant", "sessionId": session, "timestamp": ts,
            "message": {"id": mid, "model": OPUS,
                        "usage": {"input_tokens": 1, "output_tokens": 10},
                        "content": blocks}}


def _dispatch_block(use_id, agent_type, *, model=None, description="do a thing"):
    inp = {"subagent_type": agent_type, "description": description,
           "prompt": "..."}
    if model:
        inp["model"] = model
    return {"type": "tool_use", "id": use_id, "name": "Agent", "input": inp}


def _result(use_id, text="done", *, error=False, ts="2026-08-01T10:05:00Z",
            session="s"):
    block = {"type": "tool_result", "tool_use_id": use_id, "content": text}
    if error:
        block["is_error"] = True
    return {"type": "user", "sessionId": session, "timestamp": ts,
            "message": {"content": [block]}}


class TestTierInference:
    def test_repo_agents_map_to_their_tier(self):
        for name, tier in AGENT_TIERS.items():
            d = Dispatch("s", "p", "u", name)
            assert d.tier == tier

    def test_agent_names_are_matched_case_insensitively(self):
        assert Dispatch("s", "p", "u", "Route-T1").tier == "T1"

    def test_an_unknown_agent_falls_back_to_the_model(self):
        assert Dispatch("s", "p", "u", "custom", model=HAIKU).tier == "T0"

    def test_an_older_generation_lands_on_the_same_rung(self):
        """A tier is a cost tier; opus-4-8 is $5/$25 exactly like opus-5."""
        assert tier_for_model("claude-opus-4-8") == tier_for_model(OPUS)

    def test_sonnet_generations_share_a_rung(self):
        assert tier_for_model("claude-sonnet-4-6") == tier_for_model(SONNET)

    def test_an_unknown_model_yields_no_tier(self):
        assert tier_for_model("gpt-9-turbo") == ""
        assert tier_for_model("") == ""

    def test_an_unplaceable_dispatch_has_no_tier(self):
        assert Dispatch("s", "p", "u", "mystery").tier == ""


class TestScan:
    def test_a_dispatch_and_its_result_are_paired(self, write_jsonl):
        root = write_jsonl([
            _assistant("m1", [_dispatch_block("u1", "route-t1")]),
            _result("u1", "found it"),
        ])
        found = scan(root)
        assert len(found.dispatches) == 1
        d = found.dispatches[0]
        assert d.resolved and not d.escalated and d.tier == "T1"

    def test_two_dispatches_in_one_turn_are_both_seen(self, write_jsonl):
        """One record per content block: collapsing by message id loses one."""
        root = write_jsonl([
            _assistant("m1", [_dispatch_block("u1", "route-t0")]),
            _assistant("m1", [_dispatch_block("u2", "route-t1")]),
            _result("u1"), _result("u2"),
        ])
        assert len(scan(root).usable) == 2

    def test_an_error_result_is_an_escalation(self, write_jsonl):
        root = write_jsonl([
            _assistant("m1", [_dispatch_block("u1", "route-t0")]),
            _result("u1", "boom", error=True),
        ])
        d = scan(root).dispatches[0]
        assert d.escalated and d.error
        assert d.reason == "tool error"

    def test_the_escalate_marker_is_an_escalation(self, write_jsonl):
        root = write_jsonl([
            _assistant("m1", [_dispatch_block("u1", "route-t0")]),
            _result("u1", "ESCALATE: this needs multi-file reasoning"),
        ])
        d = scan(root).dispatches[0]
        assert d.escalated
        assert "multi-file" in d.reason

    def test_prose_about_escalating_is_not_an_escalation(self, write_jsonl):
        """The marker is a protocol, not a word. Matching the word would make
        every subagent that explains the escalation rule a failure."""
        root = write_jsonl([
            _assistant("m1", [_dispatch_block("u1", "route-t0")]),
            _result("u1", "I considered whether to ESCALATE but finished it."),
        ])
        assert not scan(root).dispatches[0].escalated

    def test_an_unresolved_dispatch_is_not_counted(self, write_jsonl):
        root = write_jsonl([_assistant("m1", [_dispatch_block("u1", "route-t0")])])
        found = scan(root)
        assert found.unresolved == 1
        assert found.usable == []

    def test_an_untiered_dispatch_is_counted_but_not_usable(self, write_jsonl):
        root = write_jsonl([
            _assistant("m1", [_dispatch_block("u1", "some-custom-agent")]),
            _result("u1"),
        ])
        found = scan(root)
        assert found.untiered == 1
        assert found.usable == []

    def test_a_replayed_transcript_does_not_duplicate(self, write_jsonl, tmp_path):
        records = [_assistant("m1", [_dispatch_block("u1", "route-t1")]),
                   _result("u1")]
        write_jsonl(records, name="a.jsonl")
        write_jsonl(records, name="b.jsonl")
        assert len(scan(tmp_path).dispatches) == 1

    def test_by_tier_and_by_agent(self, write_jsonl):
        root = write_jsonl([
            _assistant("m1", [_dispatch_block("u1", "route-t0")]),
            _result("u1", "ESCALATE: too hard"),
            _assistant("m2", [_dispatch_block("u2", "route-t0")]),
            _result("u2", "fine"),
        ])
        found = scan(root)
        assert found.by_tier() == {"T0": (2, 1)}
        assert found.by_agent() == {"route-t0": 2}

    def test_empty_root(self, tmp_path):
        found = scan(tmp_path)
        assert found.dispatches == []
        assert found.escalations == 0

    def test_the_old_task_tool_name_still_counts(self, write_jsonl):
        block = _dispatch_block("u1", "route-t1")
        block["name"] = "Task"
        root = write_jsonl([_assistant("m1", [block]), _result("u1")])
        assert len(scan(root).usable) == 1


class TestToOutcomes:
    def _root(self, write_jsonl):
        return write_jsonl([
            _assistant("m1", [_dispatch_block("u1", "route-t1")]),
            _result("u1", "done"),
            _assistant("m2", [_dispatch_block("u2", "route-t0")]),
            _result("u2", "ESCALATE: nope"),
        ])

    def test_rows_carry_tier_escalation_and_provenance(self, write_jsonl):
        rows = to_outcomes(scan(self._root(write_jsonl)))
        assert {r.tier for r in rows} == {"T0", "T1"}
        assert sum(r.escalated for r in rows) == 1
        assert all(r.source == "transcript" for r in rows)

    def test_task_hash_is_stable(self, write_jsonl):
        root = self._root(write_jsonl)
        a = {r.task_hash for r in to_outcomes(scan(root))}
        b = {r.task_hash for r in to_outcomes(scan(root))}
        assert a == b and len(a) == 2

    def test_known_hashes_are_skipped(self, write_jsonl):
        root = self._root(write_jsonl)
        first = to_outcomes(scan(root))
        again = to_outcomes(scan(root),
                            known_hashes={r.task_hash for r in first})
        assert again == []

    def test_timestamps_come_from_the_transcript(self, write_jsonl):
        rows = to_outcomes(scan(self._root(write_jsonl)))
        assert all(r.ts > 0 for r in rows)


class TestImportCommand:
    def _root(self, write_jsonl):
        return write_jsonl([
            _assistant("m1", [_dispatch_block("u1", "route-t1")]),
            _result("u1", "done"),
        ])

    def test_dry_run_writes_nothing(self, write_jsonl, tmp_path, capsys):
        from adder.decide.track.outcomes import load, main

        log = tmp_path / "out.jsonl"
        root = self._root(write_jsonl)
        assert main(["import", str(root), "--log", str(log)]) == 0
        assert "Dry run" in capsys.readouterr().out
        assert load(log) == []

    def test_write_appends(self, write_jsonl, tmp_path, capsys):
        from adder.decide.track.outcomes import load, main

        log = tmp_path / "out.jsonl"
        root = self._root(write_jsonl)
        assert main(["import", str(root), "--log", str(log), "--write"]) == 0
        rows = load(log)
        assert len(rows) == 1
        assert rows[0].tier == "T1"
        assert rows[0].source == "transcript"

    def test_importing_twice_is_a_no_op(self, write_jsonl, tmp_path, capsys):
        from adder.decide.track.outcomes import load, main

        log = tmp_path / "out.jsonl"
        root = self._root(write_jsonl)
        main(["import", str(root), "--log", str(log), "--write"])
        capsys.readouterr()
        main(["import", str(root), "--log", str(log), "--write"])
        assert "Nothing new" in capsys.readouterr().out
        assert len(load(log)) == 1

    def test_json_output(self, write_jsonl, tmp_path, capsys):
        from adder.decide.track.outcomes import main

        root = self._root(write_jsonl)
        assert main(["import", str(root), "--log", str(tmp_path / "o.jsonl"),
                     "--json"]) == 0
        d = json.loads(capsys.readouterr().out)
        assert d["usable"] == 1
        assert d["written"] is False

    def test_imported_rows_move_the_estimate(self, write_jsonl, tmp_path):
        from adder.decide.track.outcomes import evidence, main

        log = tmp_path / "out.jsonl"
        before = evidence("T1", None, log)
        assert before.scope == "prior"
        main(["import", str(self._root(write_jsonl)), "--log", str(log), "--write"])
        after = evidence("T1", None, log)
        assert after.scope != "prior"
        assert after.p_fail < before.p_fail


class TestSourceField:
    def test_old_rows_without_a_source_still_load(self, tmp_path):
        from adder.decide.track.outcomes import load

        log = tmp_path / "out.jsonl"
        log.write_text(json.dumps({"tier": "T1", "model": "m", "project": "p",
                                   "escalated": False, "ts": 1.0}) + "\n")
        rows = load(log)
        assert len(rows) == 1
        assert rows[0].source == "recorded"

    @pytest.mark.parametrize("source", ["recorded", "transcript"])
    def test_both_sources_count_toward_evidence(self, tmp_path, source):
        """They are blind to the same thing, so they are weighed the same."""
        from adder.decide.track.outcomes import Outcome, evidence, record

        log = tmp_path / "out.jsonl"
        for _ in range(6):
            record(Outcome(tier="T1", model="m", project="p", escalated=False,
                           source=source), log)
        assert evidence("T1", None, log).n == 6
