"""Re-reads: content the context already held, and reads that recur every session.

The two mistakes this file guards are opposites. Counting a *refresh* as waste
tells someone their test runs are the problem; counting only exact-duplicate
calls misses the case where the same file is read at two offsets. Both are
wrong in a way that survives a smoke test, so the classification is asserted
directly rather than through a dollar figure.
"""

from __future__ import annotations

import json

import pytest

from adder.measure.window import reread


def records(pairs, sid="s", model="claude-opus-5"):
    """`pairs` is [(tool, input, result_text), ...] — one tool call per turn."""
    out = []
    for i, (tool, inp, result) in enumerate(pairs):
        out.append({
            "type": "assistant", "sessionId": sid,
            "timestamp": f"2026-08-01T10:{i:02d}:00Z",
            "message": {"id": f"m-{sid}-{i}", "model": model,
                        "usage": {"input_tokens": 1,
                                  "cache_read_input_tokens": 20_000 + 500 * i,
                                  "cache_creation_input_tokens": 0,
                                  "output_tokens": 200},
                        "content": [{"type": "tool_use", "id": f"u-{sid}-{i}",
                                     "name": tool, "input": inp}]}})
        out.append({
            "type": "user", "sessionId": sid,
            "timestamp": f"2026-08-01T10:{i:02d}:30Z",
            "message": {"content": [{"type": "tool_result",
                                     "tool_use_id": f"u-{sid}-{i}",
                                     "content": result}]}})
    return out


@pytest.fixture
def root(tmp_path):
    d = tmp_path / "projects" / "proj"
    d.mkdir(parents=True)
    return d.parent


def write(root, pairs, sid="s"):
    (root / "proj" / f"{sid}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records(pairs, sid=sid)))
    return root


BIG = "x" * 8_000          # ~2,000 tokens
OTHER = "y" * 8_000


class TestIdentity:
    def test_read_is_keyed_by_path(self):
        assert reread.identity("Read", {"file_path": "/a.py"}) == "Read:/a.py"

    def test_read_offsets_share_one_identity(self):
        a = reread.identity("Read", {"file_path": "/a.py", "offset": 1})
        b = reread.identity("Read", {"file_path": "/a.py", "offset": 500})
        assert a == b

    def test_bash_whitespace_is_normalised(self):
        a = reread.identity("Bash", {"command": "cat  a.py"})
        b = reread.identity("Bash", {"command": "cat a.py\n"})
        assert a == b

    def test_grep_keys_on_pattern_and_path(self):
        assert reread.identity("Grep", {"pattern": "x", "path": "/p"}) == "Grep:x|/p"

    def test_unknown_tools_are_hashed_never_quoted(self):
        # The input of a tool this module has no shape for can be prose written
        # for a human. Identities are printed and injected into contexts, so an
        # unknown tool is identified by a hash of its input and never by it.
        ident = reread.identity("Weird", {"question": "what should we do next?"})
        assert ident.startswith("Weird#")
        assert "what should we do" not in ident

    def test_the_same_unknown_call_hashes_the_same_way(self):
        a = reread.identity("Weird", {"a": 1})
        b = reread.identity("Weird", {"a": 1})
        assert a == b != reread.identity("Weird", {"a": 2})

    def test_non_dict_input_is_survivable(self):
        assert reread.identity("Read", None) == "Read"


class TestVolatile:
    @pytest.mark.parametrize("cmd", ["git status", "pytest -q", "ls -la"])
    def test_state_reporting_commands_are_volatile(self, cmd):
        assert reread.is_volatile(f"Bash:{cmd}")

    def test_a_file_read_is_never_volatile(self):
        assert not reread.is_volatile("Read:/a.py")

    def test_cat_is_not_volatile(self):
        assert not reread.is_volatile("Bash:cat a.py")


class TestDigest:
    def test_whitespace_does_not_change_the_answer(self):
        assert reread.digest("a  b\n") == reread.digest("a b")

    def test_different_content_differs(self):
        assert reread.digest("a") != reread.digest("b")

    def test_no_content_is_retained(self):
        d = reread.digest("secret token value")
        assert "secret" not in d and len(d) == 16


class TestShorten:
    def test_keeps_the_filename_end(self):
        out = reread.shorten("Read:/very/long/path/that/goes/on/and/on/thing.py", 30)
        assert out.endswith("thing.py")
        assert len(out) <= 30

    def test_short_identities_are_untouched(self):
        assert reread.shorten("Read:/a.py") == "Read:/a.py"


class TestClassification:
    def test_identical_result_is_redundant(self, root):
        write(root, [("Read", {"file_path": "/a.py"}, BIG)] * 2)
        rep = reread.scan(root)
        r = rep.repeats[("s", "Read:/a.py")]
        assert r.calls == 2
        assert len(r.redundant) == 1
        assert r.refreshes == []

    def test_changed_result_is_a_refresh_not_waste(self, root):
        write(root, [("Read", {"file_path": "/a.py"}, BIG),
                     ("Read", {"file_path": "/a.py"}, OTHER)])
        rep = reread.scan(root)
        r = rep.repeats[("s", "Read:/a.py")]
        assert r.redundant == []
        assert len(r.refreshes) == 1
        assert r.superseded_tokens > 0

    def test_a_third_identical_copy_counts_twice(self, root):
        write(root, [("Read", {"file_path": "/a.py"}, BIG)] * 3)
        r = reread.scan(root).repeats[("s", "Read:/a.py")]
        assert len(r.redundant) == 2

    def test_returning_to_an_earlier_version_is_still_redundant(self, root):
        write(root, [("Read", {"file_path": "/a.py"}, BIG),
                     ("Read", {"file_path": "/a.py"}, OTHER),
                     ("Read", {"file_path": "/a.py"}, BIG)])
        r = reread.scan(root).repeats[("s", "Read:/a.py")]
        assert len(r.redundant) == 1
        assert len(r.refreshes) == 1

    def test_one_call_is_not_a_repeat(self, root):
        write(root, [("Read", {"file_path": "/a.py"}, BIG)])
        assert reread.scan(root).with_repeats() == []

    def test_small_results_are_filtered_out(self, root):
        write(root, [("Read", {"file_path": "/a.py"}, "tiny")] * 2)
        assert reread.scan(root).with_repeats(min_tokens=400) == []

    def test_volatile_commands_are_excluded_by_default(self, root):
        write(root, [("Bash", {"command": "git status"}, BIG)] * 2)
        rep = reread.scan(root)
        assert rep.with_repeats() == []
        assert rep.with_repeats(include_volatile=True)

    def test_a_replayed_transcript_does_not_double_count(self, root, tmp_path):
        pairs = [("Read", {"file_path": "/a.py"}, BIG)] * 2
        write(root, pairs)
        (root / "proj" / "copy.jsonl").write_text(
            (root / "proj" / "s.jsonl").read_text())
        r = reread.scan(root).repeats[("s", "Read:/a.py")]
        assert r.calls == 2


class TestRecurring:
    def test_the_same_read_in_two_sessions(self, root):
        write(root, [("Bash", {"command": "cat a.py"}, BIG)], sid="s1")
        write(root, [("Bash", {"command": "cat a.py"}, BIG)], sid="s2")
        rc = reread.scan(root).recurring["Bash:cat a.py"]
        assert rc.n_sessions == 2
        assert rc.stable

    def test_a_changing_answer_is_not_stable(self, root):
        write(root, [("Bash", {"command": "cat a.py"}, BIG)], sid="s1")
        write(root, [("Bash", {"command": "cat a.py"}, OTHER)], sid="s2")
        assert not reread.scan(root).recurring["Bash:cat a.py"].stable

    def test_within_session_repeats_count_the_session_once(self, root):
        write(root, [("Read", {"file_path": "/a.py"}, BIG)] * 4, sid="s1")
        rc = reread.scan(root).recurring["Read:/a.py"]
        assert rc.n_sessions == 1

    def test_threshold_controls_candidacy(self, root):
        for sid in ("s1", "s2"):
            write(root, [("Bash", {"command": "cat a.py"}, BIG)], sid=sid)
        rep = reread.scan(root)
        assert rep.memory_candidates(min_sessions=2)
        assert not rep.memory_candidates(min_sessions=3)

    def test_unstable_candidates_are_excluded_unless_asked_for(self, root):
        write(root, [("Bash", {"command": "cat a.py"}, BIG)], sid="s1")
        write(root, [("Bash", {"command": "cat a.py"}, OTHER)], sid="s2")
        rep = reread.scan(root)
        assert not rep.memory_candidates(min_sessions=2)
        assert rep.memory_candidates(min_sessions=2, stable_only=False)


class TestPricing:
    def test_an_early_duplicate_costs_more_than_a_late_one(self, root, make_sessions):
        write(root, [("Read", {"file_path": "/a.py"}, BIG)] * 2)
        rep = reread.scan(root)
        r = rep.repeats[("s", "Read:/a.py")]
        carry = reread._carry({})
        early = reread.price_repeat(r, {"s": ("claude-opus-5", 400)}, carry)
        late = reread.price_repeat(r, {"s": ("claude-opus-5", 3)}, carry)
        assert early > late > 0

    def test_note_budget_is_nonnegative(self, root, make_sessions):
        for sid in ("s1", "s2"):
            write(root, [("Bash", {"command": "cat a.py"}, BIG)], sid=sid)
        rep = reread.scan(root)
        sessions = make_sessions(3, 40)
        rc = rep.memory_candidates(min_sessions=2)[0]
        shape = reread._session_shape(sessions)
        budget = reread.breakeven_note_tokens(rc, shape, reread._carry(sessions),
                                              sessions)
        assert budget >= 0

    def test_more_sessions_paying_to_relearn_buys_a_bigger_note(self, make_sessions):
        sessions = make_sessions(4, 60)
        shape = reread._session_shape(sessions)
        carry = reread._carry(sessions)
        small = reread.Recurring("Read:/a.py", "Read", sessions={"s0"}, tokens=2_000)
        large = reread.Recurring("Read:/a.py", "Read",
                                 sessions=set(sessions), tokens=8_000)
        assert (reread.breakeven_note_tokens(large, shape, carry, sessions)
                > reread.breakeven_note_tokens(small, shape, carry, sessions))


class TestOutput:
    def test_report_names_the_recoverable_kind_only(self, root, make_sessions):
        write(root, [("Read", {"file_path": "/a.py"}, BIG)] * 2)
        text = reread.report(reread.scan(root), make_sessions(2, 30))
        assert "Re-read" in text
        assert "Read:/a.py" in text

    def test_report_is_calm_when_there_is_nothing(self, root, make_sessions):
        write(root, [("Read", {"file_path": "/a.py"}, BIG)])
        assert "No content was admitted twice" in reread.report(
            reread.scan(root), make_sessions(2, 30))

    def test_json_is_one_document(self, root, capsys):
        write(root, [("Read", {"file_path": "/a.py"}, BIG)] * 2)
        assert reread.main([str(root), "--json"]) == 0
        doc = json.loads(capsys.readouterr().out)
        assert doc["admissions"] == 2
        assert doc["repeats"][0]["duplicate_calls"] == 1

    def test_text_run_is_clean(self, root, capsys):
        write(root, [("Read", {"file_path": "/a.py"}, BIG)] * 2)
        assert reread.main([str(root)]) == 0
        assert "tool results" in capsys.readouterr().out


class TestReadOnly:
    def test_the_transcript_tree_is_untouched(self, root):
        write(root, [("Read", {"file_path": "/a.py"}, BIG)] * 2)
        before = {p: p.stat().st_mtime_ns for p in root.rglob("*")}
        reread.scan(root)
        assert {p: p.stat().st_mtime_ns for p in root.rglob("*")} == before
