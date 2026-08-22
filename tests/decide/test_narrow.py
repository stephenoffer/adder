"""Substitution instead of refusal, and the four rules that keep it safe.

This is the only path in the repo that emits `permissionDecision: allow`, so it
is the only one that can suppress a prompt the user would otherwise have seen.
Every rule in `narrow.py` exists because of that, and each gets an assertion:

* never widen a call the caller already bounded;
* never narrow below the point where the model just asks again;
* never invent a key the tool's schema would reject;
* never rewrite a tool whose bounded form is a different question (`Grep` to
  filenames) or whose text this module did not write (`Bash`).

Plus the two that live in the guard: it is off by default, and it is only ever
reachable where the guard was going to refuse outright — so turning it on
relaxes a denial and can never permit something the guard was silent about.
"""
from __future__ import annotations

import pytest

from adder.decide.narrow import BOUNDS, MIN_LINES, bounded_at, describe, narrow


class TestWhatMayBeRewritten:
    def test_read_gains_a_limit(self):
        got = narrow("Read", {"file_path": "/a/b.py"}, lines=300)
        assert got == {"file_path": "/a/b.py", "limit": 300}

    def test_grep_gains_a_head_limit(self):
        got = narrow("Grep", {"pattern": "def ", "output_mode": "content"}, lines=300)
        assert got["head_limit"] == 300
        assert got["pattern"] == "def "
        assert got["output_mode"] == "content", "the kind of answer must not change"

    def test_bash_is_never_rewritten(self):
        # Appending `| head` can change the exit status, cut a && chain, or
        # truncate input to a command whose output was never the point.
        assert narrow("Bash", {"command": "make test && ./deploy.sh"}, lines=300) is None

    def test_glob_and_webfetch_have_nothing_to_bound(self):
        assert narrow("Glob", {"pattern": "**/*.py"}, lines=300) is None
        assert narrow("WebFetch", {"url": "https://x.test/a"}, lines=300) is None

    def test_task_is_never_rewritten(self):
        # A subagent's brief is negotiated in words, not by truncating its input.
        assert narrow("Task", {"prompt": "audit the scheduler"}, lines=300) is None

    def test_grep_is_not_switched_to_filenames(self):
        # The cheapest bounded form of a content grep is a filename list, and it
        # answers a different question. A truncation the model can see the edge
        # of is recoverable; a substituted question is not.
        got = narrow("Grep", {"pattern": "x", "output_mode": "content"}, lines=300)
        assert got["output_mode"] == "content"


class TestNeverWiden:
    def test_a_tighter_caller_bound_is_left_alone(self):
        assert narrow("Read", {"file_path": "/a.py", "limit": 50}, lines=300) is None

    def test_an_equal_bound_is_left_alone(self):
        assert narrow("Read", {"file_path": "/a.py", "limit": 300}, lines=300) is None

    def test_a_looser_caller_bound_is_tightened(self):
        got = narrow("Read", {"file_path": "/a.py", "limit": 5_000}, lines=300)
        assert got is not None and got["limit"] == 300

    def test_offset_survives_the_rewrite(self):
        got = narrow("Read", {"file_path": "/a.py", "offset": 900}, lines=300)
        assert got == {"file_path": "/a.py", "offset": 900, "limit": 300}


class TestNeverUseless:
    def test_a_budget_too_small_to_be_worth_it_declines(self):
        assert narrow("Read", {"file_path": "/a.py"}, lines=MIN_LINES - 1) is None

    def test_at_the_floor_it_proceeds(self):
        assert narrow("Read", {"file_path": "/a.py"}, lines=MIN_LINES) is not None

    def test_zero_and_negative_budgets_decline(self):
        assert narrow("Read", {"file_path": "/a.py"}, lines=0) is None
        assert narrow("Read", {"file_path": "/a.py"}, lines=-10) is None


class TestSchemaSafety:
    def test_only_the_bounding_key_is_added(self):
        original = {"file_path": "/a.py", "offset": 3}
        got = narrow("Read", original, lines=300)
        assert set(got) - set(original) == {"limit"}

    def test_the_original_is_not_mutated(self):
        original = {"file_path": "/a.py"}
        narrow("Read", original, lines=300)
        assert original == {"file_path": "/a.py"}

    def test_an_empty_or_wrong_shaped_input_declines(self):
        assert narrow("Read", {}, lines=300) is None
        assert narrow("Read", None, lines=300) is None

    def test_the_bound_key_matches_the_table(self):
        # A drifted key name fails the harness's schema check and the hook
        # silently does nothing, which is the worst failure mode available.
        assert BOUNDS["Read"] == "limit"
        assert BOUNDS["Grep"] == "head_limit"


class TestBoundedAt:
    def test_reads_an_existing_bound(self):
        assert bounded_at("Read", {"file_path": "/a.py", "limit": 42}) == 42

    def test_unbounded_is_none_not_zero(self):
        assert bounded_at("Read", {"file_path": "/a.py"}) is None

    def test_a_nonsense_bound_is_none(self):
        assert bounded_at("Read", {"limit": "lots"}) is None
        assert bounded_at("Read", {"limit": True}) is None
        assert bounded_at("Read", {"limit": 0}) is None

    def test_an_unrewritable_tool_has_no_bound(self):
        assert bounded_at("Bash", {"command": "ls"}) is None


class TestDescribe:
    def test_names_the_parameter_and_the_way_back(self):
        original = {"file_path": "/a.py"}
        got = narrow("Read", original, lines=300)
        text = describe("Read", original, got)
        assert "limit=300" in text
        assert "unbounded" in text
        assert "re-issue" in text.lower()

    def test_names_the_previous_bound_when_there_was_one(self):
        original = {"file_path": "/a.py", "limit": 5_000}
        text = describe("Read", original, narrow("Read", original, lines=300))
        assert "5,000" in text

    def test_grep_describes_matches_not_lines(self):
        original = {"pattern": "x", "output_mode": "content"}
        text = describe("Grep", original, narrow("Grep", original, lines=300))
        assert "matches" in text

    def test_nothing_to_say_about_an_unrewritable_tool(self):
        assert describe("Bash", {"command": "ls"}, {"command": "ls"}) == ""


class TestGuardIntegration:
    """The two rules that live in the guard rather than in `narrow`."""

    def cfg(self, **kw):
        from adder.decide.guard import Settings
        base = {"enforce": "full", "min_cost": 0.25, "min_tokens": 100,
                "narrow": True}
        base.update(kw)
        return Settings(**base)

    def call(self, cfg, tool="Read", tool_input=None, tokens=60_000):
        from adder.core.shapes import Estimate, SizeModel
        from adder.decide.guard import GuardState, decide

        class Fixed(SizeModel):
            def __init__(self):
                pass

            def predict_tool(self, tool, tool_input):
                return Estimate(p50=tokens, p90=tokens, n=50, source="test")

        return decide(tool, tool_input or {"file_path": "/big.py"},
                      model="claude-opus-5", remaining_turns=300,
                      cfg=cfg, sizes=Fixed(), state=GuardState(),
                      context_tokens=150_000)

    def test_off_by_default(self):
        from adder.decide.guard import Settings
        assert Settings().narrow is False, (
            "a path that can suppress a permission prompt must not default on")

    def test_substitutes_where_it_would_have_refused(self):
        v = self.call(self.cfg())
        assert v.fire and v.narrowed is not None
        assert v.action == "narrow"
        assert v.narrowed["limit"] > 0

    def test_refuses_as_before_when_the_setting_is_off(self):
        v = self.call(self.cfg(narrow=False))
        assert v.fire and v.narrowed is None
        assert v.action == "deny"

    def test_never_reachable_below_full_enforcement(self):
        # The relaxation argument only holds where a refusal was going to
        # happen. In advisory mode there is no refusal to relax, so there must
        # be no substitution either.
        for level in ("off", "certain"):
            v = self.call(self.cfg(enforce=level))
            assert v.fire, f"the gate did not fire at all at enforce={level}"
            assert v.narrowed is None, f"substituted while enforce={level}"

    def test_a_budget_too_small_to_bother_with_falls_back_to_a_refusal(self):
        # At a $0.01 floor the affordable read is a handful of lines. Handing the
        # model four lines of a file it asked for whole guarantees it asks again,
        # which spends the turn the substitution existed to save plus one.
        v = self.call(self.cfg(min_cost=0.01))
        assert v.fire and v.narrowed is None
        assert v.action == "deny"

    def test_bash_falls_back_to_a_refusal(self):
        v = self.call(self.cfg(), tool="Bash", tool_input={"command": "cat huge.log"})
        assert v.fire, "the gate did not fire, so this proves nothing"
        assert v.narrowed is None
        assert v.action in ("deny", "advise")

    def test_the_payload_carries_allow_updatedinput_and_a_reason(self):
        v = self.call(self.cfg())
        out = v.payload()["hookSpecificOutput"]
        assert out["hookEventName"] == "PreToolUse"
        assert out["permissionDecision"] == "allow"
        assert out["updatedInput"] == v.narrowed
        assert out["permissionDecisionReason"], (
            "a substitution the model cannot see is a lie about what it read")

    def test_the_reason_names_the_bound(self):
        v = self.call(self.cfg())
        assert "limit=" in v.payload()["hookSpecificOutput"]["permissionDecisionReason"]

    def test_updatedinput_is_a_complete_tool_input(self):
        # The harness schema-validates it, so a patch missing `file_path` fails
        # and the hook silently does nothing.
        v = self.call(self.cfg())
        assert "file_path" in v.narrowed

    def test_the_saving_is_priced_against_what_actually_ran(self):
        # The bounded call *does* run, so the saving is the difference between
        # the unbounded read and the bounded one -- not the whole read, which is
        # what a refusal would claim. Asserted as the identity rather than as a
        # comparison against the refusal, because which of those two is larger
        # depends on the delegated summary size and is not the invariant.
        v = self.call(self.cfg())
        assert v.narrowed is not None
        assert 0 < v.delegated < v.inline
        assert v.saving == pytest.approx(v.inline - v.delegated)

    def test_the_bounded_cost_tracks_the_bound_that_was_set(self):
        # Two different floors produce two different budgets, and the priced
        # cost of the substitution has to follow the budget rather than being a
        # constant that happens to look plausible.
        tight = self.call(self.cfg(min_cost=0.1))
        loose = self.call(self.cfg(min_cost=1.0))
        assert tight.narrowed["limit"] < loose.narrowed["limit"]
        assert tight.delegated < loose.delegated

    def test_a_substitution_is_not_discounted_by_advice_uptake(self):
        v = self.call(self.cfg())
        assert v.uptake == 1.0, "the bounded call runs whether or not anyone agreed"

    def test_a_broken_narrower_degrades_to_the_old_behaviour(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("bad tool input shape")

        monkeypatch.setattr("adder.decide.narrow.narrow", boom)
        v = self.call(self.cfg())
        assert v.narrowed is None and v.fire, (
            "an optional path took the whole tool call down with it")

    def test_a_caller_who_already_bounded_it_is_left_alone(self):
        # Grep with head_limit set never reaches the gate at all, which is the
        # cheaper version of the same rule.
        v = self.call(self.cfg(), tool="Grep",
                      tool_input={"pattern": "x", "output_mode": "content",
                                  "head_limit": 20})
        assert v.narrowed is None


class TestNoSilentPermissionGrant:
    def test_allow_is_emitted_only_alongside_a_substitution(self):
        from adder.decide.guard import Verdict

        advise = Verdict(True, "r", message="m")
        assert "permissionDecision" not in advise.payload()["hookSpecificOutput"]

        deny = Verdict(True, "r", message="m", deny=True)
        assert deny.payload()["hookSpecificOutput"]["permissionDecision"] == "deny"

        ask = Verdict(True, "r", message="m", ask=True)
        assert ask.payload()["hookSpecificOutput"]["permissionDecision"] == "ask"

    def test_a_silent_verdict_emits_nothing_at_all(self):
        from adder.decide.guard import Verdict
        assert Verdict(False, "nothing to say", narrowed={"limit": 1}).payload() == {}

    @pytest.mark.parametrize("tool", ["Bash", "Write", "Edit", "Task", "Glob",
                                      "WebFetch", "WebSearch"])
    def test_no_write_or_shell_tool_can_be_substituted(self, tool):
        # The rewrite is only defensible because it is a strict subset of a
        # read. Nothing that mutates or executes may take this path.
        assert narrow(tool, {"command": "x", "file_path": "/a", "prompt": "p",
                             "pattern": "q", "url": "https://x.test"},
                      lines=300) is None
