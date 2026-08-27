"""Classifier and policy behaviour, including the cases that must NOT route."""
from __future__ import annotations

import pytest

from adder.decide.route.classify import Tier, classify
from adder.decide.route.policy import decide, routing_overhead

OPUS = "claude-opus-5"


class TestClassifierPrecision:
    @pytest.mark.parametrize("task", [
        "what does cost.py do",
        "where is the retry logic",
        "which function computes the rate",
    ])
    def test_short_read_only_goes_cheap(self, task):
        v = classify(task)
        assert v.tier == Tier.T0 and v.read_only and not v.abstained

    def test_enumerating_a_plural_target_no_longer_goes_to_the_weakest_rung(self):
        """"list the files in adder/" used to sit alongside the lookups above.

        It is not a lookup: the answer is a set, and a short one is
        indistinguishable from a right one. See `TestRecallCriticalTasksAbstain` below.
        """
        v = classify("list the files in adder/")
        assert v.tier == Tier.T1 and v.read_only and not v.abstained

    @pytest.mark.parametrize("task", [
        "refactor auth across the codebase",
        "why is the cache invalidating",
        "design a migration plan for the storage layer",
        "investigate the race condition in the worker pool",
        "root-cause the performance regression",
    ])
    def test_hard_vocabulary_never_routes_down(self, task):
        assert classify(task).tier >= Tier.T2

    def test_stack_trace_forces_strong_model(self):
        v = classify("this fails:\nTraceback (most recent call last):\n  File x")
        assert v.tier >= Tier.T2

    def test_ambiguous_abstains_upward(self):
        v = classify("make it better")
        assert v.tier == Tier.T2 and v.abstained

    def test_empty_task_routes_up(self):
        assert classify("").tier == Tier.T2

    def test_multi_step_mutation_is_not_cheap(self):
        v = classify("add a retry then update the tests and finally bump the version")
        assert v.tier >= Tier.T2

    def test_read_only_flag_only_on_nonmutating(self):
        assert not classify("fix the typo in README.md").read_only

    def test_read_only_is_withheld_from_a_bare_imperative(self):
        """It is a permission, not a description of the words present.

        "make it better" contains no mutating verb and is plainly a mutation,
        so the absence of one is not enough to claim the task can run on an
        agent with no write tools.
        """
        assert not classify("make it better").read_only


class TestCLIRejectsImpossibleInputs:
    """Turns and tokens do not run backwards.

    `--remaining -5` used to reach the placement gate and raise `ValueError:
    interval out of order` as an unhandled traceback out of the CLI, while
    `--read-tokens -100` and `--context -1` were accepted in silence and
    produced prices for them. The inconsistency was an accident of which flag
    happened to be used to build an interval.
    """

    @pytest.mark.parametrize("flag", ["--remaining", "--read-tokens", "--context"])
    def test_a_negative_count_is_refused_at_the_flag(self, flag, capsys):
        from adder.decide.route.policy import main

        with pytest.raises(SystemExit) as e:
            main(["where is X", flag, "-5"])
        assert e.value.code == 2
        assert "zero or more" in capsys.readouterr().err

    def test_a_non_integer_is_refused_with_a_readable_message(self, capsys):
        from adder.decide.route.policy import main

        with pytest.raises(SystemExit):
            main(["where is X", "--remaining", "lots"])
        assert "whole number" in capsys.readouterr().err

    def test_the_library_clamps_too_because_the_hook_computes_this(self):
        """The hook takes remaining turns from the horizon estimator, not a
        human, so the arithmetic defends itself rather than trusting the flag."""
        p = decide("what does prices.py do", context_tokens=-1, remaining_turns=-5)
        assert p.action in {"inline", "delegate", "downgrade"}

    def test_is_deterministic(self):
        t = "refactor the auth module"
        assert classify(t).tier == classify(t).tier

    def test_is_fast_and_offline(self):
        import time
        start = time.perf_counter()
        for _ in range(2000):
            classify("refactor the auth module across the service")
        per_call_ms = (time.perf_counter() - start) / 2000 * 1000
        assert per_call_ms < 10.0


class TestRecallCriticalTasksAbstain:
    """A misrouted easy task costs pennies only when the failure is visible.

    It is not visible for recall. A weak model asked for every hardcoded
    credential in a tree returns three of the seven, confidently; no test
    fails, nothing retries, and the audit is wrong in a way that reads right.
    So the signal is not difficulty, it is whether an incomplete answer would
    be detectable.
    """

    @pytest.mark.parametrize("task", [
        "find every hardcoded credential across all 3167 python files",
        "count all the consent-gate bypasses",
        "list any workloads that break under the new quota",
        "which of all the recommendations were promoted without review",
        "show all the call sites that write to the audit log",
    ])
    def test_a_quantifier_over_a_plural_target_abstains(self, task):
        v = classify(task)
        assert v.tier == Tier.T2 and v.abstained
        assert any("not detectable" in r for r in v.reasons)

    def test_the_same_sentence_without_the_quantifier_is_not_forced_up(self):
        """The rule is narrow on purpose: it fires on stated exhaustiveness."""
        assert classify("find the hardcoded credential in config.py").tier == Tier.T0

    def test_the_gap_this_rule_does_not_close(self):
        """Recorded rather than papered over.

        "check the audit log for tampering" is recall-critical -- an incomplete
        answer is exactly as invisible -- and states neither a quantifier nor a
        plural target, so nothing here catches it on its wording. It abstains
        for the ordinary reason instead: nothing in it is a high-precision
        signal either way. A word list broad enough to catch it would stop
        being high-precision, which is the property the whole classifier is
        built on.
        """
        v = classify("check the audit log for tampering")
        assert v.abstained
        assert not any("not detectable" in r for r in v.reasons)

    def test_an_enumeration_of_a_plural_target_leaves_the_weakest_rung(self):
        """No quantifier, so a weaker answer rather than an abstention -- but
        completeness is still part of the answer, so not the rung whose
        under-reporting nothing would catch."""
        v = classify("list the workloads that break under a new quota")
        assert v.tier == Tier.T1 and not v.abstained and v.read_only

    @pytest.mark.parametrize("task", [
        "what does ray.data.Dataset.map_batches do",
        "where is the retry logic",
        "which function computes the rate",
    ])
    def test_a_singular_lookup_is_still_cheap(self, task):
        """The plural test must not read a third-person verb as a set.

        "which function computes the rate" ends in an s and is the exact
        singular lookup the cheapest rung exists for.
        """
        assert classify(task).tier == Tier.T0

    @pytest.mark.parametrize("task", [
        "read the audit log and tell me if anything was tampered with",
        "is there anything in the diff that bypasses the consent gate",
        "is this pattern used anywhere else",
        "did everything in the queue get acknowledged",
    ])
    def test_a_quantifying_pronoun_is_its_own_plural_target(self, task):
        """`anywhere` was listed and `anything` was not.

        Both quantify over a set by construction, so demanding a separate
        plural noun beside them asks for evidence the sentence already gave --
        and "is there anything in the diff that bypasses the consent gate" is a
        whole-diff search whose short answer reads exactly like a complete one.
        """
        v = classify(task)
        assert v.tier == Tier.T2 and v.abstained
        assert any("not detectable" in r for r in v.reasons)

    @pytest.mark.parametrize("task", [
        "locate the race condition",
        "find the hardcoded credential in this tree",
        "search for the memory leak",
        "list the deadlock",
    ])
    def test_a_defect_class_is_unbounded_whatever_its_number(self, task):
        """"find the bug" is not a claim that there is one bug.

        The definite article is a convention of English, not a statement of
        cardinality, so the plurality gate cannot see this shape: `find every
        race condition` abstained and `locate the race condition` went to the
        weakest rung. Same search, same silent failure.
        """
        v = classify(task)
        assert v.tier == Tier.T2 and v.abstained
        assert any("defect class" in r for r in v.reasons)

    def test_a_named_file_is_what_bounds_a_defect_search(self):
        """The line is scope, not grammar.

        An incomplete answer about `config.py` is checkable by opening
        `config.py`. An incomplete answer about a tree is not.
        """
        assert classify("find the hardcoded credential in config.py").tier == Tier.T0

    @pytest.mark.parametrize("task", [
        "verify no credentials are committed in this repo",
        "check whether any tenant exceeded the cost ceiling",
        "confirm nothing in the payload leaks the submodule path",
        "did anything get promoted without human review",
    ])
    def test_detection_is_enumeration_with_the_list_left_out(self, task):
        """None of `_ENUMERATE`'s verbs, all of its exposure.

        A model that checked three of the seven places answers "no", and "no"
        is also what a complete answer looks like.
        """
        v = classify(task)
        assert v.tier == Tier.T2 and v.abstained

    @pytest.mark.parametrize("task", [
        "check the schema of the events table",
        "verify the retry count in config.py",
    ])
    def test_a_bounded_check_is_not_a_detection(self, task):
        """The verb alone is not the signal, or the rule stops being precise."""
        assert not any("detection" in r for r in classify(task).reasons)


class TestKeywordCollisionsDoNotEscalate:
    """`hard` decided before anything looked at the shape of the sentence.

    On a repository whose subject matter is performance, security, concurrency
    and debugging, a vocabulary match is evidence about the repository, not
    about the task, and it was firing on one-line greps at roughly five times
    the cheapest rung's price.
    """

    def test_a_topic_word_does_not_make_a_lookup_expensive(self):
        v = classify("where is the security module")
        assert v.tier == Tier.T0 and v.read_only and not v.abstained

    def test_a_topic_word_in_a_scoped_edit_does_not_reach_the_top_rung_by_signal(self):
        v = classify("add a docstring to the debug helper")
        assert v.abstained, "abstaining is honest here; a confident T2 was not"
        assert any("topic word" in r for r in v.reasons)

    def test_naming_an_exception_class_is_not_a_stack_trace(self):
        v = classify("rename the Exception class")
        assert not any("stack trace" in r for r in v.reasons)

    @pytest.mark.parametrize("text", [
        "this fails:\nTraceback (most recent call last):\n  File x",
        "ValueError: interval out of order: -1.34 <= -5.0 <= -5.0",
        'File "adder/util/risk.py", line 155',
    ])
    def test_a_real_trace_still_forces_the_strong_model(self, text):
        assert classify(text).tier >= Tier.T2

    @pytest.mark.parametrize("task", [
        "refactor auth across the codebase",
        "why is the cache invalidating",
        "design a migration plan for the storage layer",
        "investigate the race condition in the worker pool",
        "root-cause the performance regression",
    ])
    def test_the_work_verbs_still_decide(self, task):
        v = classify(task)
        assert v.tier >= Tier.T2 and not v.abstained


class TestRoutingDeclines:
    """The failure mode that sinks naive routers: routing when it costs more."""

    def test_declines_when_saving_below_overhead(self):
        """Three turns left is not enough horizon to amortize anything.

        This used to come back as a `delegate` whose saving happened to sit
        below the routing overhead, with the render appending a line telling you
        not to do the thing it had just recommended. It now declines outright,
        because placement is priced with its redo risk: at a 900K context the
        turn that catches a bad summary costs $0.46 on its own, which swamps the
        $0.04 that delegating an 8K read saves across three remaining turns.
        """
        p = decide("what does prices.py do", context_tokens=900_000, remaining_turns=3)
        assert not p.worth_it
        assert p.action == "inline"
        assert any("keep inline" in r for r in p.reasons)

    def test_render_warns_when_a_recommendation_misses_its_overhead(self):
        """The render path is still there for a plan that misses its own bar.

        Built directly rather than through `decide`, which no longer produces
        one: the gate declines first. The line is the last defence for a caller
        that constructs a Plan itself, so it stays under test.
        """
        from adder.decide.route.classify import Tier
        from adder.decide.route.policy import Plan

        p = Plan(action="delegate", tier=Tier.T0, model="claude-haiku-4-5",
                 effort="default", agent="route-t0", saving=0.01, overhead=0.46,
                 confidence=0.9, reasons=[])
        assert not p.worth_it
        assert "does not clear routing overhead" in p.render()

    def test_routes_when_saving_clears_overhead(self):
        p = decide("what does prices.py do", context_tokens=400_000, remaining_turns=500)
        assert p.worth_it and p.action == "delegate"

    def test_overhead_scales_with_context(self):
        assert routing_overhead(900_000, OPUS) > routing_overhead(100_000, OPUS)

    def test_overhead_is_material_in_big_sessions(self):
        """~$0.25 just to spend a turn deciding, at 500K context."""
        assert 0.20 < routing_overhead(500_000, OPUS) < 0.35


class TestPolicyRouting:
    def test_hard_task_delegates_to_strong_tier(self):
        p = decide("refactor auth across the codebase",
                   context_tokens=400_000, remaining_turns=400)
        assert p.agent == "route-t2" and p.model == "claude-opus-5"

    def test_cheap_task_delegates_to_haiku(self):
        p = decide("what does prices.py do", context_tokens=400_000, remaining_turns=400)
        assert p.agent == "route-t0" and p.model == "claude-haiku-4-5"

    def test_never_downgrades_a_hard_task(self):
        p = decide("redesign the storage layer", context_tokens=5_000, remaining_turns=2)
        assert p.action != "downgrade"

    def test_plan_explains_itself(self):
        p = decide("what does prices.py do", context_tokens=400_000, remaining_turns=400)
        assert p.reasons and all(isinstance(r, str) for r in p.reasons)

    def test_zero_remaining_turns_still_safe(self):
        p = decide("what does x do", context_tokens=10_000, remaining_turns=0)
        assert isinstance(p.saving, float)


class TestNoTaskGuard:
    """If argument substitution fails, refuse rather than delegate nothing."""

    @pytest.mark.parametrize("task", ["", "   ", "$ARGUMENTS", "${ARGUMENTS}", "$1"])
    def test_missing_task_refuses_to_route(self, task):
        p = decide(task, context_tokens=400_000, remaining_turns=500)
        assert p.action == "inline" and not p.worth_it and p.confidence == 0.0
        assert "no task text" in " ".join(p.reasons)

    def test_real_task_still_routes(self):
        p = decide("what does prices.py do", context_tokens=400_000, remaining_turns=500)
        assert p.action == "delegate"


class TestLiveExpiryNotice:
    """A rate change is a re-tune, not a footnote: every threshold in this repo
    is a ratio of two prices. `on` is a parameter so the notice is testable
    rather than dependent on the day the suite runs."""

    def _sess(self, model):
        from adder.core.trace import Session, Turn

        s = Session("s", "p")
        for i in range(10):
            s.turns.append(Turn("s", "p", model, 0, 50_000 + i * 1_000, 0, 400, 0,
                                False, ts=f"2026-08-15T10:{i:02d}:00Z"))
        return s

    def test_it_warns_inside_the_window(self):
        from datetime import date

        from adder.measure.session.live import render

        text = render(self._sess("claude-sonnet-5"), on=date(2026, 8, 15))
        assert "introductory rate" in text
        assert "2026-08-31" in text

    def test_it_is_quiet_well_before(self):
        from datetime import date

        from adder.measure.session.live import render

        assert "introductory" not in render(self._sess("claude-sonnet-5"),
                                            on=date(2026, 1, 1))

    def test_it_is_quiet_after_the_change(self):
        from datetime import date

        from adder.measure.session.live import render

        assert "introductory" not in render(self._sess("claude-sonnet-5"),
                                            on=date(2026, 9, 15))

    def test_a_model_with_no_intro_rate_is_never_warned_about(self):
        from datetime import date

        from adder.measure.session.live import render

        assert "introductory" not in render(self._sess("claude-opus-5"),
                                            on=date(2026, 8, 15))
