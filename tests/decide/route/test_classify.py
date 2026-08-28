"""Two ways the classifier read a sentence wrongly, both found by probing it.

The first is arithmetic: `_PATHLIKE` carried a capturing group, so `re.findall`
returned the extension rather than the path it matched, and every multi-file
request counted as one file. A three-file edit took the `scoped edit to one
named file` branch and routed to T1; a read-only question spanning four files
never reported spanning any. The count was wrong in the cheap direction, which
is the direction that costs a retry.

The second is vocabulary, and it is the more interesting failure because
nothing was broken. `_DEFECT` is a list of defect nouns and the list worked
exactly as written -- six probes simply named classes that were not on it, so
"find the security flaw" and "locate the crash" reached the weakest rung at 0.85
confidence. A whole-tree audit priced as a lookup, with the silent failure the
defect rule exists to stop. Adding six words would have bought six probes, so
the default inverted instead: an enumeration over a definite noun phrase
abstains unless the sentence says what bounds it.

They are one file because they are one lesson. A classifier is a claim about
what a sentence means, and both of these were claims that held on every example
anybody had tried.
"""
from __future__ import annotations

import pytest

from adder.decide.route.classify import (
    _PATHLIKE,
    Tier,
    _unbounded_target,
    classify,
)


class TestFileCounting:
    def test_findall_returns_paths_not_extensions(self):
        t = "update adder/cost.py and adder/trace.py and adder/policy.py"
        assert _PATHLIKE.findall(t) == [
            "adder/cost.py", "adder/trace.py", "adder/policy.py"]

    def test_distinct_files_sharing_one_extension_still_count_separately(self):
        """The regression: three `.py` paths collapsed to a single `{'py'}`."""
        t = "a/one.py b/two.py c/three.py"
        assert len(set(_PATHLIKE.findall(t))) == 3

    def test_mixed_extensions_are_counted(self):
        t = "wire up src/app.ts, src/app.tsx and config.yaml"
        assert len(set(_PATHLIKE.findall(t))) == 3

    def test_the_same_path_twice_is_one_file(self):
        assert len(set(_PATHLIKE.findall("edit a/b.py then a/b.py again"))) == 1


class TestTierConsequences:
    def test_a_multi_file_edit_is_not_a_scoped_single_file_edit(self):
        """`mutating and files == 1` is the T1 branch; two files must not take it."""
        v = classify("update adder/cost.py and adder/trace.py")
        assert v.tier >= Tier.T2

    def test_a_genuinely_single_file_edit_still_routes_to_t1(self):
        v = classify("update adder/cost.py")
        assert v.tier is Tier.T1

    def test_a_read_only_question_spanning_files_reports_the_span(self):
        v = classify("where is x defined in a/one.py and b/two.py")
        assert any("spans 2 files" in r for r in v.reasons)


# The six probes, verbatim. Five reached T0 at 0.85; the sixth ("find the bug")
# was on the list and did not.
LEAKED = [
    "find the security flaw",
    "find the data corruption",
    "locate the privilege escalation",
    "find the auth bypass",
    "locate the crash",
    "find the race",
]

# Searches that name one findable thing. Each is checkable by opening what it
# names, which is the property that makes a short answer a complete one.
BOUNDED = [
    "find the config file",
    "locate the definition of process_batch",
    "show the line where the timeout is set",
    "find the function that parses the header",
    "list the tests for this module",
    "locate the `guard_min_tokens` setting",
    "find the class DispatchLedger",
    "show the error message adder/cli/help.py prints",
]


class TestTheLeakIsClosed:
    @pytest.mark.parametrize("task", LEAKED)
    def test_an_unlisted_defect_noun_abstains(self, task):
        v = classify(task)
        assert v.abstained and v.tier >= Tier.T2, (
            f"{task!r} routed to {v.tier.name} at {v.confidence}: a search whose "
            "incomplete answer reads exactly like a complete one")

    @pytest.mark.parametrize("task", LEAKED)
    def test_the_reason_says_what_was_unbounded(self, task):
        """A refusal to route cheap has to be inspectable, or the next person
        widens the wordlist again instead of reading why it exists."""
        assert any("bounds it" in r for r in classify(task).reasons)

    def test_a_noun_nobody_has_thought_of_yet_also_abstains(self):
        """The actual claim. Not that these six are handled -- that the next six
        are, without an edit."""
        v = classify("find the tenant isolation gap")
        assert v.abstained and v.tier >= Tier.T2

    def test_it_is_still_read_only(self):
        """Abstaining routes up, not sideways: nothing here asks for a change,
        so the permission the verdict publishes must not quietly narrow."""
        assert classify("find the auth bypass").read_only


class TestOrdinaryLookupsStillRouteCheap:
    """The cost of the rule above, and the reason it is an allowlist.

    An abstention is cheap when it is rare. If "find the config file" abstained
    too, the classifier would be paying routing overhead to say "no change" on
    every lookup in the session, which is the failure `_HARD_TOPIC` was demoted
    for causing.
    """

    @pytest.mark.parametrize("task", BOUNDED)
    def test_a_bounded_search_is_not_abstained_on(self, task):
        v = classify(task)
        assert not v.abstained, f"{task!r} abstained: {v.reasons}"
        assert v.tier <= Tier.T1

    def test_a_named_path_bounds_a_search_a_bare_noun_would_not(self):
        assert _unbounded_target("find the deadlock")
        assert not _unbounded_target("find the deadlock in adder/decide/guard.py")

    def test_a_named_symbol_bounds_it_too(self):
        assert not _unbounded_target("find the caller of load_state")

    def test_a_quoted_string_bounds_it(self):
        assert not _unbounded_target('find the handler for "PreToolUse"')

    def test_an_indefinite_article_is_not_this_rule(self):
        """`find a way to do X` is not a search over a tree; the abstention it
        deserves, if any, comes from a different rule."""
        assert not _unbounded_target("find a faster hashing approach")

    def test_a_verb_that_is_not_an_enumeration_is_not_this_rule(self):
        assert not _unbounded_target("fix the crash")


class TestItDidNotBreakWhatWasWorking:
    def test_a_stated_quantifier_still_abstains(self):
        assert classify("find every hardcoded credential").abstained

    def test_a_listed_defect_noun_still_abstains(self):
        assert classify("find the memory leak").abstained

    def test_a_short_single_target_question_is_still_the_cheapest_rung(self):
        v = classify("what does map_batches do")
        assert v.tier is Tier.T0 and not v.abstained

    def test_a_scoped_edit_is_untouched(self):
        assert classify("add a docstring to adder/util/text.py").tier is Tier.T1


class TestTheProjectVocabulary:
    """What a domain codebase can teach the classifier about itself.

    Measured on a real one: twelve out of twelve task phrasings from the
    repository's own tracker abstained, every one to the top rung at confidence
    0.3. A classifier that always abstains is not being conservative -- it is
    charging routing overhead to arrive at "no change", once per task. The
    cause is not subtle: the shipped vocabulary is English about software in
    general, and a repository whose subject is scheduling has fifty nouns that
    decide what a task is and the classifier has seen none of them.

    Declared rather than learned, and the reason is a stated invariant rather
    than a preference: `outcomes.Outcome` stores `task_hash` and never the task
    text, and `track/similar.py` builds a MinHash sketch specifically so the
    terms cannot be recovered from the log. Learning a vocabulary out of that
    would undo it.
    """

    @pytest.fixture
    def domain(self, tmp_path, monkeypatch):
        import json

        (tmp_path / ".adder.json").write_text(json.dumps({
            "classify_terms": "cheap=placement group,map_batches; "
                              "hard=autoscaler,preemption"}))
        monkeypatch.chdir(tmp_path)
        return tmp_path

    def test_nothing_changes_without_it(self):
        from adder.decide.route.classify import project_terms

        assert project_terms() == {"cheap": (), "hard": ()}

    def test_a_cheap_term_bounds_a_search_that_would_otherwise_abstain(self, domain):
        assert classify("locate the shuffle buffer").abstained
        v = classify("locate the placement group")
        assert not v.abstained and v.tier is Tier.T0

    def test_a_hard_term_decides_like_the_shipped_hard_vocabulary(self, domain):
        v = classify("add a docstring to the autoscaler helper")
        assert v.tier >= Tier.T2 and not v.abstained

    def test_a_downgrade_it_buys_is_always_traceable(self, domain):
        """The whole risk of a configured vocabulary is a cheap route nobody
        can explain. Every term that decides says so, and names itself."""
        v = classify("locate the placement group")
        assert any("classify_terms" in r for r in v.reasons)

    def test_a_defect_class_still_outranks_a_cheap_term(self, domain):
        """A project term says a thing is findable, not that an audit for
        defects in it is bounded. `find the placement group leak` is still a
        search whose incomplete answer reads like a complete one."""
        assert classify("find the placement group leak").abstained

    def test_a_malformed_setting_is_ignored_rather_than_fatal(self, tmp_path,
                                                              monkeypatch):
        import json

        (tmp_path / ".adder.json").write_text(json.dumps(
            {"classify_terms": "nonsense without an equals sign"}))
        monkeypatch.chdir(tmp_path)
        from adder.decide.route.classify import project_terms

        assert project_terms() == {"cheap": (), "hard": ()}

    def test_phrases_survive_whitespace_and_case(self, tmp_path, monkeypatch):
        import json

        (tmp_path / ".adder.json").write_text(json.dumps(
            {"classify_terms": "cheap=  Placement   Group  "}))
        monkeypatch.chdir(tmp_path)
        from adder.decide.route.classify import project_terms

        assert project_terms()["cheap"] == ("placement group",)
