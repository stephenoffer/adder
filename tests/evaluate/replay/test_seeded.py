"""The one quality signal here that cannot agree with the cost model.

`adder quality` and `adder verify` read the same transcripts, through the same
parser, priced by the same cost model as the saving they are checking. If that
model is wrong, they are wrong in the same direction and agree with themselves.
That asymmetry runs the wrong way: cost is measured five ways and quality --
the thing routing would actually lose -- was measured by the cost machinery.

So this fixture supplies a corroborating signal that shares no code with any of
it. A file with K defects planted in it, a prompt, and a string match. The
tests below are about the scorer, because the scorer is the part that could
quietly turn into an advertisement: too lenient and every model passes, too
strict and no reply ever matches.

Nothing here calls a model. The API path is `adder ab --recall --run`, which is
opt-in for the same reason the rest of `ab` is.
"""

from __future__ import annotations

import ast

from adder.evaluate.replay.seeded import (
    PROMPT,
    SEEDS,
    SOURCE,
    K,
    Recall,
    Seed,
    report,
    score,
)


class TestTheFixture:
    def test_it_parses_as_python(self):
        """A fixture that does not parse is a fixture about syntax errors."""
        ast.parse(SOURCE)

    def test_every_seed_names_a_function_that_exists_in_it(self):
        tree = ast.parse(SOURCE)
        names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert {s.symbol for s in SEEDS} <= names

    def test_the_ids_are_unique(self):
        assert len({s.id for s in SEEDS}) == K

    def test_the_prompt_does_not_say_how_many_there_are(self):
        """A model told to find nine things reports nine things. The number it
        was given would then be doing the work the measurement is for."""
        assert str(K) not in PROMPT
        assert "nine" not in PROMPT.lower()

    def test_it_is_short_enough_to_be_read_whole(self):
        """Recall over a file nobody could hold is a context measurement wearing
        a quality measurement's clothes."""
        assert len(SOURCE.splitlines()) < 120


class TestTheScorerIsStrictEnough:
    def test_an_empty_reply_finds_nothing(self):
        assert score("") == set()

    def test_naming_the_function_alone_is_not_a_finding(self):
        """Otherwise a reply that lists every function in the file scores K/K,
        and one marker was enough to make that true: `"//"` normalises to the
        empty string, which is a substring of everything."""
        listing = " ".join(s.symbol for s in SEEDS)
        assert score(listing) == set()

    def test_no_marker_normalises_to_nothing(self):
        """The general form of the bug above, kept as a rule rather than as a
        fixed test for one entry."""
        from adder.evaluate.replay.seeded import _normalise

        assert all(_normalise(w) for s in SEEDS for w in s.any_of)

    def test_describing_a_defect_without_saying_where_is_not_a_finding(self):
        """A model that writes plausible sentences about software in general
        must score zero, or the fixture measures fluency."""
        assert score("There is an off-by-one error and a division by zero "
                     "somewhere, and a race condition.") == set()

    def test_the_wrong_function_does_not_earn_the_credit(self):
        assert "range-off-by-one" not in score(
            "average_cost has an off-by-one in its range")


class TestTheScorerIsLenientEnough:
    def test_a_plain_correct_answer_scores(self):
        found = score("In parse_window the range is exclusive of hi, so the "
                      "last page is dropped.")
        assert found == {"range-off-by-one"}

    def test_backticks_and_code_formatting_do_not_defeat_it(self):
        """A scorer beaten by a backtick is measuring markdown."""
        assert score("`average_cost()` raises **ZeroDivisionError** on `[]`") == \
            {"empty-division"}

    def test_a_thorough_answer_finds_every_one(self):
        """The measurement has to be reachable. If no reply can score K/K the
        denominator is decorative."""
        reply = " ".join(f"{s.symbol}: {s.any_of[0]}" for s in SEEDS)
        assert score(reply) == {s.id for s in SEEDS}

    def test_two_seeds_in_one_function_are_scored_apart(self):
        """`read_config` carries both a bare except and a leaked handle. A
        reply that finds one must not be credited with the other."""
        found = score("read_config uses a bare except that swallows everything")
        assert found == {"bare-except"}


class TestTheReport:
    def test_it_names_what_was_missed(self):
        """`6 of 9` is a number; "missed the unbounded retry" is a decision
        about what to route where."""
        text = report([Recall("m", {"range-off-by-one"})])
        assert "retry_fetch" in text and "parse_window" not in text.split("missed")[1]

    def test_a_failed_arm_reports_the_error_rather_than_zero_recall(self):
        """A model that could not be reached scored nothing; reporting that as
        0/9 would be the same bad number this whole file exists to avoid."""
        text = report([Recall("m", error="timeout")])
        assert "timeout" in text

    def test_the_scope_limit_is_stated(self):
        assert "licenses nothing" in report([Recall("m")])

    def test_recall_is_out_of_the_planted_count(self):
        r = Recall("m", {s.id for s in SEEDS[:3]})
        assert r.n == K and r.rate == 3 / K and len(r.missed) == K - 3


class TestItSharesNoCodeWithTheCostModel:
    def test_the_module_imports_nothing_from_the_pricing_or_trace_layers(self):
        """The claim in the docstring, as an assertion. A corroborating signal
        that reaches for `admitted_token_cost` stops corroborating."""
        import pathlib

        src = pathlib.Path(
            "adder/evaluate/replay/seeded.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        imported = {
            (node.module or "")
            for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        }
        banned = ("adder.pricing", "adder.core.trace", "adder.core.shapes",
                  "adder.measure")
        assert not [m for m in imported if m.startswith(banned)]

    def test_scoring_needs_nothing_but_text(self):
        assert isinstance(score("parse_window is off by one"), set)


class TestSeedMatching:
    def test_both_halves_are_required(self):
        s = Seed("x", "foo", ("leaks",), "")
        assert not s.found_in("foo is fine")
        assert not s.found_in("something leaks")
        assert s.found_in("foo leaks a handle")
