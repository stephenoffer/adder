"""Experiment design, pinned on the allocations that waste a budget.

Three failure modes, each with a test: spending on foregone conclusions,
spending everything on the single closest pair, and giving a contested pair
nothing at all.
"""

from __future__ import annotations

import json

import pytest

from adder.evaluate.claims import design
from adder.pricing.bt import Battle, fit_with_ci


def _battles(pairs):
    """`pairs` is [(a, b, wins_for_a, wins_for_b), ...]."""
    out = []
    for a, b, wa, wb in pairs:
        out += [Battle(a, b, "a")] * wa + [Battle(a, b, "b")] * wb
    return out


class TestOverlap:
    def test_disjoint_intervals_do_not_overlap(self):
        r = fit_with_ci(_battles([("a", "b", 200, 0)]), resamples=40)
        assert design.interval_overlap(r["a"], r["b"]) == 0.0

    def test_identical_models_overlap_heavily(self):
        """Two models with an identical record. The overlap does not reach 1.0
        because resampling offsets the two intervals slightly; what matters is
        that it stays high enough to keep the pair at the top of the plan."""
        r = fit_with_ci(_battles([("a", "b", 100, 100)]), resamples=40)
        assert design.interval_overlap(r["a"], r["b"]) > 0.75

    def test_overlap_is_bounded(self):
        r = fit_with_ci(_battles([("a", "b", 60, 40)]), resamples=40)
        assert 0.0 <= design.interval_overlap(r["a"], r["b"]) <= 1.0


class TestScoring:
    def test_a_coin_flip_outranks_a_foregone_conclusion(self):
        battles = _battles([("close1", "close2", 50, 50), ("strong", "weak", 200, 0)])
        ratings = fit_with_ci(battles, resamples=60)
        pairs = {p.key: p for p in design.score_pairs(ratings, battles)}
        close = pairs[("close1", "close2")]
        settled = pairs[("strong", "weak")]
        assert close.value > settled.value

    def test_a_heavily_sampled_pair_loses_priority_to_an_unsampled_one(self):
        """Diminishing returns: otherwise one cell eats the whole budget."""
        battles = _battles([("a", "b", 400, 400), ("c", "d", 4, 4)])
        ratings = fit_with_ci(battles, resamples=60)
        pairs = {p.key: p for p in design.score_pairs(ratings, battles)}
        assert pairs[("c", "d")].value > pairs[("a", "b")].value

    def test_a_settled_pair_keeps_a_small_floor(self):
        """Separated today is not separated forever if the task mix drifts."""
        battles = _battles([("strong", "weak", 300, 0)])
        ratings = fit_with_ci(battles, resamples=60)
        pair = design.score_pairs(ratings, battles)[0]
        assert pair.settled
        assert pair.value > 0.0

    def test_candidates_can_be_restricted(self):
        battles = _battles([("a", "b", 10, 10), ("c", "d", 10, 10)])
        ratings = fit_with_ci(battles, resamples=40)
        pairs = design.score_pairs(ratings, battles, candidates=[("a", "b")])
        assert [p.key for p in pairs] == [("a", "b")]

    def test_an_unknown_model_in_a_candidate_is_skipped(self):
        battles = _battles([("a", "b", 10, 10)])
        ratings = fit_with_ci(battles, resamples=40)
        assert design.score_pairs(ratings, battles, candidates=[("a", "ghost")]) == []

    def test_pair_counts_come_from_the_log(self):
        battles = _battles([("a", "b", 7, 3)])
        assert design.counts(battles) == {("a", "b"): 10}

    def test_counts_are_order_insensitive(self):
        battles = [Battle("b", "a", "a"), Battle("a", "b", "b")]
        assert design.counts(battles) == {("a", "b"): 2}


class TestAllocation:
    def test_the_budget_is_spent_exactly(self):
        battles = _battles([("a", "b", 20, 20), ("c", "d", 5, 5), ("e", "f", 1, 1)])
        ratings = fit_with_ci(battles, resamples=40)
        pairs = design.score_pairs(ratings, battles)
        alloc = design.allocate(pairs, 50)
        assert sum(alloc.values()) == 50

    def test_the_contested_pair_gets_more_than_the_settled_one(self):
        battles = _battles([("close1", "close2", 30, 30), ("strong", "weak", 200, 0)])
        ratings = fit_with_ci(battles, resamples=60)
        alloc = design.allocate(design.score_pairs(ratings, battles), 60)
        assert alloc.get(("close1", "close2"), 0) > alloc.get(("strong", "weak"), 0)

    def test_no_budget_allocates_nothing(self):
        assert design.allocate([], 10) == {}
        battles = _battles([("a", "b", 5, 5)])
        ratings = fit_with_ci(battles, resamples=20)
        assert design.allocate(design.score_pairs(ratings, battles), 0) == {}

    def test_equal_value_pairs_are_split_evenly_not_by_sort_order(self):
        pairs = [design.Pair("a", "b", 0, 0.5, 1.0, 0.0),
                 design.Pair("c", "d", 0, 0.5, 1.0, 0.0)]
        alloc = design.allocate(pairs, 10)
        assert sorted(alloc.values()) == [5, 5]

    def test_largest_remainder_does_not_invent_comparisons(self):
        pairs = [design.Pair(f"m{i}", f"n{i}", 0, 0.5, 1.0, 1.0) for i in range(7)]
        alloc = design.allocate(pairs, 10)
        assert sum(alloc.values()) == 10


class TestSeparation:
    def test_a_lopsided_pair_separates_quickly(self):
        assert design.comparisons_to_separate(0.9) < 50

    def test_a_near_tie_needs_an_impractical_sample(self):
        assert design.comparisons_to_separate(0.52) > 500

    def test_a_true_tie_never_separates(self):
        assert design.comparisons_to_separate(0.5) == 100_000

    def test_it_is_symmetric_about_a_half(self):
        assert (design.comparisons_to_separate(0.7) ==
                design.comparisons_to_separate(0.3))

    def test_a_bad_probability_is_rejected(self):
        with pytest.raises(ValueError):
            design.comparisons_to_separate(1.5)


class TestPlan:
    def test_a_plan_prices_itself(self):
        battles = _battles([("a", "b", 20, 20)])
        pl = design.plan(battles, budget=40, cost_per_comparison=0.25, resamples=40)
        assert pl.total_cost == pytest.approx(10.0)

    def test_an_empty_log_produces_an_empty_plan(self):
        pl = design.plan([], budget=40)
        assert pl.pairs == []
        assert "nothing to rank" in design.format_report(pl)

    def test_the_report_names_the_hardest_pair(self):
        battles = _battles([("close1", "close2", 40, 40), ("strong", "weak", 200, 0)])
        text = design.format_report(design.plan(battles, budget=60, resamples=60))
        assert "close1" in text

    def test_it_says_when_a_pair_is_not_worth_measuring(self):
        battles = _battles([("a", "b", 100, 100)])
        text = design.format_report(design.plan(battles, budget=60,
                                                cost_per_comparison=1.0, resamples=60))
        assert "route on price" in text or "interchangeable" in text

    def test_json_is_finite_and_complete(self):
        battles = _battles([("a", "b", 20, 20), ("c", "d", 5, 5)])
        payload = design.plan(battles, budget=30, resamples=40).to_json()
        text = json.dumps(payload)
        assert "NaN" not in text and "Infinity" not in text
        assert sum(row["comparisons"] for row in payload["plan"]) == 30


class TestLoading:
    def test_it_reads_a_comparison_log(self, tmp_path):
        p = tmp_path / "b.jsonl"
        p.write_text('{"a":"x","b":"y","winner":"a"}\n# note\n\n'
                     '{"a":"x","b":"y","winner":"tie"}\n')
        assert len(design.load_battles(p)) == 2

    def test_a_malformed_line_names_itself(self, tmp_path):
        p = tmp_path / "b.jsonl"
        p.write_text('{"a":"x","b":"y"}\nnope\n')
        with pytest.raises(ValueError, match=r":2:"):
            design.load_battles(p)

    def test_a_missing_model_is_named(self, tmp_path):
        p = tmp_path / "b.jsonl"
        p.write_text('{"a":"x"}\n')
        with pytest.raises(ValueError, match="b"):
            design.load_battles(p)


class TestCli:
    def test_no_log_exits_one_with_output(self, capsys, isolated_home):
        assert design.main([]) == 1
        assert capsys.readouterr().out.strip()

    def test_json_parses_with_no_log(self, capsys, isolated_home):
        design.main(["--json"])
        json.loads(capsys.readouterr().out)

    def test_a_missing_file_is_an_error(self, tmp_path, capsys):
        assert design.main([str(tmp_path / "nope.jsonl")]) == 1

    def test_it_plans_from_a_real_log(self, tmp_path, capsys, isolated_home):
        p = tmp_path / "b.jsonl"
        rows = ([{"a": "m1", "b": "m2", "winner": "a"}] * 20 +
                [{"a": "m1", "b": "m2", "winner": "b"}] * 20 +
                [{"a": "m3", "b": "m4", "winner": "a"}] * 30)
        p.write_text("\n".join(json.dumps(r) for r in rows))
        assert design.main([str(p), "--budget", "50", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert sum(r["comparisons"] for r in payload["plan"]) == 50
