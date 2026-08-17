"""The router metric, pinned against the cases where it would flatter itself.

Three properties matter more than the rest, and each has a test that fails if
the implementation drifts:

* a router that orders tasks at random scores 0.500, not 0.550 -- the bin
  convention is what makes every other number readable;
* a router that is *worse* than random scores below 0.5, so the metric can say
  "this is doing harm" rather than bottoming out at zero improvement;
* the dollar axis and the call axis disagree on exactly the workload this tool
  measures, and the report has to show both.
"""

from __future__ import annotations

import json

import pytest

from adder.evaluate.replay import routereval as re_
from adder.evaluate.replay.routereval import Episode


def _episodes(n=40, *, perfect=True, cost_strong=1.0, cost_weak=0.1):
    """Half the tasks need the strong model; `score` ranks them or does not."""
    out = []
    for i in range(n):
        needs_strong = i % 2 == 0
        out.append(Episode(
            key=f"t{i}",
            q_strong=1.0,
            q_weak=0.0 if needs_strong else 1.0,
            cost_strong=cost_strong,
            cost_weak=cost_weak,
            score=(1.0 if needs_strong else 0.0) if perfect else (0.0 if needs_strong else 1.0),
        ))
    return out


class TestCurve:
    def test_endpoints_are_exact(self):
        pts = re_.curve(_episodes())
        assert pts[0].call_fraction == 0.0
        assert pts[0].pgr == pytest.approx(0.0)
        assert pts[-1].call_fraction == 1.0
        assert pts[-1].pgr == pytest.approx(1.0)

    def test_the_curve_has_one_point_per_threshold(self):
        assert len(re_.curve(_episodes(n=12))) == 13

    def test_quality_is_monotone_for_a_perfect_router(self):
        pts = re_.curve(_episodes())
        qualities = [p.quality for p in pts]
        assert qualities == sorted(qualities)

    def test_cost_rises_with_the_strong_call_count(self):
        pts = re_.curve(_episodes())
        assert pts[-1].cost > pts[0].cost
        assert pts[-1].cost_fraction == pytest.approx(1.0)

    def test_an_empty_set_produces_no_curve(self):
        assert re_.curve([]) == []

    def test_the_sweep_is_stable_when_scores_tie(self):
        """A classifier emitting five discrete levels ties constantly."""
        tied = [Episode(f"t{i}", 1.0, 0.0, 1.0, 0.1, score=1.0) for i in range(8)]
        assert re_.curve(tied) == re_.curve(list(reversed(tied)))


class TestApgr:
    def test_a_random_router_scores_one_half(self):
        """The bin convention exists to make this exactly 0.5, not 0.55."""
        lo, hi = re_.random_router_ci(_episodes(n=200), trials=120)
        assert lo < 0.5 < hi
        assert abs((lo + hi) / 2 - 0.5) < 0.05

    def test_a_perfect_router_hits_the_ceiling_the_data_allows(self):
        """APGR does not top out at 1.0, and reading it as if it did misleads.

        Half these tasks genuinely need the strong model, so even a perfect
        router only reaches PGR=1 once it has spent half the calls. Its APGR is
        0.75, and that is the ceiling for *this* task mix -- which is exactly
        the oracle number, so the two must agree.
        """
        eps = _episodes(perfect=True)
        perfect = re_.apgr(re_.curve(eps))
        assert perfect == pytest.approx(0.75, abs=0.01)
        assert perfect == pytest.approx(re_.apgr(re_.oracle(eps)), abs=1e-9)

    def test_an_inverted_router_scores_below_one_half(self):
        """A router can do harm, and the metric has to be able to say so."""
        assert re_.apgr(re_.curve(_episodes(perfect=False))) < 0.5

    def test_apgr_is_bounded_by_the_oracle(self):
        eps = _episodes(perfect=True)
        assert re_.apgr(re_.curve(eps)) <= re_.apgr(re_.oracle(eps)) + 1e-9

    def test_the_oracle_is_the_best_ordering_available(self):
        eps = _episodes(perfect=False)
        assert re_.apgr(re_.oracle(eps)) > re_.apgr(re_.curve(eps))


class TestCpt:
    def test_a_perfect_router_reaches_full_quality_on_half_the_calls(self):
        pts = re_.curve(_episodes())
        # Exactly half the tasks need the strong model, and a perfect router
        # ranks all of them first.
        assert re_.cpt(pts, 1.0) == pytest.approx(0.5)

    def test_cpt_is_monotone_in_the_target(self):
        pts = re_.curve(_episodes())
        assert re_.cpt(pts, 0.5) <= re_.cpt(pts, 0.8) <= re_.cpt(pts, 1.0)

    def test_an_unreachable_target_costs_everything(self):
        assert re_.cpt(re_.curve(_episodes()), 1.5) == 1.0

    def test_cpt_on_an_empty_curve_is_one(self):
        assert re_.cpt([], 0.5) == 1.0


class TestCostAxis:
    def test_the_axes_agree_when_every_call_costs_the_same(self):
        eps = [Episode(f"t{i}", 1.0, 0.0 if i % 2 else 1.0, 1.0, 0.0,
                       score=float(i % 2)) for i in range(20)]
        pts = re_.curve(eps)
        assert re_.apgr(pts, "calls") == pytest.approx(re_.apgr(pts, "cost"), abs=0.02)

    def test_the_axes_diverge_when_the_expensive_calls_are_the_big_ones(self):
        """The agent-session case: the strong calls carry the huge contexts."""
        eps = []
        for i in range(20):
            needs_strong = i % 2 == 0
            eps.append(Episode(
                key=f"t{i}",
                q_strong=1.0,
                q_weak=0.0 if needs_strong else 1.0,
                # Tasks that need the strong model are the 190k-context ones.
                cost_strong=40.0 if needs_strong else 1.0,
                cost_weak=0.1,
                score=1.0 if needs_strong else 0.0,
            ))
        pts = re_.curve(eps)
        by_calls = re_.apgr(pts, "calls")
        by_cost = re_.apgr(pts, "cost")
        assert by_calls > by_cost
        assert by_calls - by_cost > 0.05


class TestReport:
    def test_evaluate_fills_every_summary_field(self):
        rep = re_.evaluate(_episodes(), resamples=40)
        assert rep.n == 40
        assert rep.apgr_calls > 0.5
        assert rep.apgr_ci[0] <= rep.apgr_calls <= rep.apgr_ci[1]
        assert set(rep.cpt) == {50, 80, 95}
        assert rep.separable

    def test_a_good_router_is_reported_as_beating_random(self):
        assert re_.evaluate(_episodes(n=200), resamples=60).beats_random

    def test_a_random_router_is_not_reported_as_beating_random(self):
        eps = [Episode(f"t{i}", 1.0, float(i % 2), 1.0, 0.1, score=(i * 7 % 13) / 13.0)
               for i in range(30)]
        assert not re_.evaluate(eps, resamples=60).beats_random

    def test_identical_arms_are_called_out_rather_than_scored(self):
        """No gap to recover means PGR has no denominator, not that we won."""
        eps = [Episode(f"t{i}", 1.0, 1.0, 1.0, 0.1, score=float(i)) for i in range(10)]
        rep = re_.evaluate(eps, resamples=20)
        assert not rep.separable
        assert "no quality gap" in re_.format_report(rep).lower()

    def test_an_empty_report_renders_and_does_not_divide_by_zero(self):
        rep = re_.evaluate([], resamples=10)
        assert rep.n == 0
        assert "no episodes" in re_.format_report(rep).lower()

    def test_the_modelled_share_is_disclosed(self):
        eps = [Episode(f"t{i}", 1.0, float(i % 2), 1.0, 0.1, score=float(i),
                       modelled=True) for i in range(10)]
        rep = re_.evaluate(eps, resamples=20)
        assert rep.modelled_share == 1.0
        assert "MODELLED" in re_.format_report(rep)

    def test_json_is_finite_and_complete(self):
        payload = re_.evaluate(_episodes(), resamples=20).to_json()
        text = json.dumps(payload)
        assert "NaN" not in text and "Infinity" not in text
        assert payload["apgr"]["oracle"] >= payload["apgr"]["calls"] - 1e-9
        assert len(payload["curve"]) == 41

    def test_the_report_names_the_axis_disagreement(self):
        eps = []
        for i in range(20):
            needs_strong = i % 2 == 0
            eps.append(Episode(f"t{i}", 1.0, 0.0 if needs_strong else 1.0,
                               40.0 if needs_strong else 1.0, 0.1,
                               score=1.0 if needs_strong else 0.0))
        text = re_.format_report(re_.evaluate(eps, resamples=20))
        assert "dollars" in text.lower()


class TestLoading:
    def test_reads_a_well_formed_file(self, tmp_path):
        p = tmp_path / "eps.jsonl"
        p.write_text(
            '{"key":"a","q_strong":1,"q_weak":0,"score":0.9}\n'
            "# a comment\n"
            "\n"
            '{"key":"b","q_strong":1,"q_weak":1,"score":0.1,"split":"test"}\n')
        eps = re_.load_episodes(p)
        assert [e.key for e in eps] == ["a", "b"]
        assert eps[1].split == "test"

    def test_a_malformed_line_stops_the_run_and_names_itself(self, tmp_path):
        """Skipping the bad line silently is how you score a router on 8 of 400."""
        p = tmp_path / "eps.jsonl"
        p.write_text('{"key":"a","q_strong":1,"q_weak":0,"score":1}\nnot json\n')
        with pytest.raises(ValueError, match=r":2:"):
            re_.load_episodes(p)

    def test_a_missing_field_is_named(self, tmp_path):
        p = tmp_path / "eps.jsonl"
        p.write_text('{"key":"a","q_strong":1}\n')
        with pytest.raises(ValueError, match="q_weak"):
            re_.load_episodes(p)

    def test_split_selects_a_subset(self):
        eps = [Episode("a", 1.0, 0.0, 1.0, 0.1, 1.0, split="train"),
               Episode("b", 1.0, 0.0, 1.0, 0.1, 1.0, split="test")]
        assert [e.key for e in re_.split(eps, "test")] == ["b"]


class TestFromOutcomes:
    @staticmethod
    def _row(tier="T0", escalated=False, ctx=20_000, cost=0.01, key="k"):
        from adder.decide.track.outcomes import Outcome

        return Outcome(tier=tier, model="m", project="p", escalated=escalated,
                       context_tokens=ctx, cost=cost, task_hash=key)

    def test_an_escalation_is_a_task_the_weak_arm_failed(self):
        eps = re_.episodes_from_outcomes([self._row(escalated=True)])
        assert eps[0].q_weak == 0.0
        assert eps[0].q_strong == 1.0
        assert eps[0].modelled is True

    def test_a_held_task_is_a_task_the_weak_arm_won(self):
        eps = re_.episodes_from_outcomes([self._row(escalated=False)])
        assert eps[0].q_weak == 1.0

    def test_rows_at_or_above_the_strong_reference_are_excluded(self):
        """There is no cheaper arm to compare against, so there is no episode."""
        assert re_.episodes_from_outcomes([self._row(tier="T2")]) == []
        assert re_.episodes_from_outcomes([self._row(tier="T3")]) == []

    def test_an_unknown_tier_is_dropped_not_guessed(self):
        assert re_.episodes_from_outcomes([self._row(tier="T9")]) == []

    def test_the_strong_arm_costs_more_than_the_weak_one(self):
        eps = re_.episodes_from_outcomes([self._row(tier="T0", ctx=100_000, cost=0.0)])
        assert eps[0].cost_strong > eps[0].cost_weak

    def test_the_tier_is_the_router_ranking_signal(self):
        eps = re_.episodes_from_outcomes(
            [self._row(tier="T0", key="a"), self._row(tier="T1", key="b")])
        by_key = {e.key: e.score for e in eps}
        assert by_key["b"] > by_key["a"]

    def test_context_breaks_ties_inside_a_tier(self):
        eps = re_.episodes_from_outcomes(
            [self._row(tier="T1", ctx=10_000, key="small"),
             self._row(tier="T1", ctx=400_000, key="big")])
        by_key = {e.key: e.score for e in eps}
        assert by_key["big"] > by_key["small"]


class TestCli:
    def test_no_log_and_no_file_exits_one_with_output(self, capsys, isolated_home):
        assert re_.main([]) == 1
        assert capsys.readouterr().out.strip()

    def test_no_log_still_emits_valid_json(self, capsys, isolated_home):
        assert re_.main(["--json"]) == 1
        json.loads(capsys.readouterr().out)

    def test_a_missing_file_is_an_error_not_a_traceback(self, capsys, tmp_path):
        assert re_.main([str(tmp_path / "nope.jsonl")]) == 1

    def test_bad_targets_are_a_usage_error(self, capsys, tmp_path):
        p = tmp_path / "e.jsonl"
        p.write_text('{"q_strong":1,"q_weak":0,"score":1}\n')
        assert re_.main([str(p), "--targets", "fifty"]) == 2

    def test_scores_a_real_file_end_to_end(self, capsys, tmp_path):
        p = tmp_path / "e.jsonl"
        p.write_text("\n".join(
            json.dumps({"key": f"t{i}", "q_strong": 1, "q_weak": 0 if i % 2 == 0 else 1,
                        "cost_strong": 1.0, "cost_weak": 0.1,
                        "score": 1.0 if i % 2 == 0 else 0.0})
            for i in range(20)))
        assert re_.main([str(p), "--resamples", "20", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["apgr"]["calls"] > 0.5


class TestDegenerateInputs:
    def test_a_single_episode_does_not_divide_by_zero(self):
        rep = re_.evaluate([Episode("only", 1.0, 0.0, 1.0, 0.1, 0.5)], resamples=10)
        assert rep.n == 1
        assert 0.0 <= rep.apgr_calls <= 1.0
        json.dumps(rep.to_json())

    def test_free_arms_do_not_break_the_cost_axis(self):
        """Both arms priced at zero: the cost fraction has no denominator."""
        eps = [Episode(f"t{i}", 1.0, float(i % 2), 0.0, 0.0, score=float(i))
               for i in range(10)]
        pts = re_.curve(eps)
        assert all(p.cost_fraction == 0.0 for p in pts)
        assert 0.0 <= re_.apgr(pts, "cost") <= 1.0

    def test_negative_quality_scores_are_handled(self):
        """Some graders score -1..1; PGR only needs a gap, not a range."""
        eps = [Episode(f"t{i}", 1.0, -1.0 if i % 2 == 0 else 1.0, 1.0, 0.1,
                       score=1.0 if i % 2 == 0 else 0.0) for i in range(10)]
        assert re_.apgr(re_.curve(eps)) > 0.5


class TestAFlatCostAxisIsNotAPerfectScore:
    """`separable` refuses an APGR of 1.0 from a collapsed QUALITY axis. The
    cost axis has the identical degeneracy and had no guard.

    The cost axis is normalised by `strong_total - weak_total`. When that is
    not positive every point collapses to a `cost_fraction` of 0, `_pgr_at`
    walks to the last point, and the report says APGR 1.000 with CPT(80%) at
    0% of budget -- "recover 80% of the gap for nothing".

    It is reachable on the default path: `episodes_from_outcomes` prices the
    weak arm at its RECORDED cost, session context included, against a
    MODELLED cold run for the strong arm.
    """

    def _flat(self):
        from adder.evaluate.replay.routereval import Episode

        return [Episode(key=f"e{i}", q_strong=1.0, q_weak=0.0 if i % 2 else 1.0,
                        cost_strong=1.0, cost_weak=2.0, score=float(i))
                for i in range(10)]

    def _separable(self):
        from adder.evaluate.replay.routereval import Episode

        return [Episode(key=f"e{i}", q_strong=1.0, q_weak=0.0 if i % 2 else 1.0,
                        cost_strong=3.0, cost_weak=1.0, score=float(i))
                for i in range(10)]

    def test_a_flat_axis_is_flagged(self):
        from adder.evaluate.replay.routereval import evaluate

        assert evaluate(self._flat(), resamples=20).cost_separable is False

    def test_a_real_cost_gap_is_not_flagged(self):
        from adder.evaluate.replay.routereval import evaluate

        assert evaluate(self._separable(), resamples=20).cost_separable is True

    def test_the_json_reports_null_rather_than_a_perfect_score(self):
        from adder.evaluate.replay.routereval import evaluate

        d = evaluate(self._flat(), resamples=20).to_json()
        assert d["apgr"]["cost"] is None
        assert d["cpt_cost"] is None
        assert d["cost_separable"] is False

    def test_the_text_report_says_so_instead_of_quoting_a_number(self):
        from adder.evaluate.replay.routereval import evaluate, format_report

        text = format_report(evaluate(self._flat(), resamples=20))
        assert "n/a" in text and "cost axis is flat" in text
        assert "1.000" not in text.split("APGR (cost axis)")[1][:40]

    def test_the_calls_axis_is_untouched(self):
        from adder.evaluate.replay.routereval import evaluate

        rep = evaluate(self._flat(), resamples=20)
        assert 0.0 < rep.apgr_calls < 1.0
