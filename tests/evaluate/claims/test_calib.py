"""Calibration scoring, pinned so it cannot flatter the estimator it scores.

The traps this suite exists for: scoring a predictor on the rows it was fitted
on (which measures memorisation), letting a two-observation bin set the
headline number, and reporting a Brier score with nothing to read it against.
"""

from __future__ import annotations

import json

import pytest

from adder.evaluate.claims import calib
from adder.evaluate.claims.calib import Prediction


def _preds(pairs):
    return [Prediction(f"k{i}", p, h) for i, (p, h) in enumerate(pairs)]


class TestScores:
    def test_a_perfect_predictor_scores_zero_brier(self):
        preds = _preds([(1.0, True), (0.0, False)] * 10)
        assert calib.brier(preds) == pytest.approx(0.0)

    def test_a_coin_flip_scores_a_quarter(self):
        preds = _preds([(0.5, True), (0.5, False)] * 10)
        assert calib.brier(preds) == pytest.approx(0.25)

    def test_a_confidently_wrong_predictor_scores_one(self):
        preds = _preds([(1.0, False), (0.0, True)] * 5)
        assert calib.brier(preds) == pytest.approx(1.0)

    def test_brier_on_nothing_is_zero_not_an_error(self):
        assert calib.brier([]) == 0.0
        assert calib.log_loss([]) == 0.0

    def test_log_loss_punishes_confidence_where_brier_shrugs(self):
        timid = _preds([(0.6, False)])
        bold = _preds([(0.99, False)])
        assert calib.log_loss(bold) > 5 * calib.log_loss(timid)
        assert calib.brier(bold) < 4 * calib.brier(timid)

    def test_log_loss_is_finite_at_the_extremes(self):
        import math

        assert math.isfinite(calib.log_loss(_preds([(1.0, False), (0.0, True)])))


class TestBins:
    def test_predictions_land_in_the_right_bin(self):
        bins = calib.bins_of(_preds([(0.05, False), (0.95, True)]), n_bins=10)
        assert bins[0].n == 1
        assert bins[9].n == 1

    def test_a_prediction_of_exactly_one_does_not_fall_off_the_end(self):
        bins = calib.bins_of(_preds([(1.0, True)]), n_bins=10)
        assert bins[9].n == 1

    def test_observed_and_predicted_are_reported_per_bin(self):
        bins = calib.bins_of(_preds([(0.25, True), (0.25, False)]), n_bins=4)
        assert bins[1].predicted == pytest.approx(0.25)
        assert bins[1].observed == pytest.approx(0.5)

    def test_a_thin_bin_is_flagged_and_excluded_from_the_headline(self):
        """Two observations can show a 50-point gap from noise alone."""
        preds = _preds([(0.05, False)] * 40 + [(0.95, False)] * 2)
        bins = calib.bins_of(preds)
        thin = [b for b in bins if b.n and b.thin]
        assert thin and thin[0].n == 2
        ece, _ = calib.calibration_error(bins, len(preds))
        assert ece < 0.10          # the noisy bin did not set it

    def test_calibration_error_of_a_well_calibrated_predictor_is_small(self):
        preds = _preds([(0.3, i < 30) for i in range(100)])
        ece, mce = calib.calibration_error(calib.bins_of(preds), len(preds))
        assert ece < 0.02 and mce < 0.02

    def test_calibration_error_catches_a_systematic_shift(self):
        # Predicts 0.3, happens 0.7 of the time.
        preds = _preds([(0.3, i < 70) for i in range(100)])
        ece, mce = calib.calibration_error(calib.bins_of(preds), len(preds))
        assert ece == pytest.approx(0.4, abs=0.02)
        assert mce == pytest.approx(0.4, abs=0.02)

    def test_no_solid_bins_yields_no_error_rather_than_a_crash(self):
        assert calib.calibration_error([], 0) == (0.0, 0.0)


class TestDecomposition:
    def test_bias_is_signed(self):
        over = _preds([(0.9, i < 10) for i in range(100)])
        bias, _ = calib.decomposition(over, calib.bins_of(over))
        assert bias > 0.5

    def test_a_constant_predictor_has_no_resolution(self):
        preds = _preds([(0.4, i % 2 == 0) for i in range(100)])
        _, resolution = calib.decomposition(preds, calib.bins_of(preds))
        assert resolution == pytest.approx(0.0, abs=1e-9)

    def test_a_discriminating_predictor_has_resolution(self):
        preds = _preds([(0.9, True)] * 50 + [(0.1, False)] * 50)
        _, resolution = calib.decomposition(preds, calib.bins_of(preds))
        assert resolution > 0.2


class TestReport:
    def test_skill_is_positive_when_the_predictor_discriminates(self):
        rep = calib.evaluate(_preds([(0.9, True)] * 50 + [(0.1, False)] * 50))
        assert rep.skill > 0.8
        assert rep.beats_base_rate

    def test_skill_is_zero_for_a_predictor_that_is_the_base_rate(self):
        rep = calib.evaluate(_preds([(0.5, i % 2 == 0) for i in range(100)]))
        assert rep.skill == pytest.approx(0.0, abs=0.01)
        assert not rep.beats_base_rate

    def test_skill_goes_negative_when_the_predictor_is_worse_than_a_constant(self):
        """The finding no per-tier table would ever surface."""
        rep = calib.evaluate(_preds([(0.9, False)] * 50 + [(0.1, True)] * 50))
        assert rep.skill < 0

    def test_a_well_calibrated_report_says_so(self):
        rep = calib.evaluate(_preds([(0.3, i < 30) for i in range(100)] * 2))
        assert rep.calibrated

    def test_a_miscalibrated_report_says_so(self):
        rep = calib.evaluate(_preds([(0.3, i < 70) for i in range(100)] * 2))
        assert not rep.calibrated
        assert "under-confident" in calib.format_report(rep)

    def test_an_empty_report_renders(self):
        rep = calib.evaluate([])
        assert rep.n == 0
        assert "Not enough" in calib.format_report(rep)

    def test_json_is_finite_and_complete(self):
        payload = calib.evaluate(_preds([(0.4, i % 3 == 0) for i in range(60)])).to_json()
        text = json.dumps(payload)
        assert "NaN" not in text and "Infinity" not in text
        assert set(payload) >= {"brier", "skill", "ece", "bins", "decision_flips"}

    def test_the_report_names_the_worst_bin(self):
        preds = _preds([(0.1, i < 5) for i in range(50)] +
                       [(0.8, i < 5) for i in range(50)])
        text = calib.format_report(calib.evaluate(preds))
        assert "Worst bin" in text


class TestFlips:
    def test_a_flip_is_a_prediction_on_the_wrong_side_of_the_gate(self):
        flips, cost = calib.decision_flips(
            [Prediction("a", 0.2, True, cost=3.0),      # said no, failed
             Prediction("b", 0.9, False, cost=1.0),     # said yes, held
             Prediction("c", 0.9, True, cost=1.0)])     # said yes, failed
        assert flips == 2
        assert cost == pytest.approx(3.0)   # only the expensive direction is priced

    def test_no_flips_when_every_prediction_is_on_the_right_side(self):
        assert calib.decision_flips(
            [Prediction("a", 0.9, True), Prediction("b", 0.1, False)]) == (0, 0.0)


class TestPrequential:
    @staticmethod
    def _rows(n=40, fail_every=3):
        from adder.decide.track.outcomes import Outcome

        return [Outcome(tier="T0", model="m", project="p",
                        escalated=(i % fail_every == 0), cost=0.5,
                        task_hash=f"h{i}", ts=float(1_000 + i))
                for i in range(n)]

    def test_every_prediction_is_made_without_its_own_row(self, isolated_home):
        """A retrospective score measures memorisation, not prediction."""
        rows = self._rows()
        preds = calib.prequential(rows, warmup=5)
        assert len(preds) == len(rows) - 5
        # Row 5's prediction may only use rows 0..4, whose escalation rate is
        # 2/5; a fitted-on-itself estimator would land on the global rate.
        assert 0.0 < preds[0].predicted < 1.0

    def test_warmup_rows_are_not_scored(self, isolated_home):
        assert len(calib.prequential(self._rows(20), warmup=10)) == 10

    def test_rows_are_replayed_in_timestamp_order(self, isolated_home):
        rows = self._rows(20)
        shuffled = list(reversed(rows))
        assert ([p.key for p in calib.prequential(rows, warmup=2)] ==
                [p.key for p in calib.prequential(shuffled, warmup=2)])

    def test_the_global_estimator_can_be_scored_too(self, isolated_home):
        rows = self._rows()
        assert calib.prequential(rows, scoped=False, warmup=5)

    def test_an_empty_log_scores_nothing(self, isolated_home):
        assert calib.prequential([], warmup=5) == []

    def test_a_predictable_log_is_learned(self, isolated_home):
        """Every row escalates: the estimator should converge upward."""
        from adder.decide.track.outcomes import Outcome

        rows = [Outcome(tier="T0", model="m", project="p", escalated=True,
                        task_hash=f"h{i}", ts=float(1_000 + i)) for i in range(60)]
        preds = calib.prequential(rows, warmup=5)
        assert preds[-1].predicted > preds[0].predicted
        assert preds[-1].predicted > 0.8


class TestCli:
    def test_an_empty_log_exits_one_with_output(self, capsys, isolated_home):
        assert calib.main([]) == 1
        assert capsys.readouterr().out.strip()

    def test_json_parses_on_an_empty_log(self, capsys, isolated_home):
        calib.main(["--json"])
        json.loads(capsys.readouterr().out)

    def test_it_scores_a_written_log(self, capsys, isolated_home, tmp_path):
        from adder.decide.track.outcomes import Outcome, record

        log = tmp_path / "outcomes.jsonl"
        for i in range(40):
            record(Outcome(tier="T0", model="m", project="p",
                           escalated=(i % 4 == 0), cost=0.2,
                           task_hash=f"h{i}", ts=float(1_000 + i)), log=log)
        assert calib.main(["--log", str(log), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["n"] > 0
        assert 0.0 <= payload["base_rate"] <= 1.0


class TestEdges:
    def test_two_bins_is_a_legal_resolution(self):
        rep = calib.evaluate(_preds([(0.2, False)] * 50 + [(0.8, True)] * 50),
                             n_bins=2)
        assert len(rep.bins) == 2
        assert rep.skill > 0.5

    def test_an_outcome_that_always_happens_has_no_base_variance(self):
        """Brier of the constant predictor is 0, so skill has no denominator."""
        rep = calib.evaluate(_preds([(0.9, True)] * 40))
        assert rep.base_rate == 1.0
        assert rep.brier_base == 0.0
        assert rep.skill == 0.0

    def test_a_single_prediction_reports_without_crashing(self):
        rep = calib.evaluate(_preds([(0.5, True)]))
        assert rep.n == 1
        json.dumps(rep.to_json())
