"""Effort re-fitting, and the refusals that keep it honest.

The two tests that matter are the ones that decline to fit: a single observed
level, and a level with too few turns. Both currently describe the author's
own machine, and both are the cases where an eager fit would put an invented
multiplier in front of a dollar figure.
"""

from __future__ import annotations

import json

from adder.core.trace import Session
from adder.measure.session.effort import BASE_LEVEL, MIN_TURNS, fit, report
from adder.pricing.cost import EFFORT_OUTPUT_MULT


def _sess(make_turn, spec: dict[str, tuple[int, int]]) -> dict[str, Session]:
    """`{effort: (n_turns, out_tokens_each)}` as one session."""
    s = Session("s", "p")
    for level, (n, out) in spec.items():
        for i in range(n):
            s.turns.append(make_turn(effort=level, out=out, minutes=i))
    return {"s": s}


class TestFit:
    def test_no_labels_means_no_levels(self, make_turn):
        sessions = _sess(make_turn, {"": (10, 500)})
        f = fit(sessions)
        assert f.levels == {}
        assert f.unlabelled == 10
        assert f.measured is False

    def test_one_level_is_not_a_fit(self, make_turn):
        f = fit(_sess(make_turn, {"high": (200, 500)}))
        assert f.measured is False
        assert f.multipliers() == EFFORT_OUTPUT_MULT

    def test_a_thin_level_is_refused(self, make_turn):
        f = fit(_sess(make_turn, {"high": (200, 500),
                                  "low": (MIN_TURNS - 1, 100)}))
        assert "low" not in f.fittable
        assert f.multipliers()["low"] == EFFORT_OUTPUT_MULT["low"]

    def test_a_level_with_enough_turns_is_fitted(self, make_turn):
        f = fit(_sess(make_turn, {"high": (200, 1000), "low": (200, 250)}))
        assert f.measured is True
        assert f.multipliers()["low"] == 0.25
        assert f.multipliers()[BASE_LEVEL] == 1.0

    def test_the_base_level_itself_needs_enough_data(self, make_turn):
        f = fit(_sess(make_turn, {"high": (3, 1000), "low": (200, 250)}))
        assert f.base is None
        assert f.measured is False
        assert f.multipliers() == EFFORT_OUTPUT_MULT

    def test_unfitted_levels_keep_their_priors(self, make_turn):
        f = fit(_sess(make_turn, {"high": (200, 1000), "low": (200, 250)}))
        m = f.multipliers()
        assert m["xhigh"] == EFFORT_OUTPUT_MULT["xhigh"]
        assert m["max"] == EFFORT_OUTPUT_MULT["max"]

    def test_multipliers_always_cover_every_known_level(self, make_turn):
        m = fit(_sess(make_turn, {"high": (200, 1000)})).multipliers()
        assert set(m) >= set(EFFORT_OUTPUT_MULT)

    def test_drift_reports_prior_against_measured(self, make_turn):
        f = fit(_sess(make_turn, {"high": (200, 1000), "medium": (200, 900)}))
        prior, measured = f.drift()["medium"]
        assert prior == EFFORT_OUTPUT_MULT["medium"]
        assert measured == 0.9

    def test_an_unknown_level_is_recorded_but_has_no_prior(self, make_turn):
        f = fit(_sess(make_turn, {"high": (200, 1000), "ludicrous": (200, 4000)}))
        assert "ludicrous" in f.levels
        assert f.multipliers()["ludicrous"] == 4.0


class TestLevelStats:
    def test_thinking_share(self, make_turn):
        s = Session("s", "p")
        s.turns = [make_turn(effort="high", out=1000, thinking=250)]
        assert fit({"s": s}).levels["high"].thinking_share == 0.25

    def test_empty_level_does_not_divide_by_zero(self):
        from adder.measure.session.effort import Level

        lv = Level("high")
        assert lv.mean_out == 0.0
        assert lv.thinking_share == 0.0
        assert lv.median_out == 0.0
        assert lv.enough is False


class TestReport:
    def test_no_labels_says_the_priors_stand(self, make_turn):
        text = report(fit(_sess(make_turn, {"": (5, 100)})))
        assert "MODELLED" in text

    def test_one_level_explains_why_there_is_no_fit(self, make_turn):
        text = report(fit(_sess(make_turn, {"high": (200, 500)})))
        assert "only one effort level" in text

    def test_a_real_fit_reports_the_drift(self, make_turn):
        text = report(fit(_sess(make_turn, {"high": (200, 1000), "low": (200, 250)})))
        assert "Fitted multipliers" in text
        assert "upper bound" in text

    def test_step_down_table_appears_with_sessions(self, make_turn):
        sessions = _sess(make_turn, {"high": (200, 1000)})
        text = report(fit(sessions), sessions=sessions)
        assert "high → medium" in text


class TestCli:
    def test_json(self, tmp_path, write_jsonl, capsys):
        from adder.measure.session.effort import main

        write_jsonl([{"type": "assistant", "sessionId": "s", "effort": "high",
                      "timestamp": "2026-08-01T10:00:00Z",
                      "message": {"id": "m1", "model": "claude-opus-5",
                                  "usage": {"input_tokens": 1,
                                            "cache_read_input_tokens": 900,
                                            "output_tokens": 100}, "content": []}}])
        assert main([str(tmp_path), "--json"]) == 0
        d = json.loads(capsys.readouterr().out)
        assert d["levels"]["high"]["turns"] == 1
        assert d["measured"] is False
        assert d["priors"] == EFFORT_OUTPUT_MULT

    def test_effort_is_read_off_the_record(self, tmp_path, write_jsonl):
        from adder.core.trace import iter_file

        write_jsonl([{"type": "assistant", "sessionId": "s", "effort": "xhigh",
                      "timestamp": "2026-08-01T10:00:00Z",
                      "message": {"id": "m1", "model": "claude-opus-5",
                                  "usage": {"input_tokens": 1,
                                            "output_tokens": 100}, "content": []}}])
        assert next(iter_file(tmp_path / "s.jsonl")).effort == "xhigh"
