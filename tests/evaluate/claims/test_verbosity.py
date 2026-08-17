"""The verbosity report, pinned on refusing to claim what it cannot separate.

The failure mode here is a confident coefficient table produced from a log where
style never varied within a matchup — style and skill are collinear there, and
any number printed is an artefact of the ridge penalty rather than a finding.
"""

from __future__ import annotations

import json

import pytest

from adder.evaluate.claims import verbosity as vb
from adder.pricing.style import Style


def _rows(n=400, *, seed=3, vary=True):
    """Comparisons where the longer answer usually wins."""
    import math
    import random

    rng = random.Random(seed)
    out = []
    for _ in range(n):
        la = rng.choice([300, 900, 1800, 3200]) if vary else 2000
        lb = rng.choice([300, 900, 1800, 3200]) if vary else 400
        p = 1.0 / (1.0 + math.exp(-(0.9 * (math.log1p(la) - math.log1p(lb)))))
        out.append({
            "a": "wordy", "b": "brief",
            "winner": "a" if rng.random() < p else "b",
            "a_style": {"tokens": la, "headers": 2, "lists": 2, "bold": 1},
            "b_style": {"tokens": lb, "headers": 2, "lists": 2, "bold": 1},
        })
    return out


def _write(tmp_path, rows, name="b.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


class TestLoading:
    def test_it_reads_explicit_style_counts(self, tmp_path):
        battles, styles = vb.load(_write(tmp_path, _rows(5)))
        assert len(battles) == 5
        assert styles[0][0].tokens > 0

    def test_it_measures_style_from_response_text(self, tmp_path):
        p = _write(tmp_path, [{"a": "x", "b": "y", "winner": "a",
                               "a_text": "# H\n\n- one\n- two\n**b**",
                               "b_text": "short"}])
        _battles, styles = vb.load(p)
        assert styles[0][0].headers == 1
        assert styles[0][0].tokens > styles[0][1].tokens

    def test_a_row_with_neither_text_nor_counts_has_no_style(self, tmp_path):
        p = _write(tmp_path, [{"a": "x", "b": "y", "winner": "tie"}])
        _battles, styles = vb.load(p)
        assert styles[0][0] == Style()

    def test_comments_and_blanks_are_skipped(self, tmp_path):
        p = tmp_path / "b.jsonl"
        p.write_text('# note\n\n{"a":"x","b":"y","winner":"a"}\n', encoding="utf-8")
        assert len(vb.load(p)[0]) == 1

    def test_a_malformed_line_names_itself(self, tmp_path):
        p = tmp_path / "b.jsonl"
        p.write_text('{"a":"x","b":"y"}\nnope\n', encoding="utf-8")
        with pytest.raises(ValueError, match=r":2:"):
            vb.load(p)

    def test_a_missing_model_is_named(self, tmp_path):
        p = tmp_path / "b.jsonl"
        p.write_text('{"a":"x"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="b"):
            vb.load(p)


class TestPerModelStyle:
    def test_it_averages_each_model_responses(self, tmp_path):
        battles, styles = vb.load(_write(tmp_path, _rows(200)))
        means = vb.per_model_style(battles, styles)
        assert set(means) == {"wordy", "brief"}
        assert means["wordy"].tokens > 0

    def test_an_empty_log_has_no_models(self):
        assert vb.per_model_style([], []) == {}


class TestReport:
    def test_it_prices_the_premium_per_answer(self, tmp_path):
        battles, styles = vb.load(_write(tmp_path, _rows(400)))
        text = vb.report(battles, styles, resamples=15)
        assert "$/answer" in text
        assert "controlled" in text

    def test_it_refuses_to_report_an_unidentified_fit(self, tmp_path):
        """Style constant within a matchup: nothing can be separated."""
        battles, styles = vb.load(_write(tmp_path, _rows(200, vary=False)))
        text = vb.report(battles, styles, resamples=10)
        assert "collinear" in text
        assert "$/answer" not in text

    def test_it_always_states_the_observational_caveat(self, tmp_path):
        battles, styles = vb.load(_write(tmp_path, _rows(300)))
        assert "OBSERVATIONAL" in vb.report(battles, styles, resamples=10)

    def test_an_empty_log_says_so(self):
        assert "No comparisons" in vb.report([], [])

    def test_a_longer_horizon_raises_the_priced_premium(self, tmp_path):
        """The carry half grows with the turns that remain."""
        battles, styles = vb.load(_write(tmp_path, _rows(300)))
        short = vb.report(battles, styles, remaining_turns=1, resamples=10)
        long = vb.report(battles, styles, remaining_turns=500, resamples=10)
        assert short != long


class TestCli:
    def test_no_log_exits_one_with_output(self, capsys, isolated_home):
        assert vb.main([]) == 1
        assert capsys.readouterr().out.strip()

    def test_json_parses_with_no_log(self, capsys, isolated_home):
        vb.main(["--json"])
        json.loads(capsys.readouterr().out)

    def test_a_missing_file_is_an_error(self, tmp_path, capsys):
        assert vb.main([str(tmp_path / "nope.jsonl")]) == 1

    def test_a_malformed_file_is_a_usage_error(self, tmp_path, capsys):
        p = tmp_path / "b.jsonl"
        p.write_text("nope\n", encoding="utf-8")
        assert vb.main([str(p)]) == 2

    def test_it_fits_a_real_log(self, tmp_path, capsys, isolated_home):
        p = _write(tmp_path, _rows(300))
        assert vb.main([str(p), "--resamples", "10", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["identified"] is True
        assert "mean_style" in payload
        assert payload["beta"]["length"] > 0.3
