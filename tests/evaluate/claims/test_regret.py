"""`adder regret`: the estimator is judged on the decisions it drives.

CLAUDE.md requires a mirrored test file for every command and this one had
none, so two defects in it had nowhere to be caught: the cross-validation was
run over a length distribution the estimator was never fitted on, and an empty
corpus produced a confident winner.
"""

from __future__ import annotations

import json

import pytest

from adder.evaluate.claims import regret as mod


class TestResultBest:
    def test_a_tie_names_no_winner(self):
        """Every estimator at zero regret is no measurement, not a tie broken
        by dict insertion order."""
        r = mod.Result((1, 1, 1), {"empirical": 0.0, "countdown": 0.0, "flat": 0.0})
        assert r.best == ""
        assert r.spread == 0.0

    def test_a_real_difference_names_the_cheapest(self):
        r = mod.Result((1, 1, 1), {"empirical": 1.0, "countdown": 4.0, "flat": 9.0})
        assert r.best == "empirical"
        assert r.spread == pytest.approx(8.0)

    def test_no_scores_at_all(self):
        assert mod.Result((1, 1, 1)).best == ""


class TestEvaluate:
    def test_it_returns_one_row_per_scenario(self):
        rows = mod.evaluate(list(range(20, 40)), scenarios=mod.SCENARIOS[:2],
                            probes=(10, 30))
        assert len(rows) == 2
        assert all(set(r.regret) == {"empirical", "countdown", "flat"} for r in rows)

    def test_regret_is_never_negative(self):
        rows = mod.evaluate(list(range(20, 60)), scenarios=mod.SCENARIOS[:2],
                            probes=(10, 30))
        assert all(v >= -1e-9 for r in rows for v in r.regret.values())

    def test_no_probe_lands_inside_any_session(self):
        """Every held-out length is below every probe, so nothing is scored."""
        rows = mod.evaluate([6, 7, 8], scenarios=mod.SCENARIOS[:1], probes=(1000,))
        assert rows[0].spread == 0.0 and rows[0].best == ""


class TestReport:
    def test_a_thin_corpus_says_so_rather_than_ranking(self):
        text = mod.report([10, 20, 30])
        assert f">={mod.MIN_SESSIONS}" in text

    def test_a_real_corpus_ranks(self):
        text = mod.report(list(range(20, 80, 3)))
        assert "total regret" in text and "empirical" in text


class TestTheCommand:
    def test_lengths_come_from_the_main_chain(self, make_turn, tmp_path, monkeypatch):
        """`Horizon.from_sessions` counts main-chain turns and says why: a
        subagent turn does not re-read the main context. Cross-validating
        against a different distribution than the estimator was fitted on
        measures nothing."""
        from adder.core.trace import Session

        sessions = {}
        for k in range(mod.MIN_SESSIONS + 2):
            s = Session(f"s{k}", "p")
            s.turns = [make_turn(session=f"s{k}", minutes=i) for i in range(12)]
            # A delegation-heavy session: 90 subagent turns against 12 real ones.
            s.turns += [make_turn(session=f"s{k}", sidechain=True, minutes=50 + i)
                        for i in range(90)]
            sessions[f"s{k}"] = s
        monkeypatch.setattr("adder.core.trace.load_sessions",
                            lambda *a, **k: sessions)
        captured = {}

        def _spy(lengths, **kw):
            captured["l"] = list(lengths)
            return []

        monkeypatch.setattr(mod, "evaluate", _spy)
        mod.main([str(tmp_path), "--json"])
        # 12, the conversation's own length -- not 102, the record count.
        assert captured["l"] == [12] * (mod.MIN_SESSIONS + 2)

    def test_an_empty_corpus_names_no_winner(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr("adder.core.trace.load_sessions", lambda *a, **k: {})
        assert mod.main([str(tmp_path), "--json"]) == 0
        d = json.loads(capsys.readouterr().out)
        assert d["sessions"] == 0 and d["enough_data"] is False
        assert d["scenarios"] == []

    def test_json_is_one_parseable_document(self, make_sessions, tmp_path,
                                            capsys, monkeypatch):
        monkeypatch.setattr("adder.core.trace.load_sessions",
                            lambda *a, **k: make_sessions(12, 40))
        assert mod.main([str(tmp_path), "--json"]) == 0
        d = json.loads(capsys.readouterr().out)
        assert d["enough_data"] is True and d["scenarios"]
        assert all(s["best"] in ("", "empirical", "countdown", "flat")
                   for s in d["scenarios"])
