"""Spend-report labels that said something other than what the number was.

Each test names the old text and pins the corrected one, because each of these
printed a correct figure under a heading that made it wrong: the whole
accumulated pool credited to verbosity, input+output labelled as input, and
sub-dollar spend rounded to "$0".
"""
from __future__ import annotations

import json


class TestDebt:
    def test_only_the_output_share_is_credited_to_verbosity(self, make_sessions):
        from adder.measure.spend.debt import report

        out = report(make_sessions(n=2, n_turns=30))
        assert "accumulated prior output" not in out
        pool = next(line for line in out.splitlines() if "accumulated context" in line)
        assert "what verbosity controls" not in pool
        share = next(line for line in out.splitlines() if "attributable to output" in line)
        assert "what verbosity controls" in share


class TestLimits:
    def test_the_token_column_is_not_called_input(self, make_session, tz):
        """`Block.tokens` is `total_tokens`, which includes output."""
        from adder.measure.spend.limits import build, render

        tz("UTC")
        s = make_session(n_turns=10)
        rep = build({"s": s})
        out = render(rep)
        b = rep.blocks[0]
        assert b.tokens == sum(t.context + t.out for t in s.turns)
        assert "`read` counts every token the model had to take in" not in out
        assert "`tokens` is input plus output" in out
        assert f"{b.tokens:,}" in out


class TestTrace:
    def _root(self, write_jsonl, tmp_path):
        # 1,000 cache-read tokens and 10 output on Opus: a fraction of a cent.
        return write_jsonl([
            {"type": "assistant", "sessionId": "s", "timestamp": "2026-08-01T10:00:00Z",
             "message": {"id": "m1", "model": "claude-opus-5",
                         "usage": {"input_tokens": 1, "cache_read_input_tokens": 1_000,
                                   "output_tokens": 10}, "content": []}}],
            into=tmp_path / "t")

    def test_sub_dollar_sessions_do_not_print_as_zero(self, write_jsonl, tmp_path,
                                                      isolated_home, capsys):
        from adder.measure.spend.trace import main

        assert main([str(self._root(write_jsonl, tmp_path)), "--no-cache"]) == 0
        out = capsys.readouterr().out
        # $0.000755 of spend: `,.0f` printed both of these as "$0".
        ranked = out.split("most expensive sessions:")[1].splitlines()[1]
        assert "$0.0008" in ranked
        assert "top 25% of sessions = $0.0008" in out

    def test_json_totals_carry_four_decimals_like_sessions(self, write_jsonl,
                                                           tmp_path, isolated_home,
                                                           capsys):
        from adder.measure.spend.sessions import main as sessions_main
        from adder.measure.spend.trace import main

        root = str(self._root(write_jsonl, tmp_path))
        assert main([root, "--json", "--no-cache"]) == 0
        total = json.loads(capsys.readouterr().out)["total"]
        assert sessions_main([root, "--json"]) == 0
        assert total == json.loads(capsys.readouterr().out)["total"]
        assert 0 < total < 0.01


class TestSessionsJsonOnEmpty:
    def test_empty_root_still_prints_json(self, tmp_path, isolated_home, capsys):
        from adder.measure.spend.sessions import main

        (tmp_path / "empty").mkdir()
        assert main([str(tmp_path / "empty"), "--json"]) == 1
        blob = json.loads(capsys.readouterr().out)
        assert blob["sessions"] == 0 and blob["rows"] == []
        assert blob["error"] == "no sessions"


class TestDebtMoney:
    def test_a_small_dataset_does_not_read_as_zero_dollars(self, make_sessions):
        from adder.measure.spend.debt import report

        out = report(make_sessions(n=1, n_turns=6, base=2_000, growth=100, out=20))
        gen = next(line for line in out.splitlines() if "generation cost" in line)
        assert "$0 " not in gen and "$" in gen


class TestAnomalyNamesItsThreshold:
    def test_the_threshold_used_is_the_one_printed(self, make_sessions):
        from adder.measure.spend.anomaly import report, scan

        out = report(scan(make_sessions(n=2, n_turns=30), turn_z=50.0))
        assert "more than 50 robust deviations" in out
        assert "3.5" not in out


class TestDebtRowsAreThisMachines:
    """The table printed 340, 759 and 1,854 -- the author's median, p90 and
    longest session -- on every machine as if they described it."""

    def test_with_enough_sessions_the_rows_are_measured(self, make_session):
        from adder.measure.spend.debt import report

        sessions = {f"s{i}": make_session(n, sid=f"s{i}")
                    for i, n in enumerate((10, 20, 30, 40, 50, 60))}
        out = report(sessions)
        assert "median session here" in out and "longest session here" in out
        assert "1,854" not in out
        assert any(line.split()[:1] == ["60"] for line in out.splitlines())

    def test_with_too_few_the_rows_say_they_are_illustrative(self, make_sessions):
        from adder.measure.spend.debt import report

        assert "illustrative" in report(make_sessions(n=2, n_turns=10))
