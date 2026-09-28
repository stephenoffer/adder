"""Numeric flags are validated at parse time, not trusted into a report.

Every case below used to run and print an answer: a zero-hour window, a
negative budget, a negative `--top` that silently dropped the tail of the
ranking, a negative z threshold that flagged every turn, and a negative turn
index. Argparse rejects each now with its usual usage line and exit status 2.
"""
from __future__ import annotations

import argparse
import importlib

import pytest

from adder.measure.argtypes import nonneg_float, nonneg_int, positive_float, positive_int


class TestTypes:
    @pytest.mark.parametrize("fn,good,bad", [
        (positive_int, ["1", "20"], ["0", "-1", "1.5", "x"]),
        (nonneg_int, ["0", "400"], ["-5", "x"]),
        (positive_float, ["0.5", "5"], ["0", "-1", "nan", "inf", "x"]),
        (nonneg_float, ["0", "12.5"], ["-5", "nan", "-inf", "x"]),
    ])
    def test_accepts_and_rejects(self, fn, good, bad):
        for text in good:
            assert fn(text) == float(text)
        for text in bad:
            with pytest.raises(argparse.ArgumentTypeError):
                fn(text)


@pytest.mark.parametrize("module,argv", [
    ("adder.measure.spend.limits", ["--hours", "0"]),
    ("adder.measure.spend.limits", ["--hours", "-1"]),
    ("adder.measure.spend.budget", ["--limit", "-5"]),
    ("adder.measure.spend.sessions", ["--top", "-1"]),
    ("adder.measure.spend.anomaly", ["--z", "-1"]),
    ("adder.measure.spend.anomaly", ["--top", "0"]),
    ("adder.measure.session.horizon", ["--at", "-5"]),
])
def test_commands_reject_nonsense(module, argv, tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        importlib.import_module(module).main([str(tmp_path), *argv])
    assert e.value.code == 2
    assert "error: argument" in capsys.readouterr().err


def test_budget_zero_still_means_no_budget(tmp_path, capsys):
    """Zero is the documented "disabled" value, so it must still parse."""
    from adder.measure.spend.budget import main

    (tmp_path / "t").mkdir()
    main([str(tmp_path / "t"), "--limit", "0"])        # no SystemExit


@pytest.mark.parametrize("command", [
    "tools", "reread", "compact", "memory", "agents", "spec", "trace",
    "handoff", "similar", "place", "frontier", "verbosity", "design"])
def test_every_top_flag_refuses_a_negative(command, capsys, isolated_home):
    """Only `sessions` and `anomaly` validated `--top`; the rest sliced the
    ranking with a negative index and printed the tail as "the top"."""
    from adder.cli import main

    with pytest.raises(SystemExit) as e:
        main([command, "--top", "-1"])
    assert e.value.code == 2


def test_trace_top_zero_still_means_every_row(tmp_path, capsys, isolated_home):
    from adder.measure.spend.trace import main

    main([str(tmp_path), "--top", "0"])                 # parses; 0 is "all"
