"""`adder hook NAME`, the portable spelling of a hook command.

The thing under test is not really the dispatch -- it is three lines. It is the
promise that a stale or misspelt entry in somebody's committed settings.json
cannot break their session, which is the same promise every other file in this
subtree makes about exceptions and makes here about names.
"""

from __future__ import annotations

import importlib

from adder.decide.auto import HOOKS
from adder.decide.hooks.run import HOOKS as NAMES
from adder.decide.hooks.run import main


class TestNames:
    def test_every_name_auto_writes_is_dispatchable(self):
        """`auto` writes `adder hook <name>`; a name it writes that this does
        not know is an install that fails on every tool call."""
        assert {h['name'] for h in HOOKS} <= set(NAMES)

    def test_every_target_module_imports_and_has_a_main(self):
        for module in NAMES.values():
            assert callable(importlib.import_module(module).main)

    def test_the_module_a_name_maps_to_is_the_one_auto_installs(self):
        for h in HOOKS:
            assert NAMES[h['name']] == h['module']


class TestFailingOpen:
    def test_an_unknown_name_is_not_an_error(self, capsys):
        """A settings.json outlives the version that wrote it. Exiting non-zero
        on a name we have since renamed would turn that into a failure on every
        single tool call, for a component whose entire contract is to fail open.
        """
        assert main(['no-such-hook']) == 0
        assert 'unknown hook' in capsys.readouterr().err

    def test_nothing_reaches_stdout_on_an_unknown_name(self, capsys):
        """stdout is the hook's protocol channel; anything printed there is
        parsed as a decision. Diagnostics go to stderr, where Claude Code shows
        them and the model's context never does."""
        main(['no-such-hook'])
        assert capsys.readouterr().out == ''

    def test_no_arguments_is_a_usage_error_rather_than_a_traceback(self, capsys):
        assert main([]) == 2
        assert 'usage' in capsys.readouterr().err.lower()

    def test_help_lists_every_hook(self, capsys):
        # argparse raises SystemExit(0) for `--help`, like every other command
        # in the table; `tests/cli/test_dispatch.py` asserts exactly that.
        import pytest

        with pytest.raises(SystemExit) as e:
            main(['--help'])
        assert e.value.code == 0
        out = capsys.readouterr().out
        assert all(n in out for n in NAMES)


class TestDispatch:
    def test_it_runs_the_named_hook(self, monkeypatch, capsys):
        import adder.decide.hooks.precompact_learn as learner

        monkeypatch.setattr(learner, 'main', lambda: 0)
        assert main(['compact-learn']) == 0

    def test_a_hook_that_raises_still_returns_zero(self, monkeypatch, capsys):
        """The hook modules swallow their own exceptions; this is the belt for
        the one that gets past them -- an ImportError from a half-installed
        environment, say. A PreToolUse hook that exits non-zero is a broken
        turn."""
        import adder.decide.hooks.precompact_learn as learner

        def boom():
            raise RuntimeError('half-installed')

        monkeypatch.setattr(learner, 'main', boom)
        assert main(['compact-learn']) == 0
        assert 'half-installed' in capsys.readouterr().err
