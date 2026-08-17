"""The instruction-file walks must terminate on a repository with a symlink.

Both walks bound their depth with `len(e.resolve().parts) - root_depth`. A
symlink pointing at an ancestor *resolves shorter*, so the bound is never
reached and the walk never ends: `adder memory` hangs, and so does anything
that calls it -- `adder doctor` among them. `SKIP_DIRS` only covers names
somebody thought of, and a self-referential `node_modules` link is routine.
"""

from __future__ import annotations

import os
import signal
from contextlib import contextmanager

import pytest

from adder.measure.window.memory import _nested_claude_md, _repo_paths


@contextmanager
def _within(seconds: int):
    """Fail rather than hang: the bug this file pins is an infinite loop."""
    def _bail(*_):
        raise AssertionError(f"the walk did not finish within {seconds}s")

    old = signal.signal(signal.SIGALRM, _bail)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


@pytest.fixture
def repo_with_a_loop(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "CLAUDE.md").write_text("nested", encoding="utf-8")
    (tmp_path / "sub" / "mod.py").write_text("x = 1", encoding="utf-8")
    os.symlink(str(tmp_path), str(tmp_path / "sub" / "loop"))
    return tmp_path


def test_nested_walk_terminates(repo_with_a_loop):
    with _within(20):
        got = _nested_claude_md(repo_with_a_loop)
    assert [p.name for p in got] == ["CLAUDE.md"]


def test_repo_path_walk_terminates(repo_with_a_loop):
    with _within(20):
        got = _repo_paths(repo_with_a_loop)
    assert "sub/mod.py" in got


def test_a_plain_repo_is_unaffected(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "CLAUDE.md").write_text("x", encoding="utf-8")
    assert [p.name for p in _nested_claude_md(tmp_path)] == ["CLAUDE.md"]
