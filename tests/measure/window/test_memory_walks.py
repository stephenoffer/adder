"""The instruction-file walks must terminate on a repository with a symlink.

Both walks bound their depth with `len(e.resolve().parts) - root_depth`. A
symlink pointing at an ancestor *resolves shorter*, so the bound is never
reached and the walk never ends: `adder memory` hangs, and so does anything
that calls it -- `adder doctor` among them. `SKIP_DIRS` only covers names
somebody thought of, and a self-referential `node_modules` link is routine.
"""

from __future__ import annotations

import os
import threading

import pytest

from adder.measure.window.memory import _nested_claude_md, _repo_paths


def _within(seconds: int, call):
    """Run `call`, failing rather than hanging: this file pins an infinite loop.

    The watchdog is a thread rather than `signal.alarm` because `SIGALRM` and
    `signal.alarm` are Unix-only: on the Windows leg of the matrix this raised
    AttributeError before the walk was ever called, so the file asserted
    nothing there. A daemon thread cannot be cancelled, so a real hang leaks
    one; that is the right trade, because the assertion below has already
    failed and pytest exits without waiting for it.
    """
    done: list = []
    failed: list[Exception] = []

    def run():
        try:
            done.append(call())
        except Exception as exc:        # re-raised on the calling thread below
            failed.append(exc)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(seconds)
    assert not t.is_alive(), f"the walk did not finish within {seconds}s"
    if failed:
        raise failed[0]
    return done[0]


@pytest.fixture
def repo_with_a_loop(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "CLAUDE.md").write_text("nested", encoding="utf-8")
    (tmp_path / "sub" / "mod.py").write_text("x = 1", encoding="utf-8")
    try:
        os.symlink(str(tmp_path), str(tmp_path / "sub" / "loop"))
    except (OSError, NotImplementedError) as exc:
        # Windows needs Developer Mode or an elevated account to make one. The
        # loop is the whole subject here, so there is nothing left to assert.
        pytest.skip(f"cannot create a directory symlink here: {exc}")
    return tmp_path


def test_nested_walk_terminates(repo_with_a_loop):
    got = _within(20, lambda: _nested_claude_md(repo_with_a_loop))
    assert [p.name for p in got] == ["CLAUDE.md"]


def test_repo_path_walk_terminates(repo_with_a_loop):
    got = _within(20, lambda: _repo_paths(repo_with_a_loop))
    assert "sub/mod.py" in got


def test_a_plain_repo_is_unaffected(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "CLAUDE.md").write_text("x", encoding="utf-8")
    assert [p.name for p in _nested_claude_md(tmp_path)] == ["CLAUDE.md"]
