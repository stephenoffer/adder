"""The session reports are read-only, and the `cache` setting reaches them.

Two defects, one theme. `adder horizon` fitted through `horizon.load`, the
hooks' cached entry point, so a report on CLAUDE.md's read-only list wrote
`.adder-horizon.json` on every run -- including with `cache` set to false.
And `sched`, `speed` and `spec` passed `use_cache=True` to the transcript
reader outright, so turning the parse cache off did nothing for them.

Each test counts the files that appear under an isolated HOME, so it asserts
the behaviour a user would see rather than which argument was passed.
"""
from __future__ import annotations

import json

import pytest


def _write_corpus(root, *, sessions: int = 2, turns: int = 3) -> None:
    """Short sessions with a tool call each, enough for every report to load."""
    for s in range(sessions):
        lines = []
        for i in range(turns):
            lines.append({
                "type": "assistant", "sessionId": f"s{s}",
                "timestamp": f"2026-08-0{s + 1}T10:{i:02d}:00Z",
                "message": {
                    "id": f"m{s}-{i}", "model": "claude-opus-5",
                    "usage": {"input_tokens": 5, "cache_read_input_tokens": 1_000 * (i + 1),
                              "output_tokens": 50},
                    "content": [{"type": "tool_use", "id": f"t{s}-{i}", "name": "Read",
                                 "input": {"file_path": f"/x/{i}.py"}}]}})
        d = root / f"-w-{s}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"s{s}.jsonl").write_text("\n".join(json.dumps(r) for r in lines))


def _written(home) -> set[str]:
    return {p.name for p in home.rglob("*") if p.is_file()} - {"adder.json"}


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "projects"
    _write_corpus(root)
    return root


@pytest.fixture
def long_corpus(tmp_path):
    """Sessions past the horizon's 5-turn floor: the old code stored a fit only
    when it had lengths, so a corpus of short sessions could not show the bug."""
    root = tmp_path / "long"
    _write_corpus(root, sessions=2, turns=6)
    return root


def _cache(home, on: bool) -> None:
    (home / "adder.json").write_text(json.dumps({"cache": on}))


class TestHorizonReportIsReadOnly:
    def test_never_writes_the_horizon_cache(self, isolated_home, long_corpus, capsys):
        from adder.measure.session.horizon import main

        _cache(isolated_home, True)
        assert main([str(long_corpus)]) == 0
        assert ".adder-horizon.json" not in _written(isolated_home)

    def test_cache_false_writes_nothing_at_all(self, isolated_home, long_corpus,
                                               capsys):
        from adder.measure.session.horizon import main

        _cache(isolated_home, False)
        assert main([str(long_corpus), "--json"]) == 0
        assert _written(isolated_home) == set()

    def test_hooks_entry_point_still_caches(self, isolated_home, long_corpus):
        """`load` is where the cache belongs; the report fix must not remove it."""
        from adder.measure.session.horizon import cache_path, load

        load(long_corpus)
        assert cache_path().is_file()


class TestHorizonSaysWhyItIsThePrior:
    def test_short_sessions_are_not_reported_as_none_observed(self, isolated_home,
                                                              corpus, capsys):
        """Two 3-turn sessions printed "0 sessions observed"."""
        from adder.measure.session.horizon import main

        assert main([str(corpus)]) == 0
        out = capsys.readouterr().out
        assert "0 sessions observed" not in out
        assert "too few sessions (n=0 of 2 on record, need >=5 of >=5 turns)" in out

    def test_json_carries_the_count_seen(self, isolated_home, corpus, capsys):
        from adder.measure.session.horizon import main

        assert main([str(corpus), "--json"]) == 0
        blob = json.loads(capsys.readouterr().out)
        assert (blob["sessions"], blob["observed"], blob["min_turns"]) == (0, 2, 5)


@pytest.mark.parametrize("module,extra", [
    ("adder.measure.session.sched", ["--resamples", "5"]),
    ("adder.measure.session.speed", []),
    ("adder.measure.session.speculation", ["--json"]),
])
class TestCacheSettingIsHonoured:
    def _run(self, module, extra, corpus):
        import importlib

        assert importlib.import_module(module).main([str(corpus), *extra]) == 0

    def test_cache_false_leaves_no_parse_cache(self, module, extra, isolated_home,
                                               corpus, capsys):
        _cache(isolated_home, False)
        self._run(module, extra, corpus)
        assert not any("trace-cache" in n for n in _written(isolated_home))

    def test_cache_true_still_uses_it(self, module, extra, isolated_home, corpus,
                                      capsys):
        """The control: without it the test above passes on a report that never
        reads a transcript at all."""
        _cache(isolated_home, True)
        self._run(module, extra, corpus)
        assert any("trace-cache" in n for n in _written(isolated_home))
