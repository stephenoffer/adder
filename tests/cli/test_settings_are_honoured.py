"""A setting `adder config` reports must be the one the code actually uses.

`settings.py` opens by saying it exists because "a tool whose behaviour depends
on invisible state is a tool whose numbers cannot be reproduced by the person
reading them". Three of its own declarations broke that rule: `log`, `ledger`
and `trace_cache` were read from the environment at *import* time into a module
constant, so `.adder.json` set them, `adder config` printed them, and every
reader went on opening the file under `~/.claude`.
"""

from __future__ import annotations

import json

import pytest

from adder.core.settings import configured_path, get


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / ".adder.json").write_text(json.dumps({
        "log": str(tmp_path / "mine-outcomes.jsonl"),
        "ledger": str(tmp_path / "mine-ledger.jsonl"),
        "trace_cache": str(tmp_path / "mine-cache"),
    }), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    # The env layer wins over the file, so it must be clear for this test.
    for var in ("ADDER_LOG", "ADDER_LEDGER", "ADDER_TRACE_CACHE"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


class TestTheReportedValueIsTheUsedValue:
    def test_outcome_log(self, project):
        from adder.decide.track.outcomes import log_path

        assert str(log_path()) == get("log") == str(project / "mine-outcomes.jsonl")

    def test_ledger(self, project):
        from adder.decide.track.ledger import ledger_path

        assert str(ledger_path()) == get("ledger") == str(project / "mine-ledger.jsonl")

    def test_trace_cache(self, project):
        from adder.core.trace import cache_path

        assert str(cache_path()) == get("trace_cache") == str(project / "mine-cache")

    def test_an_explicit_argument_still_wins(self, project, tmp_path):
        from adder.decide.track.outcomes import log_path

        assert log_path(tmp_path / "elsewhere.jsonl").name == "elsewhere.jsonl"


class TestTheModuleConstantStillRedirects:
    """`isolated_home` and several tests point a log somewhere by patching the
    constant. Resolving the setting unconditionally would replace it with a
    value equal to the built-in default and read the developer's real files."""

    def test_an_unset_setting_falls_back_to_the_constant(self, tmp_path, monkeypatch):
        from pathlib import Path

        monkeypatch.chdir(tmp_path)          # no .adder.json here
        monkeypatch.delenv("ADDER_LOG", raising=False)
        sentinel = Path(tmp_path / "sentinel.jsonl")
        assert configured_path("log", sentinel) == sentinel

    def test_patching_the_constant_redirects_the_reader(self, tmp_path, monkeypatch):
        import adder.decide.track.outcomes as mod

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("ADDER_LOG", raising=False)
        monkeypatch.setattr(mod, "DEFAULT_LOG", tmp_path / "patched.jsonl")
        assert mod.log_path().name == "patched.jsonl"


class TestHomeIsReadWhenAsked:
    """A default built from `Path.home()` at import is one no test can move.

    This is not fastidiousness about isolation. `adder auto on` learns a size
    model into `~/.claude`, and the guard's own latency test then ran against
    the developer's real 40,902-call model -- so the population whose suite
    went red was exactly the population that had activated the tool. The
    fixture believed it had redirected the file; the constant had been built
    before the fixture ran.
    """

    def test_a_home_derived_default_follows_home(self, tmp_path, monkeypatch):
        from adder.core.settings import resolve

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        for name in ("size_model", "root", "guard_state", "home", "trace_cache",
                     "log", "ledger", "uptake_cache"):
            got = str(resolve(cwd=tmp_path, env={})[name].value)
            assert got.startswith(str(tmp_path)), f"{name} is pinned to import-time home"

    def test_the_size_model_fallback_follows_home_too(self, tmp_path, monkeypatch):
        from adder.core.shapes import default_model_path

        monkeypatch.setenv("HOME", str(tmp_path))
        assert default_model_path() == tmp_path / ".claude" / ".adder-sizes.json"

    def test_the_fixture_moves_the_learned_model(self, isolated_home):
        """The specific leak, asserted where it was missed."""
        from adder.core.shapes import model_path

        assert isolated_home in model_path().parents

    def test_the_fixture_leaves_no_learned_model_to_find(self, isolated_home):
        from adder.core.shapes import load_model

        assert load_model().calls == 0
