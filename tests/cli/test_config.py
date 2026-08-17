"""Settings resolution, including the precedence order and the failure modes.

None of these may touch the real `~/.claude/adder.json`: a developer with a
budget set in their own config would otherwise see different test results than
CI, which is the exact class of bug the config layer exists to prevent.
"""

from __future__ import annotations

import json

import pytest

from adder.cli import config
from adder.core import settings
from adder.core.settings import ConfigError, Resolved, project_file, resolve, template


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Point the user file at a directory that starts empty."""
    monkeypatch.setattr(settings, "USER_FILE", tmp_path / "user" / "adder.json")
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestPrecedence:
    def test_default_when_nothing_is_set(self):
        r = resolve(env={})
        assert r["budget"].value == 0.0
        assert r["budget"].source == "default"
        assert r["budget"].overridden is False

    def test_user_file_beats_default(self, isolated, monkeypatch):
        f = isolated / "user" / "adder.json"
        f.parent.mkdir(parents=True)
        f.write_text(json.dumps({"budget": 40.0}))
        monkeypatch.setattr(settings, "USER_FILE", f)
        r = resolve(env={})
        assert r["budget"].value == 40.0
        assert r["budget"].source == str(f)

    def test_project_file_beats_user_file(self, isolated, monkeypatch):
        user = isolated / "user" / "adder.json"
        user.parent.mkdir(parents=True)
        user.write_text(json.dumps({"budget": 40.0}))
        monkeypatch.setattr(settings, "USER_FILE", user)
        (isolated / ".adder.json").write_text(json.dumps({"budget": 5.0}))
        assert resolve(env={})["budget"].value == 5.0

    def test_env_beats_everything(self, isolated):
        (isolated / ".adder.json").write_text(json.dumps({"budget": 5.0}))
        r = resolve(env={"ADDER_BUDGET": "99"})
        assert r["budget"].value == 99.0
        assert r["budget"].source == "$ADDER_BUDGET"

    def test_empty_env_var_does_not_override(self, isolated):
        (isolated / ".adder.json").write_text(json.dumps({"budget": 5.0}))
        assert resolve(env={"ADDER_BUDGET": ""})["budget"].value == 5.0


class TestProjectFileDiscovery:
    def test_found_from_a_subdirectory(self, isolated):
        (isolated / ".adder.json").write_text("{}")
        deep = isolated / "a" / "b" / "c"
        deep.mkdir(parents=True)
        assert project_file(deep) == isolated / ".adder.json"

    def test_absent_is_none(self, isolated):
        assert project_file(isolated) is None


class TestCasting:
    def test_bools_accept_the_usual_spellings(self):
        for truthy in ("1", "true", "TRUE", "yes", "on"):
            assert resolve(env={"ADDER_CACHE": truthy})["cache"].value is True
        for falsy in ("0", "false", "no", "off"):
            assert resolve(env={"ADDER_CACHE": falsy})["cache"].value is False

    def test_ints_and_floats_come_back_typed(self):
        r = resolve(env={"ADDER_WARN_CONTEXT": "1234", "ADDER_TARGET": "2.5"})
        assert r["warn_context"].value == 1234
        assert isinstance(r["warn_context"].value, int)
        assert r["target"].value == 2.5

    def test_paths_expand_tilde(self):
        assert not str(resolve(env={"ADDER_ROOT": "~/x"})["root"].value).startswith("~")

    def test_uncastable_value_names_the_setting_and_source(self):
        with pytest.raises(ConfigError) as e:
            resolve(env={"ADDER_TARGET": "not-a-number"})
        assert "target" in str(e.value)
        assert "ADDER_TARGET" in str(e.value)


class TestBadFiles:
    def test_malformed_json_is_reported_not_ignored(self, isolated):
        (isolated / ".adder.json").write_text("{not json")
        with pytest.raises(ConfigError):
            resolve(env={})

    def test_non_object_top_level_is_reported(self, isolated):
        (isolated / ".adder.json").write_text("[1,2,3]")
        with pytest.raises(ConfigError):
            resolve(env={})


class TestSurface:
    def test_every_setting_has_a_unique_env_var(self):
        names = [s.env_var for s in settings.SETTINGS]
        assert len(names) == len(set(names))

    def test_every_setting_has_help(self):
        assert all(s.help.strip() for s in settings.SETTINGS)

    def test_template_is_valid_json_and_round_trips(self):
        d = json.loads(template())
        assert d["model"]
        for k in d:
            assert k in settings.BY_NAME

    def test_get_rejects_an_unknown_name(self):
        with pytest.raises(KeyError):
            settings.get("nope")

    def test_get_returns_the_resolved_value(self):
        assert settings.get("ttl", env={"ADDER_TTL": "1h"}) == "1h"

    def test_resolved_exposes_the_setting(self):
        r = resolve(env={})["model"]
        assert isinstance(r, Resolved)
        assert r.name == "model"


class TestCli:
    def test_plain_run_prints_a_table(self, capsys):
        assert config.main([]) == 0
        assert "Effective configuration" in capsys.readouterr().out

    def test_named_setting_prints_only_the_value(self, capsys):
        assert config.main(["model"]) == 0
        assert capsys.readouterr().out.strip() == settings.BY_NAME["model"].default

    def test_unknown_name_is_a_usage_error(self, capsys):
        assert config.main(["nope"]) == 2
        # stderr, not stdout: `--json` writes a document to stdout and an error
        # line mixed into it is a document that does not parse.
        assert "unknown setting" in capsys.readouterr().err

    def test_json_is_machine_readable(self, capsys):
        assert config.main(["--json"]) == 0
        d = json.loads(capsys.readouterr().out)
        assert d["model"]["source"] == "default"

    def test_init_emits_a_usable_file(self, capsys):
        assert config.main(["--init"]) == 0
        assert json.loads(capsys.readouterr().out)["ttl"] == "5m"

    def test_bad_config_exits_one_with_a_message(self, isolated, capsys):
        (isolated / ".adder.json").write_text("{")
        assert config.main([]) == 1
        assert "adder config:" in capsys.readouterr().err


class TestEnvOnlySettingsAreReportedHonestly:
    """Three settings are consumed below the settings layer and cannot read a file.

    `util.render` and `pricing.*` sit under `core` and may not import
    `core.settings` -- `tests/repo/test_structure.py` enforces it -- so they
    read their environment variable directly. `resolve` was reporting a value
    from `.adder.json` as the effective one anyway, which is exactly the
    invisible state `settings.py` opens by saying it exists to remove.
    """

    def _project(self, tmp_path, monkeypatch, body):
        import json as _json

        (tmp_path / ".adder.json").write_text(_json.dumps(body), encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        for var in ("ADDER_COLOR", "ADDER_OFFLINE", "ADDER_CATALOG"):
            monkeypatch.delenv(var, raising=False)
        return tmp_path

    def test_a_file_value_is_not_reported_as_effective(self, tmp_path, monkeypatch):
        from adder.core.settings import resolve

        self._project(tmp_path, monkeypatch, {"offline": True})
        r = resolve()["offline"]
        assert r.value is False and r.source == "default"

    def test_the_environment_still_wins(self, tmp_path, monkeypatch):
        from adder.core.settings import resolve

        self._project(tmp_path, monkeypatch, {"offline": True})
        monkeypatch.setenv("ADDER_OFFLINE", "1")
        assert resolve()["offline"].value is True

    def test_a_file_setting_one_of_them_is_named(self, tmp_path, monkeypatch):
        from adder.core.settings import ignored_in_files

        self._project(tmp_path, monkeypatch, {"offline": True, "color": "always"})
        assert ignored_in_files() == ["color", "offline"]

    def test_the_report_says_so(self, tmp_path, monkeypatch, capsys):
        self._project(tmp_path, monkeypatch, {"catalog": "/nowhere.json"})
        assert config.main([]) == 0
        out = capsys.readouterr().out
        assert "env only" in out and "which nothing reads from a file" in out

    def test_a_normal_setting_is_unaffected(self, tmp_path, monkeypatch):
        from adder.core.settings import resolve

        self._project(tmp_path, monkeypatch, {"budget": 42.0})
        assert resolve()["budget"].value == 42.0
