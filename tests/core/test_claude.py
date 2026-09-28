"""What Claude Code has installed, and the names routing dispatches under.

The failure these pin down is a router that recommends an agent which does not
exist: with only the plugin installed, the tier agents answer to
`adder:route-t1`, and `route-t1` reaches nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adder.core import claude, settings
from adder.decide.route.classify import Tier


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    (h / ".claude").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: h))
    monkeypatch.setattr(settings, "USER_FILE", h / ".claude" / "adder.json")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return h


def _plugin(home: Path, on: bool = True) -> None:
    (home / ".claude" / "settings.json").write_text(
        json.dumps({"enabledPlugins": {"adder@adder": on}}))


def _agent(home: Path, name: str) -> None:
    d = home / ".claude" / "agents"
    d.mkdir(exist_ok=True)
    (d / f"{name}.md").write_text("---\nname: x\n---\n")


class TestDispatchNames:
    def test_bare_with_nothing_installed(self, home):
        """What `auto on` will install, so the recommendation is still right."""
        assert claude.dispatch_name("route-t1") == "route-t1"

    def test_namespaced_when_only_the_plugin_supplies_it(self, home):
        _plugin(home)
        assert claude.dispatch_name("route-t1") == "adder:route-t1"

    def test_an_installed_file_wins_over_the_plugin(self, home):
        _plugin(home)
        _agent(home, "route-t1")
        assert claude.dispatch_name("route-t1") == "route-t1"

    def test_a_disabled_plugin_is_not_used(self, home):
        _plugin(home, on=False)
        assert claude.dispatch_name("route-t2") == "route-t2"

    def test_the_tier_names_what_this_machine_can_dispatch(self, home):
        assert Tier.T1.agent == "route-t1"
        _plugin(home)
        assert Tier.T1.agent == "adder:route-t1"
        assert Tier.T3.agent == "adder:route-t2"

    @pytest.mark.parametrize(("given", "want"), [
        ("adder:route-t1", "route-t1"), ("ADDER:route-t0", "route-t0"),
        ("route-t2", "route-t2"), ("other:route-t1", "other:route-t1")])
    def test_bare(self, given, want):
        assert claude.bare(given) == want


class TestTheDefaultLevel:
    """Installing the plugin should cut cost without a second step."""

    def test_unset_is_off_without_the_plugin(self, home):
        assert settings.resolve(env={})["guard_enforce"].value == "off"

    def test_unset_is_certain_with_it(self, home):
        _plugin(home)
        got = settings.resolve(env={})["guard_enforce"]
        assert (got.value, got.source) == ("certain", "the adder plugin")

    def test_a_config_file_still_wins(self, home):
        """How `auto off` returns the plugin's guard to advice."""
        _plugin(home)
        (home / ".claude" / "adder.json").write_text(json.dumps({"guard_enforce": "off"}))
        assert settings.resolve(env={})["guard_enforce"].value == "off"

    def test_the_environment_still_wins(self, home):
        _plugin(home)
        got = settings.resolve(env={"ADDER_GUARD_ENFORCE": "shadow"})["guard_enforce"]
        assert got.value == "shadow"

    def test_the_guard_reads_the_same_answer(self, home):
        from adder.decide.guard import Settings
        _plugin(home)
        assert Settings.resolve(env={}).enforce == "certain"
