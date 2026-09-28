"""Activation beside the Claude Code plugin, and the skills it now installs.

Two installers can now put adder into Claude Code: `adder auto on` writing
`settings.json`, and the plugin declaring the same hooks in its own
`hooks.json`. Claude Code merges the two with no deduplication, so the failure
this file exists for is the quiet one: both installed, every hook running
twice, the guard's overhead charged twice and every refusal counted twice in
the ledger `auto status` reports as proof it is working.

The skills are the other half. They lived in `.claude/skills/`, which the wheel
prunes, so a pip user got hooks and agents and no way for the agent to reach a
report without typing the command.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adder.decide import auto
from adder.decide.auto import (
    AGENTS,
    SKILLS,
    _ephemeral,
    _interpreter_warnings,
    _is_ours,
    _render_plan,
    apply,
    plan,
    plan_off,
    plugin_enabled,
    plugin_hooks,
    skills_dir,
    status,
)


@pytest.fixture(autouse=True)
def home(tmp_path_factory, monkeypatch):
    """A home directory that is not the developer's, for every test here."""
    h = tmp_path_factory.mktemp('home')
    (h / '.claude').mkdir()
    monkeypatch.setenv('HOME', str(h))
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: h))
    monkeypatch.setenv('ADDER_GUARD_STATE', str(h / '.adder-guard.json'))
    monkeypatch.delenv('ADDER_HARNESS', raising=False)
    return h


def _enable(path: Path, value: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.loads(path.read_text()) if path.exists() else {}
    blob.setdefault('enabledPlugins', {})['adder@adder'] = value
    path.write_text(json.dumps(blob))


class TestTheSkills:
    def test_every_skill_is_in_the_package(self):
        for src in SKILLS:
            assert (skills_dir() / src / 'SKILL.md').is_file(), src

    def test_none_hardcodes_a_name(self):
        """The directory names the skill. A `name:` would give a plugin user
        `/adder:adder-doctor` and a pip user whatever the frontmatter said,
        rather than the two names `SKILLS` promises."""
        for src in SKILLS:
            head = (skills_dir() / src / 'SKILL.md').read_text().split('---')[1]
            assert not any(line.startswith('name:') for line in head.splitlines()), src

    def test_activation_copies_them_under_their_installed_names(self, tmp_path):
        p = plan(cwd=tmp_path, user=False, plugin=False)
        apply(p)
        for src, name in SKILLS.items():
            got = p.skills_path / name / 'SKILL.md'
            assert got.read_text() == (skills_dir() / src / 'SKILL.md').read_text()

    def test_routing_keeps_the_name_the_docs_use(self):
        assert SKILLS['route'] == 'adder'

    def test_a_skill_the_user_edited_is_never_overwritten(self, tmp_path):
        mine = tmp_path / '.claude' / 'skills' / 'adder-doctor' / 'SKILL.md'
        mine.parent.mkdir(parents=True)
        mine.write_text('mine')
        p = plan(cwd=tmp_path, user=False, plugin=False)
        apply(p)
        assert mine.read_text() == 'mine'
        assert 'adder-doctor' in p.skill_skips
        assert 'left alone' in '\n'.join(_render_plan(p))

    def test_running_it_twice_is_a_no_op(self, tmp_path):
        apply(plan(cwd=tmp_path, user=False, plugin=False))
        again = plan(cwd=tmp_path, user=False, plugin=False)
        assert not again.skill_writes and again.empty

    def test_off_removes_only_the_copies_nobody_edited(self, tmp_path):
        """A resident skill description costs every turn, so a turned-off tool
        should not leave four behind; an edited one is somebody's now."""
        p = plan(cwd=tmp_path, user=False, plugin=False)
        apply(p)
        edited = p.skills_path / 'adder-context' / 'SKILL.md'
        edited.write_text('edited')
        off = plan_off(cwd=tmp_path, user=False)
        assert 'adder-context' not in off.skill_removes
        apply(off)
        assert not (p.skills_path / 'adder').exists()
        assert edited.read_text() == 'edited'

    def test_off_keeps_a_directory_that_holds_something_else(self, tmp_path):
        p = plan(cwd=tmp_path, user=False, plugin=False)
        apply(p)
        extra = p.skills_path / 'adder' / 'notes.md'
        extra.write_text('mine')
        apply(plan_off(cwd=tmp_path, user=False))
        assert extra.is_file()


class TestDetectingThePlugin:
    def test_off_by_default(self, tmp_path):
        assert not plugin_enabled(tmp_path)

    def test_enabled_at_user_scope(self, home, tmp_path):
        _enable(home / '.claude' / 'settings.json')
        assert plugin_enabled(tmp_path)

    def test_any_marketplace_counts(self, home, tmp_path):
        (home / '.claude' / 'settings.json').write_text(
            json.dumps({'enabledPlugins': {'adder@a-fork': True}}))
        assert plugin_enabled(tmp_path)

    def test_a_similarly_named_plugin_does_not(self, home, tmp_path):
        (home / '.claude' / 'settings.json').write_text(
            json.dumps({'enabledPlugins': {'adderall@x': True}}))
        assert not plugin_enabled(tmp_path)

    def test_the_nearer_file_wins(self, home, tmp_path):
        _enable(home / '.claude' / 'settings.json', True)
        _enable(tmp_path / '.claude' / 'settings.json', False)
        assert not plugin_enabled(tmp_path)

    def test_a_malformed_file_is_absent_not_fatal(self, home, tmp_path):
        (home / '.claude' / 'settings.json').write_text('{not json')
        assert not plugin_enabled(tmp_path)


class TestActivationBesideThePlugin:
    def test_it_writes_no_hooks(self, home, tmp_path):
        p = plan(cwd=tmp_path, plugin=True)
        assert not p.hook_changes
        assert 'hooks come from the adder plugin' in '\n'.join(_render_plan(p))

    def test_it_removes_the_copy_an_earlier_activation_wrote(self, home, tmp_path):
        apply(plan(cwd=tmp_path, plugin=False))
        settings = home / '.claude' / 'settings.json'
        assert auto.guard.installed_in(tmp_path)
        _enable(settings)
        p = plan(cwd=tmp_path)
        assert p.plugin and len(p.hook_changes) == len(auto.HOOKS)
        apply(p)
        assert not auto.guard.installed_in(tmp_path)
        assert json.loads(settings.read_text())['enabledPlugins'] == {'adder@adder': True}

    def test_it_still_sets_the_level(self, tmp_path):
        p = plan(cwd=tmp_path, level='full', plugin=True)
        assert p.config_after['guard_enforce'] == 'full'

    def test_it_copies_only_explore(self, tmp_path):
        """A plugin's agents are namespaced and cannot replace a built-in, so
        the Explore override is the one thing only activation can deliver."""
        p = plan(cwd=tmp_path, plugin=True)
        assert p.agent_writes == [AGENTS[0]] == ['Explore.md']
        assert not p.skill_writes

    def test_status_counts_the_plugin_as_installed(self, home, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _enable(home / '.claude' / 'settings.json')
        s = status(cwd=tmp_path)
        assert s.plugin and s.active and not s.hooks_missing
        assert 'the adder plugin' in auto.render_status(s)

    def test_status_names_a_double_registration(self, home, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        apply(plan(cwd=tmp_path, plugin=False))
        _enable(home / '.claude' / 'settings.json')
        assert auto.double_registered(tmp_path)
        assert 'runs twice' in auto.render_status(status(cwd=tmp_path))


class TestThePluginHookTable:
    def test_the_command_is_recognised_as_ours(self):
        for groups in plugin_hooks()['hooks'].values():
            for g in groups:
                for e in g['hooks']:
                    assert _is_ours(e['command'])

    def test_it_covers_every_hook_with_the_same_matcher(self):
        got = plugin_hooks()['hooks']
        for h in auto.HOOKS:
            (group,) = got[h['event']]
            assert group.get('matcher', '') == h['matcher']
            assert group['hooks'][0]['timeout'] == auto.HOOK_TIMEOUT_S


class TestTemporaryInterpreters:
    @pytest.mark.parametrize('exe', [
        '/Users/a/.cache/uv/archive-v0/Xy12/bin/python',
        '/home/a/.local/pipx/.cache/9f1c/bin/python',
    ])
    def test_throwaway_environments_are_recognised(self, exe):
        assert _ephemeral(exe)

    @pytest.mark.parametrize('exe', [
        '/Users/a/.local/share/uv/tools/adder-cli/bin/python',
        '/home/a/.local/pipx/venvs/adder-cli/bin/python',
        '/opt/homebrew/bin/python3.12',
    ])
    def test_installed_ones_are_not(self, exe):
        assert not _ephemeral(exe)

    def test_a_user_install_from_one_is_warned_about(self):
        w = _interpreter_warnings(user=True, plugin=False,
                                  exe='/u/.cache/uv/archive-v0/a/bin/python')
        assert w and 'uv tool install adder-cli' in w[0]

    def test_the_plugin_does_not_use_it_so_no_warning(self):
        assert not _interpreter_warnings(user=True, plugin=True,
                                         exe='/u/.cache/uv/archive-v0/a/bin/python')
