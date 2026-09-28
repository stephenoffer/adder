"""`auto off` gives back the files `auto on` found, and takes away what it wrote.

Three ways it did not. `unmerge` deleted an event list or a `hooks` object the
user already had empty, so the result was not the file that was there before.
Every write re-serialised the file at indent 2 with non-ASCII escaped, so even
a correct removal showed up as a whole-file diff. And the agent files stayed
behind as costing "nothing without the hooks", when `Explore.md` overrides a
built-in and pins every exploration to Haiku.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adder.decide.auto import AGENTS, agents_dir, apply, merge, plan, plan_off, unmerge


@pytest.fixture(autouse=True)
def home(tmp_path_factory, monkeypatch):
    h = tmp_path_factory.mktemp('not-your-home')
    (h / '.claude').mkdir()
    monkeypatch.setenv('HOME', str(h))
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: h))
    return h


class TestUnmerge:
    def test_an_event_the_user_left_empty_survives(self):
        mine = {'hooks': {'Stop': []}}
        assert unmerge(merge(mine)[0])[0] == mine

    def test_a_group_of_the_users_with_no_hooks_survives(self):
        mine = {'hooks': {'PreToolUse': [{'matcher': 'Read', 'hooks': []}]}}
        assert unmerge(merge(mine)[0])[0] == mine


class TestTheFileComesBackByteForByte:
    ORIGINAL = ('{\n    "hooks": {\n        "Stop": []\n    },\n'
                '    "env": {"NAME": "Zoë"},\n'
                '    "permissions": {"allow": ["Bash(ls:*)"]}\n}\n')

    def test_on_then_off(self, tmp_path):
        settings = tmp_path / '.claude' / 'settings.json'
        settings.parent.mkdir()
        settings.write_text(self.ORIGINAL, encoding='utf-8')
        apply(plan(cwd=tmp_path, user=False))
        assert settings.read_text(encoding='utf-8') != self.ORIGINAL
        assert '\\u00eb' not in settings.read_text(encoding='utf-8')
        apply(plan_off(cwd=tmp_path, user=False))
        assert settings.read_text(encoding='utf-8') == self.ORIGINAL


class TestTheAgentFilesGo:
    def test_off_removes_the_copies_on_wrote(self, tmp_path):
        p = plan(cwd=tmp_path, user=False)
        apply(p)
        assert (p.agents_path / 'Explore.md').is_file()
        off = plan_off(cwd=tmp_path, user=False)
        assert sorted(off.agent_removes) == sorted(
            n for n in AGENTS if (agents_dir() / n).is_file())
        apply(off)
        assert not any((p.agents_path / n).exists() for n in AGENTS)

    def test_an_edited_copy_stays(self, tmp_path):
        p = plan(cwd=tmp_path, user=False)
        apply(p)
        mine = p.agents_path / 'Explore.md'
        mine.write_text(mine.read_text() + '\nmy own note\n')
        off = plan_off(cwd=tmp_path, user=False)
        assert 'Explore.md' not in off.agent_removes
        apply(off)
        assert mine.read_text().endswith('my own note\n')


class TestOnlyWhatThisInstallWrote:
    def test_identical_files_with_no_install_are_not_removed(self, tmp_path):
        """A team can commit the shipped files; this repository tracks a mirror
        of them. Identical is not proof `on` wrote them."""
        (tmp_path / '.claude' / 'agents').mkdir(parents=True)
        for n in AGENTS:
            src = agents_dir() / n
            if src.is_file():
                (tmp_path / '.claude' / 'agents' / n).write_bytes(src.read_bytes())
        off = plan_off(cwd=tmp_path, user=False)
        assert off.hook_changes == [] and off.agent_removes == [] and off.skill_removes == []


class TestAnOldPortableLineIsUpgraded:
    def test_a_committed_bare_line_gains_the_fail_open(self):
        old = {'hooks': {'PreToolUse': [{'hooks': [
            {'type': 'command', 'command': 'adder hook read-guard'}]}]}}
        out, changed = merge(old, portable=True)
        cmds = [e['command'] for g in out['hooks']['PreToolUse'] for e in g['hooks']]
        assert cmds == ['adder hook read-guard || true']
        assert any('fails open' in c for c in changed)
        assert merge(out, portable=True)[1] == []           # and only once


def test_off_makes_no_backup_of_what_on_created(tmp_path):
    """`on` created the config from nothing; `off` backed that up as if it
    were the user's original."""
    apply(plan(cwd=tmp_path, user=False))
    off = plan_off(cwd=tmp_path, user=False)
    apply(off)
    assert not off.config_path.with_name(off.config_path.name + '.adder.bak').exists()
