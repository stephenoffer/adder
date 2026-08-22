"""The one module here that writes to a file the user did not name.

`adder guard --install` printed a JSON block and asked you to merge it. That is
defensible for a report and indefensible for the thing the whole tool depends
on: the measured difference between installed and not is the difference between
1.6x and 1.0x, and a saving gated on a copy-and-paste is a saving most people
never get.

Writing settings for someone raises exactly one question -- what happens when
it goes wrong -- so most of what follows is about the file that was already
there. It must survive being edited by us: foreign hooks kept, unrelated keys
kept, a malformed file refused rather than replaced, and `off` removing what
`on` added and nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adder.decide import guard
from adder.decide.auto import (
    BACKUP_SUFFIX,
    HOOKS,
    Plan,
    _override_warnings,
    apply,
    merge,
    plan,
    plan_off,
    render_status,
    status,
    unmerge,
)

FOREIGN = {
    'hooks': {
        'PreToolUse': [{'matcher': 'Bash',
                        'hooks': [{'type': 'command', 'command': 'echo not-ours'}]}],
        'Stop': [{'hooks': [{'type': 'command', 'command': 'echo also-not-ours'}]}],
    },
    'permissions': {'allow': ['Bash(ls:*)']},
    'model': 'opusplan',
}


@pytest.fixture
def clean_home(tmp_path, monkeypatch):
    """A machine with no adder on it, whatever the real one has installed.

    `status` reads `~/.claude/settings.json` as one of the places a hook can be
    declared, so without this the result depends on whether the author had
    activated the tool -- the definition of a broken test in this repo.
    """
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path))
    monkeypatch.setenv('ADDER_GUARD_STATE', str(tmp_path / '.adder-guard.json'))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _commands(blob: dict) -> list[str]:
    return [e.get('command', '')
            for groups in (blob.get('hooks') or {}).values()
            for g in groups for e in (g.get('hooks') or [])]


class TestMerging:
    """Pure dict-to-dict, so the risky part can be tested without a filesystem."""

    def test_all_three_hooks_are_added_to_an_empty_file(self):
        out, changed = merge({})
        assert len(changed) == len(HOOKS)
        for h in HOOKS:
            assert any(h['script'] in c for c in _commands(out))

    def test_the_guard_is_matched_against_every_tool_it_watches(self):
        from adder.decide.guard import OBSERVED
        out, _ = merge({})
        matcher = out['hooks']['PreToolUse'][0]['matcher']
        assert set(matcher.split('|')) == set(OBSERVED), \
            "a tool missing from the matcher is a tool the guard never sees"

    def test_merging_twice_changes_nothing_the_second_time(self):
        once, _ = merge({})
        twice, changed = merge(once)
        assert changed == [] and twice == once

    def test_a_foreign_hook_on_the_same_event_is_kept(self):
        out, _ = merge(FOREIGN)
        assert 'echo not-ours' in _commands(out)

    def test_unrelated_settings_are_untouched(self):
        out, _ = merge(FOREIGN)
        assert out['permissions'] == FOREIGN['permissions']
        assert out['model'] == 'opusplan'

    def test_the_input_dict_is_not_mutated(self):
        before = json.dumps(FOREIGN, sort_keys=True)
        merge(FOREIGN)
        assert json.dumps(FOREIGN, sort_keys=True) == before

    def test_a_hooks_key_that_is_not_a_dict_is_refused_rather_than_replaced(self):
        out, changed = merge({'hooks': 'nonsense'})
        assert changed == [] and out == {'hooks': 'nonsense'}


class TestRemoving:
    """`off` has to be exact. An activation people cannot undo is one they will
    not try in the first place."""

    def test_it_removes_what_merge_added(self):
        out, changed = unmerge(merge({})[0])
        assert len(changed) == len(HOOKS)
        assert _commands(out) == []

    def test_it_leaves_foreign_hooks_alone(self):
        out, _ = unmerge(merge(FOREIGN)[0])
        assert 'echo not-ours' in _commands(out)
        assert 'echo also-not-ours' in _commands(out)

    def test_it_leaves_unrelated_settings_alone(self):
        out, _ = unmerge(merge(FOREIGN)[0])
        assert out['permissions'] == FOREIGN['permissions']

    def test_on_then_off_restores_the_original(self):
        assert unmerge(merge(FOREIGN)[0])[0] == FOREIGN

    def test_removing_from_a_file_we_never_touched_does_nothing(self):
        out, changed = unmerge(FOREIGN)
        assert changed == [] and out == FOREIGN

    def test_an_emptied_event_is_dropped_not_left_as_a_stub(self):
        out, _ = unmerge(merge({})[0])
        assert 'PreCompact' not in (out.get('hooks') or {})

    def test_a_hook_installed_from_another_checkout_is_still_removed(self):
        """Matched on the script name, not the path. Otherwise moving the
        checkout leaves a hook that fails on every single tool call."""
        stale = {'hooks': {'PreToolUse': [{'hooks': [
            {'type': 'command',
             'command': 'python3 /somewhere/else/.claude/hooks/pretooluse_read_guard.py'}]}]}}
        out, changed = unmerge(stale)
        assert changed and not out.get('hooks')


class TestPlanning:
    """`plan` reads, `apply` writes, and nothing else does either."""

    def test_planning_writes_nothing(self, tmp_path):
        plan(cwd=tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_it_reports_the_level_it_would_set(self, tmp_path):
        assert plan(cwd=tmp_path, level='full').level == 'full'
        assert plan(cwd=tmp_path).level == 'certain'

    def test_a_malformed_settings_file_blocks_rather_than_being_overwritten(self,
                                                                           tmp_path):
        p = tmp_path / '.claude' / 'settings.json'
        p.parent.mkdir()
        p.write_text('{ this is not json')
        got = plan(cwd=tmp_path)
        assert got.blocked
        with pytest.raises(ValueError):
            apply(got)
        assert p.read_text() == '{ this is not json'

    def test_a_malformed_config_file_blocks_too(self, tmp_path):
        (tmp_path / '.adder.json').write_text('nope')
        assert plan(cwd=tmp_path).blocked

    def test_a_plan_with_nothing_to_do_says_so(self, tmp_path):
        apply(plan(cwd=tmp_path))
        assert plan(cwd=tmp_path).empty


class TestApplying:
    def test_it_writes_both_files(self, tmp_path):
        p = plan(cwd=tmp_path)
        apply(p)
        assert json.loads(p.settings_path.read_text())['hooks']
        assert json.loads(p.config_path.read_text())['guard_enforce'] == 'certain'

    def test_the_original_is_backed_up_before_it_is_edited(self, tmp_path):
        s = tmp_path / '.claude' / 'settings.json'
        s.parent.mkdir()
        s.write_text(json.dumps(FOREIGN))
        apply(plan(cwd=tmp_path))
        assert json.loads(s.with_name(s.name + BACKUP_SUFFIX).read_text()) == FOREIGN

    def test_the_backup_is_not_overwritten_by_a_later_run(self, tmp_path):
        """The point of it is the state before adder touched anything."""
        s = tmp_path / '.claude' / 'settings.json'
        s.parent.mkdir()
        s.write_text(json.dumps(FOREIGN))
        apply(plan(cwd=tmp_path))
        apply(plan_off(cwd=tmp_path))
        apply(plan(cwd=tmp_path, level='full'))
        assert json.loads(s.with_name(s.name + BACKUP_SUFFIX).read_text()) == FOREIGN

    def test_the_written_file_is_valid_json_a_harness_could_read(self, tmp_path):
        p = plan(cwd=tmp_path)
        apply(p)
        blob = json.loads(p.settings_path.read_text())
        for groups in blob['hooks'].values():
            for g in groups:
                for e in g['hooks']:
                    assert e['type'] == 'command'
                    # `sys.executable`, not `python3`: see `guard.interpreter`.
                    assert e['command'].startswith(guard.interpreter() + ' ')

    def test_the_hook_scripts_it_points_at_exist(self, tmp_path):
        p = plan(cwd=tmp_path)
        apply(p)
        for cmd in _commands(json.loads(p.settings_path.read_text())):
            assert Path(cmd.split(' ', 1)[1]).is_file(), \
                "an installed hook that is not on disk fails on every tool call"

    def test_off_after_on_leaves_no_trace_in_settings(self, tmp_path):
        apply(plan(cwd=tmp_path))
        p = plan_off(cwd=tmp_path)
        apply(p)
        assert not json.loads(p.settings_path.read_text()).get('hooks')
        assert json.loads(p.config_path.read_text())['guard_enforce'] == 'off'


class TestTheAgentFiles:
    """The other half of "installed and changed nothing".

    `adder bench` prices the hooks and the tier agents together — 2.5x for the
    hooks, 3.1x for the pair — so activation that installed one and not the
    other would be quoting a multiple it does not deliver. It did exactly that
    until these tests existed.
    """

    def test_activation_installs_them(self, tmp_path):
        from adder.decide.auto import AGENTS
        p = plan(cwd=tmp_path)
        apply(p)
        for name in AGENTS:
            assert (p.agents_path / name).is_file()

    def test_they_go_beside_the_settings_file_it_wrote(self, tmp_path):
        p = plan(cwd=tmp_path)
        assert p.agents_path.parent == p.settings_path.parent

    def test_explore_is_installed_because_it_needs_no_decision(self):
        """It overrides a built-in that already runs on every exploration."""
        from adder.decide.auto import AGENTS
        assert AGENTS[0] == 'Explore.md'

    def test_an_agent_the_user_already_has_is_never_overwritten(self, tmp_path):
        """Silently replacing somebody's Explore during a command about cost is
        the worst kind of surprise: it changes what a built-in does, in a file
        they did not know we knew about."""
        mine = tmp_path / '.claude' / 'agents' / 'Explore.md'
        mine.parent.mkdir(parents=True)
        mine.write_text('mine, and I rely on it')
        p = plan(cwd=tmp_path)
        apply(p)
        assert mine.read_text() == 'mine, and I rely on it'
        assert 'Explore.md' in p.agent_skips
        assert 'Explore.md' not in p.agent_writes

    def test_the_skipped_file_is_named_in_the_plan(self, tmp_path):
        from adder.decide.auto import _render_plan
        mine = tmp_path / '.claude' / 'agents' / 'Explore.md'
        mine.parent.mkdir(parents=True)
        mine.write_text('mine')
        text = '\n'.join(_render_plan(plan(cwd=tmp_path)))
        assert 'Explore.md' in text and 'left alone' in text

    def test_an_identical_file_is_neither_written_nor_flagged(self, tmp_path):
        """Re-running activation must be a no-op, not a list of pretend work."""
        apply(plan(cwd=tmp_path))
        again = plan(cwd=tmp_path)
        assert not again.agent_writes and not again.agent_skips
        assert again.empty

    def test_off_leaves_them_in_place(self, tmp_path):
        """They cost nothing without the hooks, and one may have been edited."""
        p = plan(cwd=tmp_path)
        apply(p)
        apply(plan_off(cwd=tmp_path))
        assert (p.agents_path / 'route-t0.md').is_file()

    def test_the_installed_copy_matches_the_source(self, tmp_path):
        from adder.decide.auto import agents_dir
        p = plan(cwd=tmp_path)
        apply(p)
        assert (p.agents_path / 'route-t2.md').read_text() == \
            (agents_dir() / 'route-t2.md').read_text()

    def test_every_named_agent_exists_in_the_checkout(self):
        """A name here that is not on disk is an install that silently does
        three quarters of what it says."""
        from adder.decide.auto import AGENTS, agents_dir
        for name in AGENTS:
            assert (agents_dir() / name).is_file(), name


class TestWarnings:
    """Silent overrides that would make activation a lie."""

    def test_a_subagent_model_override_is_called_out(self):
        got = _override_warnings({'CLAUDE_CODE_SUBAGENT_MODEL': 'claude-opus-5'})
        assert got and 'outranks' in got[0]

    def test_an_env_kill_switch_is_called_out(self):
        assert _override_warnings({'ADDER_GUARD_ENFORCE': 'off'})

    def test_a_clean_environment_warns_about_nothing(self):
        assert _override_warnings({}) == []


class TestStatus:
    """Read-only, and it must never raise: it is what someone runs when they
    suspect the tool is not doing anything."""

    def test_it_reports_off_when_nothing_is_installed(self, clean_home):
        """`HOME` is redirected, because whether the author has activated adder
        on this machine must not decide whether the test passes."""
        s = status(cwd=clean_home)
        assert not s.installed and not s.active

    def test_the_off_message_says_what_to_run(self, clean_home):
        assert 'adder auto on' in render_status(status(cwd=clean_home))

    def test_it_reports_on_once_the_hooks_are_written(self, clean_home):
        apply(plan(cwd=clean_home))
        s = status(cwd=clean_home)
        assert s.installed and s.level == 'certain' and not s.hooks_missing
        assert 'ON' in render_status(s)

    def test_a_refusal_is_credited_at_par_and_advice_is_not(self):
        from adder.decide.auto import Status
        s = Status(installed=[Path('x')], level='certain', hooks_present=[],
                   hooks_missing=[], sessions=1, fires=2, prevented=8.0,
                   promised=4.0, overhead=1.0, uptake=0.5, measured_uptake=False,
                   model_calls=10, model_age_s=1.0)
        assert s.realised == pytest.approx(10.0)      # 8 at par + 4 x 0.5
        assert s.ratio == pytest.approx(10.0)

    def test_no_overhead_yet_is_not_an_infinite_return(self):
        from adder.decide.auto import Status
        s = Status(installed=[], level='off', hooks_present=[], hooks_missing=[],
                   sessions=0, fires=0, prevented=0.0, promised=0.0, overhead=0.0,
                   uptake=0.5, measured_uptake=False, model_calls=0,
                   model_age_s=float('inf'))
        assert s.ratio == 0.0

    def test_rendering_a_plan_names_every_file_it_would_touch(self, tmp_path):
        from adder.decide.auto import _render_plan
        p = plan(cwd=tmp_path)
        text = '\n'.join(_render_plan(p))
        assert str(p.settings_path) in text and str(p.config_path) in text


class TestTheCommand:
    def test_status_is_the_default_action(self, capsys):
        from adder.decide.auto import main
        assert main([]) == 0
        assert 'adder auto' in capsys.readouterr().out

    def test_dry_run_writes_nothing(self, tmp_path, monkeypatch, capsys):
        from adder.decide.auto import main
        monkeypatch.chdir(tmp_path)
        assert main(['on', '--dry-run']) == 0
        assert 'nothing written' in capsys.readouterr().out.lower()
        assert not (tmp_path / '.claude').exists()

    def test_declining_the_prompt_writes_nothing(self, tmp_path, monkeypatch):
        from adder.decide import auto
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(auto, '_confirm', lambda _: False)
        assert auto.main(['on']) == 1
        assert not (tmp_path / '.claude').exists()

    def test_json_status_is_machine_readable(self, capsys):
        from adder.decide.auto import main
        assert main(['status', '--json']) == 0
        assert 'level' in json.loads(capsys.readouterr().out)


class TestTheEnforcingThresholds:
    """`full` moves three numbers as well as the level, and which three is a
    measurement rather than a preference. These pin the shape of that claim;
    the numbers themselves are re-derivable with `--tune`."""

    def test_full_writes_the_measured_thresholds(self, tmp_path):
        from adder.decide.auto import ENFORCING_THRESHOLDS
        got = plan(cwd=tmp_path, level='full').config_after
        for k, v in ENFORCING_THRESHOLDS.items():
            assert got[k] == v

    def test_certain_leaves_the_thresholds_alone(self, tmp_path):
        """`certain` refuses only what admits nothing new. How large a call has
        to be before the guard speaks is a different question, already answered
        by whatever the user configured."""
        got = plan(cwd=tmp_path, level='certain').config_after
        assert set(got) == {'guard_enforce'}

    def test_the_floor_is_above_the_one_the_reports_solve_for(self):
        """Deliberate, and the one place the tool does not take its own advice:
        below 800 tokens the hook parses a transcript on half of all tool calls
        to find money the dollar gate has already found."""
        from adder.decide.auto import ENFORCING_THRESHOLDS
        assert ENFORCING_THRESHOLDS['guard_min_tokens'] == 800

    def test_a_tuned_answer_overrides_the_shipped_one(self, tmp_path):
        got = plan(cwd=tmp_path, level='full',
                   thresholds={'guard_min_tokens': 1234}).config_after
        assert got['guard_min_tokens'] == 1234

    def test_the_change_list_names_every_setting_it_moves(self, tmp_path):
        p = plan(cwd=tmp_path, level='full')
        assert len(p.config_changes) == 4
        assert any('guard_enforce' in c for c in p.config_changes)

    def test_a_setting_already_at_the_target_is_not_relisted(self, tmp_path):
        (tmp_path / '.adder.json').write_text(json.dumps({'guard_min_tokens': 800}))
        p = plan(cwd=tmp_path, level='full')
        assert not any(c.startswith('guard_min_tokens') for c in p.config_changes)

    def test_off_does_not_revert_a_tuned_threshold(self, tmp_path):
        """They are measurements of this workload and they mean something to an
        advisory guard too. `off` means stop refusing, not forget."""
        apply(plan(cwd=tmp_path, level='full'))
        p = plan_off(cwd=tmp_path)
        apply(p)
        after = json.loads(p.config_path.read_text())
        assert after['guard_enforce'] == 'off' and after['guard_min_tokens'] == 800


class TestTuning:
    """The sweep that lets somebody else's workload answer the same question."""

    def test_the_best_point_is_the_highest_net(self):
        from adder.decide.auto import Tuning, best
        lo = Tuning(2000, 0.25, 10, 100.0, 0.0, 5.0, 0.01)
        hi = Tuning(800, 0.10, 50, 300.0, 0.0, 20.0, 0.15)
        assert best([lo, hi]) is hi

    def test_a_tie_goes_to_the_one_that_parses_fewer_transcripts(self):
        """A lower floor always finds at least as much and always parses more
        to do it, so a pure argmax would pick the noisiest point every time."""
        from adder.decide.auto import Tuning, best
        cheap = Tuning(800, 0.10, 50, 300.0, 0.0, 20.0, 0.08)
        dear = Tuning(300, 0.10, 50, 302.0, 0.0, 20.0, 0.39)
        assert best([dear, cheap]) is cheap

    def test_a_materially_better_point_wins_despite_the_latency(self):
        """The margin is a threshold, not a veto. $138 more is worth 39%."""
        from adder.decide.auto import Tuning, best
        cheap = Tuning(800, 0.10, 50, 300.0, 0.0, 20.0, 0.08)
        dear = Tuning(300, 0.10, 90, 460.0, 0.0, 30.0, 0.39)
        assert best([dear, cheap]) is dear

    def test_the_margin_is_relative_not_absolute(self):
        from adder.decide.auto import MATERIAL_MARGIN
        assert 0 < MATERIAL_MARGIN < 0.5

    def test_a_workload_where_nothing_pays_still_answers(self):
        from adder.decide.auto import Tuning, best
        loss = Tuning(800, 0.10, 0, 0.0, 0.0, 1.0, 0.0)
        assert best([loss]) is loss

    def test_net_is_prevented_plus_advice_less_overhead(self):
        from adder.decide.auto import Tuning
        assert Tuning(800, 0.1, 1, 10.0, 4.0, 2.0, 0.1).net == pytest.approx(12.0)

    def test_no_points_is_not_a_crash(self):
        from adder.decide.auto import best
        assert best([]) is None

    def test_tuning_an_empty_corpus_returns_a_point_per_grid_row(self, tmp_path):
        from adder.decide.auto import TUNE_GRID, tune
        assert len(tune(tmp_path)) == len(TUNE_GRID)

    def test_tune_without_full_is_refused_rather_than_ignored(self, capsys):
        from adder.decide.auto import main
        assert main(['on', '--tune', '--dry-run']) == 2
        capsys.readouterr()


class TestPlanShape:
    def test_an_empty_plan_is_recognised(self):
        assert Plan(settings_path=Path('a'), config_path=Path('b'),
                    level='off', was_level='off').empty
