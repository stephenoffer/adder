"""`adder trial`: the tasks are fair, the arms are what adder ships, and nothing
spends money or writes anywhere without being asked."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from adder.evaluate.replay import trial
from adder.evaluate.replay.trial_tasks import LONG, TASKS


def _lay(root, files):
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


def _suite(root):
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                           "tests", "tests_hidden"], cwd=root, capture_output=True).returncode


class TestTheTasksAreFair:
    """A task whose own solution fails, or whose start already passes, scores noise."""

    @pytest.mark.parametrize("task", TASKS + LONG, ids=lambda t: t.id)
    def test_the_start_fails_and_the_solution_passes(self, task, tmp_path):
        start, solved = tmp_path / "start", tmp_path / "solved"
        for d in (start, solved):
            _lay(d, task.files)
            _lay(d, task.hidden)
        _lay(solved, task.solution)
        assert _suite(start) != 0
        assert _suite(solved) == 0

    @pytest.mark.parametrize("task", TASKS + LONG, ids=lambda t: t.id)
    def test_the_grader_uses_tests_the_agent_never_saw(self, task):
        assert not set(task.hidden) & set(task.files)
        assert all(p.startswith("tests_hidden/") for p in task.hidden)


class TestTheArms:
    def test_the_baseline_runs_every_part_in_one_session(self):
        multi = next(t for t in TASKS if t.multi)
        (only,) = trial.prompts(multi, trial.ARMS["baseline"])
        assert all(p in only for p in multi.parts)

    def test_an_adder_arm_restarts_per_part_with_a_handoff(self):
        multi = next(t for t in TASKS if t.multi)
        got = trial.prompts(multi, trial.ARMS["adder-sonnet"])
        assert len(got) == len(multi.parts)
        assert "already did" not in got[0] and multi.parts[0] in got[1]

    def test_only_adder_arms_get_the_hooks(self, tmp_path):
        assert trial.settings_for(trial.ARMS["baseline"], tmp_path) == {}
        hooks = trial.settings_for(trial.ARMS["adder-sonnet"], tmp_path)["hooks"]
        assert "PreToolUse" in hooks and "PostToolUse" in hooks

    def test_every_session_is_capped_and_unpersisted(self, tmp_path):
        cmd = trial.command(trial.ARMS["adder-sonnet"], "x", tmp_path / "s.json", 2.0)
        assert "--no-session-persistence" in cmd
        assert cmd[cmd.index("--max-budget-usd") + 1] == "2.00"
        assert cmd[cmd.index("--effort") + 1] == "medium"


class TestItSpendsNothingUnasked:
    def test_a_dry_run_starts_no_session(self, monkeypatch, capsys):
        monkeypatch.setattr(trial, "run_one", lambda *a, **k: pytest.fail("ran"))
        assert trial.main([]) == 0
        assert "DRY RUN" in capsys.readouterr().out

    def test_run_needs_somewhere_to_write(self, monkeypatch, capsys):
        monkeypatch.setattr(trial, "run_one", lambda *a, **k: pytest.fail("ran"))
        assert trial.main(["--run"]) == 2

    def test_the_budget_stops_the_run(self, monkeypatch, tmp_path, isolated_home):
        calls = []

        def fake(task, arm, rep, **k):
            calls.append(k["spend_left"])
            return trial.Result(task.id, arm.name, rep, True, 3.0, 1, 1, 1.0)

        monkeypatch.setattr(trial, "run_one", fake)
        monkeypatch.setattr(trial.shutil, "which", lambda _: "/bin/claude")
        out = tmp_path / "r.jsonl"
        trial.main(["--run", "--budget", "7", "--out", str(out)])
        assert len(calls) == 3 and calls[-1] == pytest.approx(1.0)
        assert len(out.read_text().splitlines()) == 3


class TestTheReport:
    def _r(self, arm, task, passed, cost):
        return trial.Result(task, arm, 0, passed, cost, 1, 5, 1.0)

    def test_cost_is_compared_on_the_same_tasks(self):
        rs = [self._r("baseline", "a", True, 1.0), self._r("baseline", "b", True, 9.0),
              self._r("adder-sonnet", "a", True, 0.25)]
        text = trial.report(rs)
        assert "4.00x cheaper" in text               # a against a only, not against 10.0

    def test_it_round_trips_through_the_file(self, tmp_path):
        out = tmp_path / "r.jsonl"
        out.write_text(json.dumps(vars(self._r("baseline", "a", False, 0.5))) + "\n")
        assert trial.load(out)[0].cost == 0.5


class TestTheCascade:
    """Cheap first, and the strong model only when the tests a person can see
    still fail. Never decided on the hidden tests."""

    def _fake(self, monkeypatch, visible):
        from types import SimpleNamespace

        models = []

        def run(cmd, **kw):
            models.append(cmd[cmd.index("--model") + 1])
            return SimpleNamespace(stdout=json.dumps({"total_cost_usd": 0.1, "num_turns": 3}))

        monkeypatch.setattr(trial.subprocess, "run", run)
        monkeypatch.setattr(trial, "visible_pass", lambda work: visible)
        monkeypatch.setattr(trial, "grade", lambda work, task: (True, "ok"))
        return models

    def test_it_escalates_only_when_the_visible_tests_fail(self, monkeypatch, tmp_path):
        task = next(t for t in TASKS if not t.multi)
        models = self._fake(monkeypatch, visible=False)
        r = trial.run_one(task, trial.ALL_ARMS["adder-cascade"], 0, workroot=tmp_path)
        assert models == ["claude-haiku-4-5", "claude-sonnet-5"]
        assert r.sessions == 2 and r.cost == pytest.approx(0.2) and "escalated" in r.detail

    def test_a_passing_cheap_run_is_not_escalated(self, monkeypatch, tmp_path):
        task = next(t for t in TASKS if not t.multi)
        models = self._fake(monkeypatch, visible=True)
        trial.run_one(task, trial.ALL_ARMS["adder-cascade"], 0, workroot=tmp_path)
        assert models == ["claude-haiku-4-5"]

    def test_haiku_arms_set_no_effort(self, tmp_path):
        cmd = trial.command(trial.ALL_ARMS["adder-haiku"], "x", tmp_path / "s.json", 1.0)
        assert "--effort" not in cmd
