"""The aggregate check, its ranking rule, and its exit code.

`doctor` computes nothing itself — every check delegates to the module that
owns the measurement. So these tests are about the contract around the checks:
findings are ordered by dollars, materiality has a floor, `--strict` means
something stable, and a check that cannot run says SKIP rather than OK.
"""

from __future__ import annotations

import json
from pathlib import Path

from adder.evaluate.doctor import (
    MATERIAL_SHARE,
    MIN_MATERIAL,
    Check,
    _material,
    check_prices,
    report,
    run,
)


class TestMateriality:
    def test_absolute_floor(self):
        assert _material(MIN_MATERIAL, 1e9) is True
        assert _material(MIN_MATERIAL - 0.01, 1e9) is False

    def test_relative_floor_catches_small_workloads(self):
        assert _material(0.5, 1.0) is True          # half of everything
        assert _material(0.5, 1e6) is False

    def test_share_threshold_is_honoured(self):
        total = 100.0
        assert _material(total * MATERIAL_SHARE, total) is True


class TestTheGuardCheck:
    """The only finding in `doctor` about money that has not been spent yet,
    which is why an advisory guard is reported as a problem and not as a
    preference. Its whole saving is multiplied by a guess."""

    def test_an_uninstalled_guard_fails_and_points_at_activation(self, monkeypatch):
        from adder.evaluate import doctor
        monkeypatch.setattr("adder.decide.guard.installed_in", lambda *a, **k: [])
        got = doctor.check_guard()
        assert not got.ok and 'adder auto on' in got.action

    def test_an_installed_but_advisory_guard_is_also_a_finding(self, monkeypatch):
        from adder.evaluate import doctor
        monkeypatch.setattr("adder.decide.guard.installed_in",
                            lambda *a, **k: [Path('/somewhere/settings.json')])
        monkeypatch.setenv('ADDER_GUARD_ENFORCE', 'off')
        got = doctor.check_guard()
        assert not got.ok and '--full' in got.action

    def test_an_enforcing_guard_passes_that_gate(self, monkeypatch):
        from adder.evaluate import doctor
        monkeypatch.setattr("adder.decide.guard.installed_in",
                            lambda *a, **k: [Path('/somewhere/settings.json')])
        monkeypatch.setenv('ADDER_GUARD_ENFORCE', 'full')
        got = doctor.check_guard()
        assert 'advisory' not in got.headline


class TestChecks:
    def test_price_expiry_is_flagged_inside_the_window(self):
        from datetime import date

        c = check_prices(date(2026, 8, 15))         # Sonnet 5 intro ends 08-31
        assert c.ok is False
        assert any("2026-08-31" in d for d in c.detail)

    def test_price_expiry_is_quiet_outside_the_window(self):
        from datetime import date

        assert check_prices(date(2026, 1, 1)).ok is True

    def test_a_check_that_cannot_run_reports_skip(self):
        assert Check("x", True, "n/a", skipped=True).status == "SKIP"

    def test_status_strings(self):
        assert Check("x", True, "").status == "OK"
        assert Check("x", False, "").status == "FIX"


class TestQualityCheck:
    """The half of the thesis a cost report cannot see."""

    def _root(self, write_jsonl, *, calls=60, errors=0):
        records = []
        for i in range(calls):
            records.append({
                "type": "assistant", "sessionId": "s",
                "timestamp": f"2026-08-01T10:{i % 60:02d}:00Z",
                "message": {"id": f"m{i}", "model": "claude-opus-5",
                            "usage": {"input_tokens": 1,
                                      "cache_read_input_tokens": 5_000,
                                      "output_tokens": 50},
                            "content": [{"type": "tool_use", "id": f"u{i}",
                                         "name": "Bash", "input": {}}]}})
            block = {"type": "tool_result", "tool_use_id": f"u{i}", "content": "x"}
            if i < errors:
                block["is_error"] = True
            records.append({"type": "user", "sessionId": "s",
                            "timestamp": f"2026-08-01T10:{i % 60:02d}:30Z",
                            "message": {"content": [block]}})
        return write_jsonl(records)

    def test_a_healthy_tool_error_rate_passes(self, write_jsonl, make_sessions):
        from adder.evaluate.doctor import check_quality

        root = self._root(write_jsonl, calls=60, errors=1)
        assert check_quality(root, make_sessions(1, 20), 100.0).ok is True

    def test_a_high_tool_error_rate_fails_and_is_priced(self, write_jsonl,
                                                        make_sessions):
        from adder.evaluate.doctor import check_quality

        root = self._root(write_jsonl, calls=60, errors=30)
        c = check_quality(root, make_sessions(1, 20), 100.0)
        assert c.ok is False
        assert c.dollars > 0

    def test_too_few_calls_to_judge_does_not_fail(self, write_jsonl, make_sessions):
        """Three failures out of ten is noise, not a finding."""
        from adder.evaluate.doctor import check_quality

        root = self._root(write_jsonl, calls=10, errors=3)
        assert check_quality(root, make_sessions(1, 20), 100.0).ok is True

    def test_no_tool_calls_is_skipped_rather_than_passed(self, tmp_path,
                                                         make_sessions):
        from adder.evaluate.doctor import check_quality

        c = check_quality(tmp_path, make_sessions(1, 5), 10.0)
        assert c.skipped is True

    def test_the_other_proxies_are_reported_without_a_verdict(self, write_jsonl,
                                                              make_sessions):
        from adder.evaluate.doctor import check_quality

        root = self._root(write_jsonl, calls=60)
        c = check_quality(root, make_sessions(1, 20), 100.0)
        assert any("before/after" in d for d in c.detail)


class TestRun:
    def test_every_check_runs_on_a_synthetic_workload(self, tmp_path, make_sessions,
                                                      isolated_home):
        checks = run(tmp_path, make_sessions(3, 60))
        names = {c.name for c in checks}
        assert {"spend", "cache", "delegation", "anomalies", "horizon",
                "prices", "catalog", "quality"} <= names

    def test_findings_are_ordered_by_dollars(self, tmp_path, make_sessions,
                                             isolated_home):
        checks = run(tmp_path, make_sessions(3, 60))
        fixes = [c for c in checks if not c.ok and not c.skipped]
        assert fixes == sorted(fixes, key=lambda c: -c.dollars)

    def test_skipped_checks_sort_last(self, tmp_path, make_sessions, isolated_home):
        checks = run(tmp_path, make_sessions(2, 20))
        skipped_at = [i for i, c in enumerate(checks) if c.skipped]
        active_at = [i for i, c in enumerate(checks) if not c.skipped]
        assert not skipped_at or min(skipped_at) > max(active_at)

    def test_order_is_stable_between_runs(self, tmp_path, make_sessions,
                                          isolated_home):
        sessions = make_sessions(3, 40)
        a = [c.name for c in run(tmp_path, sessions)]
        b = [c.name for c in run(tmp_path, sessions)]
        assert a == b

    def test_a_clean_workload_has_no_actionable_findings(self, tmp_path, make_session,
                                                         isolated_home):
        """One tiny session: nothing here is worth a dollar of anybody's attention."""
        checks = run(tmp_path, {"s": make_session(3, base=1_000, growth=0)})
        material = [c for c in checks
                    if not c.ok and not c.skipped and c.dollars >= MIN_MATERIAL]
        assert not material


class TestReport:
    def test_names_every_check(self, tmp_path, make_sessions, isolated_home):
        text = report(run(tmp_path, make_sessions(2, 30)))
        assert "spend" in text and "cache" in text

    def test_says_so_when_nothing_is_wrong(self):
        text = report([Check("spend", True, "fine")])
        assert "Nothing material to fix" in text

    def test_warns_that_levers_overlap(self, tmp_path, make_sessions, isolated_home):
        text = report(run(tmp_path, make_sessions(3, 80)))
        if "at stake" in text:
            assert "does not save the sum" in text


class TestCli:
    def _root(self, write_jsonl, n=40, read=200_000):
        return write_jsonl([
            {"type": "assistant", "sessionId": "s",
             "timestamp": f"2026-08-01T10:{i:02d}:00Z",
             "message": {"id": f"m{i}", "model": "claude-opus-5",
                         "usage": {"input_tokens": 1,
                                   "cache_read_input_tokens": read + i * 1000,
                                   "output_tokens": 400}, "content": []}}
            for i in range(n)])

    def test_json_shape(self, write_jsonl, capsys, isolated_home):
        from adder.evaluate.doctor import main

        root = self._root(write_jsonl)
        assert main([str(root), "--json"]) in (0, 1)
        d = json.loads(capsys.readouterr().out)
        assert "checks" in d and "at_stake" in d
        assert all("status" in c for c in d["checks"])

    def test_strict_returns_one_when_something_failed(self, write_jsonl, capsys,
                                                      isolated_home):
        from adder.evaluate.doctor import main

        root = self._root(write_jsonl)
        rc = main([str(root), "--strict", "--json"])
        d = json.loads(capsys.readouterr().out)
        assert rc == (1 if d["failed"] else 0)

    def test_without_strict_it_always_returns_zero(self, write_jsonl, capsys,
                                                   isolated_home):
        from adder.evaluate.doctor import main

        assert main([str(self._root(write_jsonl))]) == 0

    def test_no_sessions_exits_one(self, tmp_path, capsys, isolated_home):
        from adder.evaluate.doctor import main

        assert main([str(tmp_path)]) == 1
        assert "No sessions" in capsys.readouterr().out
