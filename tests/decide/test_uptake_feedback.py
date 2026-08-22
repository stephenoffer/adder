"""The measured uptake has to reach the gate, and must not be able to seal it shut.

`uptake()` has measured this for a long time and `adder auto status` has printed
it, but nothing consumed it: every advisory saving in the tool was discounted by
a flat 0.5 even on a machine that had measured its own rate. A measurement
nobody acts on is the same failure as a router nobody invokes.

Wiring it in creates one trap worth more than the rest of this file put
together. `advice_taken` gates whether advice is worth saying at all, so a
measured rate of zero stops the guard speaking — and a guard that does not speak
records no fires, so nothing can ever measure it again. The estimator would seal
itself shut on one bad week with no way back that did not involve editing a
config file nobody knows exists. `test_a_zero_measurement_cannot_seal_the_guard_shut`
is the assertion that the floor keeps that loop open.
"""
from __future__ import annotations

import json
import pathlib
import time

from adder.decide.guard import (
    UPTAKE_FLOOR,
    Settings,
    Uptake,
    load_uptake,
    save_uptake,
)


def write(path, *, rate, measured=True, ts=None):
    path.write_text(json.dumps({
        "rate": rate, "fires": 40, "changed": int(40 * rate),
        "measured": measured, "ts": time.time() if ts is None else ts,
    }), encoding="utf-8")
    return path


class TestCache:
    def test_round_trips_a_measurement(self, tmp_path):
        p = tmp_path / "u.json"
        save_uptake(Uptake(fires=40, changed=10), p)
        rate, measured, age = load_uptake(p)
        assert rate == 0.25 and measured is True
        assert age < 60

    def test_a_thin_measurement_is_not_marked_measured(self, tmp_path):
        p = tmp_path / "u.json"
        save_uptake(Uptake(fires=3, changed=3), p)
        _rate, measured, _age = load_uptake(p)
        assert measured is False, "three fires were treated as a measurement"

    def test_a_missing_cache_falls_back_to_the_assumption(self, tmp_path):
        rate, measured, age = load_uptake(tmp_path / "absent.json")
        assert (rate, measured) == (0.5, False)
        assert age == float("inf")

    def test_a_corrupt_cache_falls_back_rather_than_raising(self, tmp_path):
        p = tmp_path / "u.json"
        p.write_text("{not json", encoding="utf-8")
        assert load_uptake(p) == (0.5, False, float("inf"))

    def test_an_out_of_range_rate_is_refused(self, tmp_path):
        # A rate above 1 would inflate every advisory saving in the tool.
        p = tmp_path / "u.json"
        write(p, rate=1.4)
        assert load_uptake(p)[:2] == (0.5, False)
        write(p, rate=-0.2)
        assert load_uptake(p)[:2] == (0.5, False)

    def test_saving_never_raises_on_an_unwritable_path(self, tmp_path):
        # A cache is not a record: failing to write one must not take down the
        # command that happened to be refreshing it.
        save_uptake(Uptake(fires=40, changed=10), tmp_path / "nope" / "u.json")


class TestItReachesTheGate:
    def resolve(self, tmp_path, monkeypatch, *, rate, measured=True, env=None):
        p = write(tmp_path / "u.json", rate=rate, measured=measured)
        monkeypatch.setattr("adder.decide.guard.uptake_path", lambda: p)
        return Settings.resolve(cwd=tmp_path, env=env or {})

    def test_a_measured_rate_replaces_the_assumption(self, tmp_path, monkeypatch):
        cfg = self.resolve(tmp_path, monkeypatch, rate=0.8)
        assert cfg.advice_taken == 0.8, (
            "the measurement was taken and then not used, which is the bug")

    def test_an_unmeasured_cache_leaves_the_assumption_alone(self, tmp_path, monkeypatch):
        cfg = self.resolve(tmp_path, monkeypatch, rate=0.8, measured=False)
        assert cfg.advice_taken == 0.5

    def test_no_cache_at_all_behaves_exactly_as_before(self, tmp_path, monkeypatch):
        monkeypatch.setattr("adder.decide.guard.uptake_path",
                            lambda: tmp_path / "absent.json")
        assert Settings.resolve(cwd=tmp_path, env={}).advice_taken == 0.5

    def test_an_explicit_setting_beats_the_measurement(self, tmp_path, monkeypatch):
        # Somebody who wrote a number into config has said something the
        # estimator does not know. Overriding it silently would make the setting
        # decorative, which is the bug `Settings` exists to avoid.
        cfg = self.resolve(tmp_path, monkeypatch, rate=0.8,
                           env={"ADDER_GUARD_ADVICE_TAKEN": "0.2"})
        assert cfg.advice_taken == 0.2

    def test_a_broken_cache_cannot_take_the_guard_down(self, tmp_path, monkeypatch):
        def boom():
            raise OSError("no home directory")

        monkeypatch.setattr("adder.decide.guard.uptake_path", boom)
        # Resolving settings runs before every tool call. It must degrade.
        assert Settings.resolve(cwd=tmp_path, env={}).advice_taken == 0.5


class TestTheFloor:
    def resolve(self, tmp_path, monkeypatch, rate):
        p = write(tmp_path / "u.json", rate=rate)
        monkeypatch.setattr("adder.decide.guard.uptake_path", lambda: p)
        return Settings.resolve(cwd=tmp_path, env={})

    def test_a_zero_measurement_cannot_seal_the_guard_shut(self, tmp_path, monkeypatch):
        # The trap this whole file is about. At an uptake of 0 no advice ever
        # clears its own cost, so the guard stops speaking; a silent guard
        # records no fires; and with no fires nothing can re-measure. The floor
        # is what keeps a path back open.
        cfg = self.resolve(tmp_path, monkeypatch, 0.0)
        assert cfg.advice_taken >= UPTAKE_FLOOR > 0.0

    def test_a_very_low_measurement_is_lifted_to_the_floor(self, tmp_path, monkeypatch):
        assert self.resolve(tmp_path, monkeypatch, 0.01).advice_taken == UPTAKE_FLOOR

    def test_a_rate_above_the_floor_is_used_as_measured(self, tmp_path, monkeypatch):
        assert self.resolve(tmp_path, monkeypatch, 0.35).advice_taken == 0.35

    def test_a_perfect_rate_is_allowed(self, tmp_path, monkeypatch):
        # Uptake of 1.0 is a real possibility on an enforcing setup and must not
        # be clipped: clipping it would understate what advising is worth.
        assert self.resolve(tmp_path, monkeypatch, 1.0).advice_taken == 1.0

    def test_the_floor_still_discounts(self, tmp_path, monkeypatch):
        # The floor keeps the loop open; it must not become a way for advice
        # nobody follows to look worth saying.
        assert UPTAKE_FLOOR < 0.5


class TestProvenanceIsPrinted:
    def test_an_assumed_rate_is_labelled_assumed(self, tmp_path, monkeypatch):
        from adder.decide.guard import _uptake_line

        monkeypatch.setattr("adder.decide.guard.uptake_path",
                            lambda: tmp_path / "absent.json")
        line = _uptake_line(Settings())
        assert "ASSUMED" in line, (
            "an unmeasured 50% printed as a bare percentage reads as a finding")
        assert "--learn" in line, "the way to measure it is not named"

    def test_a_measured_rate_says_when(self, tmp_path, monkeypatch):
        from adder.decide.guard import _uptake_line

        p = write(tmp_path / "u.json", rate=0.7)
        monkeypatch.setattr("adder.decide.guard.uptake_path", lambda: p)
        line = _uptake_line(Settings(advice_taken=0.7))
        assert "measured" in line and "ASSUMED" not in line


class TestRefresh:
    def test_learning_writes_the_cache_the_gate_reads(self, tmp_path, monkeypatch):
        # The reason the measurement went unused: nothing wrote the cache.
        from adder.decide import guard

        p = tmp_path / "u.json"
        monkeypatch.setattr(guard, "uptake_path", lambda: p)
        monkeypatch.setattr(guard, "uptake",
                            lambda root=None, log=None: Uptake(fires=40, changed=28))
        guard.refresh_uptake(tmp_path)
        assert load_uptake(p)[0] == 0.7
        assert Settings.resolve(cwd=tmp_path, env={}).advice_taken == 0.7


class TestTheReportDoesNotGiveStaleAdvice:
    """The old report told the reader to lower `guard_advice_taken` by hand.

    That was right while nothing consumed the measurement and is wrong now: the
    gate picks it up on its own, so the instruction would be telling somebody to
    do a thing that has already happened.
    """

    def test_it_no_longer_tells_you_to_lower_the_setting(self):
        src = (pathlib.Path(__file__).parents[2]
               / "adder" / "decide" / "guard.py").read_text(encoding="utf-8")
        assert "lower guard_advice_taken" not in src, (
            "the report still instructs a manual change the gate now makes itself")
