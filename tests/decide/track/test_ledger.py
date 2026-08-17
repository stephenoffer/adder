"""The solvency invariant. If these pass, `cost_with_adder <= baseline` is an
accounting identity rather than a slogan."""
from __future__ import annotations

import json
import time

import pytest

from adder.decide.track.ledger import (
    MIN_HAIRCUT,
    MIN_VERIFIED,
    Entry,
    Ledger,
    current,
    load,
    main,
    prune,
    record,
)


@pytest.fixture
def log(tmp_path):
    return tmp_path / "ledger.jsonl"


def _entries(n, predicted=1.0, worst=0.5, overhead=0.2, realized=None, **kw):
    return [Entry(action="delegate", predicted=predicted, worst=worst,
                  overhead=overhead, realized=realized, **kw) for _ in range(n)]


class TestRoundTrip:
    def test_records_and_reads_back(self, log):
        for e in _entries(3):
            record(e, log)
        assert len(load(log)) == 3

    def test_missing_log_is_empty_not_an_error(self, tmp_path):
        assert load(tmp_path / "nope.jsonl") == []

    def test_malformed_lines_are_skipped(self, log):
        record(_entries(1)[0], log)
        with log.open("a") as fh:
            fh.write("not json\n{}\n[]\n\n")
        # `{}` is dropped too: an entry with no action and no numbers is not a
        # recommendation, and defaulting it to zero would quietly credit the
        # ledger with a free row.
        assert len(load(log)) == 1

    def test_unknown_fields_from_a_newer_version_are_ignored(self, log):
        log.write_text(json.dumps({"action": "delegate", "predicted": 1.0,
                                   "worst": 0.5, "overhead": 0.1,
                                   "invented_later": 7}) + "\n")
        assert load(log)[0].predicted == 1.0

    def test_recording_never_raises(self, tmp_path):
        """Accounting must not be able to break routing."""
        record(_entries(1)[0], tmp_path / "no" / "such" / "dir" / "l.jsonl")

    def test_prune_keeps_the_newest(self, log):
        for i in range(10):
            record(Entry("delegate", 1.0, 0.5, 0.1, ts=float(i)), log)
        assert prune(log, keep=4) == 6
        assert len(load(log)) == 4
        assert min(e.ts for e in load(log)) == 6.0


class TestSolvency:
    def test_worth_more_than_the_asking_is_solvent(self):
        led = Ledger(_entries(5, worst=0.5, overhead=0.2))
        assert led.solvent and led.margin == pytest.approx(1.5)

    def test_overhead_beyond_the_guarantee_is_insolvent(self):
        led = Ledger(_entries(5, worst=0.05, overhead=0.2))
        assert not led.solvent and led.margin < 0

    def test_solvency_uses_the_worst_case_not_the_expectation(self):
        """The point of the invariant: an expectation can be wrong in the
        direction that costs money, and a worst case cannot."""
        led = Ledger(_entries(3, predicted=10.0, worst=0.01, overhead=1.0))
        assert led.promised > led.spent and not led.solvent

    def test_declined_recommendations_cost_nothing(self):
        led = Ledger(_entries(4, worst=0.0, overhead=5.0, accepted=False))
        assert led.spent == 0.0 and led.solvent

    def test_an_empty_ledger_holds_trivially(self):
        assert Ledger([]).solvent and "nothing has been spent" in Ledger([]).describe()


class TestHaircut:
    def test_no_verified_history_means_no_correction(self):
        assert Ledger(_entries(50)).haircut() == 1.0

    def test_under_delivery_scales_future_predictions_down(self):
        led = Ledger(_entries(MIN_VERIFIED, predicted=1.0, realized=0.6))
        assert led.haircut() == pytest.approx(0.6, rel=0.01)

    def test_over_delivery_earns_no_credit(self):
        """A model allowed to inflate itself on good news will eventually
        inflate itself on noise."""
        led = Ledger(_entries(MIN_VERIFIED, predicted=1.0, realized=4.0))
        assert led.haircut() == 1.0

    def test_it_is_floored(self):
        led = Ledger(_entries(MIN_VERIFIED, predicted=1.0, realized=0.001))
        assert led.haircut() == MIN_HAIRCUT

    def test_thin_evidence_does_not_throttle(self):
        led = Ledger(_entries(MIN_VERIFIED - 1, predicted=1.0, realized=0.1))
        assert led.haircut() == 1.0

    def test_recent_evidence_weighs_more(self):
        now = time.time()
        old = _entries(MIN_VERIFIED, predicted=1.0, realized=0.2,
                       ts=now - 365 * 86400)
        new = _entries(MIN_VERIFIED, predicted=1.0, realized=1.0, ts=now)
        assert Ledger(old + new).haircut(now=now) > 0.8

    def test_zero_predictions_cannot_divide_by_zero(self):
        assert Ledger(_entries(MIN_VERIFIED, predicted=0.0, realized=0.0)).haircut() == 1.0


class TestReport:
    def test_empty_report_exits_clean(self, log, capsys):
        assert main(["--log", str(log)]) == 0
        assert "Nothing recorded yet" in capsys.readouterr().out

    def test_insolvent_report_exits_nonzero(self, log, capsys):
        for e in _entries(3, worst=0.01, overhead=1.0):
            record(e, log)
        assert main(["--log", str(log)]) == 1
        assert "INSOLVENT" in capsys.readouterr().out

    def test_json_is_machine_readable(self, log, capsys):
        for e in _entries(3, realized=0.5):
            record(e, log)
        main(["--log", str(log), "--json"])
        d = json.loads(capsys.readouterr().out)
        assert d["recommendations"] == 3 and d["solvent"] is True

    def test_current_never_raises(self, tmp_path):
        assert current(tmp_path / "absent.jsonl").entries == []


class TestAHandEditedRowDoesNotBreakTheAccounting:
    """Every failure here is invisible: `policy._haircut` wraps this in a bare
    `except` and falls back to 1.0, so a broken ledger reads as a ledger with
    nothing to say.

    `outcomes._coerce_ts` already makes the ISO-string correction and explains
    why -- "every other timestamp in this repo is an ISO string, so a caller
    writing one here is a matter of time". The ledger carries the identical
    field and made no such correction.
    """

    def _log(self, tmp_path, rows):
        import json

        p = tmp_path / "led.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        return p

    def _row(self, **kw):
        base = {"action": "delegate", "predicted": 1.0, "worst": 0.5,
                "overhead": 0.1, "realized": 0.8, "ts": 1.0}
        base.update(kw)
        return base

    def test_an_iso_timestamp_is_read_as_one(self, tmp_path):
        from adder.decide.track.ledger import load

        rows = load(self._log(tmp_path, [self._row(ts="2026-08-01T10:00:00Z")]))
        assert isinstance(rows[0].ts, float) and rows[0].ts > 0

    def test_a_string_number_still_sums(self, tmp_path):
        from adder.decide.track.ledger import Ledger, load

        led = Ledger(load(self._log(tmp_path, [self._row(predicted="1.0"),
                                               self._row(predicted=2.0)])))
        assert led.promised == pytest.approx(3.0)

    def test_a_nonsense_row_is_zeroed_not_fatal(self, tmp_path):
        from adder.decide.track.ledger import Ledger, load

        led = Ledger(load(self._log(tmp_path, [
            self._row(), self._row(predicted=None, overhead="nope",
                                   worst=float("inf"), ts="bad")])))
        assert led.promised == pytest.approx(1.0)
        assert led.spent == pytest.approx(0.1)

    def test_prune_can_still_sort(self, tmp_path):
        from adder.decide.track.ledger import prune

        p = self._log(tmp_path, [self._row(ts="2026-08-01T10:00:00Z"),
                                 self._row(ts=1.0), self._row(ts=2.0)])
        assert prune(p, keep=2) == 1

    def test_the_haircut_survives_it(self, tmp_path):
        from adder.decide.track.ledger import Ledger, load

        rows = [self._row(ts=float(i), predicted="1.0", realized="0.5")
                for i in range(10)]
        led = Ledger(load(self._log(tmp_path, rows)))
        assert 0.2 <= led.haircut() <= 1.0
