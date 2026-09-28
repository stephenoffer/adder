"""`parse_iso` must read what transcripts write, identically on 3.10 and 3.11.

Every expected value here is spelled out rather than computed with
`fromisoformat`, because `fromisoformat` is the thing that disagrees with
itself across versions.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from adder.util.when import parse_iso

UTC = timezone.utc


class TestAccepts:
    @pytest.mark.parametrize("text, want", [
        ("2026-01-02T00:00:00.123456789Z", datetime(2026, 1, 2, 0, 0, 0, 123456, UTC)),
        ("2026-01-02T00:00:00.12Z", datetime(2026, 1, 2, 0, 0, 0, 120000, UTC)),
        ("2026-01-02T00:00:00.1+00:00", datetime(2026, 1, 2, 0, 0, 0, 100000, UTC)),
        ("2026-01-02T00:00:00+0000", datetime(2026, 1, 2, tzinfo=UTC)),
        ("2026-01-02T00:00:00Z", datetime(2026, 1, 2, tzinfo=UTC)),
        ("2026-01-02T00:00:00z", datetime(2026, 1, 2, tzinfo=UTC)),
        ("2026-01-02 03:04:05", datetime(2026, 1, 2, 3, 4, 5)),
        ("2026-01-02T03:04", datetime(2026, 1, 2, 3, 4)),
        ("2026-01-02", datetime(2026, 1, 2)),
    ])
    def test_forms_transcripts_write(self, text, want):
        assert parse_iso(text) == want

    @pytest.mark.parametrize("tz, minutes", [
        ("+0530", 330), ("+05:30", 330), ("-0800", -480), ("-08", -480), ("+00", 0),
    ])
    def test_offsets_with_and_without_a_colon(self, tz, minutes):
        t = parse_iso(f"2026-01-02T12:00:00{tz}")
        assert t.utcoffset() == timedelta(minutes=minutes)

    def test_fraction_is_truncated_not_rounded(self):
        """Rounding `.9999999` would carry into the next second, and the next day."""
        t = parse_iso("2026-01-02T23:59:59.9999999Z")
        assert (t.day, t.second, t.microsecond) == (2, 59, 999999)

    def test_no_offset_is_naive(self):
        assert parse_iso("2026-01-02T00:00:00.5").tzinfo is None


class TestRejects:
    @pytest.mark.parametrize("text", [
        "", "garbage", "2026-13-01T00:00:00Z", "2026-01-02T25:00:00Z",
        "2026-01-02T00:00:00+2400", "20260102T000000", "2026-01-02T00:00:00.Z",
        "2026-01-02Tx",
    ])
    def test_raises_value_error_like_fromisoformat(self, text):
        with pytest.raises(ValueError):
            parse_iso(text)

    def test_a_non_string_is_a_type_error(self):
        with pytest.raises(TypeError):
            parse_iso(None)  # type: ignore[arg-type]
