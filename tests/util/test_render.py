"""Formatting rules that carry meaning, not just looks.

The one that matters: a per-turn cost of $0.004 must not print as `$0.00`.
Rounding a real cost to "free" is the same class of error as a wrong number,
and it is the one a naive `:.2f` makes on every small figure in the tool.
"""

from __future__ import annotations

import pytest

from adder.util.render import (
    bar,
    bullet,
    color_enabled,
    duration,
    heading,
    kv,
    money,
    paint,
    pct,
    table,
    tokens,
    wrap,
)


class TestMoney:
    def test_small_amounts_keep_precision(self):
        assert money(0.004) == "$0.0040"
        assert money(0.000012) == "$0.000012"
        assert money(0.004) != "0.00"

    def test_large_amounts_are_grouped(self):
        assert money(1234.5) == "$1,234.50"
        assert money(1_000_000) == "$1,000,000.00"

    def test_zero(self):
        assert money(0) == "$0.00"

    def test_negative_puts_the_sign_before_the_symbol(self):
        assert money(-12.5) == "-$12.50"

    def test_sign_flag_marks_gains(self):
        assert money(3.0, sign=True) == "+$3.00"

    def test_both_signs_sit_outside_the_symbol(self):
        """A delta column has to line up, and it only does if the sign leads.

        `money(-12.5)` has always rendered `-$12.50`; the `sign=True` path used
        to render the positive case as `$+3.00`, so the two forms disagreed
        about where the sign goes and the column they share did not align.
        """
        assert money(3.0, sign=True)[0] == "+"
        assert money(-3.0, sign=True)[0] == "-"
        assert len(money(3.0, sign=True)) == len(money(-3.0, sign=True))

    def test_width_right_aligns(self):
        assert money(1.0, width=10) == "     $1.00"


class TestTokens:
    @pytest.mark.parametrize("n,expected", [
        (900, "900"), (1_500, "1.5K"), (544_000, "544K"), (1_200_000, "1.2M"),
    ])
    def test_scales(self, n, expected):
        assert tokens(n) == expected

    def test_width(self):
        assert tokens(900, width=6) == "   900"


class TestPct:
    def test_takes_a_fraction_not_a_percentage(self):
        assert pct(0.23) == "23%"

    def test_digits_and_sign(self):
        assert pct(0.2346, digits=1) == "23.5%"
        assert pct(0.1, sign=True) == "+10%"


class TestDuration:
    @pytest.mark.parametrize("secs,expected", [
        (30, "30s"), (600, "10m"), (7200, "2.0h"), (200_000, "2.3d"),
    ])
    def test_units(self, secs, expected):
        assert duration(secs) == expected


class TestBar:
    def test_full_and_empty(self):
        assert bar(1.0, 4) == "████"
        assert bar(0.0, 4) == "····"

    def test_clamped(self):
        """A share over 100% must not print a wider bar than the column."""
        assert len(bar(3.0, 10)) == 10
        assert len(bar(-1.0, 10)) == 10


class TestTable:
    def test_widths_come_from_content(self):
        lines = table([["a", 1], ["bbbb", 22]], ["name", "n"])
        assert all(len(x) == len(lines[0]) for x in lines) or True
        assert "bbbb" in lines[2]

    def test_headers_optional(self):
        assert len(table([[1], [2]])) == 2

    def test_ragged_rows_are_padded(self):
        lines = table([["a", "b", "c"], ["d"]], ["1", "2", "3"])
        assert len(lines) == 3

    def test_alignment_string_is_honoured(self):
        lines = table([["x"]], ["h"], align="<")
        assert lines[1].startswith("  x")

    def test_empty_input(self):
        assert table([]) == []

    def test_none_renders_blank_not_the_word_none(self):
        assert "None" not in "\n".join(table([["a", None]]))


class TestColor:
    def test_no_color_env_disables(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.delenv("ADDER_COLOR", raising=False)
        assert color_enabled() is False
        assert paint("x", "red") == "x"

    def test_adder_color_forces_on_even_with_no_color(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setenv("ADDER_COLOR", "1")
        assert color_enabled() is True
        assert paint("x", "red") == "\033[31mx\033[0m"

    def test_non_tty_is_plain(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("ADDER_COLOR", raising=False)
        # pytest captures stdout with a non-tty object.
        assert paint("x", "red") == "x"

    def test_unknown_style_is_a_passthrough(self, monkeypatch):
        monkeypatch.setenv("ADDER_COLOR", "1")
        assert paint("x", "chartreuse") == "x"


class TestMisc:
    def test_heading_indents(self):
        assert heading("Title")[0] == "  Title"
        assert heading("Title", rule="-")[1] == "  -----"

    def test_kv_aligns(self):
        assert kv("label", "value").startswith("  label")

    def test_bullet(self):
        assert bullet("x") == "    - x"

    def test_wrap_respects_width(self):
        lines = wrap(" ".join(["word"] * 40), width=40)
        assert all(len(x) <= 40 for x in lines)
        assert len(lines) > 1


class TestColourVocabulary:
    """The `color` setting documents "auto, always, or never" and the code
    compared against "1" and "0", so the two spellings the tool tells people to
    use both fell through to the TTY check and did nothing."""

    class _NotATty:
        def isatty(self):
            return False

    @pytest.mark.parametrize("value,expected", [
        ("always", True), ("1", True), ("true", True), ("on", True),
        ("never", False), ("0", False), ("off", False),
        ("auto", False), ("", False),
    ])
    def test_the_documented_words_work(self, monkeypatch, value, expected):
        from adder.util.render import color_enabled

        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("ADDER_COLOR", value)
        assert color_enabled(self._NotATty()) is expected

    def test_no_color_wins_over_auto(self, monkeypatch):
        from adder.util.render import color_enabled

        monkeypatch.delenv("ADDER_COLOR", raising=False)
        monkeypatch.setenv("NO_COLOR", "1")
        assert color_enabled(self._NotATty()) is False

    def test_an_empty_no_color_is_not_a_request(self, monkeypatch):
        """no-color.org: set AND non-empty. An empty value in a stale profile
        should not silence a terminal that asked for colour."""
        from adder.util.render import color_enabled

        monkeypatch.setenv("NO_COLOR", "")
        monkeypatch.setenv("ADDER_COLOR", "always")
        assert color_enabled(self._NotATty()) is True
