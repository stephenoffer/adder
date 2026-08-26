"""A narrow code page must not cost the user the whole report.

`adder help` names a command as "re-fit the effort→output priors", `adder live`
warns with `⚠`, and `render.bar` draws with `█`. None of the three exist in
cp1252, which is what Python picks for a *redirected* stdout on Windows -- so
`adder help > out.txt` there raised UnicodeEncodeError and printed nothing at
all. The Windows leg of the CI matrix was the only place it showed, because an
actual Windows console does take UTF-8.

The fix is in `run`, next to the BrokenPipeError handler, for the same reason:
one place, one behaviour, every command.
"""

from __future__ import annotations

import io
import sys

import pytest

from adder.cli import _widen_output_encoding, run


def _narrow_stream() -> io.TextIOWrapper:
    """A stdout that cannot represent the characters the reports use."""
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="")


class TestWidening:
    def test_a_narrow_stream_is_moved_to_utf8(self, monkeypatch):
        out = _narrow_stream()
        monkeypatch.setattr(sys, "stdout", out)
        _widen_output_encoding()
        assert out.encoding.lower().replace("-", "") == "utf8"

    def test_a_utf8_stream_is_left_alone(self, monkeypatch):
        out = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", newline="")
        monkeypatch.setattr(sys, "stdout", out)
        _widen_output_encoding()
        assert out.errors == "strict", "a stream that was already fine was rewritten"

    def test_a_stream_that_cannot_be_reconfigured_is_not_an_error(self, monkeypatch):
        class Fixed:
            encoding = "cp1252"

            def reconfigure(self, **_):
                raise OSError("not a real stream")

        monkeypatch.setattr(sys, "stdout", Fixed())
        _widen_output_encoding()          # the report still has to be attempted

    def test_stderr_is_widened_too(self, monkeypatch):
        err = _narrow_stream()
        monkeypatch.setattr(sys, "stderr", err)
        _widen_output_encoding()
        assert err.encoding.lower().replace("-", "") == "utf8"


class TestTheCommandItself:
    def test_help_survives_a_cp1252_stdout(self, monkeypatch):
        """The failure as it happened: `python -m adder help` into a pipe."""
        out = _narrow_stream()
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "argv", ["adder", "help"])
        assert run() == 0
        out.flush()
        text = out.buffer.getvalue().decode("utf-8")
        assert "effort→output" in text, "the glyph should survive, not just the report"

    def test_the_arrow_would_have_raised_without_the_widening(self):
        """The premise, asserted rather than assumed: cp1252 has no arrow."""
        with pytest.raises(UnicodeEncodeError):
            "effort→output".encode("cp1252")
