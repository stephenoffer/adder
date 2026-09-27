"""`nothing_found` has to say which kind of empty this is.

A first run that finds nothing is the one moment a bare sentence is least
useful, and the two causes want opposite advice: an over-narrow window wants
the filter widened, an empty root wants a directory or `auto on`. Printing one
message for both is what these tests exist to prevent.
"""

from __future__ import annotations

from adder.util.render import nothing_found


def test_no_filter_points_at_the_root_and_at_activation():
    out = nothing_found("priced turns", "/tmp/projects")
    assert "No priced turns found under /tmp/projects." in out
    assert "matching" not in out
    assert ".adder.json" in out
    assert "adder auto on --full" in out


def test_an_active_filter_blames_the_filter_and_nothing_else():
    out = nothing_found("sessions", "/tmp/projects", window="since 2030-01-01")
    assert "matching since 2030-01-01." in out
    assert "excluded by that filter" in out
    # The activation pitch is wrong here: there is history, the window hid it.
    assert "auto on" not in out


def test_the_no_op_window_reads_as_no_filter():
    # `Window.describe()` returns this sentence rather than an empty string,
    # and treating it as a filter would tell someone to widen a window they
    # never narrowed.
    out = nothing_found("priced turns", "/tmp/projects", window="everything on disk")
    assert "matching" not in out
    assert "adder auto on --full" in out
