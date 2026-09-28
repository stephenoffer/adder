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


class TestUnreadFiles:
    """Files that are present but unparsed must not read as "no history yet".

    This is the first thing anyone not running Claude Code saw: point adder at
    a directory of Codex or Aider logs it cannot parse and the answer was
    advice to come back after a session or two, which is advice to wait for a
    problem waiting does not fix.
    """

    def test_present_but_unparsed_is_called_a_format_problem(self):
        out = nothing_found("priced turns", "/logs", unread=3)
        assert "3 files are there" in out
        assert "format problem" in out
        # The wait-and-retry advice belongs to the empty case only.
        assert "session or two" not in out

    def test_it_names_the_formats_it_can_read(self):
        out = nothing_found("priced turns", "/logs", unread=1)
        assert "1 file is there" in out
        for fmt in ("Claude Code", "OpenAI", "Gemini", "OpenTelemetry"):
            assert fmt in out

    def test_an_example_path_becomes_a_runnable_command(self):
        out = nothing_found("priced turns", "/logs", unread=2, example="/logs/a.jsonl")
        assert "adder trace /logs/a.jsonl" in out

    def test_a_filter_still_wins_over_the_format_message(self):
        # An active window explains the emptiness on its own; blaming the
        # format there would send someone to debug a parser that worked.
        out = nothing_found("sessions", "/logs", window="since 2030-01-01", unread=4)
        assert "excluded by that filter" in out
        assert "format problem" not in out

    def test_no_files_keeps_the_fresh_machine_message(self):
        out = nothing_found("priced turns", "/logs", unread=0)
        assert "session or two" in out
        assert "format problem" not in out
