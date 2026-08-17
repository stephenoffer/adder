"""Completion scripts are generated, so the test is that they cannot go stale.

A hand-written completion file drifts the day a command is added and nobody
notices for a year. These assert the generated script covers the live command
table and the live parsers, which is the property that makes generation worth
doing at all.
"""

from __future__ import annotations

import pytest

from adder.cli import COMMANDS
from adder.cli.completion import SHELLS, flags_for, script


class TestCoverage:
    @pytest.mark.parametrize("shell", SHELLS)
    def test_every_command_appears(self, shell):
        text = script(shell)
        missing = [c.name for c in COMMANDS if c.name not in text]
        assert not missing, f"{shell} completion omits {missing}"

    @pytest.mark.parametrize("shell", SHELLS)
    def test_output_is_not_empty(self, shell):
        assert len(script(shell).strip().splitlines()) > 5

    def test_unknown_shell_is_an_error(self):
        with pytest.raises(ValueError):
            script("csh")

    def test_bash_defines_and_registers_the_function(self):
        text = script("bash")
        assert "_adder_complete()" in text
        assert "complete -F _adder_complete adder" in text

    def test_zsh_has_a_compdef_header_first(self):
        assert script("zsh").splitlines()[0] == "#compdef adder"

    def test_fish_emits_one_line_per_subcommand(self):
        text = script("fish")
        for c in COMMANDS:
            assert f"-a '{c.name}'" in text


class TestFlagDiscovery:
    def test_flags_come_from_the_real_parser(self):
        """`trace` grew `--by` and `--strict`; discovery must see both."""
        flags = flags_for("trace")
        assert {"--json", "--by", "--strict", "--since"} <= set(flags)

    def test_shared_window_flags_are_found_on_every_windowed_command(self):
        for name in ("tools", "sessions", "anomaly", "agents", "budget", "doctor"):
            assert "--since" in flags_for(name), name

    def test_an_unknown_command_yields_nothing_rather_than_raising(self):
        assert flags_for("no-such-command") == []

    def test_discovery_does_not_execute_the_report(self, capsys):
        """Building a parser must not run the command that owns it."""
        flags_for("trace")
        assert not capsys.readouterr().out.strip()

    def test_short_flags_are_not_offered(self):
        """`-h` alone is noise in a completion list; long forms are the contract."""
        assert all(f.startswith("--") for f in flags_for("trace"))


class TestCli:
    @pytest.mark.parametrize("shell", SHELLS)
    def test_named_shell_prints_that_script(self, shell, capsys):
        from adder.cli.completion import main

        assert main([shell]) == 0
        assert "adder" in capsys.readouterr().out

    def test_shell_is_guessed_from_the_environment(self, monkeypatch, capsys):
        from adder.cli.completion import main

        monkeypatch.setenv("SHELL", "/bin/zsh")
        assert main([]) == 0
        assert capsys.readouterr().out.startswith("#compdef adder")

    def test_an_unguessable_shell_is_a_usage_error(self, monkeypatch, capsys):
        from adder.cli.completion import main

        monkeypatch.setenv("SHELL", "/usr/bin/nonsense")
        assert main([]) == 2
        assert "name a shell" in capsys.readouterr().err

    def test_an_invalid_shell_argument_is_rejected_by_argparse(self):
        from adder.cli.completion import main

        with pytest.raises(SystemExit):
            main(["powershell"])
