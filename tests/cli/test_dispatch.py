"""The dispatcher and the command table it resolves against.

These tests exist because the failure modes here are silent: a command that
falls out of the table still works via `python -m adder.<path>` and nobody
notices it vanished from `adder help`.
"""

from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

from adder import __version__
from adder.cli import BY_NAME, COMMANDS, main, usage
from adder.cli.commands import START_HERE

REPO = pathlib.Path(__file__).resolve().parents[2]


class TestCommandTable:
    def test_names_are_unique(self):
        names = [c.name for c in COMMANDS]
        assert len(names) == len(set(names))

    @pytest.mark.parametrize("cmd", COMMANDS, ids=lambda c: c.name)
    def test_module_imports_and_exposes_main(self, cmd):
        mod = importlib.import_module(cmd.module)
        assert callable(mod.main), f"{cmd.module} has no callable main()"

    @pytest.mark.parametrize("cmd", COMMANDS, ids=lambda c: c.name)
    def test_help_exits_zero(self, cmd, capsys):
        # argparse raises SystemExit(0) for --help; anything else is a broken parser.
        with pytest.raises(SystemExit) as e:
            main([cmd.name, "--help"])
        assert e.value.code == 0
        assert capsys.readouterr().out.strip()

    @pytest.mark.parametrize("cmd", COMMANDS, ids=lambda c: c.name)
    def test_appears_in_usage(self, cmd):
        assert cmd.name in usage()

    def test_every_adder_module_with_a_main_is_registered(self):
        """A new report module must be reachable from `adder`, not just importable."""
        registered = {c.module for c in COMMANDS}
        exempt = {
            "adder.cli",        # the dispatcher itself
            "adder.__main__",   # `python -m adder`
            "adder.pricing.sources",    # reachable as `adder models refresh`, not top-level
        }
        # The hooks are invoked by the harness by path, not by a user typing a
        # command name, and `adder auto on` is how they get installed. They have
        # a `main()` because they are scripts, not because they are commands.
        exempt |= {
            ".".join(path.relative_to(REPO).with_suffix("").parts)
            for path in (REPO / "adder" / "decide" / "hooks").glob("*.py")
        }
        # `rglob`, not `glob`: the package is a tree, and a report module added
        # three levels down is exactly as invisible as one added at the top.
        unregistered = []
        for path in sorted((REPO / "adder").rglob("*.py")):
            if path.name.startswith("_"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            has_main = any(
                isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body
            )
            if not has_main:
                continue
            mod = ".".join(path.relative_to(REPO).with_suffix("").parts)
            if mod in exempt or mod in registered:
                continue
            unregistered.append(mod)
        assert not unregistered, (
            f"{unregistered} define main() but are not in COMMANDS in "
            "adder/cli/commands.py, so `adder help` does not know they exist"
        )


class TestDispatch:
    def test_no_args_prints_usage(self, capsys):
        assert main([]) == 0
        assert "adder <command>" in capsys.readouterr().out

    @pytest.mark.parametrize("flag", ["help", "-h", "--help"])
    def test_help_flags(self, flag, capsys):
        assert main([flag]) == 0
        assert "Measure" in capsys.readouterr().out

    @pytest.mark.parametrize("flag", ["version", "-V", "--version"])
    def test_version_flags(self, flag, capsys):
        assert main([flag]) == 0
        assert __version__ in capsys.readouterr().out

    def test_unknown_command_is_nonzero_and_suggests(self, capsys):
        assert main(["tarce"]) == 2
        err = capsys.readouterr().err
        assert "unknown command" in err
        assert "trace" in err

    def test_unknown_command_with_no_near_match(self, capsys):
        assert main(["zzzzzzzz"]) == 2
        assert "unknown command" in capsys.readouterr().err

    def test_help_for_a_command_forwards_to_its_parser(self, capsys):
        with pytest.raises(SystemExit) as e:
            main(["help", "trace"])
        assert e.value.code == 0
        assert "usage" in capsys.readouterr().out.lower()


class TestShortHelp:
    """The screen a first run sees. It used to be all sixty commands with their
    flag syntax, and the one a new user needed was fifty lines down."""

    @pytest.mark.parametrize("entry", START_HERE, ids=lambda e: e[0])
    def test_every_start_here_entry_is_a_real_command(self, entry):
        assert entry[0].split()[0] in BY_NAME

    def test_leads_with_start_here_and_fits_a_screen(self):
        text = usage()
        assert text.index("Start here") < text.index("Every command")
        assert len(text.splitlines()) <= 40

    def test_does_not_repeat_the_flag_reference(self):
        # One command's flag syntax in the short screen means all of them are.
        assert BY_NAME["guard"].usage not in usage()
        assert BY_NAME["guard"].usage in usage(full=True)

    def test_help_all_prints_the_reference(self, capsys):
        assert main(["help", "--all"]) == 0
        out = capsys.readouterr().out
        assert all(c.summary in out for c in COMMANDS)


class TestHelpTopic:
    def test_a_misspelt_topic_is_an_unknown_command(self, capsys):
        """`adder help lve` exited 0 on the short screen; `adder lve` exits 2."""
        assert main(["help", "lve"]) == 2
        err = capsys.readouterr().err
        assert "unknown command 'lve'" in err
        assert "Did you mean: live" in err

    @pytest.mark.parametrize("word", ["version", "help"])
    def test_the_words_usage_lists_are_not_unknown(self, word, capsys):
        assert main(["help", word]) == 0
        assert "unknown command" not in capsys.readouterr().err


class TestBadConfig:
    """A config file that does not parse is a usage error, not a traceback."""

    @pytest.mark.parametrize("text, needle", [
        ("{bad", ".adder.json"),
        ('{"budget": "lots"}', "budget"),
    ])
    def test_every_command_names_the_file_and_exits_2(
            self, isolated_home, text, needle, capsys):
        (isolated_home.parent / ".adder.json").write_text(text)
        assert main(["budget", str(isolated_home / "none")]) == 2
        err = capsys.readouterr().err
        assert err.startswith("adder budget: ")
        assert needle in err
        assert "Traceback" not in err

    def test_hook_still_fails_open(self, isolated_home, monkeypatch, capsys):
        """2 means "block" to Claude Code; a config typo must not deny a tool call."""
        import io

        (isolated_home.parent / ".adder.json").write_text("{bad")
        monkeypatch.setattr("sys.stdin", io.StringIO(
            '{"tool_name": "Read", "tool_input": {"file_path": "/x"}}'))
        assert main(["hook", "read-guard"]) == 0
