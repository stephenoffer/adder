"""The dispatcher and the repo-level invariants it is supposed to preserve.

These tests exist because the failure modes here are silent: a command that
falls out of the table still works via `python -m router.<name>` and nobody
notices it vanished from `rt help`; a stray `import urllib` in a report module
breaks the offline guarantee without breaking a single other test.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
from typing import ClassVar

import pytest

from router import __version__
from router.cli import COMMANDS, main, usage

REPO = pathlib.Path(__file__).resolve().parent.parent


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

    def test_every_router_module_with_a_main_is_registered(self):
        """A new report module must be reachable from `rt`, not just importable."""
        registered = {c.module for c in COMMANDS}
        exempt = {
            "router.cli",        # the dispatcher itself
            "router.__main__",   # `python -m router`
            "router.sources",    # reachable as `rt models refresh`, not top-level
        }
        for path in sorted((REPO / "router").glob("*.py")):
            if path.name.startswith("_"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            has_main = any(
                isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body
            )
            if not has_main:
                continue
            mod = f"router.{path.stem}"
            if mod in exempt:
                continue
            assert mod in registered, (
                f"{mod} defines main() but is not in COMMANDS in router/cli.py"
            )


class TestDispatch:
    def test_no_args_prints_usage(self, capsys):
        assert main([]) == 0
        assert "rt <command>" in capsys.readouterr().out

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


class TestVersion:
    def test_is_pep440_ish(self):
        parts = __version__.split(".")
        assert len(parts) >= 2
        assert all(p[0].isdigit() for p in parts)

    def test_pyproject_reads_version_dynamically(self):
        """A hardcoded version in pyproject drifts from the package. Catch it."""
        import tomllib

        cfg = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
        assert "version" in cfg["project"].get("dynamic", []), (
            "pyproject must take version from router.__version__, not restate it"
        )


class TestRepoInvariants:
    """The rules in CLAUDE.md, as assertions rather than prose."""

    NETWORK_MODULES: ClassVar[set[str]] = {
        "urllib", "http", "socket", "requests", "httpx", "ftplib", "smtplib",
        "ssl", "asyncio",
    }

    def test_no_network_imports_outside_sources(self):
        offenders = []
        for path in sorted((REPO / "router").rglob("*.py")):
            if path.name == "sources.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    mods = [node.module]
                else:
                    continue
                for m in mods:
                    if m.split(".")[0] in self.NETWORK_MODULES:
                        offenders.append(f"{path.name}:{node.lineno} imports {m}")
        assert not offenders, (
            "router/sources.py is the only module allowed to reach the network:\n"
            + "\n".join(offenders)
        )

    def test_no_runtime_dependencies(self):
        import tomllib

        cfg = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
        assert cfg["project"]["dependencies"] == [], (
            "the tool must run from a bare checkout; see CONTRIBUTING.md"
        )

    def test_governance_files_exist(self):
        for name in ("LICENSE", "README.md", "CHANGELOG.md", "CONTRIBUTING.md",
                     "SECURITY.md", "CLAUDE.md"):
            assert (REPO / name).is_file(), f"missing {name}"

    def test_changelog_has_an_unreleased_section(self):
        text = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
        assert "## [Unreleased]" in text

    def test_changelog_documents_the_current_version(self):
        text = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
        assert f"## [{__version__}]" in text, (
            f"CHANGELOG has no section for {__version__}; the release workflow "
            "refuses to tag without one"
        )

    def test_every_command_is_in_the_docs(self):
        docs = (REPO / "docs" / "commands.md").read_text(encoding="utf-8")
        missing = [c.name for c in COMMANDS if f"rt {c.name}" not in docs]
        assert not missing, f"undocumented in docs/commands.md: {missing}"
