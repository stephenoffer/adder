"""The rules in CLAUDE.md, as assertions rather than prose.

A rule that is only written down is a rule that is followed until the first
inconvenient afternoon. Everything here is cheap to check and expensive to
discover the hard way: a stray `import urllib` in a report module breaks the
offline guarantee without breaking any other test.
"""

from __future__ import annotations

import ast
import pathlib
from typing import ClassVar

import pytest

from adder import __version__
from adder.cli import COMMANDS

REPO = pathlib.Path(__file__).resolve().parents[2]


def pyproject() -> dict:
    """`pyproject.toml` as a dict, on every interpreter the project claims.

    `tomllib` is 3.11+, and the invariants parsed out of this file are the most
    expensive ones to break. Skipping them on 3.10 would leave the oldest
    supported interpreter unguarded, so the dev extra carries `tomli` there.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            pytest.skip("needs tomllib (3.11+) or tomli; run `pip install -e '.[dev]'`")
    return tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))


class TestVersion:
    def test_is_pep440_ish(self):
        parts = __version__.split(".")
        assert len(parts) >= 2
        assert all(p[0].isdigit() for p in parts)

    def test_pyproject_reads_version_dynamically(self):
        """A hardcoded version in pyproject drifts from the package. Catch it."""
        cfg = pyproject()
        assert "version" in cfg["project"].get("dynamic", []), (
            "pyproject must take version from adder.__version__, not restate it"
        )


class TestRepoInvariants:
    """The rules in CLAUDE.md, as assertions rather than prose."""

    NETWORK_MODULES: ClassVar[set[str]] = {
        "urllib", "http", "socket", "requests", "httpx", "ftplib", "smtplib",
        "ssl", "asyncio",
    }

    def test_no_network_imports_outside_sources(self):
        offenders = []
        for path in sorted((REPO / "adder").rglob("*.py")):
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
            "adder/pricing/sources.py is the only module allowed to reach the network:\n"
            + "\n".join(offenders)
        )

    def test_no_runtime_dependencies(self):
        cfg = pyproject()
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
        missing = [c.name for c in COMMANDS if f"adder {c.name}" not in docs]
        assert not missing, f"undocumented in docs/commands.md: {missing}"
