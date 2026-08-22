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


class TestWhatTheWheelCarries:
    """Everything `adder auto on` installs has to survive `pip install`.

    It did not, for four releases. The hooks and the tier agents lived under
    `.claude/`, which `MANIFEST.in` prunes, so the wheel carried none of them:
    activation wrote three hook entries pointing at files that did not exist and
    copied zero agents, and said nothing, because `agent_plan` skips a source it
    cannot find. The only cost-*prevention* in the tool was inert for everybody
    who had not cloned the repository -- which is to say, for everybody the
    README's install line was written for.

    Nothing about that failure was loud. These are the assertions that make it
    loud: the payload lives inside the package, and anything in it that is not a
    module is named by a `package-data` glob.
    """

    def test_the_hooks_and_agents_live_inside_the_package(self):
        from adder.decide.auto import agents_dir, hooks_dir

        pkg = pathlib.Path(REPO / "adder").resolve()
        for d in (hooks_dir(), agents_dir()):
            assert pkg in d.resolve().parents or d.resolve() == pkg, (
                f"{d} is outside adder/, so a wheel cannot carry it and "
                "`pip install adder-cli && adder auto on` installs nothing"
            )

    def test_every_installed_file_exists_where_activation_looks_for_it(self):
        from adder.decide.auto import AGENTS, HOOKS, agents_dir, hooks_dir

        missing = [str(hooks_dir() / h["script"]) for h in HOOKS
                   if not (hooks_dir() / str(h["script"])).is_file()]
        missing += [str(agents_dir() / a) for a in AGENTS
                    if not (agents_dir() / a).is_file()]
        assert not missing, f"activation would install nothing for: {missing}"

    def test_the_hooks_are_modules_so_setuptools_finds_them(self):
        """`packages.find` ships a package; it does not ship a stray directory."""
        from adder.decide.auto import hooks_dir

        assert (hooks_dir() / "__init__.py").is_file()

    def test_non_python_payload_is_declared_as_package_data(self):
        """A `.md` is invisible to `packages.find`, so it needs a glob naming it."""
        from adder.decide.auto import AGENTS, agents_dir

        globs = pyproject()["tool"]["setuptools"]["package-data"]
        rel = agents_dir().resolve().relative_to(pathlib.Path(REPO / "adder").resolve())
        owner = "adder." + ".".join(rel.parts[:-1]) if len(rel.parts) > 1 else "adder"
        declared = globs.get(owner) or globs.get(f'"{owner}"') or []
        assert any(g.startswith(f"{rel.parts[-1]}/") for g in declared), (
            f"pyproject declares {declared!r} for {owner}; nothing there carries "
            f"{AGENTS[0]} into the wheel"
        )

    def test_the_repository_runs_the_agents_it_ships(self):
        """`.claude/agents/` here is a copy of what activation installs.

        Two copies of four files, on purpose. The package needs them because
        that is what a wheel can carry; this checkout needs them at
        `.claude/agents/` because that is where Claude Code looks while somebody
        is working in this repository, and a fresh clone should be dogfooding
        what it ships rather than something adjacent to it. Copies drift, so the
        test is here rather than the trust.
        """
        from adder.decide.auto import AGENTS, agents_dir

        drifted = []
        for name in AGENTS:
            mine = REPO / ".claude" / "agents" / name
            if not mine.is_file():
                continue
            if mine.read_text(encoding="utf-8") != \
                    (agents_dir() / name).read_text(encoding="utf-8"):
                drifted.append(name)
        assert not drifted, (
            f"{drifted} differ between .claude/agents/ and the packaged copy in "
            "adder/decide/agents/. The packaged one is what users get; copy it over."
        )

    def test_nothing_installable_is_read_out_of_dot_claude(self):
        """The directory the manifest prunes may not be a source of payload.

        `.claude/hooks/*.py` still exists as forwarding shims for a settings.json
        written before the move. A shim is a few lines; if one of these grows a
        decision again, it is a decision only a checkout has.
        """
        for path in sorted((REPO / ".claude" / "hooks").glob("*.py")):
            assert len(path.read_text(encoding="utf-8").splitlines()) < 30, (
                f"{path} is doing real work again; the wheel does not carry it"
            )
