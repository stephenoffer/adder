"""The rules in CLAUDE.md, as assertions rather than prose.

A rule that is only written down is a rule that is followed until the first
inconvenient afternoon. Everything here is cheap to check and expensive to
discover the hard way: a stray `import urllib` in a report module breaks the
offline guarantee without breaking any other test.
"""

from __future__ import annotations

import ast
import json
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

    def test_the_skills_and_the_plugin_hook_table_are_package_data(self):
        """The same failure one directory over: without a glob, the wheel
        carries none of the skills and `auto on` copies zero of them."""
        declared = pyproject()["tool"]["setuptools"]["package-data"]["adder.decide"]
        assert "skills/*/SKILL.md" in declared
        assert "hooks/hooks.json" in declared

    def test_the_repository_runs_the_skills_it_ships(self):
        """`.claude/skills/<installed name>` mirrors the packaged copy."""
        from adder.decide.auto import SKILLS, skills_dir

        drifted = [name for src, name in SKILLS.items()
                   if (REPO / ".claude" / "skills" / name / "SKILL.md").is_file()
                   and (REPO / ".claude" / "skills" / name / "SKILL.md").read_text(
                       encoding="utf-8")
                   != (skills_dir() / src / "SKILL.md").read_text(encoding="utf-8")]
        assert not drifted, (
            f"{drifted} differ between .claude/skills/ and adder/decide/skills/. "
            "The packaged one is what users get; copy it over."
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


class TestThePlugin:
    """The repository is a Claude Code plugin and its own marketplace.

    Claude Code reads the plugin straight out of a clone: there is no build
    step and no install hook in which to fix anything up. So every path the
    manifest names has to exist, the hook table has to be the one `auto.HOOKS`
    describes, and the launcher the hooks run has to be executable -- each of
    those failing is a plugin that installs cleanly and does nothing.
    """

    def manifest(self) -> dict:
        return json.loads((REPO / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))

    def test_its_version_is_the_package_version(self):
        assert self.manifest()["version"] == __version__

    def test_every_path_it_names_exists_inside_the_repo(self):
        m = self.manifest()
        paths = [*m["skills"], *m["agents"], m["hooks"]]
        for p in paths:
            assert p.startswith("./"), p
            full = (REPO / p).resolve()
            assert full.exists(), p
            assert REPO.resolve() in full.parents, p

    def test_it_ships_the_tier_agents_but_not_explore(self):
        """A plugin agent is namespaced, so a plugin `Explore` would be a second
        explorer rather than the override; activation installs that one."""
        from adder.decide.auto import AGENTS

        names = [pathlib.Path(p).name for p in self.manifest()["agents"]]
        assert names == list(AGENTS[1:])

    def test_its_hook_table_is_the_one_activation_describes(self):
        from adder.decide.auto import plugin_hooks

        path = REPO / self.manifest()["hooks"]
        assert json.loads(path.read_text(encoding="utf-8")) == plugin_hooks(), (
            "adder/decide/hooks/hooks.json is stale; regenerate it from "
            "adder.decide.auto.plugin_hooks()"
        )

    def test_the_launcher_its_hooks_run_is_executable(self):
        import os

        for p in ("bin/adder", "scripts/adder"):
            assert os.access(REPO / p, os.X_OK), p

    def test_the_marketplace_lists_this_repository(self):
        m = json.loads((REPO / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
        (entry,) = m["plugins"]
        assert entry["name"] == self.manifest()["name"]
        assert entry["source"] in (".", "./")


class TestTheBuildBackendFloor:
    """The declared `setuptools>=` must be able to build this project.

    `build-system.requires` said `setuptools>=68` while `[project]` declared
    `license = "MIT"` and `license-files` -- both PEP 639, which setuptools only
    understood from 77.0.0. Building with 76.1.0 dies on `project.license must
    be valid exactly by one definition`; 77.0.0 is the first that works.

    Nothing caught it because pip builds in an isolated environment and resolves
    the newest setuptools there, so the floor is never the version used. It *is*
    the version used by anyone installing the sdist with `--no-build-isolation`,
    or by a distro packaging this against a pinned toolchain, and a floor that
    cannot build the project is a false claim about what the package needs.
    """

    # The release that added PEP 639 support. Raise this only alongside a
    # metadata key that genuinely needs a newer setuptools.
    PEP639 = 77

    def _floor(self) -> tuple[str, int]:
        specs = [r for r in pyproject()["build-system"]["requires"]
                 if r.replace("_", "-").lower().startswith("setuptools")]
        assert len(specs) == 1, f"expected one setuptools requirement, got {specs}"
        spec = specs[0]
        assert ">=" in spec, f"{spec!r} declares no minimum; the floor is the claim"
        return spec, int(spec.split(">=")[1].split(".")[0].strip().rstrip(","))

    def test_the_floor_supports_the_license_metadata_actually_declared(self):
        project = pyproject()["project"]
        needs_pep639 = isinstance(project.get("license"), str) or "license-files" in project
        spec, major = self._floor()
        if needs_pep639:
            assert major >= self.PEP639, (
                f"{spec!r} cannot build this project: `license`/`license-files` are "
                f"PEP 639 and need setuptools>={self.PEP639}. Either raise the floor "
                "or go back to the `license = {text = ...}` table."
            )

    def test_the_floor_is_not_raised_past_what_is_used(self):
        """A floor set higher than necessary excludes builders for no reason."""
        _, major = self._floor()
        assert major <= self.PEP639, (
            "the floor is above the newest feature this project's metadata uses; "
            "if a newer setuptools is genuinely required, say which key needs it"
        )


class TestFixturesDoNotDependOnTheWallClock:
    """A fixture that means "fresh" may not say so with a literal date.

    `tests/decide/route/test_models.py` pinned a catalog to
    `refreshed_at: 2026-08-14`, which was one day old when it was written and
    44 days old by the time anyone noticed. The staleness limit is 21 days, so
    six weeks later `--if-stale` started fetching, the offline guard refused,
    and a test named after sockets failed for a reason that had nothing to do
    with sockets. CLAUDE.md already forbids wall-clock dependence; this is the
    one shape of it that rots silently instead of failing immediately.
    """

    def test_no_test_pins_a_catalog_timestamp_to_a_literal_date(self):
        import re

        pattern = re.compile(r'"refreshed_at"\s*:\s*"(\d{4})-')
        offenders = []
        for path in (REPO / "tests").rglob("test_*.py"):
            for i, line in enumerate(path.read_text().splitlines(), 1):
                m = pattern.search(line)
                # A deliberately ancient date is the "stale" fixture and cannot
                # rot any further; it is the recent-looking one that expires.
                if m and int(m.group(1)) >= 2024:
                    offenders.append(f"{path.relative_to(REPO)}:{i}")
        assert not offenders, (
            "these pin a catalog timestamp to a date that will age past the "
            "staleness limit; derive it from the clock instead: "
            + ", ".join(offenders))


class TestDocsOnlyNameRealCommands:
    """The reverse of the coverage check above, which only ran one way.

    `TestEveryCommandIsDocumented` fails when a command is missing from
    `docs/commands.md`. Nothing caught the opposite: prose naming a command
    that does not exist. `docs/agents.md` shipped a reference to `adder tiers`,
    which reads exactly like the real ones and sends the reader to an "unknown
    command" error. A doc that invents a command is worse than one that omits
    a real one, because it is confidently wrong.
    """

    # Dispatcher-level words that are not rows in COMMANDS.
    META = frozenset({"config", "help", "version", "auto", "hook", "completion"})

    def test_no_doc_references_a_command_that_does_not_exist(self):
        import re

        from adder.cli.commands import COMMANDS

        known = {c.name for c in COMMANDS} | self.META
        offenders = []
        docs = list(REPO.glob("*.md")) + list((REPO / "docs").glob("*.md"))
        for path in docs:
            for name in sorted(set(re.findall(r"`adder ([a-z][a-z0-9-]*)",
                                              path.read_text()))):
                if name not in known:
                    offenders.append(f"{path.relative_to(REPO)}: `adder {name}`")
        assert not offenders, (
            "these name a command that does not exist: " + ", ".join(offenders))


class TestDocsOnlyNameRealFlags:
    """A valid command with an invented flag reads exactly like working advice.

    `adder config --set harness=codex` shipped in two action lines the tool
    prints to users and in three places in the docs. `config` is a real
    command, so the check above passed it; `--set` has never existed. Following
    that instruction gets an argparse error, and the setting stays wrong.

    Flags are read out of each parser's own `--help` rather than from a list
    here, so this cannot drift: a renamed flag fails on the reference that
    still spells it the old way. Subcommands are walked too, because
    `adder outcomes import --write` hangs its flag off a subparser.
    """

    @staticmethod
    def _flags(cmd: str, sub: str | None, mods: dict, cache: dict):
        import contextlib
        import importlib
        import io
        import re

        key = (cmd, sub or "")
        if key in cache:
            return cache[key]
        name = mods.get(cmd)
        if not name:
            cache[key] = None
            return None
        buf = io.StringIO()
        try:
            module = importlib.import_module(name)
            argv = ([sub] if sub else []) + ["--help"]
            with (contextlib.redirect_stdout(buf),
                  contextlib.redirect_stderr(buf),
                  contextlib.suppress(SystemExit)):
                module.main(argv)
        except Exception:
            cache[key] = None
            return None
        text = buf.getvalue()
        cache[key] = set(re.findall(r"(--[a-z][a-z0-9-]*)", text)) if text else None
        return cache[key]

    def test_every_documented_flag_exists_on_its_command(self):
        import re

        from adder.cli.commands import COMMANDS

        mods = {c.name: c.module for c in COMMANDS}
        cache: dict = {}
        sources = (list((REPO / "adder").rglob("*.py"))
                   + list(REPO.glob("*.md"))
                   + list((REPO / "docs").glob("*.md")))
        offenders = []
        for path in sources:
            for cmd, rest in re.findall(r"`adder ([a-z][a-z0-9-]*)((?: [^`]*)?)`",
                                        path.read_text()):
                tokens = rest.split()
                sub = (tokens[0] if tokens and not tokens[0].startswith("-")
                       and tokens[0].isalpha() else "")
                for flag in re.findall(r"(--[a-z][a-z0-9-]*)", rest):
                    known = self._flags(cmd, sub or None, mods, cache)
                    if known is None and sub:
                        known = self._flags(cmd, None, mods, cache)
                    if known is not None and flag not in known:
                        offenders.append(
                            f"{path.relative_to(REPO)}: `adder {cmd}"
                            f"{' ' + sub if sub else ''} {flag}`")
        assert not offenders, (
            "these name a flag the command does not accept: "
            + ", ".join(sorted(set(offenders))))
