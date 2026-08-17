"""The package layout, as assertions rather than a diagram in a doc.

This file exists because the previous layout was 50 modules in one directory,
and nothing stopped it: every individual commit that added one more file was
obviously fine. Breadth limits only work if something enforces them on the
commit that crosses the line, so the rules are here rather than in prose.

Three rules do the work:

* **Breadth caps.** No directory holds more than `MAX_MODULES` Python files or
  `MAX_SUBPACKAGES` subdirectories. Crossing either is the signal to add a
  level, not to keep going sideways.
* **Layering.** Packages are ordered, and an import may only point *down* the
  order. This is what keeps `core` importable by the prompt hook without
  dragging a report, an argparse parser, and a print routine along with it.
* **Mirroring.** The test tree has the same shape as the package tree, so the
  tests for a module are where you would look for them and nowhere else.

Each failure message says where the new file should go, because a rule that
only says "no" costs the next person twenty minutes.
"""

from __future__ import annotations

import ast
import pathlib
import re
from typing import ClassVar

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
PKG = REPO / "adder"
TESTS = REPO / "tests"

# Depth over breadth. Both numbers are arbitrary in the way a speed limit is
# arbitrary: what matters is that there is one and that it is checked.
MAX_MODULES = 12
MAX_SUBPACKAGES = 10

# Import may point down this list, never up. The order is a claim about what
# depends on what, and it is the reason `adder.util` can be read without
# knowing what a session is.
#
#   util      no domain at all -- formatting, statistics, token estimates
#   pricing   what a token costs, per provider
#   core      reading a transcript off disk, and the settings that govern it
#   measure   read-only reports over what core read
#   decide    turning a measurement into a recommendation
#   evaluate  checking whether the recommendation held up
#   cli       the surface that dispatches to all of it
LAYERS: dict[str, int] = {
    "adder.util": 0,
    "adder.pricing": 1,
    "adder.core": 2,
    "adder.measure": 3,
    "adder.decide": 4,
    "adder.evaluate": 5,
    "adder.cli": 6,
}

# The foundation is imported by everything, so it may not carry a command: a
# module that owns an argparse parser is a module that cannot be imported
# cheaply, and the PreToolUse hook pays that cost on every submit.
NO_COMMANDS_BELOW = ("adder.util", "adder.pricing", "adder.core")

# `tests/repo` checks the repository itself, so it mirrors no package.
TEST_ONLY_DIRS = {"repo"}

MODULE_NAME = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")


def directories(root: pathlib.Path) -> list[pathlib.Path]:
    out = [root]
    out += [p for p in root.rglob("*") if p.is_dir() and "__pycache__" not in p.parts]
    return sorted(out)


def modules(d: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p for p in d.glob("*.py"))


def package_of(dotted: str) -> str | None:
    """The layer a dotted module name belongs to, or None if it is not in one."""
    parts = dotted.split(".")
    for n in (3, 2):
        candidate = ".".join(parts[:n])
        if candidate in LAYERS:
            return candidate
    return None


def dotted_name(path: pathlib.Path) -> str:
    return ".".join(path.relative_to(REPO).with_suffix("").parts)


def imports_of(path: pathlib.Path) -> list[tuple[int, str]]:
    """Every `adder.*` module imported by this file, with its line number.

    Parsed rather than grepped: a dotted name inside a docstring is prose, and
    flagging it as a layering violation would train people to ignore this test.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if node.module.startswith("adder"):
                found.append((node.lineno, node.module))
        elif isinstance(node, ast.Import):
            found += [(node.lineno, a.name) for a in node.names if a.name.startswith("adder")]
    return found


class TestBreadth:
    """Depth over breadth, enforced at the directory that crosses the line."""

    @pytest.mark.parametrize("root", [PKG, TESTS], ids=["adder", "tests"])
    def test_no_directory_holds_too_many_modules(self, root):
        wide = {
            str(d.relative_to(REPO)): len(modules(d))
            for d in directories(root)
            if len(modules(d)) > MAX_MODULES
        }
        assert not wide, (
            f"more than {MAX_MODULES} Python files in one directory: {wide}. "
            "Group the related ones into a subpackage rather than adding a "
            f"{MAX_MODULES + 1}th file here -- see docs/structure.md."
        )

    @pytest.mark.parametrize("root", [PKG, TESTS], ids=["adder", "tests"])
    def test_no_directory_holds_too_many_subpackages(self, root):
        wide = {}
        for d in directories(root):
            subs = [p for p in d.iterdir() if p.is_dir() and p.name != "__pycache__"]
            if len(subs) > MAX_SUBPACKAGES:
                wide[str(d.relative_to(REPO))] = len(subs)
        assert not wide, (
            f"more than {MAX_SUBPACKAGES} subdirectories in one directory: {wide}"
        )

    def test_the_top_of_the_package_stays_thin(self):
        """`adder/` itself holds the entry points and nothing else.

        A module that lands here is a module with no stated home, and the next
        one lands beside it. There have been fifty.
        """
        loose = {p.name for p in modules(PKG)} - {"__init__.py", "__main__.py"}
        assert not loose, (
            f"{sorted(loose)} sit at the top of the package. Put each one in the "
            f"layer it belongs to: {', '.join(sorted(LAYERS))}."
        )


class TestPackages:
    def test_every_package_directory_is_a_package(self):
        missing = [
            str(d.relative_to(REPO))
            for d in directories(PKG)
            if modules(d) and not (d / "__init__.py").exists()
        ]
        assert not missing, f"directories with modules but no __init__.py: {missing}"

    def test_every_package_says_what_it_is_for(self):
        """An `__init__.py` with no docstring is a directory, not a package.

        The docstring is where the rule for what belongs in the package lives.
        Without it the next module lands in whichever directory has the
        shortest name.
        """
        silent = []
        for d in directories(PKG):
            init = d / "__init__.py"
            if not init.exists():
                continue
            if not ast.get_docstring(ast.parse(init.read_text(encoding="utf-8"))):
                silent.append(str(init.relative_to(REPO)))
        assert not silent, f"packages with no docstring explaining what belongs: {silent}"

    def test_module_names_are_lowercase_words(self):
        bad = [
            str(p.relative_to(REPO))
            for p in PKG.rglob("*.py")
            if not p.name.startswith("__") and not MODULE_NAME.match(p.stem)
        ]
        assert not bad, f"module names must be lowercase snake_case: {bad}"


class TestLayering:
    """An import may point down the layer order, never up."""

    def test_imports_only_point_downward(self):
        violations = []
        for path in sorted(PKG.rglob("*.py")):
            src = package_of(dotted_name(path))
            if src is None:
                continue
            for lineno, imported in imports_of(path):
                dst = package_of(imported)
                if dst is None or dst == src:
                    continue
                if LAYERS[dst] > LAYERS[src]:
                    violations.append(
                        f"{path.relative_to(REPO)}:{lineno} imports {imported} "
                        f"({src} may not import {dst})"
                    )
        assert not violations, (
            "upward imports break the layering:\n  " + "\n  ".join(violations)
            + "\nMove the shared piece down to the lower layer instead."
        )

    def test_the_foundation_carries_no_commands(self):
        from adder.cli import COMMANDS

        stranded = [
            c.name for c in COMMANDS
            if any(c.module.startswith(p + ".") or c.module == p for p in NO_COMMANDS_BELOW)
        ]
        assert not stranded, (
            f"{stranded} are commands whose module lives in the foundation. Keep the "
            "computation there and put the argparse parser in measure/decide/evaluate."
        )


class TestMirroring:
    """The test tree has the same shape as the package tree."""

    EXEMPT: ClassVar[set[str]] = TEST_ONLY_DIRS

    def test_every_package_with_modules_has_a_test_directory(self):
        missing = []
        for d in directories(PKG):
            real = [p for p in modules(d) if not p.name.startswith("__")]
            if not real:
                continue
            mirror = TESTS / d.relative_to(PKG)
            if not mirror.is_dir():
                missing.append(str(mirror.relative_to(REPO)))
        assert not missing, f"no test directory mirroring these packages: {missing}"

    def test_every_test_directory_mirrors_a_package(self):
        stray = []
        for d in directories(TESTS):
            if d == TESTS:
                continue
            rel = d.relative_to(TESTS)
            if rel.parts[0] in self.EXEMPT:
                continue
            if not (PKG / rel).is_dir():
                stray.append(str(d.relative_to(REPO)))
        assert not stray, (
            f"test directories with no package to match: {stray}. Either the package "
            "moved and the tests did not, or the directory needs adding to TEST_ONLY_DIRS."
        )

    def test_every_command_has_a_mirrored_test_file(self):
        """CLAUDE.md's rule for adding a command, enforced rather than stated.

        "write `adder/<layer>/<subject>/<name>.py` ... add
        `tests/<same path>/test_<name>.py`". Directory-level mirroring was
        checked and file-level mirroring was not, so `adder regret` shipped
        with no test file at all -- and two defects in it had nowhere to be
        caught: a cross-validation run over a length distribution its estimator
        was never fitted on, and an empty corpus naming a winning estimator.
        """
        from adder.cli.commands import COMMANDS

        missing = []
        for c in COMMANDS:
            parts = c.module.split(".")[1:]          # drop the package name
            folder = TESTS.joinpath(*parts[:-1])
            if not folder.is_dir():
                missing.append(f"{c.name}: no {folder.relative_to(REPO)}/")
                continue
            if not list(folder.glob(f"test_{parts[-1]}*.py")):
                missing.append(
                    f"{c.name}: expected "
                    f"{(folder / f'test_{parts[-1]}.py').relative_to(REPO)}")
        assert not missing, (
            "commands with no mirrored test file:\n  " + "\n  ".join(missing))

    def test_no_test_file_sits_at_the_top(self):
        """A test at `tests/` root belongs to no module, and stays there forever."""
        loose = sorted(p.name for p in TESTS.glob("test_*.py"))
        assert not loose, (
            f"{loose} are not in a mirror directory. Put each beside the package it "
            "covers: tests/<the module's package>/test_<module>.py."
        )
