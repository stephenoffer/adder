"""Home-derived defaults that follow `HOME` after import.

Six modules built a default path from `Path.home()` into a module constant:
the transcript root, the parse cache, the user config file, the outcome log,
the ledger and the uptake cache. `CLAUDE.md` forbids it, because a
`monkeypatch.setenv("HOME", ...)` runs after import and moves nothing already
computed. Only `isolated_home`, which patches two of the six by name, kept the
suite off the developer's own files, so any test that skipped the fixture read
`~/.claude/adder.json` and wrote a pickle into `~/.claude`.

The constants stay, because tests and `isolated_home` repoint them by
assignment and that has to keep working. What changes is how they are read:
through `live`, which returns the constant if something moved it and otherwise
resolves the home directory again, now.
"""

from __future__ import annotations

from pathlib import Path


class HomeDefault:
    """One home-relative default, and the value its constant had at import."""

    def __init__(self, *parts: str) -> None:
        self.parts = parts
        self.at_import = Path.home().joinpath(*parts)

    def fresh(self) -> Path:
        return Path.home().joinpath(*self.parts)

    def live(self, constant: Path | str) -> Path:
        """`constant` if it was repointed since import, else the default as of now."""
        c = Path(constant)
        return c if c != self.at_import else self.fresh()

    def is_default(self, path: Path | str) -> bool:
        """Is `path` the default, by its import-time value or its current one?"""
        p = Path(path).expanduser()
        return p in (self.at_import, self.fresh())
