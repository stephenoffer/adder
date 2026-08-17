# Contributing

Thanks for looking. This project measures money, so the review bar is about
correctness of numbers more than style.

If you are an AI agent working in this repo, read [CLAUDE.md](CLAUDE.md) first —
it is the binding version of everything below.

## Setup

```bash
git clone https://github.com/stephenoffer/adder && cd adder
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make check          # lint + tests; should be clean before you change anything
```

No install is strictly required — `./scripts/adder help` works from a bare
checkout on Python 3.10+. The editable install is only to get `adder` on your
`PATH` and the dev tools available.

## The one rule that matters

**A wrong number is worse than no number.** Anything that changes a reported
figure needs the measurement behind it, in the PR description. "This looks more
correct" is not a reason; "re-ran `adder validate` on 171 files, the ratio moved
from 1.02 to 0.98, here is the output" is.

## Before you open a PR

```bash
make check
```

That runs `ruff check .` and `python3 -m pytest`. Both must be clean. CI runs
the same thing on Python 3.10 through 3.14, so a green local run should mean a
green CI run.

`ruff format` is intentionally not part of the gate. Several modules hand-align
price and capability tables into columns and the formatter destroys that for no
readability gain. Run `make fmt` to apply lint autofixes; match the surrounding
style by hand for the rest.

Checklist:

- [ ] Tests pass, and new behaviour has a new test in the same commit.
- [ ] `ruff check .` is clean.
- [ ] No new runtime dependency (see below if you think you need one).
- [ ] No network call outside `adder/pricing/sources.py`.
- [ ] Tests do not read the real `~/.claude` directory.
- [ ] `git status` has no untracked junk.
- [ ] `CHANGELOG.md` has an entry under `## [Unreleased]` if the change is
      user-visible.
- [ ] Docs updated if you added or changed a command.

## Adding a command

1. `adder/<layer>/<subject>/<name>.py` with a
   `main(argv: list[str] | None = None) -> int` and its own `argparse` parser.
   Which layer, and why: `docs/structure.md`. Reports go under `measure/`,
   recommendations under `decide/`, checks of a recommendation under
   `evaluate/`. The foundation — `util`, `pricing`, `core` — holds no commands.
2. One `Command(...)` row in `COMMANDS` in `adder/cli/commands.py`.
3. `tests/<the same path>/test_<name>.py`.
4. A row in `docs/commands.md` and, if it is user-facing, in the README table.

If adding the file makes its directory the 13th Python file, split the
directory first — `tests/repo/test_structure.py` will fail otherwise, and the
message will say where.

The dispatcher deliberately does not re-declare per-command flags. Each module
owns its own parser so `adder <name> --help` is always accurate.

## Dependencies

`[project.dependencies]` is empty and staying that way. Zero-dependency means
the tool runs anywhere Python does, including inside a CI container that has no
package index reachable. If you have a case for adding one, open an issue before
the PR and make the argument there — it is a real conversation, just not one to
have for the first time in a diff.

Dev-only tools (`pytest`, `ruff`, `build`, `twine`) go in
`[project.optional-dependencies].dev` and are fair game.

## Test conventions

- The test tree mirrors the package tree, directory for directory:
  `adder/measure/window/cache.py` -> `tests/measure/window/test_cache*.py`.
  A `test_<module>_<aspect>.py` second file is fine; a test at the top of
  `tests/` is not.
- Build fixtures with `tmp_path`. Never touch the real transcript directory.
- Warnings are errors. If a dependency-free change raises a `DeprecationWarning`
  on a newer Python, fix the code rather than filtering the warning.
- No wall-clock or network dependence. A test that fails at midnight UTC is a
  failing test.
- If a test genuinely requires local transcripts, mark it
  `@pytest.mark.transcripts`; CI deselects those.

## Commit style

```
fix: keep highest output_tokens per message id

Partial streamed records carry a running count, so keeping the first
record undercounted output by 2.6% across 18,163 turns.
```

Prefixes: `fix`, `feat`, `docs`, `refactor`, `test`, `chore`, `perf`.
Imperative mood, no trailing period, one logical change per commit.

## Review

PRs need CI green and one approving review. Maintainers may ask for the
measurement behind a number more than once; that is the process working, not
distrust.

## Reporting a bug

Open an issue with the `adder` command you ran, the output you got, the output you
expected, and your Python version. **Do not paste raw transcript content** —
it contains your source code and prompts. Redact or synthesise.

## Security

Do not open a public issue for a security problem. See [SECURITY.md](SECURITY.md).
