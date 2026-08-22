# CLAUDE.md

Working agreement for any agent (or human) changing this repository. Read it
before the first edit. It is short on purpose; everything in it is a rule that
has already been violated at least once.

## What this is

`adder` reads Claude Code transcript files off the local disk and reports
what a session costs, where the cost comes from, and what each intervention is
worth. It is a measurement tool that happens to make routing recommendations.

That framing sets the bar: **a wrong number here is worse than no number.** The
failure mode of a cost tool is not a bad recommendation, it is a confident bad
recommendation that someone acts on.

## Ground rules

1. **No runtime dependencies.** `[project.dependencies]` stays empty. The tool
   must run from a bare Python 3.10+ checkout with no install step. If you
   believe you need a dependency, say so in the PR and expect to justify it.
2. **No network in library code.** `adder/pricing/sources.py` is the single exception
   and it is opt-in, explicit, and honours `ADDER_OFFLINE=1`. Everything
   else — every report, every gate, every test — is pure computation over local
   files. Never add an implicit fetch to make a report "more accurate".
3. **No mutation of user data.** The tool reads `~/.claude/projects/**`. It does
   not write there, rename there, or delete there. Ever. Outputs go to stdout or
   to a path the user named. The one exception is `adder auto on`, which writes
   `settings.json` and `.adder.json` — it prints the change first, keeps a
   `.adder.bak`, and `adder auto off` reverses it. Nothing else may grow a
   write path.
4. **Read-only means read-only.** `adder live`, `adder trace`, `adder debt`, `adder context`,
   `adder cache`, `adder quality`, `adder horizon` must never change state on disk.
5. **Every claim is testable or it is not made.** A number in the README, a
   docstring, or a report either comes from a function under test or it is
   labelled as an estimate with its assumptions written down. `adder validate`
   exists so the foundational claims can be re-run against new data; if you
   change a claim, change `validate.py` with it.

## Invariants you will be tempted to break

These are load-bearing. Each one cost real money to discover.

- **Deduplicate assistant records by `message.id`, keeping the highest
  `output_tokens`.** Claude Code writes one JSONL record per content block, each
  repeating the whole message's `usage`. Summing raw lines multi-counts most
  turns — it inflated the original numbers by 1.78x. See `trace.iter_file` and
  `tests/core/test_trace_dedup.py`.
- **Prices are date-aware.** `prices.py` resolves a rate for a *date*, not just
  a model, because introductory pricing expires. Never hardcode a rate inline.
- **Model resolution is longest-prefix, and handles the `[1m]` context suffix.**
  `claude-sonnet-4-6-*` must not resolve as `claude-sonnet-5`.
- **The cache is model-scoped.** Moving a warm session to a cheaper model
  invalidates the prefix and makes input ~2x more expensive. Any "just use a
  cheaper model" suggestion must clear this check before it is emitted.
- **A routing recommendation must clear its own overhead.** A routing turn
  re-reads the whole context. If the modelled saving is smaller than that, the
  correct output is "just do it", not a recommendation. See `policy.decide`.
- **Context limits gate feasibility before profitability.** A cheaper model that
  cannot hold the context is not an option at any price.

## Layout

The package is a tree of seven layers and **imports point down the list, never
up**. Full rules and reasoning: `docs/structure.md`.

```
adder/
  util/       no domain at all: render, stats, risk, text
  pricing/    what a token costs: prices, catalog, providers, registry, cost,
              sources (the only module that opens a socket), data/
  core/       reading a session off disk: trace, filters, settings, shapes
  measure/    read-only reports:  spend/  window/  session/
  decide/     measurement -> choice:  route/  track/  guard.py  handoff.py
              delegate.py (the tier a delegated step should run on)
              auto.py (the one module that writes a file the user did not name)
              hooks/ and agents/ — the payload `auto on` installs, in the
              package because the wheel prunes `.claude/`
  evaluate/   did it hold up:  replay/  claims/  doctor.py
  cli/        dispatcher, command table, help, completion, config
tests/        mirrors the package tree, directory for directory
docs/         the reasoning; the README is the summary of it
scripts/adder launcher for a checkout; delegates to adder.cli
.claude/      skills, and forwarding shims for hooks installed before v0.2.
              Not shipped: `MANIFEST.in` prunes it. Nothing installable may
              live here — see `tests/repo/test_invariants.py`
```

Three limits, all enforced by `tests/repo/test_structure.py`:

- **No more than 12 Python files or 10 subdirectories in one directory.**
  Crossing either means adding a level, not another file. Split by subject.
- **No upward imports.** If a lower layer needs something from a higher one,
  move the shared piece *down*; do not add a function-level import to hide it.
- **`util`, `pricing`, and `core` carry no commands.** Everything imports the
  foundation, so it may not drag an argparse parser in. This is why `trace` is
  two files: `core/trace.py` reads transcripts, `measure/spend/trace.py` is the
  report.

**Adding a command:** write `adder/<layer>/<subject>/<name>.py` with a
`main(argv) -> int`, add one `Command(...)` row to `COMMANDS` in
`adder/cli/commands.py`, add `tests/<same path>/test_<name>.py`, and add a row
to `docs/commands.md`. The dispatcher does not duplicate flags — each module
owns its own parser.

## Style

- `from __future__ import annotations` at the top of every module.
- Module docstrings explain **why the module exists and what it got wrong**, not
  what the functions are named. Match the density of the existing ones.
- Comments explain the non-obvious decision, not the syntax.
- Type-annotate public functions. `adder/py.typed` claims the package is typed;
  keep that honest.
- Imports inside the package are **absolute** (`from adder.pricing.cost import
  turn_cost`), never relative. Ruff `TID252` enforces it: in a tree this deep the
  number of dots is load-bearing and invisible, and breaks when a file moves.
- Exception classes end in `Error`. Single-capital locals are for symbols from
  the formula in the docstring above them, and nowhere else.
- `ruff check .` clean before commit. Line length 100. The formatter is
  deliberately not a gate — several tables in `pricing` are hand-aligned.

## Testing

- `python3 -m pytest` must pass before any commit. Warnings are errors
  (`filterwarnings = ["error"]`) — do not silence one to make a test go green.
- The test tree mirrors the package tree: `adder/measure/window/cache.py` is
  tested by `tests/measure/window/test_cache*.py`. `tests/repo/` is the one
  exception — it checks the repository itself. Two test files may share a
  basename because pytest runs with `--import-mode=importlib`; leave that flag
  alone.
- Tests must not read the real `~/.claude` directory. Build fixtures with
  `tmp_path`. A test that only passes on the author's machine is a broken test;
  if one genuinely needs local transcripts, mark it `@pytest.mark.transcripts`.
- Deterministic only: no wall-clock dependence, no network, no random seeds left
  unset.
- New behaviour ships with a test in the same commit.

## File hygiene

- Never commit: `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.DS_Store`,
  `dist/`, `build/`, `*.egg-info/`, `.env`, virtualenvs, or scratch output.
  `.gitignore` covers these — if you find yourself adding `-f`, stop.
- Scratch files, one-off scripts, and exploratory notebooks do not belong in the
  repo. Use a temp directory outside it.
- No transcript data in the repo. Real transcripts contain source code,
  file paths, and prompts. Fixtures are synthetic.
- `git status` should be clean of untracked noise before you open a PR. If a new
  file is deliberate, track it; if not, delete it.

## Commits and PRs

- Conventional-ish subject: `fix:`, `feat:`, `docs:`, `refactor:`, `test:`,
  `chore:`, `perf:`. Imperative mood, no trailing period, ~72 chars.
- The body explains **why**, and names the measurement if the change moves a
  number.
- One logical change per commit. Do not bundle a refactor with a fix.
- Never commit directly to `main`. Branch, PR, let CI pass.
- Do not commit or push unless the human asked. Staging and message drafting are
  fine; the push is theirs.

## Things an agent must not do here

- Do not "fix" a failing test by weakening its assertion or deleting it.
- Do not change a headline number in the README without re-running the
  measurement that produced it and updating `validate.py`.
- Do not add a dependency, a network call, or a background process.
- Do not touch `.claude/settings.local.json` or anything gitignored as local.
- Do not rewrite git history that has been pushed.
- Do not widen scope. If you find a second bug, report it; fix the one asked for.

## Quick commands

```bash
make help          # every task below, described
make test          # pytest
make lint          # ruff check
make fmt           # ruff check --fix + format
make check         # lint + test, what CI runs
make clean         # remove caches and build artifacts
./scripts/adder help  # the tool's own command list
```
