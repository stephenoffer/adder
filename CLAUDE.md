# CLAUDE.md

Working agreement for any agent (or human) changing this repository. Read it
before the first edit. It is short on purpose; everything in it is a rule that
has already been violated at least once.

## What this is

`llm-router` reads Claude Code transcript files off the local disk and reports
what a session costs, where the cost comes from, and what each intervention is
worth. It is a measurement tool that happens to make routing recommendations.

That framing sets the bar: **a wrong number here is worse than no number.** The
failure mode of a cost tool is not a bad recommendation, it is a confident bad
recommendation that someone acts on.

## Ground rules

1. **No runtime dependencies.** `[project.dependencies]` stays empty. The tool
   must run from a bare Python 3.10+ checkout with no install step. If you
   believe you need a dependency, say so in the PR and expect to justify it.
2. **No network in library code.** `router/sources.py` is the single exception
   and it is opt-in, explicit, and honours `LLM_ROUTER_OFFLINE=1`. Everything
   else — every report, every gate, every test — is pure computation over local
   files. Never add an implicit fetch to make a report "more accurate".
3. **No mutation of user data.** The tool reads `~/.claude/projects/**`. It does
   not write there, rename there, or delete there. Ever. Outputs go to stdout or
   to a path the user named.
4. **Read-only means read-only.** `rt live`, `rt trace`, `rt debt`, `rt context`,
   `rt cache`, `rt quality`, `rt horizon` must never change state on disk.
5. **Every claim is testable or it is not made.** A number in the README, a
   docstring, or a report either comes from a function under test or it is
   labelled as an estimate with its assumptions written down. `rt validate`
   exists so the foundational claims can be re-run against new data; if you
   change a claim, change `validate.py` with it.

## Invariants you will be tempted to break

These are load-bearing. Each one cost real money to discover.

- **Deduplicate assistant records by `message.id`, keeping the highest
  `output_tokens`.** Claude Code writes one JSONL record per content block, each
  repeating the whole message's `usage`. Summing raw lines multi-counts most
  turns — it inflated the original numbers by 1.78x. See `trace.iter_file` and
  `tests/test_trace_dedup.py`.
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

```
router/            library + one module per report; each owns its own main(argv)
  cli.py           the only dispatcher; maps command name -> module, lazily
  cost.py          the cost model — the piece everything else calls
  prices.py        hand-maintained first-party Claude rates, date-aware
  catalog.py       provider-agnostic model data (bundled < cache < project)
  sources.py       the only module that opens a socket
  data/            bundled catalog snapshot shipped in the wheel
tests/             one test module per router module, same name
docs/              the reasoning; the README is the summary of it
scripts/rt         launcher for a checkout; delegates to router.cli
.claude/           agents, hooks, and skills — part of the product, tracked
```

**Adding a command:** write `router/<name>.py` with a `main(argv) -> int`, add
one `Command(...)` row to `COMMANDS` in `router/cli.py`, add `tests/test_<name>.py`,
and add a row to `docs/commands.md`. The dispatcher does not duplicate flags —
each module owns its own parser.

## Style

- `from __future__ import annotations` at the top of every module.
- Module docstrings explain **why the module exists and what it got wrong**, not
  what the functions are named. Match the density of the existing ones.
- Comments explain the non-obvious decision, not the syntax.
- Type-annotate public functions. `router/py.typed` claims the package is typed;
  keep that honest.
- `ruff check .` clean before commit. Line length 100.

## Testing

- `python3 -m pytest` must pass before any commit. Warnings are errors
  (`filterwarnings = ["error"]`) — do not silence one to make a test go green.
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
./scripts/rt help  # the tool's own command list
```
