# Package structure

How this repository is laid out, why, and what to do when you add a file.

Everything here is checked by `tests/repo/test_structure.py`. If you are reading
this because that test failed, the rule you broke is below with the reasoning
behind it — and the failure message already told you where the file should go.

## The problem this solves

`adder/` was fifty modules in one directory. No single commit was at fault:
each one added one more report next to the others, which was obviously fine at
the time. That is how breadth happens. By the end, finding the module that
priced a cache write meant reading a fifty-line `ls`, and `trace.py` — the
transcript reader that every report depends on — sat between `tools.py` and
`validate.py` with nothing marking it as the foundation.

So the rules below are not about tidiness. Two of them are about **cost**
(what does an import drag in) and one is about **findability** (where do I look).

## The rules

### Depth over breadth

**No more than 12 Python files or 10 subdirectories in one directory.**

Crossing either is the signal to add a level, not to keep going sideways. When
you split, split by *subject* — the thing the modules are about — not by kind.
`measure/window/` holds everything about what fills the context window; a
`helpers/` or `utils/` directory holds whatever nobody had a name for, and fills
up again within a month.

The numbers are arbitrary in the way a speed limit is arbitrary. What matters is
that there is one and that something checks it.

### Layers, and imports point down

Seven packages, ordered. An import may point down this list, never up.

| Layer | Package | Holds | Knows about |
|---|---|---|---|
| 0 | `adder.util` | render, stats, risk, text | nothing — no turns, no dollars |
| 1 | `adder.pricing` | prices, catalog, providers, registry, cost, bt, sources | what a token costs |
| 2 | `adder.core` | trace, filters, settings, shapes | reading a session off disk |
| 3 | `adder.measure` | the read-only reports | what happened |
| 4 | `adder.decide` | `route/`, `track/`, guard, handoff | what to do about it |
| 5 | `adder.evaluate` | `replay/`, `claims/`, doctor | whether the advice held |
| 6 | `adder.cli` | dispatcher, command table, help | how a person reaches all of it |

The test for whether something belongs in `util` is whether you can explain it
without saying the words "turn" or "session". `est_tokens` passes. Anything that
takes a `Session` does not.

Two things the layering buys:

* **The hook stays cheap.** The PreToolUse guard runs on every submit. It reaches
  `core.trace` and `pricing.cost`, and because those may not import upward, it
  cannot accidentally pull in an argparse parser, a report, or the A/B harness.
* **A cycle becomes a design question.** When `core.filters` wanted a date
  helper that lived in a report, the fix was to move the helper down into
  `core`, not to add a function-level import that hides the cycle. The rule made
  that the obvious move rather than the annoying one.

If you need something from a higher layer, the shared piece belongs in a lower
one. Move it down; don't import up.

### The foundation carries no commands

`util`, `pricing`, and `core` may not contain a module registered in `COMMANDS`.
A module that owns an argparse parser is a module that cannot be imported
cheaply, and everything imports the foundation.

This is why `trace` is two files: `core/trace.py` reads and deduplicates
transcripts, `measure/spend/trace.py` is the `adder trace` report. Same for
`core/settings.py` (resolution, read by fifteen modules) and `cli/config.py`
(the `adder config` report).

### The test tree mirrors the package tree

`adder/measure/window/cache.py` is tested by `tests/measure/window/test_cache*.py`.
Directory for directory, with one exception: `tests/repo/` checks the repository
itself and mirrors no package.

No test file sits at the top of `tests/`. A test that belongs to no module never
acquires one.

Two test modules may share a basename — `tests/cli/test_dispatch.py` and
`tests/decide/track/test_dispatch.py` are both the obvious name for what they
cover.
That works because `pytest` runs with `--import-mode=importlib`; do not remove
it.

## Naming and style

| Thing | Rule |
|---|---|
| Module file | lowercase, `^[a-z][a-z0-9]*(_[a-z0-9]+)*$`; one word where one word will do |
| Package | a noun for the subject, not a kind: `pricing`, not `helpers` |
| Command module | named for the command: `adder cache` lives in `.../cache.py` |
| Test file | `test_<module>.py`, or `test_<module>_<aspect>.py` for a second angle |
| Exception class | ends in `Error` (`UnknownModelError`), enforced by ruff `N818` |
| Single-capital local | allowed, and only, for a symbol from the formula in the docstring above it |
| Imports inside the package | absolute (`from adder.pricing.cost import turn_cost`), never relative |
| Every module | `from __future__ import annotations` first, then a docstring saying why the module exists |

Absolute imports are worth a word. In a tree this deep, `from ...pricing.cost
import turn_cost` makes the number of dots load-bearing and invisible, and it
breaks the moment the file moves one level. The absolute form says where the
code lives no matter which file you are reading it in. Ruff's `TID252` enforces
it.

## Adding to the package

**A new report or command:**

1. Pick the layer by what the module *does*: reports go in `measure/`,
   recommendations in `decide/`, checks of a recommendation in `evaluate/`.
2. Write `adder/<layer>/<subject>/<name>.py` with a module docstring, a
   `main(argv) -> int`, and its own argparse parser. The dispatcher does not
   declare flags.
3. Add one `Command(...)` row to `COMMANDS` in `adder/cli/commands.py`.
4. Add `tests/<same path>/test_<name>.py`.
5. Add a row to `docs/commands.md` — a test checks for it.

If step 2 would make the directory the 13th file, split the directory first.

**A new shared helper:** put it in the lowest layer that can hold it. If two
packages need it and it knows nothing about the domain, that is `util`. If it
takes a `Session`, that is `core`.

## What is checked, and where

| Rule | Checked by |
|---|---|
| Breadth caps, layering, mirroring, package docstrings, module names | `tests/repo/test_structure.py` |
| Absolute imports, naming, `from __future__`, module docstrings | `ruff check .` (`TID252`, `N`, `I002`, `D100`) |
| Every `main()` is registered in `COMMANDS` | `tests/cli/test_dispatch.py` |
| No network outside `pricing/sources.py`, no runtime deps | `tests/repo/test_invariants.py` |
| Every command is in `docs/commands.md` | `tests/repo/test_invariants.py` |
