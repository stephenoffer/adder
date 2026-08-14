# Command reference

Every report is computed locally from transcript files. No API key, no model
calls, and no network — with exactly one exception, `rt models refresh`, which
is opt-in and named as such below.

Generated from the command table in `router/cli.py`; `tests/test_cli.py` fails
if a command is missing here.

## Measure

Read-only reports over transcript files. None of these write anything.

| Command | What it does |
|---|---|
| `rt live [--cwd DIR]` | this session: cost/turn, next-turn cost, pressure |
| `rt trace [root] [--json] [--verify]` | total spend, by model and session |
| `rt debt [root]` | what an output token really costs ([cost-model.md](cost-model.md)) |
| `rt context [root]` | where context growth comes from |
| `rt cache [root]` | cache hit rate and rebuild waste, by cause |
| `rt quality [root] [--since DATE]` | agent-performance proxies ([quality.md](quality.md)) |
| `rt horizon [root]` | remaining-turns estimate vs the naive countdown |

## Decide

Turn a measurement into a choice.

| Command | What it does |
|---|---|
| `rt policy "<task>" [--json]` | route a task: inline vs delegate |
| `rt outcomes [--log PATH]` | escalation calibration (p_fail) |
| `rt classify "<task>"` | task-complexity classification, on its own |
| `rt pick "<task>" [--combos] [--json]` | cheapest model, or combination, that clears the quality bar |
| `rt models [list|show|ladder|refresh]` | the cross-provider catalog: what exists, at what price and rating |

## Evaluate

Check that a lever is real before trusting it.

| Command | What it does |
|---|---|
| `rt savings [root] [--max-turns N]` | what each lever is worth ([levers.md](levers.md)) |
| `rt verify --since DATE [root]` | did a change actually land? |
| `rt validate [root]` | re-test the claims everything rests on ([measurement.md](measurement.md)) |
| `rt regret [root]` | dollar regret of the horizon estimator |
| `rt simulate [root]` | replay sessions under interventions; test lever composition |
| `rt ab --help` | controlled A/B on answer quality |

## Conventions

- `root` defaults to your local Claude Code projects directory
  (`~/.claude/projects`). Pass a path to analyse a copy instead.
- `rt` with no arguments, or `rt help`, prints the command list.
- `rt help <command>` and `rt <command> --help` both show that command's flags.
  Each module owns its own parser, so the flags shown are always the real ones.
- `rt version` prints the installed version.
- Every command exits `0` on success, `1` on a reported failure, and `2` on a
  usage error.

## The one networked command

`rt models refresh` is the only command that opens a socket. It pulls the public
model catalog, writes it to your user cache, and touches nothing else. It never
runs implicitly, and `LLM_ROUTER_OFFLINE=1` makes it refuse outright. Every other
command works from a machine with no network at all.

Replay a saved capture instead of fetching with
`rt models refresh --from openrouter=models.json`.

## What ships

- **`router/`** — pricing with context limits and cache minimums, a cache-aware
  cost model, transcript parsing with deduplication, growth attribution, cache
  analysis, quality proxies, the model catalog, and the routing policy.
- **`router/cli.py`** — the dispatcher. Adding a command means one row here plus
  a module, a test, and a line in this file.
- **`.claude/agents/`** — Explore on Haiku plus three routing tiers, each with
  output-bounding rules.
- **`.claude/hooks/`** — a prompt hook that prices the session, and a
  **PreToolUse guard** that prices a large read *before* it lands in context.
  The guard is the only thing here that prevents cost rather than reporting it.
- **`.claude/skills/`** — `/route`, `/route-doctor`, `/route-init`.
- **`scripts/rt`** — launcher for a git checkout; `pip install` provides `rt`
  directly.
