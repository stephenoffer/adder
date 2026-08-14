# Command reference

Every report is computed locally from transcript files. No network, no API key,
no model calls.

## Measure

| Command | What it does |
|---|---|
| `rt live [--cwd DIR]` | this session: cost/turn, next-turn cost, pressure |
| `rt trace [root] [--json] [--verify]` | total spend, by model and session |
| `rt debt [root]` | what an output token really costs ([cost-model.md](cost-model.md)) |
| `rt context [root]` | where context growth comes from |
| `rt cache [root]` | cache hit rate and rebuild waste, by cause |
| `rt quality [root] [--since DATE]` | agent-performance proxies ([quality.md](quality.md)) |

## Decide

| Command | What it does |
|---|---|
| `rt policy "<task>" [--json]` | route a task: inline vs delegate |
| `rt outcomes [--log PATH]` | escalation calibration (p_fail) |

## Evaluate

| Command | What it does |
|---|---|
| `rt savings [root] [--max-turns N]` | what each lever is worth ([levers.md](levers.md)) |
| `rt verify --since DATE [root]` | did a change actually land? |

`root` defaults to your local Claude Code projects directory. `rt` with no
arguments prints this list.

## What ships

- **`router/`** — pricing with context limits and cache minimums, a cache-aware
  cost model, transcript parsing with deduplication, growth attribution, cache
  analysis, quality proxies, and the routing policy.
- **`.claude/agents/`** — Explore on Haiku plus three routing tiers, each with
  output-bounding rules.
- **`.claude/hooks/`** — a prompt hook that prices the session, and a
  **PreToolUse guard** that prices a large read *before* it lands in context.
  The guard is the only thing here that prevents cost rather than reporting it.
- **`.claude/skills/`** — `/route`, `/route-doctor`, `/route-init`.
