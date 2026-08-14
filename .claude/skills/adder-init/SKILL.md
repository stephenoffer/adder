---
name: adder-init
description: Install the cost-routing agents (Explore on Haiku, tier agents) into this project or user config, after showing exactly what will change. Use when asked to set up, install, or enable cost routing.
allowed-tools: Bash(/Users/stephen.offer/Desktop/llm-router/scripts/adder:*), Read, Write, Bash(diff:*), Bash(ls:*), Bash(cat:*), Bash(env:*)
disable-model-invocation: true
---

# Install cost routing

## Before changing anything

1. Check for silent overrides that would defeat routing:

   ```
   !`env | grep -E 'CLAUDE_CODE_SUBAGENT_MODEL|ANTHROPIC_MODEL' || echo "none set (good)"`
   ```

   `CLAUDE_CODE_SUBAGENT_MODEL` outranks both the per-invocation model parameter
   and agent frontmatter. If it is set, say so and stop — routing cannot work
   until it is unset.

2. Show the user what each change is worth before making it. Run
   `/Users/stephen.offer/Desktop/llm-router/scripts/adder savings` and quote the top two levers.

## What to install

Source agents live in `/Users/stephen.offer/Desktop/llm-router/.claude/agents/`:

| File | Effect |
|---|---|
| `Explore.md` | Overrides the built-in Explore to run on Haiku, read-only. Biggest zero-risk win: exploration is read-heavy and runs in a fresh context, so there is no cache to lose. |
| `route-t0.md` | Haiku, read-only, capped turns. Read-only makes escalating away from it risk-free. |
| `route-t1.md` | Sonnet, scoped edits, capped turns. |
| `route-t2.md` | Opus, the escalation target and safe default. |

Copy to `.claude/agents/` in the target project, or `~/.claude/agents/` for all
projects. **Show a diff against any existing file and get confirmation before
overwriting** — a user may already have an Explore agent they rely on.

## After installing

Tell the user plainly:

- Explore-on-Haiku takes effect immediately, with no further action.
- `/adder` is opt-in per task; it deliberately declines to route when the saving
  would not clear the cost of the routing turn itself.
- The largest remaining lever is usually **session length**, which no agent file
  can fix: context cost grows with turns × context, so splitting a long session
  cuts it roughly in proportion.
