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

## The hook, which is the part that actually prevents spend

Everything else adder ships is a report, and a report saves nothing until
somebody acts on it. The agent files and the PreToolUse guard are the only two
components that act without being obeyed, and `adder bench` prices that pair on
its own — on the author's history, **1.5x of the 1.6x "installed and changed
nothing" figure is the guard**. Offer it, and say what it does before writing
anything.

`.claude/hooks/pretooluse_read_guard.py` prices a read **before** it lands in
context and injects the price plus the alternative. It is advisory: it never
denies a call, and it never blocks silently.

```json
{"hooks": {"PreToolUse": [{"matcher": "Read|Bash|Grep",
   "hooks": [{"type": "command",
              "command": "python3 /abs/path/.claude/hooks/pretooluse_read_guard.py"}]}]}}
```

Rules for offering it:

- **Read the user's existing `settings.json` and show a diff first.** A hooks
  block is easy to clobber; another PreToolUse hook may already be registered
  and both must survive.
- It fires on a **cost** (`ADDER_GUARD_MIN_COST`, default $0.25), not a token
  count, because the same read is worth interrupting for at turn 400 and not at
  turn 3. Do not suggest tuning it before running `adder bench`, which reports
  whether the dollar gate or the token floor is the binding constraint — on this
  workload it is the floor, so tuning the dollar gate would change nothing.
- `ADDER_GUARD_BLOCK=1` escalates from advice to a confirmation prompt on very
  large reads. Off by default, and it should stay off until the user has seen
  the advisory version fire a few times.

## Bootstrap the routing evidence

The tier agents cannot route below what the classifier asked for until the
outcome log has history at that rung, and on a fresh install it has none. If the
user has delegated before, the evidence is already in their transcripts:

```bash
adder outcomes import          # dry run — shows what it found
adder outcomes import --write  # append it
```

Say what it can and cannot see: an error result or an `ESCALATE:` reply is an
observed failure; a subagent that returned a confident wrong answer is invisible,
so the rate is a lower bound. Do not run `--write` without asking.

## After installing

Tell the user plainly:

- Explore-on-Haiku takes effect immediately, with no further action.
- `/adder` is opt-in per task; it deliberately declines to route when the saving
  would not clear the cost of the routing turn itself.
- The largest remaining lever is usually **session length**, which no agent file
  can fix: context cost grows with turns × context, so splitting a long session
  cuts it roughly in proportion.
