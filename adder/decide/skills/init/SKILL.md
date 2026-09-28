---
description: Install the cost-routing agents (Explore on Haiku, tier agents) into this project or user config, after showing exactly what will change. Use when asked to set up, install, or enable cost routing.
allowed-tools: Bash(adder:*), Bash(./scripts/adder:*), Read, Write, Bash(diff:*), Bash(ls:*), Bash(cat:*), Bash(env:*)
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
   `adder savings` and quote the top two levers.

## What to install

The agents ship inside the package, at `adder/decide/agents/`. Do not copy them
by hand -- `adder auto on` installs them and reports what it skipped. What each
one is for:

| File | Effect |
|---|---|
| `Explore.md` | Overrides the built-in Explore to run on Haiku, read-only. The built-in inherits the session's model (Opus on an Opus session, per the Claude Code docs from v2.1.198), so this is a large saving on read-heavy work that runs in a fresh context with no cache to lose. It is not free: Haiku searches less well on hard questions. Say so, and if Explore's answers get worse, remove the file. |
| `route-t0.md` | Haiku, read-only, capped turns. Read-only makes escalating away from it risk-free. |
| `route-t1.md` | Sonnet, scoped edits, capped turns. |
| `route-t2.md` | Opus, the escalation target and safe default. |

`adder auto on` writes them to `~/.claude/agents/`, or to this project's
`.claude/agents/` with `--project`. With the adder plugin enabled it copies only
`Explore.md`, since the plugin already supplies the tiers and cannot override a
built-in. It never overwrites: a file whose contents
differ is reported as `keep ... (yours differs — left alone)`, because a user may
already have an `Explore` they rely on. If one is skipped, show the diff and ask
before doing anything about it.

## The hooks, which are the part that actually prevents spend

Everything else adder ships is a report, and a report saves nothing until
somebody acts on it. The agent files and the PreToolUse guard are the only two
components that act without being obeyed, and `adder bench` prices that pair on
its own — on the author's history, **1.5x of the 1.6x "installed and changed
nothing" figure is the guard, and letting it refuse rather than advise takes
that to 3.1x**.

Do not hand-write the hooks block. `adder auto` owns this now:

```bash
adder auto on --full --dry-run   # prints every change, writes nothing
adder auto on --full --yes       # only after the user has said yes to that
```

Run the dry run, show the user its output, and ask them in the conversation.
Pass `--yes` only once they agree: without it `auto on` asks on stdin, which
the Bash tool does not have, so it prints `Nothing written.` and stops. It
merges into an existing `settings.json` without disturbing hooks another tool
registered, keeps a `.adder.bak`, and `adder auto off` removes exactly what it
added. With the plugin enabled it writes no hooks at all, and removes any
earlier copy, because a hook declared twice runs twice.

What to tell them about the levels, because this is the part that changes
behaviour:

- **`certain`** (plain `adder auto on`) refuses only calls that admit nothing
  new — a read of a file already in the context, or one this session wrote.
  Refusing these cannot lose information at any price.
- **`--full`** also refuses a large read that has a strictly cheaper equal, and
  names the cheaper call. This one changes how the agent works: it will
  delegate and bound reads it would otherwise have made inline.
- Either way it never refuses the same thing twice. If the model asks again it
  gets through, so a wrong refusal costs one turn.

Do not suggest tuning the thresholds by hand. `adder auto on --full --tune`
sweeps them against the user's own transcripts, and `adder guard --explain
"<command>"` answers "why would it say nothing about this".

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
- Hooks that `adder auto on` wrote take effect in the **next** session. If the
  adder plugin is installed, its hooks are already running, and `auto on` only
  set how firmly they act.
- The routing skill (`/adder`, or `/adder:route` from the plugin) is opt-in per
  task. It declines to route when the saving would not clear the cost of the
  routing turn itself.
- `adder auto status` reports what it has been worth, and keeps the calls it
  prevented separate from the advice it gave — only the first of those needs no
  assumption about whether anyone listened.
- The largest remaining lever is usually **session length**, and no hook can
  pull it: context cost grows with turns × context, so splitting a long session
  cuts it roughly in proportion, and nothing here can restart a session for
  them. That gap is the difference between the 3.1x above and 6.4x.
