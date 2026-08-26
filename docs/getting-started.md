# Getting started

```bash
pip install adder-cli
adder auto on --full
```

That is the setup. No account, no API key, no configuration file to write. adder
never calls a model and never opens a network connection, with one exception:
`adder models refresh`, which only runs when you type it.

The second line is the one that changes the bill. Everything else on this page
is a report, and a report saves nothing until somebody acts on it. Replaying the
author's recorded turns, activation alone — installed, working the same way —
priced a $7,888 bill at $2,567, which is 3.1x, and that is the whole reason this
page starts with `auto on` rather than with a number. ([benchmark.md](benchmark.md))

You can also run it from a checkout with no install step at all:

```bash
git clone https://github.com/stephenoffer/adder && cd adder
./scripts/adder auto on --full
```

A full run over one machine's entire history takes under a second and needs
nothing but Python 3.10+. There are no dependencies at all, deliberately, so it
works on a machine with no reachable package index.

## What activation actually does

It prints the change, asks, and then writes it. Nothing is written before you
answer, `--dry-run` stops after the printing, and `adder auto off` reverses it.

```
$ adder auto on --full

  This will add:

    add    PreToolUse: pretooluse_read_guard.py
    add    PreCompact: precompact_learn.py
    add    UserPromptSubmit: session_cost_advisor.py
    in     /your/project/.claude/settings.json

    set    guard_enforce: unset -> full
    set    guard_max_fires: unset -> 200
    set    guard_min_cost: unset -> 0.1
    set    guard_min_tokens: unset -> 800
    in     /your/project/.adder.json

    copy   Explore.md
    copy   route-t0.md
    copy   route-t1.md
    copy   route-t2.md
    in     /your/project/.claude/agents

  What each hook does:

    PreToolUse        prices, and can refuse, a call before its result lands in context
    PreCompact        forgets what compaction drops, and re-learns result sizes
    UserPromptSubmit  prices compaction against a restart, once a session is expensive
    agents            what a delegated step runs on: Explore on Haiku, three tiers

  refusals              full: also a large read with a cheaper equal

  Write these changes? [y/N]
```

Three properties of that, each of which is a test rather than a promise:

- **Your `settings.json` survives it.** Hooks another tool registered are kept,
  unrelated keys are kept, and a file that does not parse is refused rather than
  replaced. The original is copied to `settings.json.adder.bak` before the first
  edit, and that backup is never overwritten by a later run.
- **An existing agent file is never overwritten.** If you already have an
  `Explore` you rely on, it is reported as `keep ... (yours differs)` and left
  exactly as it was.
- **`adder auto off` removes precisely what `on` added**, matching on the script
  name rather than the path, so it still works after you move the checkout.

`--user` writes to `~/.claude` instead of this project, which is what you want
once you have decided you like it. `--dry-run` prints the plan and stops.

Nothing runs on a timer and nothing holds a socket. "In the background" here
means the hooks your harness already fires, on events it already has, costing
nothing on a turn where there is nothing to say. The hooks take effect in the
**next** session; the agent files take effect immediately.

Then, once it has been running:

```bash
adder auto status
```

which reports what it has prevented, what it merely argued for, and what its own
messages cost you, kept apart, because only the first of those needs no
assumption about whether anyone listened.

## Before you have any history

Every report here reads a transcript you have already paid for, so on a fresh
machine `adder doctor` has nothing to measure and says so. That is the one
asymmetry worth knowing on day one: **the reports need history, the guard does
not.** Activation is useful before you have run a single session, and the size
model it predicts with re-learns from your own transcripts as they accumulate.

## Words used here

| Term | Meaning |
|---|---|
| **turn** | one exchange: your message, the agent's reply |
| **context** | everything the model re-reads to take the next turn |
| **output token** | text the model writes. Billed once at writing, then as context on every later turn |
| **prompt cache** | a discount on re-reading context you have already sent. Big, but tied to one model |
| **cache TTL** | how long that discount lasts before the context has to be re-sent at full price (5 minutes or 1 hour) |
| **inline** | doing the work in this conversation, so everything it touches stays in your context |
| **delegated** | handing the work to a subagent with its own throwaway context, and keeping only the answer |
| **remaining turns** | how many turns the session probably has left. The multiplier on everything |

## Your first report

```bash
adder live
```

```
  This session: 11 turns · 47,401 tokens in context · $0.89 spent ($0.081/turn)
  Model claude-opus-5 · cache TTL 1h · context 5% of the 1,000,000-token window
  Sessions that reach turn 11 typically run ~340 more turns → ~$28.27 total
  One more turn at this context costs ~$0.042.

  Cost of reading a file into THIS context from here:
       file size      inline   delegated   verdict
         5,000 tok  $    0.881  $    0.096   delegate
        20,000 tok  $    3.525  $    0.383   delegate
        50,000 tok  $    8.812  $    0.957   delegate
       150,000 tok  $   26.438  $    2.869   delegate

  Every 10K tokens added to this context now costs ~$1.76 over the rest of the session.
```

Line by line:

| Line | What it means |
|---|---|
| `$0.89 spent` | what this session has cost so far. This is the number a dashboard would show you |
| `~340 more turns → ~$28.27` | sessions that get this far usually keep going. **This is the bill you are actually on the hook for**, and it is 32x the number above it |
| `One more turn … ~$0.042` | what it costs to continue right now. It grows as the context grows, so this number is worse later than it is today |
| the table | what one file costs if you read it into this conversation (`inline`) versus handing it to a subagent that reads it and hands back a summary (`delegated`) |
| `Every 10K tokens … ~$1.76` | the entry fee for anything new arriving in context, paid across every remaining turn |

**The table is the point.** Reading a 50,000-token file into this session costs
**$8.81** by the time the session ends. Handing the same file to a subagent that
reads it and returns a summary costs **$0.96**. Same information, **9x apart**,
and nothing in your usage dashboard distinguishes them.

The difference is not the reading. It is that the inline version leaves 50,000
tokens sitting in your context, and you pay rent on them for the next 340 turns.
The subagent's context is thrown away when it finishes; only the summary comes
back.

So the habit adder is arguing for is short: **stop pulling large things into
long conversations. Delegate the reading, keep the answer.**

## If you only run one command

```bash
adder doctor
```

It runs every check, prices each finding, and sorts by the price. The question
after "my bill is large" is always "which of the four usual causes is it", and
the answer is usually one of them rather than all four. Each line names the
command that goes deeper. `--strict` exits non-zero when something material is
wrong, which is what makes it usable from a hook or from CI.

Nothing in `doctor` computes anything of its own; every check calls the module
that owns that measurement. That is deliberate. A summary command that
reimplements the cost model is a second answer waiting to disagree with the
first.

## What each command answers

Start with `live`. Everything else is there when you have a specific question.

| You want to know | Run |
|---|---|
| I have no idea where to start. Just tell me. | `adder doctor` |
| What is this session costing me, right now? | `adder live` |
| Where has all my money gone, across every session? | `adder trace` |
| Which sessions were the expensive ones? | `adder sessions --sort per-turn` |
| Which *tool* is filling my context? | `adder tools` |
| Which single turns cost far more than the rest, and why? | `adder anomaly` |
| Am I actually delegating, and what did I miss? | `adder agents` |
| Am I going to blow this month's budget? | `adder budget --limit 400` |
| What does an output token *really* cost here? | `adder debt` |
| Why is my context so big, and what put it there? | `adder context` |
| Is my prompt cache working, and what is it wasting? | `adder cache` |
| Am I getting worse output, not just cheaper output? | `adder quality` |
| What is in my context before I say anything, and what does it cost? | `adder memory` |
| Did I read the same thing twice? | `adder reread` |
| Should I compact this session, or is it too late to be worth it? | `adder compact` |
| If I restart, how much may I carry, and what has to be in it? | `adder handoff` |
| Should I do this task here, or delegate it? | `adder policy "<task>"` |
| Which model is cheapest for a task that still clears the quality bar? | `adder pick "<task>"` |
| Of everything I could change, which is worth the most? | `adder savings` |
| If I followed all of that, what would the bill actually be? | `adder plan` |
| I changed something last week. Did it actually work? | `adder verify --since DATE` |
| What does a token in my context really cost to carry? | `adder carry` |
| Has this tool been worth more than the turns it costs? | `adder ledger` |

`adder help` prints the full list, including the estimator and evaluation
commands (`horizon`, `regret`, `simulate`, `ab`, `validate`, `outcomes`,
`classify`, `models`, `effort`). `adder <command> --help` shows the flags for
one. Full reference: [commands.md](commands.md).

## Slicing and scripting

Every command that reads transcripts takes the same window flags:

```bash
adder trace --since 7d --project my-repo --by tool
adder sessions --since 2026-08-01 --until 2026-09-01 --sort rebuilds
adder tools --no-subagents --json | jq '.tools[0]'
```

Dates accept `2026-08-01`, `7d`, `2w`, `today`, or `yesterday`. The window is
half-open (`--since` is inclusive, `--until` is exclusive), so August plus
September equals August-and-September, exactly.

Every report also takes `--json`, and `adder export` writes the priced turns out
as CSV, JSONL, or JSON at turn, session, or day grain. Exports carry token
counts, prices, timestamps, model ids, and tool *names*, never message content.

## What adder writes

**Your transcripts are never modified.** Nothing under `~/.claude/projects` is
written, renamed, or deleted, ever. adder does keep its own files beside them,
and they are listed here rather than left for you to discover:

| File | Written by | Why |
|---|---|---|
| `~/.claude/.adder-trace-cache` | any report | memoized parses, keyed by `(mtime, size)`; delete it any time |
| `~/.claude/adder-outcomes.jsonl` | `outcomes record` / `outcomes import --write` | the dispatch history that calibrates `p_fail` |
| `~/.claude/adder-ledger.jsonl` | `policy --record` | recommendations made, so the tool can be held to them |
| `~/.claude/adder/catalog.json` | `models refresh` | the cross-vendor model snapshot |
| `~/.claude/.adder-sizes.json` | `guard --learn`, `auto on`, the PreCompact hook | what tool calls of each shape actually returned here |
| `.claude/settings.json` + `.adder.json` | **`auto on` / `auto off` only** | the hooks, and the level they enforce at |
| `.claude/agents/*.md` | **`auto on` only** | what a delegated step runs on |
| `*.adder.bak` | `auto on`, once | whatever the two files above held before adder first touched them |

Everything else is arithmetic over files you already have, printed to stdout.
`adder config` shows the resolved path of each of the above.

The three rows marked `auto` are the only writes in the tool that land in a file
you did not name on the command line, which is why that command prints the whole
change first, keeps a backup, and has an `off`. No report writes any of them:
`live`, `trace`, `debt`, `context`, `cache`, `quality` and `horizon` are
read-only, and a test asserts it.

## What adder itself costs

Nothing, in both senses that matter. It sends no requests to any model, so there
is nothing to meter.

The second sense is the one people miss: it costs no tokens either. A cost tool
that worked by asking a model to analyse your session would add that analysis to
the context, and you would then pay to re-read it on every remaining turn. adder
is plain Python doing arithmetic over local files.

Whether the *advice* pays for the turn spent asking for it is a separate
question, and it has its own page: [overhead.md](overhead.md).

## Using it inside Claude Code

The CLI reports. The hooks and agent definitions are what act on those reports,
they ship inside the package, and `adder auto on` installs them:

- **Agents.** `Explore` on Haiku plus three routing tiers (T0/T1/T2), each with
  rules that bound how much output comes back into your context. See
  [tiers.md](tiers.md).
- **Hooks.** A prompt hook that prices the session as you work, and a
  **PreToolUse read guard** that prices a tool result *before* it lands in your
  context. The guard is the only piece here that prevents cost instead of
  reporting it after the fact, so it triggers on what the call will *cost*: with
  400 turns left, 6,000 tokens is $1.24 to carry and $0.13 delegated. A fixed
  token count cannot be right at both ends of a session. It advises by default
  and never blocks silently. See [guard.md](guard.md).
- **Skills.** `/adder` routes one task, `/adder-doctor` diagnoses a session,
  `/adder-context` decides whether to compact or restart, and `/adder-init`
  walks the install. These live in this repository's `.claude/skills/` and are a
  convenience for a checkout, not part of the mechanism; activation does not
  need them.

If you want the habit without the machinery, `adder policy "<task>"` gives you
the inline-versus-delegate call for a single task, and refuses to recommend
delegating when the saving would not cover the cost of the routing turn itself.

## How much to trust the numbers

The dollar figures in these docs come from one machine's transcripts, dominated
by one workload, and they grow every session. Your absolute numbers will be
different. The *shares* are what drive the advice, and even those are worth
re-checking on your own history, which is the entire point of the tool. **Run
`adder savings` before believing any number here.**

3,260 tests, no API key, and no network outside `adder models refresh`. Two of
those tests exist only to enforce the last two clauses: one walks the code of
every module and fails if anything outside `adder/pricing/sources.py` imports a
networking library, the other fails if the dependency list stops being empty.
