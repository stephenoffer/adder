# adder

[![CI](https://github.com/stephenoffer/adder/actions/workflows/ci.yml/badge.svg)](https://github.com/stephenoffer/adder/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/adder-cli.svg)](https://pypi.org/project/adder-cli/)
[![Python](https://img.shields.io/pypi/pyversions/adder-cli.svg)](https://pypi.org/project/adder-cli/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**adder reads the agent transcript files already sitting on your disk and
tells you where your token spend actually goes — and what it is going to keep
costing for the rest of the session.**

It ships pointed at Claude Code, which is where its numbers were measured, and
it works on any provider: OpenAI, Google, DeepSeek, and hosted open-weight
endpoints are priced with *their* cache economics, not Anthropic's borrowed.
See [Providers](docs/providers.md).

```bash
pip install adder-cli
adder live
```

That's the whole setup. No account, no API key, no configuration file. adder
never calls a model and never opens a network connection — with one exception,
`adder models refresh`, which only runs when you type it.

**Your transcripts are never modified.** Nothing under `~/.claude/projects` is
written, renamed, or deleted, ever. adder does keep its own files beside them,
and they are listed here rather than left for you to discover:

| File | Written by | Why |
|---|---|---|
| `~/.claude/.adder-trace-cache` | any report | memoized parses, keyed by `(mtime, size)` — delete it any time |
| `~/.claude/adder-outcomes.jsonl` | `outcomes record` / `outcomes import --write` | the dispatch history that calibrates `p_fail` |
| `~/.claude/adder-ledger.jsonl` | `policy --record` | recommendations made, so the tool can be held to them |
| `~/.claude/adder/catalog.json` | `models refresh` | the cross-vendor model snapshot |

Everything else is arithmetic over files you already have, printed to stdout.
`adder config` shows the resolved path of each of the above.

## Is this for you?

You will get something out of adder if any of these sound familiar:

- You use Claude Code, Codex, Gemini CLI, or any agent that holds one long
  conversation, and the bill came in higher than you expected.
- You have been told to "use a cheaper model" and want to know whether that
  actually helps. (Usually it doesn't. See [below](#three-things-the-numbers-overturned).)
- You want to know whether a change you made last month saved real money, or
  just felt frugal.
- You want the tool to prove its own worth before you install it. Measured on
  the author's history: [1.6x for installing it, 6.7x for following
  it](#adder-vs-no-adder).

You do not need to understand token pricing to start. Run `adder live` and read
the walkthrough below.

## The one idea

First, two words this page uses constantly:

- A **turn** is one exchange: you say something, the agent responds.
- The **context** is everything the model has to re-read to take the next turn —
  your whole conversation so far, plus every file and command output that got
  pulled into it.

Here is the part that surprises people. **You do not pay for a piece of text
once. You pay for it on every turn after it appears.**

When the agent writes 1,000 tokens, you are billed for writing them. Then those
1,000 tokens join the context, so you are billed to re-read them on turn 2. And
turn 3. And every turn until the session ends. Write something with 340 turns
left and you pay for it 341 times.

How much that multiplies your real cost depends on how much session is left:

| turns remaining | 0 | 50 | 200 | 340 | 759 | 1,854 |
|---|---|---|---|---|---|---|
| **what a token really cost** | 1.0x | 2.0x | 5.0x | **7.8x** | 16.2x | 38.1x |

Past roughly **50 remaining turns, re-reading a token costs more than writing it
did.** Your usage dashboard shows you the 1.0x column. ([The arithmetic is in
docs/cost-model.md](docs/cost-model.md); `adder debt` recomputes it against your
own history.)

Two things follow, and they are the whole tool:

- The expensive decision is rarely *which model you used*. It is **what you let
  into the context, and how early in the session you let it in**.
- A 600-turn session is not one long session. It is the same context, re-read
  600 times.

adder measures that second bill — the carry — for your own sessions, and prices
the decisions that come out of it.

## What adder itself costs

Nothing, in both senses that matter:

- **No dollars.** It sends no requests to any model. There is nothing to meter.
- **No tokens in your context.** This is the part people miss. A cost tool that
  works by asking a model to analyse your session would add its own analysis to
  your context, and you would then pay to re-read that analysis on every
  remaining turn. adder is plain Python doing arithmetic over local files.

A full run over one machine's entire history takes under a second and needs
nothing but Python 3.10+ — no dependencies at all, deliberately, so it works on
a machine with no reachable package index.

## Your first run

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

So the habit adder is arguing for is short: **stop pulling large things into long
conversations. Delegate the reading, keep the answer.**

You can also run adder from a checkout, with no install step at all:

```bash
git clone https://github.com/stephenoffer/adder && cd adder
./scripts/adder live
```

### Words used here

| Term | Meaning |
|---|---|
| **turn** | one exchange — your message, the agent's reply |
| **context** | everything the model re-reads to take the next turn |
| **output token** | text the model writes. Billed once at writing, then as context on every later turn |
| **prompt cache** | a discount on re-reading context you have already sent. Big, but tied to one model |
| **cache TTL** | how long that discount lasts before the context has to be re-sent at full price (5 minutes or 1 hour) |
| **inline** | doing the work in this conversation, so everything it touches stays in your context |
| **delegated** | handing the work to a subagent with its own throwaway context, and keeping only the answer |
| **remaining turns** | how many turns the session probably has left. The multiplier on everything |

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
| Why is my context so big — what put it there? | `adder context` |
| Is my prompt cache working, and what is it wasting? | `adder cache` |
| Am I getting worse output, not just cheaper output? | `adder quality` |
| What is in my context before I say anything, and what does it cost? | `adder memory` |
| Did I read the same thing twice? | `adder reread` |
| Should I compact this session, or is it too late to be worth it? | `adder compact` |
| If I restart, how much may I carry — and what has to be in it? | `adder handoff` |
| Should I do this task here, or delegate it? | `adder policy "<task>"` |
| Which model is cheapest for a task that still clears the quality bar? | `adder pick "<task>"` |
| Of everything I could change, which is worth the most? | `adder savings` |
| If I followed all of that, what would the bill actually be? | `adder plan` |
| I changed something last week. Did it actually work? | `adder verify --since DATE` |
| What does a token in my context really cost to carry? | `adder carry` |
| Has this tool been worth more than the turns it costs? | `adder ledger` |

`adder help` prints the full list — including the estimator and evaluation
commands (`horizon`, `regret`, `simulate`, `ab`, `validate`, `outcomes`,
`classify`, `models`, `effort`) — and `adder <command> --help` shows the flags
for one. Full reference: [docs/commands.md](docs/commands.md).

### If you only run one command

```bash
adder doctor
```

It runs every check, prices each finding, and sorts by the price — because the
question after "my bill is large" is always "which of the four usual causes is
it", and the answer is usually one of them rather than all four. Each line names
the command that goes deeper. `--strict` exits non-zero when something material
is wrong, which is what makes it usable from a hook or from CI.

Nothing in `doctor` computes anything of its own; every check calls the module
that owns that measurement. That is deliberate — a summary command that
reimplements the cost model is a second answer waiting to disagree with the
first.

### Slicing and scripting

Every command that reads transcripts takes the same window flags:

```bash
adder trace --since 7d --project my-repo --by tool
adder sessions --since 2026-08-01 --until 2026-09-01 --sort rebuilds
adder tools --no-subagents --json | jq '.tools[0]'
```

Dates accept `2026-08-01`, `7d`, `2w`, `today`, or `yesterday`. The window is
half-open — `--since` is inclusive, `--until` is exclusive — so August plus
September equals August-and-September, exactly.

Every report also takes `--json`, and `adder export` writes the priced turns out
as CSV, JSONL, or JSON at turn, session, or day grain. Exports carry token
counts, prices, timestamps, model ids, and tool *names* — never message content.

## The context you did not write

Three of the four things in a long conversation's window arrived without anyone
deciding to put them there this turn: the instruction files that open every
session, the file you already read two hundred turns ago, and everything a
compaction has not got round to dropping. They are measured by their own family
of commands, and the reasoning is in [docs/context.md](docs/context.md).

**Memory is the only content whose carry never ends.** A tool result is
eventually compacted away. `CLAUDE.md` is not — compaction rebuilds the prefix
from the same file. So a resident token is re-read on every turn of every
session, indefinitely:

```
1,000 resident tokens = $0.31 per session
                      = $5.59 across the 18 sessions this project has on record
                      = $32.64 in a user-level file, which all 105 sessions load
```

Scope decides which session count applies: a project `CLAUDE.md` is resident
only in that project's sessions, a `~/.claude/CLAUDE.md` in all of them, and
pricing the first against every session on the machine over-states it several
fold. `adder memory` prices each file that feeds the prefix, separates what is
resident from what loads on demand (a skill's description is resident; its body
is not), and reports what is duplicated, stale, or unindexed.
`adder memory --what-if 500` prices an edit before you make it.

**The second copy of a file buys nothing.** The first one never left.
`adder reread` separates a redundant re-read from a justified refresh — only
the first is offered as a saving — and, for things re-learned in many sessions,
prints the largest resident note that would still beat re-reading them. On this
machine that budget comes out at tens of tokens, which is the arithmetic behind
a piece of advice this repo does not otherwise give: **do not write it down.**

**Compaction has a threshold, and it is low.** It pays for itself whenever

```
remaining_turns  >  kept * write_mult / (freed * read_mult)
```

— a few dozen turns at the measured multipliers. So the common mistake is not
compacting too often; it is carrying a full context for hundreds of turns.
Measured here: 9 compactions, median survival 6%, net **+$1,857**; and 18
sessions that never compacted at all, worth **$718**. `adder compact` prices
each one, and `adder live` now ends with the live verdict:

```
Context hygiene: restart — worth ~$55 over the ~350 turns expected to remain.
  compacting instead: ~$35.
```

**Restarting does not mean losing the session.** `adder handoff` solves for the
brief size at which a restart stops being ahead. At a 500K context with 300
turns left it is 467,000 tokens — so the constraint on a handoff is what you
can usefully say, not what you can afford to carry. It also lists what the
brief has to name, recovered from tool inputs alone: files edited, commands
re-run, reads ranked by what re-establishing them would cost. Never message
text.

## What is it worth to fix?

```bash
adder savings
```

This is the impact question: of everything you *could* change, which ones are
worth the trouble? Run against the author's history, it reports (abridged):

```
  Measured spend $4,818 across 53 sessions
  Root cause: $3,541 of it is prior context being re-read

  SUBSTITUTES - all attack the same pool; they do not add
  $    1,941   40.3%  [MODELLED  ] Split sessions longer than 300 turns
  $      873   18.1%  [MODELLED  ] Delegate 25% of turns to subagents
  $      865   17.9%  [MODELLED  ] Drop effort high -> medium
  $      703   14.6%  [ATTRIBUTED] Cut tool output admitted to context by 40%
  $      648   13.5%  [ATTRIBUTED] Write 30% less (leverage 4.7x downstream)

  COMBINED (substitutes compose multiplicatively on the residual):
    TOTAL             $    3,253   (68% of measured spend)
```

Read the second line first. **$3,541 of a $4,818 bill — 73% of it — was not new
work at all.** It was context that had already been paid for once, being re-read.
Everything below that line is a different way of attacking the same $3,541.

Three things worth understanding about this output:

**The levers are substitutes, not a shopping list.** They all drain the same
pool, so you cannot add them up: 40.3% + 18.1% + 17.9% is not 76%. The `COMBINED`
line composes them properly, which is why five levers worth 104% on paper come
to 68% in reality.

**The ranking is probably not what you expected.** The top lever is not writing
style or model choice. It is *ending sessions sooner* — because session length is
the multiplier on everything else.

**Every number is labelled with how much to trust it:**

| Label | Means | Trust |
|---|---|---|
| `MEASURED` | counted directly from your transcripts | high |
| `ATTRIBUTED` | a share of a measured pool, split by a stated rule | medium |
| `MODELLED` | derived from assumptions, which are printed next to the number | check the assumptions |

The `MODELLED` rows are the weak link, and the tool says so instead of quietly
rounding them up.

### If you change only one thing

Start a new session more often. On the history above that single habit was worth
40% of the bill — more than model choice, delegation and terseness combined —
because every other cost in this tool is multiplied by how many turns are left.
A long session is not a free resource you are being efficient by reusing. It is a
context you are re-buying on every turn.

Second: when a task needs to read something big, delegate it. That is the 9x in
the table above, and unlike splitting a session it costs you nothing in continuity.

## Three things the numbers overturned

**Switching to a cheaper model mid-session usually loses money — but starting on
one is the biggest lever there is.** The prompt cache is tied to one model, so
moving a *warm* conversation throws the discount away: Opus 5 re-reads cached
context at $0.50/MTok, Haiku 4.5 has to read it fresh at $1.00, and at 544K
Haiku cannot hold it at all. Per-turn routing is worth $22 out of $4,818.

That is a fact about switching, and for a long time this repo read it as a fact
about models. It is not. A session that *starts* on Sonnet never built an Opus
prefix, so there is nothing to invalidate — just a cheaper rate on the
context-carry term that is three quarters of the bill. Same transcripts, same
arithmetic: switching is worth 0.5% of spend, starting cheap is worth 60%.
`adder validate` re-runs both so they can be checked against each other.

**Verbosity is not the main lever.** Being told to write shorter answers feels
like the obvious fix. But assistant output is only half of context growth. Tool
results are another quarter, and `Bash` alone admits more context than every
other tool combined. No writing-style instruction can reach that.

**The cache was already fine.** 99.2% hit rate, 97% of writes on the 1h TTL,
$0 recoverable. The $302 of waste came from gaps longer than an hour between
turns, which no cache setting covers. The tool reports that as unavailable rather
than inventing a saving it cannot deliver.

## Cheaper is not the same as better

A tool that only counts dollars will happily talk you into worse work.

`adder quality` reads the same transcripts for signs the agent is struggling:
tool error rate, how often you correct it, how often you interrupt it, turns per
prompt, and rework. `adder verify` then refuses to certify a saving if any of
those got worse. A change that cut your bill and doubled your rework is not a
saving; it is a cost you moved somewhere the invoice cannot see.

## So how much can it actually save?

`adder savings` prices each lever on its own. That answers "which one is
biggest", not "what would my bill be". `adder plan` answers the second: it
replays every recorded turn under a **regime** — a concrete operating
configuration you could actually follow — and prices both sides of it.

```bash
adder plan --target 10
```

```
  Measured spend            $     5,025   20,808 turns, 84 sessions
  Replay of the same turns  $     5,025   residual -0.0% -- everything below is relative to this
  Restart cadence, solved rather than assumed: 19 turns: k* = sqrt(2W/(m*r*g)) at a
  $0.1033 restart [measured], 961 tok/turn of growth and a 0.115x re-read multiplier.
  A restart is charged what an opening actually costs -- 74% of it is a cache read.
  Delegation threshold, likewise: delegate reads over ~285 tok: below that the
  400-token brief and the summary cost more than the 9 re-reads they avoid.

  regime                                          total  vs baseline  tok deleg.
  -----------------------------------------------------------------------------
  as run                                    $     5,025         1.0x           -
  delegate reads over 300 tok               $     1,552         3.2x         99%
  + right-size the subagent                 $     1,039         4.8x         99%
  + split sessions at 19 turns              $       761         6.6x         99%
  + effort high -> medium                   $       746         6.7x         99%
  + 30% terser, 40% less tool output        $       735         6.8x         99%
  + start sessions on claude-sonnet-5       $       477        10.5x         99%

  Target 10x means getting $5,025 down to $503.
  The regime above reaches 10.5x. Target met, on these assumptions;
  run `adder quality` before and after, because none of this is free.
```

**The two thresholds are solved, not chosen.** `19 turns` used to be a round
`300`, and `300 tokens` used to be a round `5,000`. Both are set by the prompt
cache, and both were being guessed.

A restart does not rebuild the prefix. Measured over 46 openings, 74% of an
opening context arrives as a *cache read*, because the expensive part of the
floor — system prompt, tool schemas, `CLAUDE.md` — is identical across sessions
and still resident. That makes a restart $0.10 rather than the $0.30 a rebuild
costs, and since the optimum goes as `sqrt(W)`, the cycle that minimises average
turn cost lands at 19 turns instead of 33 — against sessions that actually run to
536. `adder prefix` shows the measurement; `adder plan` charges every restart for
it. Restarting used to be free in this replay, which is exactly how a lever gets
pushed to the end of its range for nothing.

The delegation threshold falls out of the same arithmetic, and the answer is not
the intuitive one: a shorter cycle leaves fewer re-reads to avoid, which should
*raise* the threshold, and it does — but only to ~300 tokens, because admitting a
token to an Opus context costs 2.00x its input rate as a cache write while
reading it once on Haiku costs 1.00x of a rate five times lower. Delegation is
not only a carry play, and `5,000` was leaving most of it unused.

Three things make this different from the savings table.

**It reproduces your bill before it quotes a discount.** The second line is the
whole guarantee: replay the transcripts with no regime applied and the total has
to come back as the number you actually paid. It does, to −0.0%. Every multiple
below it is a ratio against that. `adder validate` re-checks it, because two
ordering bugs in the replay were caught by exactly this line and nothing else
would have caught them.

**Both sides are on the books.** A delegated read still has to be read by
somebody, that somebody still writes a summary, and some fraction of those runs
come back wrong and get redone on Opus. All three are charged. The saving is
smaller than the version that only counts what left your context, and it is the
one you would actually get.

**Delegability is measured, not assumed.** Every earlier estimate here used
"assume 25% of turns are delegable", which is a guess with a percent sign on it
and is not a rule anyone can follow. The regime triggers on something the
transcript records exactly — how many tokens a step would pull into context — so
"delegate anything over 5,000 tokens" is checkable, followable, and the 23% that
matches is a measurement.

**10x is reachable on this history, and the tool will tell you if it is not on
yours.** When no configuration on the grid meets the target it says so and names
the floor, instead of searching until it finds a number that flatters the
question.

## adder vs no adder

`adder plan` prices the cheapest way this workload *could* have been run. It
assumes you do everything it says. The question that comes before that one is
what changes if you install adder and keep working exactly as you do now — and
only two things here can act without being obeyed: the PreToolUse guard, which
prices a read before it lands in context, and the tier files in
`.claude/agents/`, which decide what a delegated step runs on. Everything else
is a report, and a report saves nothing until somebody acts on it.

`adder bench` replays every recorded turn on both sides of that line.

```bash
adder bench
```

```
  Measured spend            $     5,846   23,922 turns, 90 sessions
  Replay of the same turns  $     5,846   residual +0.0%

  configuration                                                  total  vs no adder
  ---------------------------------------------------------------------------------
  no adder -- as run                                        $    5,846         1.0x
  + the read guard (delegate over 2,000 tok)                $    3,943         1.5x
      $0.25 at 321 expected re-reads is 1,500 tok; the floor is 2,000, so the
      token floor binds
  + the tier agents (.claude/agents)                        $    3,730         1.6x
      subagent tier chosen by expected cost, which is what route-t0/t1/t2 encode
 *+ what the reports say (over 300 tok, restart every 20)   $      869         6.7x

  Installed and nothing else changed:    1.6x
  Following what the reports say:        6.7x
```

**So the answer is two numbers, and quoting one of them would be dishonest.**
Installing it and changing nothing is worth **1.6x** on this history. Working
the way it tells you to is worth **6.7x** — and that row is the orchestrator
pattern, where 99% of admitted tokens go out to a subagent and the main session
holds only the thread. That is a different way of working, not a setting.

**The 6.7x is quoted against a sweep, not on its own.** Three inputs decide it
and no transcript can settle any of them: what a delegated read hands back, how
often a delegated step gets redone, and how many tokens a restarted session has
to be told. At the pessimistic corner of all three the multiple is **3.4x**, and
the floor is set by the summary ratio — if a subagent hands back 30% of what it
read instead of 10%, most of the carry it was supposed to avoid comes straight
back. `adder ab` is the only thing here that can test that.

**The guard's threshold is derived, not typed.** The hook fires on a cost
($0.25) rather than a token count, because the same read is worth interrupting
for at turn 400 and not at turn 3. On this workload that gate resolves to 1,500
tokens — under the hook's 2,000-token floor, so the floor is what actually
binds. Worth knowing before tuning a gate that is not doing the work.

Both headline numbers are re-tested by `adder validate` rather than remembered,
and both are workload-dependent: a workload whose sessions stay short has little
carry to remove, and the honest answer there is that the multiple is not
available. Full method in [benchmark.md](docs/benchmark.md).

## Which tier, for this task

```bash
adder policy "make the ingest step tolerate a partial batch" \
  --context 300000 --remaining 200
```

Below is what that prints once `adder outcomes` has some history to work from.
On a fresh machine the log is empty, every rung falls back to the same prior,
and the same command answers `T2` — which is the point: without evidence it
declines to get clever.

**Getting that history used to require a discipline nobody keeps.** The log was
filled by running `adder outcomes record` after every single delegation, by
hand, so in practice it stayed empty, `p_fail` never left its prior, and the
adaptive half of the tool never ran on any machine. The evidence was on disk the
whole time — a delegation is an `Agent` call with a `subagent_type`, and its
outcome is the result that came back:

```bash
adder outcomes import          # show what it found, write nothing
adder outcomes import --write  # append it to the log
```

It is idempotent, so run it whenever. What it reads is unambiguous: an error
result, or the `ESCALATE:` reply the tier agents are told to return. What it
cannot read is a subagent that returned a confident wrong answer and was
believed — and neither can a person filling in the form afterwards, so the
derived rate is a **lower bound** on failure either way. That matters, because
under-estimating `p_fail` is the direction that costs money.

```
DELEGATE -> route-t1 (claude-sonnet-5, effort=medium)
  modelled saving $5.557  routing overhead $0.160  confidence 0.30
  - no high-precision signal; abstaining and routing up
  - T1 costs $0.2032 expected against T2's $0.3890, and the outcome log backs it:
    12% over 40 project runs (recency-weighted mass 39.3)

  Tier chosen by expected cost, including the risk of redoing it:
     T0 claude-haiku-4-5     default $   0.3449   no measured history at this tier; a prior is not evidence
  -> T1 claude-sonnet-5      medium  $   0.2032   run $0.1376 + 12% chance of redoing it
     T2 claude-opus-5        high    $   0.3890   run $0.3520 + 7% chance of redoing it
     T3 claude-opus-5        xhigh   $   0.3997   run $0.3620 + 7% chance of redoing it
```

The task text is four words of nothing, so the classifier abstains — and
abstaining routes *up*, because a misrouted hard task costs a full retry and a
misrouted easy one costs pennies. That used to be the end of it: abstain, get
Opus, forever, no matter how many times Sonnet had already finished this kind of
work in this repo.

Now the tier is whichever rung has the lowest expected cost:

```
E[tier] = run(tier) + p_fail(tier) x (cost of finishing on T2 + the turn that catches it)
```

with the two directions held to different standards, because the two ways of
being wrong are not the same size. Moving **up** needs no evidence — the worst
case is that you paid for the model you would have picked anyway. Moving
**down**, below what the classifier asked for, needs three things at once: the
classifier abstained rather than matched a signal, the outcome log holds enough
recent history at that rung to be evidence rather than a prior wearing a number,
and the measured failure rate is under that rung's own break-even. Cheapness
alone never buys a downgrade — under a no-evidence prior the cheapest rung always
looks best, and that is precisely the reasoning being refused.

The whole ladder is printed, losers included, with the reason each one lost. A
router that shows only its answer is indistinguishable from one that guessed.

Two things this changed that were quietly wrong before. A failed cheap attempt
was being charged twice — `cheap + p x (cheap + expensive)` bills the cheap run
again in a branch where it is not re-run — which made every cheap tier look worse
than it is. And the turn that *notices* the failure was free: a subagent that
returns something wrong does not say so, a main-session turn has to read it and
dispatch again, and at 400K context that turn is not rounding error. Both are on
the books now.

## Which model, across vendors

The routing ladder used to be nine Claude model ids typed into a Python file.
It was correct the day it was written and stale two launches later.

`adder models refresh` builds a catalog from two public sources — LMArena for
head-to-head ratings, OpenRouter's index for prices, context windows, cache
rates and tool support — and joins them into ~500 models from every major lab,
open weights included. This is the **one** command in the whole tool that opens a
network connection. It only ever runs when you type it, and `ADDER_OFFLINE=1`
makes it refuse outright.

```bash
adder pick "port the payment adapter to the new interface" \
  --context 300000 --remaining 120 --combos
```

```
  context 300,000 tok  |  120 turns left  |  difficulty 1.0

  single       $   0.406  elo~1,582  deepseek-v4-flash
                assumes: cheap model is good enough on its own
  draft-review $   0.532  elo~1,648  deepseek-v4-flash + claude-opus-5
                assumes: a review reads 6x less than the generation did
  cascade      $   0.539  elo~1,686  deepseek-v4-flash + claude-opus-5
                assumes: 80% of failures are detected in time to escalate
  single       $   0.728  elo~1,691  claude-opus-5
                assumes: no assumption beyond the price table
  panel        $   1.271  elo~1,691  3x deepseek-v4-flash + claude-opus-5
                assumes: 3 runs fail independently -- they usually do not
```

Note what is being priced. Not "what does this request cost" — what does this
task cost *given this session*: what its tokens cost to carry for the remaining
120 turns, plus the cache rebuild that switching vendors forces on you. Each plan
prints the assumption it depends on, so you can reject the ones you don't believe.

Three gates stop the cheap answers that aren't real. A model that cannot hold the
session isn't quoted for inline work. A model nobody has rated is excluded rather
than assumed fine. And under Claude Code, a non-Claude model can be a subagent
but never the session itself.

`adder models ladder` re-derives the T0/T1/T2 rungs from the catalog and prints
the drift against the constants. It reports; it never silently repoints dispatch.
[models.md](docs/models.md) has the arithmetic and the three ways it goes wrong.

## Using it inside Claude Code

The CLI reports. The `.claude/` directory in this repo is what acts on those
reports, and it ships as part of the product:

- **Agents** — `Explore` on Haiku plus three routing tiers (T0/T1/T2), each with
  rules that bound how much output comes back into your context.
- **Hooks** — a prompt hook that prices the session as you work, and a
  **PreToolUse read guard** that prices a tool result *before* it lands in your
  context. The guard is the only piece here that prevents cost rather than
  reporting it after the fact, so it triggers on what the call will *cost* —
  with 400 turns left, 6,000 tokens is $1.24 to carry and $0.13 delegated —
  rather than on a fixed token count that cannot be right at both ends of a
  session. It advises by default and never blocks silently.

  What it will admit is **predicted from what commands of that shape actually
  returned on your machine**, not from a pattern. The version that guessed
  15,000 tokens for anything containing `cat ` was 105x over the measured
  median, fired on 903 calls whose real output was mostly under 143 tokens, and
  matched none of the eighteen largest results in the corpus. Learned
  prediction cuts the median error from 14,871 tokens to 85, fires a third as
  often, and catches more of the large calls. It also charges its own advice
  against the saving — an injected sentence is carried like any other token —
  and catches the one certain saving in the whole project: re-reading a file
  that is already in the context unchanged, which is **19.2% of unbounded reads
  of text files** here. See [docs/guard.md](docs/guard.md).

  `adder guard` says whether it is installed, what it predicts, and what it has
  cost. Start there — an uninstalled guard, a broken one and a correctly quiet
  one look identical from the outside. `adder guard --replay` prices what it
  would have said to transcripts you have already paid for: on the author's,
  236 findings across 29,464 tool calls, worth $85 against $2.58 of injected
  advice, and only 1.44% of calls cost it a transcript parse.

  Writing that replay paid for itself before it ran on anyone else's data. Its
  first output valued the guard at $1,053 and ranked its biggest findings as
  duplicate reads of PNG screenshots — all of it one bug, sizing every file as
  `bytes / 4` when an image is billed by dimensions and capped near 1,600
  tokens. The corrected number is twelve times smaller and is the one worth
  quoting.
- **Skills** — `/adder` routes one task, `/adder-doctor` diagnoses a session,
  `/adder-init` installs the agents into your project after showing you exactly
  what it will change.

If you want the habit without the machinery, `adder policy "<task>"` gives you
the inline-versus-delegate call for a single task — and refuses to recommend
delegating when the saving would not cover the cost of the routing turn itself.

## Is the tool itself cheaper than not having it?

This is the question a cost tool has to answer about itself, and it is not
rhetorical. Asking adder costs a routing turn, and a routing turn at 500K of
context on Opus is about **$0.26** before anything useful has happened. Advice
worth $0.10 a time, charged at $0.26 a time, is a more expensive way to work.

Write the bill out and there is nothing to argue about:

```
cost_with_adder = baseline - savings + overhead
```

which is below `baseline` exactly when savings cover overhead. Four things keep
that true rather than assuming it.

**Recommendations are priced against their own uncertainty, not just their
midpoint.** Three inputs to every placement decision are estimates with real
spread — how many turns are left, how often this tier fails, how big a summary
comes back. adder now reports the probability that a recommendation is cheaper
than not taking it, and declines when the expected saving is being carried by a
tail rather than by the typical outcome. It also reports the corner where the
advice would lose money, because *"this loses money if the session ends within
23 turns"* is a sentence you can check and *"worst case −$0.03"* is not.

**Delegation is priced as something that can fail.** A delegated read that comes
back missing what you needed costs the subagent run, the turn that noticed, and
the inline read anyway. That term was missing, and its absence made delegation
look free of downside.

**The bar is measured, not assumed.** `adder carry` reads the realized cost of
carrying context off your own transcripts rather than assuming a warm cache:
0.115x here, against the 0.10x the model assumed. The routing overhead every
recommendation has to clear moves with it.

**The tool keeps its own books.** `adder ledger` records the guaranteed saving of
every recommendation acted on against the overhead it cost, and measures the gap
between what predictions promised and what they delivered. If they have been
delivering 60% of face value, every future prediction is scaled by 0.6 before it
meets its gate — so a model that over-promises raises its own bar until it stops.

`adder validate` re-runs all of this against your data, including a 240-case
sweep asserting that **every recommendation the router emits saves more than the
turn that produced it.** That sweep found three counterexamples the day it was
written; they are fixed.

The cheapest advice adder gives is the kind that costs nothing to apply:

```bash
adder carry        # ends with a delegation threshold in tokens
```

A threshold is a rule a hook can apply with no routing turn behind it, so it
cannot cost more than not asking.

## Is the router any good?

Every claim above is about cost. None of them answers the question a sceptical
reader asks first: **the tool routes work to cheaper models — how often is that
the wrong call?** A saving is easy to show and means nothing on its own, because
sending 40% of work to a cheaper model always saves money. What it cost in
quality is the question, and until recently this tool could not answer it.

It can now, using the metrics the routing literature already settled on:

```
$ adder routereval
call-performance curve
  calls   budget  quality      cost     PGR
     0%       0%    0.610    $1.94   0.000
    20%      41%    0.842    $8.10   0.595
    ...

summary
  APGR (calls axis)     0.694  [0.612, 0.771]
  APGR (cost axis)      0.581
  random router         0.500  [0.437, 0.566]
  oracle ceiling        0.780
  CPT(80%)              34% of calls, 61% of budget
```

`PGR` is the share of the quality gap between the weak and strong model that the
router recovered; `APGR` is that averaged over every budget, so a router that
just sends everything to the expensive model cannot score well; `CPT(80%)` is the
share of work you have to send to the expensive model to recover 80% of the gap.
**A router that picks at random scores 0.500**, and that baseline is printed as
an interval so a score of 0.55 on forty episodes is correctly read as "not
distinguishable from random" rather than as a win.

Two things here are not in the standard metric, and both matter for agent
sessions specifically:

- **A dollar axis.** The published metric counts *calls*. A strong call on a
  190K-token context costs roughly forty times a weak call on an 8K one, so a
  router sending 30% of calls to the strong model can be spending 95% of the
  budget. When the two axes disagree, the report says so and the cost axis is
  the true one.
- **An oracle ceiling.** APGR does not top out at 1.0 — its maximum is set by
  how many tasks genuinely need the strong model. Reporting a score without the
  ceiling invites everyone to read 0.75 as a C grade when it is a perfect one.

The other half of routing is `p_fail`, the estimated failure rate every
escalation gate multiplies by. `adder calib` scores it **prequentially** — walk
the log in order, predict each row from only the rows before it, then reveal the
outcome — and reports the Brier skill against simply predicting the base rate.
If the estimator cannot beat a constant, the report says so instead of printing
a per-tier table that looks like evidence.

Underneath both sits a quality signal with a confidence interval that is wider
than people assume: at the top of the coding board the 95% interval is about ±10
points, so a "17-point lead" is two overlapping intervals and no lead at all.
`adder frontier` therefore only lets a model outrank a cheaper one when its
interval clears it — which makes the frontier *narrower* than one drawn on point
estimates, because the models it drops are exactly the ones whose lead is noise.

See [Routing](docs/routing.md) for the full reasoning.

## How much to trust the numbers

The dollar figures on this page come from one machine's transcripts, dominated by
one workload, and they grow every session. (The docs analyse a slightly earlier
snapshot of the same history, which is why their totals are a little lower.)

Your absolute numbers will be different. The *shares* are what drive the advice,
and even those are worth re-checking on your own history — which is the entire
point of the tool. **Run `adder savings` before believing any number on this
page.**

1,151 tests, no API key, and no network outside `adder models refresh`. Two of
those tests exist only to enforce the last two clauses: one walks the code of
every module and fails if anything outside `adder/pricing/sources.py` imports a
networking library, the other fails if the dependency list stops being empty.

## Read more

| | |
|---|---|
| [Benchmark](docs/benchmark.md) | adder vs no adder on the same turns: 1.6x installed, 6.7x followed |
| [Measurement](docs/measurement.md) | the counting bug that halved every number here, and what survived it |
| [Cost model](docs/cost-model.md) | the arithmetic behind 7.8x, and why this is not a model router |
| [Context](docs/context.md) | the tokens nobody put there this turn: what memory costs per turn, when a re-read is waste, when compaction pays, and what a restart may carry |
| [Levers](docs/levers.md) | what each intervention is worth, including the one that turned out not to be available |
| [Quality](docs/quality.md) | how a saving is checked against agent-performance regression |
| [Models](docs/models.md) | choosing across vendors: the catalog, the gates, and why Elo is not a failure rate |
| [Providers](docs/providers.md) | running adder on any LLM: per-provider cache economics, reading OpenAI/Gemini/OTel logs, and what is still assumed |
| [Routing](docs/routing.md) | scoring the router itself: PGR/APGR/CPT, the random baseline, and why rating intervals decide the frontier |
| [Research map](docs/research-map.md) | which published result motivated each command, what was adopted, what was deliberately not built |
| [Systems](docs/systems.md) | placement against cache locality, deadlines on the cheap path, attained service, and where to spend a measurement budget |
| [Commands](docs/commands.md) | full CLI reference and what ships |
| [Architecture](docs/architecture.md) | how the pieces fit, and which invariants are load-bearing |
| [Structure](docs/structure.md) | the package layout: seven layers, why imports only point down, and where a new file goes |
| [Releasing](docs/releasing.md) | how a version gets cut and published |
| [Naming](docs/naming.md) | what `adder` means, what it replaced, and why `pip install` says `adder-cli` |

## The name

A full adder has two outputs. One is the sum. The other is the carry — the bit
that doesn't fit in this column and has to be paid in the next one.

Every cost tool reports the sum. This one reports the carry. An output token is
billed once when it's written and again as context on every turn after it; the
write is the sum, the 340 re-reads are the carry, and on the history above the
carry was 5.7x the sum. It's also a snake, which is the entry fee for a Python
project.

It was called `llm-router` until 2026-08-14. That name advertised the least
interesting thing the tool does — `policy.decide` refuses to emit a routing
recommendation at all when the saving doesn't clear the cost of the routing turn
— and implied it sits in a request path, which it never has. When your cost-model
doc needs a section titled "Why this is not a model router", the name is the
problem. [naming.md](docs/naming.md) has the rest, including why `pip install`
says `adder-cli`.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) first; agents working in this repo should
read [CLAUDE.md](CLAUDE.md), which is the binding version of the same rules.

```bash
pip install -e ".[dev]"
make check        # ruff + pytest, the same thing CI runs
```

The bar here is about numbers, not style: anything that changes a reported figure
needs the measurement behind it in the PR. Two constraints are hard — no runtime
dependencies, and no network call outside `adder/pricing/sources.py`. Both are enforced
by tests rather than by review.

## Security

The tool reads your transcripts, which contain your source code and prompts. It
writes nothing under `~/.claude`, sends nothing anywhere, and holds no
credentials. [SECURITY.md](SECURITY.md) has the threat model and how to report a
problem privately.

## License

[MIT](LICENSE). Changes are recorded in [CHANGELOG.md](CHANGELOG.md).
