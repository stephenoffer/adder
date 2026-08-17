# adder

[![CI](https://github.com/stephenoffer/adder/actions/workflows/ci.yml/badge.svg)](https://github.com/stephenoffer/adder/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/adder-cli.svg)](https://pypi.org/project/adder-cli/)
[![Python](https://img.shields.io/pypi/pyversions/adder-cli.svg)](https://pypi.org/project/adder-cli/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Your coding agent's bill is bigger than your dashboard says.** adder reads the
transcripts already sitting on your disk, shows you where the money went, and
tells you what the rest of the session is going to cost.

```bash
pip install adder-cli
adder live
```

No account, no API key, no config file. It never calls a model, and the only
command that touches the network is `adder models refresh`, which runs when you
type it and never otherwise. Your transcripts are never modified.

Built against Claude Code, where its numbers were measured. It works on any
provider: OpenAI, Google, DeepSeek and hosted open-weight endpoints are priced
with *their own* cache economics, not Anthropic's borrowed
([providers.md](docs/providers.md)).

## The one idea

**You don't pay for a piece of text once. You pay for it on every turn after it
appears.**

When the agent writes 1,000 tokens, you're billed for writing them. Then they
join the context, so you're billed again to re-read them on turn 2. And turn 3.
Write something with 340 turns left and you pay for it 341 times.

| turns remaining | 0 | 50 | 200 | 340 | 759 |
|---|---|---|---|---|---|
| **what a token really cost** | 1.0x | 2.0x | 5.0x | **7.8x** | 16.2x |

Past roughly 50 remaining turns, re-reading a token costs more than writing it
did. Your usage dashboard shows you the 1.0x column.

Two things follow, and they're the whole tool:

- The expensive decision is rarely *which model you used*. It's **what you let
  into the context, and how early you let it in**.
- A 600-turn session isn't one long session. It's the same context, re-read 600
  times.

## What you see first

```
$ adder live
  This session: 11 turns · 47,401 tokens in context · $0.89 spent ($0.081/turn)
  Sessions that reach turn 11 typically run ~340 more turns → ~$28.27 total

  Cost of reading a file into THIS context from here:
       file size      inline   delegated   verdict
        20,000 tok  $    3.525  $    0.383   delegate
        50,000 tok  $    8.812  $    0.957   delegate
       150,000 tok  $   26.438  $    2.869   delegate
```

$0.89 spent, $28.27 owed. Then look at the table. Reading a 50,000-token file
into this session costs **$8.81** by the time the session ends. Handing that
same file to a subagent that reads it and returns a summary costs **$0.96**.
Same information, **9x apart**, and nothing in your usage dashboard tells the
two apart.

So the habit is short: **stop pulling large things into long conversations.
Delegate the reading, keep the answer.**

## What it's worth

Two numbers, because quoting one of them would be dishonest. Replaying one
machine's 23,922 recorded turns:

| | |
|---|---|
| Install it, change nothing | **1.6x cheaper** |
| Work the way it tells you to | **6.7x cheaper** |

The first row is the read guard and the tier agents, which act without being
obeyed. The second is advice, and nothing enforces it. That row is the
orchestrator pattern, where almost every step that would admit content runs
somewhere else; at the pessimistic corner of its three softest assumptions it
drops to 3.4x. Both numbers are re-tested by `adder validate` rather than
remembered. ([benchmark.md](docs/benchmark.md))

On that same history, 73% of a $4,818 bill was not new work at all. It was
context already paid for once, being re-read. `adder savings` ranks the ways to
attack that, and the top one isn't model choice or writing style. It's **ending
sessions sooner**, because session length multiplies everything else.
([levers.md](docs/levers.md))

## Three things the numbers overturned

**"Just use a cheaper model" usually loses money — but *starting* on one is the
biggest lever there is.** The prompt cache is model-scoped, so moving a warm
conversation throws the discount away. Switching mid-session is worth 0.5% of
spend; starting cheap is worth 60%.

**Verbosity isn't the main lever.** Assistant output is only half of context
growth. Tool results are another quarter, and `Bash` alone admits more than
every other tool combined. No writing-style instruction reaches that.

**The cache was already fine.** 99.2% hit rate, $0 recoverable. The $302 of
waste came from hour-long gaps between turns, which no cache setting covers. The
tool reports that as unavailable rather than inventing a saving it can't
deliver.

## Cheaper is not the same as better

A tool that only counts dollars will happily talk you into worse work.

`adder quality` reads those transcripts for signs the agent is struggling: tool
error rate, how often you correct it, how often you interrupt, turns per prompt,
rework. `adder verify` then refuses to certify a saving if any of those got
worse. A change that cut your bill and doubled your rework isn't a saving. It's
a cost you moved somewhere the invoice can't see.

## The commands you'll actually use

| You want to know | Run |
|---|---|
| Just tell me what's wrong | `adder doctor` |
| What is this session costing me right now? | `adder live` |
| Where has all my money gone? | `adder trace` |
| Why is my context so big? | `adder context` |
| Of everything I could change, which is worth most? | `adder savings` |
| If I followed all of that, what would the bill be? | `adder plan` |
| Should I do this task here, or delegate it? | `adder policy "<task>"` |
| Did the change I made last week actually work? | `adder verify --since DATE` |

Forty-odd more, each with `--json` and the same window flags:
[getting-started.md](docs/getting-started.md) walks through the first run,
[commands.md](docs/commands.md) is the full reference.

## How much to trust the numbers

Every dollar figure here comes from one machine's transcripts, dominated by one
workload. Your absolute numbers will differ. The *shares* are what drive the
advice, and even those are worth re-checking against your own history, which is
the entire point of the tool. **Run `adder savings` before believing any number
on this page.**

3,117 tests, no API key, no runtime dependencies, and no network outside `adder
models refresh`. Two of those tests exist only to enforce the last two clauses.

## Read more

| | |
|---|---|
| [Getting started](docs/getting-started.md) | your first run, the vocabulary, what each command answers, what adder writes to disk |
| [Benchmark](docs/benchmark.md) | adder vs no adder on the same turns: 1.6x installed, 6.7x followed |
| [Cost model](docs/cost-model.md) | the arithmetic behind 7.8x, and why this is not a model router |
| [Levers](docs/levers.md) | what each intervention is worth, and the one that turned out not to be available |
| [Context](docs/context.md) | the tokens nobody put there this turn: memory, re-reads, compaction, restarts |
| [Guard](docs/guard.md) | the only thing here that prevents spend rather than reporting it |
| [Tiers](docs/tiers.md) | picking a tier by expected cost, and why a downgrade needs evidence that an upgrade doesn't |
| [Models](docs/models.md) | choosing across vendors: the catalog, the gates, and why Elo is not a failure rate |
| [Routing](docs/routing.md) | scoring the router itself: PGR/APGR/CPT, the random baseline, rating intervals |
| [Overhead](docs/overhead.md) | whether the advice is worth the turn spent asking for it |
| [Quality](docs/quality.md) | how a saving is checked against agent-performance regression |
| [Measurement](docs/measurement.md) | the counting bug that halved every number here, and what survived it |
| [Providers](docs/providers.md) | per-provider cache economics, reading OpenAI/Gemini/OTel logs, what is still assumed |
| [Research map](docs/research-map.md) | which published result motivated each command, and what was deliberately not built |
| [Systems](docs/systems.md) | placement against cache locality, deadlines on the cheap path, attained service |
| [Architecture](docs/architecture.md) · [Structure](docs/structure.md) | how the pieces fit, and where a new file goes |
| [Releasing](docs/releasing.md) · [Naming](docs/naming.md) | how a version gets cut; what `adder` means |

## The name

A full adder has two outputs. One is the sum. The other is the carry — the bit
that doesn't fit in this column and has to be paid in the next one.

Every cost tool reports the sum. This one reports the carry. An output token is
billed once when it's written and again as context on every turn after; the
write is the sum, the 340 re-reads are the carry, and on the history above the
carry was 5.7x the sum. It's also a snake, which is the entry fee for a Python
project. ([naming.md](docs/naming.md))

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) first; agents working in this repo
should read [CLAUDE.md](CLAUDE.md), which is the binding version of the same
rules.

```bash
pip install -e ".[dev]"
make check        # ruff + pytest, the same thing CI runs
```

The bar here is about numbers, not style: anything that changes a reported
figure needs the measurement behind it in the PR. Two constraints are hard, and
both are enforced by tests instead of by review. No runtime dependencies, and
no network call outside `adder/pricing/sources.py`.

## Security

adder reads your transcripts, which contain your source code and prompts. It
writes nothing under `~/.claude/projects`, sends nothing anywhere, and holds no
credentials. [SECURITY.md](SECURITY.md) has the threat model and how to report a
problem privately.

## License

[MIT](LICENSE). Changes are recorded in [CHANGELOG.md](CHANGELOG.md).
