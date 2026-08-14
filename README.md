# adder

[![CI](https://github.com/stephenoffer/adder/actions/workflows/ci.yml/badge.svg)](https://github.com/stephenoffer/adder/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/adder-cli.svg)](https://pypi.org/project/adder-cli/)
[![Python](https://img.shields.io/pypi/pyversions/adder-cli.svg)](https://pypi.org/project/adder-cli/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Every cost tool tells you what a turn cost. None of them tell you what it will
keep costing.

An output token in a Claude Code session is billed once when it's written, then
again as cached input on every turn that follows. Write something 340 turns from
the end of a session and you pay to re-read it 340 more times. On one machine's
history that made the real cost of output **5.7x** the number the tools report,
and it moved the biggest saving from "write less" to "stop running 600-turn
sessions."

`adder` reads your local transcript files and shows you the same thing for your own
work. No API key, no model calls, and no network — with one opt-in exception,
`adder models refresh`, which pulls public model data and only runs when you type it.

## See your own numbers

```bash
pip install adder-cli
adder live
```

Or run it straight from a checkout, with no install step at all:

```bash
git clone https://github.com/stephenoffer/adder && cd adder
./scripts/adder live
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

That table is the pitch. Reading a 50K-token file into a warm session costs $8.81
by the time the session ends; handing it to a subagent that returns a summary
costs $0.96. Same information, 9x apart, and nothing in your usage dashboard
distinguishes them.

Python 3.10+ and zero runtime dependencies, deliberately — it runs on any
machine that has Python, including one with no reachable package index. From a
checkout, put `scripts/` on your `PATH` to drop the `./scripts/` prefix.

## Then find out what it's worth

```bash
./scripts/adder savings
```

Run against the author's history, that reports (abridged):

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

Every figure is labelled with how much it's worth trusting: `MEASURED` from
transcripts, `ATTRIBUTED` from a share of a measured pool, `MODELLED` from
stated assumptions the tool prints alongside the number. The assumptions are the
weak link and it says so.

## Three things the numbers overturned

**Switching to a cheaper model usually loses money.** The prompt cache is
model-scoped. Opus 5 reads cached context at $0.50/MTok; Haiku 4.5 has to read
it fresh at $1.00. Downgrading a warm 544K-token conversation doubles your input
bill, and at that size Haiku can't hold the context anyway. Per-turn routing was
worth $22 out of $4,818.

**Verbosity is not the main lever.** Assistant output is half of context growth.
Tool results are another quarter, and `Bash` alone admits more context than every
other tool combined. No writing-style instruction can reach that.

**The cache was already fine.** 99.2% hit rate, 97% of writes on the 1h TTL,
$0 recoverable. The $302 of rebuild waste came from gaps longer than an hour,
which no TTL setting covers. The tool reports that as unavailable instead of
inventing a saving.

## Commands

```
Measure
  adder live                     this session: cost/turn, next-turn cost, pressure
  adder trace [--json]           total spend, by model and session
  adder debt                     what an output token really costs
  adder context                  where context growth comes from
  adder cache                    cache hit rate and rebuild waste, by cause
  adder quality [--since DATE]   agent-performance proxies
  adder horizon                  remaining-turns estimate vs the naive countdown

Decide
  adder policy "<task>"          route a task: inline vs delegate
  adder outcomes                 escalation calibration (p_fail)
  adder classify "<task>"        task-complexity classification, on its own

Evaluate
  adder savings                  what each lever is worth
  adder verify --since DATE      did a change actually land?
  adder validate                 re-test the claims everything rests on
  adder regret                   dollar regret of the horizon estimator
  adder simulate                 replay sessions under interventions
  adder ab                       controlled A/B on answer quality
```

`adder help` prints the full list, `adder <command> --help` the flags for one.
Full reference: [docs/commands.md](docs/commands.md).

Cheaper per turn is not the same as better. `adder quality` tracks tool error rate,
correction rate, interrupts, turns per prompt and rework straight from the
transcripts, and `adder verify` refuses to certify a saving when any of them
regressed.

## Which model, across vendors

The routing ladder used to be nine Claude model ids typed into a Python file.
It was correct the day it was written and stale two launches later.

`adder models refresh` builds a catalog from two public sources — LMArena for
head-to-head ratings, OpenRouter's index for prices, context windows, cache
rates and tool support — and joins them into ~500 models from every major lab,
open weights included. Then:

```bash
./scripts/adder pick "port the payment adapter to the new interface" \
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

Every plan is priced through *this* session's economics, not per request: the
term that decides it is what the task's tokens cost to carry for the remaining
120 turns, plus the prefix rebuild a vendor switch forces. Three gates stop the
cheap answers that aren't real — a model that can't hold the session isn't
quoted inline, a model nobody has rated is excluded rather than assumed fine,
and under Claude Code a non-Claude model can be a subagent but never the
session itself.

`adder models ladder` re-derives the T0/T1/T2 rungs from the catalog and prints the
drift against the constants. It reports; it never silently repoints dispatch.
[models.md](docs/models.md) has the arithmetic and the three ways it goes wrong.

## Read more

| | |
|---|---|
| [Measurement](docs/measurement.md) | the counting bug that halved every number here, and what survived it |
| [Cost model](docs/cost-model.md) | the arithmetic behind 7.8x, and why this is not a model router |
| [Levers](docs/levers.md) | what each intervention is worth, including the one that turned out not to be available |
| [Quality](docs/quality.md) | how a saving is checked against agent-performance regression |
| [Models](docs/models.md) | choosing across vendors: the catalog, the gates, and why Elo is not a failure rate |
| [Commands](docs/commands.md) | full CLI reference and what ships |
| [Architecture](docs/architecture.md) | how the pieces fit, and which invariants are load-bearing |
| [Releasing](docs/releasing.md) | how a version gets cut and published |
| [Naming](docs/naming.md) | what `adder` means, what it replaced, and why `pip install` says `adder-cli` |

442 tests, no network, no API key. Two of those tests exist only to enforce
the last two clauses: one walks the AST of every module and fails if anything
outside `adder/sources.py` imports a networking library, the other fails if
`[project.dependencies]` stops being empty.

The dollar figures above are one machine's transcripts, dominated by one
workload, and they grow every session — the docs analyse a slightly earlier
snapshot of the same history, which is why their totals are a little lower. The
*shares* are what drive the advice, and yours will differ. That's the point of
measuring. Run `adder savings` before believing any number on this page.

## The name

A full adder has two outputs. One is the sum. The other is the carry — the bit
that doesn't fit in this column and has to be paid in the next one.

Every cost tool reports the sum. This one reports the carry. An output token is
billed once when it's written and again as cached input on every turn after it;
the write is the sum, the 340 re-reads are the carry, and on the history above
the carry was 5.7x the sum. It's also a snake, which is the entry fee for a
Python project.

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
dependencies, and no network call outside `adder/sources.py`. Both are enforced
by tests rather than by review.

## Security

The tool reads your transcripts, which contain your source code and prompts. It
writes nothing under `~/.claude`, sends nothing anywhere, and holds no
credentials. [SECURITY.md](SECURITY.md) has the threat model and how to report a
problem privately.

## License

[MIT](LICENSE). Changes are recorded in [CHANGELOG.md](CHANGELOG.md).
