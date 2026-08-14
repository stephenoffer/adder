# llm-router

Every cost tool tells you what a turn cost. None of them tell you what it will
keep costing.

An output token in a Claude Code session is billed once when it's written, then
again as cached input on every turn that follows. Write something 340 turns from
the end of a session and you pay to re-read it 340 more times. On one machine's
history that made the real cost of output **5.7x** the number the tools report,
and it moved the biggest saving from "write less" to "stop running 600-turn
sessions."

`rt` reads your local transcript files and shows you the same thing for your own
work. No network, no API key, no model calls.

## See your own numbers

```bash
git clone <this repo> && cd llm-router
./scripts/rt live
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

Python 3.10+, no dependencies. Put `scripts/` on your `PATH` to drop the
`./scripts/` prefix.

## Then find out what it's worth

```bash
./scripts/rt savings
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
rt live                     this session: cost/turn, next-turn cost, pressure
rt trace [--json]           total spend, by model and session
rt debt                     what an output token really costs
rt context                  where context growth comes from
rt cache                    cache hit rate and rebuild waste, by cause
rt quality [--since DATE]   agent-performance proxies
rt policy "<task>"          route a task: inline vs delegate
rt savings                  what each lever is worth
rt verify --since DATE      did a change actually land?
rt outcomes                 escalation calibration (p_fail)
```

Cheaper per turn is not the same as better. `rt quality` tracks tool error rate,
correction rate, interrupts, turns per prompt and rework straight from the
transcripts, and `rt verify` refuses to certify a saving when any of them
regressed.

## Read more

| | |
|---|---|
| [Measurement](docs/measurement.md) | the counting bug that halved every number here, and what survived it |
| [Cost model](docs/cost-model.md) | the arithmetic behind 7.8x, and why this is not a model router |
| [Levers](docs/levers.md) | what each intervention is worth, including the one that turned out not to be available |
| [Quality](docs/quality.md) | how a saving is checked against agent-performance regression |
| [Commands](docs/commands.md) | full CLI reference and what ships |

263 tests, no network, no API key.

The dollar figures above are one machine's transcripts, dominated by one
workload, and they grow every session — the docs analyse a slightly earlier
snapshot of the same history, which is why their totals are a little lower. The
*shares* are what drive the advice, and yours will differ. That's the point of
measuring. Run `rt savings` before believing any number on this page.
