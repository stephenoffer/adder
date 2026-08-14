# llm-router

Cost tooling for Claude Code agent sessions. It reads your local transcript
files and tells you where the money actually goes — no network, no API key, no
model calls.

The finding it is built around: **in a long session, an output token is a
liability, not an expense.** It is billed once when generated, then again as
cached input on every remaining turn. At a 340-turn median session that is
**7.8x** the price every other cost tool reports.

## Quick start

```bash
git clone <this repo> && cd llm-router
./scripts/rt live       # this session: cost/turn, next-turn cost, pressure
./scripts/rt trace      # total spend across all your sessions
./scripts/rt savings    # what each lever is worth on your own history
```

Python 3.10+, no dependencies. Put `scripts/` on your `PATH` to drop the
`./scripts/` prefix.

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
rt horizon                  remaining-turns estimate vs the naive countdown
rt validate                 re-test the claims everything rests on
rt regret                   dollar regret of the horizon estimator
rt ab                       controlled A/B on answer quality
```

Full reference: [docs/commands.md](docs/commands.md).

## What it found

Measured across 171 transcript files (50 sessions, 18,163 turns, $4,456
list-equivalent):

- **92% of spend is input-side**, 78% cache reads alone. Output is 8%.
- The five biggest levers compose to **~$2,999, or 67% of measured spend** —
  and the biggest of them is splitting long sessions, not writing less.
- Per-turn model downgrades are worth **$21**, and often lose money: the prompt
  cache is model-scoped, so moving a warm session to a cheaper model makes input
  2x more expensive.

## Read more

| | |
|---|---|
| [Measurement](docs/measurement.md) | the dedup bug that halved the numbers, and what survived it |
| [Cost model](docs/cost-model.md) | why an output token costs 7.8x sticker, and why this is not a model router |
| [Levers](docs/levers.md) | what each intervention is worth, including the one that wasn't available |
| [Quality](docs/quality.md) | how a saving is checked against agent-performance regression |
| [Commands](docs/commands.md) | full CLI reference and what ships |

263 tests, no network, no API key.

These figures come from one machine's transcripts, dominated by one workload.
Run `rt savings` against your own history before believing any of them.
