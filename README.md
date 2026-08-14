# llm-router

A cost router for Claude Code and Claude API agents that optimises **where work
runs**, not just which model runs it.

## Why this is not another model router

Existing routers (RouteLLM, NotDiamond, Martian, OpenRouter auto, vLLM Semantic
Router) pick a model per request and price it as `in x rate_in + out x rate_out`.
That is correct for stateless APIs. It is wrong for agent sessions, because the
conversation prefix is re-sent on **every turn**. A token admitted to a context
is billed once as a cache write and then again, as a cache read, on every
remaining turn:

```
admitted_token_cost(n, model, remaining_turns)
  = n * rate_in * 1.25                       # once
  + n * rate_in * 0.10 * remaining_turns     # forever after
```

Measured on 169 real transcripts (~$7.2K list-equivalent):

| | |
|---|---|
| Input-side spend | **90%** |
| Cache-read alone | **78%** |
| Output-side spend | 10% |
| Median session | 607 turns @ 544K context |
| Most expensive session | $1,001 in 3,478 turns |

**Model selection can only touch the output-side 10%** — and taking it means
invalidating a model-scoped cache. Opus 5 reads cached context at $0.50/MTok;
Haiku 4.5 reads it fresh at $1.00/MTok. Downgrading a warm conversation makes
input **2x more expensive**. Break-even is `output > context / 40`: at a 544K
context you need >13.6K output tokens per turn. The measured average is 818.

So a naive router loses money on exactly the long sessions it is meant to help.
This one models cache state and declines.

## What it actually does

| Lever | Worth | Confidence |
|---|---|---|
| Split sessions >300 turns | ~$3,700 | modelled |
| Delegate reads to subagents | ~$1,500 | modelled |
| Per-turn model downgrade | ~$55 | modelled |
| Run Explore/subagents on Haiku | ~$42 | **measured** |

Split and Delegate are **not additive** — both draw on the same cache-read pool.

Note the ordering: per-turn model routing, the usual product framing, is the
*smallest* lever on this workload by roughly two orders of magnitude.

## Install

```bash
git clone <this repo> && cd llm-router
python3 -m pytest tests/ -q          # 67 tests, no network, no API key
cp .claude/agents/*.md ~/.claude/agents/     # Layer 0: takes effect immediately
```

`Explore.md` overrides the built-in Explore agent to run on Haiku. As of
v2.1.198 Explore *inherits* the session model, so on an Opus session all
exploration runs at 5x the necessary rate. This is four lines of YAML and the
best effort-to-saving ratio in the repo.

## Use

```bash
scripts/rt live                      # what this session costs, per turn
scripts/rt savings                   # where money went, what each fix is worth
scripts/rt trace --verify            # reproduce the headline numbers
scripts/rt policy "refactor auth"    # route one task
scripts/rt classify "..." --json     # classification only
```

In Claude Code: `/route-doctor` (diagnose), `/route <task>` (dispatch),
`/route-init` (install).

## As a library

```python
from router.cost import admitted_token_cost, switch_is_profitable, placement_cost

admitted_token_cost(10_000, "claude-opus-5", remaining_turns=1000)   # 5.06

d = switch_is_profitable("claude-opus-5", "claude-haiku-4-5",
                         ctx_tokens=544_000, est_out_tokens=818)
bool(d)     # False
d.reason    # 'loses $0.2506: re-reading 544,000 tok uncached on claude-haiku-4-5 ...'
```

Gates return a `Decision`, which is falsy when the action loses money and always
explains why in `.reason`. Nothing returns a bare bool.

## Design decisions worth knowing

- **Declining to route is the default.** A routing turn re-reads the whole
  context: ~$0.25 at 500K tokens on Opus, before doing anything useful. If the
  modelled saving does not clear that, the router says "do it inline".
- **The classifier abstains a lot.** Text cannot predict how deep a coding task
  goes; it fires only on high-precision extremes and otherwise routes *up*. A
  misrouted hard task costs a full retry; a misrouted easy one costs pennies.
- **Cheap tiers are read-only.** `route-t0` cannot write, so escalating away from
  it can never leave half-applied edits behind. `route-t1` must escalate *before*
  its first edit.
- **Prices are date-aware.** Sonnet 5's introductory $2/$10 reverts to $3/$15 on
  2026-08-31, a 50% jump that moves every threshold.
- **Attribution is pinned to measured spend.** Two attribution bugs shipped
  during development (`cache_write` overcounts admitted content ~5x because
  Claude Code refreshes cache segments; context growth overshoots because
  compaction shrinks contexts). There is now an assertion that attributed cost
  can never exceed the recorded cache-read bill.

## Limitations, stated plainly

- **Trace replay bounds cost, not quality.** Transcripts contain only what Opus
  produced. Nothing here shows what Haiku *would* have produced, so no quality
  claim is derivable from replay. That needs live A/B against objective criteria
  (tests pass, build succeeds), which is not yet built.
- **Savings marked MODELLED rest on stated assumptions** — 30% of content
  delegable, 10:1 compression, sessions separable at turn boundaries. Only the
  Explore figure is measured.
- **Public benchmarks will show parity, not a win.** RouterArena and RouteLLM's
  MT-Bench setup are single-turn and stateless, so they cannot exercise context
  amortisation — the thing this router is built around. The defensible claim is
  "better on stateful agentic workloads", which is where the spend actually is.
- **Cost is list-price equivalent.** Subscription seats do not pay per token; the
  optimisation target is unchanged either way.
- `CLAUDE_CODE_SUBAGENT_MODEL` and org `availableModels` allowlists silently
  override model choice. `/route-init` checks the first; always verify the model
  that actually ran from the transcript rather than trusting an agent's claim.
