# llm-router

Cost tooling for Claude agent sessions, built around a measured finding: in a
long session, **an output token is a liability, not an expense**.

## The finding

Measured across 169 real transcripts (~$7.4K list-equivalent):

| | |
|---|---|
| Input-side spend | 90% (cache-read alone: 78%) |
| Output-side spend | 10% |
| **Assistant output as share of context growth** | **~105%** |

The context of a long agent session is, almost entirely, the model's own
previous words being re-read. So the input-side 90% is a *symptom*; output is the
cause. An output token is billed once at generation, then again as cached input
on every remaining turn:

```
true_cost(1 token, R remaining turns) = rate_out + rate_in * 0.10 * R
```

On Opus 5 that is `$25/MTok + $0.50/MTok per turn` — a multiple of `1 + R/50`:

| remaining turns | 0 | 50 | 200 | 607 | 1,000 | 3,478 |
|---|---|---|---|---|---|---|
| **cost vs sticker** | 1.0x | 2.0x | 5.0x | 13.1x | 21.0x | **70.6x** |

Past **50 remaining turns**, re-reading an output token costs more than
generating it did. On this dataset: **$704 generated, $5,320 in downstream
re-reads.** Every cost tool reports the $704.

## Why this is not a model router

Existing routers (RouteLLM, NotDiamond, Martian, OpenRouter auto, vLLM Semantic
Router) pick a model per request and price it `in x rate_in + out x rate_out`.
That is correct for stateless APIs and wrong for agent sessions.

Worse, the obvious move actively loses money. Opus 5 reads cached context at
$0.50/MTok; Haiku 4.5 reads it fresh at $1.00/MTok — the cache is model-scoped,
so downgrading a warm conversation makes input **2x more expensive**. Break-even
is `output > context / 40`: at the measured median context (544K) that needs
13.6K output tokens per turn. The measured average is **818**.

Per-turn model routing is worth ~**$55** of the $7.4K here. It ships, correctly
gated, as the smallest of five levers.

## One root cause, three substitutes

Because output accumulating in context is the root cause, there are exactly three
ways to attack it — and they are **substitutes, not complements**:

| Lever | Worth | Confidence |
|---|---|---|
| Split sessions >300 turns | $3,794 | modelled |
| Write 30% less (leverage **7.6x**) | $1,813 | attributed |
| Delegate 25% of turns to subagents | $1,342 | modelled |
| *(separate)* per-turn model downgrade | $55 | modelled |
| *(separate)* Explore/subagents on Haiku | $42 | **measured** |

Summing the first three double-counts. Composed multiplicatively on the residual:
**~$4,950, or 67% of measured spend.**

## The result that changes the advice

Running the built-in verifier across a real cutover date:

```
output tokens / turn    1,352 ->    760    -43.8%
cost / turn           $0.2299 -> $0.2326    +1.2%
median context        290,791 -> 353,721   +21.6%
turns / session           389 ->    776    +99.7%

  verbosity effect      x0.562
  session-length effect x1.997
  predicted context     x1.123   (product)
  actual context        x1.216
```

**Writing 44% less produced no saving, because sessions got twice as long.**
Cost per turn tracks context, context tracks *cumulative* output, and cumulative
output is `output-per-turn x session-length`. The two factors multiply, so
terseness only pays if session length is held constant.

This is why the tool reports failure rather than claiming a win — and why session
length, not verbosity or model choice, is the dominant lever.

## Install

```bash
python3 -m pytest tests/ -q          # 108 tests, no network, no API key
cp .claude/agents/*.md ~/.claude/agents/
```

`Explore.md` overrides the built-in Explore to run on Haiku. As of v2.1.198
Explore *inherits* the session model, so on an Opus session all exploration runs
at 5x the necessary rate. Four lines of YAML, no code path involved.

## Use

```bash
scripts/rt debt                       # true cost of an output token, and of writing less
scripts/rt savings                    # every lever, with confidence tiers
scripts/rt verify --since 2026-08-01  # did an intervention actually land?
scripts/rt live                       # this session: $/turn, cost of the next read
scripts/rt trace --verify             # reproduce the headline figures
scripts/rt policy "refactor auth"     # route one task
```

In Claude Code: `/route-doctor`, `/route <task>`, `/route-init`.

## As a library

```python
from router.debt import debt_multiple, decompose_read_cost
from router.cost import switch_is_profitable

debt_multiple(607)          # 13.1  -- output token cost multiple at median session
decompose_read_cost(sess)   # (measured_total, irreducible_baseline, addressable)

d = switch_is_profitable("claude-opus-5", "claude-haiku-4-5",
                         ctx_tokens=544_000, est_out_tokens=818)
bool(d)     # False
d.reason    # 'loses $0.2506: re-reading 544,000 tok uncached on ...'
```

Gates return a `Decision`, falsy when the action loses money, always explaining
why in `.reason`. Nothing returns a bare bool.

## Design decisions worth knowing

- **Declining to route is the default.** A routing turn re-reads the whole
  context: ~$0.25 at 500K tokens on Opus, before doing anything useful. If the
  modelled saving does not clear that, the router says "do it inline".
- **The classifier abstains a lot.** Text cannot predict how deep a coding task
  goes; it fires only on high-precision extremes and otherwise routes *up*.
- **Cheap tiers are read-only.** `route-t0` cannot write, so escalating away from
  it can never leave half-applied edits. `route-t1` must escalate before its
  first edit.
- **Prices are date-aware.** Sonnet 5's introductory $2/$10 reverts to $3/$15 on
  2026-08-31, a 50% jump that moves every threshold.
- **Every attribution is bounded by measured spend.** Three over-claiming bugs
  were caught during development by self-checks that assert attributed cost can
  never exceed the recorded bill:
  - `cache_write` overcounts admitted content ~5x (Claude Code refreshes cache
    segments);
  - context-growth reconstruction overshoots because compaction shrinks contexts;
  - forward-projecting output debt overshot the real bill by ~35%.

## Limitations, stated plainly

- **Nothing here proves a quality-neutral saving.** Transcripts contain only what
  Opus produced; no counterfactual quality claim is derivable from replay. The
  live A/B (objective pass criteria) is not built.
- **`verify --since` is uncontrolled**, not an A/B. Task mix changes between
  periods too. It is honest enough to report failure, which it currently does.
- **MODELLED figures rest on stated assumptions** — 25% of turns delegable, 10:1
  compression, sessions separable at turn boundaries. Only the Explore figure and
  the debt decomposition are measured.
- **Public benchmarks will show parity, not a win.** RouterArena and RouteLLM's
  MT-Bench setup are single-turn and stateless, so they cannot exercise context
  amortisation. The defensible claim is "better on stateful agentic workloads".
- **Cost is list-price equivalent.** Subscription seats do not pay per token; the
  optimisation target is unchanged either way.
- `CLAUDE_CODE_SUBAGENT_MODEL` and org `availableModels` allowlists silently
  override model choice. Verify from the transcript's `message.model`, never from
  an agent's own claim.
