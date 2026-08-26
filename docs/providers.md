# Providers: making the cost model work for any LLM

`adder` was built against Claude Code transcripts and its cost model was built
against Anthropic's pricing. Those are two different facts, and only the first
one is a limitation worth keeping.

This document is about the second: what had to change so that a cost model
written around one vendor's cache economics produces correct numbers for every
other vendor, and what is still assumed.

## The one number that decides everything

In a persistent agent session the dominant cost term is not the price of a
turn. It is the price of *carrying* the turn's tokens for the rest of the
session:

```
admitted_token_cost(n, remaining_turns)
  = n * cache_write_rate                    # once, when it enters
  + n * cache_read_rate * remaining_turns   # again, on every turn after
```

On the measured transcripts that second term is about 76% of total spend. So
`cache_read_rate` is not a detail. It is the number the whole tool turns on,
and it is the number that differs most between providers.

## What actually differs

| | Anthropic | OpenAI | Google | DeepSeek | Many hosted open-weight endpoints |
|---|---|---|---|---|---|
| Cache style | explicit | automatic | automatic | automatic | none |
| Write premium | 1.25x (5m), 2.00x (1h) | none | none | none | n/a |
| Read discount | 0.10x | 0.25x (4.x), 0.10x (5.x) | ~0.25x | ~0.10x | none |
| TTL selectable | yes | no | no | no | n/a |
| Storage billed | no | no | yes, per hour | no | no |
| Breakpoints | up to 4 | n/a | n/a | n/a | n/a |

Three consequences, in descending order of how much money they move:

**A provider with no cache has a carry term ten times larger.** Re-reading the
prefix costs full input rate, not a tenth of it. Borrowing Anthropic's 0.10x for
such a model does not make the estimate slightly optimistic — it points the
recommendation the wrong way, telling you to admit tokens to a context that
cannot amortize them. `adder` now prices an unknown provider as *no cache*,
deliberately: between a report that is too cautious and one that is confidently
wrong, this repo picks too cautious.

**Automatic caching has no write premium, because it has no write decision.**
An uncached prefix is billed as ordinary input and the cache happens as a side
effect. Applying Anthropic's 1.25x to an OpenAI transcript invents a quarter of
the write side of the bill. It also means "place a cache breakpoint" is not
advice that applies (there is nothing to place), and `adder` no longer offers
it where it cannot be taken.

**Google bills storage per hour.** Every other cost term in this tool is driven
by tokens moved. That one is driven by elapsed time, so on Google an idle
session is not free, and no amount of prompt discipline reduces it. It is
carried as its own term rather than folded into the write rate.

## Where each number comes from

Resolution runs in three layers, most authoritative first:

1. **`pricing/prices.py`**: hand-checked first-party Claude rates. Date-aware,
   so an introductory rate expires on schedule. Wins for Claude, always.
2. **`pricing/catalog.py`**: ~500 models joined from public sources, each
   carrying its provenance and its age. Where a provider publishes an absolute
   cache rate, that rate is used verbatim.
3. **`pricing/providers.py`**: the table above. Fills in the mechanics the
   catalog does not publish, which is most of them: 273 of the 510 bundled
   entries have no cache read rate at all.

`pricing/registry.py` is the single place that joins them. Every report asks it
the same question and gets a `ModelSpec` back, with `rate_provenance` saying
which layer answered: `published`, `derived from <vendor>'s published
multiplier`, or `MODELLED from a <vendor> default`. A number and the reason to
believe it travel together.

### Two data traps that were live

**A published `cache_write` is not always a write rate.** Aggregators put
storage in the same field. Google's `gemini-3.7-flash` reports `0.0208` against
an input rate of `0.75`, which is per-million *per hour* of storage; taking
it for a write rate prices a cache write at 2.8% of input, a 36x
understatement. The tell is an ordering that cannot be true: you cannot pay less
to create a cache entry than to read one back.

**A short catalog key is a greedy prefix.** The bundled snapshot contains
`~openai/gpt-latest`, a floating alias that normalizes to the bare key `gpt`.
Under a plain prefix match every unrecognised OpenAI id (anything newer than
the snapshot) matched it and was silently priced at that alias's $5/$30, *with
no warning*, because resolution had succeeded. An unknown model reported as
unknown is a caveat in the output; an unknown model priced off a wildcard is a
wrong number presented as a measurement. Prefix matching now refuses floating
aliases and refuses to cross a version boundary.

## Reading a non-Claude session

`core/ingest.py` normalizes usage records from Claude Code JSONL, the Anthropic
Messages API, OpenAI Chat Completions and Responses, Gemini, OpenTelemetry
GenAI spans, and generic logs. Point any report at a directory of them:

```bash
adder trace ./logs          # any of the above, .json or .jsonl
adder cache ./logs
adder debt ./logs
```

Format is sniffed **per record**, not per file, because a proxy log interleaves
three providers and a format decided once from the first line would mis-read the
other two.

### The conversion that matters

Providers disagree about whether the cached prefix is *inside* the input count.

- Anthropic: `input_tokens` is uncached input only. The three counts are
  disjoint and sum to the context.
- OpenAI: `prompt_tokens` is the whole prompt, and
  `prompt_tokens_details.cached_tokens` is the part of that total served from
  cache. They overlap.
- Google: same as OpenAI.

Read an OpenAI record with Anthropic's semantics and every cached token is
counted twice: once at full rate inside `prompt_tokens`, once more at the cache
read rate. A turn with `prompt_tokens: 50000` and `cached_tokens: 48000` becomes
a 98,000-token context that never existed, priced about double. Every adapter
states which convention it converts *from*, and the overlapping ones subtract.

The OpenTelemetry convention does not say which it is, and instrumentations
differ because each mirrors the SDK underneath it. So the provider decides, and
anything unrecognised is treated as disjoint, because subtracting a cache read
that was never included would understate spend, which is the failure this repo
is least willing to ship.

## Telling adder what you run

Three settings, resolvable from `~/.claude/adder.json`, `./.adder.json`, or the
environment. `adder config` prints what is in effect and which layer set it.

```bash
export ADDER_MODEL=gpt-5                                   # session model
export ADDER_HARNESS=codex                                 # agent runtime
export ADDER_LADDER="T0=gpt-5-mini,T1=gpt-5,T2=gpt-5-pro"  # dispatch tiers
```

**`harness`** is which agent runtime is driving, and it decides what placements
exist. Some harnesses pin the main conversation to one vendor by construction:
Claude Code to Anthropic, Codex to OpenAI, Gemini CLI to Google. Under those, a
model from another vendor can be a subagent, an MCP tool, or an external call,
but it cannot *be* the session, and quoting an inline price for one is quoting a
placement that does not exist. `aider`, `openhands`, `custom` and `any` impose
no pin. See `core/harness.py`; add your own with `ADDER_HARNESSES=<path>`.

**`ladder`** repoints the dispatch tiers. The default is Claude because that is
what the measurements were taken on, and the catalog deliberately *reports*
drift rather than silently repointing dispatch. Unnamed rungs keep their
default, so a partial override cannot leave a tier pointing at nothing. It can,
though, leave the ladder non-monotone, and
`classify.ladder_warnings()` says so:

```
!  T3 (claude-opus-5) is cheaper than T2 (gpt-5-pro); the ladder does not
   climb, so escalating from T2 to T3 saves money instead of spending it
```

## Extending it

Both tables are data and both take a local override, so a new vendor, a cache
tier that shipped after this was written, or a negotiated rate does not need a
fork:

```bash
export ADDER_PROVIDERS=~/.claude/adder-providers.json
export ADDER_HARNESSES=~/.claude/adder-harnesses.json
```

```json
{"providers": {"acme": {
  "cache_style": "automatic",
  "cache_read_mult": 0.2,
  "cache_write_mult": {"auto": 1.0},
  "batch_mult": 0.4
}}}
```

A file that mentions one field amends that field and leaves the rest of the
record alone: pinning a negotiated cache read rate must not silently reset the
TTL table. A corrupt override degrades to the built-in table rather than taking
down every report, and a field that cannot be a number becomes `None`, which
every gate already reads as *unknown* and never as *free*.

## What is still assumed

Worth saying plainly, since the point of this tool is that a number carries its
own caveats:

- **Provider mechanics marked `verified=False` are MODELLED.** Google's storage
  rate, DeepSeek's and xAI's discounts, Bedrock's cache behaviour. They are
  documented defaults, not read off a bill.
- **The quality signal is still arena Elo**, which measures human preference on
  chat and web-dev prompts, not multi-file agentic tool use. It is used because
  it is the only cross-vendor signal that updates within days of a launch and is
  not self-reported. `outcomes.p_fail` overrides it the moment there is measured
  history.
- **Effort vocabularies differ and are not translated.** Anthropic takes labels
  up to `max`, OpenAI adds `minimal` and stops at `high`, Google takes an
  integer thinking budget. `adder` refuses a level a model does not accept
  rather than mapping it to a neighbour, because asking for a level the model
  rejects is a 400, not a cheaper turn.
- **The carry model is fitted to Claude Code transcripts.** The *shape* of it
  (context growth per turn, survival across compaction) is a property of how an
  agent works, not of who serves it, but it has only been measured on one
  harness. `adder carry` refits it from whatever transcripts you point it at.
