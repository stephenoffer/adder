# The measurement bug that came first

*(Figures here come from a snapshot of one machine's history. The transcript
pool grows with every session, so a fresh `adder savings` run reports slightly
higher totals than the ones below.)*

The previous version of this analysis reported **$7,507 across 32,251 turns**.
That was wrong, and the correction reframes every number in these docs.

Claude Code writes **one JSONL record per content block**, and every record
repeats the whole message's `usage`. A turn with a thinking block and two tool
calls is three records, each reporting the same token counts. Summing lines
multi-counts most turns:

| | reported | actual |
|---|---|---|
| turns | 32,251 | **18,163** |
| list-equivalent spend | $7,507 | **$4,456** |
| median session length | 607 turns | **340 turns** |

Records are now grouped by `message.id`, keeping the record with the highest
`output_tokens` — partial records carry a running count that only the final one
completes, so keeping the *first* instead undercounts output by 2.6%.

This is the kind of error that makes a cost tool worse than no cost tool, so it
is tested directly (`tests/core/test_trace_dedup.py`).

## What the corrected data says

Measured across 171 transcript files (50 distinct sessions, 18,163 turns,
$4,456 list-equivalent):

| | |
|---|---|
| Input-side spend | 92% (cache-read alone: 78%) |
| Output-side spend | 8% |
| Cache hit rate | 99.1% of cacheable input tokens |
| Median session | 340 turns, 544K peak context |

## The claim that did not survive deduplication

The earlier analysis concluded that assistant output was **~105% of context
growth**, and therefore that verbosity was the whole story. That figure was an
artifact: the duplicate records inflated output ~1.78x while leaving context
deltas untouched (duplicates carry an identical context, so the delta between
them is zero).

Re-derived on deduplicated records, context growth splits roughly in half:

| source | share | basis | lever |
|---|---|---|---|
| assistant output | **50%** | billed | terseness, effort |
| tool results | ~26% | estimated | delegation, bounded reads |
| user messages | ~5% | estimated | — |

*(The remaining ~19% is tool-result estimation error and injected context —
read content, not written.)*

`Bash` alone accounts for ~4.1M tokens of admitted context, more than every
other tool combined. **No writing-style instruction can reach it.** That is why
the agents carry explicit output-bounding rules (`head`, `wc -l`, `grep -n -m`)
alongside the terseness rules.

## The same bug, twice more

Deduplication by `message.id` fixed the version of this that was costing 1.78x.
It is a *class* of bug, not one bug, and two more instances of it were found
later in the same data.

**Across files.** `iter_file` deduplicates within a transcript. Nothing
deduplicated *between* transcripts, and there are two routine reasons the same
turn appears in two files: a resumed session writes a new `.jsonl` that replays
earlier turns, and a sidechain file restates the parent turn it branched from.
Both carry the original `message.id`. `load_sessions` now keys on
`(session id, message id)` — the session id is part of the key because message
ids are only unique within a conversation, and dropping it would collapse two
genuinely different turns that happened to share one.

**Within a message, for tools.** The tool report attributes each `tool_result`
to the `tool_use` block that asked for it. Deduplicating those records by
message id discards every `tool_use` block after the first, because one record
per content block means a turn that called three tools is three records sharing
one message id. The results then reference ids the scan has never seen, and
land in an `unknown` bucket: **56% of measured context growth was attributed to
a tool called `?`** until the dedup key was changed to the block id.

The lesson both times is the same one the original bug taught. The transcript
format repeats itself in more than one dimension, so *every* aggregation over it
has to name the key it is unique on, and a bucket labelled "unknown" that holds
a majority of the total is a broken join rather than a finding.

## A third counting error, in the other direction

Not every assistant record is a turn. Claude Code writes one with the model id
`<synthetic>` when the *client* produced the message — "API Error: Connection
closed mid-response", an interrupted stream, a context that would not fit. Their
usage block is all zeros and nothing was billed.

Counted as turns they inflate the turn count and depress every per-turn average.
Counted as an unknown model they raise a "this report is a lower bound" warning
about spend that does not exist. They are neither, and are now reported as what
they are: a count of client-side failures, which `adder quality` treats as a
performance proxy rather than a cost one.

## Two traps in counting output

Both are handled here, and both are easy to get wrong:

- On Opus 5 the thinking field is returned *empty* while still being billed.
- A `tool_use` block's JSON arguments are output tokens that appear in no
  `text` block.

Estimating model output from text lengths undercounts it roughly sixtyfold.

## Caveats

These figures are from one machine's transcripts, dominated by one workload.
The *shares* (output vs tool output, cache hit rate, gap distribution) are what
drive the advice, and they will differ for you — which is the point of measuring
rather than assuming. Run `adder savings` against your own history before believing
any number here.
