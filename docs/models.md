# Choosing across vendors

The rest of this repo answers *where* work should run. This page is about
*what* runs it — and about the three ways an answer to that question goes
wrong.

`adder pick` and `adder models` exist because the previous answer was nine model ids
typed into a Python file. That table was correct the day it was written. Two
launches later it was a list of last quarter's models, and nothing in the tool
knew.

## The catalog

`adder models refresh` builds a catalog from two public sources:

**LMArena** publishes head-to-head Elo per model, per board, with vote counts
and licences. It is the only cross-vendor quality signal that updates within
days of a launch and is not self-reported by the vendor.

**OpenRouter's model index** publishes price, context window, cache rates,
modalities, and supported parameters for ~400 models, without an API key. It is
an aggregator, so its prices are *reported*, not authoritative.

They disagree on nearly every surface detail. The arena writes
`claude-opus-4-6-thinking`; the aggregator writes `anthropic/claude-opus-4.6`.
Without normalisation the join rate between them is about a third of what it
should be, and the models that fall out are disproportionately the new ones —
exactly the ones worth routing to. `catalog.normalize_key` collapses effort
suffixes, date stamps, thinking budgets, and version separators. Measured on a
live refresh (2026-08-14 captures): joining on the raw names matches 73 models
across the two sources; normalising first matches 114, a 56% increase. The 41
recovered are the ones where the two sources spell a version differently —
`claude-haiku-4-5` against `claude-haiku-4.5` — which skews new.

Everything is layered:

```
bundled snapshot  <  ~/.claude/adder/catalog.json  <  ./.adder/catalog.json  <  prices.py
```

Later layers win **field by field**, not wholesale. A refresh that fails to
price one model does not blank the price already on disk, and a project that
pins a single rate does not erase everything else known about that model.
First-party Claude rates sit on top and are the only entries marked `verified`.

`ADDER_CATALOG=<path>` replaces the whole stack with one file. A
recommendation that depends on whatever happened to be cached on the machine
that produced it is not a result anyone can check.

## Three ways this goes wrong

### 1. Treating "unrated" as "fine"

The catalog holds hundreds of models nobody has benchmarked. They are
disproportionately cheap. Rank by price with no rating gate and the output is a
list of the cheapest unknown things on the internet, presented with the same
confidence as everything else. `adder pick` **excludes unrated models by default**;
`--include-unrated` opts back in and flags every such row.

### 2. Reading Elo as a failure rate

A model that loses half its head-to-head comparisons to the frontier does not
fail half its tasks. Most losses are "the other answer was nicer", not "this
answer was wrong". Collapsing those into one number is what makes an Elo-driven
router recommend the frontier model for everything.

So the estimate is decomposed:

```
p_loss  = 1 - P(candidate preferred over reference)     # Bradley-Terry, from public Elo
p_fail  = p_loss × UNUSABLE_GIVEN_LOSS                  # one named prior, currently 0.35
```

Both are printed. `p_loss` is derived from measured public votes; `p_fail`
carries a prior that is visible and adjustable, and `outcomes.p_fail` replaces
the whole thing with measured retry history from your own sessions as soon as
there is any.

Arena Elo measures human preference on chat and web-dev prompts. It does not
measure multi-file agentic tool use, which is what these sessions actually do.
Anything derived from it is labelled MODELLED for that reason. The `webdev`
board is preferred over `text` because a router that ranks on prose picks a
prose model.

### 3. Quoting a placement that does not exist

Two gates catch this:

- **Context.** A model whose window is smaller than the current session cannot
  run inline at any price. It is priced as a subagent or not at all.
- **Harness.** Under Claude Code the main conversation is a Claude model by
  construction. A GPT or open-weight model can be a subagent, an MCP tool, or an
  external call — it cannot *be* the session. `adder pick --harness any` relaxes
  this for harnesses that route natively.

There is a third, quieter one. Anthropic charges 0.10x input to read a cached
prefix and 1.25x to write it. Other providers do not, and some publish nothing.
Since the dominant term in a long session is `cache_read × remaining_turns`, a
multiplier borrowed from Anthropic and applied to another vendor is not a
rounding error — it is the whole answer. Where the catalog has absolute cache
rates it uses them; where it does not, the row says so.

## Costing, not pricing

A candidate is not priced per request. It is priced through the session it
would run in:

```
inline    = switch + admit + carry + generate + output_carry
carry     = read_tokens × cache_read × remaining_turns
switch    = session_context × cache_write        # the cache is model-scoped
delegated = subagent_run + summary_admit + summary_carry
```

The `carry` term is why a model 5x cheaper per token that pulls 3x more into
the main context is the expensive choice by turn forty, and `switch` is why
moving a warm 300K conversation to a cheaper vendor usually loses money before
it saves any.

## Combinations

The cheapest way to clear a quality bar is often not one model. `adder pick
--combos` prices four shapes and, for each, names the assumption that actually
decides it:

| Shape | Wins when | Loses when |
|---|---|---|
| **single** | the task needs one good pass | — |
| **cascade** | `p_fail` is low | failure is not *detected*; an undetected bad answer ships |
| **draft-review** | review is much cheaper than generation | the reviewer needs the context the drafter explored |
| **panel** | the answer is checkable | N runs fail together and agreement looks like consensus |

A cascade's quality is not the strong model's rating. It is the strong model's
rating discounted by the failures detection misses, because an undetected
failure is precisely the case where the cascade did not work:

```
expected_cost = cost_cheap + p_fail × detection × cost_strong
quality       = elo_strong - p_fail × (1 - detection) × (elo_strong - elo_cheap)
```

Lower detection makes a cascade look cheaper *because it is worse*. Both
numbers are printed so that trade is visible rather than hidden in an average.

## Keeping the ladder honest

`adder models ladder` re-derives each rung of the T0/T1/T2 ladder from the catalog
and diffs it against the constants in `classify.py`:

```
rung  hardcoded              catalog says              elo  purpose
T0    claude-haiku-4-5        claude-haiku-4-5       1,326  lookups, searches, read-only triage
T1    claude-sonnet-5         claude-sonnet-5        1,541  scoped edits, mechanical refactors, tests
T2    claude-opus-5           claude-opus-5          1,691  multi-file, ambiguous, long-horizon
```

Within a price band the rung holds the *strongest* model, not the cheapest —
the band already bought the saving.

Drift is reported, never applied. A catalog scraped from two public sources is
not allowed to silently repoint where your work runs; a rung is worth changing
only if the new model also holds the context and the change survives
`adder savings` on your own history.

## The network rule

`adder models refresh` is the only command in this tool that opens a socket, it
only runs when typed, and `ADDER_OFFLINE=1` makes even that refuse. Every
report, gate, and test stays pure computation over local files — CI parses the
package and fails the build if any module except `sources.py` imports a
networking library.

A refresh that loses one source keeps the other and records the failure in the
catalog's provenance. A refresh that loses both leaves the existing catalog
untouched.

To keep it current without thinking about it:

```bash
adder models refresh --if-stale --max-age 14
```

That reads the local catalog's age and returns before opening a socket if it is
still current, so it is safe on a timer or in a session hook.
