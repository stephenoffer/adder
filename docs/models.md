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
carries a prior that is visible and adjustable.

That prior was the weakest thing in this module, so it gets two defences.

**It is fitted where it can be.** The outcome log records how often a tier
really escalated. The arena says how often that tier's model loses a comparison
to the escalation target. The ratio of those two *is* the constant:

```
unusable_given_loss = measured_escalation_rate / modelled_preference_loss
```

Wherever there is enough history the number stops being a prior and becomes a
fit with a sample size attached, and the row says which it used. Three cases
refuse the fit rather than produce a confident quotient: a tier with nothing to
escalate to (a model compared with itself gives a preference loss of 0.5, not
0, so this has to be rejected by identity and not by a threshold), an unrated
model, and a gap that sits inside the arena's own error bars.

**Where it cannot be fitted, its influence is measured instead.** `adder pick
--combos --sensitivity` sweeps the constant across the range it is plausibly
wrong over and reports whether the winning plan changes:

```
  stable: single (deepseek-v4-flash) wins across the whole plausible range of
  unusable_given_loss over [0.15, 0.60]
```

Most of the time the cost gaps between plans are wider than any plausible value
of the constant can move, and saying so is worth more than another decimal
place. When they are not, the output says `UNSTABLE`, names the value where the
winner flips, and tells you to prefer the plan that wins at the pessimistic
end — because a recommendation that turns on an unmeasured number is a coin
flip with a dollar sign on it.

Arena Elo measures human preference on chat and web-dev prompts. It does not
measure multi-file agentic tool use, which is what these sessions actually do.
Anything derived from it is labelled MODELLED for that reason. The `webdev`
board is preferred over `text` because a router that ranks on prose picks a
prose model.

### 3. Reading a rating as if it had no error bar

The arena publishes a 95% interval with every rating, and the first version of
this code threw it away. At the top of the webdev board the half-width is about
10 points, so the 17-point gap between the first and second model is two
overlapping intervals — a difference the source itself does not claim. Deriving
a confident 52% preference loss from it is inventing precision.

Comparisons are now conservative: the candidate is taken at the bottom of its
interval and the reference at the top of its own, so the estimate never claims
a substitute is closer to the reference than the evidence supports, and a
ranking says outright when the arena cannot separate two models. The correction
is worth about three points of `p_loss` on current data — small, and worth
stating at its real size rather than dramatising.

One related disclosure: the arena ranks reasoning efforts as separate
contestants (`claude-opus-5-max` and `claude-opus-5-high` are different rows)
while the price table has one price per model. The catalog keeps the best
rating and records which variant earned it, and any row quoting a max-effort
rating against default-effort pricing says so.

### 4. Quoting a placement that does not exist

Two gates catch this:

- **Context.** A model whose window is smaller than the current session cannot
  run inline at any price. It is priced as a subagent or not at all.
- **Harness.** Under Claude Code the main conversation is a Claude model by
  construction. A GPT or open-weight model can be a subagent, an MCP tool, or an
  external call — it cannot *be* the session. `adder pick --harness any` relaxes
  this for harnesses that route natively.

There is a quieter one. Anthropic charges 0.10x input to read a cached
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

The panel row reports **no** quality number. The obvious formula lifts the
rating by the best-of-N win rate, which requires the N runs to fail
independently; runs of one model on one prompt do not, and nobody has published
the correlation. Any number there would be a constant picked to make the row
look reasonable, so the column is empty and the cost — which is exact — still
gets priced.

A cascade's quality is not the strong model's rating. It is the strong model's
rating discounted by the failures detection misses, because an undetected
failure is precisely the case where the cascade did not work:

```
expected_cost = cost_cheap + p_fail × detection × cost_strong
quality       = elo_strong - p_fail × (1 - detection) × (elo_strong - elo_cheap)
```

Lower detection makes a cascade look cheaper *because it is worse*. Both
numbers are printed so that trade is visible rather than hidden in an average.

## Where a substitution is actually safe

`adder policy` will name another vendor's model, but only under one placement.

The standing objection to "just use a cheaper model" is the prompt cache: it is
model-scoped, so moving a warm session rebuilds the whole prefix, and on this
machine's history per-turn model downgrades were worth $21 out of $4,818. That
objection is about the *session*. It does not apply to a subagent, which starts
cold — no prefix to invalidate, a summary that costs the same to carry no
matter who produced it, and a failure contained to one run.

So delegation is the one placement where the vendor is genuinely free, and it
is the only one `policy.substitutes()` says anything about. Even there it
prices the substitute as a cascade rather than a swap:

```
expected = subagent_run + p_fail × cost_of_redoing_it_on_the_claude_tier
```

and it holds the substitute to the tier's quality tolerance — 120 Elo points at
T0, 40 at T2, because a lookup can afford a weaker model and a multi-file
refactor cannot.

Both sides of that comparison are the **subagent leg only** — what the model
charges to do the work. The summary it returns is admitted and carried at the
session model's rate no matter who produced it, so including that term would
apply the escalation multiplier to a cost neither candidate controls, and on a
long session it is large enough to swamp the difference the choice is actually
about.

`p_fail` itself is two numbers composed, not one. The outcome log measures how
often *this tier* escalates on this project; the arena measures how much weaker
the *substitute* is than the model that tier names. Neither answers the question
alone, so `select.blend_p_fail` treats them as independent failure modes —
`measured + (1 - measured) x elo_gap` — and each row says which basis it used.

Run it on this repo's own measurements and the answer is usually no:

```
  - cheapest cross-vendor subagent (qwen3.8) saves $0.136 against $0.160 of routing overhead;
    the placement was the lever, not the vendor
```

A 14-cent saving on a subagent that a routing turn cost 16 cents to choose is
not a saving. The substitution only clears the bar on genuinely large reads,
where the subagent's own run is a real number rather than a rounding error:

```
  Cheaper subagents that clear this tier's quality bar (a subagent starts cold, so
  there is no model-scoped cache to rebuild; priced including escalation):
    qwen3.8                      $  1.181 vs $2.020   saves $0.839  elo 1,669, p_fail 19% (elo)
    kimi-k3                      $  1.583 vs $2.020   saves $0.437  elo 1,674, p_fail 18% (elo)  [open weights]
```

That is the same shape as every other finding here. The money is in what enters
the context and how long it stays, not in the per-token price of whoever
answers.

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
