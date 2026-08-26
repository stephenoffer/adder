# What an output token really costs

An output token is billed once at generation, then again as cached input on
every remaining turn:

```
true_cost(1 token, R remaining turns) = rate_out + cache_read_rate * R
```

On Opus 5 that is `$25/MTok + $0.50/MTok per turn`, a multiple of `1 + R/50`:

| remaining turns | 0 | 50 | 200 | 340 | 759 | 1,854 |
|---|---|---|---|---|---|---|
| **cost vs sticker** | 1.0x | 2.0x | 5.0x | **7.8x** | 16.2x | 38.1x |

Past **50 remaining turns**, re-reading an output token costs more than
generating it did. Every cost tool reports only the generation cost.

`adder debt` computes this against your own transcripts.

**`cache_read_rate` is the provider's, not a constant.** It is 0.10x input on
Anthropic, which is where the `R/50` above comes from; 0.20x on the OpenAI 5.x
family; and, on a hosted endpoint with no prompt cache at all, 1.00x, because
re-reading the prefix genuinely costs full input rate there. That last case is
not a rounding difference. It makes the debt multiple roughly ten times larger,
so verbosity is ten times more expensive than an Anthropic-shaped estimate
would say, and the break-even arrives almost immediately instead of at turn 50.
`docs/providers.md` has the table; `adder debt --model <id>` prices whichever
model you name.

## Why this is not a model router

Existing routers (RouteLLM, NotDiamond, Martian, OpenRouter auto, vLLM Semantic
Router) pick a model per request and price it `in x rate_in + out x rate_out`.
That is correct for stateless APIs and wrong for agent sessions.

Worse, the obvious move actively loses money. Opus 5 reads cached context at
$0.50/MTok; Haiku 4.5 reads it fresh at $1.00/MTok, because the cache is
model-scoped,
so downgrading a warm conversation makes input **2x more expensive**. Break-even
is `output > context / 40`: at the measured median context (544K) that needs
13.6K output tokens per turn. The measured average is **783**.

And at 544K the switch is not merely unprofitable, it is **impossible**: Haiku
4.5 holds 200K. Every gate checks the context window before quoting a saving,
because a "saving" that names a model the context does not fit in is a 400
error, not a discount.

Per-turn model routing is worth ~**$21** of the $4,456 measured here. It ships,
correctly gated, as the smallest lever.

## The distinction that took too long to draw

That result is about a **switch**, and for years it was read as "the model does
not matter here". That does not follow, and the difference is worth about half
the bill.

A switch is expensive because it invalidates a prefix you already paid to build.
A session that *starts* on the cheaper model never built one. There is no
rebuild, only a cheaper rate applied from the first turn to the context-carry
term that is 76% of spend. Re-priced across the same transcripts:

| | worth |
|---|---|
| switching a warm conversation to Sonnet, per turn | **0.5%** of spend |
| starting the session on Sonnet instead | **60%** of spend, before rework |

Both numbers come from the same data and `adder validate` re-runs both, as
`model routing is a minor lever` and `starting cheap beats switching cheap`. The
first was never wrong. It was just answering a question nobody had asked, because
the router was built to decide turns and the expensive decision is made once, at
turn zero, by whoever picked the model.

The 60% is a rate substitution and nothing more. It says what the same tokens
would have cost, not that Sonnet would have done the work. That part is
capability, no transcript that only ever ran on one model can settle it, and
`adder plan` charges it as an explicit `--session-rework` fraction rather than
pretending it is free. At the default 20% the lever is still the largest one
available.

## The three terms in `1 + R/50` that are not constants

That headline formula assumes a token is re-read at 0.10x on every one of `R`
remaining turns. It is a good approximation and each of its three parts is
measurable, so none of them has to stay an assumption. `adder carry` measures
all three against local transcripts.

**The published discount is a floor, not a rate.** A re-read costs 0.10x only when the prefix is
warm on the turn that reads it. Turns miss: the 5m TTL expires while you read a
diff, a tool result lands past the cache-breakpoint lookback, a parallel fan-out
races the first write. A miss rewrites at 1.25x rather than reading at
0.10x. The realized multiplier is recoverable from the transcripts directly,
because every turn records how its input actually split:

```
realized_mult = sum(uncached + read_mult*cache_read + write_mult*cache_write)
                / sum(context)          # multipliers per turn, from its provider
```

token-weighted, first turns excluded (a session's opening turn writes its whole
prefix by construction, which measures the cost of *starting* a session rather
than continuing one). Measured on the transcripts behind this repo it is
**0.115x, or 1.15x the assumption**. The carry term was already ~76% of spend and
it was being under-priced.

**`R` is not how many turns the token is present for.** Compaction evicts it. A
token admitted now is re-read every turn until the next compaction, survives it
with roughly the share the compaction kept, and so on, so the honest count is

```
E[reads] = sum over epochs j of  survival^j * (turns in epoch j within R)
```

Detecting compaction correctly matters more than the correction does. 122 turns
out of 20,524 here show a context drop; only **7 are compactions**. The rest are
small dips from branch resumption, clustered between 0.65x and 0.98x of the
previous context. Counting all 122 fits a 4-turn compaction period, which would
price a token admitted now at 16 re-reads instead of 348, a 20x under-statement
that switches delegation off across the board. A compaction requires the context
to have been at 60% of the model's ceiling *and* to have lost half of itself;
the 7 real events sit at 999.5K–999.9K dropping to 4–6%.

**`R` is a mean, not a median.** Cost is linear in remaining turns, so its
expectation is `c * E[R]`, and `E[R]` is the conditional **mean**. Session length
is heavy-tailed, so the mean sits above the median: 351 against 305 at turn 0
here, a factor of 1.15. `horizon.remaining()` returns the median because that is
the right number to show a person; `horizon.mean_remaining()` returns the mean
because that is the one that prices carry. Using the median under-prices
admission in exactly the long sessions that hold the spend.

The two corrections push in opposite directions, which is why neither is safe to
apply alone. Together, on this workload, they move the cost of admitting 10,000
tokens at a 348-turn horizon from $1.80 to $2.07.

## Two things that fall out once the carry number is honest

**How long to run a session.** Average per-turn input cost on a `k`-turn cycle,
in a session growing at `g` tokens per turn from a floor `F` that a restart
cannot avoid, with `W` the one-off write a restart pays:

```
A(k) = m*r*F + m*r*g*(k+1)/2 + W/k        =>    k* = sqrt(2W / (m*r*g))
```

The square root is the result. Being wrong about the handoff cost by 4x, and
nothing in a transcript records how much context a person needs to resume, moves
the optimum by 2x, not by 4x. That is what makes the number usable despite its
softest input, and it is why "compact constantly" is nearly always worse advice
than it sounds. `adder carry` prints it as a sweep over handoff size rather than
as a single figure.

**When to delegate.** Both sides of the placement decision are affine in the read
size, so the break-even is one division rather than a search:

```
inline(x) = x * r_m * (w + m*E)
deleg(x)  = (b + x)*r_s + p*x*r_s_out + p*x*r_m*(w + m*E) + p_redo*(x*r_m*(w + m*E) + c)
```

This is the only advice in the repo that is free to act on. Every other
recommendation costs a routing turn and has to earn it back; a threshold is a
rule a hook applies with no turn behind it, so it cannot cost more than not
asking.
