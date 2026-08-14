# What an output token really costs

An output token is billed once at generation, then again as cached input on
every remaining turn:

```
true_cost(1 token, R remaining turns) = rate_out + rate_in * 0.10 * R
```

On Opus 5 that is `$25/MTok + $0.50/MTok per turn` — a multiple of `1 + R/50`:

| remaining turns | 0 | 50 | 200 | 340 | 759 | 1,854 |
|---|---|---|---|---|---|---|
| **cost vs sticker** | 1.0x | 2.0x | 5.0x | **7.8x** | 16.2x | 38.1x |

Past **50 remaining turns**, re-reading an output token costs more than
generating it did. Every cost tool reports only the generation cost.

`adder debt` computes this against your own transcripts.

## Why this is not a model router

Existing routers (RouteLLM, NotDiamond, Martian, OpenRouter auto, vLLM Semantic
Router) pick a model per request and price it `in x rate_in + out x rate_out`.
That is correct for stateless APIs and wrong for agent sessions.

Worse, the obvious move actively loses money. Opus 5 reads cached context at
$0.50/MTok; Haiku 4.5 reads it fresh at $1.00/MTok — the cache is model-scoped,
so downgrading a warm conversation makes input **2x more expensive**. Break-even
is `output > context / 40`: at the measured median context (544K) that needs
13.6K output tokens per turn. The measured average is **783**.

And at 544K the switch is not merely unprofitable, it is **impossible** — Haiku
4.5 holds 200K. Every gate checks the context window before quoting a saving,
because a "saving" that names a model the context does not fit in is a 400
error, not a discount.

Per-turn model routing is worth ~**$21** of the $4,456 measured here. It ships,
correctly gated, as the smallest lever.
