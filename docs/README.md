# adder documentation

Three pages cover most of what people need. The rest is evidence, kept separate
so the first three stay short.

| Start with | |
|---|---|
| **[Getting started](getting-started.md)** | install, activate, first report, and what each command answers |
| **[How it works](how-it-works.md)** | the carry arithmetic, the guard, the router, in one read |
| **[Commands](commands.md)** | every command, grouped, with flags |

## Find a page by the question

| Question | Page |
|---|---|
| Why does a long session cost so much? | [cost-model.md](cost-model.md) |
| What is adder actually worth, against no adder? | [benchmark.md](benchmark.md) |
| Which habit is worth changing first? | [levers.md](levers.md) |
| What will the guard refuse, and can it be wrong? | [guard.md](guard.md) |
| Why is the guard built that way? | [guard-internals.md](guard-internals.md) |
| Which tier should run a task? | [tiers.md](tiers.md) |
| How do I know the router helped? | [routing.md](routing.md) |
| Can I use a non-Anthropic model? | [models.md](models.md), [providers.md](providers.md) |
| What is CLAUDE.md costing me every turn? | [context.md](context.md) |
| Does the advice pay for the turn it costs? | [overhead.md](overhead.md) |
| Did cutting cost make the agent worse? | [quality.md](quality.md) |
| Why should I believe any of these numbers? | [measurement.md](measurement.md) |

## Behind the numbers

[measurement.md](measurement.md) is the deduplication bug that inflated every
early figure by 1.78x, and the two places the same mistake reappeared. Read it
before trusting a number from any cost tool, this one included.

[benchmark.md](benchmark.md) documents the replay harness: how a configuration
is re-priced against recorded turns, and why the null configuration has to
reproduce the measured bill first.

[research-map.md](research-map.md) separates what is known from what is assumed
and what is still open. [systems.md](systems.md) covers the scheduling and
caching theory the session-length results borrow from.

## Working on adder

[architecture.md](architecture.md) shows how the pieces fit and what runs where.
[structure.md](structure.md) has the seven layers and the rule that imports
point down, never up. [naming.md](naming.md) explains "adder" and the vocabulary
the reports use. [releasing.md](releasing.md) is how a version gets cut.

Agents changing this repository should read [CLAUDE.md](../CLAUDE.md), which is
the binding version of those rules.
