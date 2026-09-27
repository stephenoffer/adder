# How it works

Ten minutes, start to finish: why an agent session costs more than the invoice
suggests, what adder refuses in order to stop it, and how it picks a model for
the work that is left. Every section ends with the page that carries the
evidence.

## Context is a subscription, not a purchase

When the agent writes 1,000 tokens, you are billed for writing them. Then they
join the context, so you are billed again to re-read them on turn 2. And turn 3.
Write something with 340 turns left and you pay for it 341 times:

```
true_cost(1 token, R remaining turns) = rate_out + cache_read_rate * R
```

| turns remaining | 0 | 50 | 200 | 340 | 759 |
|---|---|---|---|---|---|
| what a token really cost | 1.0x | 2.0x | 5.0x | **7.8x** | 16.2x |

Past roughly 50 remaining turns, re-reading a token costs more than writing it
did. Your usage dashboard shows you the 1.0x column.

Priced against a live session, that looks like this:

```
$ adder live
  This session: 11 turns · 47,401 tokens in context · $0.89 spent ($0.081/turn)
  Sessions that reach turn 11 typically run ~340 more turns → ~$28.27 total

  Cost of reading a file into THIS context from here:
       file size      inline   delegated   verdict
        20,000 tok  $    3.525  $    0.383   delegate
        50,000 tok  $    8.812  $    0.957   delegate
       150,000 tok  $   26.438  $    2.869   delegate
```

$0.89 spent, $28.27 owed. Reading a 50,000-token file into this session costs
$8.81 by the time it ends. Handing the same file to a subagent that reads it and
returns a summary costs $0.96. Same information, 9x apart, and nothing in your
usage dashboard tells the two apart.

Two things follow, and they are the whole tool. The expensive decision is rarely
which model you used; it is what you let into the context and how early. And a
600-turn session is not one long session — it is the same context, re-read 600
times.

→ [cost-model.md](cost-model.md) derives the formula, and shows where each of
its three constants stops being constant.

## What the guard does while you work

Most cost tools are dashboards: they tell you what you already spent. `adder
auto on` puts adder in front of the decision instead, on hooks your harness
already fires. Nothing runs on a timer, nothing holds a socket, nothing calls a
model.

It refuses a read of something already in your context. You read `schema.py` on
turn 12; on turn 80 the agent reads it again. Nothing changed on disk, so the
second read admits every one of those tokens a second time and buys no
information at all. That call now does not happen.

It refuses a large read that has a cheaper equal, and names the equal:

```
[adder] This admits ~15,000 tokens at ~$1.19 of carry, against ~$0.13
        delegated. Read 300 lines of it, or hand it to a subagent.
```

The agent takes the cheaper path and keeps working.

It bounds what a subagent hands back. A subagent's own reads are already outside
your window; the only part that reaches your context is the return, so that is
what gets priced — against a brief, not against reading it yourself.

Three properties make refusing safe to ship:

- It never refuses the same thing twice. Ask again and it goes through, so a
  wrong refusal costs one turn.
- It forgets on compaction. "Already in your context" stops being true the
  moment the context is rebuilt.
- It prices its own sentences before saying them.

→ [guard.md](guard.md) is the whole decision, including what may be refused and
what may only be rewritten. [guard-internals.md](guard-internals.md) is the
measurement that forced the current design.

## How the router picks a model

Every delegation is a model choice, and the default answer is "whatever the main
session is running" — the most expensive model you have, picked by nobody. The
guard already sees every `Task`, so it asks at the point the decision is live,
in one clause folded into the message it was already sending:

```
[adder] Run it on route-t0 (claude-haiku-4-5): ~$0.05 cheaper, redo risk included.
```

Behind that clause, in order:

1. **Classify, but only at the extremes.** "Fix the login bug" is four words and
   unbounded work; text cannot predict how deep a coding task goes. So the
   classifier fires on high-precision signals and abstains otherwise. Abstaining
   routes up, because a misrouted hard task costs a full retry and a misrouted
   easy one costs pennies — unless the failure is silent. That argument holds
   for coding work, where a wrong answer breaks a test. It does not hold for
   recall: a weak model asked for every hardcoded credential in a tree hands
   back three of the seven, confidently, and nothing retries. So a stated
   quantifier over a plural target ("every", "all", "any") abstains however easy
   the sentence looks.
2. **Feasibility before price.** A tier whose context window cannot hold the
   task is not an option at any price.
3. **Price every rung including the cost of being wrong:**
   `run(tier) + p_fail × (redo on the strong model + the turn that catches it)`.
   `p_fail` comes from your own outcome log wherever you have history, so a tier
   that keeps failing on your project stops being recommended for it.
4. **Permissions are asymmetric.** Going up needs no evidence: the worst case is
   that you paid for the model you'd have chosen anyway. Going down needs the
   classifier to have abstained, enough recent history at that rung to be
   informative, and a measured failure rate under that rung's break-even.
   Cheapness alone never buys a downgrade.
5. **Compare against what the call would otherwise have run on** — the session
   model, not the top rung. Re-pricing a decision somebody already made quotes a
   saving nobody was going to collect.
6. **Say it only if it pays, and only once per session.** The sentence lands in
   your context and is re-read for the rest of the session, so it is priced like
   any other tool result.

It never refuses a delegation. Refusing a `Task` would refuse the largest lever
in the tool on the strength of a classifier that is deliberately
abstention-happy.

→ [tiers.md](tiers.md) is the ladder and the classifier;
[routing.md](routing.md) is how the router is scored.

### On a domain repository the router mostly says nothing

This is worth knowing before you measure it. The signals above are generic
English verbs: `refactor`, `investigate`, `why is`, `across the codebase`. A
domain workload speaks in nouns the classifier has never seen — *this Ray Data
pipeline is spilling to disk*, *the NCCL collective hangs during allreduce*,
*recommend an instance type for Llama-3.1-70B*. Twelve phrasings of that shape
produce twelve abstentions, and an abstention routes up, to where the session
already was. The routing turn gets charged to conclude "keep doing what you were
doing".

So the multiple any of these numbers reports is a property of your corpus's task
vocabulary as much as its session lengths. An `adder bench` run that comes back
all-abstention is telling you the classifier has nothing to say about your work,
not that your work is hard.

### Across vendors

`adder models refresh` builds a catalog of ~500 models from LMArena Elo and
OpenRouter's index, carrying price, context, cache rates, rating and vote counts
for each. `adder pick "<task>"` returns the cheapest that clears the quality
bar, excluding unrated models by default: rank by price with no rating gate and
the answer is the cheapest unknown thing on the internet, stated with the same
confidence as everything else.

Those cross-vendor candidates appear in `adder policy` and never in the injected
clause. A Claude Code `Task` cannot be dispatched to Qwen, and naming a model at
the moment nobody can act on it is how a router stops being read.

→ [models.md](models.md) · [providers.md](providers.md)

## What running it costs

A guard that talks costs money: every sentence it injects is carried for the
rest of the session. So adder prices its own advice and stays quiet unless the
saving covers it. Replayed over 34,592 recorded tool calls:

| | |
|---|---|
| calls it acted on | 2,554 (7.4%) |
| saving from calls that did not happen | **$517** |
| saving argued for in sentences | $23 |
| what adder's own messages cost | $27 |
| return per $1 spent | **20x** |

96% of that needs no assumption about whether anyone listened, because the calls
did not happen. Before enforcement existed, 100% of the number was a sentence
multiplied by a guess. That is the real change: not that the figure got bigger,
but that it stopped being a hope.

→ [overhead.md](overhead.md)

## Where the money actually is

`adder savings` on the same history: $5,840 of a $7,888 bill, or 74%, was not
new work. It was context already paid for once, being re-read.

| lever | worth | who pulls it |
|---|---|---|
| Split sessions longer than 300 turns | $3,127 | you |
| Drop effort high → medium | $1,477 | you |
| Delegate 25% of turns to subagents | $1,441 | adder |
| Compact sessions that ran full and never did | $1,121 | you |
| Cut tool output admitted to context by 40% | $1,112 | adder |

These are substitutes, not addends — they all attack the same pool. The largest
is session length, and no hook can pull it: nothing here can restart a session
for you. That gap is exactly the difference between the 3.1x adder delivers and
the 6.4x available, and the tool prints both rather than the flattering one. At
the pessimistic corner of its three softest assumptions, 6.4x becomes 3.3x.

→ [levers.md](levers.md)

### Three results that contradicted the obvious advice

Switching to a cheaper model mid-session loses money. The cache is model-scoped,
so a warm conversation throws the discount away; the switch is worth 0.5%, while
*starting* cheap is worth 60%.

Verbosity is not the main lever. `Bash` results alone admit more context than
every other tool combined.

The cache was already fine, at a 99.2% hit rate, so the tool reports $0
recoverable there instead of inventing a saving it cannot deliver.

## Cheaper is not the same as better

A tool that only counts dollars will happily talk you into worse work, and one
that can refuse your agent's tool calls could do real damage doing it. So `adder
quality` reads your transcripts for signs the agent is struggling: tool error
rate, corrections, interruptions, turns per prompt, rework. `adder verify`
refuses to certify a saving if any of those got worse.

A change that cut your bill and doubled your rework isn't a saving; it's a cost
you moved somewhere the invoice can't see. Start with plain `adder auto on` if
you'd rather it only refuse the reads that provably admit nothing new.

→ [quality.md](quality.md)

## How much to trust any of it

Every dollar figure on this page comes from one machine's transcripts: 118
sessions, 33,192 turns, 34,592 tool calls, dominated by one workload. Your
absolute numbers will differ. The shares are what drive the advice, and even
those are worth re-checking on your own history, which is the entire point of
the tool.

Run `adder savings` before believing any number here, and `adder bench` to price
the ladder above against your own turns. Every figure is re-run by `adder
validate`, not remembered, and the thresholds were swept rather than chosen:
`adder auto on --full --tune` re-derives them from your transcripts instead of
inheriting one machine's answer.

3,753 tests stand behind that, and two of them exist only to enforce two of the
promises above: no runtime dependencies, and no network outside `adder models
refresh`.

→ [benchmark.md](benchmark.md) · [measurement.md](measurement.md)
