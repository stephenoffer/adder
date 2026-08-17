# Is the tool cheaper than not having it?

A cost tool has to answer this about itself, and the question is not rhetorical.
Asking adder costs a routing turn, and a routing turn at 500K of context on Opus
is about **$0.26** before anything useful has happened. Advice worth $0.10 a
time, charged at $0.26 a time, is a more expensive way to work. That is the trap.

Write the bill out and there is nothing to argue about:

```
cost_with_adder = baseline - savings + overhead
```

which is below `baseline` exactly when savings cover overhead. Four things hold
that inequality up instead of assuming it.

## Recommendations are priced against their own uncertainty

Not just their midpoint. Three inputs to every placement decision are estimates
with real spread: how many turns are left, how often this tier fails, how big a
summary comes back. So adder reports the probability that a recommendation is
cheaper than not taking it, and declines when the expected saving is being
carried by a tail rather than by the typical outcome.

It also reports the corner where the advice would lose money, because *"this
loses money if the session ends within 23 turns"* is a sentence you can check
and *"worst case −$0.03"* is not.

## Delegation is priced as something that can fail

A delegated read that comes back missing what you needed costs the subagent run,
the turn that noticed, and the inline read anyway. That term used to be missing
altogether, and its absence made delegation look free of downside. It is not.

## The bar is measured, not assumed

`adder carry` reads the realized cost of carrying context off your own
transcripts instead of assuming a warm cache. It comes out at 0.115x here,
against the 0.10x the model assumed, and the routing overhead every
recommendation has to clear moves with it.

## The tool keeps its own books

`adder ledger` records the guaranteed saving of every recommendation acted on
against the overhead it cost, and measures the gap between what predictions
promised and what they delivered. If they have been delivering 60% of face
value, every future prediction is scaled by 0.6 before it meets its gate. A model
that over-promises raises its own bar until it stops.

## Re-running it

`adder validate` re-runs all of this against your data, including a 240-case
sweep asserting that **every recommendation the router emits saves more than the
turn that produced it.** The sweep found three counterexamples the day it was
written. They are fixed.

The cheapest advice adder gives is the kind that costs nothing to apply:

```bash
adder carry        # ends with a delegation threshold in tokens
```

A threshold is a rule a hook can apply with no routing turn behind it, so it
cannot cost more than not asking.
