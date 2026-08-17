# Four systems questions, asked about an agent session

`adder` began as a cost report. These four commands ask questions that serving
systems have been asking about inference workloads for years, and answer them
about the workload on your laptop instead of one in a datacentre. The answers
are different, because the workload is: one long-lived stateful conversation
rather than a stream of independent requests.

| Command | Question |
|---|---|
| `adder place` | Locality or price? Move a warm session, or keep its prefix? |
| `adder deadline` | Is the cheap, slow, unreliable path worth it before a deadline? |
| `adder sched` | Does how far a program has run predict how much is left? |
| `adder design` | Which measurement to spend the next dollar on? |

## Placement: locality has a price

A load balancer deciding where to send a follow-up request weighs two things:
the region that already holds the request's cached state, and the region with
cheaper capacity. Sending it to the cheap region means recomputing or shipping
the KV cache.

An agent session is the same trade with the numbers moved around. Its context
is cached against **one model**, and the cache is model-scoped, so moving to a
cheaper model discards it: the whole context is re-read cold and written again.

```
breakeven_turns = migration_cost / (cost_per_turn_here − cost_per_turn_there)
```

Above that many remaining turns, move. Below it, stay — however much cheaper the
other model looks per token. `adder place` sweeps the whole catalog for this and
prints the affinity you would be discarding as a number in its own right,
because that is the quantity nobody believes until they see it.

Two gates come before price, and both have bitten:

- **Feasibility.** A context window that cannot hold the session is not a cheap
  option, it is a 400.
- **Cache economics.** A provider with no prompt cache re-reads your prefix at
  full input rate, not a tenth of it. A per-token price 40% cheaper can be
  several times dearer across a long session. Providers that publish no cache
  rate are assumed to have no cache, which is the pessimistic direction.

## Deadlines: a discount you cannot collect is not a discount

Batch processing is half price — the largest single price lever available, and
one `adder` never recommended because nothing here knew what a deadline was.

The trade is not "cheaper but slower", it is **cheaper but uncertain**. The
cheap path returns work at a rate you do not control and may stall entirely.
Against a deadline that converts a discount into a risk, which needs a policy.

`adder deadline` prices four policies and names the cheapest that meets the
deadline. It deliberately does not have a favourite, because the obvious
candidate is optimal under one assumption and bad under another:

- **Greedy** (batch until the slack runs out, then sprint) wins outright when
  the guaranteed path can absorb the whole remaining queue at once — true of an
  API you can fan out against. On a 200-unit queue over 24 steps it costs
  $100.52 against the proportional policy's $130.35, and both meet every
  deadline.
- **Proportional** (keep completed work on the line `total × t / horizon`) wins
  when the guaranteed path is rate-limited, because greedy concentrates every
  expensive unit into the window with the least capacity to place them.

The proportional rule has no slack fraction, risk tolerance, or threshold to
fit — a parameter is a thing that gets set once from one workload and is then
wrong everywhere else.

One addition to that rule is load-bearing, and the tests caught its absence.
Falling behind the line is recoverable; running out of the steps in which the
guaranteed path could still finish is not. So the switch must happen *before*
that point:

```
if guaranteed_steps_needed(remaining) >= steps_left: use the guaranteed path
```

Without it the policy is a heuristic that usually finishes. With it, the
deadline is a guarantee whenever the guaranteed path alone could have met it.

Unfinished work is charged at full price in every policy's total. Otherwise the
cheapest strategy is always the one that gives up.

## Attained service: what the data would not support

Schedulers that treat an agent program as the unit prioritise by **attained
service** — programs that have consumed little go first, because they will
finish soon. That is an empirical claim about the conditional distribution of
work, and nothing here had checked it.

`adder sched` measures the mean-residual-life curve, `E[remaining | attained ≥ x]`,
and summarises it in turns per turn against a reference that needs no
calibration: **−0.5 is "every session is the same length"** (averaging over the
positions past `x` halves the 1:1 decline), and **0.0 is "length tells you
nothing"**.

This one is worth reading as a negative result, because three earlier versions
of the statistic each produced a confident answer that was an artefact:

1. **Correlating attained against remaining over pooled positions.** Inside a
   session those two sum to a constant, so they are mechanically
   anti-correlated. On a workload built to be heavy-tailed it returned −0.75 and
   would have advised the exact opposite of the truth.
2. **A rank correlation over thresholds.** Reads "bounded" for everything,
   because the ordering is decided by the truncated tail.
3. **Least squares over the full threshold range.** The deepest threshold has
   enormous leverage, so truncation decides the answer again.

And the conclusion the data will not support: **a heavy-tailed verdict.** Every
real workload is finite, so past the median length the survivors are simply
running out and the curve must fall to zero. Four synthetic workloads built with
long tails all summarised negative, for that reason rather than for lack of a
tail. So the verdict is two-way — `uniform-length` or `dispersed` — and the
module says why the third category is missing rather than printing one it
cannot defend.

What it is for: `uniform-length` means a "hand off after N turns" rule can work.
`dispersed` means it is sorting noise, and restarts should key on context size
and cost per turn, which are observed rather than inferred.

## Experiment design: the budget goes where the uncertainty is

`adder ab` is the only thing here that spends money to produce data, and it was
used the way every A/B harness is used: pick two models, run some tasks, run
more if it looks close.

That is the worst allocation of a fixed budget. Comparing a model against one it
obviously beats teaches nothing; two whose intervals overlap by 90% is where all
the uncertainty lives. `adder design` allocates by information per unit of
remaining uncertainty:

```
value(i,j) = p_ij (1 − p_ij) × overlap(i,j) / sqrt(n_ij + 1)
```

The first term is the Fisher information a comparison carries, maximised at a
coin flip. The second stops a pair that has already separated from attracting
budget. The third is diminishing returns, which is what stops the whole budget
landing in one cell. Each of the three has an obvious failure mode that the
other two cover.

It also answers the question that ends experiments: **how many comparisons
would it take to separate this pair at all?** A pair at p=0.55 needs on the
order of a thousand. At any realistic per-task cost the honest answer is that
those two models are interchangeable on quality, and the choice is price.
