# One pool, six substitutes

| Lever | Worth | Confidence |
|---|---|---|
| Split sessions >300 turns | $2,535 | modelled |
| Drop effort high → medium | $1,173 | modelled |
| Delegate 25% of turns to subagents | $1,158 | modelled |
| Cut tool output admitted to context by 40% | $908 | attributed |
| Write 30% less (leverage 4.7x) | $880 | attributed |
| Compact sessions that ran full and never did | $719 | modelled |
| *(separate)* per-turn model downgrade | $26 | modelled |
| *(separate)* Explore/subagents on Haiku | $22 | **measured** |
| *(separate)* delete duplicated resident memory | $0 | attributed |
| *(separate)* recoverable cache rebuilds | $0 | **measured** |

Summing the first six double-counts — they attack the same pool. Composed
multiplicatively on the residual: **~$4,436, or 69% of measured spend**, over
105 sessions and $6,394 of measured spend.

Compaction is in the pool, not beside it, because it is a substitute for
splitting: a session that was split never reaches the ceiling, so it has no
compaction to miss. Memory is *not* in the pool, for the opposite reason — it
is resident floor rather than accumulated context, which is the term
`debt.decompose_read_cost` calls the irreducible baseline. Part of that
baseline is a file on disk, so it was never entirely irreducible; see
[context.md](context.md).

Note what moved after deduplication (see [measurement.md](measurement.md)).
Terseness fell from the second-largest lever to near the bottom of the list,
purely because its reachable share halved. **Effort** is new and ranks third: it is the
only output-side lever that does *not* invalidate the prompt cache, so unlike a
model downgrade it costs nothing to apply mid-session.

Run `adder savings` to compute this table against your own history.

## The levers that are not trades

Every row above is a substitute for the others and every one of them costs
something: a delegated read risks a redo, a split session pays a handoff, terser
output is output somebody wanted. Three findings are not like that, which is why
they are not in the table — nothing is given up to take them.

| | measured here | mechanism |
|---|---|---|
| re-reading an unchanged file | 19.2% of unbounded text reads | `adder guard` |
| reading back a file this session wrote | — | `adder guard` |
| one command shape repeated | 47% of Bash result tokens sit in shapes that cross 20K cumulatively | `adder guard` |

The first two admit tokens that are already in the context, so the information
gained is exactly zero. The third is the one no per-call rule can see: the
largest single channel on this machine is `sed -n 'A,Bp'`, a bounded read the
guard is right to wave through 245 times and wrong to wave through the 246th.

`adder guard --replay` prices all three against your own transcripts. On the
author's, replaying 29,464 calls: 236 findings, worth $85 against $2.58 of
injected advice. That is an upper bound, and [guard.md](guard.md) says why.

## The sixth lever, and why it is not in that table

The table above holds the session's model fixed, because every lever in it was
derived from a question about a turn. Ask the question about a *session* instead
and a much larger one appears: what the same work would have cost had it started
on Sonnet.

It is missing from the table for a structural reason, not an oversight. The
pooled levers are substitutes that drain one pool, and this one is not — it changes the
*price* of whatever is left in the pool after the others, so it multiplies
rather than competes. Composing it into that table would be wrong twice over.
`adder plan` handles it instead, as the last row of a cumulative regime.

Measured on the same transcripts, and the reason the distinction matters:

```
  + 30% terser, 40% less tool output        $       735         6.8x
  + start sessions on claude-sonnet-5       $       477        10.5x
```

One row, and it nearly halves what four levers had left. It is also the least
certain number in this repo: it is a rate substitution, so it says what the same
tokens would have cost and not that the cheaper model would have produced them.
`adder plan --session-rework` is the knob for that doubt, and the default of 20%
is an assumption, not a finding.

## Cache efficiency: a lever that turned out not to be available

Rebuilding a cached prefix costs 1.25x (5m TTL) or 2.00x (1h) versus 0.10x to
read it — a 12.5x swing on the entire context. Measured here:

```
hit rate      99.1%   of cacheable input tokens served from cache
100 large rebuilds cost $317 over what a cache read would have
  idle expiry (beyond any TTL)   67 turns   $296   recoverable: no
  growth                         28 turns    $20   recoverable: no
  post-compaction                 4 turns     $1   recoverable: no
Recoverable: $0
```

97% of writes already use the 1h TTL, so "switch to 1h" is not available. The
$296 comes from gaps **longer than an hour** — which no TTL setting covers. The
tool says so rather than claiming a saving: that is a session-boundary problem,
and it reinforces the splitting lever instead.

`adder cache` reports this for your transcripts.

## The cache lever that *is* available: what a restart costs

Hit rate was the wrong place to look. The cache is already being hit 99% of the
time; what it was not being used for is the thing it is uniquely good at, which
is **holding a prefix across sessions so that throwing a context away is cheap**.

Both models of a restart in this repo were wrong, in opposite directions.
`adder plan` charged nothing for one, so splitting was a free lever and the
optimiser took it to the end of its range for nothing. `carry.optimal_split`
charged a full prefix rebuild, so the closed form kept answering "a few hundred
turns". The transcripts settle it, because every session records what its own
opening turn was billed:

```
opening context       27,953 tok
  cache read          20,622 tok     74%  @0.10x -- the shared floor, already resident
  written              7,268 tok     26%  @2.00x -- the part that is this session's

One restart on claude-opus-5, carrying a 2,000-token handoff:
  measured (prefix warm)   $  0.1033
  assumed  (full rebuild)  $  0.2995   2.9x more than it costs
```

A session opening is not a rebuild. The expensive part of the floor — system
prompt, tool schemas, `CLAUDE.md` — is byte-identical across sessions, so it is
still resident and is served at 0.10x. Only the session's own tail is written.

Because the optimum goes as `sqrt(W)`, pricing the restart correctly moves the
cadence from 33 turns to 19, and per-turn input cost falls **6.1x** against the
536-turn sessions this workload actually runs:

```
  on the assumed rebuild       33 turns   $ 0.0346 per turn
  on the measured opening      19 turns   $ 0.0271 per turn
  as run (536 turns)                      $ 0.1647 per turn
```

That is the single largest lever in `adder plan`, and together with the
delegation threshold it solves — which the same cache arithmetic sets, at ~300
tokens rather than the hand-picked 5,000 — it is why the ladder now reaches
10.5x where it used to reach 5.8x. Note that the table at the top of this
page still prices splitting at a 300-turn cadence, because `adder savings` prices
each lever in isolation against the read pool and has no restart term to solve.
`adder plan` is where the cadence is solved and the restart is charged. Two caveats travel with it:

- **The handoff is modelled, and it is now the softest input in the tool.** A
  restart every 19 turns only works if 2,000 tokens is enough to carry the
  thread. `adder plan --handoff` sweeps it: at 50,000 tokens the cadence
  stretches to 46 turns and the multiple falls to 5.5x. The direction survives
  the sweep; the magnitude does not.
- **Warmth is only relied on inside the TTL.** Openings here measure warm even
  after gaps of days, which no TTL explains, so that observation is excluded
  from the measurement. Restarting mid-work puts the previous turn seconds
  behind, which is the case the number is taken from.

`adder prefix` reports this for your transcripts.
