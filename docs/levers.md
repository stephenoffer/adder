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

Run `adder savings` to compute this table against your own history:

```
  Measured spend $4,818 across 53 sessions
  Root cause: $3,541 of it is prior context being re-read

  SUBSTITUTES - all attack the same pool; they do not add
  $    1,941   40.3%  [MODELLED  ] Split sessions longer than 300 turns
  $      873   18.1%  [MODELLED  ] Delegate 25% of turns to subagents
  $      865   17.9%  [MODELLED  ] Drop effort high -> medium
  $      703   14.6%  [ATTRIBUTED] Cut tool output admitted to context by 40%
  $      648   13.5%  [ATTRIBUTED] Write 30% less (leverage 4.7x downstream)

  COMBINED (substitutes compose multiplicatively on the residual):
    TOTAL             $    3,253   (68% of measured spend)
```

Read the second line first. **$3,541 of a $4,818 bill, 73% of it, was not new
work at all.** It was context that had already been paid for once, being
re-read. Everything below that line is a different way of attacking the same
$3,541, which is why five levers worth 104% on paper come to 68% in reality.

The ranking is probably not what you expected: the top lever is not writing
style or model choice, it is *ending sessions sooner*, because session length is
the multiplier on everything else.

### How much to trust each row

| Label | Means | Trust |
|---|---|---|
| `MEASURED` | counted directly from your transcripts | high |
| `ATTRIBUTED` | a share of a measured pool, split by a stated rule | medium |
| `MODELLED` | derived from assumptions, which are printed next to the number | check the assumptions |

The `MODELLED` rows are the weak link, and the tool says so instead of quietly
rounding them up.

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

## The whole workload under one regime

`adder savings` prices each lever on its own. That answers "which one is
biggest", not "what would my bill be". `adder plan` answers the second: it
replays every recorded turn under a **regime**, a concrete operating
configuration you could actually follow, and prices both sides of it.

```bash
adder plan --target 10
```

```
  Measured spend            $     5,025   20,808 turns, 84 sessions
  Replay of the same turns  $     5,025   residual -0.0% -- everything below is relative to this
  Restart cadence, solved rather than assumed: 19 turns: k* = sqrt(2W/(m*r*g)) at a
  $0.1033 restart [measured], 961 tok/turn of growth and a 0.115x re-read multiplier.
  A restart is charged what an opening actually costs -- 74% of it is a cache read.
  Delegation threshold, likewise: delegate reads over ~285 tok: below that the
  400-token brief and the summary cost more than the 9 re-reads they avoid.

  regime                                          total  vs baseline  tok deleg.
  -----------------------------------------------------------------------------
  as run                                    $     5,025         1.0x           -
  delegate reads over 300 tok               $     1,552         3.2x         99%
  + right-size the subagent                 $     1,039         4.8x         99%
  + split sessions at 19 turns              $       761         6.6x         99%
  + effort high -> medium                   $       746         6.7x         99%
  + 30% terser, 40% less tool output        $       735         6.8x         99%
  + start sessions on claude-sonnet-5       $       477        10.5x         99%

  Target 10x means getting $5,025 down to $503.
  The regime above reaches 10.5x. Target met, on these assumptions;
  run `adder quality` before and after, because none of this is free.
```

**Both thresholds are solved, not chosen.** `19 turns` used to be a round `300`,
and `300 tokens` used to be a round `5,000`. Both are set by the prompt cache,
and both were being guessed. The arithmetic is in the two sections above.

The delegation threshold is the less intuitive of the two. A shorter restart
cycle leaves fewer re-reads to avoid, which should *raise* the threshold, and it
does, but only to ~300 tokens. Admitting a token to an Opus context costs 2.00x
its input rate as a cache write, while reading it once on Haiku costs 1.00x of a
rate five times lower. Delegation is not only a carry play, and `5,000` was
leaving most of it unused.

Three things make this different from the savings table.

**It reproduces your bill before it quotes a discount.** The second line is the
whole guarantee: replay the transcripts with no regime applied and the total has
to come back as the number you actually paid. It does, to −0.0%. Every multiple
below it is a ratio against that. `adder validate` re-checks it, because two
ordering bugs in the replay were caught by exactly this line and nothing else
would have caught them.

**Both sides are on the books.** A delegated read still has to be read by
somebody, that somebody still writes a summary, and some fraction of those runs
come back wrong and get redone on Opus. All three are charged. The saving is
smaller than the version that only counts what left your context, and it is the
one you would actually get.

**Delegability is measured, not assumed.** Every earlier estimate here used
"assume 25% of turns are delegable", which is a guess with a percent sign on it
and is not a rule anyone can follow. The regime triggers on something the
transcript records exactly, which is how many tokens a step would pull into
context. So "delegate anything over 5,000 tokens" is checkable, followable, and
the 23% that matches is a measurement.

When no configuration on the grid meets the target, the report says so and names
the floor, instead of searching until it finds a number that flatters the
question.
