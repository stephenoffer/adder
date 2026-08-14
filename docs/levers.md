# One pool, five substitutes

| Lever | Worth | Confidence |
|---|---|---|
| Split sessions >300 turns | $1,741 | modelled |
| Delegate 25% of turns to subagents | $810 | modelled |
| Drop effort high → medium | $801 | modelled |
| Cut tool output admitted to context by 40% | $655 | attributed |
| Write 30% less (leverage 4.6x) | $601 | attributed |
| *(separate)* Explore/subagents on Haiku | $22 | **measured** |
| *(separate)* per-turn model downgrade | $21 | modelled |
| *(separate)* recoverable cache rebuilds | $0 | **measured** |

Summing the first five double-counts — they attack the same pool. Composed
multiplicatively on the residual: **~$2,999, or 67% of measured spend.**

Note what moved after deduplication (see [measurement.md](measurement.md)).
Terseness fell from the second-largest lever to the smallest of the five, purely
because its reachable share halved. **Effort** is new and ranks third: it is the
only output-side lever that does *not* invalidate the prompt cache, so unlike a
model downgrade it costs nothing to apply mid-session.

Run `adder savings` to compute this table against your own history.

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
