# Where the ideas came from, and what each one changed

This document exists so the reasoning behind the last three rounds of work is
inspectable rather than implied. Each row is a published result, the gap it
exposed in `adder`, and what was built. Where a paper's method was adopted, the
deviation is stated; where it was rejected, the reason is stated.

None of this is a claim of novelty. `adder` measures one narrow thing — the cost
of a stateful agent session on a laptop — and the value of a systems result here
is that it has already been argued over by people with better data.

## Survey → gap → build

| Published result | Gap it exposed | Built |
|---|---|---|
| Learning to route from preference data: PGR, APGR, CPT, call-performance curve | The router made recommendations and was never scored | `adder routereval` |
| Preference leaderboards as batch Bradley-Terry with bootstrap intervals, tiered to fight label sparsity | Ratings were treated as scalars; a 17-point "lead" was two overlapping intervals | `adder/pricing/bt.py`, `adder frontier` |
| **Style control**: length and markdown as covariates in the BT regression | Raw ratings reward verbosity, and verbosity is the thing this tool bills you for | `adder verbosity` |
| Active sampling to converge a ranking faster | The A/B budget was spread evenly, so the weakest link was the pair nobody sampled | `adder design` |
| Agent-first data systems: agentic speculation is high-scale, heterogeneous, redundant, steerable | Sessions were only ever read as sequences of billable turns | `adder spec` |
| Prefix/KV cache reuse: block hashing, prefix-anchored matching, radix sharing | The cache was observed, never counterfactually simulated | `adder cachesim` |
| Locality-aware cross-region serving: prefix affinity against cheaper capacity elsewhere | Moving a warm session was priced for one named pair, not swept | `adder place` |
| Spot instances under deadlines: uniform progress as a parameter-free policy | The batch discount was known and never recommended, because nothing knew what a deadline was | `adder deadline` |
| Agent programs as first-class scheduling units, prioritised by attained service | Every "restart the session" recommendation assumed attained service was predictive; nobody checked | `adder sched` |
| **Speculative decoding: performance or illusion?** — headline speedups measured at batch size 1, shrinking under realistic load | The `speed` field was recorded on every turn and never audited against its 2x premium | `adder speed` |
| Online predictor evaluation | `p_fail` was inspected on the data it was fitted to, which is a tautology | `adder calib` |
| Resource-aware batching for offline inference | Submission order was never treated as a lever, though the prefix cache is billed by the token | `adder blend` |
| Harvesting preemptible capacity, multi-region spot for batch jobs | Nothing priced what an interruption would destroy, so interruptibility could not be assessed | `adder harvest` |
| Artifact reproducibility | "It was 6.1x and now it is 4.2x" had four candidate explanations and no record | `adder repro` |
| **Controlled router benchmarks**: clustering queries and scoring per cluster matches trained matrix-factorization and graph routers; several published routers fail to beat the best single model | `p_fail` was scoped per (project, tier), which averages over task kinds that have nothing in common | `adder similar`, and `p_fail` conditioned on the task |
| Cold-start routing: a router with no history for a query has to route on something | The classifier abstained and abstention routes up, forfeiting the largest lever in the tool on exactly the tasks it could not read | neighbour-conditioned evidence, which needs no history *at this rung for this project* — only history on tasks like this one |
| Hook-mediated context trimming: dedup and truncate tool results in flight rather than reporting them afterwards | The guard priced the bounded call and then spent a turn asking the model to make it | `guard_narrow` — the guard substitutes the bounded call via `updatedInput` |
| Subscription metering: a rolling window and a weekly cap, not a per-token bill | Every figure in the package was in dollars, which is the wrong unit for a plan user by a change of kind, not a scale factor | `adder limits` |

## Three adoptions worth reading closely

### Clustering routers, and the substitution the constraints here force

The result worth taking from the controlled routing benchmarks is not that any
one router wins. It is that the field's methodological gains are smaller than
they look: on identical splits, the trained routers are matched — and in the
cost-first regime sometimes beaten — by embedding the queries, clustering them,
and scoring each model per cluster, with no network trained at all. The
implication is that most of the recoverable signal is *which kind of task is
this*, and the second finding sharpens it: swapping the embedding backbone
barely moves any of them.

That second finding is what makes the method portable here. If the choice of
embedding hardly matters, then the question is how much is lost by having no
embedding model at all — and for this workload the answer is less than it
sounds. Task descriptions in a coding transcript are short and written by a
model to a fixed prompt, so they are unusually lexically consistent. Vocabulary
overlap is a weak proxy for semantics in general and a decent one here.

So `similar.py` keeps the shape and substitutes the representation: a MinHash
sketch of terms and adjacent bigrams, Jaccard between sketches, nearest
neighbours instead of k clusters. Two deviations are deliberate. The benchmarks
predict *answer quality* from the cluster; this predicts the thing already
measured — how often that rung escalated — because every other number in this
repo is a re-priced observation and a router that starts guessing at quality
would be the first one that is not. And the estimate is not allowed to act
symmetrically: see [tiers.md](tiers.md) for why a thin neighbour set may raise
`p_fail` and may not lower it.

The limitation is inherited and worth repeating: paraphrase with no shared words
is invisible to this, and it fails silently. That is why missing has to cost
nothing, and why the fallback is the tier-wide rate rather than a guess.

### Style control, and why it matters more here than on a leaderboard

The method adds style features — normalised length difference, markdown headers,
lists — as covariates in the Bradley-Terry regression, so the strength
coefficient reflects capability rather than the judge's taste for long,
well-formatted answers. On a leaderboard that is a fairness correction.

Here it is a **cost** correction, and a sharper one. A model that ranks higher
because it writes longer answers is wrong for this tool twice over: the rating
overstates its capability, *and* the extra tokens are billed to you on the turn
they are produced and on every turn afterwards as prefix. Routing on an
uncontrolled rating therefore pays a premium for the very property that inflated
the number.

So `adder verbosity` fits the controlled model and reports the gap between the two
strengths as a **verbosity premium**, priced in dollars per answer against your
own output rates. The limitation the original analysis states is inherited and
repeated: this is observational, and length may correlate with substance.

### Speculative decoding, and the shape of the scepticism

The result worth taking is not about speculative decoding. It is that a widely
cited speedup was measured in the most favourable configuration — batch size 1,
a research prototype — and shrinks toward nothing under production load, because
the system becomes compute-bound and verification dominates.

`adder` records a `speed` field on every turn, and the fast path bills at **2x**.
That number has been in the price table since it was written and no report has
ever asked whether the speed arrived. `adder speed` audits it from the
transcripts: measured tokens per second on fast turns against standard ones,
paired within a model so a mix shift cannot fake a result, with the interval,
and the answer expressed as the break-even speedup the 2x premium requires.

The methodological point transfers exactly: measure it under the conditions you
actually run, not the ones that flatter it.

## What was surveyed and deliberately not built

- ~~**Resource-aware batching for offline inference.**~~ **Built after all, as
  `adder blend`.** The first pass rejected it because the engine belongs to
  somebody else so there is no utilisation to improve. That was the wrong
  reason to stop: you do not control the batching, but you do control the
  submission order, and the prefix cache is billed to you by the token.
  Ordering is free and the saving is real. Building it surfaced a result the
  intuition gets backwards — the saving is **not monotone in the cache TTL**. It
  peaks where the TTL sits above the grouped gap and below the scattered one,
  and falls to zero at both ends, so the report sweeps rather than quoting.
- ~~**Harvesting preemptible capacity.**~~ **Built after all, as `adder
  harvest`.** Rejected as a hardware argument, which skipped the transferable
  half: preemptibility is a property of the *work*, not the machine. Cheap
  capacity is only cheap if being interrupted is survivable, and a transcript
  records exactly how much context an interruption would destroy. The
  conclusion did **not** match the hardware version, which is why building it
  was worth it: measured on 87 real sessions, the discount dwarfs the rebuild
  (breakeven at 13.3 interruptions per session) and a 2,000-token handoff
  removes 1% of the loss rather than most of it. "Checkpointing is what makes
  preemptible capacity pay" holds only where the checkpoint is large relative
  to the state, and a handoff summary against a 500K context is not.
- **Universal text-parameter optimisation.** The system prompt, CLAUDE.md and
  skill descriptions are text parameters billed on every turn, which is a real
  and measurable cost — but optimising them requires running the agent, and this
  package does not run the agent. `adder memory` prices them; improving them is
  out of scope by construction.
- **A dollar figure for the metering window.** `adder limits` reconstructs the
  five-hour window and reports what each one read, but it asserts no capacity.
  The cap is not published in tokens, differs by model, and drains faster in
  peak hours, so the only capacity statement the data supports is "the heaviest
  window on this machine's record was served" — a floor, not a limit. A
  projection naming the minute you get cut off would be a fabrication with a
  clock face on it, so the report compares against that floor and says what it
  is. The one number quoted without qualification is the within-window slope,
  because it does not depend on the boundary rule at all.
- **Backfilling vocabulary sketches into the existing outcome log.** Rows
  written before `similar.py` existed carry no sketch and are invisible to the
  neighbour estimator, which looks like an obvious thing to fix and is not. The
  sketch would have to be written into rows already on disk, and the log is
  append-only for a reason: `record()` relies on a single atomic `O_APPEND`
  write so concurrent sessions interleave lines instead of corrupting them. A
  rewrite has no such guarantee — anything appended between the read and the
  rename is lost, and what would be lost is the evidence the router calibrates
  on. The alternative, a `task_hash -> sketch` sidecar consulted for rows
  lacking their own, works and buys a second cache file to keep coherent.
  Neither is worth it against a problem that resolves itself: new delegations
  carry sketches, the estimator needs four neighbours, and `adder similar`
  reports the coverage fraction so the gap is visible rather than silent. The
  decision is to let it heal and to say so here.
- **A heavy-tailed verdict in `adder sched`.** Attempted and abandoned: every
  finite workload's mean-residual-life curve turns down past the median length,
  so the claim is unfalsifiable on this data. The module says so rather than
  printing a category it cannot defend.
