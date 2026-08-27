# Routing, and how to tell whether it worked

This is the reasoning behind `adder routereval`, `adder calib`, and the
Bradley-Terry layer they both stand on. The short version: adder makes routing
recommendations, and until recently it had no way to say whether those
recommendations were any good. A recommendation you cannot score is an opinion.

## The problem with every routing claim

"Route the easy work to a cheaper model and save 40%" is three claims wearing
one sentence:

1. that some work is easy,
2. that a classifier can tell which,
3. that the cheaper model does that work acceptably.

Claim 1 is usually true. Claim 3 is testable and `adder ab` tests it. Claim 2 is
the one that gets asserted and never measured, and it is the one that decides
whether the whole thing works. A router that cannot separate easy from hard is
not a cheap router, it is a coin flip with extra steps, and it will still show
a saving, because sending 40% of work to a cheaper model always shows a saving.
The saving is not the question. The question is what it cost in quality, and
whether a router did better than picking at random.

## The metric

Fix a strong model and a weak model. A router sends some share of work to the
strong one. Then, with `r()` meaning average quality:

```
PGR   = (r(router) − r(weak)) / (r(strong) − r(weak))
APGR  = mean of PGR across the whole range of call fractions
CPT(x) = smallest share of strong calls that reaches PGR = x
```

PGR on its own is gameable: send everything to the strong model and PGR is 1.0
at no saving. So the summary number is APGR, the area under the
call-performance curve, which prices the whole trade-off rather than one point
on it.

**A random router scores APGR = 0.5.** That is the property that makes the
number readable, and it is why `routereval` discretises the axis at bin
midpoints rather than upper edges (upper edges score a random router 0.55, and
then every result has to be read against a baseline nobody remembers).

Three things `adder routereval` prints that the bare metric does not:

- **The random baseline as an interval, not a constant.** On 40 recorded
  episodes a random router lands anywhere between about 0.42 and 0.58. A router
  scoring 0.55 on that sample has demonstrated nothing, and the report says so
  rather than printing "+10% over random".
- **The oracle ceiling.** APGR does not top out at 1.0. If half the tasks
  genuinely need the strong model, a router with perfect foresight scores 0.75.
  Reading 0.75 as "75% of the way to perfect" is wrong: on that task mix it
  *is* perfect. The report prints the ceiling and the regret against it.
- **A dollar axis.** See below.

## Where we deviate: calls are not costs

The published metric puts *fraction of calls to the strong model* on the x axis.
That is right when calls cost about the same. In an agent session they do not,
and the difference is not small.

A strong call on a 190K-token context costs roughly forty times a weak call on
an 8K one. So a router sending 30% of *calls* to the strong model can easily be
spending 95% of the all-strong *budget*. Reporting "30% of calls" as the cost of
that router does not understate the answer, it inverts it.

`routereval` therefore computes both curves:

| axis | x is | use it for |
|---|---|---|
| `calls` | share of calls routed strong | comparing against published numbers |
| `cost` | share of the all-strong dollar budget spent | deciding anything |

When they disagree by more than 0.05 APGR, the report says so explicitly. That
gap is a property of the workload, and on agent sessions it is usually large.

## The comparison PGR structurally cannot make

PGR is 0 at the all-weak endpoint and 1 at the all-strong one, by construction.
The all-strong endpoint *is* a single model, so no PGR-derived number, APGR
included, can answer the question a sceptical reader asks first: **would picking
one model and never routing at all have done just as well?**

The published benchmarks ask it, and it is the question that embarrasses the
field. Several well-known routers (a binary-classifier one, a cascade, and a
commercial auto-router) fail to beat the best single model in at least one
regime. A metric family anchored at the weak arm cannot show that, which is
presumably part of why it went unreported for so long.

So two more numbers are printed, named as the benchmarks name them so they can
be quoted next to a published figure without translation:

- **gain vs best single**: the quality the best threshold on the curve adds over
  the better of the two fixed choices. Zero is the common answer and it is a
  finding, not a measurement failure: it means no mix beat just picking one.
  It is positive only where the two models are genuinely complementary.
- **cost saved at equal quality**: the cheapest threshold whose quality still
  matches that better fixed choice, read as a share of the all-strong budget.
  This is the figure a reader actually wants: not "how much of the gap did it
  recover" but "how much cheaper can this get before it starts costing me
  answers".

`best_single` is the *better* of the two arms and is deliberately not assumed to
be the strong one. On a task mix where the weak model wins, quoting the strong
arm as the baseline would flatter every router scored against it.

The one benchmark metric not adopted is ParetoDist, distance to the Pareto
frontier. Here the frontier is two fixed points and an oracle, so the distance to
it collapses into the oracle regret already reported. Adding it would be the
same number under a second name.

## Where the episodes come from

An episode is one task scored and priced under both arms. That is a
counterfactual, and counterfactuals are where cost tools lie. Three sources, in
descending order of how much they should be trusted:

1. **An A/B log** (`adder ab`). Both arms actually ran. Nothing is modelled.
2. **The outcome log.** One arm ran; the other is a recorded fact about what
   happened next (the task escalated, and the escalation resolved it). One side
   measured, one side inferred.
3. **A modelled arm.** Priced from the catalog, quality inferred from arena
   ratings. Labelled `MODELLED` in the output, and everything computed from it
   inherits the label.

Source 2 is the default because it needs no setup and it scores the router that
actually ran rather than a hypothetical one. Its assumption — that the strong
arm always succeeds — makes the resulting APGR an **upper bound** on measured
router skill, not a two-sided estimate. That is stated in the report, not in a
footnote.

## Ratings are estimates, and the interval is load-bearing

Underneath all of this sits a quality signal, and for cross-vendor comparison
the only public one that is updated within days of a launch and is not
self-reported is head-to-head preference. `adder` reads it from LMArena.

Two things about that signal matter more than its value:

**It is a Bradley-Terry maximum-likelihood fit, not Elo.** People say Elo and
mean this. The distinction is not pedantry: Elo is an online update rule that
weights recent games more heavily and depends on the order games arrived in, so
re-running it on a shuffled log gives different ratings. Bradley-Terry is a
batch MLE with no order dependence, reproducible from the log alone. For a tool
whose whole claim is "re-run the measurement", that is the only defensible
choice. `adder/pricing/bt.py` implements it, and `agreement()` checks a local
fit reproduces the published ordering.

**It has a confidence interval that is wider than people assume.** At the top of
the coding board the 95% interval is roughly ±10 points, so a "17-point lead" is
two overlapping intervals and no lead at all. Anything here that compares two
models can answer *indistinguishable*, and does so most of the time. When two
models cannot be told apart on quality, the only remaining difference is price,
which is a routing decision the data actually supports.

One implementation note worth stating because it changes a number: the
leaderboards resample rows of the battle log to get their intervals, which is
correct on a large log and fails silently on a small one. Six battles that a
single model swept have no outcome variation to resample, every resample refits
to the same value, and the reported interval has *zero width*. `fit_with_ci`
defaults instead to holding the matchup schedule fixed and redrawing each winner
from the fitted probabilities. On large logs the two agree within a point or
two; on small ones only the second is honest. The published method is still
available as `method="battles"`.

### Tiers, not a total order

Pairwise data between any two specific models is under 0.1% dense. "Is X better
than Y" is usually unanswerable; "is X in the top tier" is not. So `bt.tiers`
partitions models into strength classes by exact dynamic programming, the
one-dimensional k-means that has an optimal solution, so there is no
initialisation to get unlucky with and no seed to report. Everything in adder
that says "the strong model" means a tier.

## Calibrating the other half

Routing has a second predictor in it: `p_fail`, how often a tier fails and
forces an escalation. Every gate multiplies by it. It had never been scored:
`outcomes calibration` printed the escalation rate next to the data it was
fitted on, which is a tautology, not a test.

`adder calib` scores it **prequentially**: walk the log in timestamp order, ask
the estimator for a probability using only the rows before this one, then reveal
the outcome. Every prediction is out-of-sample, every row is used, and the score
is what the deployed gate actually experienced. This is the standard evaluation
for an online predictor and the only one that is honest about a recency-weighted
fit.

It reports Brier, but the number to read is the **skill score against always
predicting the base rate**. A Brier of 0.18 sounds respectable until a constant
scores 0.17, at which point the per-project scoping is decoration. Skill can go
negative, and a report that can only print "here is the rate per tier" would
never reveal that.

Building this found a real bug. The estimator decayed its recency weights toward
the wall clock, so replaying any log older than a few half-lives collapsed the
evidence mass to nothing and returned the 0.5 prior for every row, which looks
like a calibrated coin and is really an admission that no data was used. It also
made the function untestable to a fixed value. `evidence()` and `p_fail()` now
take an explicit `now`.

## What none of this establishes

Arena preference is not agentic tool use. It is a proxy, chosen because it is
public, cross-vendor, and fast to update, and it is labelled `MODELLED`
everywhere it surfaces. A high APGR computed against a modelled arm means the
router agrees with the model of quality, which is not the same as the router
being right. The only thing that settles that is running both arms, which is
what `adder ab` is for and why its sample-size calculation is printed before you
spend money rather than after.

Nor does any of it transfer between corpora on the strength of session length
alone. The classifier fires on generic English verbs -- `refactor`,
`investigate`, `why is`, `across the codebase` -- and a domain workload speaks
in nouns instead: *this Ray Data pipeline is spilling to disk*, *the NCCL
collective hangs during allreduce*, *recommend an instance type for
Llama-3.1-70B*. Twelve phrasings of that shape produce twelve abstentions, and
an abstention routes up, to where the session already was, having charged a
routing turn to say so. Whatever multiple gets measured here is a property of
the corpus's **task vocabulary** as much as its horizon, and a `adder bench`
run that comes back all-abstention is reporting that the classifier has nothing
to say about that vocabulary -- not that the work is hard.
