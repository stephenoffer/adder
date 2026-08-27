# Which tier, for this task

```bash
adder policy "make the ingest step tolerate a partial batch" \
  --context 300000 --remaining 200
```

Below is what that prints once `adder outcomes` has some history to work from.
On a fresh machine the log is empty, every rung falls back to the same prior,
and the same command answers `T2`. That is the point. Without evidence, it
declines to get clever.

## The history used to require a discipline nobody keeps

The outcome log was filled by running `adder outcomes record` after every single
delegation, by hand, so in practice it stayed empty, `p_fail` never left its
prior, and the adaptive half of the tool never ran on any machine. The evidence
was on disk the whole time. A delegation is an `Agent` call with a
`subagent_type`, and its outcome is the result that came back:

```bash
adder outcomes import          # show what it found, write nothing
adder outcomes import --write  # append it to the log
```

It is idempotent, so run it whenever. What it reads is unambiguous: an error
result, or the `ESCALATE:` reply the tier agents are told to return. What it
cannot read is a subagent that returned a confident wrong answer and was
believed. Neither can a person filling in the form afterwards, so the derived
rate is a **lower bound** on failure either way. That matters, because
under-estimating `p_fail` is the direction that costs money.

## The ladder

```
DELEGATE -> route-t1 (claude-sonnet-5, effort=medium)
  modelled saving $5.557  routing overhead $0.160  confidence 0.30
  - no high-precision signal; abstaining and routing up
  - T1 costs $0.2032 expected against T2's $0.3890, and the outcome log backs it:
    12% over 40 project runs (recency-weighted mass 39.3)

  Tier chosen by expected cost, including the risk of redoing it:
     T0 claude-haiku-4-5     default $   0.3449   no measured history at this tier; a prior is not evidence
  -> T1 claude-sonnet-5      medium  $   0.2032   run $0.1376 + 12% chance of redoing it
     T2 claude-opus-5        high    $   0.3890   run $0.3520 + 7% chance of redoing it
     T3 claude-opus-5        xhigh   $   0.3997   run $0.3620 + 7% chance of redoing it
```

The task text is four words of nothing, so the classifier abstains. Abstaining
routes *up*, because a misrouted hard task costs a full retry and a misrouted
easy one costs pennies. That used to be the end of it: abstain, get Opus,
forever, no matter how many times Sonnet had already finished this kind of work
in this repo.

Now the tier is whichever rung has the lowest expected cost:

```
E[tier] = run(tier) + p_fail(tier) x (cost of finishing on T2 + the turn that catches it)
```

## The two directions are held to different standards

Because the two ways of being wrong are not the same size. Moving **up** needs
no evidence at all: the worst case is that you paid for the model you would have
picked anyway. Moving **down**, below what the classifier asked for, needs three
things at once:

1. the classifier abstained rather than matched a signal,
2. the outcome log holds enough recent history at that rung to be evidence
   rather than a prior wearing a number,
3. and the measured failure rate is under that rung's own break-even.

Cheapness alone never buys a downgrade. Under a no-evidence prior the cheapest
rung always looks best, and that is precisely the reasoning being refused.

The whole ladder is printed, losers included, with the reason each one lost. A
router that shows only its answer is indistinguishable from one that guessed.

## Three things the classifier refuses to decide

**A task whose answer is a set.** "Abstaining routes up because a misrouted easy
task costs pennies" assumes the misroute is visible. For coding work it is: the
test fails, you retry, and the retry is the whole cost. For recall it is not. A
weak model asked for every hardcoded credential in a tree hands back three of
the seven, confidently, and nothing fails -- so the cost is not a retry, it is
an audit that is wrong and reads right. The signal that matters there is not
difficulty at all, it is whether an incomplete answer would be detectable, and a
stated quantifier over a plural target -- `every`, `all`, `any` -- abstains
however short the sentence is.

Plurality was the wrong place to stop, and it took a probe set to show it. Three
shapes are the same task, and only the first was caught:

| written | shape | before |
|---|---|---|
| `find every race condition` | quantifier over a plural | T2, abstained |
| `locate the race condition` | a defect class, singular | **T0 at 0.85** |
| `is there anything in the diff that bypasses the consent gate` | a quantifying pronoun | **T0 at 0.85** |
| `verify no credentials are committed in this repo` | a detection, answered "no" | **T0 at 0.85** |

The definite article is a convention of English, not a statement of cardinality:
"find the bug" is not a claim that there is one bug. So a **defect noun** as the
object of a search -- `bug`, `leak`, `vulnerability`, `credential`, `race
condition` -- is unbounded discovery whatever its number, and what bounds it is
scope rather than grammar. `find the hardcoded credential in config.py` stays on
T0, because an incomplete answer about one named file is checkable by opening
it; the same sentence ending `in this tree` is not.

A **quantifying pronoun** is its own plural target. `anywhere` was in the list
and `anything` was not, which was an oversight rather than a distinction.

A **detection** is an enumeration with the list left out. It has none of
`list|show|find|locate|grep|search|count`'s verbs and all of their exposure: a
model that checked three of the seven places answers "no", and "no" is what a
complete answer looks like too. It is gated on a negation or a quantifier beside
the verb, because `check the schema of the events table` is a bounded question.

It is still deliberately narrow. "Check the audit log for tampering" is just as
recall-critical, names no defect noun, states no quantifier and no negation, and
nothing here catches it; a word list wide enough to would stop being
high-precision, which is the only property the classifier has.

**A domain topic word.** `performance`, `security`, `concurrency` and `debug`
used to sit alongside `refactor` and `investigate` and decide the tier on their
own, before anything looked at the shape of the sentence. On a repository whose
subject matter *is* those words, that fires on nearly everything: "where is the
security module" is a one-line grep and was being priced as a threat model, at
roughly five times T0. They are nouns. They are evidence about the repository,
not about the task, so they now only explain an abstention rather than cause
one. What that demotion cost, and what the defect-noun rule above puts back, is
the one place a topic word is strong evidence: as the *object* of a search verb.
`where is the security module` is a lookup; `locate the race condition` is not,
and for a while nothing separated them because the noun had stopped deciding
anything at all. Demoting evidence is not the same as discarding it.

**A class named `Exception`.** The stack-trace signal matched the bare words
`Exception` and `Error:` anywhere in the text, so "rename the Exception class"
read as a crash report at 0.80 confidence. A trace has shape -- a `Traceback`
header, an indented frame, a `File "x", line n`, or an exception name followed
by a colon and a message -- and only the shape counts.

## T0 cannot be handed a write

`Verdict.read_only` was published in `adder classify --json` and `adder policy
--json` for a long time before anything in the tool read it, which left an
integrator free to gate a write permission on a field the tool itself did not
consult. It is now what admits the T0 rung at all: `route-t0` dispatches to an
agent holding Read, Grep, Glob and Bash and no write tool, so a task nothing
marked read-only cannot be carried out there at any price.

The descent rule above is exactly the path that could have sent one. Descending
needs an abstention, and an abstention is the state in which nobody has
established that the task is a read -- so with enough T0 history the
expected-cost search was free to route an edit to an agent that cannot edit.
`adder validate` sweeps it under the name *a mutation never reaches the
read-only rung*.

## Two things this fixed

A failed cheap attempt was being charged twice. Writing it as `cheap + p x (cheap
+ expensive)` bills the cheap run again in a branch where it is not re-run, which
made every cheap tier look worse than it is.

The turn that *notices* the failure was also free, and it should not have been. A
subagent that returns something wrong does not say so; a main-session turn has to
read it and dispatch again, and at 400K context that turn is not rounding error.
Both are on the books now.

Whether the ladder's `p_fail` estimates are any good out of sample is scored
separately, prequentially, by `adder calib`. See [routing.md](routing.md).

## The rate is conditioned on the task, not just the tier

A per-project failure rate is an average over a population that is not one
population. The same repository gets "where is the retry logic" and "make the
scheduler preemptible", and one T0 number across both is too timid for the
lookups and too bold for the refactors, with which error you get depending on
that week's task mix.

So `p_fail` is now conditioned on the task as well as the rung. `adder similar
"<task>"` is the same estimator with the working shown: it finds the recorded
runs whose *vocabulary* resembles the task in hand and reports what happened on
them, per tier, against the tier-wide rate the gate used before.

Similarity is a MinHash sketch of the task's terms and adjacent bigrams, with
Jaccard between sketches. Three properties earn it its place:

- **No model, no dependency, no network.** The routing benchmarks find that
  clustering queries and scoring each cluster is competitive with trained
  matrix-factorization and graph routers, which says the recoverable signal is
  mostly "which kind of task is this", and that kind survives being reduced to
  vocabulary. Nearest neighbours rather than k clusters, because the log is
  small enough that the extra resolution is free and needs no k chosen up front.
- **The task is not stored.** Each slot of the sketch is a minimum over the whole
  term set, so the terms are not recoverable from it. The outcome log has never
  held task text and still does not.
- **It cannot buy a downgrade it has not earned.** The asymmetry above applies
  again, one level deeper, because a rate over four neighbours is far easier to
  push around than one over four hundred runs. A neighbour estimate with real
  mass behind it replaces the tier-wide rate in either direction. A thin one may
  *raise* `p_fail`, which can only decline a downgrade, and declining a
  downgrade costs at most the model you would have used anyway. It is
  discarded when it is thin and optimistic, which is the exact case that would
  spend money on four rows.

When there are too few neighbours, when the log predates sketches, or when the
task shares no vocabulary with anything on record, the estimator returns nothing
and the gate uses the tier-wide rate it always used. A similarity measure that
cannot tell it has missed has to be built so that missing costs nothing.
