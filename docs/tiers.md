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
