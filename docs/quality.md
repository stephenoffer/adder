# Maintaining agent performance

Every lever trades tokens for something, and a degraded agent usually looks
*cheaper per turn* while needing more turns to finish. Cost numbers alone will
happily certify a regression.

`adder quality` reads five performance proxies straight from the transcripts:
tool error rate, correction rate, interrupt rate, turns per prompt, and rework
ratio. `adder verify` refuses to claim a clean saving when any of them
regressed:

```
  Cost per turn ROSE $0.0029. The intervention did not land.

  Agent performance across the same cutover:
    tool_error_rate          0.021 ->      0.012    -43.8%
    correction_rate          0.032 ->      0.013    -58.4%
    turns_per_prompt          30.5 ->       24.4    -20.0%
    No proxy regressed beyond noise.
```

Typical use: apply one lever, then a week later run
`adder verify --since 2026-08-01` and read both halves of the output before
concluding anything.

## Where these proxies cannot help

Both of the above read the same transcripts, through the same parser, priced by
the same cost model as the saving they are checking. That is the tool grading
its own homework: if the cost model is wrong about result sizes or the carry
multiplier, the check is wrong in the same direction and agrees with itself.

The asymmetry matters because of which side is easy. Cost is measured five ways
in this repo. Quality — the thing routing a task to a cheaper tier would
actually lose — was measured by the cost machinery.

`adder ab --recall` is the corroborating signal. It ships a small source file
with a known number of defects planted in it, asks a model to find them, and
counts how many came back:

```
  model                        found    recall  95% CI low      cost
  --------------------------------------------------------------------
  claude-haiku-4-5           6/9          67%         35%$   0.0121
  claude-opus-5              9/9         100%         70%$   0.0904

  claude-haiku-4-5 missed:
    cache_key         the key omits the model, so two models share one cache entry
    retry_fetch       there is no attempt limit, so a permanent failure loops forever
    write_atomic      a fixed `.tmp` name is a shared path between concurrent writers
```

Three things about it are deliberate.

**The denominator is known.** A real file has an unknown number of defects, so
a model that reports four findings on it cannot be scored. Recall is the
measurement that matters here — an incomplete audit reads exactly like a
complete one — and recall needs a denominator that is counted rather than
estimated.

**The prompt does not say how many.** A model told to find nine things reports
nine things, and the number it was given then does the work the measurement was
for.

**Nothing in it touches the cost model.** `adder/evaluate/replay/seeded.py`
imports nothing from `pricing`, `core.trace`, `core.shapes` or `measure`, and
`tests/evaluate/replay/test_seeded.py` asserts that it never will. A
corroborating signal that reaches for `admitted_token_cost` stops corroborating.

The misses are printed rather than summarised because they are the decision:
"Haiku found 6 of 9" is a number, and "Haiku missed the unbounded retry and the
shared temp path" is an answer about what to route to it.

Scope, stated the same way `adder ab` states its own: this measures recall on
defect-finding over supplied source. That is the task class whose failure is
silent, and the one `adder classify` abstains on for that reason. It licenses
nothing about multi-step or agentic work.
