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
