---
name: route-doctor
description: Show where Claude Code token spend actually goes and what each cost lever is worth. Read-only diagnostic - never changes configuration. Use when asked about token cost, spend, why a session is expensive, or how to reduce Claude Code cost.
allowed-tools: Bash(/Users/stephen.offer/Desktop/llm-router/scripts/rt:*), Read
disable-model-invocation: false
---

# Cost diagnosis

Both reports below were computed locally from transcript files before this
prompt was assembled. **They cost zero model tokens.** Do not recompute them.

## This session, right now

```
!`/Users/stephen.offer/Desktop/llm-router/scripts/rt live`
```

## Historical spend and what each lever is worth

```
!`/Users/stephen.offer/Desktop/llm-router/scripts/rt savings`
```

## Escalation calibration

```
!`/Users/stephen.offer/Desktop/llm-router/scripts/rt outcomes`
```

## How to read this to the user

Lead with the single number that changes their behaviour: what 10K tokens added
to the current context costs over the rest of the session. That is the figure
that makes delegation obviously correct.

Then, briefly:

- **Confidence labels are load-bearing.** MEASURED is recomputed from recorded
  tokens. ATTRIBUTED is exact arithmetic assigning real spend to a cause.
  MODELLED depends on a stated assumption. Never present a MODELLED figure as
  measured, and quote its assumption when you cite it.
- **Split and Delegate are not additive** — both draw on the same cache-read
  pool. Quoting their sum overstates the opportunity.
- **Per-turn model downgrade is the smallest lever**, typically under 1%. If the
  user expected model routing to be the answer, say plainly that the data
  disagrees and that context placement dominates.

Recommend at most the top two levers. Do not offer to change configuration
unless asked — that is `/route-init`.
