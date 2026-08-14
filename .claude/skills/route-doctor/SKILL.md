---
name: route-doctor
description: Show where Claude Code token spend actually goes, what an output token really costs, and what each lever is worth. Read-only diagnostic - never changes configuration. Use when asked about token cost, spend, why a session is expensive, or how to reduce Claude Code cost.
allowed-tools: Bash(/Users/stephen.offer/Desktop/llm-router/scripts/rt:*), Read
disable-model-invocation: false
---

# Cost diagnosis

Every report below was computed locally from transcript files before this prompt
was assembled. **They cost zero model tokens.** Do not recompute them.

## This session, right now

```
!`/Users/stephen.offer/Desktop/llm-router/scripts/rt live`
```

## What an output token really costs

```
!`/Users/stephen.offer/Desktop/llm-router/scripts/rt debt`
```

## What each lever is worth

```
!`/Users/stephen.offer/Desktop/llm-router/scripts/rt savings`
```

## How to read this to the user

**Lead with the root cause, not the symptom.** Spend looks input-dominated, but
assistant output is ~105% of context growth — the context is the model's own
prior words being re-read. Output is the cause; input cost is the symptom.

Then the number that changes behaviour: what 10K tokens added to the *current*
context costs over the rest of this session.

Rules for reporting honestly:

- **Confidence labels are load-bearing.** MEASURED is recomputed from recorded
  tokens. ATTRIBUTED is exact arithmetic assigning real spend to a cause.
  MODELLED depends on a stated assumption — quote the assumption when citing it.
- **The three pool levers are substitutes, not complements.** Never quote their
  sum; quote the combined figure, which composes them multiplicatively.
- **Session length usually dominates verbosity.** Cost per turn tracks context;
  context tracks output-per-turn x session-length. Cutting verbosity while
  sessions grow longer nets nothing — this is measured, not theoretical.
- **Per-turn model downgrade is the smallest lever**, typically under 1%. If the
  user expected model routing to be the answer, say plainly that the data
  disagrees.

Recommend at most the top two levers. To check whether a change actually landed,
point them at `rt verify --since YYYY-MM-DD`, which reports failure when cost
did not fall. Do not offer to change configuration unless asked — that is
`/route-init`.
