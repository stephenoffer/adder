---
name: adder-doctor
description: Show where Claude Code token spend actually goes, what an output token really costs, and what each lever is worth. Read-only diagnostic - never changes configuration. Use when asked about token cost, spend, why a session is expensive, or how to reduce Claude Code cost.
allowed-tools: Bash(adder:*), Bash(./scripts/adder:*), Read
disable-model-invocation: false
---

# Cost diagnosis

Every report below was computed locally from transcript files before this prompt
was assembled. **They cost zero model tokens.** Do not recompute them.

## The ranked answer

```
!`adder doctor`
```

## This session, right now

```
!`adder live`
```

## Going deeper

`adder tools`, `adder savings`, `adder debt`, `adder context`, `adder anomaly`,
`adder reread` and `adder agents` each go deeper on one finding. Run one **only**
if the two reports above point at it.

That instruction is priced, not stylistic. This prompt is context: written once,
re-read every remaining turn. The two reports above are ~1,300 tokens — **$0.20
to carry through 300 more turns on Opus 5**. Also inlining `tools` and
`savings`, which restate findings `doctor` has already ranked, made it ~2,400
tokens and **$0.38**: a third of this diagnosis spent saying the same thing
twice.

The exception is `adder guard`. It is the only component that prevents spend
rather than reporting it, and an uninstalled guard is indistinguishable from a
quiet one. `doctor` fails its `guard` check when it is not installed.

## How to read this to the user

**Lead with `doctor`'s top finding.** It is already ranked by dollars at stake,
and that ranking is the answer to "where do I start". Do not re-derive an order
from the other reports; if you disagree with the ranking, say why in a sentence
rather than silently reordering it.

**Then the split, not "write less".** Context growth is roughly half assistant
output and half read content (tool results, mostly `Bash`). Terseness only
reaches the first half; bounding tool output only reaches the second. Quote the
measured split from `adder tools` rather than assuming — it differs per
workload, and the advice inverts when reads dominate. `adder tools` also names
the specific lever for the worst tool, which is more actionable than "be
concise": pipe through `head`, pass `offset`/`limit`, delegate the read.

Then the number that changes behaviour: what 10K tokens added to the *current*
context costs over the rest of this session, from `adder live`.

Rules for reporting honestly:

- **Confidence labels are load-bearing.** MEASURED is recomputed from recorded
  tokens. ATTRIBUTED is exact arithmetic assigning real spend to a cause.
  MODELLED depends on a stated assumption — quote the assumption when citing it.
- **The pool levers are substitutes, not complements.** Never quote their sum;
  quote the combined figure, which composes them multiplicatively.
- **Session length usually dominates.** Cost per turn tracks context; context
  tracks per-turn growth x session length. The factors multiply, so cutting
  verbosity while sessions grow longer nets nothing — this is measured.
- **Effort is the cache-safe output lever.** Lowering effort cuts output volume
  without invalidating the prompt cache. A model downgrade rebuilds the entire
  prefix at up to 12.5x the cached read price; an effort change costs nothing to
  apply mid-session. Prefer it to a downgrade.
- **Per-turn model downgrade is the smallest lever**, typically well under 1%.
  If the user expected model routing to be the answer, say plainly that the data
  disagrees — and that at a typical context the cheap model cannot even hold the
  conversation.
- **Never quote a saving without the feasibility check.** Haiku holds 200K; the
  median peak context here is 544K.

- **`at stake` is not a promise.** `doctor` prices what the measurement says is
  addressable. Levers overlap, so fixing two does not save the sum of both;
  `adder plan` is the only figure with the overlap removed.

Recommend at most the top two levers. Then tell them how to check it landed:
`adder verify --since YYYY-MM-DD`, which reports failure when cost did not fall,
and `adder quality --since YYYY-MM-DD`, which reports whether the agent got worse.
A cheaper agent that needs more turns is not cheaper.

Do not offer to change configuration unless asked — that is `/adder-init`.
