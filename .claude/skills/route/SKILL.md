---
name: route
description: Pick the cheapest capable model and the cheapest place to run a task (inline vs delegated subagent), using a cache-aware cost model. Use when explicitly asked to route, delegate, or minimise cost for a specific task.
argument-hint: [task description]
allowed-tools: Bash(/Users/stephen.offer/Desktop/llm-router/scripts/rt:*), Agent, Read, Grep, Glob
disable-model-invocation: true
---

# Route: $ARGUMENTS

The recommendation below was computed locally before this prompt was assembled.
It cost **zero model tokens**. Do not recompute or second-guess the arithmetic.

```
!`/Users/stephen.offer/Desktop/llm-router/scripts/rt policy "$ARGUMENTS"`
```

## Act on it

**If the plan says INLINE, or says the saving does not clear routing overhead:**
just do the task yourself, now. Do not delegate. Say one short sentence about
why routing was skipped. Routing something that costs more to route than to do
is the single most common way a router loses money.

**If the plan says DELEGATE and the saving clears overhead:** dispatch with the
**Agent** tool, passing the named agent as `subagent_type` and the named model
as `model`. Pass the model explicitly — do not rely on the agent file's default,
so the choice is deterministic and visible in the transcript.

Write the brief so the subagent needs nothing from this conversation: it has a
fresh context and cannot see anything here. State the goal, the paths, the
constraints, and the shape of the answer you want back.

Ask for a **compact** result — findings and `file:line` citations, not file
contents. The subagent's reply is re-read on every remaining turn of this
session, so a verbose reply erases the saving that justified delegating.

**If the subagent replies `ESCALATE: <reason>`:** dispatch once more at the next
tier up (route-t0 → route-t1 → route-t2), including the reason in the new brief.
Escalate at most once. If a tier already edited files before escalating, tell the
next tier exactly which files it touched.

## Record the outcome

After the task finishes, log it so the escalation gate calibrates on real data
instead of a guess (`p_fail` starts at a cautious 0.5 prior and converges as
history accumulates):

```bash
/Users/stephen.offer/Desktop/llm-router/scripts/rt outcomes --help   # see current calibration with: rt outcomes
```

Record via python: `from router.outcomes import Outcome, record` — set
`escalated=True` if any tier replied `ESCALATE`, and pass the tier, model, and
project. This is the only mechanism that makes the router adaptive over time.

## Verify

After dispatch, confirm the model that actually ran. An organisation
`availableModels` allowlist, or a `CLAUDE_CODE_SUBAGENT_MODEL` environment
variable, silently substitutes a different model — so treat the agent's own claim
about which model it is as unreliable.
