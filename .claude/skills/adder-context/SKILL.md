---
name: adder-context
description: Decide what to do about a session's context - compact, restart, or carry on - and what a restart may carry. Also audits what CLAUDE.md and memory cost per turn. Use when context is large, a session feels expensive, or before compacting or restarting.
allowed-tools: Bash(adder:*), Bash(./scripts/adder:*), Read
disable-model-invocation: false
---

# Context hygiene

Every number below was computed locally from transcript files before this
prompt was assembled. **They cost zero model tokens.** Do not recompute them,
and do not re-derive the arithmetic — it is measured, not estimated.

```
!`adder live`
```

## The one rule

A token admitted to a persistent context is billed once when it arrives and
again on **every remaining turn**. So the decision is never "is this file big",
it is "how many more times will I pay to re-read it".

## Act on the verdict

The `Context hygiene:` line above is the decision, already priced against this
session's own horizon and cache behaviour.

**`carry on`** — do nothing. Both alternatives destroy information, and neither
is ahead here. Keep admissions small and continue.

**`compact`** — compaction is worth more than the rebuild it pays for. Say so
in one sentence, with the dollar figure, and let the user run it. Do not
compact without asking; it deletes detail that nothing in this tool prices.

**`restart`** — a fresh session is ahead of both carrying on and compacting.
Tell the user, name the figure, and offer the brief. Run:

```bash
adder handoff
```

That prints the brief budget and, ranked, what the brief has to name: files
this session edited (state that exists nowhere else), commands it re-ran, and
the reads that would cost the most to re-establish. If it says *cost is not the
constraint*, write what the next session actually needs — do not be terse to
save tokens you are not spending.

## Never admit the same thing twice

Before reading a file, ask whether it is already in this context. It probably
is — a file read on turn 8 is still resident on turn 140, and re-reading it
buys nothing while costing its whole carry a second time.

```bash
adder reread --min-sessions 2
```

Two distinctions the output makes, and you should too:

- The result is **unchanged** → do not read it again. Use what you have.
- The result **changed** → read it. The stale copy is still resident either
  way, and that is a compaction problem, not a reason to skip the call.

If the same thing is re-learned in *many sessions*, that is a memory candidate
— but check the printed **note budget** first. A resident note is re-read on
every turn of every session; a read is paid only in the sessions that make it.
Writing a large file summary into `CLAUDE.md` to save two reads loses money.

## What memory costs

```bash
adder memory
```

Resident text — `CLAUDE.md`, `MEMORY.md`, skill and agent descriptions — is
re-read on every turn of every session and **cannot be compacted away**. The
report prints the editing unit (dollars per 1,000 resident tokens per session),
so an edit can be priced before it is made: `adder memory --what-if 500`.

Practical consequences:

- Keep instruction files short. Keep skill *bodies* long — a body loads only
  when the skill runs; only its name and description are resident.
- Delete duplicated lines across `CLAUDE.md` files. They are paid for twice on
  every turn, and deleting one copy changes no behaviour.
- Fix stale paths. An instruction that names a moved file teaches every future
  session a wrong fact, and pays to teach it on every turn.

## Do not

- Do not read a large file inline "to check one thing". Delegate it, or read
  the range you need. `adder live` prices both, above.
- Do not compact or restart on a hunch when the verdict says `carry on`.
- Do not add to memory to avoid a read that happens in one or two sessions.
