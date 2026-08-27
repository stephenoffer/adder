# adder vs no adder

*(Figures from one machine's history, taken at $5,846 across 23,922 turns and 90
sessions. The transcript pool grows with every session, so `adder bench` will
report slightly different totals than the ones below. The multiples are the
stable part; the dollars are not.)*

Every other report here answers "where did the money go". This one answers the
question that comes before installing anything: **what changes if I install this
and keep working exactly as I do now?**

That number is smaller than the one `adder plan` quotes, and the gap between
them is the finding.

## The result

| configuration | total | vs no adder |
|---|---|---|
| no adder (as run) | $5,846 | 1.0x |
| + the read guard, at its shipped defaults | $3,943 | **1.5x** |
| + the tier agents in `.claude/agents/` | $3,730 | **1.6x** |
| + the threshold and cadence the reports solve | $869 | **6.7x** |

The first three rows happen without you doing anything. The fourth is advice,
and nothing in this repo enforces it.

So the honest one-line summary is two numbers, not one:

- **1.6x** for installing it and changing nothing.
- **6.7x** if you then work the way it tells you to, at nominal assumptions.
  **3.4x** at the pessimistic corner of them.

Reporting only the 6.7x would be the more impressive claim and a lie by
omission. It is not what installing the tool gets you; it is what restructuring
your work around the tool gets you.

## What moved when the guard learned to refuse

That gap was the tool's own indictment, and `adder auto` is the answer to it.
An enforcing guard does not advise the delegation threshold, it refuses the
calls above it, so most of the fourth row crossed the line. Re-run on a later
and larger corpus (118 sessions, 33,192 turns, $7,888 as run):

| configuration | total | vs no adder | who does it |
|---|---|---|---|
| no adder (as run) | $7,888 | 1.0x | — |
| + the read guard, refusing over 800 tok | $3,186 | **2.5x** | the hook |
| + the tier agents in `.claude/agents/` | $2,567 | **3.1x** | the agent files |
| + the threshold it solves for, over 300 tok | $1,667 | 4.7x | you |
| + restarting every 21 turns | $1,233 | 6.4x | you |

For comparison, the advisory install on that same corpus is $4,954, or **1.6x**.

**1.6x → 3.1x for installing it and changing nothing.** The ladder is longer by
one row because the delegation threshold and the restart cadence used to be
bundled, and they are enforceable by different things: a hook can refuse a
read, and nothing here can restart a session. Bundling them marked the whole
rung advisory and hid the fact that most of it no longer is.

Two rows are still yours, and the larger of the two is the cadence. That is the
honest ceiling of the automatic number on this workload: session length is the
biggest single lever in `adder savings` (39% of the addressable pool) and no
hook event can pull it.

The third row does not become enforced when you activate, and the reason is
worth stating: the guard refuses at 800 tokens, the reports solve for ~300, and
below 800 the hook would parse a transcript on half of all tool calls to find
money the dollar gate has already found. Crediting activation with the
difference would be crediting it with money it does not collect.

## Method

`adder bench` replays every recorded turn under each configuration and re-prices
it. It is the same replay engine `adder plan` uses, and the same fidelity check
applies: the null configuration must reproduce the measured bill. It does, to
within 0.0% here, and the report prints that residual first. A benchmark whose
baseline does not reproduce reality cannot say anything about a ratio taken
against it.

Rows are **cumulative**, because the levers are substitutes. They all attack the
same pool (tokens admitted to a context that is then re-read every turn), so
pricing them independently and adding the results counts the same dollars more
than once.

### The row this ladder was missing

The tables above have no row for `guard_enforce=certain`, the level
`adder auto on` installs by default, and the reason is that until recently
there was nothing to put in one. Two things were wrong at once. `adder reread`
keyed on `Read`'s `file_path`, so on any workload whose harness reads through
the shell it measured $0.00 — and this module asked whether the guard was set
to `full`, so `certain` was reported as unenforced even where it was on.

There is now a first rung: the results a turn admitted that its own context
already held, dropped turn by turn, with the rest of the session re-priced.
It is the only row in the ladder with no modelled input behind it — no summary
ratio, no `p_fail`, no handoff — because the call does not run. `adder reread`
measures the set and `adder bench` replays it, so the two cannot drift.

Two honest limits. The measurement estimates result sizes from characters and
cannot see a file edited by a peer process or by `sed -i`, so it is an upper
bound; and the subtraction is clamped to what each turn actually admitted,
since context growth and estimated result sizes are counted by different
methods. The figures in the tables above were taken before the row existed and
are not restated here — run `adder bench` to place it on your own workload.

### The guard's threshold is derived, not chosen

The PreToolUse guard fires on a **cost** ($0.25 by default), not a token count,
because the same read is worth interrupting for at turn 400 and not at turn 3.
Turning that into "delegate reads over N tokens" is one division: admitting a
token to a context that will be re-read `E` more times costs `(w + m·E)` times
the input rate, so the size at which that reaches $0.25 falls straight out.

On this workload `E` is 321 expected re-reads, which puts the dollar gate at
1,500 tokens, **below** the hook's 2,000-token floor. That floor exists so the
hook does not parse a transcript on every trivial read, and here it is the
binding constraint. Anyone tuning `ADDER_GUARD_MIN_COST` on this workload would
be tuning a gate that is not doing the work.

### What the tiers add

Moving a read out of the context is worth 1.48x on its own, with the subagent
held on the same model the session was already using. Letting the tier files
choose the model by expected cost adds the rest: 1.48x → 1.57x. Placement is
most of the lever; price is the remainder. That ordering is deliberate, so tier
choice cannot claim credit for the move.

## What the 6.7x rests on

Three inputs, none of which a transcript can settle, all swept rather than
asserted:

| summary ratio | p_fail | handoff | vs no adder |
|---|---|---|---|
| 10% | 15% | 2,000 | 6.7x |
| 10% | 15% | 20,000 | 4.5x |
| 10% | 30% | 2,000 | 6.0x |
| 10% | 30% | 20,000 | 4.1x |
| 30% | 15% | 2,000 | 5.3x |
| 30% | 15% | 20,000 | 3.8x |
| 30% | 30% | 2,000 | 4.7x |
| 30% | 30% | 20,000 | **3.4x** |

- **Summary ratio.** What a delegated read hands back. At 10% the content stays
  out of the context; at 30% most of the carry it was supposed to avoid comes
  back. This sets the floor of the range, and `adder ab` is the only thing here
  that can test it.
- **p_fail.** How often a delegated step has to be redone on the expensive
  model. Doubling it costs about 0.7x of the multiple, which is less than the
  summary ratio and less than the handoff.
- **Handoff.** How many tokens a restarted session has to be told. Nothing in a
  transcript records what a person needs to resume. A 10x larger handoff costs
  about 2.2x of the multiple, which makes it the second-softest input here.

At this threshold 99% of admitted tokens are delegated. That is not a tweak to
how you work. It is the orchestrator pattern, where the main session holds the
thread and almost every step that would admit content runs somewhere else. It is
worth being clear that the 6.7x is the price of adopting that pattern, not the
price of installing a hook.

## Why this is not `adder plan`

`plan` asks the optimiser's question — what is the cheapest way this workload
could have been run — and reaches 10.7x by adding effort reduction, terseness,
tool discipline and a cheaper session model on top of everything above. Those
are real levers and they are priced honestly, but each one is another thing the
reader has to do, and none of them is something the tool does.

`bench` stops at the line between *what the software does* and *what you do*,
because that is the line a person is deciding about before they install
anything.

## Re-running it

```bash
adder bench                      # the table above, against your own history
adder bench --json               # machine-readable
adder bench --guard-cost 0.10    # tighten the guard and re-price
adder validate                   # re-test both headline numbers as claims
```

Two of `adder validate`'s claims are these numbers:

- *installing it pays before you obey it*: the enforced rungs clear 1.3x.
- *the advice reaches 5x*: the solved regime clears 5x at nominal assumptions.

Both are workload-dependent and expected to fail on some workloads. A workload
whose sessions stay short has little carry to remove, and the honest answer
there is that the multiple is not available, not that the tool should look
harder. Run it against your own transcripts before believing any figure here.
