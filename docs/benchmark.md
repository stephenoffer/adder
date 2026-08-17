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
| no adder — as run | $5,846 | 1.0x |
| + the read guard, at its shipped defaults | $3,943 | **1.5x** |
| + the tier agents in `.claude/agents/` | $3,730 | **1.6x** |
| + the threshold and cadence the reports solve | $869 | **6.7x** |

The first three rows happen without you doing anything. The fourth is advice,
and nothing in this repo enforces it.

So the honest one-line summary is two numbers, not one:

- **1.6x** for installing it and changing nothing.
- **6.7x** if you then work the way it tells you to — at nominal assumptions,
  **3.4x** at the pessimistic corner of them.

Reporting only the 6.7x would be the more impressive claim and a lie by
omission. It is not what installing the tool gets you; it is what restructuring
your work around the tool gets you.

## Method

`adder bench` replays every recorded turn under each configuration and re-prices
it. It is the same replay engine `adder plan` uses, and the same fidelity check
applies: the null configuration must reproduce the measured bill. It does, to
within 0.0% here, and the report prints that residual first. A benchmark whose
baseline does not reproduce reality cannot say anything about a ratio taken
against it.

Rows are **cumulative**, because the levers are substitutes. They all attack the
same pool — tokens admitted to a context that is then re-read every turn — so
pricing them independently and adding the results counts the same dollars more
than once.

### The guard's threshold is derived, not chosen

The PreToolUse guard fires on a **cost** ($0.25 by default), not a token count,
because the same read is worth interrupting for at turn 400 and not at turn 3.
Turning that into "delegate reads over N tokens" is one division: admitting a
token to a context that will be re-read `E` more times costs `(w + m·E)` times
the input rate, so the size at which that reaches $0.25 falls straight out.

On this workload `E` is 321 expected re-reads, which puts the dollar gate at
1,500 tokens — **below** the hook's 2,000-token floor. That floor exists so the
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

- **Summary ratio** — what a delegated read hands back. At 10% the content stays
  out of the context; at 30% most of the carry it was supposed to avoid comes
  back. This sets the floor of the range, and `adder ab` is the only thing here
  that can test it.
- **p_fail** — how often a delegated step has to be redone on the expensive
  model. Doubling it costs about 0.7x of the multiple, which is less than the
  summary ratio and less than the handoff.
- **Handoff** — how many tokens a restarted session has to be told. Nothing in a
  transcript records what a person needs to resume. A 10x larger handoff costs
  about 2.2x of the multiple, which makes it the second-softest input here.

At this threshold 99% of admitted tokens are delegated. That is not a tweak to
how you work — it is the orchestrator pattern, where the main session holds the
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

- *installing it pays before you obey it* — the enforced rungs clear 1.3x.
- *the advice reaches 5x* — the solved regime clears 5x at nominal assumptions.

Both are workload-dependent and expected to fail on some workloads. A workload
whose sessions stay short has little carry to remove, and the honest answer
there is that the multiple is not available, not that the tool should look
harder. Run it against your own transcripts before believing any figure here.
