# adder vs no adder

*(Figures from one machine's history: 103 sessions, 46,687 turns, $8,757 as
run, measured 2026-09-28. The transcript pool grows with every session, so
`adder bench` will report slightly different totals. The multiples are the
stable part; the dollars are not.)*

Every other report here answers "where did the money go". This one answers the
question that comes before installing anything: **what changes if I install this
and keep working exactly as I do now?**

## The result

| configuration | total | vs no adder | who does it |
|---|---|---|---|
| no adder (as run) | $8,757 | 1.0x | — |
| + the guard's duplicate refusal | $8,581 | 1.02x | the hook |
| + the read guard, refusing over 800 tok | $8,156 | 1.07x | the hook |
| + the tier agents in `.claude/agents/` | $7,897 | **1.11x** | the agent files |
| + restarting every 34 turns | $3,131 | **2.8x** | you |

That is `adder auto on --full`. The advisory install, which only describes
what it would refuse, comes to $8,073, or 1.08x.

So the honest summary is two numbers, and the first is small:

| | multiple |
|---|---|
| installing it and changing nothing | 1.1x |
| then restarting when it says to | 2.8x nominal, 2.4x at the pessimistic corner |

Nearly all of the second number is the restart cadence. Session length is the
biggest lever in `adder savings`, and no hook event can pull it.

## Measured in real sessions

Every figure above is a replay. `adder trial` runs the experiment instead:
small repositories with failing tests, a real headless Claude Code session per
arm, and hidden tests copied in afterwards to decide whether the work was
done. The baseline is Opus 5, the model 84% of the measured history ran on, at
its default effort, with no hooks. Run on 2026-09-28, $28.88 in all.

Five short tasks (three of them in three parts), two repeats each:

| arm | passed | cost per run | vs baseline |
|---|---|---|---|
| baseline, Opus 5 | 10/10 | $0.413 | 1.0x |
| Opus 5.5, effort medium, hooks, tier agents | 10/10 | $0.307 | 1.35x |
| Sonnet 5, the same | 10/10 | $0.205 | 2.02x |
| Sonnet 5, the same, plus a terseness instruction | 10/10 | $0.180 | 2.29x |
| Haiku 4.5, the same | 9/10 | $0.166 | 2.49x |
| Haiku 4.5, escalating to Sonnet when visible tests fail | 9/10 | $0.163 | 2.54x |

On an eight-part task and on `sprawl`, sixteen failing tests across a
160K-token repository, every arm passed every run and the multiples were the
same size or smaller: 1.8x to 2.0x for Sonnet on the eight-part task, 1.7x on
`sprawl` in one session.

Three things this settles, and one it does not.

- **The session model is the lever that holds up.** Sonnet 5 lost nothing
  measurable against Opus 5 on this work, 10/10 each. With ten runs a loss of
  up to 28 points is still inside the interval.
- **Haiku is not cheaper in practice.** It is a fifth of the price per token
  and took about twice the turns, and it failed a task Sonnet did not.
- **A restart per part costs more than it saves on tasks like these.** A
  capable agent reads narrowly, so the baseline's context never grew: on
  `sprawl` it ran 57 turns in one session for $1.61. Sixteen restarts, each
  paying a ~21K-token opening, brought Sonnet to 1.18x, against 1.70x for the
  same model in one session. `plan`'s solved cadence is 34 turns, not one
  part, and the cost advisor has to make that call on the live context.
- **Not settled: the long-session carry.** The largest lever in the replay
  is restarting sessions of the length this history actually has (p90 974
  turns, peak contexts to 900K). No synthetic task here built context like
  that, so the 2.8x from restarts is still the replay's, with its 2,000-token
  handoff modelled.

Put together: measured on real sessions, adder's configuration is 2.0x to
2.3x cheaper at equivalent output on self-contained tasks. On this history's
long sessions the replay adds the restart cadence on top of that. Nothing
measured here reaches 10x.

## What the earlier 3.1x got wrong

This page used to report 3.1x hands off and 6.4x with restarts, on an older
corpus. Four errors in the replay produced most of that, and they all ran the
same way:

1. The delegation gate fired on the whole growth of the context between two
   turns. More than half of that growth is the model's own output, which no
   hook can send anywhere. The replay then priced that output at the
   subagent's rate as well.
2. Turns already running inside a subagent were delegated again. Claude Code
   subagents cannot spawn subagents, and those turns were 39% of the
   delegations.
3. A subagent was priced as one uncached pass over the read plus a 400-token
   brief, about $0.002 on Haiku. Measured over 258 real subagent runs, a
   subagent's first turn carries a median 34.6K tokens of its own system
   prompt, tools and memory, and most of it is written rather than read from
   cache.
4. The turn that dispatches a delegation, which re-reads the whole main
   context, was only charged when the delegation failed.

The replay now delegates only a tool result, only from the main chain, and only
where doing so is cheaper than carrying it to the next restart or compaction.
It uses the session's real remaining turns for that decision, which the live
guard can only estimate. That makes the hands-off row an upper bound on what
the hook collects. `placement_cost`, which the guard itself refuses on, got the
same subagent opening and dispatch turn, so at `guard_enforce=full` it no
longer refuses reads that would have been cheaper inline.

One lever is only partly measured. The tier agents choose Haiku for nearly
every delegated read, at the same 15% chance of a redo as every other tier. On
the kind of work a delegated read is, answering a question from supplied
source, `adder ab --run --backend cli` scored Haiku 4.5 12/12 against Opus
5.5's 12/12 at 20% of the cost (2026-09-28). Twelve tasks leave a loss of up
to 24 points inside the 95% interval, so it is a smoke test, not proof. On
review work Haiku does worse: `docs/quality.md` has it finding 67% of the
defects Opus finds. `adder outcomes import --write` records real escalations,
and they replace the 15% prior once there are enough.

## Method

`adder bench` replays every recorded turn under each configuration and re-prices
it. It is the same replay engine `adder plan` uses, and the same fidelity check
applies: the null configuration must reproduce the measured bill. It does, to
+0.0% here, and the report prints that residual first. The check is necessary,
not sufficient: with nothing changed the replay prices every turn at its
recorded cost by construction, so it proves the bookkeeping and says nothing
about whether a counterfactual is right. A benchmark whose baseline does not
reproduce reality cannot say anything about a ratio taken against it.

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

## What the 2.6x rests on

Three inputs, none of which a transcript can settle, all swept rather than
asserted:

| summary ratio | p_fail | handoff | vs no adder |
|---|---|---|---|
| 10% | 15% | 2,000 | 2.55x |
| 10% | 15% | 20,000 | 2.25x |
| 10% | 30% | 2,000 | 2.55x |
| 10% | 30% | 20,000 | 2.24x |
| 30% | 15% | 2,000 | 2.53x |
| 30% | 15% | 20,000 | 2.23x |
| 30% | 30% | 2,000 | 2.52x |
| 30% | 30% | 20,000 | **2.23x** |

*Handoff* is how many tokens a restarted session has to be told, and nothing in
a transcript records what a person needs to resume. It sets the floor of the
range: a 10x larger handoff costs about 0.3x of the multiple.

The *summary ratio* is what a delegated read hands back, and *p_fail* is how
often a delegated step has to be redone on the expensive model. Neither moves
the result by more than a few hundredths any more, because once a subagent is
priced with its own opening, few reads are worth delegating at all. That is the
finding behind the small hands-off number, not a flaw in the sweep.

## Why this is not `adder plan`

`plan` asks the optimiser's question — what is the cheapest way this workload
could have been run — and reaches a larger multiple by adding effort reduction, terseness,
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
