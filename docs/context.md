# The context you did not write

Four of the reports in this tool answer questions about the same object from
different sides: the tokens sitting in the window that nobody chose to put
there this turn. They are collected here because each one is misleading on its
own.

| Question | Command | What it prices |
|---|---|---|
| What is in the window before I say anything? | `adder memory` | instruction files, the memory index, skill and agent descriptions |
| What did I put in twice? | `adder reread` | content admitted again that was already resident |
| Should I throw the window away? | `adder compact` | compaction, priced against the carry it avoids |
| What may I carry when I do? | `adder handoff` | the largest brief that leaves a restart ahead |

## 1. The floor is not free, it is free *per session*

`adder prefix` measures that a session opening is ~74% cache read, and the
correct conclusion — restarts are cheap — was over-read into a wrong one: that
the size of the floor does not matter.

A floor token is not read once per session. It is read once per **turn**. On
this machine, at the measured re-read multiplier and the cost-weighted median
session length:

```
1,000 resident tokens = $0.31 per session
                      = $5.59 across the 18 sessions this project has on record
                      = $32.64 in a user-level file, which all 105 sessions load
```

That is the editing unit, and **scope decides which count applies**: a project
`CLAUDE.md` is resident only in that project's sessions, a `~/.claude/CLAUDE.md`
in every one. This repo's 2,200-token instruction file has cost **$12.48** over
the 18 sessions it has been loaded in; the same text in the user-level file
would have cost **$72**. Nothing else in a context has this shape, because everything else can be
compacted away and instruction files cannot: compaction rebuilds the prefix
from the same file, so the survival term is 1.0 forever.

Two consequences that are not obvious from file sizes:

* **A skill library is nearly free and an instruction file is not.** Only a
  skill's `name` and `description` are resident; its body loads when it runs.
  A 40,000-token skill collection costs less than a 3,000-token `CLAUDE.md`.
* **Most of the floor is not yours.** Of a 30K opening context here, 2.7K is
  files on disk. The other 28K is the system prompt and tool schemas, and
  `adder memory` reports it as `unaccounted` rather than attributing it to a
  file somebody is about to edit.

## 2. The second copy buys nothing

A file read on turn 8 and read again on turn 140 was still in the window the
whole time. The second copy is not a second purchase of the file — it is a
purchase of *nothing*, plus its own carry to the end of the session.

`adder reread` separates two cases that look identical in a transcript and are
not:

* **redundant** — the result is byte-identical to a copy already resident.
  Recoverable in full.
* **refresh** — the result changed. The call was justified; the superseded copy
  is still resident and still being re-read, but skipping the call would have
  been wrong.

Reporting them together would tell someone their test runs are waste. Only the
first is offered as a saving. On this machine that is **$6.88 across 31
identities**, against **199K tokens** of superseded copies that no call could
have avoided.

### When "write it down" is wrong

An identity read in many different sessions is not a re-read — each session
started empty. That is the one case where the fix is memory, and it has a
price, because a resident note is re-read on every turn of *every* session
while the read is paid only in the sessions that make it.

`adder reread` prints a **note budget**: the largest resident note that still
beats re-reading the thing. On this machine it comes out at 4–46 tokens for
files read in two sessions. Writing a 5,000-token file summary into `CLAUDE.md`
to avoid reading it twice loses money by two orders of magnitude — which is the
opposite of the usual advice, and the reason the number is printed.

## 3. Compaction is a trade, and it has a threshold

A compaction pays a rebuild and buys a smaller prefix on every remaining turn:

```
cost   = kept_tokens  * r * write_mult
saving = freed_tokens * r * read_mult * remaining_turns
```

so it is worth doing exactly when

```
remaining_turns  >  kept * write_mult / (freed * read_mult)
```

**Compact when more turns remain than that, not when the bar looks full.** The
threshold is small — a few dozen turns at the measured multipliers — which
means the common failure is not compacting too often. It is carrying a full
context for hundreds of turns because compaction felt destructive.

Measured here: 9 compactions, median survival **6%** (not the 35% the model
conservatively assumes), net **+$1,857**. Against that, 18 sessions carried a
near-full context and never compacted at all — **$718** of avoidable carry.

The missed-compaction figure is simulated turn by turn against what actually
happened, because a compacted context *refills*. Pricing freed tokens as a
constant saving over 348 turns invents money: the un-compacted session is
pinned against the ceiling and cannot grow, the compacted one regrows into the
gap, and the gap closes. `Miss.saving` models that; the naive version was 16%
higher.

**What is not priced anywhere here: what compaction deletes.** A detail that
has to be re-derived is paid for twice, and `adder reread` is where that bill
shows up. The verdict is therefore a bound — below the threshold a compaction
is *certainly* a loss; above it, a gain *if* nothing important was dropped.

## 4. Restarting is cheaper than compacting, and the brief is not the constraint

Compaction writes back everything it keeps. A restart writes only the handoff,
because the expensive part of a new session's floor is identical to the old
one's and is still cache-resident. At a 600K context that is an order of
magnitude apart, and `adder live` will say so:

```
Context hygiene: restart — worth ~$55 over the ~350 turns expected to remain.
  compacting instead: ~$35.
```

The objection is always the same: *I would lose the context.* `adder handoff`
answers it with the crossing point — the brief size at which the restart stops
being ahead:

```
(C - H) * r * m * R  =  floor_cost + H * r * w
```

At a 500K context with 300 turns left that is **467,000 tokens**. The budget is
not binding; it never was. Nobody writes a 467,000-token brief, so the real
constraint on a handoff is what you can usefully say, not what you can afford
to carry — and the tool says so in those words rather than printing a budget
someone might try to fill.

The budget goes *negative* near the end of a session. That is not "write a
shorter brief", it is "do not restart", and it is the case a fixed rule of
thumb gets wrong in the expensive direction.

## What an agent should do with this

In rough order of how much it is worth per unit of effort:

1. **Bound what enters.** The guard prices a read before it lands
   (`adder guard`); a summary from a subagent costs a tenth of the file.
2. **Never admit the same thing twice.** The first copy never left.
3. **Reset at boundaries, and carry a brief.** Restart beats compaction at a
   large context; the brief budget is almost never the limiting factor.
4. **Keep the instruction file short and the skill bodies long.** Resident
   tokens are re-read forever; on-demand tokens are not.
