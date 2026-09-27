# The guard

Every other command in this repository reports on money that is already gone.
The `PreToolUse` hook runs while the decision is still reversible, which makes
it the one component whose failure is both silent and expensive: a guard that
has stopped guarding still lets every tool call succeed, so nothing looks wrong.

This page is how to run it and what it will and will not do to your tool calls.
[guard-internals.md](guard-internals.md) is the measurement behind each of those
choices, which is a separate read because none of it is needed to use the thing.

## Running it

```bash
adder auto on --full             # install it, and let it refuse
adder guard                      # installed? what it predicts, what it decided, what it cost
adder guard --install            # the settings.json block, printed for hand-merging
adder guard --learn              # re-derive the size model from your transcripts
adder guard --explain "cat big.py | head -20"
```

`adder auto on` is the supported way in. `--install` predates it and still
prints the block for anyone who would rather merge it themselves, or who is
writing the file from a dotfiles repository.

The hook it points at lives inside the package, at
`adder/decide/hooks/pretooluse_read_guard.py`. It used to live in `.claude/`,
which the wheel prunes, so for four releases the install snippet named a path
that existed only in a git checkout, and activation from a `pip install` wrote
three hooks pointing at nothing. `.claude/hooks/` still holds forwarding shims
so a `settings.json` written before the move keeps working.

The report leads with whether the hook is declared in any `settings.json` it
can see, because an uninstalled guard, a broken guard and a correctly quiet
guard are indistinguishable from the outside. `doctor` fails on it for the same
reason: it is the only finding there about money that has not been spent yet.

`--install` prints the block instead of writing it. `adder config --init` set
that precedent and it applies with more force here: a hook changes what every
session does, so it should be installed on purpose and not as a side effect of
running a report.

`adder guard --learn` is worth running once after install and occasionally
after; the model is cached in `~/.claude/.adder-sizes.json` and the hook only
ever reads it.

It is advisory by default; set `ADDER_GUARD_BLOCK=1` to escalate to a
confirmation prompt above the hard threshold, or `adder auto on` to let it
refuse. It never blocks silently.

## Refusing

Advice has an uptake term, and the uptake term is the weakest number in this
project: nothing in a transcript says whether a model changed course because of
an injected sentence, so the guard discounts everything it says by an assumed
0.5. A refusal has no such term. The call does not happen, the tokens are not
admitted, and the saving is whole.

`guard_enforce` decides how far that goes. It is `off` unless you turn it on.

| level | what it refuses |
|---|---|
| `off` | nothing (the historical behaviour) |
| `shadow` | nothing — it computes what `certain` would refuse, and records it |
| `certain` | a read whose content is already in this context, by `Read` or by `cat` |
| `full` | also a large read that has a strictly cheaper equal |

`certain` is the level that needs no argument. The file was read earlier in
this session and has not changed on disk, or this session wrote it. Either way
the content is in the context already, so the read buys no information at any
price. It does not matter which tool did the reading: a `cat` of a file a
`Read` admitted is the same duplicate, and one file is one entry in the
refuse-once ledger rather than one per tool. `full` is a weaker claim: it rests
on the horizon estimate and on a subagent actually returning a brief, which is
why it is a separate opt-in and why its message always names the cheaper call
rather than only saying no. [What each level is
worth](guard-internals.md#what-each-level-is-worth) replays all four over
34,144 recorded calls.

Three properties make refusing safe to ship, and each is a test in
`tests/decide/test_guard_enforce.py`:

It never refuses the same target twice. A guard cannot be argued with, and the
model has no way to tell it something it does not know, so if the same call is
issued again it goes through. The worst case of a wrong refusal is one wasted
turn.

It always names the way through. Every refusal carries its reason and either the
content the model already has or the bounded call to make instead.

It forgets what compaction drops. "Already in this context" stops being true the
moment the context is rebuilt, so the PreCompact hook clears the read and write
memory before the compaction that invalidates it. Without that, the guard would
refuse a read of something the model no longer has, which is the one way an
enforcing guard costs more than it saves.

## Substituting, instead of refusing

A refusal at `full` costs a turn. The guard says "read at most 116 lines of it
(`limit: 116`)"; the model agrees and issues the bounded call. The outcome is
the bounded read plus one round trip spent arriving at advice the guard had
already priced. At the context sizes where the guard fires, that turn
is not rounding error.

The harness has a seam that removes it. A `PreToolUse` hook may return
`updatedInput`, and the call runs with the arguments the hook substituted. So
`guard_narrow=true` stops asking for the bounded call and makes the call bounded:

```
[adder] Run bounded to 116 lines (limit=116, was unbounded); re-issue with a
larger limit if you need the rest. Unbounded this Read admits ~60,000 tok at
~$8.83 of carry; this way ~$0.21.
```

The call executes, the model is told what changed and how to undo it, and no
turn is spent negotiating.

### Why it is off by default

The field was verified against the shipped client, not the documentation,
because the docs do not state the three things that decide whether this is safe.
In the strings of Claude Code 2.1.238:

- `PreToolUse hook for <tool> returned updatedInput that failed schema
  validation:` confirms the field exists on this event and is checked against
  the tool's own input schema, so a rewrite has to be a complete, valid input.
- `updatedInput is missing or empty, falling back to original tool input`: an
  absent rewrite is a no-op, which is what makes declining the safe default.
- `Hook satisfied user interaction for <tool> via updatedInput, bypassing
  permission prompt`: **a rewrite travels with an approval.**

That last one is the whole reason this is opt-in. A substitution can suppress a
prompt the user would otherwise have seen, and the result being *smaller* than
what was asked for does not make it authorised. The harness overrides an
approval where a `deny` or `ask` rule covers the call, so the exposure is limited
to calls that would have prompted by default, still a decision belonging to the
person whose files they are.

It is also reachable **only where the guard was going to refuse outright**.
Turning it on therefore relaxes a denial; it can never permit something the
guard would have been silent about. That is asserted, not asserted-and-hoped:
`tests/decide/test_narrow.py` fails if a substitution appears at `off` or
`certain`.

### What may be rewritten, and why so little

`Read` gains a `limit`; `Grep` gains a `head_limit`. Both are read-only, and in
both cases the substitution is a strict subset of what was asked for: fewer
lines of the same file, fewer hits of the same pattern. The near-misses are more
instructive than the hits:

`Bash` is refused rather than rewritten. Piping through `head -50` is right as
advice and wrong as an edit: appending to a command this module did not write
can change its exit status, cut a `&&` chain, or truncate the input to a command
whose output was never the point. Rewriting somebody's shell is not a bounded
operation.

`Grep` is not switched to `files_with_matches`. That is the cheapest bounded
form and it changes the *kind* of answer rather than the amount. A truncation
the model can see the edge of is recoverable; a different question answered
silently is not.

`Glob` and `WebFetch` have no bounding parameter at all, so there is nothing to
substitute that would still validate.

Four more rules, each an assertion. It never widens a call the caller already
bounded more tightly than the price floor. It never narrows below a floor of
usefulness: hand the model four lines of a file it wanted whole and it simply
asks again, spending the turn this existed to save plus one. It adds only keys
the tool already accepts, because an invented key fails schema validation and
the hook then silently does nothing. And anything unexpected makes it decline
rather than raise: an optional path must not take a tool call down with it.

The saving is booked against **what actually ran**, the bounded read, not
against the whole read a refusal would have prevented. `Verdict.action` reports
`narrow` rather than `deny` for the same reason: reporting one as the other would
overstate what enforcement is worth.

## Shadow: measuring the trade before making it

The argument above has a hole in the middle of it, and the hole is the 0.5.
Everything advisory rests on it, and the case for enforcement is that
enforcement removes it — which is a good argument that still asks somebody to
hand a hook the authority to refuse a tool call on the strength of a number this
project calls its own weakest.

`adder auto on --shadow` closes that. It runs the entire `certain` decision,
records the refusal it would have made, and refuses nothing. There is no
message, so nothing is admitted to the context, nothing is discounted, and the
fire ceiling does not apply — a ceiling here would truncate the measurement at
`guard_max_fires` findings a session and still read as complete.

What makes it a measurement rather than a brochure is the second half. A shadow
refusal the session went round is recorded as a **contradiction**: the same
target asked for again, or a duplicate `Read` refusal followed by the file
arriving through the shell instead. Under real enforcement that is the
refuse-once escape hatch firing, which happens when the model had a reason the
guard could not see, and it costs a turn. So:

```
$ adder guard --shadow

  Shadow — what it would have refused, and did not
  ================================================
  sessions              14
  would have refused    91 calls
  worth                 $38.20   no uptake assumption: a refused call does not happen
  contradicted          6 of them (7%), 9 times
  realised, worst case  $35.62   every contradicted refusal written off whole
```

`realised` writes off every contradicted refusal whole rather than partially,
which is deliberately the harsh reading: a contradicted refusal did not merely
fail to save its tokens, it would have cost a turn. It is a lower bound, and a
lower bound is the only honest number to put next to an install command.

Compaction clears the shadow record along with the rest of the context memory.
After compaction the tokens a refusal called redundant are gone, so a later read
is the session recovering them rather than the guard having been wrong, and
counting it would libel the measurement.

## The thresholds enforcement runs at

`adder auto on --full` moves three settings, and the values are swept rather
than picked:

| floor | fires | gate | refusals | prevented | overhead | net | calls that parse a transcript |
|---|---|---|---|---|---|---|---|
| 2,000 | 15 | $0.25 | 278 | $166 | $3.17 | $182 | 1.4% |
| 800 | 60 | $0.25 | 757 | $283 | $9.80 | $303 | 15.7% |
| 800 | 60 | $0.10 | 2,242 | $490 | $25.40 | $487 | 8% |
| **800** | **200** | **$0.10** | **2,421** | **$517** | **$27.45** | **$513** | **8%** |
| 800 | 1,000 | $0.10 | 2,448 | $523 | $27.79 | $519 | 8% |
| 300 | 200 | $0.10 | 4,590 | $677 | $51.54 | $651 | 39% |

The $0.25 gate is not a threshold on this lever at all. It exists to stop the
guard *interrupting* over small change, and a refusal is not an interruption,
so under enforcement it comes down to $0.10 and finds $200 more. The 15-fire
ceiling was sized for a guard that talks; one that redirects can afford 200,
which is worth $26 and costs no latency. Past 200 the curve is flat.

The floor is the only real trade in the table, and it trades latency for money
instead of being a free choice: 300 finds $138 more and takes the share of
tool calls that stop to parse a transcript from 8% to 39%. 800 ships because a
hook people uninstall saves nothing, which is the same argument the guard
applies to its own sentences, and because that ratio is a property of one
workload.

Which is the whole reason `adder auto on --full --tune` exists: it re-runs the
sweep against your transcripts and writes the best point, preferring the
quieter setting whenever the noisier one is within 5% of it. Shipping a
measurement of one machine as everybody's default is the mistake this guard was
rewritten to stop making, and the only defence is making it re-derivable.

## Naming a model for the delegation

The guard sees every `Task`, which makes it the one thing in the tool that is
present at the moment a routing decision is actually taken. `adder policy` has
been able to answer "what should this run on" since the beginning; what it
lacked was a caller at that moment, because a router nobody invokes routes
nothing. So the same gate that prices what a subagent hands *back* also names
what it should run *on*:

```
[adder] Run this on route-t0 (claude-haiku-4-5) rather than claude-opus-5:
~$0.05 cheaper in expectation, including a 15% chance of having to redo it
(short read-only question).
```

The number is `policy.right_size`'s: every rung priced as `run(tier) + p_fail *
(redo on T2 + the turn that catches it)`, with `p_fail` measured from the
outcome log where there is history and a prior where there is not. The baseline
is the model this delegation would otherwise have run on (the session model
under Claude Code), not the top rung, because comparing against a rung nobody
was going to use quotes a saving nobody was going to make.

Five ways it stays quiet, and they matter more than the one way it speaks:

- The call already names a routed agent. `route-t1`, `Explore` and the rest
  carry a decision somebody made deliberately.
- The classifier abstained and nothing measured says otherwise. Moving *up* the
  ladder needs no evidence; moving down needs the outcome log to hold enough
  recent history at that rung. Cheapness alone never buys a downgrade.
- The two rungs are the same model spelled differently. `claude-opus-5` and
  `claude-opus-5[1m]` are one rung, and a "switch" between them would throw away
  a model-scoped cache to buy nothing.
- The sentence costs more than the switch saves. It is priced like everything
  else the guard says and discounted by `guard_advice_taken`, because this one
  is advice rather than a refusal.
- It has already said it this session. The ladder does not change between two
  `Task` calls.

It never refuses a delegation. The guard may refuse a read; refusing a `Task`
would be refusing the largest lever this tool argues for, on the strength of a
classifier that is deliberately wrong-shy rather than right. `guard_route=false`
turns the clause off entirely for anyone who would rather the guard only talked
about size.

What is deliberately not in that sentence is a cross-vendor model.
`adder pick` ranks ~500 catalogued models by price against LMArena Elo and
`adder policy` reports the cheaper ones as substitutes, but under Claude Code a
`Task` cannot be dispatched to Qwen; the harness pins subagents to the vendor.
Naming one at the moment somebody cannot act on it is how a router loses trust.
The arena signal reaches this decision through the ladder instead, which
`adder models ladder` diffs against the live catalog.

## What it costs to run

A `command` hook is a process per tool call, so its own latency is part of what
it costs. Measured on this machine, against a 32.5ms floor that is the Python
interpreter starting:

| path | before | after |
|---|---|---|
| a tool the guard has no opinion about | 74.7 ms | **43.7 ms** |
| a bounded `Bash` | 86.9 ms | 75.8 ms |
| a guarded `Read` | **2,136 ms** | **139 ms** |

The two-second read was `live.analyse` re-fitting the session-length
distribution over every transcript on the machine, on every call, and the
prompt-submit advisor had been paying the same thing on every prompt. Fixing it
turned out to be two defaults: `load_sessions` defaulted its parse cache off
while the `cache` setting defaulted it on, and the fitted horizon was never
cached at all. See the CHANGELOG entries.

Latency is not dollars. It matters here because a two-second hook is one people
uninstall, and an uninstalled guard saves nothing.

## What it keeps

The guard writes one small JSON file under `~/.claude`, and it holds
identities, never contents: read paths with their mtimes, written paths with a
timestamp, command *shapes* with running totals. `shape()` drops arguments, so
a command carrying a token or a password is reduced to `curl` before anything
reaches disk. `tests/decide/test_guard.py` asserts this instead of trusting
it.

It is pruned in every dimension — 400 paths, 400 shapes, 200 sessions, and
anything untouched for a fortnight — and deleting it mid-session costs nothing
but the memory: the guard degrades to the stateless behaviour it had before.

## Several agents in one tree

The duplicate rule asks whether a file is the same bytes the context already
holds, and it asks by mtime. In a tree where several agents are working at once
the mtime moves for reasons this session had no part in: another agent formats
the file, a build writes it back, a sibling worktree touches it. The guard sees
a changed file, correctly declines to call the read a duplicate, and the lever
silently reports less than it is worth.

It fails towards saying nothing, never towards a wrong refusal, which is exactly
why nothing would ever have surfaced it. `adder guard` counts how many sessions
have written the shared state file in the last fifteen minutes and, when more
than one has, says that the figure below it is a floor rather than a total.

## Looking at what it did

`adder guard --last` lists every finding and refusal from the most recent
recorded session — action, tool, kind, what it was about, size, worth.

It exists because of how a refusal actually gets turned off. Someone suspects
the guard blocked something they needed, has no way to look, and
`guard_enforce=off` is one line. Every other report here is an aggregate over
weeks; this one answers "what did it just do to me", which is the question being
asked at that moment. Identities only: a shape, never a command; a basename,
never a path — the same promise the fires log makes on the way in.

## Diagnosing silence

Every failure path in the hook returns 0 so the tool call proceeds, which is
the only acceptable behaviour — and it also means a genuine bug reads exactly
like "there was nothing to say". Set `ADDER_GUARD_DEBUG=1` to print tracebacks
to stderr, where Claude Code shows them without them ever reaching the model's
context.

`adder guard --explain "<command>"` answers the same question for one specific
call, including the reason it would stay quiet.
