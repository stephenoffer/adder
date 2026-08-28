# The guard: the only thing here that prevents spend

Every other command in this repository reports on money that is already gone.
The PreToolUse hook runs while the decision is still reversible, which makes it
the one component whose failure is both silent and expensive. A guard that has
stopped guarding still lets every tool call succeed, so nothing looks wrong.

That asymmetry is why the decision moved out of the hook file and into
`adder/decide/guard.py`, and why the size estimate moved into
`adder/core/shapes.py`. Both are now under test. What follows is the
measurement that forced the rewrite.

## What was wrong

The guard fired when a call was predicted to admit at least 2,000 tokens and
cost at least $0.25 to carry. The prediction came from a list of substrings:

```python
_VERBOSE = ("cat ", "find ", "ls -R", "git log", "git diff", "npm ls",
            "pip list", "curl ", "grep -r", "rg ")
ASSUMED_BASH_TOKENS = 15_000
```

Measured against 222 local transcripts (27,698 answered tool calls, 23,228 of
them `Bash`):

| | value |
|---|---|
| assumed result size | 15,000 tok |
| measured median of the calls it fired on | **143 tok** |
| measured mean | 699 tok |
| measured p90 | 2,206 tok |
| measured max | 7,453 tok |
| share of its fires whose real output was under its own 2,000-token floor | **89%** |
| share of all Bash result tokens it saw at all | 9.7% |
| of the 18 largest results in the corpus, how many it matched | **0** |

So it interrupted 903 times about reads that were never going to be expensive,
and stayed silent on `for f in ...; do cat $f; done`, `wc -l a.ts b.tsx c.tsx`
and `git diff --stat`, none of which contain any of its substrings.

Two of its "already bounded" entries were also wrong in a way that mattered.
`-n ` was in the list to catch `grep -n`; it also matched every
`sed -n '1,600p'`, which is how the second-largest result in the corpus was
waved through. And the list was searched against the whole command string, so
`head -1 f && cat huge.log` counted as bounded.

## What replaced it

**Bounding is a shell question with a real answer.** Within a pipeline the last
stage decides (`cat huge | head` is small however big `huge` is); across a
sequence every command must be bounded (`git diff --stat; echo done` is not,
whatever its last word is). A filter is not a limit: `grep -v warning` changes
the output without capping it.

The parser is quote-aware and hand-written instead of `shlex`, which raises on
the unterminated quotes real transcripts contain, and a parser that raises
inside a PreToolUse hook is a guard that has silently stopped guarding.

**Size is measured, not assumed.** `SizeModel` learns result-size quantiles per
command shape from local transcripts. A shape is program names in pipeline
order with arguments dropped, so `cat src/a.ts` and `cat lib/b.py` accumulate
one sample rather than two singletons. Prediction backs off exact shape →
program → shipped prior, and refuses to quote a shape with fewer than three
observations as evidence.

Holdout (even calls train, odd calls test, 23,228 Bash calls):

| | learned model | the 15,000 constant |
|---|---|---|
| median absolute error vs the real result size | **68 tok** | 14,867 tok |
| p90 coverage (share of real sizes at or below the predicted p90) | 84.5% | 100%¹ |
| calls it would fire on | **146** | 530 |
| median real size of those calls | **1,390 tok** | 151 tok |
| of the 27 results over 5,000 tokens, how many it flags | **17** | 15 |

¹ A prediction of 15,000 covers everything because nothing in the corpus is
that large. Coverage alone is not a quality measure; it is bought with the
error in the first row.

One third the interruptions, and more of the large calls caught.

Quoting cost measurable accuracy on its own. Splitting the command with a
regex cut `grep -vE "^warning|^\s+-->"` in half at the alternation inside its
own pattern, producing 12,208 distinct shapes from 27,643 calls: almost all
singletons, all below the evidence floor, so the guard fell back to the prior
for nearly everything. Quote-aware splitting brings that to 7,027 shapes over
167 programs, and moves 2,500 decisions per holdout half from the program
backoff onto the specific shape.

## The blind spot a per-call rule cannot have

A guard that judges one call at a time cannot see a habit. Measured across the
same 222 transcripts:

- 32 session-and-shape pairs exceed 20,000 cumulative result tokens.
- Together they are **19.7% of every Bash result token in the corpus**, 1.38M
  of 7.0M.
- The largest is `sed -n 'A,Bp'`: **246 calls, 513 tokens each, 126,222 tokens**
  into a single session.

Every one of those calls is a bounded read, and the guard is right to say
nothing about any of them. It is the two hundred and forty-sixth that is the
problem, and only the running total shows it.

Aggregated by shape rather than by session-and-shape, the gap is starker and is
checked by `adder validate`: the shapes that clear the threshold cumulatively
hold **47% of all Bash result tokens**, against **4%** in calls large enough
for a per-call gate to see. A machine that only ever makes a few big calls will
fail that claim, and should: there the aggregate rule is not earning its
state.

So the guard counts what each command shape has admitted, bounded calls
included, since those are the ones that add up, and says so once when the total
is worth more than saying it. The saving is booked at half the carry:
tokens already admitted cannot be un-admitted, and only the calls still to come
can be avoided. Claiming the whole total would be claiming a refund.

## A bound that names a number is a size

`is_bounded` answers "is this capped by construction", and for a while the
guard treated a `yes` as a reason to say nothing. That was wrong for every
bound that carries a number. `sed -n '1,600p'` is bounded to six hundred
lines, which is about six thousand tokens, and it was waved through and
returned 6,079. Across the corpus, **45 supposedly-bounded calls returned over
3,000 tokens**, and the largest of them were `sed` ranges.

So a numeric bound is now read as an estimate. Measured over 16,727 local calls
carrying an explicit line bound, output runs **11.4 tokens per line at the
median and 35.6 at p90**. The spread is the point, because a line of minified
JSON and a line of Python are not the same object, and the guard is deciding
about a tail. `lines × 11.4` predicts the real result to a median absolute
error of 83 tokens, which is the accuracy the shape model reaches.

A bound also **caps** a learned estimate instead of merely standing in for
one. `cat huge.log | head -50` inherits `cat`'s history through the program
backoff, and `cat` may well have returned 40K tokens before, but fifty lines
is fifty lines. Capping is one-directional: a generous bound is not evidence
that this call will be large.

What is left of the structural rule is the bounds with no number in them
(`wc -l`, `grep -c`, a redirect to a file), which are small whatever the input.

## The prior

`PRIOR` in `adder/core/shapes.py` is the fallback when nothing local is known.
It is a measurement, not a guess, but it is a measurement of one machine's
workload, which is why `adder guard --learn` exists and why every estimate
reports whether it came from local evidence or from the prior. The report
prints both side by side so the gap is visible, not assumed away.

## The guard now charges for its own advice

A fire injects `additionalContext` into the conversation. That text is admitted
to the context exactly like a tool result: written once, re-read on every
remaining turn. The old guard fired 903 times and never counted it.

`decide` prices its own message and refuses to speak unless

```
saving x P(advice is taken)  >  cost of carrying the message
```

`P(advice is taken)` is an assumption, not a measurement: nothing in a
transcript says whether a model changed course because of an injected sentence.
It defaults to 0.5, so the guard needs a 2x margin over its own overhead, and it
is `guard_advice_taken` in `.adder.json` for anyone who wants to test a
different value. Setting it to 0 silences the guard entirely, which is the
correct behaviour for someone who believes advice is never acted on.

Three further limits follow from the same accounting: at most one fire per
command shape per session, at most 15 fires per session, and a running
per-session ledger of promised saving against advice cost that `adder guard`
prints.

## The saving that needed no model at all

**19.2% of unbounded `Read` calls on text files re-read something already in
the context** (44 of 229). The guard could not see it because it had no memory
between calls.

That number was first quoted as 27.4% over all unbounded reads, and the
correction is worth keeping visible. 138 of the 182 duplicates in this corpus
are screenshots, and an image is capped near 1,600 tokens whatever its file
size, so re-reading one is cents, not dollars. The headline was true and
misleading, which for a measurement tool is the same as wrong.

It is the cheapest saving in this project: nothing to delegate, no horizon to
forecast, no trade-off. The tokens are already in the context, so re-reading
buys no information at all. `GuardState` remembers each read path with the
file's mtime, so a re-read after an edit (which is the correct thing to do)
is not flagged, and only an unchanged re-read is.

There is a second way for a file's content to already be in the context, and it
is worth separating: **a file this session wrote**. A `Write` puts the whole
content in the context as the tool call's own input, so reading it back admits
every one of those tokens a second time. The guard watches `Write` without ever
advising on it, since admitting a write costs nothing when the content is the
input, purely so it can catch the read back later.

`Edit` is deliberately not watched. An edit puts a *hunk* in the context, not a
file, so re-reading an edited file can be the only way to see the rest of it.
Counting that as waste would mean advising against the correct move.

### When the harness reads with `cat`

All of the above was keyed on `Read`'s `file_path`, which is a second way for
the guard to see nothing. Under `bypassPermissions` — how agent harnesses run
unattended — the guidance routes file access to the shell, so `cat`, `sed -n`
and `grep` do the reading and `file_path` is never populated. On one 8-session
corpus (2,313 turns, $369 as run) the rule reported **0 identities, $0.00**
while 25.8% of every Bash result token in it — 314,771 tokens — was a path the
session had already read. Apportioning that corpus's $53.07 of measured Bash
carry by token share puts it near $13.70; that is an apportionment, not an
independent pricing run, so read it as an order of magnitude.

Zero and "this instrumentation cannot observe the reads on this machine" printed
identically, and the first is the one that gets believed. `adder/core/reads.py`
closes it: it names the files a call read, for `Read` and `Bash` alike, so which
tool the harness picked stops deciding whether the saving exists.

Two rules carry over unchanged, and one is new:

- **A bounded read is a slice, not a file.** `sed -n '1,50p' f`, `head -20 f`
  and `grep pat f` admit part of `f`, so none of them may record `f` as
  resident — exactly as a `Read` with a `limit` does not. A slice *of a file
  already held whole* may still be refused, because those lines are
  demonstrably already there.
- **Every ambiguity resolves towards admitting less.** No glob, no variable, no
  `cd`, no redirect, and a whole-file claim only from a single pipeline stage,
  because `cat f | grep x` admits matches rather than a file. Missing a path
  costs a saving; inventing one costs a refusal of a read that was needed.
- **The harness truncates shell output.** A `cat` of a file larger than
  `BASH_MAX_OUTPUT_LENGTH` (30,000 characters by default) returns a truncated
  result, so the file is *not* in the context. Without that check the guard
  would refuse the read that would have got the rest.

The refusal is written in the language the caller is working in. A model reading
with `cat` cannot act on advice about `limit:`, and telling it to use `Read`
instead is advice about how the harness is configured rather than about the
call in front of it.

`adder reread` measures the same family after the fact and more thoroughly,
comparing result digests to separate a genuine duplicate from a refresh. It
cannot see the read-after-write case, because a write's result is
`"file written"` rather than the content, which is why the guard, seeing the
tool *input*, is where that one belongs.

## The step before a delegation

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

- **The call already names a routed agent.** `route-t1`, `Explore` and the rest
  carry a decision somebody made deliberately.
- **The classifier abstained and nothing measured says otherwise.** Moving *up*
  the ladder needs no evidence; moving down needs the outcome log to hold enough
  recent history at that rung. Cheapness alone never buys a downgrade.
- **The two rungs are the same model spelled differently.** `claude-opus-5` and
  `claude-opus-5[1m]` are one rung, and "switch" between them would throw away a
  model-scoped cache to buy nothing.
- **The sentence costs more than the switch saves.** Priced like every other
  thing the guard says, and discounted by `guard_advice_taken`, because this one
  is advice, not a refusal.
- **It has already said it this session.** The ladder does not change between
  two `Task` calls.

It never refuses a delegation. The guard may refuse a read; refusing a `Task`
would be refusing the largest lever this tool argues for, on the strength of a
classifier that is deliberately wrong-shy rather than right. `guard_route=false`
turns the clause off entirely for anyone who would rather the guard only talked
about size.

**What is deliberately not in that sentence: a cross-vendor model.**
`adder pick` ranks ~500 catalogued models by price against LMArena Elo and
`adder policy` reports the cheaper ones as substitutes, but under Claude Code a
`Task` cannot be dispatched to Qwen; the harness pins subagents to the vendor.
Naming one at the moment somebody cannot act on it is how a router loses trust.
The arena signal reaches this decision through the ladder instead, which
`adder models ladder` diffs against the live catalog.

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

It is advisory by default; set
`ADDER_GUARD_BLOCK=1` to escalate to a confirmation prompt above the hard
threshold, or `adder auto on` to let it refuse. It never blocks silently.

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

`certain` is the level that needs no argument. The file was read earlier in this
session and has not changed on disk, or this session wrote it. Either way the
content is in the context already, so the read buys no information at any price.
It does not matter which tool did the reading: a `cat` of a file a `Read`
admitted is the same duplicate, and one file is one entry in the refuse-once
ledger rather than one per tool.
`full` is a weaker claim: it rests on the horizon estimate and on a subagent
actually returning a brief, which is why it is a separate opt-in and why its
message always names the cheaper call rather than only saying no.

Three properties make refusing safe to ship, and each is a test in
`tests/decide/test_guard_enforce.py`:

- **It never refuses the same target twice.** A guard cannot be argued with, and
  the model has no way to tell it something it does not know. If the same call
  is issued again it goes through. The worst case of a wrong refusal is one
  wasted turn.
- **It always names the way through.** Every refusal carries its reason and
  either the content the model already has or the bounded call to make instead.
- **It forgets what compaction drops.** "Already in this context" stops being
  true the moment the context is rebuilt, so the PreCompact hook clears the
  read and write memory before the compaction it is invalidated by. Without
  that, the guard would refuse a read of something the model no longer has,
  which is the one way an enforcing guard costs more than it saves.

## Substituting, instead of refusing

A refusal at `full` costs a turn. The guard says "read at most 116 lines of it
(`limit: 116`)", the model reads that, agrees, and issues the bounded call, so
the outcome is the bounded read plus one round trip spent arriving at advice the
guard had already priced. At the context sizes where the guard fires, that turn
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

- **`Bash` is refused, not rewritten.** Piping through `head -50` is right as
  advice and wrong as an edit: appending to a command this module did not write
  can change its exit status, cut a `&&` chain, or truncate the input to a
  command whose output was never the point. Rewriting somebody's shell is not a
  bounded operation.
- **`Grep` is not switched to `files_with_matches`.** That is the cheapest
  bounded form and it changes the *kind* of answer, not the amount. A truncation
  the model can see the edge of is recoverable; a different question answered
  silently is not.
- **`Glob` and `WebFetch` have no bounding parameter**, so there is nothing to
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

### Shadow: measuring the trade before making it

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

### What the levels are worth

`adder guard --replay` over 34,144 recorded tool calls on the author's machine,
at the thresholds each level ships with:

| level | fires | of which refusals | prevented | argued for | cost | net | rests on the assumption |
|---|---|---|---|---|---|---|---|
| `off` | 265 | 0 | — | $96.28 | $2.81 | $93.47 | 100% |
| `certain` | 315 | 52 | $13.13 | $95.56 | $3.19 | $105.50 | 88% |
| `full`, advisory thresholds | 315 | 278 | $166.00 | $19.13 | $3.17 | $181.96 | 10% |
| **`full`, as shipped** | **2,554** | **2,356** | **$516.65** | **$23.36** | **$26.97** | **$513.04** | **4%** |

The last column is the one to read. `full` roughly doubles the net, but the
more important change is that nine tenths of it stops depending on whether
anybody takes the advice.

### The thresholds enforcement runs at

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

## What it would have done here

`adder guard --replay` runs the guard over transcripts that have already been
paid for. On this machine, 29,464 tool calls across 80 sessions:

| | |
|---|---|
| times it would speak | **236** (0.80% of calls) |
| calls costing a transcript parse | 1.44% |
| findings | 206 size, 15 subagent brief, 13 aggregate, 2 duplicate |
| worth, at 50% assumed uptake | **$85.40** |
| cost of saying it | $2.58 |

It is an upper bound and is labelled as one: the horizon is the one the guard
would have projected rather than the turns that really remained, the saving
assumes the advice is acted on, and a call it talked someone out of would have
changed everything after it.

Writing this replay was worth it before it ever ran on someone else's data. Its
first output ranked the eight largest findings as duplicate reads of PNG
screenshots worth $25–$31 each, and put the guard's value at $1,053. All of it
was one bug: `read_estimate` sized every file as `bytes / 4`, and an image is
billed by its dimensions, capped near 1,600 tokens however many megabytes it
is on disk. A 1MB screenshot was being priced at 250,000 tokens. The corrected
number is twelve times smaller and is the one worth having.

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

## The last assumption, made measurable

`guard_advice_taken` is the one number the solvency gate rests on, and nothing
measured it: no transcript says whether a model changed course because of an
injected sentence. It can be *estimated*, though, and the estimate uses only
formats this project writes itself.

Each fire is appended to `~/.claude/adder-guard-fires.jsonl`: a shape, never a
command; a basename, never a path. `adder guard` then asks the transcript what
happened next. For a command finding: were later calls of that program, in that
session, bounded more often than earlier ones? For a duplicate read: was the
file read again? Both are observable, and neither proves causation; the model
may have bounded the next call for its own reasons.

One detail cost a rewrite. The first version matched later calls on the full
command *shape*, which can never see the improvement it is measuring: `cat f`
is what was advised about and `cat f | head -20` is what compliance looks like,
and those are different shapes. It matches on the leading program instead.

Below ten judged findings the report says so and the assumption stands.

### And then acted on

For a long time that was where it stopped. `validate` and `doctor` switched to
the measured rate when reporting, and **the gate did not**: every advisory
saving in the tool was still multiplied by a flat 0.5 on machines that had
measured their own rate and printed it. A measurement nobody acts on is the same
failure as a router nobody invokes, and it is the failure this project keeps
finding in itself.

The gate now reads it. `adder guard --learn` measures and caches the rate to
`~/.claude/.adder-uptake.json`; the hook reads that cached number and not
re-scanning transcripts before every tool call, which is the same split the size
model already uses for the same reason. An explicitly configured
`guard_advice_taken` still wins: somebody who wrote a number into
`.adder.json` has said something the estimator does not know, and overriding it
silently is how a setting becomes decorative.

**There is a floor at 10%, and it is not caution.** `advice_taken` gates whether
advice is worth saying at all, so a measured rate near zero stops the guard
speaking, and a guard that does not speak records no fires, so nothing can ever
re-measure it. Without a floor the estimator seals itself shut on one bad week,
with no way back that does not involve editing a config file nobody knows
exists. The floor is what keeps that loop open, and the report says when it is
holding one up.

`adder guard` prints the number with its provenance attached, because an
unmeasured 50% shown as a bare percentage reads as a finding:

```
uptake   50% of advice acted on — ASSUMED; run `adder guard --learn` to measure it
```

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

## Per-tool floors, and why one number was two

`guard_min_tokens` is an I/O gate rather than a judgement: below it the guard
returns before parsing anything, so a call below it is invisible to every rule
that needs a price. `guard_max_fires` is an interruption budget. Both were
single numbers shared by every tool the guard watches, and the tools are not
alike. Measured on one machine:

| tool | calls in a session | p90 result | against a 2,000 floor |
|---|---|---|---|
| `Bash` | 2,490 | 1.2K tok | almost never priced |
| `Read` | 58 | 5.9K tok | routinely priced |

The ceiling has the same shape: `Bash` can spend all fifteen fires before `Read`
has said anything. `guard_min_tokens_by_tool` and `guard_max_fires_by_tool`
override per tool, written `Bash=800,Read=6000`. Both ship empty, so nothing
changes until something is set, and a per-tool ceiling can only lower the global
one — it exists to stop one loud tool starving the others, not to raise the
total.

Nothing per-tool is shipped, because a table would be one machine's workload
asserted as everyone's, which is the mistake the size prior already made here
once. `adder guard --floors` derives it instead: each tool's own distribution,
what the current floor prices, and the floor that would price its top decile —
which is that tool's p90, by the definition of p90.

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
