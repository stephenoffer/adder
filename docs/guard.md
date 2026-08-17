# The guard: the only thing here that prevents spend

Every other command in this repository reports on money that is already gone.
The PreToolUse hook runs while the decision is still reversible, which makes it
the one component whose failure is both silent and expensive — a guard that has
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

The parser is quote-aware and hand-written rather than `shlex`, which raises on
the unterminated quotes real transcripts contain — and a parser that raises
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
own pattern, producing 12,208 distinct shapes from 27,643 calls — almost all
singletons, all below the evidence floor, so the guard fell back to the prior
for nearly everything. Quote-aware splitting brings that to 7,027 shapes over
167 programs, and moves 2,500 decisions per holdout half from the program
backoff onto the specific shape.

## The blind spot a per-call rule cannot have

A guard that judges one call at a time cannot see a habit. Measured across the
same 222 transcripts:

- 32 session-and-shape pairs exceed 20,000 cumulative result tokens.
- Together they are **19.7% of every Bash result token in the corpus** — 1.38M
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
fail that claim, and should — there the aggregate rule is not earning its
state.

So the guard counts what each command shape has admitted — bounded calls
included, since those are the ones that add up — and says so once when the
total is worth more than saying it. The saving is booked at half the carry:
tokens already admitted cannot be un-admitted, and only the calls still to come
can be avoided. Claiming the whole total would be claiming a refund.

## A bound that names a number is a size

`is_bounded` answers "is this capped by construction", and for a while the
guard treated a `yes` as a reason to say nothing. That was wrong for every
bound that carries a number. `sed -n '1,600p'` is bounded — to six hundred
lines, which is about six thousand tokens — and it was waved through and
returned 6,079. Across the corpus, **45 supposedly-bounded calls returned over
3,000 tokens**, and the largest of them were `sed` ranges.

So a numeric bound is now read as an estimate. Measured over 16,727 local calls
carrying an explicit line bound, output runs **11.4 tokens per line at the
median and 35.6 at p90** — the spread is the point, because a line of minified
JSON and a line of Python are not the same object, and the guard is deciding
about a tail. `lines × 11.4` predicts the real result to a median absolute
error of 83 tokens, which is the accuracy the shape model reaches.

A bound also **caps** a learned estimate rather than merely standing in for
one. `cat huge.log | head -50` inherits `cat`'s history through the program
backoff, and `cat` may well have returned 40K tokens before — but fifty lines
is fifty lines. Capping is one-directional: a generous bound is not evidence
that this call will be large.

What is left of the structural rule is the bounds with no number in them —
`wc -l`, `grep -c`, a redirect to a file — which are small whatever the input.

## The prior

`PRIOR` in `adder/core/shapes.py` is the fallback when nothing local is known.
It is a measurement, not a guess — but it is a measurement of one machine's
workload, which is why `adder guard --learn` exists and why every estimate
reports whether it came from local evidence or from the prior. The report
prints both side by side so the gap is visible rather than assumed away.

## The guard now charges for its own advice

A fire injects `additionalContext` into the conversation. That text is admitted
to the context exactly like a tool result: written once, re-read on every
remaining turn. The old guard fired 903 times and never counted it.

`decide` prices its own message and refuses to speak unless

```
saving x P(advice is taken)  >  cost of carrying the message
```

`P(advice is taken)` is an assumption, not a measurement — nothing in a
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
size — so re-reading one is cents, not dollars. The headline was true and
misleading, which for a measurement tool is the same as wrong.

It is the cheapest saving in this project: nothing to delegate, no horizon to
forecast, no trade-off. The tokens are already in the context, so re-reading
buys no information at all. `GuardState` remembers each read path with the
file's mtime, so a re-read after an edit — which is the correct thing to do —
is not flagged, and only an unchanged re-read is.

There is a second way for a file's content to already be in the context, and it
is worth separating: **a file this session wrote**. A `Write` puts the whole
content in the context as the tool call's own input, so reading it back admits
every one of those tokens a second time. The guard watches `Write` without ever
advising on it — admitting a write costs nothing, since the content is the
input — purely so it can catch the read back later.

`Edit` is deliberately not watched. An edit puts a *hunk* in the context, not a
file, so re-reading an edited file can be the only way to see the rest of it.
Counting that as waste would mean advising against the correct move.

`adder reread` measures the same family after the fact and more thoroughly,
comparing result digests to separate a genuine duplicate from a refresh. It
cannot see the read-after-write case, because a write's result is
`"file written"` rather than the content — which is why the guard, which sees
the tool *input*, is where that one belongs.

## Running it

```bash
adder guard                      # installed? what it predicts, what it decided, what it cost
adder guard --install            # the settings.json block to merge
adder guard --learn              # re-derive the size model from your transcripts
adder guard --explain "cat big.py | head -20"
```

The report leads with whether the hook is declared in any `settings.json` it
can see, because an uninstalled guard, a broken guard and a correctly quiet
guard are indistinguishable from the outside. `doctor` fails on it for the same
reason: it is the only finding there about money that has not been spent yet.

`--install` prints the block rather than writing it. `adder config --init` set
that precedent and it applies with more force here — a hook changes what every
session does, so it should be installed on purpose and not as a side effect of
running a report.

`adder guard --learn` is worth running once after install and occasionally
after; the model is cached in `~/.claude/.adder-sizes.json` and the hook only
ever reads it.

It is advisory by default; set
`ADDER_GUARD_BLOCK=1` to escalate to a confirmation prompt above the hard
threshold. It never blocks silently, and it never denies.

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
billed by its dimensions — capped near 1,600 tokens however many megabytes it
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
distribution over every transcript on the machine, on every call — and the
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

Each fire is appended to `~/.claude/adder-guard-fires.jsonl` — a shape, never a
command; a basename, never a path. `adder guard` then asks the transcript what
happened next. For a command finding: were later calls of that program, in that
session, bounded more often than earlier ones? For a duplicate read: was the
file read again? Both are observable, and neither proves causation — the model
may have bounded the next call for its own reasons.

One detail cost a rewrite. The first version matched later calls on the full
command *shape*, which can never see the improvement it is measuring: `cat f`
is what was advised about and `cat f | head -20` is what compliance looks like,
and those are different shapes. It matches on the leading program instead.

Below ten judged findings the report says so and the assumption stands. Above
it, `validate` and `doctor` switch to the measured rate on their own, so the
solvency claim gets stronger or weaker on evidence rather than staying anchored
to a default nobody checked.

## What it keeps

The guard writes one small JSON file under `~/.claude`, and it holds
identities, never contents: read paths with their mtimes, written paths with a
timestamp, command *shapes* with running totals. `shape()` drops arguments, so
a command carrying a token or a password is reduced to `curl` before anything
reaches disk. `tests/decide/test_guard.py` asserts this rather than trusting
it.

It is pruned in every dimension — 400 paths, 400 shapes, 200 sessions, and
anything untouched for a fortnight — and deleting it mid-session costs nothing
but the memory: the guard degrades to the stateless behaviour it had before.

## Diagnosing silence

Every failure path in the hook returns 0 so the tool call proceeds, which is
the only acceptable behaviour — and it also means a genuine bug reads exactly
like "there was nothing to say". Set `ADDER_GUARD_DEBUG=1` to print tracebacks
to stderr, where Claude Code shows them without them ever reaching the model's
context.

`adder guard --explain "<command>"` answers the same question for one specific
call, including the reason it would stay quiet.
