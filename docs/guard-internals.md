# Why the guard is built this way

The guard's job is to refuse a tool call before it admits tokens you will pay
for on every remaining turn. Getting that wrong in either direction is
expensive, so almost every rule in it replaced an earlier rule that measurement
showed was wrong. This page is that record.

[guard.md](guard.md) is the page you want if you just need to run it.

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

### Bounding is a shell question with a real answer

Within a pipeline the last stage decides (`cat huge | head` is small however big `huge` is); across a
sequence every command must be bounded (`git diff --stat; echo done` is not,
whatever its last word is). A filter is not a limit: `grep -v warning` changes
the output without capping it.

The parser is quote-aware and hand-written instead of `shlex`, which raises on
the unterminated quotes real transcripts contain, and a parser that raises
inside a PreToolUse hook is a guard that has silently stopped guarding.

### Size is measured, not assumed

`SizeModel` learns result-size quantiles per command shape from local
transcripts. A shape is program names in pipeline order with arguments dropped,
so `cat src/a.ts` and `cat lib/b.py` accumulate one sample rather than two
singletons. Prediction backs off exact shape → program → shipped prior, and
refuses to quote a shape with fewer than three observations as evidence.

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

A bounded read is a slice, not a file. `sed -n '1,50p' f`, `head -20 f` and
`grep pat f` admit part of `f`, so none of them may record `f` as resident,
exactly as a `Read` with a `limit` does not. A slice *of a file already held
whole* may still be refused, because those lines are demonstrably already there.

Every ambiguity resolves towards admitting less. No glob, no variable, no `cd`,
no redirect, and a whole-file claim only from a single pipeline stage, because
`cat f | grep x` admits matches rather than a file. Missing a path costs a
saving; inventing one costs a refusal of a read that was needed.

And the harness truncates shell output. A `cat` of a file larger than
`BASH_MAX_OUTPUT_LENGTH` (30,000 characters by default) returns a truncated
result, so the file is *not* in the context. Without that check the guard would
refuse the read that would have got the rest.

The refusal is written in the language the caller is working in. A model reading
with `cat` cannot act on advice about `limit:`, and telling it to use `Read`
instead is advice about how the harness is configured rather than about the
call in front of it.

`adder reread` measures the same family after the fact and more thoroughly,
comparing result digests to separate a genuine duplicate from a refresh. It
cannot see the read-after-write case, because a write's result is
`"file written"` rather than the content, which is why the guard, seeing the
tool *input*, is where that one belongs.

## What each level is worth

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
the measured rate when reporting, and the gate did not: every advisory
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

There is a floor at 10%, and it is not caution. `advice_taken` gates whether
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

