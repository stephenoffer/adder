"""How big will this tool call's result be, asked *before* it runs.

Every other module here prices tokens that already exist. The PreToolUse guard
is the only component that decides about tokens that do not, and until this
module it did that by pattern matching: a list of substrings (`cat `, `find `,
`git log`) meant "verbose", and verbose meant a fabricated 15,000 tokens.

Both halves were wrong, and measurably so. On 222 local transcripts the calls
that matcher fires on produce a **median of 143 result tokens** and a p90 of
2,206 -- the constant overstated the median by 105x, and 89% of the calls it
fired on came in under the guard's own 2,000-token floor. Meanwhile the matcher
was silent on the eighteen largest Bash results in the corpus (109K tokens),
because `for f in ...; do cat $f; done` and `wc -l a.ts b.ts c.ts` contain none
of its substrings, and `sed -n '1,600p'` was waved through by a `-n ` entry in
the "already bounded" list that was meant to match `grep -n`.

So the pattern list decides nothing here. Two questions are separated instead:

* **Is the output bounded by construction?** That is a shell question with a
  real answer: it depends on the *last* stage of the pipeline. `cat big | head`
  is bounded; `head -100 big | grep x` is bounded; `cargo test | grep -v warn`
  is not, because a filter is not a limit.
* **If it is not bounded, how big is it likely to be?** That is an empirical
  question, and the transcripts answer it per command shape. `SizeModel` learns
  the distribution and the guard reads a quantile off it.

The shape key is deliberately coarse -- program names in pipeline order, plus
the flags that change the output's magnitude. Arguments are dropped, because
`cat src/a.ts` and `cat src/b.ts` are the same decision and splitting them
gives every shape a sample size of one.

Nothing here is a claim about *this* machine until `SizeModel.learn` has run.
`PRIOR` is the fallback, and it is a measurement (see `docs/guard.md`), not a
guess -- but it is a measurement of somebody else's workload, which is why
`adder guard --learn` exists and why the report says which of the two answered.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from adder.core.trace import DEFAULT_ROOT, transcripts
from adder.util.records import mapping
from adder.util.stats import quantile as _stats_quantile
from adder.util.text import CHARS_PER_TOKEN, est_tokens, flatten_text

# Where a learned model is cached. Small (a few KB), read by a PreToolUse hook
# on every call, so it is a flat JSON file and never a scan.
MODEL_VERSION = 1

# Both are resolved through `model_path()` and `max_age_s()` rather than read
# here, because a constant captured at import time is one no test can redirect
# and no `.adder.json` can override -- and this one names a file under the
# user's home directory, so an import-time capture means the suite reads and
# writes the developer's own model. `render.color_enabled` and
# `guard.Settings` made the same correction for the same reason.
#
# It was still captured at import, one layer down: the fallback below read a
# module constant, and `core.settings` held the same path in a `Setting`
# default built at import. `monkeypatch.setenv("HOME", ...)` moves neither, so
# a machine that had run `adder auto on` -- which learns a model into
# `~/.claude` -- ran its own suite against 40,902 of the developer's real tool
# calls. A function, so `Path.home()` is read when it is asked for.


def default_model_path() -> Path:
    """Where a learned model lives when nothing has been configured."""
    return Path.home() / ".claude" / ".adder-sizes.json"

# Re-learn if the cached model is older than this. A day is chosen so that a
# machine whose habits changed converges within a working week, without any
# command paying for a full transcript scan more than once a day.
DEFAULT_MAX_AGE_S = 86_400.0


def model_path() -> Path:
    """Where the learned size model lives. Resolved per call, never cached."""
    try:
        from adder.core.settings import get as _setting

        return Path(str(_setting("size_model")))
    except Exception:
        return Path(os.environ.get("ADDER_SIZE_MODEL") or default_model_path())


def max_age_s() -> float:
    """Seconds before a learned model is considered stale."""
    try:
        from adder.core.settings import get as _setting

        return float(_setting("size_max_age"))
    except Exception:
        return DEFAULT_MAX_AGE_S

# The fallback distribution, in result tokens, when nothing local is known.
#
# Derived, not chosen, and derived from the population the guard actually has
# to predict: **unbounded calls only** -- a `Bash` whose output is not capped by
# construction, a content-mode `Grep` with no `head_limit`. Including bounded
# calls would describe a different question than the one being asked.
#
# Measured over 222 transcripts on the author's machine:
#
#     tool        n        p50     p90     p99
#     Bash        16,456   103     603     2,359
#     Read           876    25   1,678     6,705
#     WebFetch       353   230     595     2,102
#     WebSearch      267   697     909     1,653
#     Agent           15   193   3,723     3,945
#     (pooled non-Bash/Read fallback: n=3,706, p50 42, p90 322)
#
# The first version of this table was invented rather than measured and was
# wrong in the expensive direction on every line -- `WebFetch` was quoted at
# 12,000 tokens p90 against a measured 595, a **20x** over-statement, and the
# generic fallback at 3,000 against a measured 322. A guard whose fallback
# over-states is a guard that interrupts constantly on a machine that has not
# learned anything yet, which is the exact failure this module was written to
# remove -- it had simply moved from the hook into the default.
#
# `Grep`, `Glob` and `Task` have no observations here at all, so they inherit
# the pooled fallback rather than a number someone liked the look of. The
# consequence is deliberate: below the guard's 2,000-token floor, nothing
# without local evidence fires. On a fresh machine only `Read` is guarded, and
# `Read` is sized off the filesystem rather than predicted. Everything else
# waits for `adder guard --learn`.
PRIOR: dict[str, tuple[int, int]] = {
    # tool or shape -> (p50, p90)
    "Bash": (103, 603),
    "Read": (25, 1_678),
    "Grep": (42, 322),
    "Glob": (42, 322),
    "WebFetch": (230, 595),
    "WebSearch": (697, 909),
    "Task": (193, 3_723),
    "Agent": (193, 3_723),
    "*": (42, 322),
}

# Where the numbers above came from, so `adder guard` can say how far the
# shipped prior is from this machine and a test can fail if the table drifts
# from its own documentation.
PRIOR_SOURCE = {
    "transcripts": 222,
    "population": "unbounded calls only",
    "observations": {"Bash": 16_456, "Read": 876, "WebFetch": 353,
                     "WebSearch": 267, "Agent": 15, "pooled": 3_706},
}

# Programs whose *final* position in a pipeline bounds what comes back. A
# filter (`grep -v`, `sed 's/x/y/'`) is not on this list on purpose: it changes
# the output, it does not cap it.
#
# `cut` was on this list and is a filter by that exact definition: it narrows
# each line to some columns and emits one line per input line, so `cat big.csv
# | cut -f1` returns as many rows as it was given. Output size is O(n) in the
# input, which is what everything else here is O(1) in. The three that remain
# each collapse an arbitrary input to a fixed-size answer.
BOUNDING: frozenset[str] = frozenset({"head", "tail", "wc"})

# Flags that bound a command's own output, keyed by program. Checked only on
# the last pipeline stage, for the same reason.
BOUNDING_FLAGS: dict[str, tuple[str, ...]] = {
    "grep": ("-c", "--count", "-l", "-L", "--files-with-matches", "-m", "--max-count"),
    "rg": ("-c", "--count", "-l", "--files-with-matches", "-m", "--max-count"),
    "git": ("--stat", "--shortstat", "--name-only", "--name-status", "--oneline"),
    "sed": (),      # `sed -n '1,200p'` is handled by the range detector below
    "ls": ("-d",),
    "find": ("-maxdepth",),
}

# Tokens per line of command output, measured over 16,727 local calls that
# carried an explicit line bound. The spread is the point: p10 is 2.2 and p90 is
# 35.6, because a "line" of minified JSON and a line of Python are not the same
# object. The guard reads the p90, since it is deciding about a tail.
#
# `lines * 11.4` predicts the real result to a median absolute error of 83
# tokens, which is the same accuracy the shape model reaches -- so a numeric
# bound is a *size*, and treating it as a free pass was the largest remaining
# error in this file. `sed -n '1,600p'` was waved through as "bounded" and
# returned 6,079 tokens; 45 supposedly-bounded calls in the corpus returned
# over 3,000.
TOKENS_PER_LINE = (11.4, 35.6)          # p50, p90

# The same measurement for a bounded `Read`, which is a different population:
# source files rather than command output, so the tail is far thinner. Measured
# over 133 bounded reads whose file is still on disk. The whole-file estimator
# was checked at the same time and needed no change -- `CHARS_PER_TOKEN = 4.0`
# predicts an unbounded read to a median absolute error of 265 tokens, and
# re-fitting it to the observed 5.1 made that worse, because files edited since
# they were read contaminate the sample in one direction only.
READ_TOKENS_PER_LINE = (10.9, 13.7)     # p50, p90

# `sed -n 'A,Bp'` and `awk 'NR<=N'` are bounded reads: the range is the bound.
_SED_RANGE = re.compile(r"-n\s*['\"]?\s*(\d+)\s*,\s*(\d+)\s*p")
_AWK_RANGE = re.compile(r"NR\s*<=?\s*(\d+)")
# `head -c 4000` bounds bytes rather than lines, which is a tighter statement
# than a line count and needs no tokens-per-line assumption at all.
_HEAD_BYTES = re.compile(r"\b(?:head|tail)\b[^|;&]*?-c\s*(\d+)")
_HEAD_TAIL_N = re.compile(r"\b(?:head|tail)\b(?:\s+-n)?\s+-?(\d+)")

# Two different operators, and conflating them is a bug this file had for one
# afternoon. `|` composes a pipeline: only its last stage reaches the context,
# so only the last stage's bound counts. `;`, `&&` and `||` run separate
# commands whose outputs *concatenate*, so a sequence is bounded only if every
# command in it is -- `git diff --stat; echo done` was passing as bounded
# because the sequence's last word was `echo`.
#
# Splitting is quote-aware rather than regex, which is not fastidiousness. A
# regex split produced 12,208 distinct shapes from 27,643 calls on this
# machine, because `grep -vE "^warning|^\s+-->"` was being cut in half at the
# alternation inside its own pattern and every regex became its own "program".
# Almost every shape then had a sample size of one, which is below the evidence
# floor, so the guard fell back to the shipped prior for nearly everything --
# the exact failure the size model was written to remove.
_SEQ_OPS = ("&&", "||", ";", "\n")
_PIPE_OPS = ("|",)

# `cmd <<'EOF' ... EOF` embeds a whole document in the command line. Its body is
# data, not a pipeline, and parsing it as one turned every heredoc into a shape
# named after whatever Python happened to be inside it.
_HEREDOC = re.compile(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?")


def _strip_heredocs(command: str) -> str:
    """Drop heredoc bodies, keeping the command that opened them.

    Coerces to `str` first, and that is the guarantee the rest of this parsing
    API rests on. A tool input is whatever the model emitted, not a contract:
    `{"command": 12}` reached the regexes as an int and raised `TypeError:
    expected string or bytes-like object` out of `shape()` and `is_bounded()`.
    In the PreToolUse guard that exception is swallowed by a blanket handler,
    so the symptom is not an error -- it is the guard silently declining to
    guard, which this package calls its worst failure mode because it is
    invisible. Fixed here rather than at the eight call sites, because this is
    where "untrusted text in" is promised.
    """
    command = str(command) if command is not None else ""
    m = _HEREDOC.search(command)
    if not m:
        return command or ""
    lines = command.split("\n")
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        opened = _HEREDOC.search(line)
        i += 1
        if opened:
            delim = opened.group(1)
            while i < len(lines) and lines[i].strip() != delim:
                i += 1
            i += 1                     # skip the closing delimiter too
    return "\n".join(out)


def _split_unquoted(text: str, ops: tuple[str, ...]) -> list[str]:
    """Split on `ops`, ignoring any that fall inside quotes or after a backslash.

    Hand-written rather than `shlex`, which raises on the unterminated quotes
    real transcripts contain -- and a parser that raises inside a PreToolUse
    hook is a guard that has silently stopped guarding.
    """
    parts: list[str] = []
    buf: list[str] = []
    quote = ""
    i = 0
    n = len(text or "")
    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n:
            buf.append(ch)
            buf.append(text[i + 1])
            i += 2
            continue
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        hit = next((op for op in ops if text.startswith(op, i)), None)
        if hit:
            parts.append("".join(buf))
            buf = []
            i += len(hit)
            continue
        buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts

# A leading `VAR=value` or a `cd somewhere` contributes nothing to output size.
_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

# Redirection is plumbing, not a program. `for f in a b; do cat $f; done 2>&1`
# was producing the shape `for|echo|cat|2>&1`, which splits one shape's sample
# across two keys purely on whether stderr was merged.
_REDIRECT = re.compile(r"^(?:[0-9]*[<>]{1,2}&?[0-9]*|&>>?)$")

# Shell keywords that wrap a real command; the interesting program is inside.
_WRAPPERS = frozenset({
    "cd", "then", "do", "done", "fi", "else", "elif", "if", "for", "while",
    "time", "env", "nohup", "exec", "sudo", "command", "builtin",
})


def _words(segment: str) -> list[str]:
    """Split a pipeline segment into words without dying on unbalanced quotes.

    `shlex.split` raises on `echo "unterminated`, which a real transcript
    contains, and a hook that raises is a hook that has failed open at best.
    """
    return [w for w in re.split(r"\s+", segment.strip()) if w]


def _head(segment: str) -> tuple[str, list[str]]:
    """The program a pipeline segment runs, and its arguments.

    Skips assignments and wrappers, so `cd ~/x && time sudo cat f` reports
    `cat`. Returns `("", [])` for a segment with nothing runnable in it.
    """
    words = _words(segment)
    i = 0
    while i < len(words):
        w = words[i]
        if _ASSIGN.match(w) or w in ("(", "{", "!"):
            i += 1
            continue
        if _REDIRECT.match(w) or w[0] in "<>":
            i += 1
            continue
        base = w.rsplit("/", 1)[-1]
        if base in _WRAPPERS:
            # `for f in a b c` — skip to the body, which is a later segment.
            if base in ("for", "while", "if"):
                return base, words[i + 1:]
            i += 1
            # `cd path &&` puts the real command in the next segment; but
            # `cd path; cat f` splits on `;` so this only skips the word.
            if base == "cd" and i < len(words):
                i += 1
            continue
        return base, words[i + 1:]
    return "", []


def segments(command: str) -> list[tuple[str, list[str]]]:
    """The (program, args) of each pipeline stage, in order, wrappers removed."""
    out = []
    for seg in _split_unquoted(_strip_heredocs(command), _SEQ_OPS + _PIPE_OPS):
        prog, args = _head(seg)
        if prog:
            out.append((prog, args))
    return out


def _redirects_to_file(text: str) -> bool:
    """Whether this command sends stdout to a file, ignoring quoted text.

    Scanned rather than matched for the same reason the splitter is: a `-->`
    inside `grep -vE "^warning|^\\s+-->"` is not a redirect, and reading it as
    one marked the command bounded and silenced the guard on it.

    `2>&1` and `>&2` are stderr plumbing, not an output file, so a digit before
    the `>` or an `&` after it both disqualify.
    """
    quote = ""
    i, n = 0, len(text or "")
    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n:
            i += 2
            continue
        if quote:
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            i += 1
            continue
        if ch == ">":
            before = text[i - 1] if i else ""
            j = i + 1
            if j < n and text[j] == ">":            # `>>` appends; still a file
                j += 1
            while j < n and text[j] == " ":
                j += 1
            after = text[j] if j < n else ""
            if not before.isdigit() and after and after != "&":
                return True
            i = j + 1
            continue
        i += 1
    return False


def bound_lines(command: str) -> int | None:
    """Lines the last stage of this command will emit, when it says so.

    `None` means no explicit number -- either unbounded, or bounded by
    something with no size in it (`wc -l`, `grep -c`, a redirect), all of which
    are small whatever the input.
    """
    pipes = pipelines(command)
    if not pipes:
        return None
    best: int | None = None
    for stages in pipes:
        prog, args = stages[-1]
        joined = " ".join(args)
        n = None
        if prog == "sed":
            m = _SED_RANGE.search(joined)
            if m:
                lo, hi = int(m.group(1)), int(m.group(2))
                n = max(0, hi - lo + 1)
        elif prog == "awk":
            m = _AWK_RANGE.search(joined)
            n = int(m.group(1)) if m else None
        elif prog in ("head", "tail"):
            byte = _HEAD_BYTES.search(f"{prog} {joined}")
            if byte:
                # Bytes, not lines: convert directly and skip the per-line
                # assumption, which is the weakest term in this estimate.
                n = max(1, int(int(byte.group(1)) / CHARS_PER_TOKEN
                               / TOKENS_PER_LINE[0]))
            else:
                m = _HEAD_TAIL_N.search(f"{prog} {joined}")
                n = int(m.group(1)) if m else None
        if n is None:
            continue
        # A sequence concatenates, so the sizes add.
        best = n if best is None else best + n
    return best


def bound_estimate(command: str) -> Estimate | None:
    """Size implied by an explicit line bound, or None if there is not one."""
    n = bound_lines(command)
    if not n:
        return None
    p50, p90 = TOKENS_PER_LINE
    return Estimate(int(n * p50), int(n * p90), 0, "bound")


def _stage_is_bounded(prog: str, args: list[str]) -> bool:
    joined = " ".join(args)
    if prog in BOUNDING:
        return True
    if prog == "sed" and _SED_RANGE.search(joined):
        return True
    if prog == "awk" and _AWK_RANGE.search(joined):
        return True
    for flag in BOUNDING_FLAGS.get(prog, ()):  # exact flag words, not substrings
        if flag in args or any(a.startswith(flag + "=") for a in args):
            return True
    # A short-form bundle like `grep -rln` contains a bounding letter.
    if prog in ("grep", "rg"):
        for a in args:
            if re.fullmatch(r"-[a-zA-Z]+", a) and ("l" in a[1:] or "c" in a[1:]):
                return True
    return False


def pipelines(command: str) -> list[list[tuple[str, list[str]]]]:
    """The command split into sequenced commands, each a list of pipeline stages."""
    out = []
    for part in _split_unquoted(_strip_heredocs(command), _SEQ_OPS):
        stages = []
        for seg in _split_unquoted(part, _PIPE_OPS):
            prog, args = _head(seg)
            if prog:
                stages.append((prog, args))
        if stages:
            out.append(stages)
    return out


def is_bounded(command: str) -> bool:
    """Whether the command's output is capped by construction.

    Within a pipeline the **last** stage decides; across a sequence **every**
    command must be bounded. The previous implementation searched the whole
    string for `head`, so `head -1 f && cat huge.log` counted as bounded, and
    searched it for `-n `, so every `sed -n '1,600p'` did too -- while a genuine
    bound at the end of a long pipeline was invisible whenever an earlier stage
    matched a "verbose" pattern.
    """
    parts = _split_unquoted(_strip_heredocs(command), _SEQ_OPS)
    pipes = pipelines(command)
    if not pipes:
        return False
    for raw, stages in zip((p for p in parts if _head(p)[0]), pipes, strict=False):
        # Redirecting stdout to a file admits nothing but stderr to the context.
        if _redirects_to_file(raw):
            continue
        prog, args = stages[-1]
        if not _stage_is_bounded(prog, args):
            return False
    return True


def shape(command: str) -> str:
    """A stable key for "commands like this one".

    Program names in pipeline order, with the arguments dropped and a `+n`
    suffix when a stage carries a numeric bound. Coarse on purpose: the shape
    exists to accumulate a sample, and a key that includes file paths has a
    sample size of one for every entry.
    """
    segs = segments(command)
    if not segs:
        return "?"
    parts = []
    for prog, args in segs:
        m = _SED_RANGE.search(" ".join(args)) if prog == "sed" else None
        if m:
            parts.append("sed+range")
        elif prog in ("head", "tail"):
            n = _HEAD_TAIL_N.search(f"{prog} {' '.join(args)}")
            parts.append(f"{prog}+{n.group(1)}" if n else prog)
        elif prog == "git" and args:
            parts.append(f"git {args[0]}")
        elif prog in ("npm", "npx", "cargo", "pytest", "python3", "python", "go"):
            sub = next((a for a in args if not a.startswith("-")), "")
            parts.append(f"{prog} {sub}".strip() if sub else prog)
        else:
            parts.append(prog)
    return "|".join(parts[:4])


def _guard_would_predict(tool: str, inp: dict | None) -> bool:
    """Whether this call is one the guard has to size, rather than wave through.

    The definition `PRIOR` is derived from. A bounded command and a
    `files_with_matches` grep are decided by their shape, never by a predicted
    size, so including them would describe a different question.
    """
    inp = inp or {}
    if tool == "Bash":
        return not is_bounded(inp.get("command") or "")
    if tool == "Grep":
        return ((inp.get("output_mode") or "files_with_matches") == "content"
                and not inp.get("head_limit"))
    if tool == "Read":
        return not inp.get("limit")
    return True


@dataclass(frozen=True)
class Estimate:
    """A predicted result size, and where the prediction came from.

    `source` is reported rather than hidden because the guard's message changes
    meaning with it: "commands like this one returned 6K tokens the last 40
    times" is evidence, and "commands like this one usually return a lot" is
    not.
    """

    p50: int
    p90: int
    n: int                  # observations behind it; 0 means the prior answered
    source: str             # "shape", "head", "stat", "prior", "declared"

    @property
    def measured(self) -> bool:
        """Whether this rests on evidence rather than on the shipped prior.

        `stat` counts. A file's size is read off the filesystem, which is a
        stronger measurement than any quantile over past calls -- it is the
        actual thing about to be admitted. Tying `measured` to `n > 0` alone
        made the report describe a byte count as "no local history", which is
        the one kind of error this project exists to avoid: labelling a
        measurement as a guess reads as hedging, and labelling a guess as a
        measurement is worse.
        """
        return self.n > 0 or self.source in ("stat", "bound", "image")

    def describe(self) -> str:
        if self.source == "stat":
            return f"~{self.p90:,} tok (the file's size on disk)"
        if self.source == "bound":
            return f"~{self.p90:,} tok (the line bound this command asks for)"
        if self.source == "image":
            return f"~{self.p90:,} tok (an image, billed by dimensions not bytes)"
        if self.n <= 0:
            return f"~{self.p90:,} tok (prior; no local history for this shape)"
        return f"~{self.p90:,} tok (p90 of {self.n:,} local call{'s' if self.n != 1 else ''})"


def _quantile(sorted_xs: list[int], q: float) -> int:
    """Interpolated quantile, rounded to a token count.

    Delegates to `stats.quantile` rather than indexing at
    `round(q * (n - 1))`. That was the fourth private copy of the estimator
    `stats` exists to replace -- its docstring names three others and calls the
    nearest-rank form "biased" -- and here it was worse than biased. Python
    rounds halves to even, so at n=6 `round(0.9 * 5)` is 4 rather than 5, and
    the p90 becomes the fifth of six samples instead of interpolating toward
    the sixth. On a heavy-tailed size distribution the sixth is the tail.

    Measured over this machine's 7,575 learned shapes, the p90 disagreed with
    the interpolated value by a mean of 56.7% on samples of 6-10 -- 122 of 271
    shapes off by more than 10%, the worst by 68x (14 tokens against 966). The
    guard reads exactly this number: `est.p90` decides whether a call is priced
    at all, and small samples are the common case, so the tail estimate was
    wrong precisely where the evidence is thinnest.
    """
    if not sorted_xs:
        return 0
    return round(_stats_quantile(sorted_xs, q))


@dataclass
class SizeModel:
    """Result-size quantiles per command shape, learned from local transcripts.

    Stored as quantiles rather than samples: the file is read by a hook on
    every tool call, and a hook that parses megabytes is a hook someone turns
    off. Three levels of backoff, because a new command shape must still get an
    answer -- exact shape, program head, then the shipped prior.
    """

    shapes: dict[str, tuple[int, int, int]] = field(default_factory=dict)   # p50,p90,n
    heads: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    tools: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    built: float = 0.0
    calls: int = 0

    # -- prediction ------------------------------------------------------

    def _lookup(self, table: dict, key: str, source: str, min_n: int) -> Estimate | None:
        row = table.get(key)
        if not row or row[2] < min_n:
            return None
        p50, p90, n = row
        return Estimate(int(p50), int(p90), int(n), source)

    def predict_command(self, command: str, *, min_n: int = 3) -> Estimate:
        """Expected result size of a shell command.

        `min_n` guards against a shape whose whole history is one outlier.
        Below it the estimate backs off to the program, then to the prior --
        which is the honest answer, not a confident one off a single sample.
        """
        segs = segments(command)
        est = self._lookup(self.shapes, shape(command), "shape", min_n)
        if est is None and segs:
            est = self._lookup(self.heads, segs[-1][0], "head", min_n)
        if est is None and segs:
            # The *first* stage often decides volume (`cat big | grep x`).
            est = self._lookup(self.heads, segs[0][0], "head", min_n)
        bound = bound_estimate(command)
        if est is None:
            # An explicit line bound beats the prior: it is a fact about this
            # call rather than a distribution over other people's.
            est = bound
        if est is None:
            p50, p90 = PRIOR.get("Bash", PRIOR["*"])
            est = Estimate(p50, p90, 0, "prior")
        elif bound is not None and bound.p90 < est.p90:
            # A bound *caps* a learned estimate, it does not merely stand in for
            # one. `cat huge.log | head -50` inherits `cat`'s history through
            # the program backoff, and `cat` here has returned 40K tokens -- but
            # this call cannot, because fifty lines is fifty lines.
            est = Estimate(min(est.p50, bound.p50), bound.p90, est.n, "bound")
        return est

    def predict_tool(self, tool: str, inp: dict, *, min_n: int = 3) -> Estimate:
        """Expected result size of any tool call, before it runs.

        `Read` is the one tool whose answer is not statistical: the file is on
        disk and can be measured. Everything else falls back to history.
        """
        inp = inp or {}
        if tool == "Read":
            return read_estimate(inp)
        if tool == "Bash":
            return self.predict_command(inp.get("command") or "", min_n=min_n)
        est = self._lookup(self.tools, tool, "shape", min_n)
        if est is not None:
            return est
        p50, p90 = PRIOR.get(tool, PRIOR["*"])
        return Estimate(p50, p90, 0, "prior")

    # -- learning --------------------------------------------------------

    @classmethod
    def learn(cls, root: Path | str = DEFAULT_ROOT, *, window=None) -> SizeModel:
        """Build a model from every tool result under `root`.

        Bounded calls are learned too. Excluding them would bias every shape
        upward, and the guard needs to know that `git diff | head -50` is small
        so that it stays quiet about it.
        """
        by_shape: dict[str, list[int]] = {}
        by_head: dict[str, list[int]] = {}
        by_tool: dict[str, list[int]] = {}
        calls = 0
        for tool, inp, size in iter_results(root, window=window):
            calls += 1
            # `by_tool` is the population `PRIOR` describes, so it has to mean
            # the same thing: calls the guard would actually have to predict.
            # Counting bounded ones here made the report compare a prior
            # derived from unbounded calls against an average over all of them,
            # and the two disagreed for reasons that had nothing to do with the
            # machine being different.
            if _guard_would_predict(tool, inp):
                by_tool.setdefault(tool, []).append(size)
            if tool != "Bash":
                continue
            cmd = (inp or {}).get("command") or ""
            by_shape.setdefault(shape(cmd), []).append(size)
            segs = segments(cmd)
            if segs:
                by_head.setdefault(segs[-1][0], []).append(size)
                if segs[0][0] != segs[-1][0]:
                    by_head.setdefault(segs[0][0], []).append(size)

        def summarize(d: dict[str, list[int]]) -> dict[str, tuple[int, int, int]]:
            out = {}
            for k, xs in d.items():
                xs.sort()
                out[k] = (_quantile(xs, 0.5), _quantile(xs, 0.9), len(xs))
            return out

        return cls(shapes=summarize(by_shape), heads=summarize(by_head),
                   tools=summarize(by_tool), built=time.time(), calls=calls)

    # -- persistence -----------------------------------------------------

    def to_json(self) -> dict:
        return {
            "version": MODEL_VERSION,
            "built": self.built,
            "calls": self.calls,
            "shapes": {k: list(v) for k, v in self.shapes.items()},
            "heads": {k: list(v) for k, v in self.heads.items()},
            "tools": {k: list(v) for k, v in self.tools.items()},
        }

    @classmethod
    def from_json(cls, d: dict) -> SizeModel:
        if not isinstance(d, dict) or d.get("version") != MODEL_VERSION:
            raise ValueError("size model: unrecognised version")

        def rows(x) -> dict[str, tuple[int, int, int]]:
            out = {}
            for k, v in (x or {}).items():
                if isinstance(v, (list, tuple)) and len(v) == 3:
                    out[str(k)] = (int(v[0]), int(v[1]), int(v[2]))
            return out

        return cls(shapes=rows(d.get("shapes")), heads=rows(d.get("heads")),
                   tools=rows(d.get("tools")), built=float(d.get("built") or 0.0),
                   calls=int(d.get("calls") or 0))

    def save(self, path: Path | None = None) -> Path:
        path = Path(path) if path is not None else model_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Unique per writer. Several Claude Code sessions share one machine
        # and run this from a hook, so a fixed `.tmp` name is a shared
        # mutable path: one writer's `replace` moves the file out from
        # under another's, and the loser raises FileNotFoundError into an
        # `except OSError` that drops it. Measured at 45% of writes lost
        # under three concurrent writers. `trace._cache_store` already
        # carries the pid for exactly this reason.
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(self.to_json()), encoding="utf-8")
            tmp.replace(path)                  # atomic: a hook may be reading it
        finally:
            tmp.unlink(missing_ok=True)
        return path

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.built) if self.built else float("inf")


def empty_model() -> SizeModel:
    """A model that answers entirely from the prior. Never raises, never reads."""
    return SizeModel()


def load_model(path: Path | None = None) -> SizeModel:
    """The cached model, or a prior-only one. Never raises.

    Called from a PreToolUse hook, so every failure mode -- missing file, bad
    JSON, a version from a future release -- has to degrade to "use the prior"
    rather than to an exception. A guard that crashes is a guard that has
    stopped guarding.
    """
    try:
        target = Path(path) if path is not None else model_path()
        return SizeModel.from_json(json.loads(target.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return empty_model()


# A file whose bytes are not text. Sizing these as `bytes / 4` is not an
# approximation, it is a category error: an image is billed by its dimensions,
# not its file size, and the API caps a single image near 1,600 tokens however
# many megabytes it is on disk. Left unfixed, the guard's replay ranked its
# eight largest findings as duplicate reads of PNG screenshots worth $25-$31
# each -- a confident wrong number, which is the exact failure mode this
# project exists to avoid.
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
                            ".svg", ".ico", ".tiff", ".heic", ".avif"})

# p50/p90 for one image. The ceiling is the API's own: a 1568px image is about
# 1,600 tokens, and anything larger is downscaled to fit it.
IMAGE_TOKENS = (750, 1_600)

# Read either refuses these or converts them, and in neither case is the byte
# count the token count.
OPAQUE_SUFFIXES = frozenset({".zip", ".gz", ".tar", ".pdf", ".woff", ".woff2",
                             ".ttf", ".otf", ".mp4", ".mov", ".mp3", ".wav",
                             ".so", ".dylib", ".o", ".a", ".class", ".pyc",
                             ".wasm", ".db", ".sqlite"})


def read_estimate(inp: dict) -> Estimate:
    """Size of a `Read`, measured off the filesystem rather than guessed.

    A bounded read (`limit`) is charged for the lines it asked for, not the
    file it asked them from -- the previous guard returned zero for any read
    with a limit, which made `limit: 100000` invisible to it.
    """
    fp = (inp or {}).get("file_path")
    if not fp:
        return Estimate(0, 0, 0, "declared")
    suffix = Path(str(fp)).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return Estimate(IMAGE_TOKENS[0], IMAGE_TOKENS[1], 0, "image")
    if suffix in OPAQUE_SUFFIXES:
        # No honest estimate is available, and a guess would be priced. The
        # guard says nothing rather than something it cannot support.
        return Estimate(0, 0, 0, "declared")
    try:
        # `str(fp)` as three lines above: a tool input is whatever the model
        # emitted, and `Path(12)` raises TypeError, which `except OSError` does
        # not catch. The guard runs in a PreToolUse hook, where an uncaught
        # exception is a guard that has silently stopped guarding.
        size = Path(str(fp)).stat().st_size
    except (OSError, ValueError):
        return Estimate(0, 0, 0, "declared")
    whole = int(size / CHARS_PER_TOKEN)
    offset = (inp or {}).get("offset")
    if offset:
        # A read from an offset with no limit runs to the end of the file, so
        # the whole-file size over-states it by everything already skipped.
        with contextlib.suppress(TypeError, ValueError, OverflowError):
            whole = max(0, whole - int(int(offset) * READ_TOKENS_PER_LINE[0]))
    limit = (inp or {}).get("limit")
    if limit:
        try:
            n = int(limit)
        except (TypeError, ValueError, OverflowError):
            # OverflowError is `int(inf)`. It is not a ValueError, and leaving
            # it out is how a non-finite tool input reaches a hook uncaught.
            return Estimate(whole, whole, 0, "stat")
        # Measured over 133 bounded reads whose file is still on disk: 10.9
        # tokens per requested line at the median, 13.7 at p90. Carried as a
        # spread rather than a point for the same reason every other estimate
        # here is -- the guard reads the p90.
        lo, hi = (int(n * k) for k in READ_TOKENS_PER_LINE)
        return Estimate(min(whole, lo), min(whole, hi), 0, "stat")
    return Estimate(whole, whole, 0, "stat")


def iter_results(root: Path | str = DEFAULT_ROOT, *,
                 window=None) -> Iterable[tuple[str, dict, int]]:
    """Yield `(tool, input, result_tokens)` for every answered tool call.

    The pairing is by `tool_use_id`, which can cross records and files, so the
    pending map lives for the whole scan. Unanswered calls are dropped: a call
    the session never got a reply to has no size, and imputing one would put a
    zero into a distribution the guard reads a p90 off.
    """
    pending: dict[str, tuple[str, dict]] = {}
    seen_use: set[str] = set()
    answered: set[str] = set()
    for path in transcripts(root):
        try:
            fh = path.open(encoding="utf-8", errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                if '"tool_use"' not in line and '"tool_result"' not in line:
                    continue
                try:
                    d = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if window is not None and not window.keeps_record(d, path.parent.name):
                    continue
                content = mapping(d, "message").get("content")
                if not isinstance(content, list):
                    continue
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "tool_use":
                        uid = str(b.get("id") or "")
                        if uid in seen_use:
                            continue
                        seen_use.add(uid)
                        pending[uid] = (str(b.get("name") or "?"), b.get("input") or {})
                    elif b.get("type") == "tool_result":
                        uid = str(b.get("tool_use_id") or "")
                        if not uid or uid in answered:
                            continue
                        answered.add(uid)
                        got = pending.pop(uid, None)
                        if got is None:
                            continue
                        tool, inp = got
                        yield tool, inp, est_tokens(flatten_text(b.get("content")))


def iter_calls(root: Path | str = DEFAULT_ROOT, *,
               window=None) -> Iterable[tuple[str, str, str, dict, str]]:
    """Yield `(session, model, tool, input, timestamp)` for every call, in order.

    `iter_results` pairs a call with its answer and is the right shape for
    learning sizes. This one is the right shape for *replaying a decision*: it
    keeps the order and the session, which is what a guard's state depends on,
    and it does not need the result -- the guard never sees one.
    """
    for path in transcripts(root):
        try:
            fh = path.open(encoding="utf-8", errors="replace")
        except OSError:
            continue
        seen: set[str] = set()
        model = ""
        with fh:
            for line in fh:
                if '"tool_use"' not in line:
                    continue
                try:
                    d = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if window is not None and not window.keeps_record(d, path.parent.name):
                    continue
                if d.get("type") != "assistant":
                    continue
                msg = mapping(d, "message")
                model = msg.get("model") or model
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                session = str(d.get("sessionId") or path.stem)
                for b in content:
                    if not isinstance(b, dict) or b.get("type") != "tool_use":
                        continue
                    uid = str(b.get("id") or "")
                    if uid:
                        if uid in seen:
                            continue
                        seen.add(uid)
                    yield (session, model, str(b.get("name") or "?"),
                           b.get("input") or {}, d.get("timestamp") or "")


def refresh(root: Path | str = DEFAULT_ROOT, *, path: Path | None = None,
            max_age: float | None = None, force: bool = False) -> SizeModel:
    """Return a model, re-learning it only when the cache is stale.

    Never called from the hook's hot path. `adder guard --learn` calls it, and
    so does `adder doctor`, so a machine that runs either occasionally keeps a
    current model without any single tool call paying for a scan.
    """
    path = Path(path) if path is not None else model_path()
    max_age = max_age_s() if max_age is None else max_age
    if not force:
        cached = load_model(path)
        if cached.calls and cached.age_s < max_age:
            return cached
    model = SizeModel.learn(root)
    with contextlib.suppress(OSError):
        model.save(path)
    return model
