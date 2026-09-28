"""Which files a call read, as opposed to how much it returned.

`shapes` answers "how big will this result be". This module answers the other
question the guard needs and never had: **which file did this call put in the
context, and did it put all of it there.**

Both were answered from `Read`'s `file_path`, which works right up until the
harness stops using `Read`. Under `bypassPermissions` -- the mode agent
harnesses run unattended in -- the guidance routes file access to the shell, so
`cat`, `sed -n` and `grep` do the reading and `file_path` is never populated.
The duplicate-read rule, the one saving in this project that needs no model and
no forecast, then has nothing to match on. It reported zero re-reads, which
prints identically to "there were none". On one 8-session corpus what printed
as $0.00 was 25.8% of every Bash result token: content the session had already
read. A cost tool may say it cannot see. It may not say zero when it means
that.

Whole file or a slice
---------------------
The distinction is load-bearing, and it is the same one `guard.observe`
already makes for a `Read` with a `limit`. `cat f` admits the file. `sed -n
'1,50p' f`, `head -20 f` and `grep pat f` admit *part* of it, and recording
those as "f is in the context" would make the guard refuse the one call that
would have got the rest. So a target carries `whole`, and only a whole read may
ever justify a refusal.

Conservative on purpose
-----------------------
A path is extracted only where the shell is unambiguous about it: no glob, no
variable, no `cd` in the command (which moves the directory the path is
relative to), no redirect (which sends the output to a file instead of the
context), and -- for a whole-file claim -- a single pipeline stage, because
`cat f | grep x` admits matches rather than a file. Every ambiguity resolves
towards admitting less.

The two mistakes are not symmetric. Missing a path costs a saving that was
available. Inventing one costs a refusal of a read that was needed, in the
component that is the only thing here able to change what happens.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from adder.core.shapes import pipelines

# Programs that put a *whole* named file into the context, and programs that
# put only some of one there. Membership is about what reaches the context, not
# about what the program opens: `wc -l f` reads all of `f` and admits a number,
# so it is in neither set.
WHOLE_FILE_READERS: frozenset[str] = frozenset({"cat", "bat", "nl", "tac"})
PARTIAL_FILE_READERS: frozenset[str] = frozenset({
    "head", "tail", "sed", "awk", "grep", "rg", "cut", "jq", "less", "more",
    "diff",
})

# Claude Code truncates a Bash result before it reaches the context, so a file
# larger than the ceiling was *not* fully admitted however whole-file the
# command was. Without this a `cat` of a 400KB log would teach the guard that
# the log is in the context, and the next read of it -- the one that would have
# got the rest -- is the one that gets refused.
#
# Read from the environment because a hook runs inside the harness that set it;
# the default is the harness's own default.
DEFAULT_MAX_BASH_OUTPUT_CHARS = 30_000

# `cd` changes what a relative path means, and this module resolves relative
# paths against one directory. Rather than model the walk, a command that moves
# is not read at all -- the failure direction is a saving not taken.
_MOVES = re.compile(r"(?:^|[;&|]|\n)\s*(?:cd|pushd|popd)\b")

# A word that is plumbing rather than an argument: `>`, `>>`, `2>&1`, `<file`.
_REDIRECT_WORD = re.compile(r"^(?:[0-9]*[<>]{1,2}&?[0-9]*|&>>?)$")

# Glob, variable, brace, subshell, escape. Any of them and the word on the
# command line is not the path that was opened.
_UNSAFE = re.compile(r"[*?\[\]{}$`\\!\n]")

# Flags that swallow the word after them. Only the ones belonging to the
# programs above, because that is the only place this is consulted -- a general
# table would be a claim about every program in POSIX.
_FLAG_TAKES_VALUE: dict[str, frozenset[str]] = {
    "grep": frozenset({"-e", "-f", "-m", "-A", "-B", "-C", "-d", "--regexp",
                       "--file", "--max-count", "--include", "--exclude"}),
    "rg": frozenset({"-e", "-f", "-m", "-A", "-B", "-C", "-g", "-t",
                     "--regexp", "--file", "--max-count", "--glob", "--type"}),
    "head": frozenset({"-n", "-c", "--lines", "--bytes"}),
    "tail": frozenset({"-n", "-c", "--lines", "--bytes"}),
    "sed": frozenset({"-e", "-f", "-i", "--expression", "--file"}),
    "awk": frozenset({"-v", "-f", "--file"}),
    "cut": frozenset({"-d", "-f", "-b", "-c", "--delimiter", "--fields"}),
    "nl": frozenset({"-w", "-s", "-v", "-i", "-b"}),
    "jq": frozenset({"--arg", "--argjson", "--slurpfile", "--rawfile"}),
}

# Programs whose first positional word is a program of their own -- a pattern,
# a script, a filter -- and not a file. `sed -n '1,50p' f` names one file, not
# two, and counting `1,50p` as a path is how a parser starts inventing them.
_SCRIPT_FIRST: frozenset[str] = frozenset({"grep", "rg", "sed", "awk", "jq"})

# ...unless the script was supplied by flag instead, in which case every
# positional really is a file.
_SCRIPT_BY_FLAG: tuple[str, ...] = ("-e", "-f", "--regexp", "--file",
                                    "--expression")

# Flags that turn a reader into a writer.
_EDITS_IN_PLACE: dict[str, tuple[str, ...]] = {"sed": ("-i", "--in-place")}


def max_bash_output_chars() -> int:
    """Characters a Bash result carries inline before the harness spills it.

    Past the inline ceiling a result arrives as a file path and a 2,000
    character preview, so the file is not in the context. `BASH_MAX_OUTPUT_LENGTH`
    only sizes the read-back window and "doesn't raise the inline ceilings"
    (tools reference), yet it was read as the ceiling: with it at 100,000 a
    `cat` of a 49K file was recorded as read in full and the Read after it
    refused. It may lower the figure used here, never raise it. The setting
    that really raises it, `bashOutputMaxChars`, is not read, which errs
    toward claiming less is in the context than is.
    """
    try:
        n = int(str(os.environ.get("BASH_MAX_OUTPUT_LENGTH")))
    except (TypeError, ValueError):
        return DEFAULT_MAX_BASH_OUTPUT_CHARS
    return min(n, DEFAULT_MAX_BASH_OUTPUT_CHARS) if n > 0 else DEFAULT_MAX_BASH_OUTPUT_CHARS


@dataclass(frozen=True, order=True)
class ReadTarget:
    """One file a call read.

    `whole` is a statement about the **command**, not about the context: the
    call asked for the entire file rather than a slice. That is necessary for a
    refusal and it is not sufficient, because the harness truncates a shell
    result -- so `cat` of a file above the ceiling asked for all of it and
    admitted the first 30,000 characters.

    `whole_reads` is the only correct consumer for that purpose: it applies the
    ceiling and stats the file, and the guard calls it. This docstring
    previously said `whole` meant "the entire file reached the context", which
    is a claim `tool_targets` has no way to make -- it does not touch the
    filesystem, and for `reread`, replaying a transcript written on another
    day, today's file size would answer a question about the past.

    `order=True` because a caller reaching for the richer API will sort a list
    of these, and a frozen dataclass without it raises `TypeError` when they
    do.
    """

    path: str
    whole: bool


def _unquote(word: str) -> str:
    if len(word) >= 2 and word[0] == word[-1] and word[0] in ("'", '"'):
        return word[1:-1]
    return word


def _is_pathlike(word: str) -> bool:
    """Whether this word can only be a filename by the time the shell ran it."""
    if not word or word in ("-", ".", ".."):
        return False
    if _REDIRECT_WORD.match(word) or word[0] in "<>&|":
        return False
    if "'" in word or '"' in word:
        # A leftover quote means the splitter cut a quoted argument in half --
        # `grep -n 'def foo' f` splits into `'def` and `foo'`, and the second
        # half is not a file. The splitter is whitespace-based on purpose
        # (`shapes._words`: real transcripts contain unbalanced quotes and a
        # hook may not raise), so the check belongs here.
        return False
    return not _UNSAFE.search(word)


def _file_args(prog: str, args: list[str]) -> list[str]:
    """The words in `args` that name files, for a program in the sets above."""
    if any(a.startswith(f) for a in args for f in _EDITS_IN_PLACE.get(prog, ())):
        # `sed -i` is an edit. It admits nothing and it changes the file, so
        # reporting it as a read is wrong twice over.
        return []
    takes = _FLAG_TAKES_VALUE.get(prog, frozenset())
    skip_script = prog in _SCRIPT_FIRST and not any(
        a in _SCRIPT_BY_FLAG or a.startswith(tuple(f + "=" for f in _SCRIPT_BY_FLAG))
        for a in args
    )
    out: list[str] = []
    i, only_files = 0, False
    while i < len(args):
        a = args[i]
        if a == "--" and not only_files:
            only_files = True
            i += 1
            continue
        if not only_files and a.startswith("-") and a != "-":
            i += 2 if a in takes else 1
            continue
        if skip_script:
            skip_script = False
            i += 1
            continue
        word = _unquote(a)
        if _is_pathlike(word):
            out.append(word)
        i += 1
    return out


def read_targets(command: str) -> list[ReadTarget]:
    """Every file this shell command reads, and whether it read all of it.

    Paths come back exactly as the command wrote them -- relative stays
    relative. Resolving them needs a directory this module has no opinion
    about; `resolve` takes one.
    """
    if not command or _MOVES.search(command):
        return []
    whole: dict[str, bool] = {}
    for stages in pipelines(command):
        if any(_REDIRECT_WORD.match(a) or a.startswith(">")
               for _, args in stages for a in args):
            # Output goes to a file, so nothing here reached the context. Also
            # the shape of `cat > f <<'EOF'`, which is a write.
            continue
        for prog, args in stages:
            if prog in WHOLE_FILE_READERS:
                # A later stage filters what the reader emitted, so only a lone
                # stage can claim the whole file. `cat f | grep x` admits
                # matching lines, exactly like `grep x f` does.
                entire = len(stages) == 1
            elif prog in PARTIAL_FILE_READERS:
                entire = False
            else:
                continue
            for p in _file_args(prog, args):
                whole[p] = whole.get(p, False) or entire
    return [ReadTarget(p, w) for p, w in whole.items()]


# Programs whose output is bytes of the file and nothing else. A command built
# only from these, over files the context already holds, admits nothing new.
# `grep`, `awk`, `cut`, `jq` and `diff` read the same bytes but return a
# search, a projection or a comparison, which the model would otherwise have
# to compute by reading hundreds of lines in its head: not a duplicate.
BYTE_READERS: frozenset[str] = frozenset({
    "cat", "bat", "nl", "tac", "head", "tail", "less", "more"})


# `sed -n 40,80p` prints lines; any other sed program transforms or searches.
_SED_PRINT = re.compile(r"(\d+|\$)(,(\d+|\$))?p")
_SED_QUIET = frozenset({"-n", "--quiet", "--silent"})
_SED_HARMLESS = _SED_QUIET | {"-E", "-r", "-s", "-u", "--posix"}


def _sed_prints_lines(args: list[str]) -> bool:
    """Is this `sed` call nothing but `-n` and line-range prints?

    The script is judged unquoted and by position. The old test looked at raw
    words and let anything not ending in `p` through as a filename, so
    `sed -n 's/a/b/p' f` (its last character is the quote) and
    `sed -n '/def /p' f` -- a grep -- both counted as printing bytes, and the
    guard refused them as duplicates of a file already read.
    """
    words = [_unquote(a) for a in args]
    if not _SED_QUIET & set(words):
        return False
    scripts: list[str] = []
    positional: list[str] = []
    it = iter(words)
    for w in it:
        if w in ("-e", "--expression"):
            scripts.append(next(it, ""))
        elif w.startswith("--expression="):
            scripts.append(w.split("=", 1)[1])
        elif w.startswith("-") and w != "-":
            if w not in _SED_HARMLESS:
                return False                  # -i writes, -f reads a script we cannot see
        else:
            positional.append(w)
    if not scripts:
        if not positional:
            return False
        scripts.append(positional.pop(0))
    return all(
        _SED_PRINT.fullmatch(part.strip())
        for sc in scripts for part in sc.split(";") if part.strip())


def only_prints_files(command: str) -> bool:
    """Is every program this command runs one that just prints file bytes?

    The test a duplicate refusal has to pass. Without it `cat a.py && pytest`
    was refused whole because its one read target was known -- test run and
    all -- and `grep -n foo a.py` was refused because a slice of a known file
    "adds nothing", when what it adds is the search.
    """
    stages = [s for p in pipelines(command) for s in p]
    if not stages:
        return False
    for prog, args in stages:
        if prog == "sed":
            if not _sed_prints_lines(args):
                return False
            continue
        if prog not in BYTE_READERS:
            return False
        if prog in ("tail", "less") and any(
                a in ("-f", "-F", "+F") or a.startswith("--follow") for a in args):
            return False                      # prints what is appended, not what is there
    return True


def resolve(path: str, cwd: str | None = None) -> str:
    """`path` as an absolute path, or "" when that cannot be answered honestly.

    A relative path without a directory to resolve against is dropped rather
    than guessed. Guessing would key it against whatever directory the *report*
    happens to run in, which is how one file becomes two identities and two
    files become one.
    """
    if not path:
        return ""
    p = os.path.expanduser(path)
    if not os.path.isabs(p):
        if not cwd:
            return ""
        p = os.path.join(str(cwd), p)
    return os.path.normpath(p)


def tool_targets(tool: str, inp: dict | None, *, cwd: str | None = None) -> list[ReadTarget]:
    """What a tool call read, for the two tools that read files.

    One entry point for `Read` and `Bash` so that a path read by one and
    re-read by the other is the same identity. That is the whole point: the
    harness picks which of them does the reading, and the saving does not
    depend on its choice.

    Parsing only -- nothing here opens a file. A `whole` target is what the
    command asked for; `whole_reads` is what a refusal may rest on. See
    `ReadTarget`.
    """
    inp = inp or {}
    if tool == "Read":
        fp = inp.get("file_path")
        if not fp:
            return []
        path = resolve(str(fp), cwd)
        # A `limit` or an `offset` admitted a slice, not the file -- the rule
        # `guard.observe` has held for `Read` since it learned to remember one.
        entire = not inp.get("limit") and not inp.get("offset")
        return [ReadTarget(path, entire)] if path else []
    if tool == "Bash":
        out = []
        for t in read_targets(str(inp.get("command") or "")):
            path = resolve(t.path, cwd)
            if path:
                out.append(ReadTarget(path, t.whole))
        return out
    return []


# What the Read tool itself returns of a file, which is less than the file past
# any of three limits: the first 2,000 lines, each cut at 2,000 characters, and
# an error instead of content once the file is over the output-token ceiling
# (25,000 by default, `CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS`). A whole Read
# of a file past one of them did not put the file in the context, and the guard
# remembering it as if it had refused the `offset` read that would have got the
# rest -- with "use the copy you have", about a copy that stops at line 2,000.
READ_MAX_LINES = 2000
READ_MAX_LINE_CHARS = 2000
DEFAULT_READ_MAX_TOKENS = 25_000
# Characters per token, set low on purpose: it over-counts tokens, so a file
# near the ceiling is taken not to fit. That errs toward the guard staying quiet.
_CHARS_PER_TOKEN = 3


def read_max_tokens() -> int:
    """Tokens a Read returns before the harness refuses the file."""
    try:
        n = int(str(os.environ.get("CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS")))
    except (TypeError, ValueError):
        return DEFAULT_READ_MAX_TOKENS
    return n if n > 0 else DEFAULT_READ_MAX_TOKENS


def read_fits(path: str) -> bool:
    """Does a Read with no offset or limit return the whole of this file?

    False for anything that is not plain text, since what an image or a PDF
    puts in the context is not the file's bytes.
    """
    cap = read_max_tokens() * _CHARS_PER_TOKEN
    try:
        if os.path.getsize(path) > cap:
            return False
        with open(path, "rb") as f:
            blob = f.read(cap + 1)
    except OSError:
        return False
    if b"\0" in blob[:8192]:
        return False
    lines = blob.splitlines()                 # a trailing newline ends a line, not starts one
    return len(lines) <= READ_MAX_LINES and all(len(x) <= READ_MAX_LINE_CHARS for x in lines)


def whole_reads(tool: str, inp: dict | None, *, cwd: str | None = None,
                max_chars: int | None = None) -> list[str]:
    """Paths this call put into the context in full, and that are still there.

    The size check is not belt and braces. A `cat` of a file bigger than the
    harness's output ceiling returns a truncated result, so the file is *not*
    in the context -- and a guard that thinks it is refuses the read that would
    have got the rest.
    """
    cap = max_bash_output_chars() if max_chars is None else max_chars
    if tool == "Bash" and isinstance(inp, dict) and inp.get("run_in_background"):
        # A background command's output goes to a task, not into the result;
        # the model sees it only if it asks, so nothing is in the context yet.
        return []
    out = []
    used = 0
    for t in tool_targets(tool, inp, cwd=cwd):
        if not t.whole:
            continue
        if tool == "Bash":
            # One result, one ceiling: `cat a b` spills when the two together
            # pass it, even if neither does alone.
            try:
                used += os.path.getsize(t.path)
            except OSError:
                continue
            if used > cap:
                return []
        elif tool == "Read" and not read_fits(t.path):
            continue
        out.append(t.path)
    return out
