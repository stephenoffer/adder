"""`adder help`, rendered from the command table rather than restated.

Help text that is maintained by hand drifts from the code within about two
commits, and the drift is silent: the command still works, it is just invisible.
So this module owns no list of its own -- it formats `commands.COMMANDS`.

There are two screens, because the one screen that existed failed its first
reader. Sixty commands, each with its flag syntax, in declaration order, put
`adder doctor` fifty lines below `adder cachesim` and gave a new user no way to
tell which of them mattered. `usage()` is now the short screen: the handful of
commands a first run needs, in plain words, then every command name by group so
nothing is hidden. `usage(full=True)` (`adder help --all`) is the reference.
"""

from __future__ import annotations

import textwrap

from adder.cli.commands import COMMANDS, GROUP_BLURB, GROUPS, START_HERE

TAGLINE = "adder — what your coding-agent sessions cost, and how to spend less"


def _short() -> str:
    width = max(len(f"adder {inv}") for inv, _ in START_HERE)
    out = [TAGLINE, "", "usage: adder <command> [args]", "", "  Start here"]
    out += [f"    {('adder ' + inv).ljust(width)}  {what}" for inv, what in START_HERE]
    out += ["", "  Every command, by group"]
    label = max(len(g) for g in GROUPS)
    for group in GROUPS:
        names = " ".join(c.name for c in COMMANDS if c.group == group)
        wrapped = textwrap.wrap(names, width=78 - label - 6) or [""]
        out.append(f"    {group.ljust(label)}  {wrapped[0]}")
        out += [f"    {'':<{label}}  {rest}" for rest in wrapped[1:]]
    out += [
        "",
        "  adder help <command>   what one command does, and its flags",
        "  adder help --all       every command with a one-line description",
        "",
        "Reads the transcripts Claude Code, Codex CLI, Gemini CLI and OpenCode already",
        "keep, and finds them on its own. `adder trace codex` picks one agent; any path",
        "picks a directory or a single API/proxy log. No account, no model calls, no",
        "network (except `adder models refresh`, when you run it), and no transcript is",
        "ever modified.",
    ]
    return "\n".join(out)


def _full() -> str:
    meta = (
        ("help [--all | <command>]", "the short screen, this one, or one command's flags"),
        ("version, --version, -V", "print the installed version"),
    )
    width = max(
        max(len(f"{c.name} {c.usage}") for c in COMMANDS),
        max(len(left) for left, _ in meta),
    )
    out = [
        "adder <command> [args]   —  " + TAGLINE.split(" — ", 1)[1],
        "",
    ]
    for group in GROUPS:
        blurb = GROUP_BLURB.get(group, "")
        out.append(f"  {group}" + (f" ({blurb})" if blurb else ""))
        for c in COMMANDS:
            if c.group == group:
                left = f"{c.name} {c.usage}".rstrip()
                out.append(f"    {left.ljust(width)}  {c.summary}")
        out.append("")
    out.append("  Meta")
    out += [f"    {left.ljust(width)}  {right}" for left, right in meta]
    out += [
        "",
        "Every report is computed locally from transcript files. New here? `adder doctor`.",
        "Per-command flags: `adder <command> --help`.",
    ]
    return "\n".join(out)


def usage(full: bool = False) -> str:
    return _full() if full else _short()
