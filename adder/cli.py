"""Single entry point for every `adder` subcommand.

Each analysis module owns its own `main(argv) -> int` and its own argparse
parser. This dispatcher does not re-declare their flags; it resolves a command
name to a module, imports that module lazily, and hands off the remaining argv.
Lazy import matters: `adder live` should not pay to import the A/B harness.
"""

from __future__ import annotations

import difflib
import importlib
import os
import sys
from typing import NamedTuple

from adder import __version__


class Command(NamedTuple):
    name: str
    module: str
    group: str
    usage: str
    summary: str


COMMANDS: tuple[Command, ...] = (
    # Measure — read-only reports over transcript files.
    Command("live", "adder.live", "Measure", "[--cwd DIR]",
            "this session: cost/turn, next-turn cost, pressure"),
    Command("trace", "adder.trace", "Measure", "[root] [--json] [--verify]",
            "total spend, by model and session"),
    Command("debt", "adder.debt", "Measure", "[root]",
            "what an output token really costs"),
    Command("context", "adder.context", "Measure", "[root]",
            "where context growth comes from"),
    Command("cache", "adder.cache", "Measure", "[root]",
            "cache hit rate and rebuild waste, by cause"),
    Command("quality", "adder.quality", "Measure", "[root] [--since DATE]",
            "agent-performance proxies"),
    Command("horizon", "adder.horizon", "Measure", "[root]",
            "remaining-turns estimate vs the naive countdown"),
    # Decide — turn a measurement into a routing choice.
    Command("policy", "adder.policy", "Decide", '"<task>" [--json]',
            "route a task: inline vs delegate"),
    Command("outcomes", "adder.outcomes", "Decide", "[--log PATH]",
            "escalation calibration (p_fail)"),
    Command("classify", "adder.classify", "Decide", '"<task>"',
            "task-complexity classification, on its own"),
    Command("pick", "adder.select", "Decide", '"<task>" [--combos] [--json]',
            "cheapest model, or combination, that clears the quality bar"),
    Command("models", "adder.models", "Decide", "[list|show|ladder|refresh]",
            "the cross-provider catalog: what exists, at what price and rating"),
    # Evaluate — check that a lever is real before trusting it.
    Command("savings", "adder.savings", "Evaluate", "[root] [--max-turns N]",
            "what each lever is worth"),
    Command("verify", "adder.verify", "Evaluate", "--since DATE [root]",
            "did a change actually land?"),
    Command("validate", "adder.validate", "Evaluate", "[root]",
            "re-test the claims everything rests on"),
    Command("regret", "adder.regret", "Evaluate", "[root]",
            "dollar regret of the horizon estimator"),
    Command("simulate", "adder.simulate", "Evaluate", "[root]",
            "replay sessions under interventions; test lever composition"),
    Command("ab", "adder.ab", "Evaluate", "--help",
            "controlled A/B on answer quality"),
)

BY_NAME: dict[str, Command] = {c.name: c for c in COMMANDS}

GROUP_BLURB = {
    "Measure": "read-only, no API calls, no network",
    # `models refresh` is the single exception to the no-network rule, and it
    # only ever runs when typed.
    "Decide": "offline except `models refresh`",
    "Evaluate": "",
}


def usage() -> str:
    width = max(
        max(len(f"{c.name} {c.usage}") for c in COMMANDS),
        len("version, --version, -V"),
    )
    out = [
        "adder <command> [args]   —  cost tooling for Claude agent sessions",
        "",
    ]
    for group in ("Measure", "Decide", "Evaluate"):
        blurb = GROUP_BLURB.get(group, "")
        out.append(f"  {group}" + (f" ({blurb})" if blurb else ""))
        for c in COMMANDS:
            if c.group == group:
                left = f"{c.name} {c.usage}".rstrip()
                out.append(f"    {left.ljust(width)}  {c.summary}")
        out.append("")
    meta = (
        ("help, --help, -h", "this message"),
        ("version, --version, -V", "print the installed version"),
    )
    out.append("  Meta")
    out += [f"    {left.ljust(width)}  {right}" for left, right in meta]
    out += [
        "",
        "Every report is computed locally from transcript files. Start with `adder live`.",
        "Per-command flags: `adder <command> --help`.",
    ]
    return "\n".join(out)


def _unknown(name: str) -> int:
    print(f"adder: unknown command {name!r}", file=sys.stderr)
    close = difflib.get_close_matches(name, list(BY_NAME), n=3, cutoff=0.5)
    if close:
        print(f"Did you mean: {', '.join(close)}?", file=sys.stderr)
    print("Run `adder help` for the full list.", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("help", "-h", "--help"):
        # `adder help <command>` forwards to that command's own parser.
        if len(argv) > 1 and argv[1] in BY_NAME:
            return main([argv[1], "--help"])
        print(usage())
        return 0

    if argv[0] in ("version", "-V", "--version"):
        print(f"adder {__version__}")
        return 0

    cmd = BY_NAME.get(argv[0])
    if cmd is None:
        return _unknown(argv[0])

    module = importlib.import_module(cmd.module)
    rc = module.main(argv[1:])
    return 0 if rc is None else int(rc)


def run() -> int:
    """Console-script wrapper: turn expected interruptions into clean exits.

    A report piped into `head` closes the pipe early; without this the user sees
    a BrokenPipeError traceback instead of the output they asked for.
    """
    try:
        return main()
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:
        # Redirect stdout to devnull so the interpreter's own flush at exit does
        # not raise a second BrokenPipeError after we have already handled this.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 0


if __name__ == "__main__":
    sys.exit(run())
