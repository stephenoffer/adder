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

from adder import __version__
from adder.cli.commands import BY_NAME, COMMANDS, Command
from adder.cli.help import usage
from adder.pricing.prices import UnknownModelError as _PricesUnknownModel
from adder.pricing.registry import UnknownModelError as _RegistryUnknownModel

# Two classes, deliberately unrelated (see `registry.UnknownModelError`), and
# a command can raise either depending on which table it consulted.
_UNKNOWN_MODEL = (_PricesUnknownModel, _RegistryUnknownModel)

__all__ = ["BY_NAME", "COMMANDS", "Command", "main", "run", "usage"]


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
    try:
        rc = module.main(argv[1:])
    except _UNKNOWN_MODEL as e:
        # A model named on the command line that nothing can price. This is
        # user input, not a defect, and every command that takes `--model`,
        # `--weak`, `--strong` or `--sub-model` could raise it from somewhere
        # deep in the pricing layer -- `adder cascade --weak typo` printed a
        # forty-line traceback whose last line was already the right message.
        # Handled here for the same reason `run` handles BrokenPipeError: one
        # place, one behaviour, every command.
        print(f"adder {cmd.name}: {str(e).strip(chr(34))}", file=sys.stderr)
        return 2
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
