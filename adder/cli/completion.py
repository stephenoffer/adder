"""Shell completion, generated from the command table rather than maintained.

A hand-written completion script is a second copy of the command list, and a
second copy of a list is a list that is wrong. This one is generated from
`cli.COMMANDS` and from each module's own argparse parser at the moment it is
printed, so a command added today completes today and a flag renamed today
stops being suggested today.

Flags are discovered by importing the module and reading its parser, which is
the only source that cannot disagree with `--help`. That import costs a moment
at generation time and nothing at completion time: the output is a static
script the shell sources once.

Why the generated script is static rather than calling back into Python: a
completion that shells out to `python3 -c` on every Tab press adds ~80ms to
every keystroke sequence, which is the difference between completion that feels
free and completion people turn off.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import re
from dataclasses import dataclass, field

from adder.cli import COMMANDS

SHELLS = ("bash", "zsh", "fish")

# Flags the dispatcher owns, which no module's parser knows about.
META = ("help", "version", "--help", "-h", "--version", "-V")


def _parsers_for(command: str) -> list[argparse.ArgumentParser]:
    """The top-level parser(s) that command's `main` builds, or `[]`."""
    cmd = next((c for c in COMMANDS if c.name == command), None)
    if cmd is None:
        return []
    try:
        mod = importlib.import_module(cmd.module)
    except Exception:
        return []

    captured: list[argparse.ArgumentParser] = []
    real_parse = argparse.ArgumentParser.parse_args

    def spy(self, args=None, namespace=None):
        captured.append(self)
        raise _StopError

    argparse.ArgumentParser.parse_args = spy       # type: ignore[method-assign]
    try:
        # Every module builds its parser inside `main`, so the parser only
        # exists during a call. Running it with `--help` would exit; running it
        # with no arguments would do the work. So `parse_args` is intercepted:
        # by the time it is reached the parser is fully built.
        with contextlib.suppress(_StopError, SystemExit, Exception), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            mod.main([])
    finally:
        argparse.ArgumentParser.parse_args = real_parse  # type: ignore[method-assign]
    return captured


# What a shell script can carry unquoted. A choice outside it is dropped rather
# than escaped: no current one needs escaping, and one that did would be a
# value nobody types at a prompt.
_WORD = re.compile(r"^[A-Za-z0-9_.:=+-]+$")


def _long_flags(parser: argparse.ArgumentParser) -> list[str]:
    return sorted({s for a in parser._actions for s in a.option_strings
                   if s.startswith("--")})


@dataclass
class Spec:
    """What one command completes: its flags, its second word, and per-subcommand flags.

    Discovery used to read only `parser._actions` option strings off the
    top-level parser, so everything that is not a flag was invisible: the
    `on|off|status` of `adder auto`, the `list|show|ladder|refresh` of
    `adder models`, the shell names of `adder completion`, the hook names of
    `adder hook`. Worse, a subcommand's flags live on the subparser, so
    `adder models refresh --from` and `adder outcomes record --tier` were
    never offered at all.
    """

    flags: list[str] = field(default_factory=list)
    words: list[str] = field(default_factory=list)
    subs: dict[str, list[str]] = field(default_factory=dict)


def spec_for(command: str) -> Spec:
    """Read one command's completion `Spec` off the parser its `main` builds.

    Returns an empty `Spec` rather than raising when a module cannot be
    imported or does not build its parser in `main`: a completion script that
    fails to generate because one command is broken is worse than one missing
    a few flags.
    """
    spec = Spec()
    for parser in _parsers_for(command):
        spec.flags = sorted(set(spec.flags) | set(_long_flags(parser)))
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for name, sub in action.choices.items():
                    if _WORD.match(name):
                        spec.subs[name] = _long_flags(sub)
                        spec.words.append(name)
            elif not action.option_strings and action.choices:
                spec.words.extend(str(c) for c in action.choices
                                  if _WORD.match(str(c)))
    spec.words = sorted(set(spec.words))
    return spec


def flags_for(command: str) -> list[str]:
    """Every option string that command accepts, its subcommands' included."""
    spec = spec_for(command)
    out = set(spec.flags)
    for flags in spec.subs.values():
        out.update(flags)
    return sorted(out)


class _StopError(Exception):
    """Raised to unwind out of `main` once the parser has been built."""


def _table() -> list[tuple[str, Spec]]:
    return [(c.name, spec_for(c.name)) for c in COMMANDS]


def bash() -> str:
    rows = _table()
    names = " ".join(n for n, _ in rows) + " " + " ".join(META)
    cases = "\n".join(
        f'    {name}) opts="{" ".join(sp.flags)}"; words="{" ".join(sp.words)}" ;;'
        for name, sp in rows if sp.flags or sp.words)
    subcases = "\n".join(
        f'    "{name} {sub}") opts="{" ".join(flags)}" ;;'
        for name, sp in rows for sub, flags in sp.subs.items())
    subblock = (f"""  case "${{COMP_WORDS[1]}} ${{COMP_WORDS[2]}}" in
{subcases}
  esac
""" if subcases else "")
    return f"""# adder completion for bash. Generated by `adder completion bash`.
# Install:  adder completion bash > /etc/bash_completion.d/adder
#      or:  eval "$(adder completion bash)"
_adder_complete() {{
  local cur opts words
  COMPREPLY=()
  cur="${{COMP_WORDS[COMP_CWORD]}}"
  if [ "$COMP_CWORD" -eq 1 ]; then
    COMPREPLY=( $(compgen -W "{names}" -- "$cur") )
    return 0
  fi
  opts=""
  words=""
  case "${{COMP_WORDS[1]}}" in
{cases}
  esac
  if [ "$COMP_CWORD" -gt 2 ]; then
{subblock}    :
  fi
  if [[ "$cur" == -* ]]; then
    COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
  elif [ "$COMP_CWORD" -eq 2 ] && [ -n "$words" ]; then
    COMPREPLY=( $(compgen -W "$words" -- "$cur") )
  else
    COMPREPLY=( $(compgen -f -- "$cur") )
  fi
}}
complete -F _adder_complete adder
"""


def zsh() -> str:
    rows = _table()
    descriptions = "\n".join(
        f"    '{c.name}:{c.summary.replace(chr(39), '')}'" for c in COMMANDS)
    blocks = []
    for name, sp in rows:
        if not (sp.flags or sp.words):
            continue
        flag_case = [f"          {sub}) _values 'option' {' '.join(flags)} ;;"
                     for sub, flags in sp.subs.items() if flags]
        top = (f"_values 'option' {' '.join(sp.flags)}" if sp.flags else "_files")
        flag_case.append(f"          *) {top} ;;")
        body = "\n".join(flag_case)
        words = (f"""        if (( CURRENT == 3 )) && [[ $PREFIX != -* ]]; then
          compadd -- {' '.join(sp.words)}
          return
        fi
""" if sp.words else "")
        blocks.append(f"""      {name})
{words}        case "${{words[3]}}" in
{body}
        esac ;;""")
    cases = "\n".join(blocks)
    return f"""#compdef adder
# adder completion for zsh. Generated by `adder completion zsh`.
# Install:  adder completion zsh > "${{fpath[1]}}/_adder"
_adder() {{
  local -a commands
  commands=(
{descriptions}
  )
  if (( CURRENT == 2 )); then
    _describe 'command' commands
    return
  fi
  case "${{words[2]}}" in
{cases}
      *) _files ;;
  esac
}}
_adder "$@"
"""


def fish() -> str:
    rows = _table()
    lines = [
        "# adder completion for fish. Generated by `adder completion fish`.",
        "# Install:  adder completion fish > ~/.config/fish/completions/adder.fish",
        "complete -c adder -f",
    ]
    for c in COMMANDS:
        desc = c.summary.replace("'", "")
        lines.append(
            f"complete -c adder -n '__fish_use_subcommand' -a '{c.name}' -d '{desc}'")
    for name, sp in rows:
        for flag in sp.flags:
            lines.append(
                f"complete -c adder -n '__fish_seen_subcommand_from {name}' "
                f"-l '{flag.lstrip('-')}'")
        if sp.words:
            # Only as the word straight after the command: `adder auto on`
            # takes no second action, and a hook name is not a file.
            lines.append(
                f"complete -c adder -n '__fish_seen_subcommand_from {name}; "
                f"and test (count (commandline -opc)) -eq 2' "
                f"-a '{' '.join(sp.words)}'")
        for sub, flags in sp.subs.items():
            for flag in flags:
                lines.append(
                    f"complete -c adder -n '__fish_seen_subcommand_from {name}; "
                    f"and test (count (commandline -opc)) -ge 3; "
                    f"and test (commandline -opc)[3] = {sub}' "
                    f"-l '{flag.lstrip('-')}'")
    return "\n".join(lines) + "\n"


def script(shell: str) -> str:
    if shell not in SHELLS:
        raise ValueError(f"unknown shell {shell!r}; known: {', '.join(SHELLS)}")
    return {"bash": bash, "zsh": zsh, "fish": fish}[shell]()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="adder completion",
        description="Print a shell completion script generated from the command table.")
    ap.add_argument("shell", nargs="?", choices=SHELLS,
                    help="which shell (default: guess from $SHELL)")
    a = ap.parse_args(argv)

    shell = a.shell
    if shell is None:
        import os

        name = os.path.basename(os.environ.get("SHELL", ""))
        shell = next((s for s in SHELLS if s == name), None)
        if shell is None:
            print("adder completion: name a shell — "
                  f"{', '.join(SHELLS)}", file=__import__("sys").stderr)
            return 2
    print(script(shell), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
