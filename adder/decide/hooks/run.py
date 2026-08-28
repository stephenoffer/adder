"""`adder hook NAME`: run one hook through the console script, not by path.

Why this exists is one line in a settings file. `auto on` used to write the
absolute path of both the interpreter and the script:

    /home/ray/anaconda3/bin/python3 /mnt/store/adder/adder/decide/hooks/pretooluse_read_guard.py

Project scope was the default and `.claude/settings.json` is commonly tracked
in git, so a committed activation was broken for every other contributor on the
repository -- an interpreter that is not theirs, pointing at a checkout that is
not theirs. The hook fails open, so what they saw was nothing at all, which is
the failure mode this whole subtree is written to avoid.

`adder hook read-guard` resolves through PATH to the console script of whatever
environment each contributor installed `adder-cli` into, and a console script
carries its own interpreter in its shebang. That is the only form here that is
both portable across machines and correct about which Python can import the
package: `python3 -m adder...` is portable and wrong on any macOS where the
first `python3` on PATH is the system 3.9, and an absolute `sys.executable` is
correct and wrong on anybody else's machine. `auto` writes the module form for
a user-scope install, where the file never leaves the machine, and this form
for a project-scope one.

The cost is one `adder.cli` import per tool call, about 25ms measured here, on
top of the ~30ms the interpreter costs anyway. That is why it is not the
default: it is the price of portability, paid only where portability is what
was asked for.
"""

from __future__ import annotations

import importlib
import sys

# The hook names `auto` writes, and the modules behind them. Short, stable
# names rather than module paths: this string ends up in a file people commit,
# and a rename inside the package must not break a checked-in settings.json.
HOOKS: dict[str, str] = {
    'read-guard': 'adder.decide.hooks.pretooluse_read_guard',
    'compact-learn': 'adder.decide.hooks.precompact_learn',
    'cost-advisor': 'adder.decide.hooks.session_cost_advisor',
}


def _parser():
    import argparse

    ap = argparse.ArgumentParser(
        prog='adder hook',
        description='Run one harness hook. Claude Code calls this; you do not.',
        epilog='hooks: ' + ', '.join(f'{n} ({m})' for n, m in sorted(HOOKS.items())))
    ap.add_argument('name', nargs='?', choices=sorted(HOOKS), metavar='NAME',
                    help='which hook to run: ' + ', '.join(sorted(HOOKS)))
    return ap


def main(argv: list[str] | None = None) -> int:
    """Dispatch to one hook's `main`. Never raises, for the reason hooks do not.

    An unrecognised name does not exit non-zero, and that is the one place this
    departs from every other command in the table. The argument does not come
    from a person at a prompt: it comes from a `settings.json` that may have
    been committed by a colleague, months ago, against a version that spelled
    the hook differently. Failing loudly there means failing on every single
    tool call, for a component whose entire contract is to fail open. So the
    complaint goes to stderr -- where Claude Code shows it and the model's
    context never does -- and the turn proceeds.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = _parser()
    if argv and argv[0] not in HOOKS and not argv[0].startswith('-'):
        print(f'adder hook: unknown hook {argv[0]!r}; '
              f'expected one of {", ".join(sorted(HOOKS))}', file=sys.stderr)
        return 0
    a = ap.parse_args(argv)
    if not a.name:
        ap.print_usage(sys.stderr)
        print('adder hook: name is required', file=sys.stderr)
        return 2
    try:
        return int(importlib.import_module(HOOKS[a.name]).main() or 0)
    except Exception as e:                        # a hook must never break the turn
        print(f'adder hook {a.name}: {e}', file=sys.stderr)
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
