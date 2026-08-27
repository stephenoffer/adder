"""`adder config`: what is in effect, and which layer put it there.

The resolution rules live in `adder.core.settings` because every package reads
them. What is here is only the presentation -- a table of resolved values and
their provenance. Splitting the two keeps `core` free of a command, and keeps
this file from being imported by fifteen report modules that want one value.
"""

from __future__ import annotations

import json
from pathlib import Path

from adder.core.settings import (
    PROJECT_FILE,
    SETTINGS,
    USER_FILE,
    ConfigError,
    ignored_in_files,
    project_file,
    resolve,
    template,
)


def report(*, cwd: Path | str | None = None) -> str:
    from adder.util.render import table

    res = resolve(cwd=cwd)
    rows = []
    for s in SETTINGS:
        r = res[s.name]
        val = str(r.value)
        if len(val) > 44:
            val = "…" + val[-43:]
        rows.append([s.name, val, r.source if r.overridden else "",
                     s.env_var + (" (env only)" if s.env_only else "")])
    lines = ["  Effective configuration", ""]
    lines += table(rows, ["setting", "value", "from", "env var"], align="<<<<")
    lines.append("")
    env_only = [s.name for s in SETTINGS if s.env_only]
    lines.append("  (env only): read from the environment and never from a config "
                 "file — the")
    lines.append(f"  code that consumes {', '.join(env_only)} sits below the settings "
                 "layer and")
    lines.append("  cannot import it.")
    ignored = ignored_in_files(cwd=cwd)
    if ignored:
        lines.append("")
        lines.append(f"  ! your config file sets {', '.join(ignored)}, which nothing "
                     f"reads from a file.")
        names = ", ".join(s.env_var for s in SETTINGS if s.name in ignored)
        lines.append(f"    Export {names} instead.")
    lines.append("")
    pf = project_file(cwd)
    lines.append(f"  user file     {USER_FILE}{'' if USER_FILE.is_file() else '  (absent)'}")
    lines.append(f"  project file  {pf if pf else f'./{PROJECT_FILE}  (absent)'}")
    lines.append("")
    lines.append("  Precedence: default < user file < project file < environment.")
    lines.append("  `adder config --init > .adder.json` writes a starting point.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        prog="adder config",
        description="Show the settings in effect and where each one came from.")
    ap.add_argument("name", nargs="?", help="print one setting's value and exit")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    ap.add_argument("--init", action="store_true",
                    help="print a config file template to stdout")
    ap.add_argument("--explain", action="store_true",
                    help="include the description of every setting")
    a = ap.parse_args(argv)

    if a.init:
        print(template())
        return 0

    try:
        res = resolve()
    except ConfigError as e:
        print(f"adder config: {e}", file=sys.stderr)
        return 1

    if a.name:
        if a.name not in res:
            print(f"adder config: unknown setting {a.name!r}", file=sys.stderr)
            return 2
        r = res[a.name]
        if a.json:
            print(json.dumps({"name": r.name, "value": r.value, "source": r.source,
                              "env": r.setting.env_var, "help": r.setting.help}))
        else:
            print(r.value)
        return 0

    if a.json:
        print(json.dumps({
            n: {"value": r.value, "source": r.source, "env": r.setting.env_var,
                "default": r.setting.initial, "help": r.setting.help}
            for n, r in res.items()
        }, indent=2))
        return 0

    print()
    print(report())
    if a.explain:
        print()
        for s in SETTINGS:
            print(f"  {s.name:<22}{s.help}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
