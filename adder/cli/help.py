"""`adder help`, rendered from the command table rather than restated.

Help text that is maintained by hand drifts from the code within about two
commits, and the drift is silent: the command still works, it is just invisible.
So this module owns no list of its own -- it formats `commands.COMMANDS`.
"""

from __future__ import annotations

from adder.cli.commands import COMMANDS, GROUP_BLURB, GROUPS


def usage() -> str:
    width = max(
        max(len(f"{c.name} {c.usage}") for c in COMMANDS),
        len("version, --version, -V"),
    )
    out = [
        "adder <command> [args]   —  cost tooling for Claude agent sessions",
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
