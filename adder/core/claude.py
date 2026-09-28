"""What Claude Code itself has installed: the adder plugin, and the tier agents.

Routing is only correct if the agent it names exists. With `adder auto on` the
tier agents are files in `~/.claude/agents` and dispatch as `route-t1`; with
the Claude Code plugin they come from the plugin, which namespaces them, and
dispatch as `adder:route-t1`. `adder policy` named the bare form either way,
so on a plugin-only machine every recommendation to delegate was a
recommendation to call an agent that did not exist -- and the two places that
read dispatches back, the guard's "already routed" check and the outcome
importer, did not recognise the namespaced form either, so the router never
learned from the delegations that did happen.

This sits in `core` because `decide/route`, `decide/track`, `decide/auto` and
`evaluate/doctor` all need it, and the rule is that a shared piece moves down.
Read-only, and every function here returns an answer rather than raising: it
runs inside hooks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# The plugin's name, as `enabledPlugins` keys it: `adder@<marketplace>`.
PLUGIN = 'adder'
PREFIX = f'{PLUGIN}:'


def _cwd(cwd: Path | str | None) -> Path | None:
    try:
        return Path(cwd or os.getcwd()).resolve()
    except (OSError, ValueError):
        return None                   # a working directory deleted from under us


def settings_files(cwd: Path | str | None = None) -> list[Path]:
    """Every Claude Code settings file that can enable a plugin, nearest last."""
    out = [Path.home() / '.claude' / 'settings.json']
    base = _cwd(cwd)
    if base is not None:
        out += [base / '.claude' / 'settings.json', base / '.claude' / 'settings.local.json']
    return out


def _read(path: Path) -> dict:
    try:
        blob = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    return blob if isinstance(blob, dict) else {}


def plugin_enabled(cwd: Path | str | None = None) -> bool:
    """Is the adder plugin enabled here? The nearer settings file wins."""
    merged: dict = {}
    for path in settings_files(cwd):
        ep = _read(path).get('enabledPlugins')
        if isinstance(ep, dict):
            merged.update(ep)
    return any(str(k).split('@')[0] == PLUGIN and bool(v) for k, v in merged.items())


def bare(agent: str) -> str:
    """`adder:route-t1` -> `route-t1`, for matching a dispatch against a tier."""
    name = str(agent).strip()
    return name[len(PREFIX):] if name.lower().startswith(PREFIX) else name


def agent_file(name: str, cwd: Path | str | None = None) -> Path | None:
    """The user- or project-level definition of this agent, if there is one."""
    dirs = [Path.home() / '.claude' / 'agents']
    base = _cwd(cwd)
    if base is not None:
        dirs.append(base / '.claude' / 'agents')
    for d in dirs:
        if (d / f'{name}.md').is_file():
            return d / f'{name}.md'
    return None


def dispatch_name(agent: str, cwd: Path | str | None = None) -> str:
    """The `subagent_type` that reaches this tier agent on this machine.

    A file installed by `auto on` wins, because it is the definition the user
    can see and edit and it answers to the bare name. Failing that, the
    plugin's copy answers to the namespaced one. With neither, the bare name
    is still right: it is what `auto on` will install.
    """
    name = bare(agent)
    if agent_file(name, cwd) is None and plugin_enabled(cwd):
        return PREFIX + name
    return name
