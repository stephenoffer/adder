"""`adder mcp`: the reports as tools, for an agent that is not Claude Code.

A skill is how Claude Code reaches adder, and a skill is a Claude Code idea.
Codex, Gemini CLI and Cursor all read transcripts adder can price, and none of
them could ask it anything: the numbers existed, but only for somebody willing
to leave the agent and type the command. MCP is the one integration all of
them speak, so this is a Model Context Protocol server over stdio -- newline
delimited JSON-RPC 2.0 -- written against the stdlib, because the rule about
dependencies does not bend for a protocol.

Three decisions, each of which is a test:

* **Read-only, by allowlist.** `TOOLS` names the reports an agent may run, and
  nothing else is reachable: not `auto`, which writes settings; not
  `outcomes record` or `export`, which write files; not `models refresh`, the
  one command that opens a socket. A server that could be talked into
  `auto on --yes` would be a second write path, and CLAUDE.md allows one.
* **Each call is a fresh process.** A tool runs `python -m adder <command>`
  with stdin closed and its output captured. In process would be wrong in a
  way no single call shows: `native._latest_model` is memoised for the life of
  the process, so a server started on Monday would go on pricing Tuesday's
  sessions at Monday's model. A subprocess also means a
  stray `print` cannot reach the protocol stream and an `input()` cannot stall
  it. The price is one interpreter start per call, which an agent waiting on
  a model does not notice.
* **Few tools, short descriptions.** An MCP tool definition is resident in the
  agent's context on every turn of every session, which is exactly the charge
  `adder memory` exists to price. Nine one-line tools is the budget; a report
  that is not here is still one shell command away.

It is not a background process in the sense CLAUDE.md forbids: the agent
starts it, talks to it over a pipe, and ends it with the session. It opens no
socket and runs no timer.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from adder import __version__

# Newest first. A client asking for one of these gets it back; anything else
# gets the newest, which is what the spec says a server should answer with.
PROTOCOL_VERSIONS: tuple[str, ...] = ('2025-06-18', '2025-03-26', '2024-11-05')

# A report can run to thousands of lines on a large history, and every
# character returned lands in the calling agent's context for the rest of its
# session. Past this the reply is cut, and says how to narrow it.
MAX_CHARS = 40_000
# A full `doctor` over one machine's history is under a second; a cold parse
# of a very large one is tens. Past this something is wrong, and an agent
# blocked on a tool call cannot say so.
TIMEOUT_S = 300.0

_WINDOW = {
    'root': 'transcript directory or agent name (claude, codex, gemini, opencode)',
    'since': 'only turns on or after this date: YYYY-MM-DD, 7d, 2w, today',
    'project': 'only projects whose directory name contains this',
}


@dataclass(frozen=True)
class Tool:
    """One report an agent may run, and how its arguments become argv."""
    name: str
    command: str
    description: str
    params: dict[str, str]
    required: tuple[str, ...] = ()
    positional: str = ''          # the param passed bare rather than as --flag

    def schema(self) -> dict:
        return {'type': 'object',
                'properties': {k: {'type': 'string', 'description': v}
                               for k, v in self.params.items()},
                'required': list(self.required),
                'additionalProperties': False}

    def argv(self, args: dict[str, Any]) -> list[str]:
        """The child's argv, with no way for a value to become a flag.

        Values come from the calling agent, which means from whatever text it
        was handed. Passed bare, `{"task": "--record"}` reached `policy` as the
        flag and wrote the ledger through a server that promises to be read-only;
        `{"root": "--help"}` returned usage. Options are joined to their value
        with `=`, and the positional sits after `--`, so argparse reads each one
        as data whatever it starts with.
        """
        out: list[str] = []
        for k in self.params:
            if k != self.positional and args.get(k):
                out.append(f'--{k}={args[k]}')
        if self.positional and args.get(self.positional):
            out += ['--', str(args[self.positional])]
        return out


TOOLS: tuple[Tool, ...] = (
    Tool('doctor', 'doctor', 'Every cost check, ranked by dollars at stake. Start here.',
         _WINDOW, positional='root'),
    Tool('live', 'live', "This session's cost per turn, next-turn cost and context pressure.",
         {'cwd': 'working directory of the session (default: the server\'s)'}),
    Tool('trace', 'trace', 'Total spend, by model and by session.',
         _WINDOW, positional='root'),
    Tool('sessions', 'sessions', 'The most expensive sessions, ranked.',
         _WINDOW, positional='root'),
    Tool('savings', 'savings', 'What each lever would have saved on this history.',
         {'root': _WINDOW['root']}, positional='root'),
    Tool('context', 'context', 'Where context growth comes from.',
         _WINDOW, positional='root'),
    Tool('policy', 'policy', 'Do a task inline or delegate it, and on which model.',
         {'task': 'the task, in a sentence'}, required=('task',), positional='task'),
    Tool('pick', 'pick', 'The cheapest capable model for a task, across vendors.',
         {'task': 'the task, in a sentence'}, required=('task',), positional='task'),
    Tool('handoff', 'handoff', 'Whether to restart, and what a fresh session needs told.',
         {'cwd': 'working directory of the session (default: the server\'s)'}),
)
BY_TOOL: dict[str, Tool] = {t.name: t for t in TOOLS}

INSTRUCTIONS = (
    'adder prices coding-agent sessions from the transcripts on this machine. '
    'Every number is computed locally and costs no model tokens; quote it rather '
    'than re-deriving it. Call doctor first; the others go deeper on one finding.')


def run_tool(tool: Tool, args: dict[str, Any], *, timeout: float = TIMEOUT_S) -> tuple[str, bool]:
    """Run one report in a child process. Returns (text, is_error). Never raises."""
    import subprocess

    bad = [k for k in args if k not in tool.params]
    bad += [k for k, v in args.items() if k in tool.params and not isinstance(v, str)]
    missing = [k for k in tool.required if not args.get(k)]
    if bad or missing:
        return (f'invalid arguments: unknown or non-string {sorted(set(bad))}, '
                f'missing {missing}'), True
    cmd = [sys.executable, '-m', 'adder', tool.command, *tool.argv(args)]
    # The child imports the package this server was imported from, not
    # whichever one its working directory happens to find: from a clone that
    # is the only way it imports at all.
    env = dict(os.environ)
    here = str(Path(__file__).resolve().parents[2])
    env['PYTHONPATH'] = os.pathsep.join(filter(None, (here, env.get('PYTHONPATH'))))
    try:
        done = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True,
                              text=True, timeout=timeout, check=False, env=env)
    except subprocess.TimeoutExpired:
        return f'adder {tool.command} took longer than {timeout:.0f}s and was stopped', True
    except OSError as e:
        return f'adder {tool.command} could not start: {e}', True
    text = done.stdout
    if done.returncode:
        text = ((text + done.stderr).strip()
                or f'adder {tool.command} exited {done.returncode}')
    if len(text) > MAX_CHARS:
        text = (text[:MAX_CHARS] + f'\n\n[cut at {MAX_CHARS:,} characters; narrow it '
                'with since or project]')
    return text, bool(done.returncode)


def _result(msg_id: Any, result: dict) -> dict:
    return {'jsonrpc': '2.0', 'id': msg_id, 'result': result}


def _error(msg_id: Any, code: int, message: str) -> dict:
    return {'jsonrpc': '2.0', 'id': msg_id, 'error': {'code': code, 'message': message}}


def handle(msg: Any) -> dict | None:
    """The response to one decoded message, or None for a notification."""
    if not isinstance(msg, dict) or msg.get('jsonrpc') != '2.0' \
            or not isinstance(msg.get('method'), str):
        return _error(msg.get('id') if isinstance(msg, dict) else None,
                      -32600, 'invalid request')
    if 'id' not in msg:
        return None                           # notifications/initialized and friends
    msg_id, method = msg['id'], msg['method']
    params = msg.get('params') if isinstance(msg.get('params'), dict) else {}
    if method == 'initialize':
        asked = params.get('protocolVersion')
        return _result(msg_id, {
            'protocolVersion': asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
            'capabilities': {'tools': {'listChanged': False}},
            'serverInfo': {'name': 'adder', 'version': __version__},
            'instructions': INSTRUCTIONS})
    if method == 'ping':
        return _result(msg_id, {})
    if method == 'tools/list':
        return _result(msg_id, {'tools': [
            {'name': t.name, 'description': t.description, 'inputSchema': t.schema()}
            for t in TOOLS]})
    if method == 'tools/call':
        tool = BY_TOOL.get(str(params.get('name')))
        if tool is None:
            return _error(msg_id, -32602, f"unknown tool {params.get('name')!r}")
        args = params.get('arguments') or {}
        if not isinstance(args, dict):
            return _error(msg_id, -32602, 'arguments must be an object')
        text, is_error = run_tool(tool, args)
        return _result(msg_id, {'content': [{'type': 'text', 'text': text}],
                                'isError': is_error})
    return _error(msg_id, -32601, f'method not found: {method}')


def serve(stdin: IO[str] | None = None, stdout: IO[str] | None = None) -> int:
    """Read requests until end of input. One JSON object per line each way."""
    rx = sys.stdin if stdin is None else stdin
    tx = sys.stdout if stdout is None else stdout   # taken before any capture
    for line in rx:
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            reply: dict | None = _error(None, -32700, 'parse error')
        else:
            reply = handle(msg)
        if reply is not None:
            tx.write(json.dumps(reply, separators=(',', ':')) + '\n')
            tx.flush()
    return 0


def _checkout_launcher() -> str | None:
    """`scripts/adder` beside this package, when running from a clone.

    A clone imports only because the launcher put it on PYTHONPATH, so a
    snippet naming `python -m adder` would start a server that cannot import
    itself. The launcher carries that setting with it.
    """
    launcher = Path(__file__).resolve().parents[2] / 'scripts' / 'adder'
    return str(launcher) if launcher.is_file() else None


def config_snippet(agent: str, exe: str | None = None) -> str:
    """What to paste into one agent's MCP configuration. Printed, never written."""
    import shutil

    if exe is None:
        exe = shutil.which('adder') or _checkout_launcher()
    # Absolute, because an agent launching a server does not promise the PATH
    # of the shell that installed it. Without a console script or a checkout
    # launcher, the interpreter running this is the one known to import it.
    command, args = (exe, ['mcp']) if exe else (sys.executable, ['-m', 'adder', 'mcp'])
    blob = json.dumps({'mcpServers': {'adder': {'command': command, 'args': args}}},
                      indent=2)
    if agent == 'codex':
        return ('# ~/.codex/config.toml\n[mcp_servers.adder]\n'
                f'command = {json.dumps(command)}\nargs = {json.dumps(args)}\n')
    if agent == 'gemini':
        return f'// ~/.gemini/settings.json\n{blob}\n'
    if agent == 'cursor':
        return f'// ~/.cursor/mcp.json\n{blob}\n'
    return f"claude mcp add adder -- {shlex.join([command, *args])}\n"


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog='adder mcp',
        description='Serve the read-only reports as MCP tools over stdio. Your agent '
                    'starts this; you register it once with --print-config.')
    ap.add_argument('--print-config', metavar='AGENT',
                    choices=('codex', 'gemini', 'cursor', 'claude'),
                    help='print the config snippet for codex, gemini, cursor or claude, '
                         'and exit (writes nothing)')
    ap.add_argument('--list', action='store_true', help='list the tools served, and exit')
    a = ap.parse_args(argv)
    if a.print_config:
        print(config_snippet(a.print_config), end='')
        return 0
    if a.list:
        for t in TOOLS:
            print(f'  {t.name:<10}{t.description}')
        return 0
    return serve()
