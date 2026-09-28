"""The MCP server: its protocol, its allowlist, and the stream it must not dirty.

Driven through `serve()` with string streams rather than a pipe, so every
test here is deterministic and reads nothing but a `tmp_path` home. The tool
calls do start a child process -- that is the design -- and inherit the
isolated HOME through the environment.
"""

from __future__ import annotations

import io
import json

import pytest

from adder.cli import mcp
from adder.cli.commands import BY_NAME


def _talk(*messages) -> list[dict]:
    rx = io.StringIO(''.join((m if isinstance(m, str) else json.dumps(m)) + '\n'
                             for m in messages))
    tx = io.StringIO()
    assert mcp.serve(rx, tx) == 0
    return [json.loads(line) for line in tx.getvalue().splitlines()]


def _req(i, method, **params):
    return {'jsonrpc': '2.0', 'id': i, 'method': method, 'params': params}


class TestTheHandshake:
    def test_initialize_names_the_server_and_its_one_capability(self):
        (r,) = _talk(_req(1, 'initialize', protocolVersion='2025-06-18'))
        res = r['result']
        assert r['id'] == 1
        assert res['serverInfo']['name'] == 'adder'
        assert res['capabilities'] == {'tools': {'listChanged': False}}
        assert res['protocolVersion'] == '2025-06-18'

    def test_an_unknown_version_gets_the_newest(self):
        (r,) = _talk(_req(1, 'initialize', protocolVersion='1999-01-01'))
        assert r['result']['protocolVersion'] == mcp.PROTOCOL_VERSIONS[0]

    def test_a_notification_gets_no_reply(self):
        assert _talk({'jsonrpc': '2.0', 'method': 'notifications/initialized'}) == []

    def test_ping(self):
        assert _talk(_req(7, 'ping')) == [{'jsonrpc': '2.0', 'id': 7, 'result': {}}]

    def test_an_unknown_method_is_an_error_not_silence(self):
        (r,) = _talk(_req(2, 'resources/list'))
        assert r['error']['code'] == -32601

    def test_a_line_that_is_not_json_is_a_parse_error(self):
        (r,) = _talk('{not json')
        assert r['error']['code'] == -32700 and r['id'] is None

    def test_blank_lines_are_skipped(self):
        assert _talk('', '   ') == []


class TestTheAllowlist:
    """Nothing reachable here may write a file or open a socket."""

    WRITES: tuple[str, ...] = ('auto', 'hook', 'outcomes', 'export', 'models',
                               'guard', 'config', 'mcp')

    def test_the_listed_tools_are_exactly_the_allowlist(self):
        (r,) = _talk(_req(1, 'tools/list'))
        assert [t['name'] for t in r['result']['tools']] == [t.name for t in mcp.TOOLS]

    def test_every_tool_runs_a_registered_command(self):
        for t in mcp.TOOLS:
            assert t.command in BY_NAME, t.command

    def test_no_writing_command_is_exposed(self):
        assert not {t.command for t in mcp.TOOLS} & set(self.WRITES)

    @pytest.mark.parametrize('name', ['auto', 'outcomes', 'models', 'export'])
    def test_calling_one_anyway_is_refused(self, name):
        (r,) = _talk(_req(3, 'tools/call', name=name, arguments={}))
        assert r['error']['code'] == -32602

    def test_every_schema_forbids_extra_arguments(self):
        for t in mcp.TOOLS:
            assert t.schema()['additionalProperties'] is False

    def test_descriptions_stay_one_line(self):
        """A tool definition is resident on every turn of every session."""
        for t in mcp.TOOLS:
            assert '\n' not in t.description and len(t.description) <= 80, t.name


class TestArguments:
    def test_flags_and_the_positional_become_argv(self):
        t = mcp.BY_TOOL['doctor']
        assert t.argv({'root': 'codex', 'since': '7d'}) == ['--since=7d', '--', 'codex']

    @pytest.mark.parametrize(('tool', 'args'), [
        ('policy', {'task': '--record'}), ('doctor', {'root': '--help'}),
        ('trace', {'since': '--help'})])
    def test_a_value_can_never_become_a_flag(self, tool, args, isolated_home, tmp_path):
        """`{"task": "--record"}` used to reach `policy` as the flag and write
        the ledger through a server whose whole promise is that it reads."""
        before = sorted(isolated_home.rglob('*'))
        text, _ = mcp.run_tool(mcp.BY_TOOL[tool], args)
        assert sorted(isolated_home.rglob('*')) == before
        assert 'show this help message' not in text

    def test_an_unknown_argument_is_an_error_result(self):
        text, err = mcp.run_tool(mcp.BY_TOOL['trace'], {'yes': 'please'})
        assert err and 'yes' in text

    def test_a_non_string_argument_is_an_error_result(self):
        _, err = mcp.run_tool(mcp.BY_TOOL['trace'], {'since': 7})
        assert err

    def test_a_missing_task_is_an_error_result(self):
        _, err = mcp.run_tool(mcp.BY_TOOL['policy'], {})
        assert err


class TestCalling:
    def test_a_report_runs_and_its_text_comes_back(self, isolated_home, tmp_path):
        empty = tmp_path / 'no-transcripts'
        empty.mkdir()
        (r,) = _talk(_req(4, 'tools/call', name='trace', arguments={'root': str(empty)}))
        content = r['result']['content']
        assert content[0]['type'] == 'text' and content[0]['text'].strip()

    def test_every_line_on_stdout_is_json_rpc(self, isolated_home, tmp_path):
        """The report's own output goes into a result, never beside it."""
        empty = tmp_path / 'none'
        empty.mkdir()
        rx = io.StringIO('\n'.join(json.dumps(m) for m in (
            _req(1, 'initialize'),
            _req(2, 'tools/call', name='doctor', arguments={'root': str(empty)}),
            _req(3, 'tools/call', name='sessions', arguments={'root': str(empty)}),
        )) + '\n')
        tx = io.StringIO()
        mcp.serve(rx, tx)
        lines = tx.getvalue().splitlines()
        assert len(lines) == 3
        assert all(json.loads(line)['jsonrpc'] == '2.0' for line in lines)

    def test_a_long_reply_is_cut_and_says_so(self, monkeypatch, isolated_home, tmp_path):
        monkeypatch.setattr(mcp, 'MAX_CHARS', 10)
        text, _ = mcp.run_tool(mcp.BY_TOOL['trace'], {'root': str(tmp_path)})
        assert 'cut at 10 characters' in text


class TestPrintedConfig:
    """Printed, never written: `auto on` stays the one write path."""

    def test_codex_gets_toml(self):
        got = mcp.config_snippet('codex', exe='/x/adder')
        assert '[mcp_servers.adder]' in got and 'command = "/x/adder"' in got

    @pytest.mark.parametrize('agent', ['gemini', 'cursor'])
    def test_the_json_agents_get_mcp_servers(self, agent):
        blob = mcp.config_snippet(agent, exe='/x/adder').split('\n', 1)[1]
        assert json.loads(blob)['mcpServers']['adder'] == {
            'command': '/x/adder', 'args': ['mcp']}

    def test_claude_gets_the_one_line_command(self):
        assert mcp.config_snippet('claude', exe='/x/adder') == \
            'claude mcp add adder -- /x/adder mcp\n'

    def test_a_path_with_a_space_survives_the_paste(self):
        got = mcp.config_snippet('claude', exe='/my tools/adder')
        assert got == "claude mcp add adder -- '/my tools/adder' mcp\n"

    def test_printing_writes_nothing(self, isolated_home, capsys):
        before = sorted(p for p in isolated_home.rglob('*'))
        assert mcp.main(['--print-config', 'codex']) == 0
        assert sorted(p for p in isolated_home.rglob('*')) == before
        assert 'mcp_servers.adder' in capsys.readouterr().out
