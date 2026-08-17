"""Every `--json` mode must emit exactly one parseable JSON document.

This is a contract test, not a content test. The failure it exists to catch is
the one that only appears in a pipeline: a report that prints a friendly line
before its JSON, or emits a bare `NaN`, or forgets to return before falling
through to the text renderer. All three produce output that looks fine on a
terminal and breaks `jq` -- and by then it is in somebody's script.

Anything not listed here either has no `--json` (`export` uses `--format json`,
`ab` runs a live harness) or is exercised in its own module's tests.
"""

from __future__ import annotations

import contextlib
import importlib
import json
from typing import ClassVar

import pytest

from adder.cli import COMMANDS

# Extra argv each command needs before `--json` will do anything useful.
EXTRA: dict[str, list[str]] = {
    "verify": ["--since", "2026-08-01"],
    "policy": ["read the config file"],
    "classify": ["read the config file"],
    "pick": ["summarise a file"],
    "models": ["list"],
    "budget": ["--period", "all"],
    "outcomes": [],
    "ledger": [],
    "completion": ["bash"],
    # `live` looks up a directory, not a transcript root. Pointing it at one
    # with no session exercises the error path, which must also be JSON.
    "live": ["--cwd", "/nonexistent-directory-for-adder-tests"],
    # `handoff` prices a live session by working directory, like `live`. The
    # hypothetical mode is the one with no machine state in it, so it is the
    # one a contract test can assert on.
    "handoff": ["--context", "400000", "--remaining", "200"],
}

# Commands that do not read a transcript root, so must not be handed one.
NO_ROOT = {"policy", "classify", "pick", "models", "config", "outcomes", "ledger",
           "live", "completion", "routereval", "calib", "cascade", "frontier",
           "handoff", "design", "deadline", "place", "verbosity", "blend"}

# `export` offers JSON through `--format json` rather than `--json`, so the
# discovery below does not see it; `TestExportSurface` covers it directly.
# `ab` runs a live A/B harness and has no offline mode to assert on. Everything
# else with a `--json` flag is covered by the parametrized tests below,
# discovered from the source rather than listed here -- a hand-maintained list
# is a list that goes stale.
EXEMPT = {"export", "ab"}


def _records(n: int = 30) -> list[dict]:
    """A small but non-degenerate session: growing context, tools, a subagent."""
    out: list[dict] = []
    for i in range(n):
        out.append({
            "type": "assistant", "sessionId": "s", "effort": "high",
            "timestamp": f"2026-08-01T10:{i:02d}:00Z",
            "isSidechain": i == 5,
            "message": {"id": f"m{i}", "model": "claude-opus-5",
                        "usage": {"input_tokens": 2,
                                  "cache_read_input_tokens": 20_000 + 1_000 * i,
                                  "cache_creation_input_tokens": 100,
                                  "output_tokens": 400,
                                  "output_tokens_details": {"thinking_tokens": 50}},
                        "content": [{"type": "tool_use", "id": f"u{i}",
                                     "name": "Bash", "input": {}}]}})
        out.append({
            "type": "user", "sessionId": "s",
            "timestamp": f"2026-08-01T10:{i:02d}:30Z",
            "message": {"content": [{"type": "tool_result", "tool_use_id": f"u{i}",
                                     "content": "x" * 2_000}]}})
    return out


@pytest.fixture
def root(write_jsonl, tmp_path):
    """Transcripts in their own directory, not the temp root.

    `isolated_home` puts adder's own cache under `tmp_path`, and the read-only
    test below compares the transcript tree before and after. If the two shared
    a directory, the parse cache would show up as adder writing to your
    transcripts -- which is exactly the thing that test exists to detect, so it
    has to be able to tell them apart.
    """
    return write_jsonl(_records(), into=tmp_path / "projects")


def _json_commands():
    out = []
    for c in COMMANDS:
        if c.name in EXEMPT:
            continue
        mod = importlib.import_module(c.module)
        src = (mod.__file__ or "")
        with open(src, encoding="utf-8") as fh:
            if '"--json"' in fh.read():
                out.append(c.name)
    return out


JSON_COMMANDS = _json_commands()


class TestJsonSurface:
    def test_the_list_is_not_empty(self):
        assert len(JSON_COMMANDS) > 15, JSON_COMMANDS

    @pytest.mark.parametrize("name", JSON_COMMANDS)
    def test_emits_one_parseable_document(self, name, root, capsys, isolated_home):
        from adder.cli import BY_NAME

        mod = importlib.import_module(BY_NAME[name].module)
        argv = list(EXTRA.get(name, []))
        if name not in NO_ROOT:
            argv.append(str(root))
        argv.append("--json")

        rc = mod.main(argv)
        assert rc in (0, 1), f"{name} returned {rc}"
        out = capsys.readouterr().out
        assert out.strip(), f"{name} --json printed nothing"
        payload = json.loads(out)          # raises if anything else was printed
        assert isinstance(payload, (dict, list))

    @pytest.mark.parametrize("name", JSON_COMMANDS)
    def test_holds_no_nan_or_infinity(self, name, root, capsys, isolated_home):
        """`json.dumps` writes bare `NaN`/`Infinity`, which is not valid JSON.

        Python's own loader accepts them, so a round-trip test passes while
        every other parser in the world rejects the file.
        """
        from adder.cli import BY_NAME

        mod = importlib.import_module(BY_NAME[name].module)
        argv = list(EXTRA.get(name, []))
        if name not in NO_ROOT:
            argv.append(str(root))
        argv.append("--json")
        mod.main(argv)
        out = capsys.readouterr().out
        json.loads(out, parse_constant=_reject)


def _reject(token):
    raise AssertionError(f"non-finite JSON constant emitted: {token}")


class TestTextSurface:
    """Every command must also *run*, not merely answer `--help`.

    CI smoke-tests `--help` for each command, which proves the parser builds
    and nothing else. It would not have caught a report that raises on the
    first real record, or one whose JSON branch references a field the
    dataclass does not have -- both of which happened while this suite was
    being written.
    """

    # `ab` runs a live A/B against the API; `plan` runs a regime solver that
    # takes twenty seconds on real data and is covered by its own tests;
    # `bench` belongs to a different change in flight.
    RUNNABLE: ClassVar[list[str]] = [
        c.name for c in COMMANDS if c.name not in {"ab", "bench", "plan"}]

    def test_the_list_covers_most_of_the_table(self):
        assert len(self.RUNNABLE) > 20

    @pytest.mark.parametrize("name", RUNNABLE)
    def test_runs_against_a_real_fixture(self, name, root, capsys, isolated_home):
        from adder.cli import BY_NAME

        mod = importlib.import_module(BY_NAME[name].module)
        argv = list(EXTRA.get(name, []))
        if name not in NO_ROOT:
            argv.append(str(root))
        rc = mod.main(argv)
        assert rc in (0, 1), f"adder {name} exited {rc}"
        assert capsys.readouterr().out.strip(), f"adder {name} printed nothing"


class TestReadOnly:
    """CLAUDE.md rule 3, as an assertion instead of a promise.

    "The tool reads `~/.claude/projects/**`. It does not write there, rename
    there, or delete there. Ever." A static check cannot prove that; running
    every command over a fixture and comparing the tree before and after can.
    """

    @staticmethod
    def _snapshot(root):
        import hashlib

        out = {}
        for p in sorted(root.rglob("*")):
            if p.is_file():
                st = p.stat()
                out[str(p)] = (st.st_size, st.st_mtime_ns,
                               hashlib.sha1(p.read_bytes(),
                                            usedforsecurity=False).hexdigest())
        return out

    def test_no_command_touches_the_transcripts(self, root, capsys, isolated_home):
        from adder.cli import BY_NAME

        before = self._snapshot(root)
        assert before, "fixture produced no files to guard"

        for name in TestTextSurface.RUNNABLE:
            mod = importlib.import_module(BY_NAME[name].module)
            argv = list(EXTRA.get(name, []))
            if name not in NO_ROOT:
                argv.append(str(root))
            with contextlib.suppress(SystemExit):
                mod.main(argv)
            capsys.readouterr()

        after = self._snapshot(root)
        assert after == before, (
            "a command modified the transcript tree:\n"
            f"  appeared: {sorted(set(after) - set(before))}\n"
            f"  vanished: {sorted(set(before) - set(after))}\n"
            f"  changed:  {sorted(k for k in set(after) & set(before) if after[k] != before[k])}"
        )

    def test_export_still_refuses_to_overwrite_without_force(self, root, tmp_path):
        """The one command that writes at all writes only where it is told."""
        from adder.measure.spend.export import main

        dest = tmp_path / "taken.csv"
        dest.write_text("mine")
        assert main([str(root), "-o", str(dest)]) == 1
        assert dest.read_text() == "mine"


class TestExemptions:
    def test_export_offers_json_through_format(self, root, capsys):
        from adder.measure.spend.export import main

        assert main([str(root), "--format", "json"]) == 0
        assert json.loads(capsys.readouterr().out)["columns"]

    def test_every_exempt_command_still_exists(self):
        names = {c.name for c in COMMANDS}
        assert EXEMPT - names == set()


class TestExportSurface:
    """`export` is the one machine-readable surface the discovery above misses.

    It spells its flag `--format json`, so it fell outside the parametrized
    contract -- and it is the surface most likely to be piped straight into
    something else, which is the whole reason the contract exists.
    """

    GRAINS = ("turn", "session", "day")

    @pytest.mark.parametrize("grain", GRAINS)
    def test_json_is_one_parseable_document(self, grain, root, capsys, isolated_home):
        from adder.measure.spend import export

        assert export.main([str(root), "--grain", grain, "--format", "json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["rows"] and payload["columns"]

    @pytest.mark.parametrize("grain", GRAINS)
    def test_jsonl_is_one_document_per_line(self, grain, root, capsys, isolated_home):
        from adder.measure.spend import export

        assert export.main([str(root), "--grain", grain, "--format", "jsonl"]) == 0
        lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
        assert lines and all(isinstance(json.loads(ln), dict) for ln in lines)

    @pytest.mark.parametrize("grain", GRAINS)
    def test_no_nan_or_infinity(self, grain, root, capsys, isolated_home):
        from adder.measure.spend import export

        assert export.main([str(root), "--grain", grain, "--format", "json"]) == 0
        out = capsys.readouterr().out
        assert "NaN" not in out and "Infinity" not in out

    @pytest.mark.parametrize("grain", GRAINS)
    def test_csv_has_a_header_and_matching_widths(self, grain, root, capsys,
                                                  isolated_home):
        import csv as _csv
        import io as _io

        from adder.measure.spend import export

        assert export.main([str(root), "--grain", grain, "--format", "csv"]) == 0
        rows = list(_csv.reader(_io.StringIO(capsys.readouterr().out)))
        assert len(rows) > 1
        assert all(len(r) == len(rows[0]) for r in rows[1:])

    def test_no_message_content_leaves_the_tool(self, root, capsys, isolated_home):
        """The module's own promise: token counts, prices, ids and tool NAMES."""
        from adder.measure.spend import export

        assert export.main([str(root), "--grain", "turn", "--format", "json"]) == 0
        assert "x" * 100 not in capsys.readouterr().out
