"""Every text file this tool reads or writes is UTF-8, whatever the locale is.

Transcripts under `~/.claude/projects` are UTF-8: they hold prompts, source
code and file paths in whatever script the user works in. Python below 3.15
opens text files in the *locale's* encoding, which under `LC_ALL=C` is ASCII.

Two failures follow, and both are silent:

* A read with `errors="replace"` turns each multi-byte character into three
  replacement characters. `est_tokens` is `len(text) / 4`, so the same tool
  result measures **750 tokens instead of 250** -- and that number is what the
  size model the PreToolUse guard predicts from is learned off.
* A read without `errors=` raises `UnicodeDecodeError`, which is a `ValueError`
  and is caught by the `except (OSError, ValueError)` every loader here wraps
  itself in. The catalog, the ledger and the outcome log all come back empty
  with nothing reported.

So this is a repository-level invariant rather than a unit test: no text-mode
open in the package may inherit the locale.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[2] / "adder"

# `open` in binary mode carries no encoding, and `os.open` is a file descriptor.
_BINARY = ("rb", "wb", "ab", "r+b", "w+b")


def _text_io_calls():
    """Every `.read_text`/`.write_text`/`.open` call site in the package."""
    for path in sorted(PKG.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            name = node.func.attr
            if name not in ("read_text", "write_text", "open"):
                continue
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
                continue                      # os.open takes a descriptor
            mode = next((a.value for a in node.args
                         if isinstance(a, ast.Constant) and isinstance(a.value, str)), "")
            if mode in _BINARY:
                continue
            yield path, node.lineno, name, {k.arg for k in node.keywords}


def test_every_text_open_names_its_encoding():
    missing = [f"{p.relative_to(PKG.parent)}:{line} {name}()"
               for p, line, name, kwargs in _text_io_calls()
               if "encoding" not in kwargs]
    assert not missing, (
        "text I/O inheriting the locale encoding:\n  " + "\n  ".join(missing))


@pytest.fixture
def ascii_locale(monkeypatch):
    """Make `locale.getpreferredencoding` report ASCII, as `LC_ALL=C` does."""
    import locale

    monkeypatch.setattr(locale, "getpreferredencoding", lambda *a, **k: "ascii")
    return None


def test_a_transcript_scan_sizes_the_same_under_an_ascii_locale(tmp_path, ascii_locale):
    from adder.core.shapes import iter_results

    d = tmp_path / "proj"
    d.mkdir()
    rows = [
        {"type": "assistant", "sessionId": "s", "timestamp": "2026-08-01T10:00:00Z",
         "message": {"id": "a1", "model": "claude-opus-5",
                     "usage": {"input_tokens": 10, "output_tokens": 20},
                     "content": [{"type": "tool_use", "id": "t1", "name": "Bash",
                                  "input": {"command": "ls"}}]}},
        {"type": "user", "sessionId": "s", "timestamp": "2026-08-01T10:01:00Z",
         "message": {"role": "user",
                     "content": [{"type": "tool_result", "tool_use_id": "t1",
                                  "content": "結果" * 500}]}},
    ]
    (d / "s.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    got = list(iter_results(tmp_path))
    # 1000 characters at 4 chars/token. Decoded as ASCII with replacement it
    # would be 3000 characters and 750 tokens.
    assert [size for _, _, size in got] == [250]
