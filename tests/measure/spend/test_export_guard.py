"""`export -o` never writes into a transcript directory, whatever the flags say.

`export -o <root>/proj/s1.jsonl --force` replaced a transcript with its CSV:
`--force` was the only guard, and it guards against the wrong thing. CLAUDE.md
rule 3 forbids writing under the transcript tree at all, so these refuse with
exit status 2 and leave the file byte-identical.
"""
from __future__ import annotations

import json

import pytest

RECORD = {"type": "assistant", "sessionId": "s", "timestamp": "2026-08-01T10:00:00Z",
          "message": {"id": "m1", "model": "claude-opus-5",
                      "usage": {"input_tokens": 1, "cache_read_input_tokens": 900,
                                "output_tokens": 10}, "content": []}}


@pytest.fixture
def root(tmp_path):
    d = tmp_path / "projects" / "proj"
    d.mkdir(parents=True)
    (d / "s1.jsonl").write_text(json.dumps(RECORD))
    return tmp_path / "projects"


def _run(*argv):
    from adder.measure.spend.export import main

    return main([str(a) for a in argv])


class TestRefusesTheTranscriptRoot:
    def test_overwriting_a_transcript_is_refused_even_with_force(self, root, capsys):
        target = root / "proj" / "s1.jsonl"
        before = target.read_bytes()
        assert _run(root, "-o", target, "--force") == 2
        assert target.read_bytes() == before
        assert "transcript directory" in capsys.readouterr().err

    def test_a_new_file_inside_the_root_is_refused(self, root, capsys):
        assert _run(root, "-o", root / "proj" / "out.csv") == 2
        assert not (root / "proj" / "out.csv").exists()

    def test_a_symlink_pointing_in_is_refused(self, root, tmp_path, capsys):
        link = tmp_path / "innocent.csv"
        link.symlink_to(root / "proj" / "s1.jsonl")
        before = (root / "proj" / "s1.jsonl").read_bytes()
        assert _run(root, "-o", link, "--force") == 2
        assert (root / "proj" / "s1.jsonl").read_bytes() == before

    def test_the_default_claude_projects_tree_is_refused(self, root, tmp_path,
                                                         monkeypatch, capsys):
        """Protected even when a different root is being read."""
        home = tmp_path / "home"
        (home / ".claude" / "projects").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        dest = home / ".claude" / "projects" / "x.csv"
        assert _run(root, "-o", dest) == 2
        assert not dest.exists()

    def test_outside_the_root_still_writes(self, root, tmp_path, capsys):
        dest = tmp_path / "exports" / "turns.csv"
        assert _run(root, "-o", dest) == 0
        assert dest.read_text().startswith("timestamp,")


class TestProtectedRoot:
    def test_resolves_dot_dot(self, tmp_path):
        from adder.measure.spend.export import protected_root

        r = tmp_path / "r"
        (r / "a").mkdir(parents=True)
        assert protected_root(r / "a" / ".." / "x.csv", [r]) == r
        assert protected_root(tmp_path / "rr" / "x.csv", [r]) is None
