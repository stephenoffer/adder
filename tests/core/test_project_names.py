"""Which project a transcript belongs to, and what a person is shown for it.

Two failures, both visible in `adder sessions` on a real machine:

* a subagent transcript lives at `<slug>/<session>/subagents/agent-*.jsonl`,
  so labelling by parent directory made "subagents" a project -- and, because
  the subagent's turns carry the parent session id, made it the label of whole
  sessions;
* the slug was truncated to fit a column from the left, printing
  `phen-offer-Desktop-wolfgang-v2` for `~/Desktop/wolfgang-v2`.
"""

from __future__ import annotations

import json
from pathlib import Path

from adder.core.trace import iter_file, project_name, project_slug


def _slug(path: Path) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in str(path))


class TestProjectSlug:
    def test_a_main_transcript_is_its_parent_directory(self):
        path = Path("/x/projects/-Users-jo-app/0f3a.jsonl")
        assert project_slug(path) == "-Users-jo-app"

    def test_a_subagent_transcript_belongs_to_the_session_project(self):
        path = Path("/x/projects/-Users-jo-app/0f3a/subagents/agent-1.jsonl")
        assert project_slug(path) == "-Users-jo-app"

    def test_iter_file_labels_subagent_turns_with_the_project(self, tmp_path):
        path = tmp_path / "-Users-jo-app" / "0f3a" / "subagents" / "agent-1.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            "type": "assistant", "sessionId": "0f3a", "isSidechain": True,
            "message": {"id": "m1", "model": "claude-opus-5",
                        "usage": {"input_tokens": 10, "output_tokens": 5}},
        }) + "\n")
        (turn,) = list(iter_file(path))
        assert turn.project == "-Users-jo-app"


class TestProjectName:
    def test_drops_home_and_an_existing_container(self, isolated_home):
        (isolated_home / "Desktop").mkdir()
        assert project_name(_slug(isolated_home) + "-Desktop-wolfgang-v2") == "wolfgang-v2"

    def test_keeps_a_container_word_that_is_not_a_directory(self, isolated_home):
        assert project_name(_slug(isolated_home) + "-Desktop-app") == "Desktop-app"

    def test_a_directory_named_like_a_container_is_not_split(self, isolated_home):
        (isolated_home / "code").mkdir()
        (isolated_home / "code-review").mkdir()
        assert project_name(_slug(isolated_home) + "-code-review") == "code-review"

    def test_home_itself(self, isolated_home):
        assert project_name(_slug(isolated_home)) == "~"

    def test_outside_home_loses_only_the_leading_dash(self, isolated_home):
        assert project_name("-srv-build-app") == "srv-build-app"

    def test_a_slug_that_is_not_a_path_is_unchanged(self, isolated_home):
        assert project_name("subagents") == "subagents"
