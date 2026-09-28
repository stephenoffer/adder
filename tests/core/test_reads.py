"""What a call read: the identity half of the guard, which `Read` used to own.

Two failure directions, and they are not symmetric. Missing a path costs a
saving that was available. Inventing one costs a *refusal* of a read that was
needed, in the only component here that changes what happens -- so most of this
file is about words that look like filenames and are not: a `sed` script, half
of a quoted grep pattern, a glob, a redirect target.

The other half is the whole/slice line. `cat f` puts the file in the context;
`sed -n '1,50p' f` puts fifty lines there, and recording the second as the
first is how the guard ends up refusing the read that would have got the rest.
"""

from __future__ import annotations

import pytest

from adder.core import reads


def paths(command):
    return sorted(t.path for t in reads.read_targets(command))


def whole(command):
    return sorted(t.path for t in reads.read_targets(command) if t.whole)


class TestWholeFileReads:
    @pytest.mark.parametrize("cmd", ["cat a.py", "bat a.py", "nl a.py", "tac a.py"])
    def test_a_dump_admits_the_file(self, cmd):
        assert whole(cmd) == ["a.py"]

    def test_several_files_are_all_whole(self):
        assert whole("cat a.py b.py") == ["a.py", "b.py"]

    def test_a_sequence_reads_both(self):
        assert whole("cat a.py; cat b.py") == ["a.py", "b.py"]

    def test_flags_are_not_files(self):
        assert whole("cat -n a.py") == ["a.py"]

    def test_a_double_dash_ends_the_flags(self):
        assert whole("cat -- -weird.txt") == ["-weird.txt"]


class TestSlicesAreNotWholeReads:
    @pytest.mark.parametrize("cmd", [
        "sed -n '1,50p' a.py",
        "head -20 a.py",
        "tail -n +5 a.py",
        "awk 'NR<=10' a.py",
        "grep -n pat a.py",
        "cut -d, -f1 a.py",
    ])
    def test_a_bounded_reader_reads_part_of_it(self, cmd):
        assert paths(cmd) == ["a.py"]
        assert whole(cmd) == []

    def test_a_pipeline_filters_what_the_dump_emitted(self):
        # `cat f | grep x` admits matches, exactly like `grep x f` does. The
        # guard may not treat the file as resident on the strength of it.
        assert paths("cat a.py | grep x") == ["a.py"]
        assert whole("cat a.py | grep x") == []

    def test_a_dump_bounded_by_head_is_a_slice(self):
        assert whole("cat a.py | head -5") == []


class TestWordsThatAreNotFiles:
    def test_a_sed_script_is_not_a_path(self):
        assert paths("sed -n '1,50p' a.py") == ["a.py"]

    def test_half_a_quoted_grep_pattern_is_not_a_path(self):
        # The splitter is whitespace-based (a real transcript contains
        # unbalanced quotes and a hook may not raise), so `'def foo'` arrives
        # as two words and the second one ends in a quote.
        assert paths("grep -n 'def foo' src/x.py") == ["src/x.py"]

    def test_a_flag_value_is_not_a_path(self):
        assert paths("head -n 20 a.py") == ["a.py"]
        assert paths("cut -d , -f 1 a.py") == ["a.py"]

    @pytest.mark.parametrize("cmd", ["cat $FILE", "cat *.py", "cat {a,b}.py",
                                     "cat `which x`", "cat a[12].py"])
    def test_shell_expansion_is_never_a_path(self, cmd):
        assert paths(cmd) == []

    def test_a_dot_is_a_tree_walk_not_a_file(self):
        assert paths("grep -rn foo .") == []

    def test_a_redirect_target_is_not_read(self):
        assert paths("cat a.py > out.txt") == []

    def test_a_heredoc_body_is_data(self):
        assert paths("cat <<'EOF' > f\nhello\nEOF") == []

    def test_sed_in_place_is_an_edit(self):
        assert paths("sed -i '' 's/a/b/' a.py") == []
        assert paths("sed -i.bak s/a/b/ a.py") == []

    def test_a_command_that_moves_is_not_read_at_all(self):
        # `cd` changes what a relative path means, and this module resolves
        # against one directory. Skipping costs a saving; guessing costs a
        # refusal of a file that was never read.
        assert paths("cd /elsewhere && cat a.py") == []

    def test_a_program_that_admits_no_content_is_not_a_read(self):
        assert paths("wc -l a.py") == []


class TestResolve:
    def test_an_absolute_path_is_normalised(self):
        assert reads.resolve("/a/b/../c.py") == "/a/c.py"

    def test_a_relative_path_needs_a_directory(self):
        assert reads.resolve("c.py") == ""

    def test_a_relative_path_resolves_against_the_one_given(self):
        assert reads.resolve("c.py", "/repo") == "/repo/c.py"

    def test_nothing_resolves_to_nothing(self):
        assert reads.resolve("") == ""


class TestToolTargets:
    def test_a_list_of_targets_can_be_sorted(self):
        """`frozen=True` without `order=True` raises on `sorted()`."""
        ts = reads.tool_targets("Bash", {"command": "cat /b.py /a.py"})
        assert [t.path for t in sorted(ts)] == ["/a.py", "/b.py"]

    def test_read_and_cat_name_the_same_file(self):
        by_tool = reads.tool_targets("Read", {"file_path": "/repo/a.py"})
        by_shell = reads.tool_targets("Bash", {"command": "cat a.py"}, cwd="/repo")
        assert [t.path for t in by_tool] == [t.path for t in by_shell] == ["/repo/a.py"]
        assert by_tool[0].whole and by_shell[0].whole

    def test_a_bounded_read_is_a_slice_on_both_sides(self):
        r = reads.tool_targets("Read", {"file_path": "/a.py", "limit": 50})
        b = reads.tool_targets("Bash", {"command": "sed -n '1,50p' /a.py"})
        assert not r[0].whole
        assert not b[0].whole

    def test_an_offset_read_is_a_slice(self):
        r = reads.tool_targets("Read", {"file_path": "/a.py", "offset": 500})
        assert not r[0].whole

    def test_a_relative_shell_path_without_a_cwd_is_dropped(self):
        assert reads.tool_targets("Bash", {"command": "cat a.py"}) == []

    def test_a_tool_that_reads_nothing_has_no_targets(self):
        assert reads.tool_targets("WebFetch", {"url": "https://x"}) == []

    def test_a_missing_input_is_survivable(self):
        assert reads.tool_targets("Read", None) == []
        assert reads.tool_targets("Bash", {}) == []


class TestWholeReads:
    def test_a_file_the_harness_would_truncate_is_not_admitted(self, tmp_path):
        # A `cat` of a file over the harness's output ceiling returns a
        # truncated result, so the file is *not* in the context -- and a guard
        # that thinks it is refuses the read that would have got the rest.
        big = tmp_path / "big.log"
        big.write_text("x" * 5_000)
        cmd = {"command": f"cat {big}"}
        assert reads.whole_reads("Bash", cmd, max_chars=10_000) == [str(big)]
        assert reads.whole_reads("Bash", cmd, max_chars=1_000) == []

    def test_a_file_that_is_not_there_is_not_admitted(self, tmp_path):
        cmd = {"command": f"cat {tmp_path / 'gone.py'}"}
        assert reads.whole_reads("Bash", cmd) == []

    def test_a_slice_never_counts_however_small(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x")
        assert reads.whole_reads("Bash", {"command": f"sed -n '1,5p' {f}"}) == []

    def test_the_ceiling_comes_from_the_harness(self, monkeypatch):
        monkeypatch.setenv("BASH_MAX_OUTPUT_LENGTH", "1234")
        assert reads.max_bash_output_chars() == 1234

    @pytest.mark.parametrize("value", ["", "not-a-number", "0", "-5"])
    def test_an_unusable_ceiling_falls_back(self, monkeypatch, value):
        monkeypatch.setenv("BASH_MAX_OUTPUT_LENGTH", value)
        assert reads.max_bash_output_chars() == reads.DEFAULT_MAX_BASH_OUTPUT_CHARS

    def test_whole_is_a_claim_about_the_command_not_the_context(self, tmp_path):
        """The two APIs answer different questions, and the docstring says so.

        `tool_targets` never opens a file, so it cannot know that the harness
        truncated the result; `whole_reads` stats it and applies the ceiling.
        A refusal may rest on the second and never on the first.
        """
        big = tmp_path / "big.log"
        big.write_text("x" * 60_000)
        cmd = {"command": f"cat {big}"}
        assert reads.tool_targets("Bash", cmd)[0].whole is True
        assert reads.whole_reads("Bash", cmd, max_chars=30_000) == []

    @pytest.mark.parametrize(("command", "want"), [
        ("cat a.py", True), ("cat a.py | head -5", True), ("sed -n 1,20p a.py", True),
        ("head -50 a.py b.py", True), ("sed s/a/b/ a.py", False),
        ("sed -n /foo/p a.py", False), ("grep -n foo a.py", False),
        ("cat a.py && pytest", False), ("diff a.py b.py", False),
        ("tail -f log", False), ("awk 1 a.py", False), ("", False)])
    def test_only_prints_files(self, command, want):
        assert reads.only_prints_files(command) is want

    def test_the_shell_ceiling_is_not_applied_to_read(self, tmp_path):
        # The Bash ceiling is a fact about shell output. A file over it that the
        # Read tool returns whole is still a whole read.
        f = tmp_path / "a.py"
        f.write_text(("x" * 79 + "\n") * 500)
        assert reads.whole_reads("Read", {"file_path": str(f)}, max_chars=1_000) == [str(f)]

    def test_a_read_the_tool_truncates_is_not_whole(self, tmp_path):
        """The harness paginates Read: 2,000 lines, 2,000 characters a line,
        and an error past its token ceiling. None of those put the file in the
        context, and remembering one as read refuses the offset read that
        would have got the rest."""
        long = tmp_path / "long.py"
        long.write_text("x\n" * 2_001)
        wide = tmp_path / "wide.py"
        wide.write_text("x" * 2_001)
        huge = tmp_path / "huge.py"
        huge.write_text(("x" * 79 + "\n") * 1_900)
        for f in (long, wide, huge):
            assert reads.whole_reads("Read", {"file_path": str(f)}) == [], f.name

    def test_exactly_the_line_ceiling_is_whole(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x\n" * 2_000)
        assert reads.whole_reads("Read", {"file_path": str(f)}) == [str(f)]

    def test_a_missing_or_binary_file_is_not_whole(self, tmp_path):
        png = tmp_path / "a.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n\0\0")
        assert reads.whole_reads("Read", {"file_path": str(png)}) == []
        assert reads.whole_reads("Read", {"file_path": str(tmp_path / "nope.py")}) == []

    def test_the_token_ceiling_follows_the_harness_setting(self, tmp_path, monkeypatch):
        f = tmp_path / "a.py"
        f.write_text(("x" * 79 + "\n") * 500)            # 40,000 chars
        monkeypatch.setenv("CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS", "5000")
        assert reads.whole_reads("Read", {"file_path": str(f)}) == []
