"""Size prediction, which is what the guard decides on.

The constant this module replaced -- 15,000 tokens for any command matching
`cat ` or `git log` -- overstated the measured median by 105x. The tests that
matter here are therefore not "does it return a number" but "does it return a
number for the right reason, and does it say so when it is guessing".
"""

from __future__ import annotations

import json

import pytest

from adder.core.shapes import (
    PRIOR,
    TOKENS_PER_LINE,
    Estimate,
    SizeModel,
    _quantile,
    empty_model,
    is_bounded,
    load_model,
    pipelines,
    read_estimate,
    segments,
    shape,
)


class TestBounding:
    """Whether output is capped by construction is a shell question, not a guess."""

    @pytest.mark.parametrize("cmd", [
        "wc -l a.ts b.tsx",
        "sed -n '1,400p' foo.jsonl",
        "grep -rn foo src/ | head -50",
        "git log --oneline | head -30",
        "python3 -m pytest -q 2>&1 | tail -15",
        "npx eslint . > /tmp/out.txt",
        "git status --short | head -20 && git diff --stat",
        "grep -c pattern file.txt",
        "grep -l pattern src/",
    ])
    def test_bounded(self, cmd):
        assert is_bounded(cmd), f"{cmd!r} caps its own output"

    @pytest.mark.parametrize("cmd", [
        "cd ~/x && cat src/measure.py",
        "cargo test 2>&1 | grep -vE '^warning'",
        'for f in a.py b.py; do echo "== $f"; cat $f; done',
        "cd /Users/x; git diff --stat; echo done",
        "head -1 f.txt && cat huge.log",
        "ls -la",
        "cat a.txt > /tmp/x; cat b.txt",
    ])
    def test_unbounded(self, cmd):
        assert not is_bounded(cmd), f"{cmd!r} can return arbitrarily much"

    def test_a_pipeline_is_decided_by_its_last_stage(self):
        """`cat huge | head` is small however big `huge` is."""
        assert is_bounded("cat huge.log | head -20")
        assert not is_bounded("head -20 small.log | cat")

    def test_a_sequence_needs_every_command_bounded(self):
        """The old matcher searched the whole string, so a trailing `echo`
        made `git diff --stat; echo done` look bounded when the diff is not."""
        assert is_bounded("wc -l a; wc -l b")
        assert not is_bounded("wc -l a; cat b")

    def test_a_filter_is_not_a_limit(self):
        """`grep -v` changes the output; it does not cap it."""
        assert not is_bounded("cargo build 2>&1 | grep -v warning")

    def test_a_sed_range_is_a_bounded_read(self):
        """`-n ` was in the old 'already bounded' list to catch `grep -n`, and
        it waved through every `sed -n '1,600p'` in the corpus."""
        assert is_bounded("sed -n '1,200p' big.py")
        assert not is_bounded("sed 's/a/b/' big.py")

    def test_grep_dash_n_is_not_a_bound(self):
        assert not is_bounded("grep -n pattern big.log")

    def test_empty_and_garbage_never_raise(self):
        for cmd in ["", "   ", "|||", "'unterminated", "&&"]:
            assert is_bounded(cmd) in (True, False)


class TestShape:
    def test_paths_and_arguments_are_dropped(self):
        """A key that includes the filename has a sample size of one."""
        assert shape("cat src/a.ts") == shape("cat lib/b.py") == "cat"

    def test_a_leading_cd_is_not_the_command(self):
        assert shape("cd ~/project && cat file.py") == "cat"

    def test_pipeline_order_is_kept(self):
        assert shape("git log | head -30") == "git log|head+30"

    def test_a_subcommand_is_part_of_the_shape(self):
        """`git log` and `git status` have nothing to do with each other."""
        assert shape("git diff") != shape("git status")

    def test_wrappers_are_skipped(self):
        assert shape("time sudo cat f") == "cat"

    def test_nothing_runnable_is_a_known_key(self):
        assert shape("") == "?"

    def test_segments_and_pipelines_agree_on_the_programs(self):
        cmd = "cat a | grep b; wc -l c"
        assert [p for p, _ in segments(cmd)] == ["cat", "grep", "wc"]
        assert [len(stages) for stages in pipelines(cmd)] == [2, 1]


class TestReadEstimate:
    def test_a_file_is_sized_from_the_filesystem(self, tmp_path):
        f = tmp_path / "big.py"
        f.write_text("x" * 40_000)
        assert read_estimate({"file_path": str(f)}).p90 == 10_000

    def test_a_bounded_read_is_charged_for_what_it_asked_for(self, tmp_path):
        """The old guard returned zero for any read with a limit, which made
        `limit: 100000` invisible to it."""
        f = tmp_path / "big.py"
        f.write_text("x" * 400_000)
        whole = read_estimate({"file_path": str(f)}).p90
        part = read_estimate({"file_path": str(f), "limit": 50}).p90
        assert 0 < part < whole

    def test_a_limit_can_never_exceed_the_file(self, tmp_path):
        f = tmp_path / "small.py"
        f.write_text("x" * 400)
        assert read_estimate({"file_path": str(f), "limit": 100_000}).p90 == 100

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        assert read_estimate({"file_path": str(tmp_path / "nope")}).p90 == 0

    def test_no_file_path_is_not_an_error(self):
        assert read_estimate({}).p90 == 0


class TestPrediction:
    @staticmethod
    def _model():
        return SizeModel(
            shapes={"cat": (200, 6_000, 40), "rare": (10, 20, 1)},
            heads={"cat": (200, 6_000, 40), "jq": (50, 900, 12)},
            tools={"WebFetch": (3_000, 9_000, 20)},
            built=1.0, calls=52,
        )

    def test_an_exact_shape_wins(self):
        e = self._model().predict_command("cd x && cat f.py")
        assert (e.p90, e.source, e.n) == (6_000, "shape", 40)

    def test_it_backs_off_to_the_program(self):
        e = self._model().predict_command("jq '.a' file.json")
        assert e.source == "head" and e.p90 == 900

    def test_it_backs_off_to_the_prior_when_nothing_is_known(self):
        e = self._model().predict_command("some-unheard-of-binary --all")
        assert e.source == "prior" and e.p90 == PRIOR["Bash"][1] and e.n == 0

    def test_one_observation_is_not_a_distribution(self):
        """A shape whose whole history is a single outlier must not be quoted
        as if it were evidence."""
        e = self._model().predict_command("rare --thing")
        assert e.source != "shape"

    def test_the_prior_is_not_dressed_up_as_a_measurement(self):
        e = self._model().predict_command("unknown-tool")
        assert not e.measured and "prior" in e.describe()

    def test_a_measurement_says_how_many_observations(self):
        assert "40 local calls" in self._model().predict_command("cat f").describe()

    def test_an_empty_model_answers_from_the_prior_for_every_tool(self):
        m = empty_model()
        for tool in ("Bash", "Grep", "Glob", "WebFetch", "Task"):
            assert m.predict_tool(tool, {}).source == "prior"

    def test_read_is_measured_not_predicted(self, tmp_path):
        """A file's size is on disk; there is no reason to guess at it."""
        f = tmp_path / "f.py"
        f.write_text("x" * 8_000)
        assert self._model().predict_tool("Read", {"file_path": str(f)}).source == "stat"

    def test_a_tool_with_history_uses_it(self):
        assert self._model().predict_tool("WebFetch", {}).source == "shape"


class TestLearning:
    @staticmethod
    def _records(cmd, sizes):
        """One tool_use/tool_result pair per size, as Claude Code writes them."""
        out = []
        for i, n in enumerate(sizes):
            uid = f"u{i}"
            out.append({"type": "assistant", "message": {
                "id": f"m{i}", "content": [
                    {"type": "tool_use", "id": uid, "name": "Bash",
                     "input": {"command": cmd}}]}})
            out.append({"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": uid, "content": "x" * (n * 4)}]}})
        return out

    def test_it_learns_a_shape_from_transcripts(self, write_jsonl, tmp_path):
        d = write_jsonl(self._records("cat f.py", [100, 200, 300, 9_000]),
                        into=tmp_path / "proj")
        m = SizeModel.learn(d)
        e = m.predict_command("cat other.py")
        assert e.source == "shape" and e.n == 4
        assert e.p50 < e.p90, "a distribution, not a point"

    def test_bounded_calls_are_learned_too(self, write_jsonl, tmp_path):
        """Excluding them would bias every shape upward, and the guard needs to
        know that `git log | head` is small so it stays quiet about it."""
        d = write_jsonl(self._records("git log | head -5", [10, 12, 14]),
                        into=tmp_path / "proj")
        assert SizeModel.learn(d).shapes.get("git log|head+5", (0, 0, 0))[2] == 3

    def test_an_unanswered_call_contributes_no_size(self, write_jsonl, tmp_path):
        """Imputing zero for a call the session never got a reply to would drag
        down a quantile the guard reads directly."""
        recs = [{"type": "assistant", "message": {"id": "m", "content": [
            {"type": "tool_use", "id": "u", "name": "Bash",
             "input": {"command": "cat f"}}]}}]
        m = SizeModel.learn(write_jsonl(recs, into=tmp_path / "proj"))
        assert m.calls == 0

    def test_learning_from_an_empty_tree_is_not_an_error(self, tmp_path):
        assert SizeModel.learn(tmp_path).calls == 0


class TestPersistence:
    def test_round_trip(self, tmp_path):
        m = SizeModel(shapes={"cat": (1, 2, 3)}, heads={}, tools={}, built=5.0, calls=3)
        m.save(tmp_path / "sizes.json")
        back = load_model(tmp_path / "sizes.json")
        assert back.shapes == m.shapes and back.calls == 3

    def test_the_write_is_atomic(self, tmp_path):
        """A hook may be reading this file while another writes it."""
        p = tmp_path / "sizes.json"
        SizeModel(calls=1, built=1.0).save(p)
        assert p.exists() and not p.with_suffix(".tmp").exists()

    @pytest.mark.parametrize("blob", ["", "{", "[]", '{"version": 999}', "null"])
    def test_a_corrupt_model_degrades_to_the_prior(self, tmp_path, blob):
        """A guard that raises on a bad cache file is a guard that has stopped
        guarding, and the tool call still succeeds so nobody notices."""
        p = tmp_path / "sizes.json"
        p.write_text(blob)
        assert load_model(p).predict_command("cat f").source == "prior"

    def test_a_missing_model_degrades_to_the_prior(self, tmp_path):
        assert load_model(tmp_path / "absent.json").calls == 0

    def test_a_future_version_is_not_read_as_this_one(self, tmp_path):
        p = tmp_path / "sizes.json"
        p.write_text(json.dumps({"version": 99, "shapes": {"cat": [1, 2, 3]}}))
        assert load_model(p).shapes == {}


class TestEstimateContract:
    def test_measured_is_exactly_having_observations(self):
        assert Estimate(1, 2, 3, "shape").measured
        assert not Estimate(1, 2, 0, "prior").measured

    def test_a_file_size_counts_as_measured(self):
        """It is the actual thing about to be admitted, not a quantile over
        past calls. Describing a byte count as "no local history" labels a
        measurement as a guess."""
        e = Estimate(1_000, 1_000, 0, "stat")
        assert e.measured and "size on disk" in e.describe()

    def test_the_prior_is_still_named_as_a_prior(self):
        assert "prior" in Estimate(1, 2, 0, "prior").describe()

    def test_the_prior_covers_every_guarded_tool(self):
        """A tool with no prior entry falls to `*`; that is fine, but the ones
        the guard names should be deliberate rather than incidental."""
        for tool in ("Bash", "Grep", "Glob", "WebFetch", "Task"):
            assert tool in PRIOR


class TestQuotingAndHeredocs:
    """A regex split produced 12,208 shapes from 27,643 calls, because every
    regex argument containing `|` became its own program. Almost all of them
    then had a sample size of one, below the evidence floor, so the guard fell
    back to the shipped prior for nearly everything."""

    def test_a_pipe_inside_a_quoted_pattern_is_not_a_pipeline(self):
        cmd = 'cargo test 2>&1 | grep -vE "^warning|^\\s+-->"'
        assert shape(cmd) == "cargo test|grep"

    def test_a_semicolon_inside_quotes_is_not_a_sequence(self):
        assert shape('echo "a; b" && cat f') == "echo|cat"

    def test_an_arrow_inside_a_pattern_is_not_a_redirect(self):
        """`-->` in a grep pattern marked the command bounded and silenced the
        guard on everything it piped."""
        assert not is_bounded('cargo build 2>&1 | grep -vE "^note|-->"')

    def test_stderr_plumbing_is_not_an_output_file(self):
        assert not is_bounded("cat f 2>&1")
        assert not is_bounded("echo hi >&2")

    def test_a_real_redirect_still_bounds(self):
        assert is_bounded("npx eslint . > /tmp/out.txt")
        assert is_bounded("cat f >> /tmp/log")

    def test_a_heredoc_body_is_data_not_a_pipeline(self):
        cmd = "python3 <<'EOF'\nimport os\nfor x in range(3): print(x)\nEOF"
        assert "for" not in shape(cmd) and "import" not in shape(cmd)

    def test_a_command_after_a_heredoc_is_still_seen(self):
        cmd = "cat <<'EOF' > /tmp/f\nbody\nEOF\ncat /tmp/f"
        assert shape(cmd).endswith("cat")

    def test_an_unterminated_quote_never_raises(self):
        """`shlex.split` raises on this, and a parser that raises inside a
        PreToolUse hook is a guard that has silently stopped guarding."""
        for cmd in ["echo 'unterminated", 'grep "half', "cat 'a|b"]:
            assert isinstance(shape(cmd), str)
            assert is_bounded(cmd) in (True, False)


class TestPathsResolveAtCallTime:
    """A constant captured at import time is one no test can redirect and no
    `.adder.json` can override -- and this one names a file under the user's
    home, so an import-time capture means the suite reads and writes the
    developer's own model."""

    def test_the_model_path_follows_the_environment(self, monkeypatch, tmp_path):
        from adder.core.shapes import model_path

        monkeypatch.setenv("ADDER_SIZE_MODEL", str(tmp_path / "sizes.json"))
        assert model_path() == tmp_path / "sizes.json"

    def test_it_is_not_captured_once(self, monkeypatch, tmp_path):
        from adder.core.shapes import model_path

        monkeypatch.setenv("ADDER_SIZE_MODEL", str(tmp_path / "a.json"))
        first = model_path()
        monkeypatch.setenv("ADDER_SIZE_MODEL", str(tmp_path / "b.json"))
        assert model_path() != first

    def test_the_max_age_follows_the_environment(self, monkeypatch):
        from adder.core.shapes import max_age_s

        monkeypatch.setenv("ADDER_SIZE_MAX_AGE", "60")
        assert max_age_s() == 60.0

    def test_refresh_does_not_rescan_a_fresh_model(self, monkeypatch, tmp_path):
        """The whole point of the cache: `adder guard` and `doctor` may scan,
        and must not scan every time they are run."""
        from adder.core.shapes import SizeModel, refresh

        p = tmp_path / "sizes.json"
        SizeModel(shapes={"cat": (1, 2, 5)}, built=__import__("time").time(),
                  calls=5).save(p)

        def explode(*a, **k):                  # pragma: no cover - must not run
            raise AssertionError("rescanned a model that was still fresh")

        monkeypatch.setattr(SizeModel, "learn", classmethod(explode))
        assert refresh(tmp_path, path=p).calls == 5

    def test_refresh_rescans_a_stale_model(self, monkeypatch, tmp_path, write_jsonl):
        from adder.core.shapes import SizeModel, refresh

        p = tmp_path / "sizes.json"
        SizeModel(shapes={"cat": (1, 2, 5)}, built=1.0, calls=5).save(p)
        recs = [
            {"type": "assistant", "message": {"id": "m", "content": [
                {"type": "tool_use", "id": "u", "name": "Bash",
                 "input": {"command": "ls -la"}}]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "u", "content": "x" * 400}]}},
        ]
        d = write_jsonl(recs, into=tmp_path / "proj")
        assert refresh(d, path=p, max_age=1.0).calls == 1


class TestThePrior:
    """The first version of this table was invented and was wrong in the
    expensive direction on every line -- `WebFetch` quoted at 12,000 tokens p90
    against a measured 595. A guard whose fallback over-states interrupts
    constantly on a machine that has learned nothing yet, which is the failure
    the size model was written to remove; it had simply moved from the hook
    into the default."""

    def test_every_entry_is_a_distribution(self):
        for tool, (p50, p90) in PRIOR.items():
            assert 0 < p50 <= p90, f"{tool} is not ordered"

    def test_it_says_where_it_came_from(self):
        from adder.core.shapes import PRIOR_SOURCE

        assert PRIOR_SOURCE["transcripts"] > 0
        assert "unbounded" in PRIOR_SOURCE["population"]

    def test_every_guarded_tool_has_an_entry(self):
        from adder.decide.guard import GUARDED

        for tool in GUARDED:
            assert tool in PRIOR or "*" in PRIOR

    def test_nothing_without_evidence_clears_the_guards_floor(self):
        """`Grep`, `Glob` and `Task` have no local observations, so they inherit
        the pooled fallback. Below the floor nothing fires without evidence: on
        a fresh machine only `Read` is guarded, and `Read` is sized off the
        filesystem rather than predicted."""
        from adder.decide.guard import Settings

        floor = Settings().min_tokens
        for tool in ("Grep", "Glob"):
            assert PRIOR[tool][1] < floor, f"{tool} would fire on no evidence"
        assert PRIOR["*"][1] < floor

    def test_the_generic_fallback_is_not_the_largest_entry(self):
        """A catch-all bigger than the tools it stands in for is how an unknown
        tool becomes the noisiest one."""
        assert PRIOR["*"][1] <= max(p90 for k, (_, p90) in PRIOR.items() if k != "*")

    def test_read_is_never_answered_from_the_prior_when_the_file_exists(self, tmp_path):
        f = tmp_path / "f.py"
        f.write_text("x" * 4_000)
        assert empty_model().predict_tool("Read", {"file_path": str(f)}).source == "stat"


class TestTheLearnedPopulation:
    """`by_tool` has to mean what `PRIOR` means, or the report compares a prior
    derived from unbounded calls against an average over all of them and the
    two disagree for reasons that have nothing to do with the machine."""

    @staticmethod
    def _pair(uid, tool, inp, size):
        return [
            {"type": "assistant", "message": {"id": f"m{uid}", "content": [
                {"type": "tool_use", "id": uid, "name": tool, "input": inp}]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": uid, "content": "x" * size * 4}]}},
        ]

    def test_a_bounded_command_is_not_in_the_tool_population(self, write_jsonl, tmp_path):
        recs = []
        for i in range(4):
            recs += self._pair(f"b{i}", "Bash", {"command": "wc -l f"}, 100)
        m = SizeModel.learn(write_jsonl(recs, into=tmp_path / "proj"))
        assert "Bash" not in m.tools

    def test_an_unbounded_command_is(self, write_jsonl, tmp_path):
        recs = []
        for i in range(4):
            recs += self._pair(f"u{i}", "Bash", {"command": "cat f"}, 100)
        m = SizeModel.learn(write_jsonl(recs, into=tmp_path / "proj"))
        assert m.tools["Bash"][2] == 4

    def test_a_bounded_read_is_not_in_it_either(self, write_jsonl, tmp_path):
        recs = []
        for i in range(4):
            recs += self._pair(f"r{i}", "Read", {"file_path": "/f", "limit": 20}, 100)
        m = SizeModel.learn(write_jsonl(recs, into=tmp_path / "proj"))
        assert "Read" not in m.tools

    def test_the_shape_model_still_learns_bounded_commands(self, write_jsonl, tmp_path):
        """They are excluded from the *tool* population, not from the shape
        model -- the guard needs to know `git log | head` is small."""
        recs = []
        for i in range(4):
            recs += self._pair(f"s{i}", "Bash", {"command": "git log | head -5"}, 10)
        m = SizeModel.learn(write_jsonl(recs, into=tmp_path / "proj"))
        assert m.shapes["git log|head+5"][2] == 4


class TestNumericBounds:
    """A bound that names a number is a size. Treating it as a free pass was
    the largest remaining error in this module: 45 supposedly-bounded calls in
    the corpus returned over 3,000 tokens, and the biggest was a `sed` range."""

    def test_a_sed_range_gives_its_line_count(self):
        from adder.core.shapes import bound_lines

        assert bound_lines("sed -n '1,600p' f") == 600
        assert bound_lines("sed -n '458,900p' f") == 443

    def test_head_and_tail_give_theirs(self):
        from adder.core.shapes import bound_lines

        assert bound_lines("cat f | head -50") == 50
        assert bound_lines("cat f | tail -n 20") == 20

    def test_a_bound_with_no_number_is_none(self):
        from adder.core.shapes import bound_lines

        assert bound_lines("wc -l a b") is None
        assert bound_lines("grep -c x f") is None

    def test_a_sequence_adds_its_bounds(self):
        from adder.core.shapes import bound_lines

        assert bound_lines("head -10 a; head -20 b") == 30

    def test_the_estimate_uses_the_measured_spread(self):
        from adder.core.shapes import TOKENS_PER_LINE, bound_estimate

        e = bound_estimate("sed -n '1,100p' f")
        assert e.p50 == int(100 * TOKENS_PER_LINE[0])
        assert e.p90 == int(100 * TOKENS_PER_LINE[1])
        assert e.p50 < e.p90, "the tail is the point; a line is not a fixed size"

    def test_a_bound_caps_a_learned_estimate(self):
        m = SizeModel(shapes={"cat": (200, 40_000, 40)},
                      heads={"cat": (200, 40_000, 40)}, built=1.0, calls=40)
        assert m.predict_command("cat huge | head -50").p90 == \
            int(50 * TOKENS_PER_LINE[1])

    def test_a_bound_larger_than_the_history_does_not_inflate_it(self):
        """Capping is one-directional: a generous bound is not evidence that
        this call will be large."""
        m = SizeModel(shapes={"cat": (200, 4_000, 40)},
                      heads={"cat": (200, 4_000, 40)}, built=1.0, calls=40)
        assert m.predict_command("cat huge | head -9000").p90 == 4_000

    def test_a_bound_is_evidence_not_a_prior(self):
        from adder.core.shapes import bound_estimate

        assert bound_estimate("head -100 f").measured
        assert "line bound" in bound_estimate("head -100 f").describe()


class TestNonTextFiles:
    """Sizing an image as `bytes / 4` is not an approximation, it is a category
    error. Left unfixed, the guard's replay ranked its eight largest findings
    as duplicate reads of PNG screenshots worth $25-$31 each, and put the
    guard's whole modelled value at $1,053 instead of $85."""

    def _file(self, tmp_path, name, size=1_000_000):
        f = tmp_path / name
        f.write_bytes(b"x" * size)
        return f

    def test_an_image_is_billed_by_dimensions_not_bytes(self, tmp_path):
        from adder.core.shapes import IMAGE_TOKENS

        e = read_estimate({"file_path": str(self._file(tmp_path, "shot.png"))})
        assert e.p90 == IMAGE_TOKENS[1] and e.source == "image"

    def test_a_huge_image_is_still_capped(self, tmp_path):
        from adder.core.shapes import IMAGE_TOKENS

        big = read_estimate({"file_path": str(self._file(tmp_path, "a.png", 50_000_000))})
        assert big.p90 == IMAGE_TOKENS[1], "the API downscales; bytes do not bill"

    @pytest.mark.parametrize("name", ["a.png", "a.jpg", "a.jpeg", "a.gif", "a.webp",
                                      "a.svg", "A.PNG"])
    def test_the_common_image_types_are_recognised(self, tmp_path, name):
        assert read_estimate(
            {"file_path": str(self._file(tmp_path, name))}).source == "image"

    def test_an_image_never_clears_the_guards_floor(self, tmp_path):
        """Re-reading a screenshot costs about 1,600 tokens, not thirty dollars."""
        from adder.decide.guard import Settings

        e = read_estimate({"file_path": str(self._file(tmp_path, "shot.png"))})
        assert e.p90 < Settings().min_tokens

    @pytest.mark.parametrize("name", ["a.zip", "a.pdf", "a.woff2", "a.sqlite", "a.mp4"])
    def test_an_opaque_file_gets_no_estimate_at_all(self, tmp_path, name):
        """No honest estimate is available, and a guess would be priced."""
        assert read_estimate({"file_path": str(self._file(tmp_path, name))}).p90 == 0

    def test_a_source_file_is_still_sized_from_disk(self, tmp_path):
        e = read_estimate({"file_path": str(self._file(tmp_path, "code.py", 40_000))})
        assert e.source == "stat" and e.p90 == 10_000

    def test_an_image_says_what_it_is(self, tmp_path):
        e = read_estimate({"file_path": str(self._file(tmp_path, "s.png"))})
        assert "billed by dimensions" in e.describe()


class TestOffsets:
    def test_an_offset_read_is_smaller_than_the_whole_file(self, tmp_path):
        f = tmp_path / "big.py"
        f.write_text("x" * 400_000)
        whole = read_estimate({"file_path": str(f)}).p90
        part = read_estimate({"file_path": str(f), "offset": 5_000}).p90
        assert 0 < part < whole

    def test_an_offset_past_the_end_is_not_negative(self, tmp_path):
        f = tmp_path / "small.py"
        f.write_text("x" * 100)
        assert read_estimate({"file_path": str(f), "offset": 999_999}).p90 == 0

    def test_a_limit_still_wins_over_an_offset(self, tmp_path):
        """`offset` says where to start; `limit` says how much, and how much is
        the thing being priced."""
        f = tmp_path / "big.py"
        f.write_text("x" * 400_000)
        e = read_estimate({"file_path": str(f), "offset": 50, "limit": 100})
        assert e.p90 < 3_000

    def test_a_junk_offset_is_survived(self, tmp_path):
        f = tmp_path / "f.py"
        f.write_text("x" * 4_000)
        assert read_estimate({"file_path": str(f), "offset": "banana"}).p90 > 0


class TestOtherBoundForms:
    def test_an_awk_line_range_is_a_bound(self):
        from adder.core.shapes import bound_lines

        assert is_bounded("awk 'NR<=50' f") and bound_lines("awk 'NR<=50' f") == 50

    def test_awk_without_a_range_is_not(self):
        assert not is_bounded("awk '{print}' f")

    def test_a_byte_bound_skips_the_per_line_assumption(self):
        """`head -c` states bytes, which is a tighter fact than a line count and
        needs no tokens-per-line term at all."""
        from adder.core.shapes import bound_estimate

        e = bound_estimate("head -c 4000 f")
        assert e is not None and e.p90 < 4_000

    def test_a_byte_bound_scales_with_the_bytes(self):
        from adder.core.shapes import bound_lines

        assert bound_lines("head -c 100000 f") > bound_lines("head -c 4000 f")


class TestOneQuantileEstimator:
    """The fourth private copy of the estimator `stats` exists to replace.

    `stats.py`'s docstring names three modules that each grew their own and
    calls the nearest-rank form biased. This one indexed at
    `round(q * (n - 1))`, and Python rounds halves to even -- so at n=6,
    `round(0.9 * 5)` is 4, and the p90 became the fifth of six samples instead
    of interpolating toward the sixth. On a heavy-tailed size distribution the
    sixth *is* the tail.

    Measured over 7,575 learned shapes: the p90 disagreed with the interpolated
    value by a mean of 56.7% on samples of 6-10, 122 of 271 shapes off by more
    than 10%, worst 68x. The guard reads exactly this number -- `est.p90`
    decides whether a call is priced at all -- and small samples are the common
    case, so the tail estimate was worst where the evidence is thinnest.
    """

    def test_it_agrees_with_the_canonical_estimator(self):
        import random

        from adder.util.stats import quantile as canon
        rng = random.Random(11)
        for _ in range(500):
            xs = sorted(rng.randint(0, 1000) for _ in range(rng.randint(1, 25)))
            for q in (0.0, 0.1, 0.5, 0.9, 0.95, 1.0):
                assert _quantile(xs, q) == round(canon(xs, q))

    def test_the_banker_rounding_case(self):
        """n=6, q=0.9: the index lands on exactly 4.5."""
        xs = [7, 9, 14, 20, 55, 966]
        assert _quantile(xs, 0.9) == 510      # was 14

    def test_the_tail_is_not_discarded(self):
        """A p90 below the second-largest sample is not a p90."""
        xs = [1, 1, 1, 1, 1, 1000]
        assert _quantile(xs, 0.9) >= xs[-2]

    def test_endpoints_are_exact(self):
        xs = [3, 5, 9, 40]
        assert _quantile(xs, 0.0) == 3
        assert _quantile(xs, 1.0) == 40

    def test_degenerate_inputs(self):
        assert _quantile([], 0.9) == 0
        assert _quantile([42], 0.9) == 42
