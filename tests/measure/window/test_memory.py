"""The always-loaded prefix: what is resident, what it costs, what is stale.

The failure this file guards is a number that reads as reassuring. If a skill
body is counted as resident, a 40,000-token skill library prints as the most
expensive thing on the machine and the reader deletes the wrong file. If the
walk escapes into `node_modules`, the report is about somebody else's
instruction files.

Nothing here reads a real `~/.claude`: every test builds its own home.
"""

from __future__ import annotations

import json

import pytest

from adder.measure.window import memory


@pytest.fixture
def home(tmp_path):
    """A Claude home with a transcript root, wired the way `discover` expects."""
    h = tmp_path / "claude"
    (h / "projects").mkdir(parents=True)
    return h


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / ".claude").mkdir(parents=True)
    return r


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class TestFrontmatter:
    def test_scalar_keys(self):
        meta, body = memory.frontmatter("---\nname: x\ndescription: does y\n---\nbody\n")
        assert meta == {"name": "x", "description": "does y"}
        assert body.strip() == "body"

    def test_no_frontmatter_returns_whole_text(self):
        meta, body = memory.frontmatter("# just a heading\n")
        assert meta == {}
        assert body == "# just a heading\n"

    def test_continuation_lines_are_folded_into_the_key(self):
        meta, _ = memory.frontmatter("---\ndescription: one\n  two\n---\nb\n")
        assert meta["description"] == "one two"

    def test_quotes_are_stripped(self):
        meta, _ = memory.frontmatter('---\nname: "quoted"\n---\n')
        assert meta["name"] == "quoted"


class TestDiscover:
    def test_finds_instruction_files_at_both_scopes(self, home, repo):
        write(home / "CLAUDE.md", "user rules\n")
        write(repo / "CLAUDE.md", "project rules\n")
        docs = memory.discover(repo, home=home, root=home / "projects")
        scopes = {(d.kind, d.scope) for d in docs}
        assert ("claude-md", "user") in scopes
        assert ("claude-md", "project") in scopes
        assert all(d.load == "always" for d in docs if d.kind == "claude-md")

    def test_claude_md_is_resident_in_full(self, home, repo):
        write(repo / "CLAUDE.md", "x" * 4_000)
        doc = memory.discover(repo, home=home, root=home / "projects")[0]
        assert doc.resident == doc.tokens > 0

    def test_skill_body_is_not_resident(self, home, repo):
        write(repo / ".claude" / "skills" / "s" / "SKILL.md",
              "---\nname: s\ndescription: short\n---\n" + "b" * 40_000)
        doc = memory.discover(repo, home=home, root=home / "projects")[0]
        assert doc.kind == "skill"
        assert doc.tokens > 5_000
        assert doc.resident < 100
        assert doc.on_demand_tokens == doc.tokens - doc.resident

    def test_agent_description_is_resident(self, home, repo):
        write(repo / ".claude" / "agents" / "a.md",
              "---\nname: a\ndescription: " + "d" * 400 + "\n---\nbody\n")
        doc = memory.discover(repo, home=home, root=home / "projects")[0]
        assert doc.kind == "agent"
        assert doc.resident > 90

    def test_memory_files_are_on_demand_and_the_index_is_not(self, home, repo, monkeypatch):
        mem = home / "projects" / memory_slug(repo) / "memory"
        write(mem / "MEMORY.md", "- [A](a.md) — hook\n")
        write(mem / "a.md", "---\nname: a\n---\nfact\n")
        docs = memory.discover(repo, home=home, root=home / "projects")
        kinds = {d.kind: d for d in docs}
        assert kinds["memory-index"].load == "always"
        assert kinds["memory-index"].resident > 0
        assert kinds["memory"].load == "on-demand"
        assert kinds["memory"].resident == 0

    def test_nested_claude_md_is_on_demand(self, home, repo):
        write(repo / "CLAUDE.md", "top\n")
        write(repo / "sub" / "CLAUDE.md", "nested\n")
        docs = memory.discover(repo, home=home, root=home / "projects")
        loads = {d.path.parent.name: d.load for d in docs if d.kind == "claude-md"}
        assert loads["sub"] == "on-demand"

    def test_the_walk_does_not_descend_into_build_output(self, home, repo):
        write(repo / "node_modules" / "pkg" / "CLAUDE.md", "not mine\n")
        write(repo / ".git" / "CLAUDE.md", "not mine\n")
        docs = memory.discover(repo, home=home, root=home / "projects")
        assert not any("node_modules" in str(d.path) for d in docs)
        assert not any(".git" in str(d.path) for d in docs)

    def test_missing_directories_are_not_an_error(self, tmp_path):
        assert memory.discover(tmp_path / "nope", home=tmp_path / "h",
                               root=tmp_path / "h" / "projects") == []


def memory_slug(repo):
    from adder.measure.session.live import slug_for

    return slug_for(repo)


class TestPricing:
    def test_prior_when_there_are_no_sessions(self):
        p = memory.Pricing.measure({})
        assert not p.measured
        assert p.sessions == 0

    def test_measured_from_sessions(self, make_sessions):
        p = memory.Pricing.measure(make_sessions(4, 60))
        assert p.measured
        assert p.sessions == 4
        assert p.turns >= 1

    def test_cost_is_linear_in_tokens(self):
        p = memory.Pricing(turns=100, sessions=10)
        assert p.session_cost(2_000) == pytest.approx(2 * p.session_cost(1_000))

    def test_window_cost_is_per_session_times_sessions(self):
        p = memory.Pricing(turns=100, sessions=7)
        assert p.window_cost(1_000) == pytest.approx(7 * p.session_cost(1_000))

    def test_a_longer_session_carries_more(self):
        short = memory.Pricing(turns=10, sessions=1).session_cost(1_000)
        long = memory.Pricing(turns=500, sessions=1).session_cost(1_000)
        assert long > 10 * short

    def test_a_warm_opening_is_cheaper_than_a_cold_one(self):
        cold = memory.Pricing(turns=1, warm_share=0.0).session_cost(1_000)
        warm = memory.Pricing(turns=1, warm_share=0.9).session_cost(1_000)
        assert warm < cold

    def test_per_1k_is_the_editing_unit(self):
        p = memory.Pricing(turns=100, sessions=3)
        assert p.per_1k() == pytest.approx(p.session_cost(1_000))


class TestScope:
    """A project file is resident in that project's sessions and nowhere else."""

    def test_project_scope_counts_fewer_sessions(self):
        p = memory.Pricing(turns=100, sessions=100, project_sessions=10)
        assert p.window_cost(1_000, scope="project") < p.window_cost(1_000)

    def test_user_scope_is_the_default(self):
        p = memory.Pricing(turns=100, sessions=100, project_sessions=10)
        assert p.window_cost(1_000) == p.window_cost(1_000, scope="user")

    def test_a_project_doc_is_priced_at_project_scope(self, home, repo,
                                                      make_sessions):
        write(repo / "CLAUDE.md", "x" * 4_000)
        sessions = make_sessions(6, 30)          # none of them in this repo
        rep = memory.analyse(sessions, repo, home=home, root=home / "projects")
        assert rep.pricing.project_sessions == 0
        assert rep.cost(rep.ranked()[0]) == 0.0

    def test_sessions_in_the_project_are_counted(self, home, repo, make_session):
        write(repo / "CLAUDE.md", "x" * 4_000)
        slug = memory_slug(repo)
        sessions = {"a": make_session(30, sid="a", project=slug),
                    "b": make_session(30, sid="b", project="somewhere-else")}
        rep = memory.analyse(sessions, repo, home=home, root=home / "projects")
        assert rep.pricing.project_sessions == 1
        assert rep.pricing.sessions == 2
        assert rep.cost(rep.ranked()[0]) > 0

    def test_project_sessions_matches_the_slug_case_insensitively(self, home, repo,
                                                                  make_session):
        slug = memory_slug(repo).upper()
        sessions = {"a": make_session(10, sid="a", project=slug)}
        assert len(memory.project_sessions(sessions, repo)) == 1


class TestFindings:
    def test_duplicate_lines_across_resident_docs(self, home, repo):
        line = "This is a long shared instruction that appears in two files.\n"
        write(home / "CLAUDE.md", line)
        write(repo / "CLAUDE.md", line)
        found = memory.duplicates(memory.discover(repo, home=home,
                                                  root=home / "projects"))
        assert found and found[0].tokens > 0

    def test_short_lines_are_not_duplicates(self, home, repo):
        write(home / "CLAUDE.md", "## Rules\n")
        write(repo / "CLAUDE.md", "## Rules\n")
        assert memory.duplicates(memory.discover(repo, home=home,
                                                 root=home / "projects")) == []

    def test_unindexed_memory_is_reported(self, home, repo):
        mem = home / "projects" / memory_slug(repo) / "memory"
        write(mem / "MEMORY.md", "- [A](a.md) — hook\n")
        write(mem / "a.md", "a\n")
        write(mem / "b.md", "b\n")
        kinds = [f.kind for f in
                 memory.index_drift(memory.discover(repo, home=home,
                                                    root=home / "projects"))]
        assert "unindexed" in kinds

    def test_index_rows_pointing_at_nothing_are_reported(self, home, repo):
        mem = home / "projects" / memory_slug(repo) / "memory"
        write(mem / "MEMORY.md", "- [Gone](gone.md) — hook\n")
        kinds = [f.kind for f in
                 memory.index_drift(memory.discover(repo, home=home,
                                                    root=home / "projects"))]
        assert "dangling-row" in kinds

    def test_memory_with_no_index_is_reported_once(self, home, repo):
        mem = home / "projects" / memory_slug(repo) / "memory"
        write(mem / "a.md", "a\n")
        found = memory.index_drift(memory.discover(repo, home=home,
                                                   root=home / "projects"))
        assert [f.kind for f in found] == ["no-index"]

    def test_dangling_wiki_links(self, home, repo):
        mem = home / "projects" / memory_slug(repo) / "memory"
        write(mem / "MEMORY.md", "- [A](a.md) — hook\n")
        write(mem / "a.md", "---\nname: a\n---\nsee [[b]] and [[a]]\n")
        found = memory.stale_links(memory.discover(repo, home=home,
                                                   root=home / "projects"))
        assert [f.kind for f in found] == ["stale-link"]
        assert "[[b]]" in found[0].detail

    def test_stale_paths_only_flag_paths(self, home, repo):
        write(repo / "CLAUDE.md",
              "see `adder/gone.py` and `settings.json` and `adder/here.py`\n")
        write(repo / "adder" / "here.py", "x\n")
        found = memory.stale_paths(
            memory.discover(repo, home=home, root=home / "projects"), repo)
        assert len(found) == 1
        assert "adder/gone.py" in found[0].detail
        assert "settings.json" not in found[0].detail
        assert "here.py" not in found[0].detail

    def test_a_path_named_from_inside_the_package_is_not_stale(self, home, repo):
        # An instruction file names paths from wherever the writer was standing.
        # Resolving only against the repo root reported every real layout
        # reference as stale, which is how a checker gets ignored.
        write(repo / "CLAUDE.md", "the parser lives in `core/trace.py`\n")
        write(repo / "adder" / "core" / "trace.py", "x\n")
        assert memory.stale_paths(
            memory.discover(repo, home=home, root=home / "projects"), repo) == []

    def test_a_leading_dot_slash_is_the_same_claim(self, home, repo):
        write(repo / "CLAUDE.md", "see `./adder/here.py`\n")
        write(repo / "adder" / "here.py", "x\n")
        assert memory.stale_paths(
            memory.discover(repo, home=home, root=home / "projects"), repo) == []

    def test_duplicates_name_the_directory_when_the_leaf_is_a_convention(
            self, home, repo):
        line = "A long shared line that appears in two different skill files.\n"
        for name in ("alpha", "beta"):
            write(repo / ".claude" / "skills" / name / "SKILL.md",
                  "---\nname: " + name + "\ndescription: d\n---\n" + line)
        # Both files are called SKILL.md; "SKILL.md and SKILL.md" names nothing.
        found = memory.duplicates(memory.discover(repo, home=home,
                                                  root=home / "projects"))
        assert found
        assert "alpha/SKILL.md" in found[0].detail
        assert "beta/SKILL.md" in found[0].detail

    def test_long_descriptions_are_charged(self, home, repo):
        write(repo / ".claude" / "skills" / "s" / "SKILL.md",
              "---\nname: s\ndescription: " + "d" * 3_000 + "\n---\nbody\n")
        found = memory.oversize_descriptions(
            memory.discover(repo, home=home, root=home / "projects"))
        assert found and found[0].tokens > 100


class TestAccounting:
    def test_unaccounted_is_the_residual_of_the_measured_floor(self):
        docs = [memory.Doc(path=__import__("pathlib").Path("x"), kind="claude-md",
                           scope="project", load="always", bytes_=0, tokens=1_000,
                           resident=1_000)]
        assert memory.unaccounted(docs, 30_000) == 29_000

    def test_unaccounted_never_goes_negative(self):
        docs = [memory.Doc(path=__import__("pathlib").Path("x"), kind="claude-md",
                           scope="project", load="always", bytes_=0, tokens=50_000,
                           resident=50_000)]
        assert memory.unaccounted(docs, 30_000) == 0

    def test_by_kind_is_ordered_by_resident_cost(self, home, repo):
        write(repo / "CLAUDE.md", "x" * 8_000)
        write(repo / ".claude" / "agents" / "a.md",
              "---\nname: a\ndescription: d\n---\n")
        kinds = list(memory.by_kind(memory.discover(repo, home=home,
                                                    root=home / "projects")))
        assert kinds[0] == "claude-md"


class TestReport:
    def test_report_names_the_editing_unit(self, home, repo, make_sessions):
        write(repo / "CLAUDE.md", "x" * 4_000)
        rep = memory.analyse(make_sessions(3, 40), repo, home=home,
                             root=home / "projects")
        text = memory.report(rep)
        assert "Editing unit" in text
        assert "resident" in text

    def test_report_survives_an_empty_project(self, tmp_path):
        rep = memory.analyse({}, tmp_path / "empty", home=tmp_path / "h",
                             root=tmp_path / "h" / "projects")
        assert "Nothing found" in memory.report(rep)

    def test_json_is_one_document_with_the_pricing_terms(self, home, repo,
                                                         make_sessions, capsys):
        write(repo / "CLAUDE.md", "x" * 4_000)
        rc = memory.main([str(home / "projects"), "--repo", str(repo),
                          "--home", str(home), "--json"])
        assert rc == 0
        doc = json.loads(capsys.readouterr().out)
        assert doc["resident_tokens"] > 0
        assert doc["pricing"]["per_1k_per_session"] > 0
        assert doc["docs"][0]["path"].endswith("CLAUDE.md")

    def test_what_if_prices_an_edit(self, home, repo, capsys):
        write(repo / "CLAUDE.md", "x\n")
        memory.main([str(home / "projects"), "--repo", str(repo),
                     "--home", str(home), "--what-if", "1000", "--json"])
        doc = json.loads(capsys.readouterr().out)
        assert doc["what_if"]["tokens"] == 1_000
        assert doc["what_if"]["session_cost"] > 0

    def test_text_run_is_clean(self, home, repo, capsys):
        write(repo / "CLAUDE.md", "x" * 2_000)
        assert memory.main([str(home / "projects"), "--repo", str(repo),
                            "--home", str(home)]) == 0
        assert "resident" in capsys.readouterr().out


class TestStrict:
    def test_a_clean_project_exits_zero(self, home, repo, capsys):
        write(repo / "CLAUDE.md", "one clear instruction\n")
        assert memory.main([str(home / "projects"), "--repo", str(repo),
                            "--home", str(home), "--strict"]) == 0

    def test_a_stale_reference_fails_at_any_size(self, home, repo, capsys):
        write(repo / "CLAUDE.md", "see `adder/gone.py`\n")
        assert memory.main([str(home / "projects"), "--repo", str(repo),
                            "--home", str(home), "--strict"]) == 1

    def test_duplication_under_the_allowance_passes(self, home, repo, capsys):
        line = ("A shared instruction line, comfortably longer than the "
                "duplicate threshold, repeated in two files.\n")
        write(home / "CLAUDE.md", line)
        write(repo / "CLAUDE.md", line)
        assert memory.main([str(home / "projects"), "--repo", str(repo),
                            "--home", str(home), "--strict",
                            "--max-waste", "10000"]) == 0

    def test_duplication_over_the_allowance_fails(self, home, repo, capsys):
        line = ("A shared instruction line, comfortably longer than the "
                "duplicate threshold, repeated in two files.\n")
        write(home / "CLAUDE.md", line * 50)
        write(repo / "CLAUDE.md", line * 50)
        assert memory.main([str(home / "projects"), "--repo", str(repo),
                            "--home", str(home), "--strict",
                            "--max-waste", "10"]) == 1

    def test_strict_also_applies_to_json(self, home, repo, capsys):
        write(repo / "CLAUDE.md", "see `adder/gone.py`\n")
        assert memory.main([str(home / "projects"), "--repo", str(repo),
                            "--home", str(home), "--strict", "--json"]) == 1
        assert json.loads(capsys.readouterr().out)["wrong"] == 1


class TestReadOnly:
    def test_nothing_under_home_is_written(self, home, repo):
        write(repo / "CLAUDE.md", "x\n")
        before = {p: p.stat().st_mtime_ns for p in home.rglob("*")}
        memory.analyse({}, repo, home=home, root=home / "projects")
        after = {p: p.stat().st_mtime_ns for p in home.rglob("*")}
        assert before == after


def _size_of(finding):
    """The one-copy size the detail line leads with."""
    return int(finding.detail.split()[0].replace(",", ""))


class TestADuplicateReturnsEveryExtraCopy:
    """`Finding.tokens` is "the resident tokens it would return".

    A line living in three always-loaded files is paid for three times and two
    copies can be deleted. `duplicates` reported the size of one copy, which is
    right for a two-file group and understates every wider one -- and the widest
    group is the likeliest to be boilerplate nobody meant to keep. On this
    repository the three-way group went from +112 to +224 tokens.
    """

    @staticmethod
    def _docs(tmp_path, n_files):
        from adder.measure.window.memory import Doc

        line = ("this sentence is long enough to clear the duplicate floor and "
                "is repeated verbatim in every one of these files\n")
        docs = []
        for i in range(n_files):
            p = tmp_path / f"f{i}.md"
            p.write_text(line)
            docs.append(Doc(path=p, kind="memory", scope="project",
                            load="always", bytes_=len(line),
                            tokens=len(line) // 4, resident=len(line) // 4))
        return docs

    def test_two_files_return_one_copy(self, tmp_path):
        found = memory.duplicates(self._docs(tmp_path, 2))
        assert len(found) == 1
        size = _size_of(found[0])
        assert found[0].tokens == size

    def test_three_files_return_two_copies(self, tmp_path):
        found = memory.duplicates(self._docs(tmp_path, 3))
        assert found[0].tokens == 2 * _size_of(found[0])

    def test_four_files_return_three_copies(self, tmp_path):
        found = memory.duplicates(self._docs(tmp_path, 4))
        assert found[0].tokens == 3 * _size_of(found[0])

    def test_the_detail_still_names_one_copys_size(self, tmp_path):
        found = memory.duplicates(self._docs(tmp_path, 3))
        assert "paid for 3x" in found[0].detail

    def test_a_line_in_one_file_is_not_a_duplicate(self, tmp_path):
        assert memory.duplicates(self._docs(tmp_path, 1)) == []
