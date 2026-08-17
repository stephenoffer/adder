"""The manifest, pinned on the one property that makes it worth having.

If the digest is not stable across runs over identical inputs, every drift
report is noise and nobody reads the second one. Most of this file is that
property, approached from the angles that break it: wall clock, filesystem
ordering, and modification times that copying does not preserve.
"""

from __future__ import annotations

import json

import pytest

from adder.evaluate.claims import repro


@pytest.fixture
def tree(tmp_path):
    """A small transcript tree with two projects."""
    root = tmp_path / "projects"
    for project, names in (("proj-a", ["s1.jsonl", "s2.jsonl"]), ("proj-b", ["s3.jsonl"])):
        d = root / project
        d.mkdir(parents=True)
        for n in names:
            (d / n).write_text('{"type":"assistant"}\n')
    return root


class TestDigest:
    def test_the_same_inputs_give_the_same_digest(self, tree):
        assert (repro.fingerprint_transcripts(tree).digest ==
                repro.fingerprint_transcripts(tree).digest)

    def test_a_new_file_changes_it(self, tree):
        before = repro.fingerprint_transcripts(tree).digest
        (tree / "proj-a" / "s4.jsonl").write_text('{"type":"assistant"}\n')
        assert repro.fingerprint_transcripts(tree).digest != before

    def test_appending_to_a_file_changes_it(self, tree):
        before = repro.fingerprint_transcripts(tree).digest
        p = tree / "proj-a" / "s1.jsonl"
        p.write_text(p.read_text() + '{"type":"assistant"}\n')
        assert repro.fingerprint_transcripts(tree).digest != before

    def test_touching_a_file_does_not(self, tree):
        """mtimes do not survive copying, so drift on them is a false alarm."""
        import os

        before = repro.fingerprint_transcripts(tree).digest
        p = tree / "proj-a" / "s1.jsonl"
        os.utime(p, (0, 0))
        assert repro.fingerprint_transcripts(tree).digest == before

    def test_a_same_size_edit_is_caught_only_by_deep(self, tree):
        """Size collides more often than people expect."""
        p = tree / "proj-a" / "s1.jsonl"
        shallow_before = repro.fingerprint_transcripts(tree).digest
        deep_before = repro.fingerprint_transcripts(tree, deep=True).digest
        text = p.read_text()
        p.write_text(text.replace("assistant", "assistanX"))
        assert len(p.read_text()) == len(text)
        assert repro.fingerprint_transcripts(tree).digest == shallow_before
        assert repro.fingerprint_transcripts(tree, deep=True).digest != deep_before

    def test_file_and_byte_counts_are_recorded(self, tree):
        fp = repro.fingerprint_transcripts(tree)
        assert fp.files == 3
        assert fp.bytes > 0

    def test_an_empty_root_digests_cleanly(self, tmp_path):
        fp = repro.fingerprint_transcripts(tmp_path)
        assert fp.files == 0
        assert fp.digest

    def test_an_unreadable_file_hashes_to_empty_rather_than_raising(self, tmp_path):
        assert repro.hash_file(tmp_path / "missing.jsonl") == ""


class TestInputs:
    def test_prices_are_hashed_as_data(self):
        fp = repro.fingerprint_prices()
        assert fp.digest
        assert fp.detail.get("models", 0) > 0

    def test_the_price_digest_is_stable(self):
        assert repro.fingerprint_prices().digest == repro.fingerprint_prices().digest

    def test_the_catalog_records_its_own_staleness(self):
        fp = repro.fingerprint_catalog()
        assert fp.files > 0
        assert "oldest_days" in fp.detail

    def test_the_code_digest_covers_the_package(self):
        fp = repro.fingerprint_code()
        assert fp.files > 20
        assert fp.detail.get("version")


class TestManifest:
    def test_two_runs_over_the_same_inputs_agree(self, tree):
        assert repro.manifest(tree)["digest"] == repro.manifest(tree)["digest"]

    def test_the_environment_block_is_outside_the_digest(self, tree):
        """A digest containing a timestamp differs from itself a second later."""
        a = repro.manifest(tree, command="one")
        b = repro.manifest(tree, command="two")
        assert a["environment"]["command"] != b["environment"]["command"]
        assert a["digest"] == b["digest"]

    def test_every_input_is_fingerprinted(self, tree):
        inputs = repro.manifest(tree)["inputs"]
        assert set(inputs) == {"transcripts", "prices", "catalog", "code"}
        assert all(block["digest"] for block in inputs.values())

    def test_it_serialises(self, tree):
        text = json.dumps(repro.manifest(tree), sort_keys=True)
        assert "NaN" not in text
        json.loads(text)


class TestCompare:
    def test_identical_manifests_show_no_drift(self, tree):
        man = repro.manifest(tree)
        assert repro.compare(man, man) == []

    def test_added_transcripts_are_reported_with_the_delta(self, tree):
        before = repro.manifest(tree)
        (tree / "proj-b" / "s9.jsonl").write_text('{"type":"assistant"}\n')
        drifts = repro.compare(before, repro.manifest(tree))
        assert [d.name for d in drifts] == ["transcripts"]
        assert "+1 files" in drifts[0].note

    def test_removed_transcripts_show_a_negative_delta(self, tree):
        before = repro.manifest(tree)
        (tree / "proj-a" / "s1.jsonl").unlink()
        drifts = repro.compare(before, repro.manifest(tree))
        assert "-1 files" in drifts[0].note

    def test_code_drift_is_listed_first(self, tree):
        """It is the most complete explanation for a number that moved."""
        before = repro.manifest(tree)
        after = json.loads(json.dumps(before))
        after["inputs"]["code"]["digest"] = "different"
        after["inputs"]["transcripts"]["digest"] = "different"
        assert [d.name for d in repro.compare(before, after)] == ["code", "transcripts"]

    def test_a_shallow_comparison_says_it_was_shallow(self, tree):
        before = repro.manifest(tree)
        (tree / "proj-b" / "s9.jsonl").write_text("x\n")
        note = repro.compare(before, repro.manifest(tree))[0].note
        assert "--deep" in note

    def test_a_missing_input_block_does_not_raise(self, tree):
        assert repro.compare({}, repro.manifest(tree))


class TestReport:
    def test_it_renders_a_manifest(self, tree):
        text = repro.report(repro.manifest(tree))
        assert "reproducibility manifest" in text
        assert "transcripts" in text

    def test_it_says_when_nothing_moved(self, tree):
        man = repro.manifest(tree)
        assert "Identical" in repro.report(man, against=man)

    def test_it_explains_what_moved(self, tree):
        before = repro.manifest(tree)
        (tree / "proj-b" / "s9.jsonl").write_text("x\n")
        text = repro.report(repro.manifest(tree), against=before)
        assert "transcripts" in text
        # `render.wrap` breaks lines, so match a fragment that survives it.
        assert "explanation" in text


class TestCli:
    def test_it_runs_and_prints(self, tree, capsys, isolated_home):
        assert repro.main([str(tree)]) == 0
        assert "reproducibility manifest" in capsys.readouterr().out

    def test_json_parses(self, tree, capsys, isolated_home):
        assert repro.main([str(tree), "--json"]) == 0
        assert "digest" in json.loads(capsys.readouterr().out)

    def test_write_then_check_is_clean(self, tree, tmp_path, capsys, isolated_home):
        man = tmp_path / "m.json"
        assert repro.main([str(tree), "--write", str(man)]) == 0
        assert repro.main([str(tree), "--check", str(man)]) == 0

    def test_check_exits_one_when_an_input_moved(self, tree, tmp_path, capsys,
                                                 isolated_home):
        """A drift check that always exits 0 is one nobody wires into CI."""
        man = tmp_path / "m.json"
        repro.main([str(tree), "--write", str(man)])
        (tree / "proj-b" / "s9.jsonl").write_text("x\n")
        assert repro.main([str(tree), "--check", str(man)]) == 1

    def test_an_unreadable_manifest_is_a_usage_error(self, tree, tmp_path, capsys,
                                                     isolated_home):
        assert repro.main([str(tree), "--check", str(tmp_path / "nope.json")]) == 2

    def test_a_corrupt_manifest_is_a_usage_error(self, tree, tmp_path, capsys,
                                                 isolated_home):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        assert repro.main([str(tree), "--check", str(bad)]) == 2

    def test_drift_appears_in_the_json_surface(self, tree, tmp_path, capsys,
                                               isolated_home):
        man = tmp_path / "m.json"
        repro.main([str(tree), "--write", str(man)])
        (tree / "proj-b" / "s9.jsonl").write_text("x\n")
        capsys.readouterr()          # drop the text report from the write run
        repro.main([str(tree), "--check", str(man), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["drift"][0]["input"] == "transcripts"
