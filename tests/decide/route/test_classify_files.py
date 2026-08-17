"""How many files a task names, which is a tier signal and was silently always 1.

`_PATHLIKE` carried a capturing group, so `re.findall` returned the extension
rather than the path it matched. Every multi-file request counted as one file:
a three-file edit took the `scoped edit to one named file` branch and routed to
T1, and a read-only question spanning four files never reported spanning any.
The count was wrong in the cheap direction, which is the direction that costs a
retry.
"""
from __future__ import annotations

from adder.decide.route.classify import _PATHLIKE, Tier, classify


class TestFileCounting:
    def test_findall_returns_paths_not_extensions(self):
        t = "update adder/cost.py and adder/trace.py and adder/policy.py"
        assert _PATHLIKE.findall(t) == [
            "adder/cost.py", "adder/trace.py", "adder/policy.py"]

    def test_distinct_files_sharing_one_extension_still_count_separately(self):
        """The regression: three `.py` paths collapsed to a single `{'py'}`."""
        t = "a/one.py b/two.py c/three.py"
        assert len(set(_PATHLIKE.findall(t))) == 3

    def test_mixed_extensions_are_counted(self):
        t = "wire up src/app.ts, src/app.tsx and config.yaml"
        assert len(set(_PATHLIKE.findall(t))) == 3

    def test_the_same_path_twice_is_one_file(self):
        assert len(set(_PATHLIKE.findall("edit a/b.py then a/b.py again"))) == 1


class TestTierConsequences:
    def test_a_multi_file_edit_is_not_a_scoped_single_file_edit(self):
        """`mutating and files == 1` is the T1 branch; two files must not take it."""
        v = classify("update adder/cost.py and adder/trace.py")
        assert v.tier >= Tier.T2

    def test_a_genuinely_single_file_edit_still_routes_to_t1(self):
        v = classify("update adder/cost.py")
        assert v.tier is Tier.T1

    def test_a_read_only_question_spanning_files_reports_the_span(self):
        v = classify("where is x defined in a/one.py and b/two.py")
        assert any("spans 2 files" in r for r in v.reasons)
