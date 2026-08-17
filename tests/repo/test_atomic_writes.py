"""No two writers may share a scratch path.

Every user-level file adder writes -- the guard state, the horizon cache, the
shape model, the outcome log, the ledger, the catalog -- is written from a hook
that runs in every Claude Code session, and several sessions share one machine.
Writing through a fixed `.tmp` name makes that scratch path shared mutable
state: writer A creates it, writer B truncates and rewrites it, A's `replace`
moves it away, and B's `replace` raises FileNotFoundError into an
`except OSError` that swallows it. Nothing errors; the write just does not
happen. Measured at 45% of writes lost under three concurrent writers.

`trace._cache_store` already carried the pid and said why. The other six sites
did not, and two of them even carried a comment about `replace` being atomic
for the *reader* -- which it is, and which is a different hazard.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

ADDER = Path(__file__).resolve().parents[2] / "adder"


def _sources():
    return [p for p in ADDER.rglob("*.py") if "__pycache__" not in str(p)]


class TestNoFixedTempNames:
    def test_no_module_writes_through_a_fixed_tmp_path(self):
        offenders = []
        for p in _sources():
            for i, line in enumerate(p.read_text().splitlines(), 1):
                if 'with_suffix(".tmp")' in line or "with_suffix('.tmp')" in line:
                    offenders.append(f"{p.relative_to(ADDER.parent)}:{i}")
        assert not offenders, (
            "these write through a scratch path another session can own at the "
            f"same time: {offenders}. Include os.getpid() in the name.")

    def test_every_temp_name_carries_the_pid(self):
        """The positive form: if a module makes a `.tmp`, it must be its own."""
        bad = []
        for p in _sources():
            for i, line in enumerate(p.read_text().splitlines(), 1):
                if ".tmp" in line and "with_suffix" in line and "getpid" not in line:
                    bad.append(f"{p.relative_to(ADDER.parent)}:{i}")
        assert not bad, f"temp names not unique per writer: {bad}"


class TestTheRaceItself:
    """A direct demonstration, so the rule above has a reason attached."""

    @staticmethod
    def _hammer(target: Path, name_for, writers=3, rounds=120):
        lost = []

        def run(tag):
            for _ in range(rounds):
                tmp = name_for(target)
                try:
                    tmp.write_text(json.dumps({"who": tag, "pad": [tag] * 500}))
                    tmp.replace(target)
                except OSError:
                    lost.append(tag)

        ts = [threading.Thread(target=run, args=(t,)) for t in "ABC"[:writers]]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return len(lost)

    def test_a_shared_scratch_path_loses_writes(self, tmp_path):
        lost = self._hammer(tmp_path / "s.json", lambda p: p.with_suffix(".tmp"))
        assert lost > 0, "expected the shared-path race to drop writes"

    def test_a_per_writer_scratch_path_loses_none(self, tmp_path):
        lost = self._hammer(
            tmp_path / "s.json",
            lambda p: p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp"))
        assert lost == 0

    def test_the_final_file_is_always_readable_either_way(self, tmp_path):
        """`replace` does protect the reader; that was never the bug."""
        t = tmp_path / "s.json"
        self._hammer(t, lambda p: p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp"))
        assert json.loads(t.read_text())["pad"]


class TestScratchFilesAreCleanedUp:
    def test_saving_a_catalog_leaves_no_tmp_behind(self, tmp_path):
        from adder.pricing.catalog import Catalog, Entry

        Catalog([Entry(key="m", id="m", inp=1.0, out=2.0)]).save(tmp_path / "c.json")
        assert [p.name for p in tmp_path.iterdir()] == ["c.json"]
