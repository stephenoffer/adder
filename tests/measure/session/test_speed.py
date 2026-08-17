"""The fast-path audit, pinned on what it must refuse to claim.

Two traps. Pooling models turns a workload that runs one model fast and another
standard into a "speedup" that is entirely a difference between the models. And
timing from inter-turn gaps measures tool execution and human reading, so an
absolute throughput printed as a measurement would be wrong — only the paired
ratio survives.
"""

from __future__ import annotations

import json

import pytest

from adder.core.trace import Session, Turn
from adder.measure.session import speed as sp


def _session(sid, *, n=30, speed="standard", model="claude-opus-5",
             gap_s=10, out=500, project="p"):
    s = Session(sid, project)
    for i in range(n):
        secs = i * gap_s
        ts = f"2026-08-01T10:{secs // 60:02d}:{secs % 60:02d}Z"
        s.turns.append(Turn(sid, project, model, uncached_in=0,
                            cache_read=20_000, cache_write=0, out=out,
                            thinking=0, sidechain=False, ts=ts, speed=speed))
    return s


class TestSamples:
    def test_a_gap_becomes_a_throughput_sample(self):
        rows = sp.samples({"s": _session("s", n=5, gap_s=10, out=500)})
        assert len(rows) == 4                 # n-1 gaps
        assert rows[0].tokens_per_second == pytest.approx(50.0)

    def test_a_long_gap_is_dropped_as_a_human_not_a_model(self):
        rows = sp.samples({"s": _session("s", n=5, gap_s=600)})
        assert rows == []

    def test_the_gap_cap_is_configurable(self):
        rows = sp.samples({"s": _session("s", n=5, gap_s=600)}, max_gap_s=1_000)
        assert len(rows) == 4

    def test_turns_with_no_output_are_dropped(self):
        rows = sp.samples({"s": _session("s", n=5, out=0)})
        assert rows == []

    def test_turns_with_no_timestamp_are_dropped(self):
        s = Session("s", "p")
        for _ in range(4):
            s.turns.append(Turn("s", "p", "claude-opus-5", uncached_in=0,
                                cache_read=10, cache_write=0, out=100,
                                thinking=0, sidechain=False, ts=None))
        assert sp.samples({"s": s}) == []

    def test_an_empty_workload(self):
        assert sp.samples({}) == []


class TestCompare:
    @staticmethod
    def _mixed(fast_gap, std_gap, n=40, model="claude-opus-5"):
        return {
            "f": _session("f", n=n, speed="fast", gap_s=fast_gap, model=model),
            "s": _session("s", n=n, speed="standard", gap_s=std_gap, model=model),
        }

    def test_a_real_speedup_is_measured(self):
        rows = sp.samples(self._mixed(fast_gap=5, std_gap=20))
        c = sp.compare(rows)[0]
        assert c.paired
        assert c.ratio == pytest.approx(4.0, abs=0.1)
        assert c.faster
        assert c.clears_the_premium

    def test_a_speedup_below_the_premium_is_not_credited_as_one(self):
        """Measurably faster and still not worth 2x is the interesting case."""
        rows = sp.samples(self._mixed(fast_gap=15, std_gap=20))
        c = sp.compare(rows)[0]
        assert c.faster
        assert not c.clears_the_premium

    def test_no_speedup_at_all(self):
        rows = sp.samples(self._mixed(fast_gap=20, std_gap=20))
        c = sp.compare(rows)[0]
        assert not c.faster

    def test_models_are_compared_separately(self):
        """Pooling makes a model difference look like a speedup."""
        sessions = {
            "f": _session("f", n=40, speed="fast", gap_s=5, model="claude-opus-5"),
            "s": _session("s", n=40, speed="standard", gap_s=20,
                          model="claude-haiku-4-5"),
        }
        comparisons = sp.compare(sp.samples(sessions))
        assert len(comparisons) == 2
        # Neither model has both arms, so neither is a usable comparison.
        assert not any(c.paired for c in comparisons)

    def test_a_thin_arm_is_not_reported_as_a_result(self):
        sessions = {
            "f": _session("f", n=5, speed="fast", gap_s=5),
            "s": _session("s", n=40, speed="standard", gap_s=20),
        }
        c = sp.compare(sp.samples(sessions))[0]
        assert not c.paired
        assert not c.faster

    def test_the_interval_is_reproducible(self):
        rows = sp.samples(self._mixed(fast_gap=5, std_gap=20))
        assert sp.compare(rows)[0].ratio_ci == sp.compare(rows)[0].ratio_ci


class TestReport:
    def test_a_workload_that_never_used_it_says_so(self):
        rep = sp.analyse({"s": _session("s", n=30)})
        assert not rep.ever_used
        text = sp.format_report(rep)
        assert "never used it" in text
        assert "premium, always fast" in text

    def test_the_premium_is_priced_against_output_only(self):
        """Doubling the whole bill overstates it, and input is most of the bill."""
        rep = sp.analyse({"s": _session("s", n=30)})
        assert 0 < rep.median_session_output_cost < rep.median_session_cost
        assert rep.premium_per_session == pytest.approx(
            rep.median_session_output_cost)

    def test_a_measured_speedup_is_reported(self):
        rep = sp.analyse(TestCompare._mixed(fast_gap=5, std_gap=20))
        assert rep.ever_used
        assert "clears the" in sp.format_report(rep)

    def test_a_speedup_that_does_not_clear_the_premium_is_called_out(self):
        rep = sp.analyse(TestCompare._mixed(fast_gap=15, std_gap=20))
        text = sp.format_report(rep)
        assert "not a saving" in text or "less than double" in text

    def test_the_wall_clock_caveat_is_always_printed(self):
        rep = sp.analyse(TestCompare._mixed(fast_gap=5, std_gap=20))
        assert "WALL CLOCK, NOT GENERATION" in sp.format_report(rep)

    def test_a_workload_with_no_usable_gaps(self):
        rep = sp.analyse({})
        assert rep.total_turns == 0
        assert "nothing can be timed" in sp.format_report(rep)

    def test_json_is_finite_and_complete(self):
        payload = sp.analyse(TestCompare._mixed(fast_gap=5, std_gap=20)).to_json()
        text = json.dumps(payload)
        assert "NaN" not in text and "Infinity" not in text
        assert payload["wall_clock_only"] is True
        assert payload["multiplier"] == pytest.approx(sp.FAST_MULTIPLIER)

    def test_the_multiplier_comes_from_the_price_table(self):
        assert sp._fast_multiplier() == pytest.approx(sp.FAST_MULTIPLIER)


class TestCli:
    def test_it_runs_against_a_fixture(self, write_jsonl, capsys, isolated_home):
        recs = [{
            "type": "assistant", "sessionId": "s",
            "timestamp": f"2026-08-01T10:{i:02d}:00Z",
            "message": {"id": f"m{i}", "model": "claude-opus-5",
                        "usage": {"input_tokens": 2,
                                  "cache_read_input_tokens": 20_000,
                                  "cache_creation_input_tokens": 100,
                                  "output_tokens": 400}}} for i in range(8)]
        root = write_jsonl(recs, into=None)
        assert sp.main([str(root)]) == 0
        assert capsys.readouterr().out.strip()

    def test_json_parses(self, write_jsonl, capsys, isolated_home):
        recs = [{
            "type": "assistant", "sessionId": "s",
            "timestamp": f"2026-08-01T10:{i:02d}:00Z",
            "message": {"id": f"m{i}", "model": "claude-opus-5",
                        "usage": {"input_tokens": 2,
                                  "cache_read_input_tokens": 20_000,
                                  "cache_creation_input_tokens": 100,
                                  "output_tokens": 400}}} for i in range(4)]
        root = write_jsonl(recs, into=None)
        assert sp.main([str(root), "--json"]) == 0
        json.loads(capsys.readouterr().out)

    def test_an_empty_root_exits_one(self, tmp_path, capsys, isolated_home):
        assert sp.main([str(tmp_path)]) == 1
        assert capsys.readouterr().out.strip()
