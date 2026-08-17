"""A corpus that bills nothing still has to produce a report.

Every share line in `adder trace` divided by the total, and a workload can
legitimately total zero: interrupted streams, zero-token replies, or a filter
window that keeps only turns whose usage block is all zeros. The turns are real
-- they are counted, the model is known, the records are well-formed -- so none
of the "no priced turns" early returns fire, and the first percentage line
raised ZeroDivisionError halfway through printing.

`stats.share` exists for exactly this ("`part/whole`, or 0.0 when the whole is
zero. The division every report does"); the report simply was not using it.
"""
from __future__ import annotations

import json

import pytest

from adder.measure.spend import trace as trace_cmd


@pytest.fixture
def zero_cost_root(tmp_path):
    """Well-formed assistant records whose usage is entirely zero."""
    d = tmp_path / "proj"
    d.mkdir()
    (d / "zero.jsonl").write_text("\n".join(json.dumps({
        "type": "assistant", "sessionId": "zero",
        "timestamp": f"2026-08-10T12:{i:02d}:00Z",
        "message": {"id": f"msg_{i}", "model": "claude-opus-5",
                    "usage": {"input_tokens": 0, "cache_read_input_tokens": 0,
                              "cache_creation_input_tokens": 0, "output_tokens": 0},
                    "content": [{"type": "text", "text": ""}]},
    }) for i in range(12)))
    return tmp_path


class TestTheReportSurvivesAZeroTotal:
    def test_the_text_report_does_not_raise(self, zero_cost_root, capsys):
        assert trace_cmd.main([str(zero_cost_root)]) in (0, 1)
        assert "12 turns" in capsys.readouterr().out

    def test_every_share_reads_as_zero_rather_than_crashing(self, zero_cost_root, capsys):
        trace_cmd.main([str(zero_cost_root)])
        out = capsys.readouterr().out
        assert "input-side" in out and "cache-read" in out

    @pytest.mark.parametrize("flag", ["--json", "--verify", "--strict"])
    def test_the_other_surfaces_survive_it_too(self, zero_cost_root, capsys, flag):
        assert trace_cmd.main([str(zero_cost_root), flag]) in (0, 1)

    @pytest.mark.parametrize("by", ["model", "day", "tool", "session"])
    def test_every_grouping_survives_it(self, zero_cost_root, capsys, by):
        assert trace_cmd.main([str(zero_cost_root), "--by", by]) in (0, 1)

    def test_the_json_surface_is_still_valid_json(self, zero_cost_root, capsys):
        trace_cmd.main([str(zero_cost_root), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["turns"] == 12
        assert payload["total"] == 0


class TestAnUnpriceableCorpusSaysWhy:
    """"No priced turns" with no reason is the worst version of the failure
    this module's docstring exists to prevent.

    A model missing from the price table "makes every total here quietly too
    small, and a quietly-too-small total is the failure mode this project
    exists to avoid". A corpus of nothing but unpriceable turns is that failure
    at its most extreme -- a total of zero -- and the tally that explains it
    was reported in every case except that one.
    """

    def _corpus(self, write_jsonl, model="totally-made-up-9"):
        return write_jsonl([{
            "type": "assistant", "sessionId": "s",
            "timestamp": f"2026-08-01T10:0{i}:00Z",
            "message": {"id": f"m{i}", "model": model,
                        "usage": {"input_tokens": 5, "output_tokens": 50}},
        } for i in range(3)])

    def test_the_text_report_names_the_model(self, write_jsonl, capsys,
                                             isolated_home):
        from adder.measure.spend import trace as mod

        assert mod.main([str(self._corpus(write_jsonl))]) == 1
        out = capsys.readouterr().out
        assert "no price" in out.lower() and "totally-made-up-9" in out

    def test_the_json_carries_the_tally(self, write_jsonl, capsys, isolated_home):
        import json

        from adder.measure.spend import trace as mod

        assert mod.main([str(self._corpus(write_jsonl)), "--json"]) == 1
        d = json.loads(capsys.readouterr().out)
        assert d["unknown_turns"] == 3
        assert d["unknown_models"] == {"totally-made-up-9": 3}

    def test_strict_fails_differently(self, write_jsonl, capsys, isolated_home):
        """`--strict` promises a non-zero exit for an unpriced model; an empty
        report exits 1 for its own reason, so the two must be tellable apart."""
        from adder.measure.spend import trace as mod

        assert mod.main([str(self._corpus(write_jsonl)), "--strict"]) == 2

    def test_an_empty_directory_still_just_says_empty(self, tmp_path, capsys,
                                                      isolated_home):
        from adder.measure.spend import trace as mod

        assert mod.main([str(tmp_path)]) == 1
        assert "used a model with no price" not in capsys.readouterr().out
