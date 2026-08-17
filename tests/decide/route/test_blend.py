"""Queue ordering, pinned on the one thing that can make it worthless.

Grouping only pays while a prefix survives between two members of its group. A
report that quotes an ordering saving without checking the TTL is quoting a
number that is zero in practice, which is the failure this suite guards.
"""

from __future__ import annotations

import json

import pytest

from adder.decide.route import blend as bl
from adder.decide.route.blend import Task

RATES = {"in_rate": 15.0, "cache_read_rate": 1.5,
         "cache_write_rate": 18.75, "out_rate": 75.0}


def _queue(groups=3, per_group=4, prefix=50_000):
    """Interleaved arrival order, so grouping has something to fix."""
    out = []
    for i in range(per_group):
        for g in range(groups):
            out.append(Task(key=f"g{g}-{i}", group=f"g{g}",
                            prefix_tokens=prefix, own_tokens=1_000))
    return out


class TestOrdering:
    def test_grouping_puts_a_group_together(self):
        ordered = bl.order_grouped(_queue())
        groups = [t.group for t in ordered]
        assert groups == sorted(groups, key=groups.index)
        assert len(set(groups)) == 3

    def test_arrival_order_is_untouched(self):
        q = _queue()
        assert [t.key for t in bl.order_arrival(q)] == [t.key for t in q]

    def test_the_largest_prefix_group_goes_first(self):
        q = [Task("a", "small", 1_000), Task("b", "big", 90_000)]
        assert bl.order_grouped(q)[0].group == "big"


class TestSimulate:
    def test_grouping_turns_writes_into_reads(self):
        q = _queue()
        arrival = bl.simulate(q, order="arrival", ttl_s=120.0, task_s=60.0, **RATES)
        grouped = bl.simulate(q, order="grouped", ttl_s=120.0, task_s=60.0, **RATES)
        assert grouped.prefix_reads > arrival.prefix_reads
        assert grouped.cost < arrival.cost

    def test_a_ttl_shorter_than_a_task_makes_ordering_useless(self):
        """The check that stops this reporting a saving nobody collects."""
        q = _queue()
        arrival = bl.simulate(q, order="arrival", ttl_s=1.0, task_s=60.0, **RATES)
        grouped = bl.simulate(q, order="grouped", ttl_s=1.0, task_s=60.0, **RATES)
        assert grouped.cost == pytest.approx(arrival.cost)

    def test_a_queue_with_no_shared_prefix_gains_nothing(self):
        q = [Task(f"t{i}", f"g{i}", 10_000) for i in range(8)]
        arrival = bl.simulate(q, order="arrival", **RATES)
        grouped = bl.simulate(q, order="grouped", **RATES)
        assert grouped.cost == pytest.approx(arrival.cost)

    def test_the_unique_part_is_charged_either_way(self):
        cheap = bl.simulate([Task("a", "g", 1_000, own_tokens=0)],
                            order="grouped", **RATES)
        dear = bl.simulate([Task("a", "g", 1_000, own_tokens=50_000)],
                           order="grouped", **RATES)
        assert dear.cost > cheap.cost

    def test_an_unknown_order_is_rejected(self):
        with pytest.raises(ValueError):
            bl.simulate([Task("a", "g", 1)], order="sideways", **RATES)

    def test_an_empty_queue_costs_nothing(self):
        assert bl.simulate([], order="grouped", **RATES).cost == 0.0

    def test_reuse_rate_is_a_fraction(self):
        run = bl.simulate(_queue(), order="grouped", **RATES)
        assert 0.0 <= run.reuse_rate <= 1.0


class TestReport:
    def test_it_reports_a_saving_when_grouping_helps(self):
        rep = bl.analyse(_queue(), ttl_s=120.0, task_s=60.0)
        assert rep.worth_ordering
        assert rep.saving > 0
        assert "Grouping saves" in bl.format_report(rep)

    def test_it_says_when_nothing_is_shared(self):
        q = [Task(f"t{i}", f"g{i}", 0, own_tokens=5_000) for i in range(6)]
        rep = bl.analyse(q)
        assert not rep.worth_ordering
        assert "shared prefix" in bl.format_report(rep)

    def test_the_saving_is_not_monotone_in_the_ttl(self):
        """A longer cache lifetime is not always worth more.

        Too short and even a grouped run goes cold between its own members; too
        long and the prefix survives the interleaving without any help. The
        saving peaks between the two and returns to zero at both ends.
        """
        rep = bl.analyse(_queue(), ttl_s=120.0, task_s=60.0)
        assert len(rep.sweep) == 4
        savings = [v for _, v in rep.sweep]
        assert all(v >= 0 for v in savings)
        assert max(savings) > 0
        assert savings[-1] < max(savings)      # the longest TTL is past the peak

    def test_it_defers_latency_to_the_deadline_command(self):
        assert "deadline" in bl.format_report(bl.analyse(_queue()))

    def test_an_empty_queue_says_so(self):
        assert "Empty queue" in bl.format_report(bl.analyse([]))

    def test_json_is_finite_and_complete(self):
        payload = bl.analyse(_queue(), ttl_s=120.0, task_s=60.0).to_json()
        text = json.dumps(payload)
        assert "NaN" not in text and "Infinity" not in text
        assert payload["worth_ordering"] is True
        assert payload["ttl_sweep"]


class TestLoading:
    def test_it_reads_a_queue(self, tmp_path):
        p = tmp_path / "q.jsonl"
        p.write_text('{"group":"a","prefix_tokens":100}\n# note\n\n'
                     '{"group":"b","prefix_tokens":200,"own_tokens":10}\n',
                     encoding="utf-8")
        assert len(bl.load(p)) == 2

    def test_a_malformed_line_names_itself(self, tmp_path):
        p = tmp_path / "q.jsonl"
        p.write_text('{"group":"a"}\nnope\n', encoding="utf-8")
        with pytest.raises(ValueError, match=r":2:"):
            bl.load(p)

    def test_a_missing_group_is_named(self, tmp_path):
        p = tmp_path / "q.jsonl"
        p.write_text('{"prefix_tokens":10}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="group"):
            bl.load(p)


class TestCli:
    def test_no_queue_exits_one_with_output(self, capsys, isolated_home):
        assert bl.main([]) == 1
        assert capsys.readouterr().out.strip()

    def test_json_parses_with_no_queue(self, capsys, isolated_home):
        bl.main(["--json"])
        json.loads(capsys.readouterr().out)

    def test_a_missing_file_is_an_error(self, tmp_path, capsys):
        assert bl.main([str(tmp_path / "nope.jsonl")]) == 1

    def test_it_prices_a_real_queue(self, tmp_path, capsys, isolated_home):
        p = tmp_path / "q.jsonl"
        rows = [{"group": f"g{i % 3}", "prefix_tokens": 50_000, "own_tokens": 1_000}
                for i in range(12)]
        p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        assert bl.main([str(p), "--ttl", "120", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["saving_usd"] > 0
