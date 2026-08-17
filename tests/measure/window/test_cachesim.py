"""The cache simulator, pinned on the rules that make a simulator honest.

A cache simulator's failure mode is optimism, and it has three classic forms:
matching blocks out of prefix order, forgetting that entries expire, and
letting capacity be unbounded while claiming it was not. Each has a test here.
"""

from __future__ import annotations

import json

import pytest

from adder.measure.window import cachesim as cs
from adder.measure.window.cachesim import Request


def _req(session="s", project="p", tokens=1000, shared=0, when=0.0, model="claude-opus-5"):
    return Request(session=session, project=project, model=model, tokens=tokens,
                   shared_prefix=shared, when=when,
                   cost_read=0.15 / 1e6, cost_write=1.875 / 1e6)


class TestSpans:
    def test_a_request_is_two_contiguous_ranges(self):
        (ns_a, a), (ns_b, b) = cs._spans(_req(tokens=1600, shared=800), 16)
        assert a == 50 and b == 50
        assert ns_a.startswith("p:") and ns_b.startswith("s:")

    def test_the_boundary_block_cannot_be_shared(self):
        """A block straddling the shared line contains bytes from both sides."""
        (_, shared), (_, own) = cs._spans(_req(tokens=100, shared=50), 16)
        assert shared == 3          # floor(50/16), not 4
        assert shared + own == 6    # floor(100/16): the partial tail is not stored

    def test_a_shared_prefix_longer_than_the_request_is_clamped(self):
        (_, shared), (_, own) = cs._spans(_req(tokens=100, shared=100_000), 16)
        assert own == 0
        assert shared == 6

    def test_only_whole_blocks_are_stored(self):
        """A partial block has no stable key, so it is never cached.

        Rounding it up credits the next request with tokens that were never
        stored, and the error grows with the block size -- it showed up as a
        1024-token block reporting a better hit rate than a 16-token one.
        """
        for tokens in (1, 15, 16, 17, 1000):
            (_, a), (_, b) = cs._spans(_req(tokens=tokens), 16)
            assert (a + b) * 16 <= tokens
            assert tokens - (a + b) * 16 < 16

    def test_a_bad_block_size_is_rejected(self):
        with pytest.raises(ValueError):
            cs._spans(_req(), 0)

    def test_different_models_never_share_a_namespace(self):
        """Caches are model-scoped; sharing across them is the classic bug."""
        (a, _), _ = cs._spans(_req(model="claude-opus-5", shared=1000, tokens=2000), 16)
        (b, _), _ = cs._spans(_req(model="claude-haiku-4-5", shared=1000, tokens=2000), 16)
        assert a != b


class TestSimulate:
    def test_a_cold_run_is_all_miss(self):
        res = cs.simulate([_req(tokens=1000)])
        assert res.hit_tokens == 0
        assert res.miss_tokens == 1000
        assert res.hit_rate == 0.0

    def test_an_immediate_repeat_is_all_hit(self):
        res = cs.simulate([_req(when=0.0), _req(when=1.0)])
        # First cold, second warm apart from the trailing partial block, which
        # is never stored: 992 of 1000 tokens hit on the second request.
        assert res.hit_rate == pytest.approx(992 / 2000, abs=1e-6)

    def test_two_sessions_share_the_project_prefix(self):
        """The whole point: session two does not pay for the shared opening."""
        reqs = [_req(session="a", tokens=2000, shared=1600, when=0.0),
                _req(session="b", tokens=2000, shared=1600, when=1.0)]
        shared = cs.simulate(reqs)
        isolated = cs.simulate([_req(session="a", tokens=2000, shared=0, when=0.0),
                                _req(session="b", tokens=2000, shared=0, when=1.0)])
        assert shared.hit_tokens > isolated.hit_tokens
        assert shared.cost < isolated.cost

    def test_matching_is_prefix_anchored(self):
        """A resident private tail is unusable when the shared head is gone."""
        reqs = [
            _req(session="a", tokens=2000, shared=1000, when=0.0),
            # Far enough later that everything expired, then a request whose
            # private range was resident before but whose shared head was not.
            _req(session="a", tokens=2000, shared=1000, when=10_000.0),
        ]
        res = cs.simulate(reqs, ttl_s=300.0)
        assert res.hit_tokens == 0

    def test_expiry_turns_a_warm_prefix_cold(self):
        warm = cs.simulate([_req(when=0.0), _req(when=100.0)], ttl_s=300.0)
        cold = cs.simulate([_req(when=0.0), _req(when=100_000.0)], ttl_s=300.0)
        assert warm.hit_tokens > 0
        assert cold.hit_tokens == 0
        assert cold.expiries > 0

    def test_a_zero_ttl_disables_expiry(self):
        res = cs.simulate([_req(when=0.0), _req(when=10**9)], ttl_s=0.0)
        assert res.hit_tokens > 0
        assert res.expiries == 0

    def test_capacity_pressure_evicts_and_lowers_the_hit_rate(self):
        """Interleaved sessions, so an evicted prefix is asked for again.

        Sequential write-then-read is LRU-friendly and shows no capacity effect
        no matter how small the cache -- which is correct, and is why the
        pattern here round-robins instead.
        """
        reqs = []
        when = 0.0
        for _round in range(4):
            for i in range(20):
                reqs.append(_req(session=f"s{i}", tokens=100_000, when=when))
                when += 1.0
        roomy = cs.simulate(reqs, capacity_tokens=10_000_000, ttl_s=0.0)
        tight = cs.simulate(reqs, capacity_tokens=200_000, ttl_s=0.0)
        assert tight.hit_rate < roomy.hit_rate
        assert tight.evictions > 0
        assert roomy.evictions == 0

    def test_peak_residency_never_exceeds_capacity(self):
        reqs = [_req(session=f"s{i}", tokens=50_000, when=float(i)) for i in range(30)]
        res = cs.simulate(reqs, capacity_tokens=200_000, block_size=16)
        assert res.peak_resident <= 200_000 + 16

    def test_cold_and_capacity_misses_are_counted_apart(self):
        reqs = []
        when = 0.0
        for _round in range(3):
            for i in range(10):
                reqs.append(_req(session=f"s{i}", tokens=100_000, when=when))
                when += 1.0
        res = cs.simulate(reqs, capacity_tokens=200_000, ttl_s=0.0)
        assert res.cold_misses > 0
        assert res.capacity_misses > 0

    def test_a_bad_capacity_is_rejected(self):
        with pytest.raises(ValueError):
            cs.simulate([_req()], capacity_tokens=0)

    def test_no_requests_produces_an_empty_result(self):
        res = cs.simulate([])
        assert res.requests == 0 and res.hit_rate == 0.0

    def test_cost_is_between_all_hit_and_all_miss(self):
        reqs = [_req(when=float(i)) for i in range(5)]
        res = cs.simulate(reqs)
        total = res.total_tokens
        assert total * 0.15 / 1e6 <= res.cost <= total * 1.875 / 1e6

    def test_replay_order_is_by_timestamp_not_list_order(self):
        forward = cs.simulate([_req(when=0.0), _req(when=1.0)])
        backward = cs.simulate([_req(when=1.0), _req(when=0.0)])
        assert forward.hit_tokens == backward.hit_tokens

    def test_it_finishes_on_a_realistic_workload(self):
        """500K contexts at a 16-token block is 31k blocks; per-block is hopeless."""
        reqs = [_req(session=f"s{i % 20}", tokens=500_000, shared=30_000,
                     when=float(i)) for i in range(2_000)]
        res = cs.simulate(reqs, block_size=16)
        assert res.requests == 2_000


class TestBlockSize:
    def test_bigger_blocks_waste_more_at_the_boundary(self):
        reqs = [_req(session="a", tokens=1000, shared=500, when=0.0),
                _req(session="b", tokens=1000, shared=500, when=1.0)]
        small = cs.simulate(reqs, block_size=16)
        large = cs.simulate(reqs, block_size=1024)
        assert small.hit_tokens >= large.hit_tokens


class TestFromSessions:
    def test_requests_carry_the_project_shared_line(self, make_sessions):
        sessions = make_sessions(n=2, n_turns=5, base=20_000, growth=1_000)
        reqs = cs.requests_from(sessions)
        assert reqs
        assert all(r.shared_prefix > 0 for r in reqs)
        assert all(r.shared_prefix <= r.tokens for r in reqs)

    def test_turns_with_no_context_are_dropped(self, make_session):
        s = make_session(3, base=0, growth=0)
        for t in s.turns:
            t.cache_read = 0
            t.cache_write = 0
            t.uncached_in = 0
        assert cs.requests_from({"s": s}) == []

    def test_the_measured_baseline_is_a_rate_and_a_bill(self, make_sessions):
        hit_rate, cost = cs.measured_baseline(make_sessions(n=2, n_turns=5))
        assert 0.0 <= hit_rate <= 1.0
        assert cost > 0.0

    def test_the_baseline_of_nothing_is_zero(self):
        assert cs.measured_baseline({}) == (0.0, 0.0)


class TestSweep:
    def test_the_hit_rate_is_monotone_in_capacity(self, make_sessions):
        sw = cs.sweep(make_sessions(n=3, n_turns=20))
        rates = [r.hit_rate for r in sw.by_capacity]
        assert rates == sorted(rates)

    def test_the_knee_is_the_cheapest_capacity_that_still_performs(self, make_sessions):
        sw = cs.sweep(make_sessions(n=3, n_turns=20))
        knee = sw.knee
        assert knee is not None
        best = max(r.hit_rate for r in sw.by_capacity)
        assert knee.hit_rate >= best - 0.01
        assert knee.capacity_tokens <= max(r.capacity_tokens for r in sw.by_capacity)

    def test_an_empty_workload_sweeps_to_nothing(self):
        sw = cs.sweep({})
        assert sw.by_capacity == []
        assert sw.knee is None

    def test_json_is_finite_and_flags_itself_as_simulated(self, make_sessions):
        payload = cs.sweep(make_sessions(n=2, n_turns=10)).to_json()
        text = json.dumps(payload)
        assert "NaN" not in text and "Infinity" not in text
        assert payload["simulated"] is True


class TestReport:
    def test_it_labels_every_number_as_simulated(self, make_sessions):
        text = cs.report(make_sessions(n=2, n_turns=10))
        assert "SIMULATED" in text
        assert "measured hit rate" in text

    def test_an_empty_workload_says_so(self):
        assert "No turns" in cs.report({})

    def test_the_cli_runs_against_a_fixture(self, write_jsonl, capsys, isolated_home):
        recs = []
        for i in range(6):
            recs.append({
                "type": "assistant", "sessionId": "s",
                "timestamp": f"2026-08-01T10:{i:02d}:00Z",
                "message": {"id": f"m{i}", "model": "claude-opus-5",
                            "usage": {"input_tokens": 2,
                                      "cache_read_input_tokens": 20_000 + 500 * i,
                                      "cache_creation_input_tokens": 100,
                                      "output_tokens": 300}}})
        root = write_jsonl(recs, into=None)
        assert cs.main([str(root)]) == 0
        assert "SIMULATED" in capsys.readouterr().out

    def test_the_cli_json_parses(self, write_jsonl, capsys, isolated_home):
        recs = [{
            "type": "assistant", "sessionId": "s",
            "timestamp": "2026-08-01T10:00:00Z",
            "message": {"id": "m0", "model": "claude-opus-5",
                        "usage": {"input_tokens": 2,
                                  "cache_read_input_tokens": 20_000,
                                  "cache_creation_input_tokens": 100,
                                  "output_tokens": 300}}}]
        root = write_jsonl(recs, into=None)
        assert cs.main([str(root), "--json"]) == 0
        json.loads(capsys.readouterr().out)

    def test_an_empty_root_exits_one_with_output(self, tmp_path, capsys, isolated_home):
        assert cs.main([str(tmp_path)]) == 1
        assert capsys.readouterr().out.strip()


class TestBoundaries:
    def test_a_request_larger_than_the_whole_cache(self):
        """Nothing can be retained, and the simulator must still terminate."""
        res = cs.simulate([_req(tokens=1_000_000, when=0.0),
                           _req(tokens=1_000_000, when=1.0)],
                          capacity_tokens=1_000, block_size=16)
        assert res.requests == 2
        assert res.hit_rate < 1.0

    def test_an_entry_exactly_at_the_ttl_is_still_live(self):
        """Strictly greater than, not greater-or-equal: an off-by-one here
        silently halves the hit rate on a workload that ticks at the TTL."""
        res = cs.simulate([_req(when=0.0), _req(when=300.0)], ttl_s=300.0)
        assert res.hit_tokens > 0
        assert res.expiries == 0

    def test_one_second_past_the_ttl_is_gone(self):
        res = cs.simulate([_req(when=0.0), _req(when=301.0)], ttl_s=300.0)
        assert res.hit_tokens == 0

    def test_a_request_smaller_than_one_block_is_all_miss(self):
        """It stores nothing, and its tokens must still be counted and billed."""
        res = cs.simulate([_req(tokens=1), _req(tokens=1, when=1.0)])
        assert res.miss_tokens == 2
        assert res.hit_tokens == 0
        assert res.cost > 0

    def test_the_longer_ttl_is_priced_at_the_higher_write_rate(self, make_sessions):
        """Simulating a 1h TTL at the 5m write rate would make it look free."""
        sessions = make_sessions(n=1, n_turns=4)
        short = cs.requests_from(sessions, ttl_s=300.0)
        long = cs.requests_from(sessions, ttl_s=3600.0)
        assert long[0].cost_write > short[0].cost_write
        assert long[0].cost_read == short[0].cost_read


class TestSubagentsGetTheirOwnNamespace:
    """A sidechain turn carries the PARENT's session id, and is not its prefix.

    Without separating them, a subagent's short brief and the parent's large
    context share one cache namespace: the brief's blocks score as hits against
    blocks the parent wrote, and the simulator over-reports its hit rate. This
    module's own docstring calls counting an entry that is not there "the bug
    that makes a simulator optimistic".
    """

    def _sessions(self, make_turn):
        from adder.core.trace import Session

        s = Session("s", "proj")
        s.turns = [make_turn(session="s", read=200_000, minutes=i) for i in range(4)]
        # Two separate delegated runs, both on the parent's session id.
        for i, agent in enumerate(("ag1", "ag2")):
            t = make_turn(session="s", read=20_000, sidechain=True, minutes=10 + i)
            t.agent_id = agent
            s.turns.append(t)
        return {"s": s}

    def test_each_run_has_a_distinct_private_namespace(self, make_turn):
        from adder.measure.window.cachesim import _spans, requests_from

        reqs = requests_from(self._sessions(make_turn))
        private = {_spans(r, 16)[1][0] for r in reqs}
        assert len(private) == 3        # the main chain, plus two subagent runs

    def test_a_subagent_claims_none_of_the_project_shared_line(self, make_turn):
        from adder.measure.window.cachesim import requests_from

        reqs = requests_from(self._sessions(make_turn))
        assert all(r.shared_prefix == 0 for r in reqs if r.agent)
        assert any(r.shared_prefix > 0 for r in reqs if not r.agent)
