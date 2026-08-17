"""Remaining-turns estimation. The countdown model is wrong in the expensive
direction, so these tests pin the corrected behaviour."""
from __future__ import annotations

import pytest

from adder.measure.session.horizon import DEFAULT_REMAINING, MIN_SAMPLES, Horizon


class TestEmptyHorizon:
    def test_falls_back_to_flat_prior(self):
        h = Horizon.default()
        assert h.remaining(0) == DEFAULT_REMAINING
        assert h.remaining(10_000) == DEFAULT_REMAINING

    def test_prior_is_flat_not_a_countdown(self):
        """A countdown would decay to zero; the measured process does not."""
        h = Horizon.default()
        assert h.remaining(0) == h.remaining(5_000)


class TestSurvivorEstimator:
    @pytest.fixture
    def h(self):
        # Heavy-tailed, like the measured distribution.
        return Horizon(sorted([50] * 10 + [200] * 10 + [600] * 10 +
                              [1200] * 10 + [3000] * 10))

    def test_conditions_on_surviving_to_turn_n(self, h):
        """Past a threshold only longer sessions remain, so the estimate RISES.

        Survivors of turn 700 are the 1200s and 3000s, whose median remaining is
        (500 + 2300)/2 = 1400 -- far more than a countdown would ever report.
        """
        assert h.remaining(700) == 1400

    def test_does_not_collapse_to_zero_late(self, h):
        """The countdown's fatal flaw: it reports 0 while turns remain."""
        assert h.countdown(1500) == 0
        assert h.remaining(1500) > 0

    def test_countdown_underestimates_late_in_session(self, h):
        for n in (600, 1000):
            assert h.remaining(n) > h.countdown(n)

    def test_thin_tail_falls_back_to_prior(self, h):
        """With fewer than MIN_SAMPLES survivors, don't fake precision."""
        assert sum(1 for L in h.lengths if L > 3_000) < MIN_SAMPLES
        assert h.remaining(3_000) == DEFAULT_REMAINING

    def test_uses_data_while_samples_remain(self, h):
        """10 survivors is plenty; the prior must not kick in early."""
        assert sum(1 for L in h.lengths if L > 2_999) >= MIN_SAMPLES
        assert h.remaining(2_999) == 1

    def test_monotone_sample_shrinkage(self, h):
        counts = [sum(1 for L in h.lengths if n < L) for n in (0, 100, 500, 1000)]
        assert counts == sorted(counts, reverse=True)

    def test_error_table_shape(self, h):
        rows = h.error_table((10, 100))
        assert len(rows) == 2 and all(len(r) == 3 for r in rows)


class TestFromSessions:
    def test_ignores_trivially_short_sessions(self, make_session):
        h = Horizon.from_sessions(
            {"a": make_session(2, sid="a"), "b": make_session(100, sid="b"),
             "c": make_session(300, sid="c")}, min_turns=5)
        assert h.lengths == [100, 300]

    def test_empty_input_is_safe(self):
        assert Horizon.from_sessions({}).remaining(50) == DEFAULT_REMAINING

    def test_a_session_is_as_long_as_its_conversation(self, make_session,
                                                      make_turn):
        """Subagent turns are not turns of the conversation being forecast.

        The number this feeds is multiplied by the carry rate to count how many
        times a token admitted now is re-read, and a subagent turn does not
        re-read the main context -- that is what delegating bought. One of the
        eighty sessions here reported 716 records for a 207-turn conversation.
        """
        s = make_session(40, sid="s")
        s.turns += [make_turn(session="s", sidechain=True) for _ in range(60)]
        assert Horizon.from_sessions({"s": s}).lengths == [40]

    def test_a_session_with_no_delegation_is_unchanged(self, make_session):
        s = make_session(40, sid="s")
        assert Horizon.from_sessions({"s": s}).lengths == [40]

    def test_a_session_short_on_the_main_chain_is_excluded(self, make_session,
                                                           make_turn):
        """310 records, but only three of them are the conversation."""
        s = make_session(3, sid="s")
        s.turns += [make_turn(session="s", sidechain=True) for _ in range(307)]
        assert Horizon.from_sessions({"s": s}, min_turns=5).lengths == []


class TestTheFitIsCached:
    """Who calls this is the whole argument for caching it.

    `live.analyse` calls `horizon.load`, and both hooks call `analyse` -- so
    every prompt submission and every guarded read was re-fitting a
    distribution over every session on the machine. Measured: 2,089ms to fit,
    4ms to read back, to move an estimate built from ~100 sessions by at most
    one session.
    """

    def test_a_second_load_does_not_refit(self, isolated_home, monkeypatch,
                                          make_sessions):
        from adder.measure.session import horizon

        monkeypatch.setattr("adder.core.trace.load_sessions",
                            lambda *a, **k: make_sessions(3, 40))
        first = horizon.load()
        assert first.lengths

        def explode(*a, **k):                  # pragma: no cover - must not run
            raise AssertionError("refitted a horizon that was already cached")

        monkeypatch.setattr("adder.core.trace.load_sessions", explode)
        assert horizon.load().lengths == first.lengths

    def test_a_stale_fit_is_refitted(self, isolated_home, monkeypatch, make_sessions):
        from adder.measure.session import horizon

        monkeypatch.setattr("adder.core.trace.load_sessions",
                            lambda *a, **k: make_sessions(3, 40))
        horizon.load()
        monkeypatch.setattr("adder.core.trace.load_sessions",
                            lambda *a, **k: make_sessions(5, 90))
        assert horizon.load(max_age_s=-1).lengths == [90] * 5

    def test_the_cache_can_be_bypassed(self, isolated_home, monkeypatch, make_sessions):
        from adder.measure.session import horizon

        monkeypatch.setattr("adder.core.trace.load_sessions",
                            lambda *a, **k: make_sessions(3, 40))
        horizon.load()
        monkeypatch.setattr("adder.core.trace.load_sessions",
                            lambda *a, **k: make_sessions(2, 77))
        assert horizon.load(use_cache=False).lengths == [77] * 2

    def test_a_corrupt_cache_is_refitted_not_raised(self, isolated_home, monkeypatch,
                                                    make_sessions):
        from adder.measure.session import horizon

        horizon.cache_path().parent.mkdir(parents=True, exist_ok=True)
        horizon.cache_path().write_text("{not json")
        monkeypatch.setattr("adder.core.trace.load_sessions",
                            lambda *a, **k: make_sessions(3, 40))
        assert horizon.load().lengths == [40] * 3

    def test_a_future_version_is_not_read_as_this_one(self, isolated_home, monkeypatch,
                                                      make_sessions):
        import json

        from adder.measure.session import horizon

        horizon.cache_path().parent.mkdir(parents=True, exist_ok=True)
        horizon.cache_path().write_text(json.dumps({"v": 999, "built": 9e9,
                                              "lengths": [1, 2, 3]}))
        monkeypatch.setattr("adder.core.trace.load_sessions",
                            lambda *a, **k: make_sessions(3, 40))
        assert horizon.load().lengths == [40] * 3

    def test_it_writes_only_where_it_was_told_to(self, isolated_home):
        from adder.measure.session import horizon

        assert str(isolated_home) in str(horizon.cache_path()), \
            "must not default into the real home during a test"

    def test_an_empty_fit_is_not_cached(self, isolated_home, monkeypatch):
        """Caching "no sessions" for an hour would hide the first real one."""
        from adder.measure.session import horizon

        monkeypatch.setattr("adder.core.trace.load_sessions", lambda *a, **k: {})
        horizon.load()
        assert not horizon.cache_path().exists()


class TestTheFitIsKeyedToItsRoot:
    """One cache file held one fit, whatever directory it was built from.

    `adder carry <dir>` passes its root explicitly, and so do several evaluate
    commands. With no root in the key they read back a distribution fitted to
    whichever directory had been analysed first -- and every carry, placement
    and delegation number downstream was then computed from somebody else's
    session lengths.
    """

    def _corpus(self, root, n_sessions, n_turns):
        import json

        d = root / "-Users-x-p"
        d.mkdir(parents=True, exist_ok=True)
        for s in range(n_sessions):
            rows = [{
                "type": "assistant", "sessionId": f"s{s}",
                "timestamp": f"2026-08-01T10:{i:02d}:00Z",
                "message": {"id": f"m{s}-{i}", "model": "claude-opus-5",
                            "usage": {"input_tokens": 1,
                                      "cache_read_input_tokens": 1000,
                                      "output_tokens": 10}},
            } for i in range(n_turns)]
            (d / f"s{s}.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        return root

    def test_two_roots_do_not_share_a_fit(self, tmp_path, monkeypatch):
        from adder.measure.session.horizon import load

        monkeypatch.setenv("ADDER_HOME", str(tmp_path / "home"))
        a = self._corpus(tmp_path / "a", 6, 30)
        b = self._corpus(tmp_path / "b", 6, 12)
        ha = load(a, use_cache=True)
        hb = load(b, use_cache=True)
        assert ha.lengths != hb.lengths
        assert set(ha.lengths) == {30} and set(hb.lengths) == {12}

    def test_the_same_root_still_hits_the_cache(self, tmp_path, monkeypatch):
        from adder.measure.session import horizon as mod

        monkeypatch.setenv("ADDER_HOME", str(tmp_path / "home"))
        a = self._corpus(tmp_path / "a", 6, 30)
        first = mod.load(a, use_cache=True)
        calls = {"n": 0}
        real = mod.Horizon.from_sessions

        def counting(*args, **kw):
            calls["n"] += 1
            return real(*args, **kw)

        monkeypatch.setattr(mod.Horizon, "from_sessions", counting)
        again = mod.load(a, use_cache=True)
        assert again.lengths == first.lengths
        assert calls["n"] == 0          # served from the cache, not re-fitted
