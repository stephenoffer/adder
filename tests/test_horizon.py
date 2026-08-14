"""Remaining-turns estimation. The countdown model is wrong in the expensive
direction, so these tests pin the corrected behaviour."""

import pytest

from router.horizon import DEFAULT_REMAINING, MIN_SAMPLES, Horizon


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
    def test_ignores_trivially_short_sessions(self):
        class S:
            def __init__(self, n):
                self.turns = list(range(n))
        h = Horizon.from_sessions({"a": S(2), "b": S(100), "c": S(300)}, min_turns=5)
        assert h.lengths == [100, 300]

    def test_empty_input_is_safe(self):
        assert Horizon.from_sessions({}).remaining(50) == DEFAULT_REMAINING
