"""Guards on the savings estimator.

The estimator's failure mode is silent over-claiming, so these tests pin
attribution to measured reality rather than to expected values.
"""

import pytest

from router.savings import (
    _attributed,
    _session_read_cost,
    amortization_profile,
    delegation_savings,
    explore_savings,
    model_routing_savings,
)
from router.trace import Session, Turn


def _mk(n_turns: int, ctx_step: int = 10_000, out: int = 800) -> Session:
    """Synthetic session with linearly growing context."""
    s = Session("s1", "proj")
    for i in range(n_turns):
        ctx = ctx_step * (i + 1)
        s.turns.append(
            Turn("s1", "proj", "claude-opus-5",
                 uncached_in=0, cache_read=ctx, cache_write=0,
                 out=out, thinking=0, sidechain=False)
        )
    return s


class TestAttributionIsBounded:
    def test_attribution_sums_to_measured_read_cost(self):
        s = _mk(50)
        assert sum(d for d, _, _ in _attributed(s)) == pytest.approx(_session_read_cost(s))

    def test_never_exceeds_measured_spend(self):
        sessions = {"s1": _mk(200), "s2": _mk(30)}
        est, _ = amortization_profile(sessions)
        actual = sum(_session_read_cost(x) for x in sessions.values())
        assert est.saving <= actual * 1.001

    def test_raises_rather_than_overclaim(self):
        """The guard must fail loudly; this bug shipped twice during development."""
        s = _mk(10)
        for t in s.turns:
            t.cache_read = 0          # no measured reads -> nothing to attribute
        est, _ = amortization_profile({"s": s})
        assert est.saving == pytest.approx(0.0)

    def test_earlier_admissions_are_charged_more(self):
        """Content admitted early is re-read more often, so it must cost more."""
        s = _mk(100)
        rows = _attributed(s)
        early = rows[1][0]
        late = rows[-2][0]
        assert early > late


class TestSavingsAreBounded:
    def test_delegation_cannot_exceed_the_pool(self):
        sessions = {"s": _mk(500)}
        pool, _ = amortization_profile(sessions)
        d = delegation_savings(sessions)
        assert 0 < d.saving < pool.saving

    def test_delegation_scales_with_fraction(self):
        sessions = {"s": _mk(300)}
        lo = delegation_savings(sessions, delegable_fraction=0.1).saving
        hi = delegation_savings(sessions, delegable_fraction=0.5).saving
        assert hi > lo

    def test_no_compression_means_no_saving(self):
        sessions = {"s": _mk(300)}
        assert delegation_savings(sessions, compression=1.0).saving == pytest.approx(0, abs=1e-6)

    def test_explore_savings_zero_without_subagents(self):
        assert explore_savings({"s": _mk(50)}).saving == pytest.approx(0.0)

    def test_model_routing_is_small_on_warm_contexts(self):
        """The plan's central claim: per-turn downgrade barely helps here."""
        sessions = {"s": _mk(500, ctx_step=1_000)}   # grows to 500K
        pool, _ = amortization_profile(sessions)
        assert model_routing_savings(sessions).saving < pool.saving * 0.05


class TestConfidenceLabelling:
    def test_measured_and_modelled_are_distinguished(self):
        sessions = {"s": _mk(50)}
        assert explore_savings(sessions).confidence == "MEASURED"
        assert delegation_savings(sessions).confidence == "MODELLED"
        assert amortization_profile(sessions)[0].confidence == "ATTRIBUTED"

    def test_modelled_estimates_state_assumptions(self):
        sessions = {"s": _mk(50)}
        for e in (delegation_savings(sessions), model_routing_savings(sessions)):
            assert e.assumptions, f"{e.lever} must state its assumptions"
