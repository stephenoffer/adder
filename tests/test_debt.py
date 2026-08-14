"""Context debt: the novel core. Over-claiming here would discredit everything."""

import pytest

from router.debt import (
    breakeven_remaining_turns,
    debt_multiple,
    decompose_read_cost,
    token_lifetime_cost,
    verbosity_saving,
)
from router.trace import Session, Turn

OPUS = "claude-opus-5"


def _sess(n_turns: int, out: int = 500, ctx_step: int = 5_000, base: int = 25_000) -> Session:
    s = Session("s", "p")
    for i in range(n_turns):
        ctx = base + ctx_step * i
        s.turns.append(Turn("s", "p", OPUS, uncached_in=0, cache_read=ctx,
                            cache_write=0, out=out, thinking=0, sidechain=False))
    return s


class TestDebtMultiple:
    def test_no_remaining_turns_is_sticker_price(self):
        assert debt_multiple(0, OPUS) == pytest.approx(1.0)

    def test_breakeven_is_fifty_turns_on_opus(self):
        """Opus 5: $25/MTok out vs $0.50/MTok/turn re-read -> parity at 50 turns."""
        assert breakeven_remaining_turns(OPUS) == 50
        assert debt_multiple(50, OPUS) == pytest.approx(2.0)

    @pytest.mark.parametrize("turns,mult", [(200, 5.0), (607, 13.14), (1000, 21.0), (3478, 70.56)])
    def test_measured_session_multiples(self, turns, mult):
        assert debt_multiple(turns, OPUS) == pytest.approx(mult, rel=1e-2)

    def test_grows_linearly(self):
        a, b = debt_multiple(100, OPUS), debt_multiple(200, OPUS)
        assert (b - 1) == pytest.approx(2 * (a - 1))

    def test_cheaper_model_has_same_ratio_shape(self):
        """Haiku: $5 out, $0.10/turn re-read -> also 50 turns to parity."""
        assert breakeven_remaining_turns("claude-haiku-4-5") == 50


class TestLifetimeCost:
    def test_matches_manual_arithmetic(self):
        # 1M tokens, 100 remaining turns: $25 generation + 100 * $0.50 re-read
        assert token_lifetime_cost(1_000_000, 100, OPUS) == pytest.approx(75.0)

    def test_negative_turns_clamped(self):
        assert token_lifetime_cost(1000, -5, OPUS) == token_lifetime_cost(1000, 0, OPUS)


class TestDecompositionIsBounded:
    """The bug that shipped twice: attributed cost exceeding measured spend."""

    def test_parts_sum_to_measured_total(self):
        sessions = {"a": _sess(300), "b": _sess(50)}
        total, base, acc = decompose_read_cost(sessions)
        assert base + acc == pytest.approx(total)

    def test_never_exceeds_measured_read_cost(self):
        sessions = {"a": _sess(1000)}
        total, base, acc = decompose_read_cost(sessions)
        measured = sum(
            t.cache_read * 5 * 0.1 / 1e6 for s in sessions.values() for t in s.turns
        )
        assert total == pytest.approx(measured) and base + acc <= measured * 1.001

    def test_baseline_dominates_a_flat_session(self):
        """No growth -> everything is irreducible baseline, nothing to save."""
        sessions = {"a": _sess(100, ctx_step=0)}
        _, base, acc = decompose_read_cost(sessions)
        assert acc == pytest.approx(0.0, abs=1e-9) and base > 0

    def test_growth_creates_addressable_pool(self):
        sessions = {"a": _sess(300, ctx_step=5_000)}
        _, base, acc = decompose_read_cost(sessions)
        assert acc > base

    def test_empty_sessions_are_safe(self):
        assert decompose_read_cost({}) == (0.0, 0.0, 0.0)


class TestVerbositySaving:
    def test_bounded_by_accumulated_pool(self):
        sessions = {"a": _sess(500)}
        _, _, acc = decompose_read_cost(sessions)
        v = verbosity_saving(sessions, reduction=1.0)
        assert v.reread_saved <= acc * 1.001

    def test_scales_with_reduction(self):
        sessions = {"a": _sess(300)}
        assert verbosity_saving(sessions, reduction=0.5).total > \
               verbosity_saving(sessions, reduction=0.1).total

    def test_zero_reduction_saves_nothing(self):
        assert verbosity_saving({"a": _sess(300)}, reduction=0.0).total == pytest.approx(0.0)

    def test_leverage_exceeds_one_in_long_sessions(self):
        """The whole point: downstream saving dwarfs generation saving."""
        v = verbosity_saving({"a": _sess(600)}, reduction=0.3)
        assert v.leverage > 1.0

    def test_leverage_is_near_zero_in_short_sessions(self):
        v = verbosity_saving({"a": _sess(3)}, reduction=0.3)
        assert v.leverage < 1.0
