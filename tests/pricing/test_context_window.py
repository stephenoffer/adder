"""A context window nobody published is None, and None is not a number.

53 of the 510 bundled catalog entries carry no context length. Eight call sites
did arithmetic or `:,` formatting on `context_limit()` directly, and every one
of them is reachable by pointing the `ladder` setting at such a model -- which
`adder config` documents as a thing to do.
"""

from __future__ import annotations

import pytest

from adder.pricing.registry import context_limit, context_window, limit_str

UNBOUNDED = "grok-3-mini"          # bundled, priced, no published window


def test_the_fixture_model_really_has_no_window():
    assert context_limit(UNBOUNDED) is None


class TestContextWindow:
    def test_unknown_window_is_the_caller_s_default(self):
        assert context_window(UNBOUNDED) == 0
        assert context_window(UNBOUNDED, 9_000) == 9_000

    def test_a_known_window_is_returned_as_is(self):
        assert context_window("claude-haiku-4-5") == 200_000

    def test_the_result_is_always_usable_as_a_number(self):
        assert min(500, context_window(UNBOUNDED, 500)) == 500


class TestLimitStr:
    def test_a_known_window_reads_as_a_token_count(self):
        assert limit_str("claude-haiku-4-5") == "200,000-token"

    def test_an_unknown_window_never_renders_as_none(self):
        assert "None" not in limit_str(UNBOUNDED)
        assert "undeclared" in limit_str(UNBOUNDED)

    def test_an_unresolvable_model_is_a_sentence_not_an_exception(self):
        assert "undeclared" in limit_str("no-such-model-anywhere")


class TestTheCallSitesThatUsedToRaise:
    def test_context_pressure_on_a_model_with_no_window(self):
        from adder.measure.session.live import LiveReport

        r = LiveReport(turns=1, context=50_000, spent=0.0, per_turn=0.0,
                       projected_remaining=0, projected_total=0.0,
                       model=UNBOUNDED)
        assert r.context_pressure == 0.0

    def test_right_size_prices_a_ladder_rung_with_no_window(self):
        from adder.decide.route.classify import Verdict, classify
        from adder.decide.route.policy import right_size

        v: Verdict = classify("read a file and summarise it")
        _tier, ladder, _ = right_size(v, need_tokens=9_000, est_out_tokens=800,
                                     retry_overhead=0.0)
        assert ladder and all(isinstance(r.expected, float) for r in ladder)

    @pytest.mark.parametrize("target", [0, -1])
    def test_plan_solve_refuses_a_non_positive_target(self, target):
        from adder.evaluate.replay.plan import solve

        with pytest.raises(ValueError, match="positive multiple"):
            solve({}, target=target, baseline=1.0, output_share=0.1)
