"""Cascade economics, pinned on the terms that get left out.

Two of these tests exist because the published version of this analysis comes
from batch inference, where a failed attempt is discarded. In a session it is
not discarded, and the carry term it leaves behind is frequently larger than the
attempt's own cost.
"""

from __future__ import annotations

import json

import pytest

from adder.decide.route import cascade as cas
from adder.decide.route.cascade import Setup, carry_cost
from adder.pricing.cost import Rates

WEAK = "claude-haiku-4-5"
STRONG = "claude-opus-5"


def _setup(**kw):
    base = {"weak_model": WEAK, "strong_model": STRONG, "ctx_tokens": 50_000,
            "p_fail": 0.2, "remaining_turns": 100}
    base.update(kw)
    return Setup(**base)


class TestValidation:
    @pytest.mark.parametrize("field", ["p_fail", "false_negative", "false_positive"])
    def test_probabilities_are_range_checked(self, field):
        with pytest.raises(ValueError):
            _setup(**{field: 1.5})
        with pytest.raises(ValueError):
            _setup(**{field: -0.1})

    def test_a_negative_verify_share_is_rejected(self):
        with pytest.raises(ValueError):
            _setup(verify_share=-1.0)


class TestProbabilities:
    def test_a_perfect_check_escalates_exactly_the_failures(self):
        s = _setup(p_fail=0.3)
        assert s.p_escalate == pytest.approx(0.3)
        assert s.p_ships_broken == 0.0

    def test_a_blind_check_escalates_nothing_and_ships_everything(self):
        s = _setup(p_fail=0.3, false_negative=1.0)
        assert s.p_escalate == pytest.approx(0.0)
        assert s.p_ships_broken == pytest.approx(0.3)

    def test_over_firing_escalates_good_answers_too(self):
        s = _setup(p_fail=0.0, false_positive=0.25)
        assert s.p_escalate == pytest.approx(0.25)
        assert s.p_ships_broken == 0.0

    def test_a_half_blind_check_halves_the_failures_it_catches(self):
        s = _setup(p_fail=0.2, false_negative=0.5)
        assert s.p_escalate == pytest.approx(0.1)
        assert s.p_ships_broken == pytest.approx(0.1)


class TestCarry:
    def test_the_dead_end_is_priced_against_the_remaining_turns(self):
        short = cas.carry_cost(4_000, _setup(remaining_turns=1))
        long = cas.carry_cost(4_000, _setup(remaining_turns=200))
        assert long == pytest.approx(200 * short)

    def test_no_turns_left_means_no_carry(self):
        assert cas.carry_cost(4_000, _setup(remaining_turns=0)) == 0.0
        assert cas.carry_cost(0, _setup()) == 0.0

    def test_delegating_the_attempt_collapses_the_carry(self):
        """The reason the recommendation is usually 'cascade in a subagent'."""
        inline = cas.strategies(_setup(inline=True))
        deleg = cas.strategies(_setup(inline=False))
        inline_cost = next(s for s in inline if s.name == "cascade").cost
        deleg_cost = next(s for s in deleg if s.name == "cascade").cost
        assert deleg_cost < inline_cost

    def test_the_carry_term_can_dominate_the_attempt_itself(self):
        """On a long session the dead end costs more than generating it did."""
        from adder.pricing.cost import run_cost

        s = _setup(ctx_tokens=100_000, remaining_turns=300, p_fail=1.0)
        attempt = run_cost(WEAK, s.ctx_tokens, s.est_out_tokens)
        assert cas.carry_cost(s.ctx_tokens + s.est_out_tokens, s) > attempt


class TestStrategies:
    def test_every_strategy_is_priced(self):
        names = {s.name for s in cas.strategies(_setup())}
        assert {"always strong", "always weak", "cascade",
                "cascade (delegated)"} <= names

    def test_always_weak_ships_its_own_failure_rate(self):
        weak = next(s for s in cas.strategies(_setup(p_fail=0.2))
                    if s.name == "always weak")
        assert weak.p_broken == pytest.approx(0.2)

    def test_always_strong_is_the_reliable_baseline(self):
        strong = next(s for s in cas.strategies(_setup()) if s.name == "always strong")
        assert strong.p_broken == 0.0
        assert strong.usable == 1.0

    def test_a_context_the_weak_model_cannot_hold_kills_the_cascade(self):
        """Feasibility gates profitability; it is not a price question."""
        rows = cas.strategies(_setup(ctx_tokens=100_000_000))
        casc = next(s for s in rows if s.name == "cascade")
        assert casc.cost == float("inf")
        assert "cannot hold" in casc.detail

    def test_a_higher_failure_rate_makes_the_cascade_dearer(self):
        cheap = next(s for s in cas.strategies(_setup(p_fail=0.05)) if s.name == "cascade")
        dear = next(s for s in cas.strategies(_setup(p_fail=0.8)) if s.name == "cascade")
        assert dear.cost > cheap.cost

    def test_an_over_firing_check_costs_money_without_shipping_failures(self):
        clean = next(s for s in cas.strategies(_setup(p_fail=0.0)) if s.name == "cascade")
        noisy = next(s for s in cas.strategies(_setup(p_fail=0.0, false_positive=0.5))
                     if s.name == "cascade")
        assert noisy.cost > clean.cost
        assert noisy.p_broken == 0.0


class TestBest:
    def test_it_does_not_recommend_always_weak_on_dollars_alone(self):
        """The cheapest row is nearly always 'always weak'. It is rarely right."""
        assert cas.best(_setup(p_fail=0.3)).name != "always weak"

    def test_a_tier_that_never_fails_may_be_used_bare(self):
        assert cas.best(_setup(p_fail=0.0)).name in ("always weak", "cascade",
                                                     "cascade (delegated)")

    def test_a_blind_check_is_not_recommended(self):
        """It ships failures, so it cannot beat the reliable baseline."""
        assert cas.best(_setup(p_fail=0.3, false_negative=1.0)).name == "always strong"

    def test_an_infeasible_weak_model_falls_back_to_strong(self):
        assert cas.best(_setup(ctx_tokens=100_000_000)).name == "always strong"


class TestBreakeven:
    def test_the_breakeven_is_a_probability(self):
        assert 0.0 <= cas.breakeven_p_fail(_setup()) <= 1.0

    def test_below_the_breakeven_the_cascade_wins(self):
        from dataclasses import replace

        s = _setup(inline=False)
        be = cas.breakeven_p_fail(s)
        if 0.02 < be < 0.98:
            rows = cas.strategies(replace(s, p_fail=be - 0.02))
            strong = next(r for r in rows if r.name == "always strong").cost
            casc = min(r.cost for r in rows if r.name.startswith("cascade"))
            assert casc < strong

    def test_above_the_breakeven_the_cascade_loses(self):
        from dataclasses import replace

        s = _setup(inline=False)
        be = cas.breakeven_p_fail(s)
        if 0.02 < be < 0.98:
            rows = cas.strategies(replace(s, p_fail=be + 0.02))
            strong = next(r for r in rows if r.name == "always strong").cost
            casc = min(r.cost for r in rows if r.name.startswith("cascade"))
            assert casc > strong

    def test_a_carry_heavy_session_has_a_lower_breakeven(self):
        short = cas.breakeven_p_fail(_setup(remaining_turns=1, inline=True))
        long = cas.breakeven_p_fail(_setup(remaining_turns=400, inline=True))
        assert long <= short

    def test_an_infeasible_weak_model_never_breaks_even(self):
        assert cas.breakeven_p_fail(_setup(ctx_tokens=100_000_000)) == 0.0


class TestVerifierBudget:
    def test_a_flaky_tier_demands_a_sharper_check(self):
        assert cas.max_false_negative(_setup(p_fail=0.4), budget=0.05) == pytest.approx(0.125)

    def test_a_reliable_tier_tolerates_a_blunt_check(self):
        assert cas.max_false_negative(_setup(p_fail=0.02), budget=0.05) == 1.0

    def test_a_tier_that_never_fails_needs_no_check(self):
        assert cas.max_false_negative(_setup(p_fail=0.0)) == 1.0


class TestReport:
    def test_it_names_the_carry_penalty_of_running_inline(self):
        text = cas.report(_setup(inline=True, remaining_turns=300))
        assert "subagent" in text
        assert "re-read" in text

    def test_it_always_discloses_the_modelled_inputs(self):
        assert "MODELLED" in cas.report(_setup())

    def test_it_reports_the_boundary(self):
        assert "cascade wins below" in cas.report(_setup())

    def test_json_is_finite_and_complete(self):
        payload = cas.to_json(_setup())
        text = json.dumps(payload)
        assert "NaN" not in text and "Infinity" not in text
        assert payload["modelled"] is True
        assert set(payload) >= {"strategies", "best", "breakeven_p_fail"}

    def test_an_infeasible_row_serialises_as_null_not_infinity(self):
        payload = cas.to_json(_setup(ctx_tokens=100_000_000))
        casc = next(s for s in payload["strategies"] if s["name"] == "cascade")
        assert casc["cost_usd"] is None
        json.loads(json.dumps(payload))


class TestCli:
    def test_it_runs_and_prints(self, capsys, isolated_home):
        assert cas.main(["--p-fail", "0.2"]) == 0
        assert capsys.readouterr().out.strip()

    def test_json_parses(self, capsys, isolated_home):
        assert cas.main(["--p-fail", "0.2", "--json"]) == 0
        json.loads(capsys.readouterr().out)

    def test_a_bad_probability_is_a_usage_error_not_a_traceback(self, capsys,
                                                                isolated_home):
        assert cas.main(["--p-fail", "2.0"]) == 2

    def test_it_falls_back_to_the_measured_failure_rate(self, capsys, isolated_home):
        assert cas.main(["--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert 0.0 <= payload["setup"]["p_fail"] <= 1.0


class TestEdges:
    def test_a_free_check_is_allowed(self):
        rows = cas.strategies(_setup(verify_share=0.0))
        assert next(s for s in rows if s.name == "cascade").cost > 0

    def test_a_check_as_dear_as_the_attempt_is_priced_as_such(self):
        cheap = next(s for s in cas.strategies(_setup(verify_share=0.0))
                     if s.name == "cascade").cost
        dear = next(s for s in cas.strategies(_setup(verify_share=1.0))
                    if s.name == "cascade").cost
        assert dear > cheap

    def test_a_zero_turn_session_has_no_carry_and_a_higher_breakeven(self):
        assert (cas.breakeven_p_fail(_setup(remaining_turns=0, inline=True)) >=
                cas.breakeven_p_fail(_setup(remaining_turns=500, inline=True)))


class TestCarryIsPricedByTheProvider:
    """The carry term decides most cascade questions, so its rate has to be real.

    `carry_cost` multiplied the session model's *input* rate by Anthropic's
    0.10x cache discount. `session_model` is free text -- `fits` and `run_cost`
    both resolve anything in the catalog -- and on a provider with no prompt
    cache a re-read costs the full input rate. Assuming the discount understated
    the term tenfold on those models, in the direction that recommends them.
    """

    @staticmethod
    def _setup(session_model, **kw):
        fields = {"weak_model": "claude-haiku-4-5", "strong_model": "claude-opus-5",
                  "session_model": session_model, "remaining_turns": 400,
                  "ctx_tokens": 100_000}
        fields.update(kw)
        return Setup(**fields)

    def test_a_claude_session_is_unchanged(self):
        """Anthropic publishes 0.10x, so the old arithmetic was right here."""
        got = carry_cost(100_000, self._setup("claude-opus-5"))
        r = Rates.for_model("claude-opus-5")
        assert got == pytest.approx(100_000 * 400 * r.inp * 0.10 / 1_000_000)

    def test_it_uses_the_published_cache_read_rate(self):
        got = carry_cost(100_000, self._setup("claude-opus-5"))
        r = Rates.for_model("claude-opus-5")
        assert got == pytest.approx(100_000 * 400 * r.cache_read / 1_000_000)

    def test_a_provider_with_no_cache_pays_full_input_rate(self):
        r = Rates.for_model("cohere/command-a")
        assert r.cache_read == pytest.approx(r.inp)      # guards the premise
        got = carry_cost(100_000, self._setup("cohere/command-a"))
        assert got == pytest.approx(100_000 * 400 * r.inp / 1_000_000)

    def test_and_that_is_ten_times_the_old_figure(self):
        got = carry_cost(100_000, self._setup("cohere/command-a"))
        r = Rates.for_model("cohere/command-a")
        assert got == pytest.approx(10 * 100_000 * 400 * r.inp * 0.10 / 1_000_000)

    def test_no_turns_left_means_no_carry(self):
        assert carry_cost(100_000, self._setup("claude-opus-5", remaining_turns=0)) == 0.0
