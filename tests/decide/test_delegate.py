"""The routing step that runs at the moment of delegation.

`policy.decide` has always been able to answer "what should this run on". What
it lacked was a caller at the point the question is live, and this module is that
caller: a `Task` goes past the guard, and the guard names the tier.

Which puts it in the one place in this repository where being wrong is not just
an unhelpful report -- it is a sentence in somebody's context, carried for the
rest of the session, arguing for a cheaper model on a task that may need the
expensive one. So the tests here are mostly about silence: when it must say
nothing, and that it can never refuse.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from adder.decide.delegate import Advice, advise, stem, task_text
from adder.decide.route.classify import Tier


@dataclass(frozen=True)
class _Rung:
    """Enough of `policy.Rung` to price a comparison."""

    tier: Tier
    model: str
    expected: float
    p_fail: float = 0.1


@dataclass
class _Plan:
    """Enough of `policy.Plan` to stand in for one, so no outcome log is read."""

    tier: Tier
    agent: str
    ladder: list
    reasons: list


def _plan(chosen=Tier.T1, *, t2_cost=0.42, t1_cost=0.21, t0_cost=0.15):
    return _Plan(
        tier=chosen,
        agent=chosen.agent,
        ladder=[_Rung(Tier.T0, "claude-haiku-4-5", t0_cost),
                _Rung(Tier.T1, "claude-sonnet-5", t1_cost),
                _Rung(Tier.T2, "claude-opus-5", t2_cost)],
        reasons=["matches a scoped-edit signal"],
    )


def _advise(tool_input=None, *, plan=None, model="claude-opus-5", remaining=300,
            context=200_000, **kw):
    return advise(tool_input if tool_input is not None
                  else {"description": "rename a helper in one file"},
                  session_model=model, remaining_turns=remaining,
                  context_tokens=context, plan=plan or _plan(), **kw)


class TestWhenItSpeaks:
    def test_a_cheaper_tier_is_named_with_the_saving(self):
        got = _advise()
        assert got.fire
        assert got.agent == "route-t1" and "claude-sonnet-5" in got.message
        assert got.saving == pytest.approx(0.42 - 0.21)

    def test_the_message_says_what_it_would_otherwise_have_run_on(self):
        """A recommendation without a baseline is unfalsifiable."""
        assert "claude-opus-5" in _advise().message

    def test_the_chance_of_a_redo_is_in_the_sentence(self):
        """A cheaper tier is only cheaper net of the risk of doing it twice, and
        a reader who cannot see that number cannot check the claim."""
        assert "10%" in _advise().message

    def test_it_prices_its_own_sentence(self):
        got = _advise()
        assert got.overhead > 0 and got.net > 0


class TestWhenItStaysQuiet:
    def test_a_call_that_already_names_a_routed_agent_is_left_alone(self):
        for name in ("route-t0", "route-t1", "route-t2", "Explore", "explore"):
            got = _advise({"description": "look at it", "subagent_type": name})
            assert not got.fire, name
            assert "already" in got.reason

    def test_an_unrouted_custom_agent_is_still_advised(self):
        """Only the tiers carry a decision. A project's own agent does not."""
        assert _advise({"description": "audit the schema",
                        "subagent_type": "schema-reviewer"}).fire

    def test_no_task_text_means_no_guess(self):
        got = _advise({})
        assert not got.fire and "guess" in got.reason

    def test_the_tier_it_would_already_use_produces_silence(self):
        got = _advise(plan=_plan(Tier.T2), model="claude-opus-5")
        assert not got.fire and "cheapest tier" in got.reason

    def test_a_session_model_off_the_ladder_is_not_compared_against(self):
        """The difference between two rungs is a number. The difference between
        a rung and an unpriced unknown is not."""
        got = _advise(model="gpt-5-codex")
        assert not got.fire and "not on the ladder" in got.reason

    def test_a_more_expensive_tier_is_never_argued_for_here(self):
        got = _advise(plan=_plan(Tier.T1, t1_cost=0.90), model="claude-opus-5")
        assert not got.fire

    def test_a_saving_under_the_floor_is_not_worth_a_sentence(self):
        got = _advise(plan=_plan(Tier.T1, t2_cost=0.42, t1_cost=0.41), min_cost=0.10)
        assert not got.fire and "floor" in got.reason

    def test_a_saving_that_does_not_cover_the_words_is_not_said(self):
        """The guard's own solvency test, applied to this sentence. A short
        session carries the words for a few turns and saves little either way."""
        got = _advise(plan=_plan(Tier.T1, t2_cost=0.4201, t1_cost=0.42),
                      remaining=800, context=900_000, min_cost=0.0)
        assert not got.fire

    def test_uptake_discounts_the_saving_before_it_is_weighed(self):
        """It is advice, not a refusal, so it is worth what gets acted on."""
        cheap = _advise(advice_taken=1.0)
        assert cheap.net > _advise(advice_taken=0.1).net


class TestItNeverRefuses:
    def test_there_is_no_way_for_this_to_deny_a_call(self):
        """Refusing a delegation would refuse the largest lever in the tool, on
        the strength of a classifier that abstains by design."""
        fields = Advice.__dataclass_fields__
        assert "deny" not in fields and "ask" not in fields


class TestTaskText:
    def test_description_and_prompt_are_both_read(self):
        got = task_text({"description": "fix it", "prompt": "across six files"})
        assert "fix it" in got and "across six files" in got

    def test_it_is_bounded(self):
        assert len(task_text({"prompt": "x" * 10_000})) <= 600


class TestModelIdentity:
    """Two spellings of one model must not look like two rungs."""

    @pytest.mark.parametrize("a,b", [
        ("claude-opus-5", "claude-opus-5[1m]"),
        ("claude-opus-5", "claude-opus-5-20260214"),
        ("Claude-Opus-5", "claude-opus-5"),
    ])
    def test_the_same_model_spelled_two_ways(self, a, b):
        assert stem(a) == stem(b)

    def test_different_models_stay_different(self):
        assert stem("claude-opus-5") != stem("claude-sonnet-5")

    def test_a_context_suffix_does_not_trigger_a_recommendation(self):
        """The failure this exists to stop: recommending a switch from a model
        to itself, at the price of the cache the switch throws away."""
        assert not _advise(plan=_plan(Tier.T2), model="claude-opus-5[1m]").fire


class _Sizes:
    """A size model that predicts one number, so the brief gate can be put on
    either side of its threshold without depending on the shipped prior."""

    def __init__(self, tokens):
        self.tokens = tokens

    def predict_tool(self, _tool, _input):
        from adder.core.shapes import Estimate
        return Estimate(p50=self.tokens, p90=self.tokens, n=9, source='test')


class TestThroughTheGuard:
    """The wiring. `advise` is stubbed here -- what is under test is that the
    guard asks, that one call produces one message rather than two, and that the
    message is never a refusal."""

    @pytest.fixture
    def wired(self, monkeypatch):
        def stub(_input, **kw):
            return Advice(True, 'stub', message='[adder] Run this on route-t1.',
                          agent='route-t1', model='claude-sonnet-5',
                          saving=0.20, overhead=0.001, target='Task:tier:T1')
        monkeypatch.setattr('adder.decide.delegate.advise', stub)

    def _decide(self, tokens=10, *, state=None, task='do a thing', **kw):
        from adder.decide.guard import GuardState, Settings, decide
        cfg = Settings(min_cost=0.0, min_tokens=1, max_fires=99, **kw)
        return decide('Task', {'description': task}, model='claude-opus-5',
                      remaining_turns=400, sizes=_Sizes(tokens), cfg=cfg,
                      state=state if state is not None else GuardState(),
                      context_tokens=200_000)

    def test_a_small_return_still_gets_the_tier_advice(self, wired):
        """The return size having nothing to say is not a reason to say nothing."""
        v = self._decide(10)
        assert v.fire and v.kind == 'tier' and 'route-t1' in v.message

    def test_a_large_return_gets_one_message_carrying_both(self, wired):
        """Two sentences about one call are carried twice for the rest of the
        session, so the tier clause joins the brief message instead."""
        v = self._decide(40_000)
        assert v.fire and v.kind == 'brief'
        assert 'route-t1' in v.message and v.message.count('[adder]') <= 2
        assert v.tier_target == 'Task:tier:T1'

    def test_the_tier_advice_is_never_a_denial(self, wired):
        for level in ('off', 'certain', 'full'):
            v = self._decide(10, enforce=level)
            assert v.fire and not v.deny and not v.ask, level

    def test_it_is_said_once_per_session(self, wired):
        from adder.decide.guard import GuardState, observe
        state = GuardState()
        first = self._decide(10, state=state, task='a')
        observe('Task', {'description': 'a'}, state, first)
        second = self._decide(10, state=state, task='b')
        assert first.fire and not second.fire

    def test_turning_it_off_leaves_the_brief_gate_alone(self, wired):
        from adder.decide.guard import Settings, decide
        cfg = Settings(min_cost=0.0, min_tokens=1, max_fires=99, route=False)
        v = decide('Task', {'description': 'a'}, model='claude-opus-5',
                   remaining_turns=400, sizes=_Sizes(40_000), cfg=cfg,
                   context_tokens=200_000)
        assert v.fire and 'route-t1' not in v.message

    def test_a_router_that_raises_does_not_take_the_tool_call_down(self, monkeypatch):
        """A guard that dies on a `Task` is worse than one with nothing to say."""
        def boom(*a, **k):
            raise RuntimeError('no catalog, no outcomes, no anything')
        monkeypatch.setattr('adder.decide.delegate.advise', boom)
        assert self._decide(10) is not None

    def test_the_dollar_floor_applies_to_a_standalone_message_only(self, monkeypatch):
        """`min_cost` is the bar for *interrupting*. A clause appended to a
        message the guard was already sending interrupts nothing, so it is
        gated by whether it pays for its own words and by nothing else."""
        def stub(_input, **kw):
            assert kw['min_cost'] == 0.0, 'the floor belongs to the caller, not the router'
            return Advice(True, 'stub', message='[adder] Run this on route-t0.',
                          agent='route-t0', model='claude-haiku-4-5',
                          saving=0.05, overhead=0.001, target='Task:tier:T0')
        monkeypatch.setattr('adder.decide.delegate.advise', stub)
        from adder.decide.guard import GuardState, Settings, decide
        cfg = Settings(min_cost=0.25, min_tokens=1, max_fires=99)
        call = {'model': 'claude-opus-5', 'remaining_turns': 400, 'cfg': cfg,
                'context_tokens': 200_000}
        alone = decide('Task', {'description': 'a'}, sizes=_Sizes(10),
                       state=GuardState(), **call)
        along = decide('Task', {'description': 'a'}, sizes=_Sizes(40_000),
                       state=GuardState(), **call)
        assert not alone.fire, 'a $0.05 switch is not worth interrupting for'
        assert along.fire and 'route-t0' in along.message
