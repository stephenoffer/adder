"""The four gates: feasibility, placement, escalation risk, and overhead."""
from __future__ import annotations

import time

import pytest

from adder.core.trace import Session, Turn
from adder.decide.route.classify import Tier
from adder.decide.route.policy import choose_effort, decide
from adder.evaluate.claims.savings import cache_discipline, effort_reduction, tool_output_discipline
from adder.pricing.cost import escalation_is_profitable, placement_cost, switch_is_profitable

OPUS, HAIKU = "claude-opus-5", "claude-haiku-4-5"


def _sess(n=100, ctx=200_000, out=500):
    s = Session("s", "p")
    for i in range(n):
        s.turns.append(Turn("s", "p", OPUS, 0, ctx + i * 1000, 0, out, 0, False,
                            ts=f"2026-08-14T10:{i % 60:02d}:00Z"))
    return s


class TestFeasibilityGate:
    def test_switch_to_a_model_that_cannot_hold_the_context_is_refused(self):
        d = switch_is_profitable(OPUS, HAIKU, 544_000, 100_000)
        assert not d and "context limit" in d.reason

    def test_pure_economics_can_be_probed_separately(self):
        d = switch_is_profitable(OPUS, HAIKU, 544_000, 100_000, check_context=False)
        assert d          # 100K output clears the break-even, ignoring feasibility

    def test_delegation_refuses_a_read_larger_than_the_subagent_window(self):
        _, _, d = placement_cost(
            tokens_read=500_000, summary_tokens=5_000, remaining_turns=300,
            main_model=OPUS, sub_model=HAIKU)
        assert not d and "cannot delegate" in d.reason

    def test_escalation_refuses_a_cheap_tier_that_cannot_hold_the_context(self):
        d = escalation_is_profitable(HAIKU, OPUS, ctx_tokens=544_000,
                                     est_out_tokens=500, p_fail=0.0)
        assert not d and "exceeds" in d.reason

    def test_a_huge_read_escalates_the_tier_for_feasibility(self):
        p = decide("what is in the log", context_tokens=100_000, remaining_turns=300,
                   est_read_tokens=400_000, p_fail=0.0)
        assert p.tier >= Tier.T1
        assert any("feasibility" in w for w in p.warnings)


class TestEscalationGate:
    def test_a_tier_that_always_fails_is_not_recommended(self):
        p = decide("what does prices.py do", context_tokens=400_000,
                   remaining_turns=300, p_fail=1.0)
        assert p.model == OPUS

    def test_a_reliable_cheap_tier_is_used(self):
        p = decide("what does prices.py do", context_tokens=400_000,
                   remaining_turns=300, p_fail=0.0)
        assert p.model == HAIKU

    def test_p_fail_is_reported_on_the_plan(self):
        p = decide("what does prices.py do", context_tokens=400_000,
                   remaining_turns=300, p_fail=0.25)
        assert p.p_fail == 0.25

    def test_p_fail_out_of_range_is_rejected(self):
        with pytest.raises(ValueError):
            escalation_is_profitable(HAIKU, OPUS, ctx_tokens=1000,
                                     est_out_tokens=1, p_fail=1.5)


class TestEffortChoice:
    def test_haiku_gets_no_effort_flag(self):
        assert choose_effort(Tier.T0, HAIKU) == "default"

    def test_opus_gets_the_tier_effort(self):
        assert choose_effort(Tier.T2, OPUS) == "high"
        assert choose_effort(Tier.T3, OPUS) == "xhigh"

    def test_plans_always_name_an_effort(self):
        p = decide("refactor the whole system", context_tokens=100_000,
                   remaining_turns=100)
        assert p.effort


class TestNewLevers:
    def test_tool_discipline_targets_the_read_half_of_the_pool(self):
        sessions = {"a": _sess()}
        e = tool_output_discipline(sessions, ".")
        assert e.saving >= 0 and 0.0 <= e.pool_fraction <= 1.0

    def test_effort_reduction_is_bounded_and_labelled(self):
        e = effort_reduction({"a": _sess()})
        assert e.confidence == "MODELLED"
        assert 0.0 <= e.pool_fraction <= 1.0
        assert "prior" in e.assumptions

    def test_effort_reduction_rejects_unknown_levels(self):
        with pytest.raises(ValueError):
            effort_reduction({"a": _sess()}, to_effort="turbo")

    def test_cache_discipline_is_measured_and_never_negative(self):
        e = cache_discipline({"a": _sess()})
        assert e.confidence == "MEASURED" and e.saving >= 0


class TestCrossVendorSubstitution:
    """Gate 5: another vendor's model standing in for a Claude subagent.

    The standing objection to a cheaper model is the model-scoped prompt cache,
    and it is an objection about the *session*. A subagent starts cold, so the
    objection does not apply there — which makes delegation the one placement
    where the vendor is genuinely free, and the only one this is offered for.
    """

    @staticmethod
    def _catalog(tmp_path, monkeypatch, extra=()):
        from adder.pricing.catalog import Catalog, Entry

        base = [
            Entry(key="claude-opus-5", id="claude-opus-5", org="Anthropic",
                  license="Proprietary", inp=5.0, out=25.0, cache_read=0.5,
                  cache_write=6.25, context=1_000_000, params=("tools", "reasoning"),
                  elo={"webdev": 1690.0}, verified=True),
            Entry(key="claude-haiku-4-5", id="claude-haiku-4-5", org="Anthropic",
                  license="Proprietary", inp=1.0, out=5.0, cache_read=0.1,
                  cache_write=1.25, context=200_000, params=("tools",),
                  elo={"webdev": 1330.0}, verified=True),
        ]
        path = tmp_path / "pinned.json"
        Catalog(list(base) + list(extra),
                provenance={"refreshed_at": "2026-08-14T00:00:00+00:00"}).save(path)
        monkeypatch.setenv("ADDER_CATALOG", str(path))
        return path

    @staticmethod
    def _peer(**kw):
        from adder.pricing.catalog import Entry

        defaults = {
            "key": "peer", "id": "vendor/peer", "org": "Vendor",
            "license": "Apache 2.0", "inp": 1.0, "out": 5.0,
            "cache_read": 0.1, "cache_write": 1.25, "context": 1_000_000,
            "params": ("tools", "reasoning"), "elo": {"webdev": 1670.0},
            "verified": False,
        }
        return Entry(**{**defaults, **kw})

    def _delegating_plan(self, read=400_000):
        p = decide("investigate why the ingest pipeline drops records across the codebase",
                   context_tokens=300_000, remaining_turns=150,
                   est_read_tokens=read, p_fail=0.0)
        assert p.action == "delegate"
        return p

    def test_a_warm_session_is_never_offered_another_vendor(self, tmp_path,
                                                            monkeypatch):
        """Inline means the model-scoped cache is in play — a different question.

        Built directly rather than via `decide()`: delegation wins so reliably
        in this cost model that an inline plan is hard to provoke, and the gate
        under test is `action != "delegate"`, not how you got there.
        """
        from adder.decide.route.policy import Plan, substitutes

        self._catalog(tmp_path, monkeypatch, [self._peer(inp=0.1, out=0.5)])
        inline = Plan(action="inline", tier=Tier.T2, model="claude-opus-5",
                      effort="high", agent=None, saving=0.0, overhead=0.16,
                      confidence=0.8, reasons=[])
        assert substitutes(inline, est_read_tokens=400_000, context_tokens=300_000,
                           remaining_turns=150) == []

    def test_the_empty_task_guard_offers_nothing_either(self, tmp_path, monkeypatch):
        """A refusal to route must not sprout a shopping list."""
        from adder.decide.route.policy import substitutes

        self._catalog(tmp_path, monkeypatch, [self._peer(inp=0.1, out=0.5)])
        p = decide("", context_tokens=300_000, remaining_turns=150)
        assert p.action == "inline"
        assert substitutes(p, est_read_tokens=400_000) == []

    def test_a_cheaper_peer_that_clears_the_bar_is_offered(self, tmp_path, monkeypatch):
        from adder.decide.route.policy import substitutes

        self._catalog(tmp_path, monkeypatch, [self._peer()])
        p = self._delegating_plan()
        got = substitutes(p, est_read_tokens=400_000, context_tokens=300_000,
                          remaining_turns=150)
        assert [s.model for s in got] == ["vendor/peer"]
        assert got[0].expected < got[0].baseline

    def test_escalation_is_priced_in_not_ignored(self, tmp_path, monkeypatch):
        """A subagent that fails and gets redone on Opus is not cheap."""
        from adder.decide.route.policy import substitutes

        # Same price, far weaker: every dollar saved comes back as retry risk.
        self._catalog(tmp_path, monkeypatch,
                      [self._peer(inp=4.9, out=24.0, elo={"webdev": 1655.0})])
        got = substitutes(self._delegating_plan(), est_read_tokens=400_000,
                          context_tokens=300_000, remaining_turns=150)
        assert got == []

    def test_a_weaker_substitute_is_more_expensive_than_a_stronger_one(self,
                                                                        tmp_path,
                                                                        monkeypatch):
        """The redo term, stated as behaviour rather than as its own formula.

        Two candidates at an identical price, one further below the tier it
        replaces: the weaker one must cost more in expectation, because the
        gap it opens is paid for in escalations. Asserting the arithmetic back
        to itself would pass even if the formula were wrong.
        """
        from adder.decide.route.policy import substitutes

        strong = self._peer(key="strong", id="v/strong", inp=0.2, out=1.0,
                            elo={"webdev": 1685.0})
        weak = self._peer(key="weak", id="v/weak", inp=0.2, out=1.0,
                          elo={"webdev": 1655.0})
        self._catalog(tmp_path, monkeypatch, [strong, weak])
        got = {s.model: s for s in substitutes(
            self._delegating_plan(), est_read_tokens=400_000,
            context_tokens=300_000, remaining_turns=150, limit=5)}
        assert got["v/weak"].direct == pytest.approx(got["v/strong"].direct)
        assert got["v/weak"].expected > got["v/strong"].expected

    def test_the_cheapest_run_is_not_automatically_the_cheapest_plan(self, tmp_path,
                                                                     monkeypatch):
        """A model cheap enough per run can still lose once the redo is priced."""
        from adder.decide.route.policy import substitutes

        cheap_but_bad = self._peer(key="bad", id="v/bad", inp=0.01, out=0.05,
                                   elo={"webdev": 1655.0})
        self._catalog(tmp_path, monkeypatch, [cheap_but_bad])
        plan = self._delegating_plan(read=20_000)
        got = substitutes(plan, est_read_tokens=20_000, context_tokens=300_000,
                          remaining_turns=150)
        for s in got:
            assert s.expected > s.direct        # the redo is always priced in
            assert s.expected < s.baseline      # and it still had to win to be here

    def test_the_quality_bar_tightens_with_the_tier(self, tmp_path, monkeypatch):
        """A T2 refactor tolerates a 40-point deficit; a T0 lookup tolerates 120."""
        from adder.decide.route.policy import SUBSTITUTE_TOLERANCE, substitutes

        assert SUBSTITUTE_TOLERANCE[Tier.T0] > SUBSTITUTE_TOLERANCE[Tier.T2]
        self._catalog(tmp_path, monkeypatch,
                      [self._peer(elo={"webdev": 1600.0}, inp=0.1, out=0.5)])
        got = substitutes(self._delegating_plan(), est_read_tokens=400_000,
                          context_tokens=300_000, remaining_turns=150)
        assert got == []          # 90 points below Opus is outside T2 tolerance

    def test_an_unrated_baseline_offers_nothing(self, tmp_path, monkeypatch):
        """With no rating on the model being replaced there is no bar to clear."""
        from adder.decide.route.policy import substitutes
        from adder.pricing.catalog import Catalog, Entry

        path = tmp_path / "unrated.json"
        Catalog([Entry(key="claude-opus-5", id="claude-opus-5", org="Anthropic",
                       inp=5.0, out=25.0, context=1_000_000, params=("tools",)),
                 self._peer()]).save(path)
        monkeypatch.setenv("ADDER_CATALOG", str(path))
        assert substitutes(self._delegating_plan(), est_read_tokens=400_000) == []

    def test_a_saving_under_the_routing_overhead_is_not_shown_as_advice(
            self, tmp_path, monkeypatch):
        """A 0.6-cent saving is noise with a dollar sign on it."""
        from adder.decide.route.policy import substitutes

        self._catalog(tmp_path, monkeypatch, [self._peer(inp=4.99, out=24.99)])
        p = self._delegating_plan(read=8_000)
        p.substitutes = substitutes(p, est_read_tokens=8_000, context_tokens=300_000,
                                    remaining_turns=150)
        if p.substitutes:
            assert all(s.saving < p.overhead for s in p.substitutes)
            out = p.render()
            assert "the placement was the lever, not the vendor" in out
            assert "vendor/peer" in out          # named, with its number

    def test_a_material_saving_is_shown_as_a_table(self, tmp_path, monkeypatch):
        from adder.decide.route.policy import substitutes

        self._catalog(tmp_path, monkeypatch, [self._peer(inp=0.2, out=1.0,
                                                         elo={"webdev": 1680.0})])
        p = self._delegating_plan()
        p.substitutes = substitutes(p, est_read_tokens=400_000, context_tokens=300_000,
                                    remaining_turns=150)
        out = p.render()
        assert "vendor/peer" in out and "saves $" in out
        assert "not as a Claude Code subagent" in out

    def test_a_missing_catalog_entry_returns_nothing_rather_than_guessing(
            self, tmp_path, monkeypatch):
        from adder.decide.route.policy import substitutes
        from adder.pricing.catalog import Catalog

        path = tmp_path / "empty.json"
        Catalog([]).save(path)
        monkeypatch.setenv("ADDER_CATALOG", str(path))
        monkeypatch.setenv("ADDER_HOME", str(tmp_path / "home"))
        # The first-party layer still supplies Opus, but with no arena rating.
        assert substitutes(self._delegating_plan(), est_read_tokens=400_000) == []

    def test_measured_escalation_history_feeds_into_the_estimate(self, tmp_path,
                                                                 monkeypatch):
        """The one real measurement in the building is used where it exists."""
        import adder.decide.track.outcomes as outcomes_mod
        from adder.decide.route.policy import substitutes
        from adder.decide.track.outcomes import Outcome, record

        # Cheap enough to survive the extra risk the history adds; the point of
        # the test is the estimate, not the filter.
        self._catalog(tmp_path, monkeypatch, [self._peer(inp=0.05, out=0.25)])
        log = tmp_path / "outcomes.jsonl"
        monkeypatch.setattr(outcomes_mod, "DEFAULT_LOG", log)
        now = time.time()
        for i in range(30):
            record(Outcome(tier="T2", model=OPUS, project="p", escalated=i % 3 == 0,
                           cost=0.1, ts=now - i * 60), log)

        plan = self._delegating_plan()
        with_history = substitutes(plan, est_read_tokens=400_000,
                                   context_tokens=300_000, remaining_turns=150,
                                   project="p")
        monkeypatch.setattr(outcomes_mod, "DEFAULT_LOG", tmp_path / "empty.jsonl")
        without = substitutes(plan, est_read_tokens=400_000, context_tokens=300_000,
                              remaining_turns=150, project="p")
        assert with_history and without
        assert with_history[0].p_fail > without[0].p_fail
        assert "history" in with_history[0].basis
        assert without[0].basis == "elo"

    def test_an_unreadable_outcome_log_falls_back_to_elo(self, tmp_path, monkeypatch):
        """A broken log must degrade the estimate, not the recommendation."""
        import adder.decide.track.outcomes as outcomes_mod
        from adder.decide.route.policy import substitutes

        self._catalog(tmp_path, monkeypatch, [self._peer(inp=0.2, out=1.0)])
        bad = tmp_path / "bad.jsonl"
        bad.write_text("{not json\n")
        monkeypatch.setattr(outcomes_mod, "DEFAULT_LOG", bad)
        got = substitutes(self._delegating_plan(), est_read_tokens=400_000,
                          context_tokens=300_000, remaining_turns=150)
        assert got and got[0].basis == "elo"

    def test_prices_from_an_aggregator_are_flagged_on_every_row(self, tmp_path,
                                                                monkeypatch):
        from adder.decide.route.policy import substitutes

        self._catalog(tmp_path, monkeypatch, [self._peer(inp=0.2, out=1.0)])
        for s in substitutes(self._delegating_plan(), est_read_tokens=400_000,
                             context_tokens=300_000, remaining_turns=150):
            assert not s.verified
            assert "unverified" in s.reachable


class TestRightSizing:
    """Gate 3 as a ladder search rather than a one-way escalation."""

    @staticmethod
    def _log(tmp_path, monkeypatch, rows):
        """Point the outcome log at a temp file. Never touch the real one."""
        import json
        import time

        path = tmp_path / "outcomes.jsonl"
        now = time.time()
        with path.open("w") as fh:
            for tier, n, fails in rows:
                for i in range(n):
                    fh.write(json.dumps({
                        "tier": tier, "model": "m", "project": "proj",
                        "escalated": i < fails, "ts": now - i * 3600,
                    }) + "\n")
        monkeypatch.setattr("adder.decide.track.outcomes.DEFAULT_LOG", path)
        return path

    def test_an_abstention_alone_never_buys_a_downgrade(self, tmp_path, monkeypatch):
        """No evidence means the cheapest rung is not an option, however cheap."""
        self._log(tmp_path, monkeypatch, [])
        p = decide("make the ingest step tolerate a partial batch",
                   context_tokens=300_000, remaining_turns=200, project="proj")
        assert p.tier == Tier.T2
        cheaper = [r for r in p.ladder if r.tier < Tier.T2]
        assert cheaper and all(not r.allowed for r in cheaper)
        assert all("prior is not evidence" in r.note for r in cheaper)

    def test_measured_history_does_buy_one(self, tmp_path, monkeypatch):
        """The escalation loop is supposed to observe failure, not only fear it."""
        self._log(tmp_path, monkeypatch, [("T1", 40, 4), ("T2", 40, 2)])
        p = decide("make the ingest step tolerate a partial batch",
                   context_tokens=300_000, remaining_turns=200, project="proj")
        assert p.tier == Tier.T1 and p.model == "claude-sonnet-5"
        chosen = next(r for r in p.ladder if r.tier is p.tier)
        t2 = next(r for r in p.ladder if r.tier is Tier.T2)
        assert chosen.expected < t2.expected

    def test_a_confident_classification_is_not_overridden_by_evidence(
            self, tmp_path, monkeypatch):
        """Cheapness may not contradict a signal the classifier actually matched."""
        self._log(tmp_path, monkeypatch, [("T0", 60, 1), ("T1", 60, 1)])
        p = decide("refactor the storage layer across the codebase",
                   context_tokens=300_000, remaining_turns=200, project="proj")
        assert p.tier >= Tier.T2
        assert any("matched a signal for" in r.note for r in p.ladder if r.tier < Tier.T2)

    def test_a_tier_over_its_own_breakeven_is_refused_even_with_history(
            self, tmp_path, monkeypatch):
        self._log(tmp_path, monkeypatch, [("T1", 40, 38)])
        p = decide("make the ingest step tolerate a partial batch",
                   context_tokens=300_000, remaining_turns=200, project="proj")
        assert p.tier == Tier.T2
        t1 = next(r for r in p.ladder if r.tier is Tier.T1)
        assert "break-even" in t1.note

    def test_failure_rate_is_monotone_up_the_ladder(self, tmp_path, monkeypatch):
        """T3 is T2's model at higher effort; it cannot be likelier to fail."""
        self._log(tmp_path, monkeypatch, [("T2", 40, 2)])
        p = decide("make the ingest step tolerate a partial batch",
                   context_tokens=300_000, remaining_turns=200, project="proj")
        rates = [r.p_fail for r in sorted(p.ladder, key=lambda r: r.tier)]
        assert rates == sorted(rates, reverse=True)

    def test_an_infeasible_rung_is_reported_not_silently_dropped(self):
        p = decide("what is in the log", context_tokens=100_000, remaining_turns=300,
                   est_read_tokens=400_000)
        t0 = next(r for r in p.ladder if r.tier is Tier.T0)
        assert not t0.feasible and "holds" in t0.note

    def test_the_ladder_is_shown_in_the_rendered_plan(self):
        p = decide("what does prices.py do", context_tokens=400_000, remaining_turns=400)
        out = p.render()
        assert "Tier chosen by expected cost" in out
        assert all(t.name in out for t in Tier)

    def test_the_prior_tracks_classifier_confidence(self):
        from adder.decide.route.policy import prior_p_fail

        assert prior_p_fail(0.85) < prior_p_fail(0.60) <= prior_p_fail(0.30) == 0.5
        assert prior_p_fail(1.0) >= 0.05

    def test_a_failed_cheap_run_is_charged_once_not_twice(self):
        """`cheap + p*(exp + overhead)`; the cheap attempt happened one time."""
        from adder.pricing.cost import escalation_is_profitable, run_cost

        cheap = run_cost(HAIKU, 20_000, 2_000)
        exp = run_cost(OPUS, 20_000, 2_000)
        d = escalation_is_profitable(HAIKU, OPUS, ctx_tokens=20_000,
                                     est_out_tokens=2_000, p_fail=0.5)
        assert d.saving == pytest.approx(exp - (cheap + 0.5 * exp))

    def test_the_turn_that_catches_a_failure_is_priced(self):
        from adder.pricing.cost import max_tolerable_p_fail

        free = max_tolerable_p_fail(HAIKU, OPUS, ctx_tokens=9_200, est_out_tokens=800)
        paid = max_tolerable_p_fail(HAIKU, OPUS, ctx_tokens=9_200, est_out_tokens=800,
                                    retry_overhead=0.21)
        assert paid < free, "a failure the main session has to notice is not free"


class TestGateFive:
    """The gate the other four were missing: is the saving positive, or only
    positive at the midpoint of three estimates nobody measured the spread of?"""

    def _h(self, lengths):
        from adder.measure.session.horizon import Horizon

        return Horizon(sorted(lengths))

    def test_guarantee_midpoint_matches_the_point_decision(self):
        """If these disagreed, the gate would be reporting one number and acting
        on another."""
        from adder.decide.route.policy import assess_placement

        point, g = assess_placement(
            tokens_read=20_000, summary_tokens=2_000, remaining_turns=400,
            session_model=OPUS, sub_model=HAIKU, overhead=0.2, p_redo=0.15)
        assert g.expected == pytest.approx(point.saving)

    def test_a_wide_horizon_lowers_confidence(self):
        """Same expected saving, different certainty. The point estimate cannot
        tell these apart and the gate must."""
        from adder.decide.route.policy import assess_placement

        tight = self._h([400] * 40)
        wide = self._h([5] * 20 + [800] * 20)
        _, a = assess_placement(tokens_read=20_000, summary_tokens=2_000,
                                remaining_turns=400, session_model=OPUS,
                                sub_model=HAIKU, overhead=0.2, p_redo=0.15,
                                horizon=tight)
        _, b = assess_placement(tokens_read=20_000, summary_tokens=2_000,
                                remaining_turns=400, session_model=OPUS,
                                sub_model=HAIKU, overhead=0.2, p_redo=0.15,
                                horizon=wide)
        assert a.confidence > b.confidence

    def test_no_measured_inputs_claims_no_uncertainty(self):
        """A caller with nothing to measure gets the old behaviour, not an
        invented interval."""
        p = decide("what does prices.py do", context_tokens=400_000,
                   remaining_turns=500)
        assert p.action == "delegate" and p.guarantee.confidence == 1.0

    def test_an_impossible_bar_declines_everything(self):
        p = decide("what does prices.py do", context_tokens=400_000,
                   remaining_turns=500, min_confidence=1.01)
        assert p.action == "inline"
        assert any("not confident enough" in r for r in p.reasons)

    def test_declining_does_not_also_argue_for_delegating(self):
        """A reason list that recommends delegation inside an `inline` plan is
        how a router loses an argument with the person reading it."""
        p = decide("what does prices.py do", context_tokens=400_000,
                   remaining_turns=500, min_confidence=1.01)
        assert not any(r.startswith("delegate:") for r in p.reasons)

    def test_dominance_is_reported_when_it_holds(self):
        p = decide("what does prices.py do", context_tokens=400_000,
                   remaining_turns=500)
        assert p.guarantee.dominant
        assert "cheaper at every admissible input" in p.render()


class TestHaircut:
    """Bounds protect the gate from uncertain inputs. Nothing in them protects
    it from the model being biased, which is what this measures."""

    def _led(self, predicted, realized, n=12):
        from adder.decide.track.ledger import Entry, Ledger

        return Ledger([Entry("delegate", predicted, predicted / 2, 0.1,
                             realized=realized) for _ in range(n)])

    def test_an_over_promising_model_throttles_itself(self):
        base = decide("what does prices.py do", context_tokens=400_000,
                      remaining_turns=500)
        cut = decide("what does prices.py do", context_tokens=400_000,
                     remaining_turns=500, ledger=self._led(1.0, 0.25))
        assert cut.guarantee.expected < base.guarantee.expected

    def test_the_discount_is_visible_not_silent(self):
        p = decide("what does prices.py do", context_tokens=400_000,
                   remaining_turns=500, ledger=self._led(1.0, 0.25))
        assert any("delivered 25%" in w for w in p.warnings)

    def test_an_honest_model_is_not_penalised(self):
        base = decide("what does prices.py do", context_tokens=400_000,
                      remaining_turns=500)
        same = decide("what does prices.py do", context_tokens=400_000,
                      remaining_turns=500, ledger=self._led(1.0, 1.0))
        assert same.guarantee.expected == pytest.approx(base.guarantee.expected)

    def test_a_missing_ledger_never_penalises(self):
        from adder.decide.route.policy import _haircut

        class Broken:
            def haircut(self):
                raise OSError("disk gone")

        assert _haircut(Broken()) == 1.0 and _haircut(None) == 1.0


class TestBatching:
    """Overhead is charged per routing turn, not per recommendation. Asking once
    about five steps costs one turn."""

    def _tasks(self, n):
        return ["what does prices.py do"] * n

    def test_the_batch_pays_one_overhead(self):
        from adder.decide.route.policy import routing_overhead, schedule

        b = schedule(self._tasks(4), context_tokens=400_000, remaining_turns=400)
        assert b.overhead == pytest.approx(routing_overhead(400_000, OPUS))

    def test_savings_add_but_overhead_does_not(self):
        from adder.decide.route.policy import schedule

        one = schedule(self._tasks(1), context_tokens=400_000, remaining_turns=400)
        four = schedule(self._tasks(4), context_tokens=400_000, remaining_turns=400)
        assert four.saving == pytest.approx(4 * one.saving)
        assert four.overhead == pytest.approx(one.overhead)

    def test_batching_raises_confidence(self):
        """The interesting consequence: the horizon risk is shared across steps
        and does not average down, but the total clears a fixed overhead by a
        wider margin as steps are added."""
        from adder.decide.route.policy import schedule
        from adder.measure.session.horizon import Horizon

        h = Horizon(sorted([20] * 10 + [60] * 10 + [400] * 10 + [900] * 10))
        one = schedule(self._tasks(1), context_tokens=400_000,
                       remaining_turns=400, horizon=h)
        six = schedule(self._tasks(6), context_tokens=400_000,
                       remaining_turns=400, horizon=h)
        assert six.confidence >= one.confidence

    def test_an_empty_batch_is_not_worth_it(self):
        from adder.decide.route.policy import Batch

        assert not Batch().worth_it
        assert Batch().render() == "nothing to route"


class TestOverheadBar:
    """The bar every recommendation clears. Understating it is the one error
    that makes adder emit advice too eagerly."""

    def test_a_measured_multiplier_raises_the_bar(self):
        from adder.decide.route.policy import routing_overhead
        from adder.measure.window.carry import Carry

        measured = Carry(read_mult=0.20, source="measured")
        assert routing_overhead(500_000, OPUS, carry=measured) > \
            routing_overhead(500_000, OPUS)

    def test_the_default_is_the_cache_read_rate(self):
        from adder.decide.route.policy import routing_overhead
        from adder.measure.window.carry import Carry
        from adder.pricing.prices import CACHE_READ_MULT

        assert routing_overhead(500_000, OPUS) == pytest.approx(
            routing_overhead(500_000, OPUS,
                             carry=Carry(read_mult=CACHE_READ_MULT)))

    def test_a_dearer_bar_can_flip_a_marginal_recommendation(self):
        from adder.measure.window.carry import Carry

        cheap = decide("what does prices.py do", context_tokens=900_000,
                       remaining_turns=6, est_read_tokens=20_000)
        dear = decide("what does prices.py do", context_tokens=900_000,
                      remaining_turns=6, est_read_tokens=20_000,
                      carry=Carry(read_mult=0.60, source="measured"))
        assert dear.overhead > cheap.overhead


class TestLedgerRecording:
    """`decide` reads the ledger for its haircut; until now nothing wrote one."""

    @staticmethod
    def _ledger(tmp_path, monkeypatch):
        path = tmp_path / "ledger.jsonl"
        monkeypatch.setattr("adder.decide.track.ledger.DEFAULT_LEDGER", path)
        return path

    def test_a_recommendation_is_booked_with_both_sides(self, tmp_path, monkeypatch):
        from adder.decide.route.policy import _record_to_ledger
        from adder.decide.track.ledger import load

        path = self._ledger(tmp_path, monkeypatch)
        p = decide("what does prices.py do", context_tokens=400_000,
                   remaining_turns=400, project="demo")
        _record_to_ledger(p, project="demo", session="s1")
        e = load(path)[0]
        assert e.action == p.action
        assert e.predicted == pytest.approx(p.saving)
        assert e.overhead == pytest.approx(p.overhead)
        assert e.project == "demo" and e.session == "s1"

    def test_the_worst_case_comes_from_the_guarantee_not_the_midpoint(
            self, tmp_path, monkeypatch):
        """Solvency is judged on the worst case; booking the midpoint would flatter it."""
        from adder.decide.route.policy import _record_to_ledger
        from adder.decide.track.ledger import load

        path = self._ledger(tmp_path, monkeypatch)
        p = decide("what does prices.py do", context_tokens=400_000,
                   remaining_turns=400, project="demo")
        _record_to_ledger(p)
        e = load(path)[0]
        if p.guarantee is not None:
            assert e.worst == pytest.approx(p.guarantee.worst)
            assert e.worst <= e.predicted

    def test_a_declined_recommendation_is_still_booked(self, tmp_path, monkeypatch):
        """It cost a routing turn. Counting only the hits makes the tool look free."""
        from adder.decide.route.policy import _record_to_ledger
        from adder.decide.track.ledger import load

        path = self._ledger(tmp_path, monkeypatch)
        p = decide("what does prices.py do", context_tokens=900_000, remaining_turns=3)
        assert not p.worth_it
        _record_to_ledger(p)
        assert load(path)[0].accepted is False

    def test_an_unwritable_ledger_never_breaks_routing(self, tmp_path, monkeypatch):
        from adder.decide.route.policy import _record_to_ledger

        monkeypatch.setattr("adder.decide.track.ledger.DEFAULT_LEDGER", tmp_path / "x" / "\0bad")
        p = decide("what does prices.py do", context_tokens=400_000, remaining_turns=400)
        _record_to_ledger(p)          # must not raise

    def test_recording_is_opt_in(self, tmp_path, monkeypatch, capsys):
        """A read-mostly command must not start writing because it was run."""
        from adder.decide.route.policy import main as policy_main
        from adder.decide.track.ledger import load

        path = self._ledger(tmp_path, monkeypatch)
        policy_main(["what does prices.py do", "--context", "400000",
                     "--remaining", "400", "--no-cross-vendor", "--no-measure"])
        capsys.readouterr()
        assert load(path) == []


class TestTheRuntimeDecidesWhichPlacementsExist:
    """`harness.supports_subagents` and `supports_model_switch` were never read.

    `harness.py` says why they exist -- delegation "is not available, and a
    report that recommends it anyway is recommending a feature the user does
    not have"; a runtime that cannot switch model mid-session makes the
    downgrade question "moot and should not be raised". Nothing outside that
    module consulted either field, so `decide` emitted both recommendations
    whatever was running.
    """

    TASK = "read every file under src and summarise the architecture"

    def _plan(self, harness, **kw):
        return decide(self.TASK, context_tokens=400_000, remaining_turns=500,
                      harness=harness, **kw)

    def test_claude_code_still_delegates(self, isolated_home):
        assert self._plan("claude-code").action == "delegate"

    def test_a_single_conversation_runtime_does_not(self, isolated_home):
        assert self._plan("aider").action != "delegate"

    def test_and_says_why(self, isolated_home):
        warned = " ".join(self._plan("aider").warnings)
        assert "no subagent" in warned

    def test_the_reasons_do_not_contradict_the_plan(self, isolated_home):
        """An `inline` plan whose reasons end in "delegate: saves $1.05"."""
        p = self._plan("aider")
        assert not any(r.startswith("delegate:") for r in p.reasons)

    def test_the_saving_is_not_claimed(self, isolated_home):
        assert self._plan("aider").saving == 0.0

    def test_a_runtime_that_cannot_switch_model_is_not_offered_a_downgrade(
            self, isolated_home, monkeypatch):
        import adder.core.harness as h

        fixed = h.replace(h.get("aider"), supports_model_switch=False)
        monkeypatch.setattr(h, "get", lambda name: fixed)
        p = decide("what is x", context_tokens=400_000, remaining_turns=500,
                   harness="aider")
        assert p.action != "downgrade"
        assert any("cannot switch model" in r for r in p.reasons)


class TestTheDecisionBoundaries:
    """Mutation testing found 20 of 62 comparisons here untested.

    Every one turned out to be the correct operator -- no bug -- but a gate
    whose boundary no test pins is a gate that can be flipped in a refactor
    without anything going red. These are the comparisons that decide whether
    money is spent, so each gets its edge pinned.
    """

    @staticmethod
    def _batch(saving, overhead, confidence, threshold=0.0, acted=True):
        from adder.decide.route.classify import Tier
        from adder.decide.route.policy import Batch, Plan

        plans = [Plan(action="delegate" if acted else "inline", tier=Tier.T2,
                      model="claude-opus-5", effort="high", agent=None,
                      saving=saving, overhead=0.0, confidence=1.0, reasons=[])]
        return Batch(plans=plans, overhead=overhead, confidence=confidence,
                     threshold=threshold)

    def test_a_batch_exactly_at_its_overhead_is_not_worth_it(self):
        """`saving > overhead`: breaking even does not pay for the turn."""
        assert not self._batch(1.0, 1.0, 1.0).worth_it

    def test_a_batch_just_over_its_overhead_is(self):
        assert self._batch(1.01, 1.0, 1.0).worth_it

    def test_confidence_exactly_at_the_threshold_passes(self):
        """`confidence >= threshold`, matching `Guarantee.safe`."""
        assert self._batch(5.0, 1.0, 0.5, threshold=0.5).worth_it
        assert not self._batch(5.0, 1.0, 0.49, threshold=0.5).worth_it

    def test_a_batch_that_acted_on_nothing_is_never_worth_it(self):
        assert not self._batch(5.0, 1.0, 1.0, acted=False).worth_it

    def test_worth_it_uses_the_same_rule_as_the_guarantee(self):
        """`Batch.worth_it` mirrors `Guarantee.safe`; they must not drift."""
        from adder.util.risk import Guarantee

        g = Guarantee(expected=1.0, worst=0.0, confidence=0.5, overhead=1.0,
                      threshold=0.5)
        assert g.safe is False                       # saving == overhead
        assert self._batch(1.0, 1.0, 0.5, threshold=0.5).worth_it is False
