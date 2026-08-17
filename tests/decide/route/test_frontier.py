"""The frontier, pinned on the comparison that decides what gets bought.

The property under test throughout: a model whose "lead" sits inside its own
confidence interval does not get to stay on the frontier just because its point
estimate is higher.
"""

from __future__ import annotations

import json

import pytest

from adder.decide.route import frontier as fr_
from adder.decide.route.frontier import Node, build
from adder.decide.route.select import Need
from adder.pricing.catalog import Catalog, Entry


def _node(name, cost, rating, half=0.0, org="Org"):
    e = Entry(key=name, id=name, name=name, org=org, inp=1.0, out=5.0)
    return Node(e, cost=cost, rating=rating, lo=rating - half, hi=rating + half)


class TestDomination:
    def test_cheaper_and_measurably_better_dominates(self):
        cheap_good = _node("a", 1.0, 1300, half=5)
        dear_bad = _node("b", 5.0, 1200, half=5)
        assert cheap_good.dominates(dear_bad)
        assert not dear_bad.dominates(cheap_good)

    def test_cheaper_and_indistinguishable_dominates(self):
        """The rule that removes the models whose lead is noise."""
        cheap = _node("a", 1.0, 1241, half=12)
        dear = _node("b", 4.0, 1247, half=12)
        assert cheap.dominates(dear)
        assert not dear.dominates(cheap)

    def test_cheaper_but_measurably_worse_does_not_dominate(self):
        cheap = _node("a", 1.0, 1100, half=5)
        dear = _node("b", 4.0, 1300, half=5)
        assert not cheap.dominates(dear)
        assert not dear.dominates(cheap)

    def test_same_cost_needs_a_measurable_edge(self):
        a = _node("a", 2.0, 1250, half=20)
        b = _node("b", 2.0, 1245, half=20)
        assert not a.dominates(b)
        assert not b.dominates(a)

    def test_same_cost_with_a_real_gap_does_dominate(self):
        a = _node("a", 2.0, 1400, half=5)
        b = _node("b", 2.0, 1200, half=5)
        assert a.dominates(b)

    def test_beats_requires_disjoint_intervals(self):
        assert _node("a", 1.0, 1300, half=5).beats(_node("b", 1.0, 1200, half=5))
        assert not _node("a", 1.0, 1300, half=60).beats(_node("b", 1.0, 1200, half=60))

    def test_a_point_estimate_is_a_zero_width_interval(self):
        assert not _node("a", 1.0, 1200).measured
        assert _node("a", 1.0, 1201).beats(_node("b", 1.0, 1200))


class TestFrontier:
    @staticmethod
    def _fr(nodes):
        f = fr_.Frontier(considered=len(nodes))
        for n in nodes:
            if any(o.dominates(n) for o in nodes if o is not n):
                f.dominated.append(n)
            else:
                f.nodes.append(n)
        f.nodes.sort(key=lambda n: n.cost)
        f.dominated.sort(key=lambda n: n.cost)
        return f

    def test_the_noise_lead_model_is_dropped(self):
        f = self._fr([_node("cheap", 1.0, 1241, half=12),
                      _node("dear", 4.0, 1247, half=12),
                      _node("best", 9.0, 1400, half=8)])
        assert [n.id for n in f.nodes] == ["cheap", "best"]
        assert [n.id for n in f.dominated] == ["dear"]

    def test_a_real_trade_off_keeps_both(self):
        f = self._fr([_node("cheap", 1.0, 1100, half=5),
                      _node("dear", 4.0, 1300, half=5)])
        assert len(f.nodes) == 2

    def test_cheapest_and_best_are_reported(self):
        f = self._fr([_node("a", 1.0, 1100, half=5), _node("b", 4.0, 1300, half=5)])
        assert f.cheapest.id == "a"
        assert f.best.id == "b"

    def test_an_empty_frontier_has_no_cheapest(self):
        f = fr_.Frontier()
        assert f.cheapest is None and f.best is None

    def test_equivalents_are_found_among_the_dominated(self):
        """The free substitutions are exactly what the cheapest one displaced."""
        f = self._fr([_node("cheap", 1.0, 1241, half=12),
                      _node("dear", 4.0, 1247, half=12),
                      _node("best", 9.0, 1400, half=8)])
        assert [n.id for n in f.equivalent_to("cheap")] == ["dear"]

    def test_no_equivalents_when_everything_is_separable(self):
        f = self._fr([_node("a", 1.0, 1100, half=2), _node("b", 4.0, 1300, half=2)])
        assert f.equivalent_to("a") == []

    def test_equivalence_of_an_unknown_model_is_empty(self):
        assert fr_.Frontier().equivalent_to("nope") == []

    def test_json_is_finite_and_complete(self):
        f = self._fr([_node("a", 1.0, 1100, half=5), _node("b", 4.0, 1300, half=5)])
        payload = f.to_json()
        text = json.dumps(payload)
        assert "NaN" not in text and "Infinity" not in text
        assert {r["id"] for r in payload["frontier"]} == {"a", "b"}


class TestMarginal:
    def test_the_step_price_is_dollars_per_rating_point(self):
        f = TestFrontier._fr([_node("a", 1.0, 1100, half=5),
                              _node("b", 3.0, 1200, half=5)])
        steps = fr_.marginal(f)
        assert len(steps) == 1
        _, _, price = steps[0]
        assert price == pytest.approx(2.0 / 100.0)

    def test_a_step_with_no_quality_gain_is_not_reported(self):
        f = TestFrontier._fr([_node("a", 1.0, 1200, half=90),
                              _node("b", 3.0, 1200, half=90)])
        assert fr_.marginal(f) == []

    def test_no_steps_on_a_single_model(self):
        assert fr_.marginal(TestFrontier._fr([_node("a", 1.0, 1200)])) == []


class TestReport:
    def test_it_names_the_free_substitutions(self):
        f = TestFrontier._fr([_node("cheap", 1.0, 1241, half=12),
                              _node("dear", 4.0, 1247, half=12)])
        text = fr_.report(f)
        assert "indistinguishable" in text.lower()
        assert "dear" in text

    def test_it_says_when_the_trade_off_is_real(self):
        f = TestFrontier._fr([_node("a", 1.0, 1100, half=2),
                              _node("b", 4.0, 1300, half=2)])
        assert "real trade-off" in fr_.report(f)

    def test_it_always_discloses_the_proxy(self):
        f = TestFrontier._fr([_node("a", 1.0, 1100, half=2)])
        assert "MODELLED" in fr_.report(f)

    def test_an_empty_frontier_says_so(self):
        assert "No model" in fr_.report(fr_.Frontier())


class TestBuild:
    def test_it_builds_against_the_bundled_catalog(self):
        from adder.decide.route.select import Need
        from adder.pricing.catalog import load

        f = fr_.build(load(), Need(context_tokens=50_000, remaining_turns=20))
        assert f.considered > 0
        assert f.nodes
        # Nothing on the frontier may be dominated by anything else on it.
        for a in f.nodes:
            assert not any(b.dominates(a) for b in f.nodes if b is not a)

    def test_a_board_nobody_publishes_yields_nothing(self):
        from adder.decide.route.select import Need
        from adder.pricing.catalog import load

        f = fr_.build(load(), Need(), board="no-such-board")
        assert f.nodes == []
        assert f.unmeasured > 0


class TestCli:
    def test_it_runs_and_prints(self, capsys, isolated_home):
        assert fr_.main(["fix a failing test"]) in (0, 1)
        assert capsys.readouterr().out.strip()

    def test_json_parses(self, capsys, isolated_home):
        fr_.main(["--json"])
        payload = json.loads(capsys.readouterr().out)
        assert "frontier" in payload

    def test_an_unknown_board_exits_one(self, capsys, isolated_home):
        assert fr_.main(["--board", "nope"]) == 1
        assert capsys.readouterr().out.strip()


class TestUnrated:
    def test_every_considered_model_is_accounted_for(self):
        """Models dropped for having no price must be counted, not vanish."""
        from adder.decide.route.select import Need
        from adder.pricing.catalog import load

        f = fr_.build(load(), Need(), board=fr_.DEFAULT_BOARD)
        assert f.considered == (len(f.nodes) + len(f.dominated)
                                + f.unmeasured + f.unpriced)

    def test_a_point_estimate_produces_a_confident_ranking_it_should_not(self):
        """The known limitation of a catalog entry with no published interval.

        A rating with no interval is treated as zero-width, so a one-point
        difference reads as a decisive one and the cheaper model does NOT
        dominate. That is wrong in substance and right in mechanism: the fix is
        a catalog that publishes intervals, not arithmetic that invents one.
        The report marks these rows `no interval` so the reader can discount
        them.
        """
        a, b = _node("a", 1.0, 1200), _node("b", 2.0, 1201)
        assert not a.measured and not b.measured
        assert b.beats(a)
        assert not a.dominates(b)

    def test_a_published_interval_removes_that_false_confidence(self):
        a, b = _node("a", 1.0, 1200, half=12), _node("b", 2.0, 1201, half=12)
        assert not b.beats(a)
        assert a.dominates(b)


class TestAFreeModelBelongsOnTheFrontier:
    """Zero is a price. A rated model that costs nothing dominates its whole row.

    `build` rejected any candidate whose cost came out `<= 0` and counted it as
    "rated but unpriced" -- a different question, and one `_eligible` has
    already answered. Free endpoints are real: the catalog carries sixteen
    `:free` rows, and they only became reachable once `Entry.to_json` stopped
    erasing a zero price on the way to disk.
    """

    @staticmethod
    def _free(**kw):
        return Entry(key="freebie", id="freebie", org="X", inp=0.0, out=0.0,
                     context=1_000_000, params=("tools",),
                     elo={"webdev": 1500.0}, elo_lo={"webdev": 1490.0},
                     elo_hi={"webdev": 1510.0}, **kw)

    @staticmethod
    def _paid():
        return Entry(key="paid", id="paid", org="Y", inp=5.0, out=25.0,
                     context=1_000_000, params=("tools",),
                     elo={"webdev": 1400.0}, elo_lo={"webdev": 1390.0},
                     elo_hi={"webdev": 1410.0})

    def _need(self, **kw):
        return Need(context_tokens=50_000, remaining_turns=100,
                    est_read_tokens=10_000, **kw)

    def test_a_zero_cost_candidate_is_not_counted_as_unpriced(self):
        free = self._free()
        fr = build(Catalog([free]), self._need(session_model="freebie"), session=free)
        assert fr.unpriced == 0

    def test_it_reaches_the_frontier_at_zero(self):
        free = self._free()
        fr = build(Catalog([free]), self._need(session_model="freebie"), session=free)
        assert [(n.entry.id, n.cost) for n in fr.nodes] == [("freebie", 0.0)]

    def test_it_dominates_a_dearer_and_weaker_model(self):
        paid = self._paid()
        fr = build(Catalog([self._free(), paid]), self._need(), session=paid)
        assert [n.entry.id for n in fr.nodes] == ["freebie"]
        assert [n.entry.id for n in fr.dominated] == ["paid"]

    def test_an_entry_with_no_price_at_all_is_still_excluded(self):
        """The distinction the old test collapsed."""
        unpriced = Entry(key="mystery", id="mystery", org="X", context=1_000_000,
                         params=("tools",), elo={"webdev": 1600.0},
                         elo_lo={"webdev": 1590.0}, elo_hi={"webdev": 1610.0})
        paid = self._paid()
        fr = build(Catalog([unpriced, paid]), self._need(), session=paid)
        assert "mystery" not in [n.entry.id for n in fr.nodes]


class TestDifficultyIsNotConfidence:
    """The frontier passed the classifier's *confidence* as the task difficulty.

    They are different quantities and, in the case that matters, close to
    inverses. An abstention carries confidence 0.3 and routes the task UP --
    the classifier saying it cannot tell how deep the work goes. Read as a
    difficulty, 0.3 is the easiest setting there is: it widens the quality
    floor (`150 / difficulty`) and shrinks the modelled Elo gap, so the tasks
    the classifier understood least were offered the weakest models with the
    loosest tolerance.
    """

    def test_a_lookup_is_easier_than_a_redesign(self):
        from adder.decide.route.classify import difficulty_of

        easy = difficulty_of("where is the config file")
        hard = difficulty_of("redesign the storage layer across the whole service")
        assert easy < hard

    def test_an_abstention_is_not_the_easiest_setting(self):
        from adder.decide.route.classify import classify, difficulty_of

        v = classify("make it work")
        assert v.abstained
        assert difficulty_of("make it work") > v.confidence
        assert difficulty_of("make it work") >= difficulty_of("where is the config file")

    def test_every_tier_has_a_difficulty_and_they_climb(self):
        from itertools import pairwise

        from adder.decide.route.classify import TIER_DIFFICULTY, Tier

        assert set(TIER_DIFFICULTY) == set(Tier)
        vals = [TIER_DIFFICULTY[t] for t in sorted(Tier)]
        assert all(a < b for a, b in pairwise(vals))

    def test_policy_and_classify_share_one_table(self):
        from adder.decide.route import classify as c
        from adder.decide.route import policy as p

        assert p.TIER_DIFFICULTY is c.TIER_DIFFICULTY
