"""Placement, pinned on the trade a per-token price table cannot see.

The recommendation this suite exists to prevent: moving a session with thirty
turns left onto a model whose migration costs a hundred turns of savings.
"""

from __future__ import annotations

import json

import pytest

from adder.decide.route import place as pl
from adder.pricing.catalog import Catalog, Entry

OPUS = "claude-opus-5"
HAIKU = "claude-haiku-4-5"


class TestAffinity:
    def test_a_resident_prefix_is_worth_the_discount_it_earns(self):
        v = pl.affinity_value(190_000, OPUS, 100)
        assert v > 0

    def test_it_scales_with_the_turns_that_remain(self):
        short = pl.affinity_value(190_000, OPUS, 10)
        long = pl.affinity_value(190_000, OPUS, 100)
        assert long == pytest.approx(10 * short)

    def test_a_session_about_to_end_has_nothing_to_protect(self):
        assert pl.affinity_value(190_000, OPUS, 0) == 0.0
        assert pl.affinity_value(0, OPUS, 100) == 0.0


class TestMigration:
    def test_a_move_costs_a_cold_read_and_a_write(self):
        assert pl.migration_cost(100_000, OPUS) > 0

    def test_it_scales_with_the_context(self):
        small = pl.migration_cost(10_000, OPUS)
        large = pl.migration_cost(100_000, OPUS)
        assert large == pytest.approx(10 * small)

    def test_an_empty_context_is_free_to_move(self):
        assert pl.migration_cost(0, OPUS) == 0.0

    def test_moving_to_a_cheap_model_costs_less_than_to_a_dear_one(self):
        assert pl.migration_cost(100_000, HAIKU) < pl.migration_cost(100_000, OPUS)


class TestPerTurn:
    def test_a_warm_turn_reads_the_prefix_cached(self):
        warm = pl.per_turn_cost(OPUS, 100_000, 1_000)
        cold = pl.migration_cost(100_000, OPUS)
        assert warm < cold

    def test_a_cheaper_model_costs_less_per_turn(self):
        assert (pl.per_turn_cost(HAIKU, 100_000, 1_000) <
                pl.per_turn_cost(OPUS, 100_000, 1_000))


class TestBreakeven:
    @staticmethod
    def _placement(per_turn, migration):
        e = Entry(key="x", id="x", name="x", org="Org", inp=1.0, out=5.0)
        return pl.Placement(entry=e, per_turn=per_turn, migration=migration)

    def test_breakeven_is_migration_over_the_per_turn_gain(self):
        p = self._placement(per_turn=1.0, migration=50.0)
        assert p.breakeven_turns(incumbent=2.0) == pytest.approx(50.0)

    def test_a_dearer_model_never_breaks_even(self):
        p = self._placement(per_turn=3.0, migration=10.0)
        assert p.breakeven_turns(incumbent=2.0) == float("inf")

    def test_an_equal_cost_model_never_breaks_even(self):
        p = self._placement(per_turn=2.0, migration=10.0)
        assert p.breakeven_turns(incumbent=2.0) == float("inf")

    def test_a_short_horizon_does_not_justify_a_move(self):
        """The whole point: cheaper per token is not cheaper."""
        p = self._placement(per_turn=1.0, migration=100.0)
        assert not p.worth_it(incumbent=2.0, remaining_turns=30)

    def test_a_long_horizon_does(self):
        p = self._placement(per_turn=1.0, migration=100.0)
        assert p.worth_it(incumbent=2.0, remaining_turns=500)

    def test_the_margin_keeps_a_marginal_move_from_firing(self):
        """Remaining turns is an estimate, and being wrong is asymmetric."""
        p = self._placement(per_turn=1.0, migration=100.0)
        assert not p.worth_it(incumbent=2.0, remaining_turns=110)
        assert p.worth_it(incumbent=2.0, remaining_turns=110, margin=1.0)

    def test_an_infeasible_placement_is_never_worth_it(self):
        e = Entry(key="x", id="x", name="x", org="Org", inp=0.01, out=0.01)
        p = pl.Placement(entry=e, per_turn=0.0, migration=0.0, feasible=False)
        assert not p.worth_it(incumbent=2.0, remaining_turns=10_000)


class TestEvaluate:
    def test_it_prices_the_bundled_catalog(self):
        from adder.pricing.catalog import load

        opts = pl.evaluate(load(), incumbent=OPUS, ctx_tokens=100_000,
                           remaining_turns=200)
        assert opts.considered > 0
        assert opts.places
        assert opts.incumbent_per_turn > 0

    def test_candidates_are_sorted_cheapest_first(self):
        from adder.pricing.catalog import load

        opts = pl.evaluate(load(), incumbent=OPUS, ctx_tokens=50_000,
                           remaining_turns=100)
        costs = [p.per_turn for p in opts.places]
        assert costs == sorted(costs)

    def test_a_model_that_cannot_hold_the_context_is_excluded(self):
        cat = Catalog([
            Entry(key="tiny", id="tiny", name="tiny", org="O", inp=0.01, out=0.01,
                  context=8_000),
        ])
        opts = pl.evaluate(cat, incumbent=OPUS, ctx_tokens=500_000,
                           remaining_turns=100)
        assert opts.places == []
        assert opts.infeasible == 1

    def test_the_incumbent_is_not_offered_as_a_destination(self):
        from adder.pricing.catalog import load

        opts = pl.evaluate(load(), incumbent=OPUS, ctx_tokens=50_000,
                           remaining_turns=100)
        assert all(p.id != OPUS for p in opts.places)

    def test_unpriced_entries_are_skipped(self):
        cat = Catalog([Entry(key="free", id="free", name="free", org="O")])
        opts = pl.evaluate(cat, incumbent=OPUS, ctx_tokens=1_000,
                           remaining_turns=10)
        assert opts.places == []

    def test_rated_only_narrows_the_field(self):
        from adder.pricing.catalog import load

        cat = load()
        wide = pl.evaluate(cat, incumbent=OPUS, ctx_tokens=50_000, remaining_turns=100)
        narrow = pl.evaluate(cat, incumbent=OPUS, ctx_tokens=50_000,
                             remaining_turns=100, require_rating=True)
        assert narrow.considered <= wide.considered

    def test_a_long_session_finds_somewhere_to_move(self):
        from adder.pricing.catalog import load

        opts = pl.evaluate(load(), incumbent=OPUS, ctx_tokens=50_000,
                           remaining_turns=100_000)
        assert opts.best is not None

    def test_a_session_about_to_end_stays_put_when_the_move_is_dear(self):
        """Controlled catalog: one destination, modestly cheaper per turn.

        The bundled catalog is not the right fixture for this property. It
        contains models cheap enough that even a one-turn horizon repays the
        move, which is arithmetically correct and makes the test say nothing.
        """
        cat = Catalog([
            Entry(key="mild", id="mild", name="mild", org="O",
                  inp=10.0, out=40.0, cache_read=1.0, cache_write=12.5,
                  context=1_000_000),
        ])
        opts = pl.evaluate(cat, incumbent=OPUS, ctx_tokens=190_000,
                           remaining_turns=1)
        assert opts.best is None

    def test_a_cheap_enough_destination_repays_even_a_short_horizon(self):
        """Stated explicitly so nobody 'fixes' the rule into a blanket refusal."""
        cat = Catalog([
            Entry(key="tiny", id="tiny", name="tiny", org="O",
                  inp=0.05, out=0.20, cache_read=0.005, cache_write=0.06,
                  context=1_000_000),
        ])
        opts = pl.evaluate(cat, incumbent=OPUS, ctx_tokens=190_000,
                           remaining_turns=5)
        assert opts.best is not None


class TestReport:
    @staticmethod
    def _opts(remaining=100):
        from adder.pricing.catalog import load

        return pl.evaluate(load(), incumbent=OPUS, ctx_tokens=100_000,
                           remaining_turns=remaining)

    def test_it_prices_the_prefix_being_discarded(self):
        text = pl.report(self._opts())
        assert "resident prefix worth" in text

    def test_it_recommends_staying_when_no_move_repays(self):
        cat = Catalog([
            Entry(key="mild", id="mild", name="mild", org="O",
                  inp=10.0, out=40.0, cache_read=1.0, cache_write=12.5,
                  context=1_000_000),
        ])
        opts = pl.evaluate(cat, incumbent=OPUS, ctx_tokens=190_000,
                           remaining_turns=2)
        text = pl.report(opts)
        assert "Stay" in text or "cheapest thing you own" in text

    def test_it_recommends_moving_on_a_long_one(self):
        assert "Move to" in pl.report(self._opts(remaining=100_000))

    def test_it_warns_about_providers_with_no_published_cache(self):
        cat = Catalog([
            Entry(key="nocache", id="nocache", name="nocache", org="O",
                  inp=0.05, out=0.10, context=1_000_000),
        ])
        opts = pl.evaluate(cat, incumbent=OPUS, ctx_tokens=100_000,
                           remaining_turns=500)
        assert opts.places, "an entry with a price should be considered"
        # `render.wrap` breaks lines, so compare on collapsed whitespace.
        flat = " ".join(pl.report(opts).split())
        assert "no prompt cache" in flat

    def test_it_always_defers_quality_to_the_frontier(self):
        # `render.wrap` may break the command name across lines.
        assert "frontier" in pl.report(self._opts())
        assert "Quality is not considered here" in pl.report(self._opts())

    def test_an_empty_field_says_so(self):
        opts = pl.evaluate(Catalog([]), incumbent=OPUS, ctx_tokens=1_000,
                           remaining_turns=10)
        assert "No other model" in pl.report(opts)

    def test_json_is_finite_and_complete(self):
        payload = self._opts().to_json()
        text = json.dumps(payload)
        assert "NaN" not in text and "Infinity" not in text
        assert "affinity_value_usd" in payload

    def test_an_unreachable_breakeven_serialises_as_null(self):
        from adder.pricing.catalog import load

        opts = pl.evaluate(load(), incumbent=HAIKU, ctx_tokens=50_000,
                           remaining_turns=10)
        payload = opts.to_json()
        json.loads(json.dumps(payload))
        assert any(c["breakeven_turns"] is None for c in payload["candidates"])


class TestBreakevenFormatting:
    def test_an_immediate_repayment_reads_as_less_than_one(self):
        """"0" would read as "no break-even computed", not "repays at once"."""
        assert pl._breakeven_str(0.2) == "<1"

    def test_a_real_horizon_is_rounded(self):
        assert pl._breakeven_str(42.4) == "42"

    def test_an_unreachable_horizon_says_never(self):
        assert pl._breakeven_str(float("inf")) == "never"


class TestCli:
    def test_it_runs_and_prints(self, capsys, isolated_home):
        assert pl.main(["--turns", "100"]) in (0, 1)
        assert capsys.readouterr().out.strip()

    def test_json_parses(self, capsys, isolated_home):
        pl.main(["--json"])
        assert "candidates" in json.loads(capsys.readouterr().out)

    def test_a_huge_context_narrows_the_field(self, capsys, isolated_home):
        pl.main(["--context", "5000000", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["infeasible"] > 0
