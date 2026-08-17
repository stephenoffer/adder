"""An unpriced model has no cache rates, and zero is not a stand-in for them.

`Entry` keeps prices Optional so that "nobody published a price" and "this model
is free" stay apart, and this module's docstring is explicit about the stakes:
"a missing cache read rate is the difference between a real saving and a
fantasy". `_cache_rates` answered a missing input price with `(0.0, 0.0)`.

The candidates are filtered `priced_only`, so the exposure was the *session*
model -- the one every delegated summary is carried at, and therefore the one
that sets the dominant term. With an unpriced session model a delegated cost
came out at $0.05 against a true $8.07: a 175x understatement, in the direction
that recommends delegating.
"""
from __future__ import annotations

import pytest

from adder.decide.route.select import (
    Need,
    UnpricedEntryError,
    _cache_rates,
    cost_of,
    rank,
)
from adder.pricing.catalog import Entry

CANDIDATE = Entry(key="c", id="c", org="X", inp=1.0, out=5.0,
                  context=1_000_000, params=("tools",))
PRICED_SESSION = Entry(key="s", id="s", inp=5.0, out=25.0, context=1_000_000)
UNPRICED = Entry(key="mystery", id="mystery", context=1_000_000)
NEED = Need(context_tokens=500_000, remaining_turns=400, est_read_tokens=40_000)


class TestCacheRatesRefuseToInventAPrice:
    def test_an_unpriced_entry_raises(self):
        with pytest.raises(UnpricedEntryError):
            _cache_rates(UNPRICED)

    def test_the_message_names_the_model(self):
        with pytest.raises(UnpricedEntryError, match="mystery"):
            _cache_rates(UNPRICED)

    def test_a_priced_entry_still_returns_rates(self):
        read, write, assumed = _cache_rates(CANDIDATE)
        assert read > 0 and write > 0 and assumed


class TestAnUnpricedSessionModelIsRefused:
    def test_cost_of_raises_rather_than_carrying_for_free(self):
        with pytest.raises(UnpricedEntryError):
            cost_of(CANDIDATE, NEED, session=UNPRICED)

    def test_a_priced_session_still_prices(self):
        got = cost_of(CANDIDATE, NEED, session=PRICED_SESSION)
        assert got.delegated > 0

    def test_the_carry_term_is_the_dominant_one(self):
        """Guards the premise: if the summary carry were small, none of this
        would matter."""
        got = cost_of(CANDIDATE, NEED, session=PRICED_SESSION)
        sub_only = (NEED.est_read_tokens * CANDIDATE.inp
                    + NEED.est_out_tokens * CANDIDATE.out) / 1_000_000
        assert got.delegated > 10 * sub_only


class TestTheCliReportsItAsAUserError:
    def test_an_unpriced_session_model_exits_two(self, capsys):
        from adder.decide.route import select
        assert select.main(["x", "--session-model", "openrouter/auto",
                            "--limit", "1"]) == 2

    def test_an_unknown_session_model_exits_two(self, capsys):
        from adder.decide.route import select
        assert select.main(["x", "--session-model", "not-a-model",
                            "--limit", "1"]) == 2
        assert "not in the catalog" in capsys.readouterr().err


class TestAHarnessWithNoSubagentsIsNotOfferedOne:
    """`harness.supports_subagents` was declared load-bearing and never read.

    Its own module says why it exists: "Without this the dominant lever in this
    repo -- delegation -- is not available, and a report that recommends it
    anyway is recommending a feature the user does not have." Nothing outside
    `harness.py` consulted the field, so an aider user -- one conversation, no
    subagent to hand work to -- was quoted a delegated placement and an 8x
    saving against an inline one, for a placement that does not exist.
    """

    ENTRY = Entry(key="c", id="c", org="Anthropic", inp=5.0, out=25.0,
                  context=1_000_000, params=("tools",))

    def _need(self, harness):
        return Need(context_tokens=300_000, remaining_turns=400,
                    est_read_tokens=80_000, harness=harness,
                    session_model="c", reference="c")

    def test_claude_code_can_still_delegate(self):
        c = cost_of(self.ENTRY, self._need("claude-code"), session=self.ENTRY)
        assert c.delegate_feasible
        assert c.placement == "delegate"

    def test_aider_cannot(self):
        c = cost_of(self.ENTRY, self._need("aider"), session=self.ENTRY)
        assert not c.delegate_feasible

    def test_so_it_is_priced_inline(self):
        c = cost_of(self.ENTRY, self._need("aider"), session=self.ENTRY)
        assert c.placement == "inline"
        assert c.best == c.inline

    def test_the_cheaper_placement_is_not_quoted_when_unavailable(self):
        """The delegated number is still computed; it just is not the answer."""
        c = cost_of(self.ENTRY, self._need("aider"), session=self.ENTRY)
        assert c.delegated < c.inline
        assert c.best > c.delegated

    def test_the_reason_says_why(self):
        c = cost_of(self.ENTRY, self._need("aider"), session=self.ENTRY)
        assert "no subagent" in c.delegate_blocked

    def test_a_model_with_neither_placement_is_not_a_candidate(self):
        """Aider does not pin the session, so this needs a window too small."""
        small = Entry(key="tiny", id="tiny", org="X", inp=1.0, out=2.0,
                      context=1_000, params=("tools",))
        c = cost_of(small, self._need("aider"), session=self.ENTRY)
        assert not c.usable

    def test_every_aider_pick_is_inline(self):
        picks = rank(Need(context_tokens=200_000, remaining_turns=300,
                          est_read_tokens=40_000, harness="aider"), limit=5)
        assert picks and all(p.placement == "inline" for p in picks)
