"""A subagent opening its own context is not a cache miss on the main chain.

`adder cache` splits rebuild waste into causes, and `model switch` is one of
the two it calls RECOVERABLE -- with the stated fix *"delegate to a subagent
instead of switching the main loop"*. Walking the combined turn list made the
main-chain/sidechain boundary look like exactly that: a different model, at a
different context size, writing its whole prefix. So the report charged the
reader for having taken its own advice.

`Session.cache_misses` and `carry.measured_read_mult` both walk the chains
separately for the same reason; this pins the third.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from adder.core.trace import Session, Turn
from adder.measure.window.cache import EXPIRY_FIXABLE, analyse

START = datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc)


def _turn(i, model, *, side=False, read=0, write=0, minutes=None):
    when = START + timedelta(minutes=i if minutes is None else minutes)
    return Turn("s", "p", model, 0, read, write, 100, 0, side,
                ts=when.isoformat(), msg_id=f"m{i}")


def _session_with_a_subagent():
    s = Session("s", "p")
    s.turns = [_turn(0, "claude-opus-5", write=200_000)]
    s.turns += [_turn(i, "claude-opus-5", read=200_000) for i in range(1, 5)]
    # The delegated run: its own model, its own cold context.
    s.turns += [_turn(5, "claude-haiku-4-5", side=True, write=60_000)]
    s.turns += [_turn(6, "claude-haiku-4-5", side=True, read=60_000)]
    s.turns += [_turn(7, "claude-opus-5", read=200_000)]
    return {"s": s}


def test_a_subagent_opening_is_not_a_model_switch():
    rep = analyse(_session_with_a_subagent())
    assert "model switch" not in rep.by_cause()


def test_no_recoverable_waste_is_invented_by_delegating():
    assert analyse(_session_with_a_subagent()).recoverable == 0.0


def test_spend_still_counts_every_turn():
    rep = analyse(_session_with_a_subagent())
    assert rep.n_turns == 8
    assert rep.write_tokens == 260_000


def test_a_real_mid_chain_model_switch_is_still_caught():
    s = Session("s", "p")
    s.turns = [_turn(0, "claude-opus-5", write=200_000)]
    s.turns += [_turn(i, "claude-opus-5", read=200_000) for i in range(1, 5)]
    # Same chain, different model: the prefix really is rebuilt.
    s.turns += [_turn(5, "claude-sonnet-5", write=200_000)]
    assert "model switch" in analyse({"s": s}).by_cause()


def test_an_idle_gap_past_the_ttl_is_still_caught():
    s = Session("s", "p")
    s.turns = [_turn(0, "claude-opus-5", write=200_000)]
    s.turns += [_turn(1, "claude-opus-5", read=200_000)]
    # 20 minutes later: past 5m, inside 1h.
    s.turns += [_turn(2, "claude-opus-5", write=200_000, minutes=21)]
    assert EXPIRY_FIXABLE in analyse({"s": s}).by_cause()


def test_mixed_naive_and_aware_stamps_do_not_raise():
    s = Session("s", "p")
    s.turns = [_turn(0, "claude-opus-5", write=200_000)]
    naive = _turn(1, "claude-opus-5", write=200_000)
    naive.ts = (START + timedelta(minutes=30)).replace(tzinfo=None).isoformat()
    s.turns.append(naive)
    assert analyse({"s": s}).n_turns == 2


# ---------------------------------------------------------------------------
# Rebuilds on providers that cache automatically. OpenAI and Google report no
# cache write: the prefix that missed arrives as plain uncached input, and that
# is where the cache is rebuilt. Reading only `cache_write` made a Codex history
# report a 100% hit rate and "$0.00 spent rebuilding prefixes" on sessions that
# had re-sent their prefix at full price.
# ---------------------------------------------------------------------------

GPT = "gpt-5.3-codex"


def _session(make_turn, rows, model=GPT):
    s = Session("s", "p")
    for i, (uncached, read) in enumerate(rows):
        s.turns.append(make_turn(model=model, uncached=uncached, read=read,
                                 write=0, minutes=i, ttl="auto"))
    return s


class TestAutomaticCaching:
    def test_uncached_input_counts_against_the_hit_rate(self, make_turn):
        s = _session(make_turn, [(40_000, 0), (1_000, 40_000), (1_000, 41_000)])
        rep = analyse({"s": s})
        assert rep.missed_tokens == 42_000
        assert 0.6 < rep.hit_rate < 0.7

    def test_a_rebuilt_prefix_is_a_miss(self, make_turn):
        s = _session(make_turn, [(40_000, 0), (1_000, 40_000), (45_000, 0)])
        rep = analyse({"s": s})
        assert len(rep.misses) == 1
        assert rep.misses[0].tokens == 45_000
        assert rep.waste > 0

    def test_the_turn_rebuilt_property_follows_the_provider(self, make_turn):
        gpt = make_turn(model=GPT, uncached=30_000, read=0, write=0)
        claude = make_turn(model="claude-opus-5", uncached=30_000, read=0, write=0)
        assert gpt.rebuilt == 30_000
        assert claude.rebuilt == 0         # explicit caching: no write, no rebuild

    def test_session_cache_misses_sees_them_too(self, make_turn):
        s = _session(make_turn, [(40_000, 0), (1_000, 40_000), (45_000, 0)])
        assert len(s.cache_misses()) == 1


class TestExplicitCachingIsUnchanged:
    def test_claude_uncached_input_is_not_a_miss(self, make_turn):
        s = Session("s", "p")
        for i in range(3):
            s.turns.append(make_turn(model="claude-opus-5", uncached=20_000,
                                     read=40_000, write=0, minutes=i))
        rep = analyse({"s": s})
        assert rep.missed_tokens == 0 and rep.hit_rate == 1.0 and not rep.misses
