"""The dispatch ladder a vendor-pinned harness can actually reach.

On Codex or Gemini CLI the shipped Claude ladder names models that cannot be
dispatched, and the fix used to be a setting written by hand with model ids
looked up elsewhere. `vendor_ladder` derives it. These tests check the
properties that make it a ladder, not which ids it picks, because the ids move
every time the catalog is refreshed.
"""

from __future__ import annotations

import pytest

from adder.core import harness, settings
from adder.pricing.registry import provider_for, rate


def _price(model: str) -> float:
    return rate(model).inp


@pytest.mark.parametrize("name,org", [("codex", "openai"), ("gemini-cli", "google")])
class TestVendorLadder:
    def test_every_rung_is_the_pinned_vendor(self, name, org):
        got = harness.vendor_ladder(harness.get(name), "")
        assert got is not None and set(got) == {"T0", "T1", "T2", "T3"}
        assert {provider_for(m).name for m in got.values()} == {org}

    def test_climbing_never_gets_cheaper(self, name, org):
        got = harness.vendor_ladder(harness.get(name), "")
        prices = [_price(got[r]) for r in ("T0", "T1", "T2", "T3")]
        assert prices == sorted(prices)


class TestAnchor:
    def test_the_top_rung_is_the_model_you_run(self):
        got = harness.vendor_ladder(harness.get("codex"), "gpt-5.3-codex")
        assert got["T2"] == got["T3"] == "gpt-5.3-codex"
        assert _price(got["T0"]) < _price("gpt-5.3-codex")

    def test_an_anchor_from_another_vendor_is_ignored(self):
        got = harness.vendor_ladder(harness.get("codex"), "claude-opus-5")
        assert provider_for(got["T2"]).name == "openai"


class TestWhereItApplies:
    def test_claude_code_keeps_the_shipped_ladder(self):
        assert harness.vendor_ladder(harness.get("claude-code"), "") is None

    def test_a_harness_that_pins_nothing_is_left_alone(self):
        assert harness.vendor_ladder(harness.get("opencode"), "") is None

    def test_classify_uses_it_on_codex(self, isolated_home, monkeypatch):
        from adder.decide.route.classify import ladder, ladder_mismatch

        monkeypatch.setenv("ADDER_HARNESS", "codex")
        assert {provider_for(m).name for m in ladder().values()} == {"openai"}
        assert ladder_mismatch() == []

    def test_a_partial_override_keeps_vendor_rungs(self, isolated_home, monkeypatch):
        from adder.decide.route.classify import ladder

        monkeypatch.setenv("ADDER_HARNESS", "codex")
        monkeypatch.setenv("ADDER_LADDER", "T0=gpt-5-nano")
        got = ladder()
        assert got["T0"] == "gpt-5-nano"
        assert provider_for(got["T2"]).name == "openai"

    def test_the_subagent_estimate_is_priced_on_the_vendor(self, isolated_home,
                                                           monkeypatch):
        monkeypatch.setenv("ADDER_HARNESS", "codex")
        assert provider_for(settings.sub_model()).name == "openai"
