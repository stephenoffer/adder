"""Which placements the agent runtime actually makes available.

The rule this replaced was a string comparison against the literal
`"claude-code"`, with `"any"` as the only alternative. That is not a
description of the world: Codex pins the main session to OpenAI for exactly the
same structural reason Claude Code pins it to Anthropic, and written the old
way a Codex user got OpenAI models refused as main-session candidates and
Claude models offered -- precisely backwards.
"""

from __future__ import annotations

import json

import pytest

from adder.core import harness


class TestPinning:
    @pytest.mark.parametrize("name,org", [
        ("claude-code", "anthropic"),
        ("codex", "openai"),
        ("gemini-cli", "google"),
    ])
    def test_a_pinning_harness_admits_only_its_own_vendor(self, name, org):
        h = harness.get(name)
        assert h.pins_main_session
        assert h.allows_main_session(org)
        assert not h.allows_main_session("some-other-vendor")

    def test_the_pin_is_symmetric_across_harnesses(self):
        """Not "Claude Code is special". Each pins to its own vendor."""
        assert not harness.get("codex").allows_main_session("Anthropic")
        assert not harness.get("claude-code").allows_main_session("OpenAI")

    def test_a_routing_harness_admits_everything(self):
        for name in ("any", "openhands", "custom"):
            h = harness.get(name)
            assert not h.pins_main_session
            assert h.allows_main_session("anybody at all")

    def test_org_matching_is_case_and_space_insensitive(self):
        h = harness.get("claude-code")
        assert h.allows_main_session("Anthropic")
        assert h.allows_main_session("  anthropic ")

    def test_an_unknown_vendor_does_not_pass_a_pinning_gate(self):
        """A gate that passes because it could not identify the vendor is not
        a gate."""
        assert not harness.get("claude-code").allows_main_session("")


class TestOtherCapabilities:
    def test_a_harness_without_subagents_says_so(self):
        """Delegation is the dominant lever here. Recommending it to someone
        whose runtime has no subagents recommends a feature they do not have.
        """
        assert not harness.get("aider").supports_subagents
        assert harness.get("claude-code").supports_subagents

    def test_automatic_caching_harnesses_do_not_expose_breakpoints(self):
        assert not harness.get("codex").exposes_cache_control
        assert not harness.get("gemini-cli").exposes_cache_control
        assert harness.get("claude-code").exposes_cache_control


class TestResolution:
    @pytest.mark.parametrize("alias,expect", [
        ("claude", "claude-code"), ("cc", "claude-code"),
        ("codex-cli", "codex"), ("gemini", "gemini-cli"),
        ("", "any"), ("none", "any"),
    ])
    def test_aliases(self, alias, expect):
        assert harness.get(alias).name == expect

    def test_an_unrecognised_name_relaxes_rather_than_raising(self):
        """A cost report must not refuse to run over a spelling. The worst
        case is a gate that does not fire, never one that fires wrongly."""
        got = harness.get("weyland-yutani-agent")
        assert got is harness.ANY_HARNESS
        assert not got.pins_main_session

    def test_names_are_offered_for_the_cli(self):
        assert "claude-code" in harness.names()
        assert "codex" in harness.names()


class TestDefault:
    def test_it_is_claude_code_when_nothing_is_set(self, monkeypatch):
        monkeypatch.delenv("ADDER_HARNESS", raising=False)
        assert harness.default() == "claude-code"

    def test_the_environment_changes_it(self, monkeypatch):
        """So a Codex user sets it once instead of passing a flag every call."""
        monkeypatch.setenv("ADDER_HARNESS", "codex")
        assert harness.default() == "codex"

    def test_an_unknown_value_falls_back_to_any_not_to_claude_code(self,
                                                                  monkeypatch):
        """Someone who set the variable meant *not* Claude Code. Silently
        reinstating it would apply a pin they explicitly tried to leave."""
        monkeypatch.setenv("ADDER_HARNESS", "something-else")
        assert harness.default() == "any"


class TestInference:
    def test_a_single_vendor_workload_infers_that_vendor_s_harness(self):
        assert harness.infer_from_models(["gpt-5", "gpt-5-mini"]).name == "codex"
        assert harness.infer_from_models(["claude-opus-5"]).name == "claude-code"

    def test_a_mixed_workload_infers_nothing(self):
        """Two vendors in the main chain means no pin can be right."""
        assert harness.infer_from_models(["gpt-5", "claude-opus-5"]) is harness.ANY_HARNESS

    def test_an_empty_workload_infers_nothing(self):
        assert harness.infer_from_models([]) is harness.ANY_HARNESS


class TestOverrides:
    def test_a_new_harness_can_be_declared_without_a_code_change(self, tmp_path,
                                                                 monkeypatch):
        f = tmp_path / "h.json"
        f.write_text(json.dumps({"harnesses": {"acme-agent": {
            "main_session_org": "acme", "supports_subagents": False}}}))
        monkeypatch.setenv("ADDER_HARNESSES", str(f))
        got = harness.get("acme-agent")
        assert got.allows_main_session("acme")
        assert not got.supports_subagents

    def test_amending_one_field_leaves_the_rest_of_the_record_alone(self,
                                                                   tmp_path,
                                                                   monkeypatch):
        f = tmp_path / "h.json"
        f.write_text(json.dumps({"harnesses": {"claude-code": {
            "supports_subagents": False}}}))
        monkeypatch.setenv("ADDER_HARNESSES", str(f))
        got = harness.get("claude-code")
        assert not got.supports_subagents
        assert got.main_session_org == "anthropic"     # untouched

    def test_a_corrupt_file_degrades_to_the_built_ins(self, tmp_path, monkeypatch):
        f = tmp_path / "h.json"
        f.write_text("{ not json")
        monkeypatch.setenv("ADDER_HARNESSES", str(f))
        assert harness.get("claude-code").main_session_org == "anthropic"


class TestRoundTrip:
    def test_every_builtin_survives_json(self):
        for name in harness.names():
            h = harness.get(name)
            assert harness.Harness.from_json(h.to_json()) == h


class TestItGatesSelection:
    def test_select_refuses_inline_placement_for_the_wrong_vendor(self):
        """The behaviour the whole module exists for, end to end."""
        from adder.decide.route.select import Need, cost_of
        from adder.pricing.catalog import load

        cat = load()
        gpt = cat.get("gpt-5")
        assert gpt is not None
        need = Need(context_tokens=100_000, remaining_turns=100,
                    est_read_tokens=20_000, harness="claude-code")
        assert not cost_of(gpt, need).inline_feasible

        need_codex = Need(context_tokens=100_000, remaining_turns=100,
                          est_read_tokens=20_000, harness="codex")
        assert cost_of(gpt, need_codex).inline_feasible

    def test_and_the_message_names_the_harness_that_blocked_it(self):
        from adder.decide.route.select import Need, cost_of
        from adder.pricing.catalog import load

        gpt = load().get("gpt-5")
        blocked = cost_of(gpt, Need(context_tokens=1, remaining_turns=1,
                                    est_read_tokens=1,
                                    harness="claude-code")).inline_blocked
        assert "claude-code" in blocked and "anthropic" in blocked


class TestConfigurableLadder:
    """The ladder stays pinned by default; the catalog reports drift but never
    repoints dispatch. What changed is that a non-Claude user can now say so
    without editing the source."""

    def test_the_default_is_unchanged(self, monkeypatch):
        monkeypatch.delenv("ADDER_LADDER", raising=False)
        from adder.decide.route.classify import DEFAULT_LADDER, ladder
        assert ladder() == DEFAULT_LADDER

    def test_it_can_be_repointed_at_another_vendor(self, monkeypatch):
        monkeypatch.setenv("ADDER_LADDER", "T0=gpt-5-mini,T1=gpt-5,T2=gpt-5-pro")
        from adder.decide.route.classify import Tier, ladder
        assert ladder()["T0"] == "gpt-5-mini"
        assert Tier.T2.model == "gpt-5-pro"

    def test_an_unnamed_rung_keeps_its_default(self, monkeypatch):
        """A partial override must not leave a tier pointing at nothing."""
        monkeypatch.setenv("ADDER_LADDER", "T0=gpt-5-mini")
        from adder.decide.route.classify import DEFAULT_LADDER, ladder
        got = ladder()
        assert got["T0"] == "gpt-5-mini"
        assert got["T2"] == DEFAULT_LADDER["T2"]

    def test_a_typo_is_ignored_rather_than_repointing_dispatch(self, monkeypatch):
        monkeypatch.setenv("ADDER_LADDER", "T9=nonsense,garbage,T0=")
        from adder.decide.route.classify import DEFAULT_LADDER, ladder
        assert ladder() == DEFAULT_LADDER


class TestLadderSanity:
    """A ladder is only useful if climbing it costs more. Once the rungs became
    configurable, several ways to break that became reachable, and all of them
    are silent."""

    def test_the_default_ladder_is_sound(self, monkeypatch):
        monkeypatch.delenv("ADDER_LADDER", raising=False)
        from adder.decide.route.classify import ladder_warnings
        assert ladder_warnings() == []

    def test_a_partial_override_that_inverts_the_ladder_is_reported(self,
                                                                   monkeypatch):
        """The one that actually happened: repoint T0..T2 at a new vendor and
        leave T3 on the old default, so the most capable rung is the cheapest.
        """
        monkeypatch.setenv("ADDER_LADDER", "T0=gpt-5-mini,T1=gpt-5,T2=gpt-5-pro")
        from adder.decide.route.classify import ladder_warnings
        got = ladder_warnings()
        assert any("does not climb" in w for w in got)
        assert any("T3" in w and "T2" in w for w in got)

    def test_a_rung_naming_an_unknown_model_is_reported(self, monkeypatch):
        monkeypatch.setenv("ADDER_LADDER", "T0=frobnicator-7")
        from adder.decide.route.classify import ladder_warnings
        assert any("not in the catalog" in w for w in ladder_warnings())

    def test_warnings_never_raise_on_a_crooked_ladder(self, monkeypatch):
        """Dispatch still works; it just stops being an argument for anything."""
        monkeypatch.setenv("ADDER_LADDER", "T0=frobnicator-7,T2=also-not-real")
        from adder.decide.route.classify import Tier, ladder_warnings
        assert len(ladder_warnings()) >= 2
        assert Tier.T0.model == "frobnicator-7"
