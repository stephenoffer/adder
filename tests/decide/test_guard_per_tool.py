"""One floor and one ceiling, serving tools that are nothing alike.

`guard_min_tokens` is an I/O gate rather than a judgement: below it the guard
returns before parsing anything, so a call below it is invisible to every rule
that needs a price. `guard_max_fires` is an interruption budget. Both were
single numbers shared by every tool the guard watches, and the tools differ by
an order of magnitude in both directions.

Measured on the machine this was written for: `Bash` returns a p90 of 1.2K
tokens over 2,490 calls in a session, `Read` 5.9K over 58. Against a shared
2,000-token floor the first is almost never priced and the second usually is,
and against a shared 15-fire ceiling the first can spend the entire budget
before the second has said a word.

Nothing is shipped per-tool. A table of numbers would be one machine's workload
asserted as everyone's -- the mistake the size prior already made here once --
so the defaults are empty, the behaviour with them empty is exactly what it was,
and `adder guard --floors` derives the numbers from the reader's transcripts.
"""

from __future__ import annotations

import json

import pytest

from adder.core.shapes import SizeModel
from adder.decide.guard import (
    GUARDED,
    GuardState,
    Settings,
    concurrent_sessions,
    decide,
    floors_report,
    needs_pricing,
    observe,
    save_state,
)

OPUS = "claude-opus-5"


@pytest.fixture(autouse=True)
def _no_local_config(isolated_home):
    """`Settings.resolve` reads `~/.claude/adder.json`, and these tests are
    about what the resolved defaults are."""
    return isolated_home


@pytest.fixture
def sizes():
    return SizeModel(
        shapes={"cat": (200, 40_000, 40)}, heads={"cat": (200, 40_000, 40)},
        tools={"Bash": (100, 1_200, 2_490), "Read": (900, 5_900, 58)},
        built=1.0, calls=2_548,
    )


class TestTheDefaultIsUnchanged:
    """The whole mechanism has to be invisible until somebody sets something."""

    def test_no_overrides_means_the_global_floor_for_every_tool(self):
        cfg = Settings()
        assert all(cfg.min_tokens_for(t) == cfg.min_tokens for t in GUARDED)

    def test_no_overrides_means_the_global_ceiling_for_every_tool(self):
        cfg = Settings()
        assert all(cfg.max_fires_for(t) == cfg.max_fires for t in GUARDED)

    def test_an_empty_setting_parses_to_no_overrides(self):
        cfg = Settings.resolve(env={"ADDER_GUARD_MIN_TOKENS_BY_TOOL": ""})
        assert cfg.min_tokens_by_tool == {}


class TestParsing:
    def test_a_per_tool_floor_is_read(self, monkeypatch):
        monkeypatch.setenv("ADDER_GUARD_MIN_TOKENS_BY_TOOL", "Bash=400,Read=6000")
        cfg = Settings.resolve()
        assert cfg.min_tokens_for("Bash") == 400
        assert cfg.min_tokens_for("Read") == 6_000
        assert cfg.min_tokens_for("Grep") == cfg.min_tokens

    def test_a_tool_the_guard_does_not_watch_is_ignored(self, monkeypatch):
        monkeypatch.setenv("ADDER_GUARD_MIN_TOKENS_BY_TOOL", "TodoWrite=1")
        assert Settings.resolve().min_tokens_by_tool == {}

    def test_a_typo_is_dropped_rather_than_raising(self, monkeypatch):
        """A guard that raises is a guard that has stopped guarding, and the
        symptom is silence. Same rule `ladder()` follows for the same reason."""
        monkeypatch.setenv("ADDER_GUARD_MIN_TOKENS_BY_TOOL", "Bash=lots,Read=6000")
        cfg = Settings.resolve()
        assert cfg.min_tokens_for("Bash") == cfg.min_tokens
        assert cfg.min_tokens_for("Read") == 6_000

    def test_a_per_tool_ceiling_may_not_exceed_the_global_one(self, monkeypatch):
        """It exists to stop one tool starving the others, not to raise the
        total. A per-tool number that could exceed the global would do the
        second while claiming to do the first."""
        monkeypatch.setenv("ADDER_GUARD_MAX_FIRES_BY_TOOL", "Bash=500")
        cfg = Settings.resolve()
        assert cfg.max_fires_for("Bash") == cfg.max_fires


class TestTheFloorActuallyBinds:
    def test_a_lowered_floor_makes_a_small_bash_call_worth_pricing(self, sizes):
        # A shape the model has never seen, so the prediction falls back to
        # the tool-level distribution -- which is the one the floor is wrong
        # about.
        small = {"command": "jq .name one.json"}
        default = Settings()
        lowered = Settings(min_tokens_by_tool={"Bash": 100})
        assert not needs_pricing("Bash", small, sizes=sizes, cfg=default)
        assert needs_pricing("Bash", small, sizes=sizes, cfg=lowered)

    def test_a_raised_floor_silences_a_read_the_global_one_would_price(self,
                                                                      sizes, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x" * 40_000)
        cfg = Settings(min_tokens_by_tool={"Read": 10_000_000})
        v = decide("Read", {"file_path": str(f)}, model=OPUS, remaining_turns=300,
                   sizes=sizes, state=GuardState(), cfg=cfg)
        assert not v.fire and "floor for Read" in v.reason

    def test_the_bare_min_tokens_argument_still_works(self, sizes):
        """Kept for the callers that genuinely mean one number -- the sweeps in
        `auto.tune` among them."""
        assert needs_pricing("Bash", {"command": "jq .name one.json"}, sizes=sizes,
                             min_tokens=100)


class TestTheCeilingActuallyBinds:
    def test_one_tool_cannot_spend_the_whole_budget(self, sizes, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x" * 40_000)
        cfg = Settings(max_fires=15, max_fires_by_tool={"Read": 2})
        state = GuardState()
        for _ in range(4):
            v = decide("Read", {"file_path": str(f)}, model=OPUS,
                       remaining_turns=300, sizes=sizes, state=state, cfg=cfg)
            observe("Read", {"file_path": str(f)}, state, v)
        assert state.fires_by_tool["Read"] == 2
        assert state.fires == 2, "the global budget must have 13 left for others"

    def test_the_global_ceiling_still_applies(self, sizes, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x" * 40_000)
        cfg = Settings(max_fires=1)
        state = GuardState()
        for _ in range(3):
            v = decide("Read", {"file_path": str(f)}, model=OPUS,
                       remaining_turns=300, sizes=sizes, state=state, cfg=cfg)
            observe("Read", {"file_path": str(f)}, state, v)
        assert state.fires == 1

    def test_the_count_survives_a_round_trip_through_the_state_file(self):
        state = GuardState(fires_by_tool={"Bash": 3})
        assert GuardState.from_json(state.to_json()).fires_by_tool == {"Bash": 3}


class TestTheFloorsReport:
    def test_it_names_only_tools_the_guard_can_act_on(self, sizes):
        sizes.tools["TodoWrite"] = (5, 5, 900)
        text = "\n".join(floors_report(sizes))
        assert "Bash" in text and "TodoWrite" not in text

    def test_it_suggests_each_tools_own_p90(self, sizes):
        """Derived, not chosen: a floor at p90 prices the top decile of that
        tool's calls by the definition of p90."""
        text = "\n".join(floors_report(sizes))
        assert "Bash=1200" in text and "Read=5900" in text

    def test_a_tool_with_too_few_calls_is_not_evidence(self, sizes):
        sizes.tools["Grep"] = (10, 20, 2)
        assert "Grep" not in "\n".join(floors_report(sizes))

    def test_no_learned_model_says_so_rather_than_inventing_one(self):
        empty = SizeModel(shapes={}, heads={}, tools={}, built=0.0, calls=0)
        assert "--learn" in "\n".join(floors_report(empty))


class TestConcurrency:
    """The duplicate rule keys on mtime, and in a tree several agents share the
    mtime moves for reasons this session had no part in. It fails towards
    saying nothing, which is why nothing would ever have surfaced it."""

    def test_one_session_is_not_concurrency(self, tmp_path):
        path = tmp_path / "state.json"
        save_state("solo", GuardState(), path)
        assert concurrent_sessions(path) == 1

    def test_several_recent_sessions_are_counted(self, tmp_path):
        path = tmp_path / "state.json"
        for i in range(3):
            save_state(f"s{i}", GuardState(), path)
        assert concurrent_sessions(path) == 3

    def test_a_session_from_last_week_is_not_concurrent(self, tmp_path):
        path = tmp_path / "state.json"
        save_state("now", GuardState(), path)
        blob = json.loads(path.read_text())
        blob["ancient"] = {"touched": 1.0}
        path.write_text(json.dumps(blob))
        assert concurrent_sessions(path) == 1

    def test_a_missing_file_is_zero_rather_than_fatal(self, tmp_path):
        assert concurrent_sessions(tmp_path / "nope.json") == 0
