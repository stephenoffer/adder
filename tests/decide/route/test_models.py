"""The `adder models` surface: browsing, drift detection, and the refresh command.

The interesting assertion in this file is `TestLadder`. adder dispatches
to three model ids written into `classify.LADDER` as constants. Constants do
not know that a model shipped last week. `adder models ladder` is the thing that
notices, so its drift detection has to keep working even when -- especially
when -- the catalog disagrees with the code.
"""

from __future__ import annotations

import json

import pytest

from adder.decide.route.models import main
from adder.pricing.catalog import Catalog, Entry


def _cat(tmp_path, monkeypatch, entries, *, refreshed="2026-08-14T00:00:00+00:00"):
    """Pin the catalog to exactly these entries.

    `ADDER_CATALOG` replaces the whole layer stack, so a test asserts on
    what it wrote rather than on whatever snapshot happens to be bundled.
    """
    path = tmp_path / "pinned.json"
    Catalog(entries, provenance={"refreshed_at": refreshed}).save(path)
    monkeypatch.setenv("ADDER_CATALOG", str(path))
    monkeypatch.setenv("ADDER_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)


CHEAP = Entry(key="claude-haiku-4.5", id="anthropic/claude-haiku-4.5", org="Anthropic",
              license="Proprietary", inp=1.0, out=5.0, context=200_000,
              params=("tools",), elo={"webdev": 1300.0})
NEWER = Entry(key="claude-haiku-9", id="anthropic/claude-haiku-9", org="Anthropic",
              license="Proprietary", inp=1.0, out=5.0, context=500_000,
              params=("tools",), elo={"webdev": 1555.0})
OSS = Entry(key="qwen4", id="qwen/qwen4-max", org="Alibaba", license="Apache 2.0",
            inp=1.2, out=6.0, context=262_144, params=("tools", "reasoning"),
            elo={"webdev": 1600.0}, intelligence=60.0, coding=75.0,
            released="2026-05-01")


class TestList:
    def test_default_view_ranks_by_rating(self, tmp_path, monkeypatch, capsys):
        _cat(tmp_path, monkeypatch, [CHEAP, OSS])
        assert main([]) == 0
        out = capsys.readouterr().out
        # The first-party layer supplies the canonical Anthropic id.
        assert out.index("qwen4-max") < out.index("claude-haiku-4-5")

    def test_open_weights_filter(self, tmp_path, monkeypatch, capsys):
        _cat(tmp_path, monkeypatch, [CHEAP, OSS])
        assert main(["list", "--open-weights"]) == 0
        out = capsys.readouterr().out
        assert "qwen4-max" in out and "claude-haiku-4-5" not in out

    def test_json_is_machine_readable(self, tmp_path, monkeypatch, capsys):
        _cat(tmp_path, monkeypatch, [OSS])
        assert main(["list", "--json"]) == 0
        rows = json.loads(capsys.readouterr().out)
        assert rows and rows[0]["id"] == "qwen/qwen4-max"

    def test_a_stale_catalog_says_so(self, tmp_path, monkeypatch, capsys):
        _cat(tmp_path, monkeypatch, [OSS], refreshed="2020-01-01T00:00:00+00:00")
        main([])
        assert "stale" in capsys.readouterr().out

    def test_the_generated_claude_layer_does_not_hide_staleness(self, tmp_path,
                                                                monkeypatch):
        """first_party() is rebuilt on every load; counting it as fresh data
        would make an abandoned catalog claim to be current forever."""
        from adder.pricing.catalog import load

        _cat(tmp_path, monkeypatch, [OSS], refreshed="2020-01-01T00:00:00+00:00")
        assert load().is_stale()


class TestShow:
    def test_reports_provenance_and_verification(self, tmp_path, monkeypatch, capsys):
        _cat(tmp_path, monkeypatch, [OSS])
        assert main(["show", "qwen4"]) == 0
        out = capsys.readouterr().out
        assert "Apache 2.0" in out and "unverified" in out

    def test_first_party_claude_rates_are_marked_verified(self, tmp_path,
                                                          monkeypatch, capsys):
        _cat(tmp_path, monkeypatch, [OSS])
        assert main(["show", "claude-opus-5"]) == 0
        assert "(verified)" in capsys.readouterr().out

    def test_an_unknown_name_fails_loudly(self, tmp_path, monkeypatch, capsys):
        _cat(tmp_path, monkeypatch, [OSS])
        assert main(["show", "not-a-model"]) == 1
        assert "no model matching" in capsys.readouterr().out


class TestShowSurfacesUncertainty:
    """A rating printed without its interval invites a claim the source declines."""

    def test_the_interval_is_printed_beside_the_rating(self, tmp_path, monkeypatch,
                                                       capsys):
        from adder.pricing.catalog import Entry

        rated = Entry(key="peer", id="v/peer", org="V", license="Apache 2.0",
                      inp=1.0, out=5.0, context=1_000_000, params=("tools",),
                      elo={"webdev": 1674.0}, elo_lo={"webdev": 1663.0},
                      elo_hi={"webdev": 1686.0}, votes=11_969)
        _cat(tmp_path, monkeypatch, [rated])
        assert main(["show", "peer"]) == 0
        out = capsys.readouterr().out
        assert "1,674" in out and "95% [1,663, 1,686]" in out

    def test_a_max_effort_rating_says_the_price_is_the_default(self, tmp_path,
                                                               monkeypatch, capsys):
        from adder.pricing.catalog import Entry

        e = Entry(key="peer", id="v/peer", org="V", inp=1.0, out=5.0,
                  context=1_000_000, params=("tools",), elo={"webdev": 1674.0},
                  rating_variant="peer-max")
        _cat(tmp_path, monkeypatch, [e])
        main(["show", "peer"])
        out = capsys.readouterr().out
        assert "rated as   peer-max" in out and "model's default" in out


class TestLadder:
    def test_a_matching_ladder_reports_no_drift(self, tmp_path, monkeypatch, capsys):
        _cat(tmp_path, monkeypatch, [])          # first-party layer supplies the rungs
        assert main(["ladder"]) == 0
        assert "Ladder matches the catalog" in capsys.readouterr().out

    def test_a_better_model_in_the_band_shows_up_as_drift(self, tmp_path,
                                                          monkeypatch, capsys):
        """The whole point: a launch the constants have not caught up with."""
        _cat(tmp_path, monkeypatch, [NEWER])
        assert main(["ladder"]) == 0
        out = capsys.readouterr().out
        assert "claude-haiku-9" in out and "differ from the constants" in out

    def test_drift_is_reported_not_applied(self, tmp_path, monkeypatch, capsys):
        """The catalog reports; it never silently repoints dispatch."""
        from adder.decide.route.classify import LADDER, Tier

        _cat(tmp_path, monkeypatch, [NEWER])
        main(["ladder"])
        assert Tier.T0.model == LADDER["T0"] == "claude-haiku-4-5"

    def test_claude_code_harness_excludes_other_vendors_from_the_rungs(
            self, tmp_path, monkeypatch, capsys):
        _cat(tmp_path, monkeypatch, [OSS])
        main(["ladder"])
        assert "qwen" not in capsys.readouterr().out

    def test_json_view_flags_each_rung(self, tmp_path, monkeypatch, capsys):
        _cat(tmp_path, monkeypatch, [NEWER])
        assert main(["ladder", "--json"]) == 0
        rows = {r["rung"]: r for r in json.loads(capsys.readouterr().out)}
        assert rows["T0"]["drift"] and rows["T0"]["suggested"] == "anthropic/claude-haiku-9"


class TestRefresh:
    def test_replays_captures_without_a_socket(self, tmp_path, monkeypatch, capsys, arena_page, openrouter_page):
        monkeypatch.setenv("ADDER_OFFLINE", "1")
        (tmp_path / "a.html").write_text(arena_page)
        (tmp_path / "o.json").write_text(openrouter_page)
        out = tmp_path / "written.json"
        rc = main(["refresh", "--from", f"lmarena={tmp_path / 'a.html'}",
                   "--from", f"openrouter={tmp_path / 'o.json'}", "--out", str(out)])
        assert rc == 0 and out.is_file()
        assert "unverified" in capsys.readouterr().out

    def test_every_source_failing_leaves_the_catalog_alone(self, tmp_path,
                                                           monkeypatch, capsys):
        monkeypatch.setenv("ADDER_OFFLINE", "1")
        out = tmp_path / "untouched.json"
        assert main(["refresh", "--out", str(out)]) == 1
        assert not out.exists()
        assert "left unchanged" in capsys.readouterr().out

    def test_if_stale_returns_before_opening_a_socket(self, tmp_path, monkeypatch,
                                                      capsys):
        """Safe to put on a timer: a fresh catalog costs nothing to check."""
        monkeypatch.setenv("ADDER_OFFLINE", "1")   # a fetch here would raise
        _cat(tmp_path, monkeypatch, [OSS])
        assert main(["refresh", "--if-stale"]) == 0
        assert "nothing to do" in capsys.readouterr().out

    def test_if_stale_does_refresh_once_the_catalog_ages_out(self, tmp_path,
                                                             monkeypatch, capsys, openrouter_page):
        monkeypatch.setenv("ADDER_OFFLINE", "1")
        _cat(tmp_path, monkeypatch, [OSS], refreshed="2020-01-01T00:00:00+00:00")
        (tmp_path / "o.json").write_text(openrouter_page)
        out = tmp_path / "fresh.json"
        rc = main(["refresh", "--if-stale", "--from",
                   f"openrouter={tmp_path / 'o.json'}", "--out", str(out)])
        assert rc == 0 and out.is_file()
        assert "refreshing" in capsys.readouterr().out

    def test_a_malformed_from_spec_is_rejected(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("ADDER_OFFLINE", "1")
        assert main(["refresh", "--from", "lmarena"]) == 2

    def test_refresh_is_the_only_networked_path(self):
        """`adder models list` must not be able to reach a socket, even indirectly."""
        import ast
        import pathlib

        tree = ast.parse(pathlib.Path("adder/decide/route/models.py").read_text(encoding="utf-8"))
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = [getattr(n, "module", "") or "" for n in top]
        assert "adder.pricing.sources" not in names and "sources" not in names


@pytest.mark.parametrize("argv", [[], ["list"], ["ladder"], ["show", "claude-opus-5"]])
def test_offline_commands_never_fetch(argv, tmp_path, monkeypatch):
    """With the network switched off, every non-refresh view still works."""
    monkeypatch.setenv("ADDER_OFFLINE", "1")
    monkeypatch.setenv("ADDER_HOME", str(tmp_path / "empty"))
    monkeypatch.chdir(tmp_path)
    assert main(argv) == 0
