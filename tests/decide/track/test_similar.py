"""The neighbour estimator must be unbiased, private, and unable to buy a downgrade.

Three properties carry the module and each has a way of failing quietly:

* the similarity has to be a Jaccard estimate rather than something merely
  monotone in overlap, because a threshold is compared against it;
* the sketch must not be a way to read the task back out of the log;
* thin, optimistic neighbour evidence must never reach a gate that would spend
  it on a cheaper tier.
"""
from __future__ import annotations

import json

import pytest

from adder.decide.track.outcomes import HALF_LIFE_DAYS, Evidence, Outcome, record
from adder.decide.track.similar import (
    MIN_NEIGHBOURS,
    SIM_FLOOR,
    SLOTS,
    coverage,
    evidence_like,
    main,
    neighbours,
    report,
    sharpen,
    similarity,
    sketch,
    terms,
)

NOW = 1_700_000_000.0
DAY = 86_400.0


def row(task, tier="T0", escalated=False, *, age_days=0.0, project="p", cost=0.01):
    return Outcome(tier=tier, model="m", project=project, escalated=escalated,
                   cost=cost, ts=NOW - age_days * DAY, sketch=list(sketch(task)))


class TestSketch:
    def test_is_fixed_width_regardless_of_input_length(self):
        short = sketch("read the config")
        long = sketch(" ".join(f"word{i} thing{i}" for i in range(400)))
        assert len(short) == len(long) == SLOTS

    def test_empty_text_has_no_opinion(self):
        # Not a sketch of nothing, which would match other empty sketches
        # perfectly and manufacture neighbours out of missing data.
        assert sketch("") == ()
        assert sketch("a an") == ()   # every term below MIN_TERM

    def test_is_deterministic_across_calls(self):
        assert sketch("migrate the scheduler") == sketch("migrate the scheduler")

    def test_does_not_carry_the_task_text(self):
        task = "delete the customer records in billing.py"
        s = sketch(task)
        blob = json.dumps(list(s))
        for word in ("delete", "customer", "records", "billing"):
            assert word not in blob

    def test_bigrams_separate_opposite_verbs_on_shared_nouns(self):
        # "read the config" and "write the config" share every surviving
        # unigram. If the sketch were unigrams only these would be neighbours,
        # and they are opposite tiers.
        read = sketch("read the config")
        write = sketch("write the config")
        assert similarity(read, write) < 1.0

    def test_terms_drop_boilerplate_and_short_words(self):
        assert "the" not in terms("the retry logic")
        assert "retry" in terms("the retry logic")


class TestSimilarity:
    def test_identical_text_is_one(self):
        assert similarity(sketch("fix the flaky test"), sketch("fix the flaky test")) == 1.0

    def test_disjoint_vocabulary_is_near_zero(self):
        a = sketch("migrate postgres schema partitions")
        b = sketch("tune webpack bundle splitting")
        assert similarity(a, b) < 0.2

    def test_missing_sketch_is_zero_not_a_match(self):
        assert similarity((), sketch("anything")) == 0.0
        assert similarity(sketch("anything"), ()) == 0.0
        assert similarity(None, None) == 0.0

    def test_estimates_jaccard_rather_than_merely_ranking(self):
        # Half the terms shared by construction, so a Jaccard estimator should
        # land near 1/3 (|A n B| / |A u B| = 3/9 over unigrams+bigrams). The
        # sketch is 16 slots, so tolerance is wide; the point is that it is not
        # systematically inflated, which is what a bottom-k sketch would be on
        # texts this short and is how an unrelated row becomes a neighbour.
        a = sketch("alpha bravo charlie delta")
        b = sketch("alpha bravo charlie zulu")
        assert 0.15 < similarity(a, b) < 0.75

    def test_compares_over_the_common_prefix_of_unequal_sketches(self):
        full = sketch("resize the connection pool")
        assert similarity(full, full[:8]) == 1.0


class TestNeighbours:
    def test_finds_the_on_topic_rows_and_orders_them(self):
        rows = [row("update the retry backoff in the client"),
                row("tune the webpack bundle"),
                row("change the retry backoff for the client")]
        got = neighbours("adjust the retry backoff on the client", rows)
        assert got, "no neighbours found for near-identical vocabulary"
        assert got[0][0] >= got[-1][0]
        assert all(s >= SIM_FLOOR for s, _ in got)

    def test_skips_rows_written_before_sketches_existed(self):
        legacy = Outcome(tier="T0", model="m", project="p", escalated=True, ts=NOW)
        assert legacy.sketch == []
        assert neighbours("retry backoff client", [legacy]) == []

    def test_ignores_project_because_the_task_is_the_scope(self):
        # The whole reason this module exists: a refactor in another repo is
        # better evidence about refactors than a lookup in this one.
        rows = [row("refactor the session store", project="other-repo")
                for _ in range(3)]
        assert neighbours("refactor the session store", rows)

    def test_tier_filter_is_applied(self):
        rows = [row("rename the settings module", tier="T0"),
                row("rename the settings module", tier="T2")]
        assert len(neighbours("rename the settings module", rows, tier="T2")) == 1

    def test_no_sketchable_task_finds_nothing(self):
        assert neighbours("", [row("anything at all")]) == []


class TestEvidenceLike:
    def similar_rows(self, n, escalated, **kw):
        # Distinct-but-similar phrasings, so this is a set of neighbours rather
        # than one row repeated -- which would pass a count threshold without
        # being independent evidence of anything.
        verbs = ["update", "change", "adjust", "revise", "amend", "edit",
                 "patch", "tweak", "modify", "alter"]
        return [row(f"{verbs[i % len(verbs)]} the retry backoff in the client",
                    escalated=escalated, **kw) for i in range(n)]

    def test_abstains_below_the_neighbour_floor(self):
        rows = self.similar_rows(MIN_NEIGHBOURS - 1, False)
        assert evidence_like("update the retry backoff in the client", "T0",
                             rows, now=NOW) is None

    def test_abstains_when_the_log_has_no_sketches(self):
        legacy = [Outcome(tier="T0", model="m", project="p", escalated=True, ts=NOW)
                  for _ in range(50)]
        assert evidence_like("retry backoff client", "T0", legacy, now=NOW) is None

    def test_measures_the_neighbour_rate(self):
        rows = self.similar_rows(8, True) + self.similar_rows(2, False)
        ev = evidence_like("update the retry backoff in the client", "T0",
                           rows, now=NOW)
        assert ev is not None
        assert ev.scope == "neighbours"
        assert ev.p_fail > 0.5, "8 of 10 neighbours escalated and the rate is under half"
        assert ev.n >= MIN_NEIGHBOURS

    def test_smoothed_by_the_same_prior_as_the_tier_wide_rate(self):
        # All clean, so an unsmoothed estimate would say 0% and a gate would
        # read that as a guarantee.
        rows = self.similar_rows(6, False)
        ev = evidence_like("update the retry backoff in the client", "T0",
                           rows, now=NOW)
        assert ev is not None and 0.0 < ev.p_fail < 0.5

    def test_old_neighbours_weigh_less(self):
        fresh = evidence_like("update the retry backoff in the client", "T0",
                              self.similar_rows(6, True), now=NOW)
        stale = evidence_like("update the retry backoff in the client", "T0",
                              self.similar_rows(6, True, age_days=6 * HALF_LIFE_DAYS),
                              now=NOW)
        assert fresh is not None and stale is not None
        assert stale.weight < fresh.weight
        # Decayed toward the prior rather than staying at a confident 100%.
        assert stale.p_fail < fresh.p_fail

    def test_wall_clock_is_not_required(self):
        # `now=None` must not raise; determinism elsewhere is what the explicit
        # parameter is for, but the default path is the one hooks take.
        assert evidence_like("update the retry backoff in the client", "T0",
                             self.similar_rows(6, True)) is not None


class TestSharpen:
    def wide(self, p, weight, scope="global"):
        return Evidence(p, 40, weight, scope, fails=p * weight)

    def nb(self, p, weight, n=6):
        return Evidence(p, n, weight, "neighbours", fails=p * weight)

    def test_no_neighbours_changes_nothing(self):
        w = self.wide(0.2, 50.0)
        assert sharpen(w, None) is w

    def test_informative_neighbours_replace_the_tier_wide_rate(self):
        nb = self.nb(0.05, 30.0)
        assert nb.informative
        assert sharpen(self.wide(0.4, 50.0), nb) is nb

    def test_thin_and_pessimistic_raises_the_rate(self):
        # Free to act on: a raised p_fail can only decline a downgrade, and the
        # worst case of declining one is the model you would have used anyway.
        out = sharpen(self.wide(0.1, 50.0), self.nb(0.6, 2.0))
        assert out is not None and out.p_fail == pytest.approx(0.6)
        assert "neighbours" in out.scope
        assert out.weight == 50.0, "the tier-wide mass is what is known; it is kept"

    def test_thin_and_pessimistic_keeps_the_interval_around_its_own_mean(self):
        out = sharpen(self.wide(0.1, 50.0), self.nb(0.6, 2.0))
        b = out.bounds()
        assert b.lo <= out.p_fail <= b.hi, "the printed interval excludes the mean"

    def test_thin_and_optimistic_is_discarded(self):
        # The case the asymmetry exists for: four cheap-looking rows must not
        # talk the router down a rung.
        w = self.wide(0.4, 50.0)
        assert sharpen(w, self.nb(0.02, 2.0)) is w

    def test_neighbours_stand_alone_when_there_is_no_tier_wide_estimate(self):
        nb = self.nb(0.3, 3.0)
        assert sharpen(None, nb) is nb
        assert sharpen(Evidence(0.5, 0, 0.0, "prior", 0.0), nb) is nb

    def test_a_thin_neighbour_estimate_is_never_informative_on_its_own(self):
        # Standing alone is not the same as being trusted: a non-informative
        # estimate cannot clear `_may_descend`, which is the gate that spends money.
        assert not sharpen(None, self.nb(0.02, 2.0)).informative


class TestCoverage:
    def test_counts_sketched_rows(self):
        rows = [row("a retry backoff change"),
                Outcome(tier="T0", model="m", project="p", escalated=False, ts=NOW)]
        assert coverage(rows) == (1, 2)


class TestRouterIntegration:
    """The point of the module: it must reach `policy.decide` and change nothing
    it has not earned."""

    def test_pessimistic_neighbours_block_a_downgrade_the_tier_rate_allowed(self, tmp_path, monkeypatch):
        from adder.decide.route.classify import Tier
        from adder.decide.route.policy import _evidence

        log = tmp_path / "outcomes.jsonl"
        # A tier-wide record that is clean and heavy: T0 looks safe overall.
        for i in range(60):
            record(row(f"list the {i} exported symbols", escalated=False), log)
        # ...and a small cluster of failures on one kind of task.
        for i in range(6):
            record(row(f"debug the {i} race condition in the scheduler",
                       escalated=True), log)
        monkeypatch.setenv("ADDER_LOG", str(log))
        monkeypatch.setattr("adder.decide.track.outcomes.DEFAULT_LOG", log)

        wide = _evidence(Tier.T0, None)
        sharp = _evidence(Tier.T0, None, "debug the race condition in the scheduler")
        assert wide is not None and sharp is not None
        assert sharp.p_fail > wide.p_fail, (
            "the failures clustered on this kind of task did not reach the gate")

    def test_a_broken_sharpener_leaves_the_tier_rate_in_place(self, tmp_path, monkeypatch):
        from adder.decide.route import policy
        from adder.decide.route.classify import Tier


        log = tmp_path / "outcomes.jsonl"
        for i in range(30):
            record(row(f"list the {i} exported symbols"), log)
        monkeypatch.setattr("adder.decide.track.outcomes.DEFAULT_LOG", log)

        def boom(*a, **k):
            raise OSError("no log")

        monkeypatch.setattr("adder.decide.track.similar.evidence_like", boom)
        # A sharper estimate is an improvement, not a dependency: the tier-wide
        # rate must survive the neighbour half falling over.
        ev = policy._evidence(Tier.T0, None, "anything at all")
        assert ev is not None and ev.n == 30

    def test_decide_still_routes_with_no_task_similarity_available(self, tmp_path, monkeypatch):
        from adder.decide.route.policy import decide

        monkeypatch.setattr("adder.decide.track.outcomes.DEFAULT_LOG",
                            tmp_path / "empty.jsonl")
        plan = decide("find where the retry logic is configured",
                      context_tokens=20_000, remaining_turns=100)
        assert plan.tier is not None and plan.ladder


class TestCli:
    def test_reports_an_empty_log_rather_than_nothing(self, tmp_path, capsys):
        assert main(["--log", str(tmp_path / "none.jsonl"), "a", "task"]) == 0
        assert "empty" in capsys.readouterr().out

    def test_names_the_sketch_coverage(self, tmp_path):
        log = tmp_path / "outcomes.jsonl"
        for i in range(4):
            record(row(f"update the retry backoff {i} in the client"), log)
        out = report("update the retry backoff in the client", log=log, now=NOW)
        assert "4 of 4" in out

    def test_says_so_when_no_tier_has_enough_neighbours(self, tmp_path):
        log = tmp_path / "outcomes.jsonl"
        record(row("something entirely unrelated"), log)
        out = report("update the retry backoff in the client", log=log, now=NOW)
        assert "falls back to the tier-wide rate" in out

    def test_json_is_machine_readable(self, tmp_path, capsys):
        log = tmp_path / "outcomes.jsonl"
        for i in range(8):
            record(row(f"update the retry backoff {i} in the client",
                       escalated=i < 6), log)
        assert main(["--log", str(log), "--json",
                     "update the retry backoff in the client"]) == 0
        d = json.loads(capsys.readouterr().out)
        assert d["sketched_rows"] == 8
        assert d["by_tier"]["T0"]["p_fail"] > 0.5

    def test_no_task_is_an_error_not_an_empty_report(self, capsys):
        assert main([""]) == 2


class TestImportPath:
    def test_transcript_import_writes_sketches(self):
        from adder.decide.track.dispatch import Dispatch, Scan, to_outcomes

        d = Dispatch(session="s", project="p", use_id="u", agent_type="route-t0",
                     description="find the retry backoff constant", ts="",
                     model="claude-haiku-4-5", resolved=True)
        scan = Scan(dispatches=[d])
        scan.usable.append(d)
        rows = to_outcomes(scan)
        assert rows and rows[0].sketch
        assert len(rows[0].sketch) == SLOTS

    def test_a_dispatch_with_no_description_gets_no_sketch(self):
        from adder.decide.track.dispatch import Dispatch, Scan, to_outcomes

        d = Dispatch(session="s", project="p", use_id="u", agent_type="route-t0",
                     description="", ts="", model="claude-haiku-4-5", resolved=True)
        scan = Scan(dispatches=[d])
        scan.usable.append(d)
        assert to_outcomes(scan)[0].sketch == []
