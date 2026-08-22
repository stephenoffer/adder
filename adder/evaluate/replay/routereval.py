"""Score the router the way the routing literature scores routers.

The gap this closes
-------------------
Every other module here answers "what did this cost" or "what should I do".
None of them answered the question a reader is entitled to ask first: **is the
router any good?** A tool that recommends a cheaper model on 40% of turns has
made 40% of a claim, and until you can put a number on how much quality that
bought and how much it gave up, the claim is unfalsifiable.

There is an established way to measure this and no reason to invent another
one. Fix a strong model and a weak model. Let the router send some fraction of
calls to the strong one. Then:

    PGR   = (quality(router) - quality(weak)) / (quality(strong) - quality(weak))
    APGR  = the average of PGR across the whole range of call fractions
    CPT(x) = the smallest fraction of strong calls that reaches PGR = x

PGR alone is gameable -- route everything to the strong model and PGR is 1.0
with no saving -- which is why the summary number is APGR, the area under the
call-performance curve. A router that picks at random scores APGR = 0.5 by
construction; that is the number every result here is reported against, because
"our router beat sending every other request to the expensive model" is the
comparison that actually matters and the one people skip.

**APGR does not top out at 1.0.** Its ceiling is set by the task mix, not by
the router: if half the tasks genuinely need the strong model, a router with
perfect foresight still cannot reach PGR=1 until it has spent half the calls,
and its APGR is 0.75. Reading 0.75 as "75% of the way to perfect" is wrong --
it *is* perfect on that data. This is why every report here prints the oracle
ceiling next to the score and a regret against it, rather than leaving the
reader to assume a ceiling that does not exist.

Where this deviates, and why
----------------------------
The published metric puts *fraction of calls to the strong model* on the x
axis. That is right when calls cost about the same, and it is badly wrong for
agent sessions, which is the workload this tool measures. A strong call on a
190k-token context costs roughly forty times a weak call on an 8k one, so a
router that sends 30% of calls to the strong model can easily be spending 95%
of the all-strong budget. Reporting "30% of calls" as the cost of that router
is not a rounding error, it is the whole answer inverted.

So this module computes both curves and prints both:

* `calls` -- x axis is the fraction of calls routed strong. Comparable to
  published numbers, and what to quote when comparing against them.
* `cost` -- x axis is the fraction of the all-strong dollar budget actually
  spent, using the same date-aware prices as every other report here.

When those two disagree, the second one is the one that is true, and the size
of the disagreement is itself a finding worth printing.

What an episode is
------------------
One task, evaluated both ways: what it would have cost and scored on the strong
model, and on the weak one. That is a counterfactual, and counterfactuals are
where cost tools lie. Two honest sources:

* an A/B log, where both arms really were run (`adder ab`), so both sides are
  measured; and
* an outcome log, where one arm was run and the other was escalated to, so one
  side is measured and the other is a recorded fact about what happened next.

A third source -- modelling the unrun arm from arena ratings -- is supported
because it is often all there is, and it is labelled MODELLED in the output for
exactly as long as it stays a model. Anything derived from it inherits the
label. The failure mode of a cost tool is not a bad recommendation, it is a
confident bad recommendation, and a router that scores well against its own
assumptions is the purest form of that.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from adder.util import render
from adder.util.stats import DEFAULT_SEED, mean, quantile

# The published metric discretises the call-fraction axis into ten bins. Bin
# midpoints rather than upper edges, because midpoints are what make a random
# router score exactly 0.500 -- upper edges score it 0.550, and every result
# then has to be read against a baseline that is not where the reader thinks
# it is.
BINS = 10
BIN_CENTRES: tuple[float, ...] = tuple((i + 0.5) / BINS for i in range(BINS))


@dataclass(frozen=True)
class Episode:
    """One task priced and scored under both arms.

    `score` is the router's propensity to send this task to the strong model:
    any real number, higher meaning "more likely to need the strong model".
    Sweeping a threshold over it is what produces the curve, so a router only
    has to expose a ranking, not a calibrated probability.
    """

    key: str
    q_strong: float
    q_weak: float
    cost_strong: float
    cost_weak: float
    score: float
    split: str = "test"
    modelled: bool = False

    @property
    def gain(self) -> float:
        """Quality given up by routing this task to the weak model."""
        return self.q_strong - self.q_weak

    @property
    def saving(self) -> float:
        """Dollars saved by routing this task to the weak model."""
        return self.cost_strong - self.cost_weak


@dataclass(frozen=True)
class Point:
    """One threshold on the call-performance curve."""

    n_strong: int
    call_fraction: float
    cost_fraction: float
    quality: float
    cost: float
    pgr: float


@dataclass
class Report:
    """Everything `routereval` measured, in the order a reader needs it."""

    n: int
    q_strong: float
    q_weak: float
    cost_strong: float
    cost_weak: float
    curve: list[Point] = field(default_factory=list)
    apgr_calls: float = 0.0
    apgr_cost: float = 0.0
    apgr_ci: tuple[float, float] = (0.0, 0.0)
    random_apgr_ci: tuple[float, float] = (0.0, 0.0)
    oracle_apgr: float = 0.0
    cpt: dict[int, float] = field(default_factory=dict)
    cpt_cost: dict[int, float] = field(default_factory=dict)
    modelled_share: float = 0.0

    @property
    def separable(self) -> bool:
        """False when the two arms scored the same and PGR has no denominator.

        This is not an edge case to swallow. If the strong and weak models
        scored identically on this task set, there is no performance gap to
        recover, every router trivially recovers all of it, and the correct
        report is "these two models are indistinguishable here, route on price"
        -- not an APGR of 1.0 that looks like a triumph.
        """
        return abs(self.q_strong - self.q_weak) > 1e-12

    @property
    def cost_separable(self) -> bool:
        """False when the cost axis has no denominator, exactly as `separable`.

        The cost axis is normalised by `strong_total - weak_total`. When that
        is not positive every point collapses to a `cost_fraction` of 0, so
        `_pgr_at` walks to the last point and returns the all-strong PGR: an
        APGR of 1.000 and a CPT of 0% -- "recover 80% of the gap for nothing".
        Both are artifacts of a flat axis and both read as a triumph.

        This is not hypothetical on the default path. `episodes_from_outcomes`
        prices the weak arm at its RECORDED cost, which includes the session
        context it ran in, against a MODELLED cold run for the strong arm; the
        recorded number is routinely the larger of the two. `separable` already
        makes this argument for the quality axis and refuses to swallow it.
        """
        return (self.cost_strong - self.cost_weak) > 1e-12

    @property
    def beats_random(self) -> bool:
        """True only when the interval clears the random router's interval."""
        return self.apgr_ci[0] > self.random_apgr_ci[1]

    @property
    def regret(self) -> float:
        """How much of the achievable APGR the router leaves on the table."""
        return max(0.0, self.oracle_apgr - self.apgr_calls)

    @property
    def best_single(self) -> float:
        """Quality from picking one model and never routing at all.

        Usually the strong arm, and deliberately not assumed to be: on a task
        mix where the weak model wins, quoting the strong arm as the baseline
        would flatter every router scored against it.
        """
        return max(self.q_strong, self.q_weak)

    @property
    def gain_at_best(self) -> float:
        """Quality the router adds over the better of the two fixed choices.

        The published benchmarks report this because it is the comparison that
        embarrasses routers: several well-known ones fail to beat simply picking
        one model, and a metric family anchored at the weak arm cannot show it.
        PGR cannot either -- it is 1.0 at the all-strong endpoint by
        construction, and that endpoint *is* a single model.

        **Zero is the common answer and it is not a bug.** It means no threshold
        on this curve beat the better fixed choice, which is a finding rather
        than a failure of measurement. It is positive only where some mix of the
        two arms is genuinely better than either alone.
        """
        if not self.curve:
            return 0.0
        return max(0.0, max(p.quality for p in self.curve) - self.best_single)

    @property
    def cost_save(self) -> float:
        """Largest share of the all-strong budget saved at no loss of quality.

        The benchmark's headline cost figure, and the one a reader actually
        wants: not "how much of the gap did it recover" but "how much cheaper
        can this get before it starts costing me answers". Read off the curve as
        the cheapest threshold whose quality still matches the better fixed
        choice.

        0.0 means no threshold held quality, which includes the honest case
        where the router has nothing to offer. Requires a cost axis with a
        denominator, for the reason `cost_separable` gives.
        """
        if not self.curve or not self.cost_separable:
            return 0.0
        holding = [p.cost_fraction for p in self.curve
                   if p.quality >= self.best_single - 1e-12]
        return max(0.0, 1.0 - min(holding)) if holding else 0.0

    def to_json(self) -> dict:
        return {
            "n": self.n,
            "quality": {"strong": self.q_strong, "weak": self.q_weak},
            "cost": {"strong": self.cost_strong, "weak": self.cost_weak},
            "apgr": {
                "calls": self.apgr_calls,
                "cost": self.apgr_cost if self.cost_separable else None,
                "ci95": list(self.apgr_ci),
                "random_ci95": list(self.random_apgr_ci),
                "oracle": self.oracle_apgr,
                "regret": self.regret,
                "beats_random": self.beats_random,
            },
            # Named as the published benchmarks name them, so a number here can
            # be quoted next to one from a paper without a translation step.
            "vs_best_single": {
                "best_single_quality": self.best_single,
                "gain_at_best": self.gain_at_best,
                "beats_best_single": self.gain_at_best > 0.0,
                "cost_save": self.cost_save if self.cost_separable else None,
            },
            "cpt": {f"{k}%": v for k, v in sorted(self.cpt.items())},
            "cpt_cost": ({f"{k}%": v for k, v in sorted(self.cpt_cost.items())}
                         if self.cost_separable else None),
            "separable": self.separable,
            "cost_separable": self.cost_separable,
            "modelled_share": self.modelled_share,
            "curve": [
                {
                    "n_strong": p.n_strong,
                    "call_fraction": p.call_fraction,
                    "cost_fraction": p.cost_fraction,
                    "quality": p.quality,
                    "cost": p.cost,
                    "pgr": p.pgr,
                }
                for p in self.curve
            ],
        }


def _order(episodes: Sequence[Episode]) -> list[Episode]:
    """Highest router score first, ties broken by key so the sweep is stable.

    Without the tiebreak, two runs over the same data can produce different
    curves whenever scores collide -- which they do constantly, because a
    classifier that emits five discrete difficulty levels has ties everywhere.
    """
    return sorted(episodes, key=lambda e: (-e.score, e.key))


def curve(episodes: Sequence[Episode]) -> list[Point]:
    """The full call-performance curve: route the top-k scored tasks strong.

    Every k from 0 (all weak) to n (all strong), so the curve needs no
    interpolation to answer a threshold question and the endpoints are exact.
    """
    ordered = _order(episodes)
    n = len(ordered)
    if n == 0:
        return []

    q_weak_total = sum(e.q_weak for e in ordered)
    q_strong_total = sum(e.q_strong for e in ordered)
    c_weak_total = sum(e.cost_weak for e in ordered)
    c_strong_total = sum(e.cost_strong for e in ordered)
    gap = (q_strong_total - q_weak_total) / n

    out: list[Point] = []
    quality_sum = q_weak_total
    cost_sum = c_weak_total
    denom_cost = c_strong_total - c_weak_total
    for k in range(n + 1):
        if k > 0:
            e = ordered[k - 1]
            quality_sum += e.gain
            cost_sum += e.saving
        q = quality_sum / n
        pgr = (q - q_weak_total / n) / gap if abs(gap) > 1e-12 else 1.0
        cost_fraction = ((cost_sum - c_weak_total) / denom_cost) if denom_cost > 0 else 0.0
        out.append(Point(k, k / n, cost_fraction, q, cost_sum, pgr))
    return out


def _pgr_at(points: Sequence[Point], fraction: float, axis: str) -> float:
    """PGR at a given budget, interpolated between the two bracketing points."""
    if not points:
        return 0.0
    key = (lambda p: p.call_fraction) if axis == "calls" else (lambda p: p.cost_fraction)
    lo = points[0]
    for p in points:
        if key(p) <= fraction:
            lo = p
        else:
            span = key(p) - key(lo)
            if span <= 0:
                return p.pgr
            w = (fraction - key(lo)) / span
            return lo.pgr * (1 - w) + p.pgr * w
    return lo.pgr


def apgr(points: Sequence[Point], axis: str = "calls") -> float:
    """Average performance gap recovered: the summary number.

    The mean of PGR over the ten bin centres of the budget axis. A router that
    orders tasks at random scores 0.5; one that always picks correctly scores
    close to 1; one that is actively anti-correlated scores below 0.5, and
    seeing that is the point -- a router can be worse than nothing, and without
    this metric nobody notices.
    """
    if not points:
        return 0.0
    return mean([_pgr_at(points, c, axis) for c in BIN_CENTRES])


def cpt(points: Sequence[Point], target: float, axis: str = "calls") -> float:
    """Smallest budget fraction reaching `target` PGR; 1.0 if it never does.

    The operational number: "to recover 80% of the quality gap, you have to
    send this share of work to the expensive model". Returned on whichever axis
    was asked for, and the cost axis is the one to quote to anyone paying.
    """
    if not points:
        return 1.0
    key = (lambda p: p.call_fraction) if axis == "calls" else (lambda p: p.cost_fraction)
    best = 1.0
    found = False
    for p in points:
        if p.pgr >= target - 1e-12:
            best = min(best, key(p)) if found else key(p)
            found = True
    return best if found else 1.0


def oracle(episodes: Sequence[Episode]) -> list[Point]:
    """The best curve any router could have drawn on this data.

    Rank by the quality actually gained from the strong model. Nothing that
    only sees the prompt can do better, so the gap between this and the real
    router is the headroom -- and if the headroom is small, a better router is
    not the lever worth building.
    """
    ranked = [
        Episode(e.key, e.q_strong, e.q_weak, e.cost_strong, e.cost_weak,
                score=e.gain, split=e.split, modelled=e.modelled)
        for e in episodes
    ]
    return curve(ranked)


def random_router_ci(
    episodes: Sequence[Episode],
    *,
    trials: int = 200,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """Interval for the APGR of a router that ranks tasks at random.

    The published baseline is 0.500 in expectation, but on 40 recorded episodes
    a random router lands anywhere from 0.42 to 0.58, and a router reporting
    0.55 on that sample has demonstrated nothing. This is the interval that
    claim has to clear.
    """
    if not episodes:
        return (0.0, 0.0)
    rng = random.Random(seed)
    scores = []
    for _ in range(trials):
        shuffled = [
            Episode(e.key, e.q_strong, e.q_weak, e.cost_strong, e.cost_weak,
                    score=rng.random(), split=e.split, modelled=e.modelled)
            for e in episodes
        ]
        scores.append(apgr(curve(shuffled)))
    return (quantile(scores, alpha / 2.0), quantile(scores, 1.0 - alpha / 2.0))


def apgr_ci(
    episodes: Sequence[Episode],
    *,
    resamples: int = 200,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
    axis: str = "calls",
) -> tuple[float, float]:
    """Bootstrap interval for the router's own APGR, resampling episodes."""
    if len(episodes) < 2:
        return (0.0, 1.0)
    rng = random.Random(seed)
    n = len(episodes)
    draws = []
    for _ in range(resamples):
        sample = [episodes[rng.randrange(n)] for _ in range(n)]
        draws.append(apgr(curve(sample), axis))
    return (quantile(draws, alpha / 2.0), quantile(draws, 1.0 - alpha / 2.0))


def evaluate(
    episodes: Sequence[Episode],
    *,
    resamples: int = 200,
    seed: int = DEFAULT_SEED,
    targets: Sequence[int] = (50, 80, 95),
) -> Report:
    """Everything above, computed once, for one set of episodes."""
    n = len(episodes)
    if n == 0:
        return Report(0, 0.0, 0.0, 0.0, 0.0)
    pts = curve(episodes)
    rep = Report(
        n=n,
        q_strong=mean([e.q_strong for e in episodes]),
        q_weak=mean([e.q_weak for e in episodes]),
        cost_strong=sum(e.cost_strong for e in episodes),
        cost_weak=sum(e.cost_weak for e in episodes),
        curve=pts,
        apgr_calls=apgr(pts, "calls"),
        apgr_cost=apgr(pts, "cost"),
        apgr_ci=apgr_ci(episodes, resamples=resamples, seed=seed),
        random_apgr_ci=random_router_ci(episodes, trials=resamples, seed=seed),
        oracle_apgr=apgr(oracle(episodes), "calls"),
        cpt={t: cpt(pts, t / 100.0, "calls") for t in targets},
        cpt_cost={t: cpt(pts, t / 100.0, "cost") for t in targets},
        modelled_share=sum(1 for e in episodes if e.modelled) / n,
    )
    return rep


# --- inputs ----------------------------------------------------------------

def load_episodes(path: Path) -> list[Episode]:
    """Read episodes from JSONL. Unparseable lines are an error, not a skip.

    A silently skipped line is how an evaluation ends up scoring a router on
    the eight episodes that happened to parse. If a line is malformed, the run
    stops and says which one.
    """
    out: list[Episode] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: not JSON ({exc.msg})") from exc
            missing = [k for k in ("q_strong", "q_weak", "score") if k not in d]
            if missing:
                raise ValueError(f"{path}:{lineno}: missing {', '.join(missing)}")
            out.append(Episode(
                key=str(d.get("key", lineno)),
                q_strong=float(d["q_strong"]),
                q_weak=float(d["q_weak"]),
                cost_strong=float(d.get("cost_strong", 1.0)),
                cost_weak=float(d.get("cost_weak", 0.0)),
                score=float(d["score"]),
                split=str(d.get("split", "test")),
                modelled=bool(d.get("modelled", False)),
            ))
    return out


def split(episodes: Sequence[Episode], name: str) -> list[Episode]:
    """The named split, for the held-out evaluation that makes a score mean something."""
    return [e for e in episodes if e.split == name]


# Output tokens assumed for a modelled counterfactual run. A subagent answer
# in this workload is a few hundred tokens to a couple of thousand; the number
# only scales the output term of both arms, so the *ratio* the metric depends
# on barely moves with it. Named rather than inlined so the sensitivity can be
# checked by changing one thing.
ASSUMED_OUT_TOKENS = 1_200


def _positive_int(text: str) -> int:
    """An argparse type for a resample count. Zero produces a fabricated result.

    `bootstrap_ci(resamples=0)` returns `(0.0, 0.0)` -- a zero-width interval
    at zero, which reads as perfect precision about a number nothing estimated.
    """
    import argparse

    try:
        n = int(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from e
    if n < 1:
        raise argparse.ArgumentTypeError(
            f"--resamples must be at least 1, got {n}: with none, every interval "
            "comes back as an exact zero, which is not a measurement")
    return n


def _episode_date(row):
    """The calendar day an outcome row was recorded, or None.

    `Outcome.ts` is epoch seconds. A row with no usable stamp falls back to
    None, which prices at today -- there is nothing better to use, and it is
    what every one of these calls did unconditionally before.
    """
    from datetime import datetime

    ts = getattr(row, "ts", None)
    try:
        return datetime.fromtimestamp(float(ts)).date() if ts else None
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def episodes_from_outcomes(rows: Sequence[object]) -> list[Episode]:
    """Turn the recorded outcome log into router episodes.

    The outcome log is already a routing evaluation set and nobody was reading
    it as one. Each row says: this task went to this tier, and it either held
    or it had to be escalated. That is exactly a (weak arm, did the weak arm
    suffice) pair, which is the only thing PGR needs.

    Two things here are assumptions, and both are labelled MODELLED in the
    output rather than buried:

    * **The strong arm always succeeds.** `q_strong = 1`. It is what escalation
      means in this log -- the row exists because the escalation resolved the
      task -- but it is still an assumption, and it makes APGR an upper bound on
      the router's measured skill rather than a two-sided estimate.
    * **The unrun arm's cost is priced, not observed.** The strong model was
      never run on the tasks that did not escalate, so its cost comes from the
      date-aware price table and the recorded context size.

    The router's ranking signal is the tier it actually chose. That makes this
    an evaluation of the deployed router, not of a hypothetical one: if the
    tiers it assigned correlate with which tasks really needed the strong
    model, APGR clears 0.5, and if they do not, it does not.
    """
    from adder.decide.route.classify import Tier
    from adder.pricing.cost import run_cost

    strong_model = Tier.T2.model
    out: list[Episode] = []
    for i, row in enumerate(rows):
        tier_name = str(getattr(row, "tier", "") or "").upper()
        tier = Tier.__members__.get(tier_name)
        if tier is None:
            continue
        ctx = max(0, int(getattr(row, "context_tokens", 0) or 0))
        escalated = bool(getattr(row, "escalated", False))
        weak_model = tier.model
        recorded = float(getattr(row, "cost", 0.0) or 0.0)
        # Price the counterfactual arm on the day the episode ran, not today.
        # `recorded` was billed at the rates in force then; comparing it against
        # a strong-arm cost quoted at today's rates makes every episode logged
        # before a rate change score differently after it, with nothing about
        # the router having changed. `Turn.pricing_date` makes the same
        # correction for recorded turns and says why.
        when = _episode_date(row)
        cost_weak = recorded if recorded > 0 else run_cost(
            weak_model, ctx, ASSUMED_OUT_TOKENS, when)
        cost_strong = run_cost(strong_model, ctx, ASSUMED_OUT_TOKENS, when)
        # A tier at or above the strong reference has no cheaper arm to
        # compare against; including it would credit the router for a choice
        # it never made.
        if tier >= Tier.T2:
            continue
        out.append(Episode(
            key=str(getattr(row, "task_hash", "") or f"row{i}"),
            q_strong=1.0,
            q_weak=0.0 if escalated else 1.0,
            cost_strong=cost_strong,
            cost_weak=cost_weak,
            # Ties inside a tier are broken by context size: bigger context is
            # the router's own next-strongest signal for "this one is hard".
            score=float(int(tier)) + min(0.99, ctx / 1_000_000.0),
            split=str(getattr(row, "source", "recorded")),
            modelled=True,
        ))
    return out


# --- report ----------------------------------------------------------------

def _curve_rows(rep: Report, rows: int = 11) -> list[list[str]]:
    """Subsample the curve to a readable number of rows, endpoints kept."""
    pts = rep.curve
    if not pts:
        return []
    if len(pts) <= rows:
        chosen = pts
    else:
        idx = sorted({round(i * (len(pts) - 1) / (rows - 1)) for i in range(rows)})
        chosen = [pts[i] for i in idx]
    return [
        [
            render.pct(p.call_fraction),
            render.pct(p.cost_fraction),
            f"{p.quality:.3f}",
            render.money(p.cost),
            f"{p.pgr:+.3f}" if p.pgr < 0 else f"{p.pgr:.3f}",
        ]
        for p in chosen
    ]


def format_report(rep: Report, *, label: str = "router") -> str:
    out: list[str] = []
    out += render.heading(f"router evaluation — {label}", rule="=")
    if rep.n == 0:
        out.append("  no episodes. Nothing to score.")
        return "\n".join(out)

    out.append(render.kv("episodes", str(rep.n)))
    out.append(render.kv("quality strong / weak", f"{rep.q_strong:.3f} / {rep.q_weak:.3f}"))
    out.append(render.kv("cost strong / weak",
                         f"{render.money(rep.cost_strong)} / {render.money(rep.cost_weak)}"))
    if rep.modelled_share > 0:
        out.append(render.warn(
            f"  MODELLED: {render.pct(rep.modelled_share)} of episodes have a "
            "counterfactual arm that was never run."))
    if not rep.separable:
        out.append("")
        out.append(render.warn(
            "  The two models scored identically on this set. There is no quality gap "
            "to recover, so PGR has no denominator and every router trivially scores "
            "1.0. Route on price."))
        return "\n".join(out)

    out.append("")
    out += render.heading("call-performance curve")
    out += render.table(
        _curve_rows(rep),
        ["calls", "budget", "quality", "cost", "PGR"],
        align=">>>>>",
    )

    out.append("")
    out += render.heading("summary")
    lo, hi = rep.apgr_ci
    rlo, rhi = rep.random_apgr_ci
    out.append(render.kv("APGR (calls axis)", f"{rep.apgr_calls:.3f}  [{lo:.3f}, {hi:.3f}]"))
    out.append(render.kv("APGR (cost axis)",
                         f"{rep.apgr_cost:.3f}" if rep.cost_separable
                         else "n/a — the two arms did not differ in cost here"))
    out.append(render.kv("random router", f"0.500  [{rlo:.3f}, {rhi:.3f}]"))
    out.append(render.kv("oracle ceiling", f"{rep.oracle_apgr:.3f}"))
    out.append(render.kv("regret vs oracle", f"{rep.regret:.3f}"))
    out.append(render.kv("gain vs best single", f"{rep.gain_at_best:+.3f}"))
    if rep.cost_separable:
        out.append(render.kv("cost saved at equal quality", render.pct(rep.cost_save)))
    for target in sorted(rep.cpt):
        calls = rep.cpt[target]
        budget = (f", {render.pct(rep.cpt_cost[target])} of budget"
                  if rep.cost_separable else "")
        out.append(render.kv(f"CPT({target}%)", f"{render.pct(calls)} of calls{budget}"))

    out.append("")
    if rep.beats_random:
        out += render.wrap(
            f"The router beats random ordering: its interval [{lo:.3f}, {hi:.3f}] "
            f"clears random's [{rlo:.3f}, {rhi:.3f}].")
    else:
        out.append(render.warn(
            f"  Not distinguishable from random ordering on {rep.n} episodes "
            f"(router [{lo:.3f}, {hi:.3f}] vs random [{rlo:.3f}, {rhi:.3f}]). "
            "Collect more episodes before trusting the ranking."))

    if rep.gain_at_best <= 0.0:
        out.append("")
        out += render.wrap(
            "No threshold on this curve beats simply using "
            f"{'the strong' if rep.q_strong >= rep.q_weak else 'the weak'} model "
            "for everything. That is the comparison the published benchmarks "
            "found several well-known routers fail, and it is reported here for "
            "the same reason: a router that cannot beat one fixed choice is a "
            "cost lever, not a quality one. Read the cost-saved figure, not the "
            "APGR.")

    if not rep.cost_separable:
        out.append("")
        out += render.wrap(
            "The cost axis is flat here: the strong arm did not cost more in "
            "total than the weak one, so there is no budget to normalise "
            "against and every cost-axis figure would read 1.000 by "
            "construction. On the default path the weak arm is priced at what "
            "it really cost -- session context included -- against a modelled "
            "cold run for the strong arm, which is not a like-for-like "
            "comparison. Quote the calls axis.")
    drift = rep.apgr_calls - rep.apgr_cost
    if rep.cost_separable and abs(drift) > 0.05:
        worse = "worse" if drift > 0 else "better"
        out += render.wrap(
            f"Counting calls flatters the router by {abs(drift):.3f} APGR: it looks "
            f"{worse} once the x axis is dollars instead of call count. Agent turns "
            "do not cost the same, so the cost axis is the one to quote.")
    return "\n".join(out)


# --- cli -------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="adder routereval",
        description="Score a router: PGR, APGR, and CPT against a random baseline.",
    )
    ap.add_argument("path", nargs="?", type=Path,
                    help="JSONL of episodes; defaults to the recorded outcome log")
    ap.add_argument("--log", type=Path, default=None,
                    help="outcome log to derive episodes from (default $ADDER_LOG)")
    ap.add_argument("--split", default="", help="evaluate only this split (e.g. test)")
    ap.add_argument("--targets", default="50,80,95",
                    help="PGR levels to report CPT for (percent, comma-separated)")
    ap.add_argument("--resamples", type=_positive_int, default=200,
                    help="bootstrap resamples for the intervals (default 200)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="resampling seed; fixed so two runs agree")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    try:
        targets = tuple(int(t) for t in args.targets.split(",") if t.strip())
    except ValueError as exc:
        print(f"adder routereval: bad --targets: {exc}", file=sys.stderr)
        return 2

    env_path = os.environ.get("ADDER_EPISODES", "")
    path = args.path or (Path(env_path) if env_path else None)
    if path is not None:
        if not path.exists():
            print(f"adder routereval: no such file: {path}", file=sys.stderr)
            return 1
        try:
            episodes = load_episodes(path)
        except ValueError as exc:
            print(f"adder routereval: {exc}", file=sys.stderr)
            return 2
        label = path.name
    else:
        # No file given: score the router that actually ran, from its own
        # outcome log. This is the mode that needs no setup, and the one whose
        # answer is about this machine rather than about a benchmark.
        from adder.decide.track.outcomes import load, log_path

        log = log_path(args.log)
        episodes = episodes_from_outcomes(load(log))
        label = f"outcome log ({log.name})"

    if args.split:
        episodes = split(episodes, args.split)
        label = f"{label} [{args.split}]"

    if not episodes:
        rep = Report(0, 0.0, 0.0, 0.0, 0.0)
        if args.json:
            print(json.dumps(rep.to_json(), indent=2, sort_keys=True))
        else:
            print(format_report(rep, label=label))
            print(render.wrap(
                "Nothing to score yet. Either record escalations with "
                "`adder outcomes record`, or pass a JSONL of episodes with "
                "q_strong, q_weak and score per line.")[0])
        return 1

    rep = evaluate(episodes, resamples=args.resamples, seed=args.seed, targets=targets)
    if args.json:
        print(json.dumps(rep.to_json(), indent=2, sort_keys=True))
    else:
        print(format_report(rep, label=label))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
