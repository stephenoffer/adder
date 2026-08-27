"""Routing decisions: combine classification, session state, and the cost model.

The most important behaviour here is *declining to route*.

A routing step is not free. If `/adder` costs an extra turn, that turn re-reads
the whole context: at 500K tokens on Opus that is ~$0.25 before adder has
done anything useful. So a recommendation is only emitted when the modelled
saving clears that overhead. Otherwise the honest answer is "just do it".

Four gates, in the order they can each veto
-------------------------------------------
1. **Feasibility.** A tier whose context window cannot hold the task is not an
   option, however cheap it looks.
2. **Placement.** Delegating keeps the read out of a context that is re-read on
   every remaining turn. This is the largest lever and is checked first.
3. **Escalation risk.** A cheap tier that fails costs its own run *plus* the
   expensive one. `p_fail` comes from the measured outcome log, not a guess, so
   a tier that keeps failing on this project stops being recommended.
4. **Overhead.** If none of the above clears the cost of the routing turn
   itself, the plan is "inline".

Effort is chosen alongside the model because it is the one lever that reduces
output volume *without* invalidating the cache -- a downgrade rebuilds the
prefix, a lower effort level does not.

How the tier is actually picked
-------------------------------
Gate 3 used to be a single yes/no: "is the classifier's tier better than Opus?"
That could only ever escalate. When the classifier abstained -- which it does
deliberately and often, because text does not predict how deep a coding task
goes -- the answer was Opus, permanently, no matter how many times the same
project had shown that Sonnet finishes this kind of work.

So the tier is now chosen by minimising expected cost across the whole ladder:

    E[tier] = run(tier) + p_fail(tier) * (run(tier) + run(T2) + retry_overhead)

with two asymmetric permissions, because the two directions of error are not
symmetric. Moving *up* the ladder needs no evidence: the worst case is that you
paid for the model you would have chosen anyway. Moving *down*, below what the
classifier asked for, needs all three of: the classifier abstained rather than
matched a signal, the outcome log holds enough recent history at that tier to
be informative (not the 0.5 prior wearing a number), and the measured failure
rate sits under that tier's own break-even. Cheapness alone never buys a
downgrade -- under a no-evidence prior the cheapest rung always looks best, and
that is exactly the reasoning this refuses to do.

Why cross-vendor substitution only appears under `delegate`
-----------------------------------------------------------
The standing objection to "just use a cheaper model" is the prompt cache: it is
model-scoped, so moving a warm session rebuilds the whole prefix and usually
costs more than it saves. That objection is about the *session*. It does not
apply to a subagent, which starts cold: there is no prefix to invalidate, the
summary it returns is priced at the session model's rate no matter who produced
it, and a failure is contained to one run.

So delegation is the one placement where the vendor is genuinely free to
choose, and it is the only one where `substitutes()` will say anything. Even
there it prices the substitute as a *cascade*, not a swap -- a cheap subagent
that fails a quarter of the time and gets redone on Opus is not cheap -- and
stays silent unless the modelled saving clears the same routing overhead every
other recommendation here has to clear.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from adder.core import harness as _harness
from adder.core import settings as _settings
from adder.decide.route.classify import TIER_DIFFICULTY as _TIER_DIFFICULTY
from adder.decide.route.classify import Tier, Verdict, classify
from adder.pricing.cost import (
    EFFORT_OUTPUT_MULT,
    Decision,
    Rates,
    effort_saving,
    max_tolerable_p_fail,
    placement_cost,
    run_cost,
    switch_is_profitable,
)
from adder.pricing.registry import (
    context_window,
    fits,
    limit_str,
    supports_effort,
)
from adder.util.risk import DEFAULT_ALPHA, DEFAULT_CONFIDENCE, Guarantee, Interval, guarantee

M = 1_000_000.0

# Cost of the routing turn itself: it re-reads context and emits a dispatch.
ROUTING_TURN_OUTPUT_TOKENS = 400

# How far the summary a delegated read hands back may miss its modelled size.
#
# The compression ratio is the softest number in the placement decision: nothing
# in a transcript records how big a summary "should" have been, and a subagent
# that hands back twice what was asked for has not failed, it has just cost
# twice as much to carry. A factor of two either way is a stated assumption, not
# a measurement, and it is here so that the assumption is visible in the gate
# rather than hidden in a default argument.
COMPRESSION_BAND = 2.0

# How far the estimated output volume of a turn may miss. Same shape of
# assumption as COMPRESSION_BAND and the same justification: `est_out_tokens`
# defaults to 800 and nothing measures it per task, so the gate should see the
# width rather than pretend to a number it does not have.
OUTPUT_BAND = 2.0

# Pseudo-observations behind the classifier's prior, used only to give it a
# width. With no outcome log there is no measured spread on `p_redo`, and a
# point estimate cannot be integrated over. Four pseudo-observations centred on
# `prior_p_fail(confidence)` is a deliberately weak belief -- it widens as
# confidence falls, and twelve real observations swamp it. It is an assumption
# about *uncertainty*, not about the rate, and the rate itself is still the
# classifier's, unchanged.
PRIOR_PSEUDO_COUNT = 4.0

# Literal placeholders that mean argument substitution did not happen.
_PLACEHOLDER = re.compile(r"\$?\{?ARGUMENTS\}?|\$[0-9]", re.I)


# Quality tolerance when swapping a subagent for another vendor's model, in
# arena Elo points, by tier. A lookup can afford a weaker model than a
# multi-file refactor can; the numbers are deliberately tight, because the
# rating being compared is a preference proxy and not a measurement of agentic
# work. See docs/models.md.
SUBSTITUTE_TOLERANCE = {Tier.T0: 120.0, Tier.T1: 80.0, Tier.T2: 40.0, Tier.T3: 25.0}

# Task difficulty by tier, used to turn an Elo gap into a failure probability.
# Defined in `classify`, beside the enum it is keyed on, because `frontier`
# needs it too and had been substituting the classifier's confidence for it.
TIER_DIFFICULTY = _TIER_DIFFICULTY


@dataclass
class Substitute:
    """Another vendor's model standing in for a Claude subagent.

    `expected` is the number that decides it: the substitute's own run plus the
    modelled cost of redoing the failures on the tier it replaced. A swap that
    only looks cheap before escalation is not a saving, it is a deferred bill.
    """

    model: str
    org: str
    rating: float | None
    p_fail: float
    direct: float               # the substitute's own run (no summary handling)
    baseline: float             # the same run on the Claude tier model
    expected: float             # direct + p_fail * baseline
    open_weights: bool
    verified: bool
    reachable: str
    basis: str = "elo"          # where p_fail came from: "elo" or a measured scope

    @property
    def saving(self) -> float:
        return self.baseline - self.expected

    def render(self) -> str:
        elo = f"elo {self.rating:,.0f}" if self.rating else "unrated"
        delta = (f"saves ${self.saving:,.3f}" if self.saving >= 0
                 else f"costs ${-self.saving:,.3f} more")
        return (f"    {self.model:<28} ${self.expected:>7,.3f} vs ${self.baseline:,.3f}"
                f"   {delta}  {elo}, p_fail {self.p_fail:.0%} ({self.basis})"
                f"{'  [open weights]' if self.open_weights else ''}")


# Failure-rate prior when the outcome log has nothing to say about a tier.
#
# Beta(1,1)'s flat 0.5 is the right prior for "no information", and it is the
# wrong one here, because the classifier is information: a high-precision regex
# matched, and its stated confidence is a precision estimate. Handing the gate a
# flat 0.5 on day one -- which is every user's first session, since the log
# starts empty -- makes a bounded lookup look like a coin flip and sends it to
# Opus. Reading the prior off the classifier's own confidence keeps the gate
# honest without pretending to evidence it does not have; measured history
# replaces it as soon as there is any.
def prior_p_fail(confidence: float) -> float:
    """Failure prior implied by the classifier's confidence, capped at the flat prior."""
    return min(0.5, max(0.05, 1.0 - confidence))


@dataclass(frozen=True)
class Rung:
    """One tier priced as a candidate, kept so the choice can be audited.

    Every rung is reported, including the ones that lost and the ones that were
    never eligible. A router that prints only its answer is indistinguishable
    from a router that guessed.
    """

    tier: Tier
    model: str
    effort: str
    feasible: bool
    allowed: bool
    p_fail: float
    run: float                  # one cold run of the task on this tier
    expected: float             # run + p_fail * (redo on T2 + the turn that catches it)
    note: str = ""

    def render(self, chosen: Tier) -> str:
        mark = "->" if self.tier is chosen else "  "
        if not self.feasible:
            body = f"{'infeasible':>10}   {self.note}"
        elif not self.allowed:
            body = f"${self.expected:>9,.4f}   {self.note}"
        else:
            body = (f"${self.expected:>9,.4f}   run ${self.run:,.4f} "
                    f"+ {self.p_fail:.0%} chance of redoing it")
        return f"  {mark} {self.tier.name} {self.model:<20} {self.effort:<8}{body}"


@dataclass
class Plan:
    action: str                 # "inline" | "delegate" | "downgrade"
    tier: Tier
    model: str
    effort: str
    agent: str | None
    saving: float
    overhead: float
    confidence: float
    reasons: list[str]
    p_fail: float = 0.0
    warnings: list[str] = field(default_factory=list)
    substitutes: list[Substitute] = field(default_factory=list)
    ladder: list[Rung] = field(default_factory=list)
    guarantee: Guarantee | None = None

    @property
    def worth_it(self) -> bool:
        return self.action != "inline" and self.saving > self.overhead

    def render(self) -> str:
        head = f"{self.action.upper()}"
        if self.action != "inline":
            head += f" -> {self.agent} ({self.model}, effort={self.effort})"
        lines = [
            head,
            f"  modelled saving ${self.saving:,.3f}  routing overhead ${self.overhead:,.3f}"
            f"  confidence {self.confidence:.2f}",
        ]
        if self.guarantee is not None:
            g = self.guarantee
            lines.append(
                f"  {'guaranteed' if g.dominant else 'confidence'} "
                f"{g.confidence:.0%} that this is cheaper than not doing it"
                + ("; cheaper at every admissible input" if g.dominant else ""))
        lines += [f"  - {r}" for r in self.reasons]
        for w in self.warnings:
            lines.append(f"  ! {w}")
        if self.action != "inline" and not self.worth_it:
            lines.append("  - saving does not clear routing overhead; do it inline instead")
        lines += self._render_ladder()
        lines += self._render_substitutes()
        return "\n".join(lines)

    def _render_ladder(self) -> list[str]:
        if not self.ladder:
            return []
        return ["", "  Tier chosen by expected cost, including the risk of redoing it:"] + [
            r.render(self.tier) for r in self.ladder
        ]

    def _render_substitutes(self) -> list[str]:
        """Show a substitute only if it clears the same bar everything else here does.

        A 0.6-cent saving on a subagent run is not a recommendation, it is
        noise with a dollar sign on it. When nothing clears the routing
        overhead, one line saying so is more useful than a table -- it answers
        "should I be running this on a cheaper vendor" with a number instead of
        silence.
        """
        if not self.substitutes:
            return []
        worth = [s for s in self.substitutes if s.saving > self.overhead]
        if not worth:
            best = max(self.substitutes, key=lambda s: s.saving)
            return ["",
                    f"  - cheapest cross-vendor subagent ({best.model}) saves "
                    f"${best.saving:,.3f} against ${self.overhead:,.3f} of routing "
                    f"overhead;",
                    "    the placement was the lever, not the vendor"]
        out = ["",
               "  Cheaper subagents that clear this tier's quality bar (a subagent "
               "starts cold, so",
               "  there is no model-scoped cache to rebuild; priced including "
               "escalation):"]
        out += [s.render() for s in worth]
        out.append(f"    {worth[0].reachable}")
        return out


def _harness_default() -> str:
    from adder.core import harness as _h

    return _h.default()


def routing_overhead(context_tokens: int, session_model: str, on: date | None = None,
                     *, carry=None) -> float:
    """Marginal cost of spending one extra turn deciding how to route.

    This is the bar every recommendation has to clear, so understating it is the
    one error here that makes adder emit advice too eagerly. The default
    assumes the routing turn re-reads the context at the cache-read rate, which
    is the best case. Pass a measured `carry.Carry` and it uses what turns
    actually paid instead -- 15% more, on the transcripts this was written
    against, and the bar moves with it.
    """
    r = Rates.for_model(session_model, on=on)
    # Without a measured carry model, the best case is that the routing turn
    # re-reads the context at whatever this provider charges for a cache read.
    # On a provider with no cache that is the full input rate, which raises the
    # bar tenfold -- correctly, because on that provider a routing turn really
    # does cost ten times more, and advice that cannot clear it should not be
    # emitted.
    per_token = r.cache_read if carry is None else r.inp * carry.read_mult
    return (
        context_tokens * per_token
        + ROUTING_TURN_OUTPUT_TOKENS * r.out
    ) / M


def choose_effort(tier: Tier, model: str) -> str:
    """Lowest effort level that suits the tier, if the model accepts one.

    Effort is free to lower mid-session: unlike a model switch it does not
    invalidate the prompt cache, so it is the safe way to cut output volume.
    """
    want = tier.effort
    if not supports_effort(model, want):
        # Not every model takes an effort level, and the ones that do disagree
        # on the vocabulary: Haiku 4.5 rejects `effort` outright, OpenAI adds
        # `minimal` and has no `xhigh`/`max`, and Google takes an integer
        # thinking budget rather than a label. Asking for a level the model
        # does not accept is a 400, not a cheaper turn.
        return "default"
    return want


def _effort_out_tokens(est_out_tokens: int, effort: str) -> int:
    """Output volume at a given effort, quoted against `high` = 1.0.

    T2 and T3 are the same model; the only thing separating them is how much
    thinking they are asked to do. If the ladder priced them identically, the
    search would be indifferent between them and would pick whichever came
    first, which is not a decision.
    """
    base = EFFORT_OUTPUT_MULT["high"]
    mult = EFFORT_OUTPUT_MULT.get(effort, base)
    return max(1, round(est_out_tokens * mult / base))


def right_size(
    v: Verdict,
    *,
    need_tokens: int,
    est_out_tokens: int,
    retry_overhead: float,
    floor_tier: Tier | None = None,
    project: str | None = None,
    p_fail_override: float | None = None,
    on: date | None = None,
    task: str = "",
) -> tuple[Tier, list[Rung], list[str]]:
    """Pick the tier with the lowest expected cost, and show the whole ladder.

    Returns `(tier, ladder, reasons)`. `ladder` holds every rung that was
    considered, eligible or not, so the caller can print the working.

    The escalation target is always T2: it is the tier the classifier falls back
    to when it abstains, so it is the thing a cheap attempt is being measured
    against. T3 is priced as T2's model at a higher effort rather than as an
    escalation target of its own -- there is nothing above it to escalate to.
    """
    floor = floor_tier if floor_tier is not None else v.tier
    reasons: list[str] = []

    # Pass 1: what each rung is, what it costs to run once, and how often it
    # fails. Failure rates are settled before any expected cost is computed,
    # because every rung below T2 is priced against T2's *expected* cost and
    # that number has to already carry the clamp below.
    specs: list[tuple[Tier, str, str, bool, float, float, object]] = []
    prev_pf = 1.0
    for tier in Tier:
        model = tier.model
        effort = choose_effort(tier, model)
        out = _effort_out_tokens(est_out_tokens, effort)
        feasible = fits(model, need_tokens)
        run = run_cost(model, min(need_tokens, context_window(model, need_tokens)),
                       out, on)

        if p_fail_override is not None:
            pf, ev = p_fail_override, None
        else:
            ev = _evidence(tier, project, task)
            # One prior across the whole ladder when the log is silent. Giving
            # the higher rungs a lower prior would be inventing evidence, and
            # the invention is not harmless: it lets a made-up number, rather
            # than a measurement, push work up the ladder. With no data, this
            # reduces to "the cheapest tier the classifier permits", which is
            # at least a rule someone can argue with.
            pf = ev.p_fail if (ev is not None and ev.informative) else prior_p_fail(v.confidence)

        # A stronger rung cannot be more likely to fail than a weaker one, so
        # the sequence is clamped monotone. Without this, a tier nobody has
        # logged sits at the prior while the tier below it sits at a measured
        # rate, and the ladder reports T3 (Opus at xhigh) as seven times more
        # failure-prone than T2 (the same model at high) purely because of
        # which label the log happens to carry.
        pf = min(pf, prev_pf)
        prev_pf = pf
        specs.append((tier, model, effort, feasible, pf, run, ev))

    # Expected cost of finishing on the escalation target, which every cheaper
    # rung is priced against. T2 is the tier the classifier falls back to when
    # it abstains, so it is what a cheap attempt is being measured against.
    t2 = next(sp for sp in specs if sp[0] is Tier.T2)
    t2_expected = t2[5] + t2[4] * (t2[5] + retry_overhead)

    # Pass 2: expected cost, and whether this rung is even on the table.
    rungs: list[Rung] = []
    for tier, model, effort, feasible, pf, run, ev in specs:
        # Below T2 a failure escalates and pays T2's own expected cost; at or
        # above T2 there is nowhere to escalate to, so a failure is a redo on
        # the same model. Either way the turn that catches the failure is
        # charged, and the failed run is charged once -- it already happened.
        if tier >= Tier.T2:
            expected = run + pf * (run + retry_overhead)
        else:
            expected = run + pf * (t2_expected + retry_overhead)

        allowed, note = True, ""
        if not feasible:
            allowed = False
            note = f"holds {limit_str(model)}, task needs ~{need_tokens:,}"
        elif tier is Tier.T0 and not v.read_only:
            # T0 dispatches to `route-t0`, which holds Read, Grep, Glob and
            # Bash and no write tool. A task the classifier could not read as
            # read-only cannot be carried out there whatever it costs, so this
            # is a feasibility question wearing a permission's clothes -- and
            # it is the only thing that consumes `Verdict.read_only`.
            allowed = False
            note = ("nothing marks this read-only and T0's agent has no write "
                    "tools; it could not carry out a change there")
        elif tier < floor:
            allowed, note = _may_descend(
                tier, floor, v, ev, pf, need_tokens,
                _effort_out_tokens(est_out_tokens, effort), retry_overhead, on)
        rungs.append(Rung(tier, model, effort, feasible, allowed, pf, run,
                          expected, note))

    eligible = [r for r in rungs if r.feasible and r.allowed]
    fell_back = not eligible
    if fell_back:
        # This used to carry `# pragma: no cover - T2 always fits`, and the
        # assumption behind the pragma was wrong: `--read-tokens 5000000` marks
        # every rung on the ladder infeasible, so the branch is one flag away.
        # The old fallback then re-admitted the infeasible rungs silently and
        # named a model that provably could not hold the task. The two ways to
        # get here are different problems and are answered differently.
        fits_at_least = [r for r in rungs if r.feasible]
        if fits_at_least:
            eligible = fits_at_least
            reasons.append(
                "no rung was both feasible and permitted; falling back to the "
                "cheapest that at least holds the task")
        else:
            widest = max(rungs, key=lambda r: context_window(r.model, 0))
            eligible = [widest]
            reasons.append(
                f"nothing on the ladder holds ~{need_tokens:,} tokens: "
                f"{widest.model} has the largest window ({limit_str(widest.model)}) "
                f"and will still overflow. Split the read rather than dispatching this")
    best = min(eligible, key=lambda r: r.expected)

    if fell_back:
        # The fallback already said why this rung was picked. The comparisons
        # below all claim the ladder chose it on merit, and one of them cites
        # the outcome log as backing a descent the log had no part in.
        pass
    elif best.tier < floor:
        reasons.append(
            f"{best.tier.name} costs ${best.expected:,.4f} expected against "
            f"{floor.name}'s ${next(r.expected for r in rungs if r.tier is floor):,.4f}, "
            f"and the outcome log backs it: "
            f"{_evidence_note(best.tier, project, task)}")
    elif best.tier > floor:
        reasons.append(
            f"escalating {floor.name} -> {best.tier.name}: at p_fail "
            f"{next(r.p_fail for r in rungs if r.tier is floor):.0%} the redo risk "
            f"costs more than running it once on {best.model}")
    else:
        reasons.append(
            f"{best.tier.name} is the cheapest tier in expectation "
            f"(${best.expected:,.4f}, including a {best.p_fail:.0%} chance of a redo)")
    return best.tier, rungs, reasons


def _may_descend(tier: Tier, floor: Tier, v: Verdict, ev, pf: float,
                 need_tokens: int, out_tokens: int, retry_overhead: float,
                 on: date | None) -> tuple[bool, str]:
    """Three conditions before dropping below what the classifier asked for."""
    if not v.abstained:
        return False, f"below the classifier's {floor.name}, which it matched a signal for"
    if ev is None or not ev.informative:
        return False, "no measured history at this tier; a prior is not evidence"
    cap = max_tolerable_p_fail(tier.model, Tier.T2.model, ctx_tokens=need_tokens,
                               est_out_tokens=out_tokens,
                               retry_overhead=retry_overhead, on=on)
    if pf >= cap:
        return False, f"measured p_fail {pf:.0%} is at or above its {cap:.0%} break-even"
    return True, ""


def _haircut(ledger) -> float:
    """How much of a predicted saving to believe, from the ledger's own record.

    Bounds protect the gate from its inputs being uncertain. Nothing in them
    protects it from the model being biased, because a bias moves every corner
    of the box together. That only shows up as realized savings falling short of
    predicted ones, so it is measured and applied here. Missing or unreadable
    ledger means no correction, never a penalty: a tool that throttles itself
    because a file could not be opened is worse than one that does not try.
    """
    if ledger is None:
        return 1.0
    try:
        return float(ledger.haircut())
    except Exception:                                     # pragma: no cover
        return 1.0


def _evidence(tier: Tier, project: str | None, task: str = ""):
    """The outcome log's view of a tier, or None if it cannot be read.

    `DEFAULT_LOG` is resolved on every call rather than captured as a default
    argument, so a test can point the log somewhere harmless. A router that can
    only be tested against the caller's real `~/.claude` is not testable.

    With a `task`, the tier-wide rate is sharpened by the rate over recorded
    runs whose vocabulary resembles it -- but only in the directions
    `similar.sharpen` permits, which are not symmetric. Without one, or with a
    log that carries no sketches, this is exactly the estimate it always was:
    the neighbour half is additive and its absence costs nothing.
    """
    try:
        from adder.decide.track.outcomes import evidence

        # No explicit path: `outcomes.log_path` resolves the `log` setting per
        # call, so a project that points its outcome log somewhere is actually
        # read. Passing `DEFAULT_LOG` here pinned it to the import-time
        # environment and made the setting decorative.
        ev = evidence(tier.name, project)
    except Exception:
        return None
    if not task:
        return ev
    try:
        from adder.decide.track.similar import evidence_like, sharpen

        return sharpen(ev, evidence_like(task, tier.name))
    except Exception:
        # A sharper estimate is an improvement, not a dependency. Anything that
        # goes wrong reading it leaves the tier-wide rate in place rather than
        # taking the gate down with it.
        return ev


def _evidence_note(tier: Tier, project: str | None, task: str = "") -> str:
    ev = _evidence(tier, project, task)
    return ev.describe() if ev is not None else "no outcome log"


def assess_placement(
    *,
    tokens_read: int,
    summary_tokens: int,
    remaining_turns: int,
    session_model: str,
    sub_model: str,
    overhead: float,
    p_redo: float,
    p_redo_bounds: Interval | None = None,
    p_redo_quantiles=None,
    horizon=None,
    turn_index: int = 0,
    carry=None,
    context_tokens: int = 0,
    haircut: float = 1.0,
    alpha: float = DEFAULT_ALPHA,
    threshold: float = DEFAULT_CONFIDENCE,
    on: date | None = None,
) -> tuple[Decision, Guarantee]:
    """Price delegation at its midpoint, its worst corner, and in probability.

    The saving is multilinear in the three quantities that are not known --
    remaining turns, the redo rate, and how big the summary comes back -- so
    `risk.worst_case` finds the true worst corner by enumerating vertices rather
    than searching, and `risk.p_cheaper` integrates the same function over the
    marginals. Both are exact for this function; neither needs a sample.

    `haircut` is the ledger's measured over-promise correction, applied to the
    saving before it meets its gate. It is the one term here that protects
    against the *model* being wrong rather than its inputs being uncertain.

    Turn and token counts are clamped at zero on the way in. None of them has a
    meaning below it, and a negative one used to build an `Interval` whose
    point sat under its own lower bound and raise `interval out of order` from
    four frames down. The CLI rejects negatives at the flag now, but the hook
    does not take this number from a human -- it computes it from the horizon
    estimator -- so the arithmetic defends itself too.
    """
    remaining_turns = max(0, int(remaining_turns))
    tokens_read = max(0, int(tokens_read))
    summary_tokens = max(1, int(summary_tokens))
    context_tokens = max(0, int(context_tokens))
    turn_index = max(0, int(turn_index))

    def saving(remaining: float, p: float, summary: float) -> float:
        _, _, d = placement_cost(
            tokens_read=tokens_read,
            summary_tokens=max(1, int(summary)),
            remaining_turns=max(0, int(remaining)),
            main_model=session_model,
            sub_model=sub_model,
            p_redo=max(0.0, min(1.0, p)),
            redo_overhead=overhead,
            carry=carry,
            context_tokens=context_tokens,
            on=on,
        )
        return haircut * d.saving

    if horizon is not None:
        r_bounds = horizon.bounds(turn_index, alpha=alpha)
        # Anchor on the horizon the caller is actually acting under, and let the
        # measured distribution supply only the spread. Substituting the whole
        # estimate would make a `--remaining` flag decorative.
        scale = remaining_turns / r_bounds.point if r_bounds.point > 0 else 1.0
        r_bounds = Interval(r_bounds.lo * scale, float(remaining_turns),
                            max(r_bounds.hi * scale, float(remaining_turns)))
        r_marg = [q * scale for q in _horizon_quantiles(horizon, turn_index)]
    else:
        r_bounds = Interval.exact(float(remaining_turns))
        r_marg = [float(remaining_turns)]

    if p_redo_bounds is None:
        from adder.util.risk import bounds_from_mean, quantiles_from_mean

        # Moment-matched so the interval is centred on the rate the caller
        # actually holds. Adding a Beta(1,1) prior on top of a prior would move
        # the point estimate as a side effect of asking for its width, and the
        # midpoint of the gate would stop matching the number reported next to it.
        p_bounds = bounds_from_mean(p_redo, PRIOR_PSEUDO_COUNT, alpha=alpha)
        p_marg = quantiles_from_mean(p_redo, PRIOR_PSEUDO_COUNT)
    else:
        p_bounds = p_redo_bounds
        p_marg = list(p_redo_quantiles or [p_redo_bounds.point])

    summ = float(max(1, summary_tokens))
    s_bounds = Interval(summ / COMPRESSION_BAND, summ, summ * COMPRESSION_BAND)

    g = guarantee(
        saving,
        {"remaining": r_bounds, "p": p_bounds, "summary": s_bounds},
        overhead=overhead,
        marginals={"remaining": r_marg, "p": p_marg,
                   "summary": [s_bounds.lo, summ, s_bounds.hi]},
        alpha=alpha, threshold=threshold,
    )

    _, _, point = placement_cost(
        tokens_read=tokens_read, summary_tokens=summary_tokens,
        remaining_turns=remaining_turns, main_model=session_model,
        sub_model=sub_model, p_redo=p_redo, redo_overhead=overhead,
        carry=carry, context_tokens=context_tokens, on=on,
    )
    return point, g


def _switch_guarantee(from_model: str, to_model: str, ctx: int, est_out: int,
                      overhead: float, haircut: float, threshold: float,
                      on: date | None) -> Guarantee:
    """Price an in-session downgrade against the one number it turns on.

    The break-even for a switch is `out > ctx * (rate_to_in - 0.1*rate_from_in)
    / (rate_from_out - rate_to_out)`, so the whole decision rides on the output
    estimate -- and that estimate is a default argument, not a measurement. A
    band around it is the honest input; the saving is linear in it, so the
    vertex enumeration is exact.
    """
    def saving(out: float) -> float:
        return haircut * switch_is_profitable(
            from_model, to_model, ctx, max(1, int(out)), on=on).saving

    band = Interval(est_out / OUTPUT_BAND, float(est_out), est_out * OUTPUT_BAND)
    strata = [band.lo, band.lo * 1.5, band.point, band.point * 1.5, band.hi]
    return guarantee(saving, {"out": band}, overhead=overhead,
                     marginals={"out": strata}, threshold=threshold)


def _horizon_quantiles(horizon, turn_index: int, strata: int = 8) -> list[float]:
    """Quantile ladder of remaining turns, falling back to the point estimate."""
    try:
        from adder.util.risk import empirical_quantiles

        alive = horizon.survivors(turn_index)
        if len(alive) >= 5:
            return empirical_quantiles(alive, strata=strata)
    except Exception:                                     # pragma: no cover
        pass
    return [float(horizon.mean_remaining(turn_index))]


def decide(
    task: str,
    *,
    context_tokens: int,
    remaining_turns: int,
    session_model: str | None = None,
    est_read_tokens: int | None = None,
    est_out_tokens: int = 800,
    compression: float = 10.0,
    project: str | None = None,
    p_fail: float | None = None,
    on: date | None = None,
    horizon=None,
    carry=None,
    turn_index: int = 0,
    ledger=None,
    min_confidence: float = DEFAULT_CONFIDENCE,
    harness: str | None = None,
) -> Plan:
    """Recommend where and on what model to run `task`.

    `horizon`, `carry` and `ledger` are the three sources of measured
    uncertainty. Supplying them turns the placement gate from "positive at the
    midpoint" into "positive with at least `min_confidence` probability, after
    discounting for how much this model has over-promised in the past". Leaving
    them as `None` claims no uncertainty and reproduces the older behaviour,
    which is the right default for a caller that has nothing to measure.
    """
    session_model = session_model or _settings.session_model()
    # Same reason as `assess_placement`: these arrive from the horizon
    # estimator and the size model, not from a human, and every gate below
    # multiplies by them. A negative turn count is not a pessimistic estimate,
    # it is a broken one, and it used to surface as a traceback.
    context_tokens = max(0, int(context_tokens))
    remaining_turns = max(0, int(remaining_turns))
    if est_read_tokens is not None:
        est_read_tokens = max(0, int(est_read_tokens))
    est_out_tokens = max(1, int(est_out_tokens))
    overhead = routing_overhead(context_tokens, session_model, on, carry=carry)
    warnings: list[str] = []

    # Guard: an empty or unsubstituted task must never produce a confident
    # dispatch. If `$ARGUMENTS` fails to substitute in the skill, the string
    # arrives literally -- refuse rather than delegate an imaginary task.
    stripped = (task or "").strip()
    if not stripped or _PLACEHOLDER.fullmatch(stripped):
        return Plan(
            action="inline", tier=Tier.T2, model=session_model, effort="high",
            agent=None, saving=0.0, overhead=overhead, confidence=0.0,
            reasons=["no task text received; refusing to route "
                     "(check that the task was passed through)"],
        )

    v: Verdict = classify(task)
    reasons = list(v.reasons)

    # Estimate how much the task will pull into context if run inline.
    if est_read_tokens is None:
        est_read_tokens = {Tier.T0: 8_000, Tier.T1: 20_000,
                           Tier.T2: 60_000, Tier.T3: 120_000}[v.tier]

    # --- Gate 1: feasibility. A tier that cannot hold the task is not an option.
    tier, model = v.tier, v.tier.model
    need = est_read_tokens + max(200, int(est_read_tokens / compression)) + 400
    while not fits(model, need) and tier < Tier.T2:
        warnings.append(
            f"{model} holds {limit_str(model)} but the task needs "
            f"~{need:,}; escalating a tier for feasibility, not capability")
        tier = Tier(int(tier) + 1)
        model = tier.model

    # --- Gate 2: escalation risk, priced across the whole ladder.
    #
    # `retry_overhead` is the routing turn again, and it belongs here: a
    # subagent that returns something wrong does not say so. A main-session
    # turn has to read the result, judge it, and dispatch again, and at this
    # context size that turn is not rounding error.
    tier, ladder, ladder_reasons = right_size(
        v, need_tokens=need, est_out_tokens=est_out_tokens,
        retry_overhead=overhead, floor_tier=tier, project=project,
        p_fail_override=p_fail, on=on, task=task)
    model = tier.model
    reasons += ladder_reasons
    p_fail = next((r.p_fail for r in ladder if r.tier is tier), p_fail or 0.0)

    effort = choose_effort(tier, model)

    # --- Gate 3: placement, priced against its own uncertainty.
    #
    # A delegated read that comes back missing what the session needed is not a
    # neutral outcome: it costs the subagent run, the turn that noticed, and
    # then the inline read anyway. `p_redo` is that risk, and it is the tier's
    # own measured escalation rate rather than a new number -- an escalated
    # subagent run IS a delegation that did not hold.
    ev = _evidence(tier, project, task)
    if ev is not None and ev.informative:
        p_redo, p_bounds, p_marg = ev.p_fail, ev.bounds(), ev.quantiles()
    else:
        p_redo, p_bounds, p_marg = p_fail, None, None

    haircut = _haircut(ledger)
    place, g = assess_placement(
        tokens_read=est_read_tokens,
        summary_tokens=max(200, int(est_read_tokens / compression)),
        remaining_turns=remaining_turns,
        session_model=session_model,
        sub_model=model,
        overhead=overhead,
        p_redo=p_redo,
        p_redo_bounds=p_bounds,
        p_redo_quantiles=p_marg,
        horizon=horizon,
        turn_index=turn_index,
        carry=carry,
        context_tokens=context_tokens,
        haircut=haircut,
        threshold=min_confidence,
        on=on,
    )
    if haircut < 1.0:
        warnings.append(
            f"past recommendations delivered {haircut:.0%} of what they promised; "
            f"this saving is discounted to match")

    # What this runtime can actually be told to do. `harness.py` carries the
    # answer and says why it must be consulted: "a report that recommends it
    # anyway is recommending a feature the user does not have". Aider is one
    # conversation -- there is no subagent to hand a read to -- and several
    # harnesses cannot switch model mid-session, which makes the downgrade
    # question moot rather than negative. Both fields existed and neither was
    # read, so both recommendations were emitted regardless of the runtime.
    rig = _harness.get(harness)

    if place and g.safe and rig.supports_subagents:
        reasons.append(place.reason)
        reasons.append(g.describe())
        return Plan(
            action="delegate", tier=tier, model=model, effort=effort,
            agent=tier.agent, saving=place.saving, overhead=overhead,
            confidence=v.confidence, reasons=reasons, p_fail=p_fail,
            warnings=warnings, ladder=ladder, guarantee=g,
        )
    if place and g.safe and not rig.supports_subagents:
        warnings.append(
            f"delegation would save ${place.saving:,.3f} here, but {rig.label} "
            "runs one conversation: there is no subagent to delegate to")

    declined = bool(place) and not g.safe
    if declined:
        # The midpoint says delegate and the distribution does not: the expected
        # saving is being carried by a tail rather than by the typical outcome.
        # Declining is the whole point of having measured the spread.
        reasons.append(g.describe())

    # Delegation not worth it. Would an in-session downgrade help? Usually not.
    if tier < Tier.T2 and not rig.supports_model_switch:
        reasons.append(
            f"{rig.label} cannot switch model mid-session, so the downgrade "
            "question does not arise here")
    elif tier < Tier.T2:
        sw: Decision = switch_is_profitable(
            session_model, model, context_tokens, est_out_tokens, on=on
        )
        reasons.append(sw.reason)
        # This branch used to emit without checking its own overhead, which the
        # `emitted advice clears its own overhead` sweep in `validate.py` caught
        # at three points out of 240: a $0.011 downgrade recommended by a turn
        # that cost $0.015 to spend. The saving was real and smaller than the
        # asking, which is the exact failure this tool exists to prevent
        # elsewhere and was not preventing here.
        sw_g = _switch_guarantee(session_model, model, context_tokens,
                                 est_out_tokens, overhead, haircut,
                                 min_confidence, on)
        if sw and not sw_g.safe:
            reasons.append(sw_g.describe())
        if sw and sw_g.safe:
            return Plan(
                action="downgrade", tier=tier, model=model, effort=effort,
                agent=None, saving=sw.saving, overhead=overhead,
                confidence=v.confidence, reasons=reasons, p_fail=p_fail,
                warnings=warnings, ladder=ladder, guarantee=sw_g,
            )
    else:
        reasons.append("classified as needing the strong model; no downgrade considered")

    # Last resort that keeps the cache warm: lower the effort level instead of
    # the model. Same prefix, same cache, fewer output tokens.
    inline_effort = "high"
    if supports_effort(session_model, "medium") and v.tier <= Tier.T1:
        saved, ed = effort_saving(
            est_out_tokens, session_model, from_effort="high", to_effort="medium",
            remaining_turns=remaining_turns, on=on)
        if ed and saved > 0:
            reasons.append(ed.reason)
            inline_effort = "medium"

    if not declined and rig.supports_subagents:
        # Do not re-state the delegation case after declining it; the guarantee
        # line above already says why, and a reason list that recommends
        # delegating inside an `inline` plan is how a router loses an argument
        # with the person reading it. The same holds when the runtime has no
        # subagent at all: the warning says so, and a reason line underneath it
        # reading "delegate: saves $1.05" contradicts the plan it belongs to.
        reasons.append(place.reason)
    return Plan(
        action="inline", tier=Tier.T2, model=session_model, effort=inline_effort,
        agent=None, saving=0.0, overhead=overhead,
        confidence=v.confidence, reasons=reasons, p_fail=p_fail,
        warnings=warnings, ladder=ladder, guarantee=g,
    )


@dataclass
class Batch:
    """Several routing decisions taken in one turn instead of several.

    The routing overhead in this repo is charged per recommendation, and that is
    only correct if each one costs its own turn. Ask once about five steps and
    the context is re-read once, not five times -- so the batch pays one
    overhead and the individual savings are measured against it collectively.

    Two consequences, and the second is the interesting one.

    The obvious one: a step whose saving is real but small never clears a whole
    routing turn on its own and is always declined, however many of them there
    are. Batched, five such steps clear it together.

    The less obvious one: **batching raises confidence**. The saving on each
    step carries two kinds of risk -- the horizon, which every step in a session
    shares, and its own redo risk, which is idiosyncratic. Summing k steps
    leaves the shared risk untouched and averages the idiosyncratic part down by
    a factor of sqrt(k). So the batch is a more certain bet than its members,
    and the confidence reported here is computed that way rather than by
    multiplying k marginals together as if the horizon were independent across
    steps of the same session. It is not; there is one session.
    """

    plans: list[Plan] = field(default_factory=list)
    overhead: float = 0.0
    confidence: float = 0.0
    threshold: float = DEFAULT_CONFIDENCE

    @property
    def acted(self) -> list[Plan]:
        return [p for p in self.plans if p.action != "inline"]

    @property
    def saving(self) -> float:
        return sum(p.saving for p in self.acted)

    @property
    def worth_it(self) -> bool:
        return (bool(self.acted) and self.saving > self.overhead
                and self.confidence >= self.threshold)

    def render(self) -> str:
        if not self.plans:
            return "nothing to route"
        head = [
            f"BATCH of {len(self.plans)}: {len(self.acted)} worth delegating",
            f"  saving ${self.saving:,.3f} against ONE routing overhead "
            f"${self.overhead:,.3f}  confidence {self.confidence:.0%}",
        ]
        if not self.worth_it:
            head.append("  the batch does not clear its own turn; do these inline")
        for p in self.plans:
            mark = "->" if p.action != "inline" else "  "
            head.append(f"  {mark} {p.action:<9} {p.model:<20} ${p.saving:>8,.3f}")
        return "\n".join(head)


def schedule(
    tasks: list[str],
    *,
    context_tokens: int,
    remaining_turns: int,
    session_model: str | None = None,
    project: str | None = None,
    horizon=None,
    carry=None,
    ledger=None,
    turn_index: int = 0,
    min_confidence: float = DEFAULT_CONFIDENCE,
    on: date | None = None,
) -> Batch:
    """Route several steps in one turn, and price them against one turn's cost.

    Each step is decided with its own overhead set to zero -- it is not paying
    for a turn of its own -- and the batch is then checked against the single
    overhead they share. That is the exact optimum here rather than a heuristic:
    the only cost shared between the steps is the one turn, so the best subset
    is "every step with a positive saving", and the only remaining question is
    whether their total clears it.
    """
    session_model = session_model or _settings.session_model()
    overhead = routing_overhead(context_tokens, session_model, on, carry=carry)
    plans = [
        decide(t, context_tokens=context_tokens, remaining_turns=remaining_turns,
               session_model=session_model, project=project, horizon=horizon,
               carry=carry, ledger=ledger, turn_index=turn_index,
               min_confidence=0.0, on=on)
        for t in tasks
    ]
    # `min_confidence=0.0` above so a step is not rejected for failing to clear
    # a turn it is not paying for. The confidence test happens once, on the sum.
    acted = [p for p in plans if p.action != "inline"]
    conf = _batch_confidence(acted, overhead=overhead, horizon=horizon,
                             turn_index=turn_index, remaining_turns=remaining_turns)
    for p in plans:
        p.overhead = overhead / max(1, len(acted)) if p.action != "inline" else overhead
    return Batch(plans=plans, overhead=overhead, confidence=conf,
                 threshold=min_confidence)


def _batch_confidence(acted: list[Plan], *, overhead: float, horizon,
                      turn_index: int, remaining_turns: int) -> float:
    """P(total saving > one routing turn), with the horizon shared across steps.

    The horizon is the common shock: every step in a session lives or dies by the
    same remaining-turn count, so it is enumerated over its quantile ladder and
    the total is recomputed at each. Redo risk is idiosyncratic and is left at
    its expectation, which is what averaging k independent draws converges to.
    """
    if not acted:
        return 0.0
    if horizon is None:
        return 1.0 if sum(p.saving for p in acted) > overhead else 0.0
    ladder = _horizon_quantiles(horizon, turn_index)
    ref = horizon.mean_remaining(turn_index) or float(remaining_turns)
    if ref <= 0:
        return 0.0
    wins = 0
    for r in ladder:
        # Saving is linear in remaining turns once the write term is small, so
        # scaling by the horizon ratio is the right first-order recomputation
        # and avoids re-running every gate for every quantile.
        scale = (r * (remaining_turns / ref)) / max(1.0, float(remaining_turns))
        if sum(p.saving for p in acted) * scale > overhead:
            wins += 1
    return wins / len(ladder)


def substitutes(
    plan: Plan,
    *,
    est_read_tokens: int,
    est_out_tokens: int = 800,
    context_tokens: int = 0,
    remaining_turns: int = 0,
    session_model: str | None = None,
    limit: int = 3,
    only_cheaper: bool = True,
    # None means "whatever the `harness` setting says", so a Codex or Gemini
    # CLI user gets their own placement rules without passing a flag on every
    # call. An explicit string still wins, for callers that know.
    harness: str | None = None,
    project: str | None = None,
    # The task text, for the neighbour-conditioned failure rate. Optional
    # because a caller holding only a `Plan` no longer has it, and the
    # tier-wide rate this used before is still a correct answer.
    task: str = "",
) -> list[Substitute]:
    """Other vendors' models that could run this delegation for less.

    Returns nothing unless the plan is a delegation -- see the module docstring
    for why a warm session is a different question. Four gates here:

    1. the substitute holds the task's context,
    2. it supports tool use,
    3. its arena rating is within this tier's tolerance of the Claude model,
    4. its *expected* cost, including escalation, beats the Claude model's.

    The fifth gate -- does the saving clear the routing overhead -- belongs to
    `Plan.render`, which owns that comparison for every other recommendation
    too. Keeping it there means a caller can still see a candidate that saves
    two cents and decide for itself, without this function pretending two cents
    is advice.

    Where `p_fail` comes from, and why it is two numbers rather than one
    --------------------------------------------------------------------
    The outcome log measures how often *this tier* escalates on this project.
    That is a real measurement, and it is about the tier -- not about the
    vendor being proposed. Using it alone would claim measured evidence for a
    model that has never run here; using the Elo gap alone would throw away the
    only real data in the building.

    So they compose as independent failure modes:

        p_fail = p_measured + (1 - p_measured) * p_elo_gap

    which is `select.blend_p_fail`, defined once and used by `adder pick` too.
    The task fails if it would have failed on the Claude tier anyway, or if it
    survives that and the substitute is weaker. With no recorded outcomes the
    first term drops out and this is the pure Elo estimate; with a tier that
    escalates constantly it stays high no matter how good the substitute looks.
    `Substitute.basis` records which happened.

    The catalog is imported here rather than at module scope so that `decide()`
    -- which runs in a prompt hook on every turn -- never pays to read it.
    """
    if plan.action != "delegate":
        return []
    try:
        from adder.decide.route.select import (
            UNUSABLE_GIVEN_LOSS,
            Need,
            blend_p_fail,
            calibrate_unusable_given_loss,
            cost_of,
            p_fail_from_elo,
            rank,
        )
        from adder.pricing.catalog import load
    except ImportError:                                   # pragma: no cover
        return []

    cat = load()
    baseline_entry = cat.get(plan.model)
    if baseline_entry is None or not baseline_entry.priced:
        return []

    tier = plan.tier
    difficulty = TIER_DIFFICULTY[tier]
    tolerance = SUBSTITUTE_TOLERANCE[tier]
    floor = baseline_entry.rating()
    if floor is None:
        # No rating for the model being replaced means no bar to clear. Offering
        # a substitute here would be comparing a price against nothing.
        return []
    floor -= tolerance

    need = Need(
        context_tokens=context_tokens or est_read_tokens,
        remaining_turns=remaining_turns,
        est_read_tokens=est_read_tokens,
        est_out_tokens=est_out_tokens,
        difficulty=difficulty,
        reference=plan.model,
        session_model=session_model or _settings.session_model(),
        harness=harness if harness is not None else _harness_default(),
    )
    # Compare the substitutable leg only. Both candidates return a summary that
    # the session model admits and carries at its own rate, so including that
    # term would apply the escalation multiplier to a cost neither model
    # controls -- and it is large enough to swamp the difference that the
    # choice is actually about.
    session_model = session_model or _settings.session_model()
    baseline = cost_of(baseline_entry, need, session=cat.get(session_model)).subagent

    # Measured escalation history for this tier, if there is enough of it to
    # act against a prior. `Evidence.informative` is what distinguishes "0.5
    # over 200 runs" from "0.5 because nothing was ever recorded".
    measured, basis = 0.0, "elo"
    ugl = UNUSABLE_GIVEN_LOSS
    ev = _evidence(tier, project, task)
    if ev is not None and ev.informative:
        measured, basis = ev.p_fail, f"elo + {ev.scope} history"
        # With history for a tier that has somewhere to escalate to, the
        # loss-to-redo prior stops being a guess: it is the ratio of what was
        # observed to the preference loss that should have produced it. Fitted
        # on the tier-vs-escalation-target gap and then applied to the
        # substitute-vs-tier gap, which assumes the ratio transfers between
        # gaps -- a stated assumption, and a smaller one than picking 0.35.
        target = cat.get(Tier.T2.model)
        if target is not None and tier < Tier.T2:
            ugl, why = calibrate_unusable_given_loss(
                baseline_entry, target, measured, difficulty=difficulty)
            basis = f"elo + {why}"

    stale_days = cat.age_days()
    stale_note = ""
    if cat.is_stale():
        stale_note = ("catalog is "
                      + (f"{stale_days:.0f} days old" if stale_days is not None
                         else "undated")
                      + "; prices and ratings may have moved (`adder models refresh`)")

    out: list[Substitute] = []
    for pick in rank(need, cat=cat, quality_floor=floor, limit=limit * 6):
        e = pick.entry
        if e.key == baseline_entry.key:
            continue
        direct = cost_of(e, need, session=cat.get(session_model)).subagent
        gap = p_fail_from_elo(e, baseline_entry, difficulty=difficulty,
                              unusable_given_loss=ugl)
        if gap is None:
            continue
        pf = blend_p_fail(measured, gap)
        expected = direct + pf * baseline
        if only_cheaper and expected >= baseline:
            continue
        out.append(Substitute(
            model=e.id, org=e.org, rating=e.rating(), p_fail=pf,
            direct=direct, baseline=baseline, expected=expected,
            open_weights=e.open_weights, verified=e.verified,
            reachable=("reachable as an MCP tool or an external call, not as a "
                       "Claude Code subagent; prices are unverified"
                       + (f". {stale_note}" if stale_note else "")),
            basis=basis,
        ))
        if len(out) >= limit:
            break
    out.sort(key=lambda s: s.expected)
    return out


def _record_to_ledger(plan: Plan, *, project: str = "", session: str = "") -> None:
    """Write one ledger entry for a recommendation as it is emitted.

    This is the moment the overhead is actually incurred and the prediction is
    actually made, so it is the only honest place to book both. Until something
    wrote here, `Ledger.haircut()` had nothing to correct with and `decide`'s
    over-promise correction was a branch that could never be taken -- the tool
    read its own record of whether it was worth using and always found the page
    blank.

    `accepted` is False when the gate declined to emit, which still costs the
    routing turn. Counting only the recommendations makes the tool look free
    every time it says "just do it inline", which is most of the time.

    Never raises. Accounting must not be able to break routing.
    """
    try:
        # `DEFAULT_LEDGER` is resolved per call, not captured as a default
        # argument, so a test can point the ledger somewhere harmless.
        from adder.decide.track.ledger import Entry, record

        g = plan.guarantee
        record(Entry(
            action=plan.action,
            predicted=plan.saving,
            worst=(g.worst if g is not None else min(0.0, plan.saving)),
            overhead=plan.overhead,
            accepted=plan.worth_it,
            project=project or "",
            session=session or "",
            note=f"{plan.tier.name} {plan.model} effort={plan.effort}",
        ))
    except Exception:
        pass


def _nonneg(raw: str) -> int:
    """Argparse type for a turn or token count. Rejects negatives at the flag.

    `adder policy "..." --remaining -5` used to reach `assess_placement` and
    raise `ValueError: interval out of order` as an unhandled traceback out of
    the CLI, while `--read-tokens -100` and `--context -1` were accepted
    silently and produced prices for them. None of the three means anything
    below zero; the difference in how they failed was an accident of which one
    happened to be used to build an interval.
    """
    import argparse

    try:
        n = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {raw!r}") from None
    if n < 0:
        raise argparse.ArgumentTypeError(
            f"must be zero or more, got {n}; turns and tokens do not run backwards")
    return n


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    import sys

    from adder.measure.session.horizon import load as load_horizon
    from adder.measure.session.live import analyse, current_session

    ap = argparse.ArgumentParser(prog="adder policy")
    ap.add_argument("task", nargs="*")
    ap.add_argument("--context", type=_nonneg, default=None)
    ap.add_argument("--remaining", type=_nonneg, default=None)
    ap.add_argument("--read-tokens", type=_nonneg, default=None)
    ap.add_argument("--project", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-cross-vendor", action="store_true",
                    help="skip the cross-vendor subagent comparison")
    ap.add_argument("--cross-vendor", action="store_true",
                    help="show cross-vendor subagents even when they save nothing")
    ap.add_argument("--min-confidence", type=float, default=DEFAULT_CONFIDENCE,
                    help="probability a recommendation must be cheaper before it "
                         "is emitted (default %(default)s)")
    ap.add_argument("--batch", action="store_true",
                    help="read one task per line from stdin and price them as a "
                         "single routing turn")
    ap.add_argument("--record", action="store_true",
                    help="append this recommendation to the ledger, so "
                         "`adder ledger` can later ask whether it paid for itself")
    ap.add_argument("--no-measure", action="store_true",
                    help="ignore local horizon, carry and ledger data; price at "
                         "the point estimates only")
    ap.add_argument("--harness", default=_harness.default(),
                    choices=_harness.names(),
                    help="agent runtime this advice is for; decides which "
                         "placements exist (default: %(default)s)")
    a = ap.parse_args(argv)

    ctx, rem, model, project = (a.context, a.remaining,
                                _settings.session_model(), a.project)
    turn_index = 0
    if ctx is None or rem is None:
        s = current_session()
        if s is not None:
            r = analyse(s)
            ctx = ctx if ctx is not None else r.context
            # The MEAN, not the median: every gate below multiplies a cost by
            # this, and carry cost is linear in remaining turns.
            rem = rem if rem is not None else round(r.carry_turns)
            model = r.model
            project = project or s.project
            turn_index = getattr(r, "turns", 0) or 0
    ctx = ctx if ctx is not None else 100_000

    # The three measured inputs. Each one is optional and each one is loaded
    # here rather than inside `decide`, so a library caller keeps a pure
    # function and the CLI gets the version that knows what it does not know.
    horizon = None if a.no_measure else load_horizon()
    carry = None
    ledger = None
    if not a.no_measure:
        try:
            from adder.core.trace import DEFAULT_ROOT, load_sessions
            from adder.decide.track.ledger import current as current_ledger
            from adder.measure.window.carry import Carry

            carry = Carry.measure(load_sessions(DEFAULT_ROOT))
            ledger = current_ledger()
        except Exception:
            carry = ledger = None
    if rem is None:
        # The MEAN, not the median: carry cost is linear in remaining turns, so
        # the expectation is what prices it. See `horizon.mean_remaining`.
        rem = int(horizon.mean_remaining(0)) if horizon else 450

    if a.batch:
        tasks = [line.strip() for line in sys.stdin if line.strip()]
        if not tasks:
            print("no tasks on stdin", file=sys.stderr)
            return 2
        b = schedule(tasks, context_tokens=ctx, remaining_turns=rem,
                     session_model=model, project=project, horizon=horizon,
                     carry=carry, ledger=ledger, turn_index=turn_index,
                     min_confidence=a.min_confidence)
        if a.json:
            print(json.dumps({
                "batch": True, "overhead": round(b.overhead, 4),
                "saving": round(b.saving, 4), "confidence": round(b.confidence, 3),
                "worth_it": b.worth_it,
                "plans": [{"action": p.action, "model": p.model,
                           "saving": round(p.saving, 4)} for p in b.plans],
            }))
        else:
            print(b.render())
        return 0

    p = decide(" ".join(a.task), context_tokens=ctx, remaining_turns=rem,
               session_model=model, est_read_tokens=a.read_tokens, project=project,
               horizon=horizon, carry=carry, ledger=ledger, turn_index=turn_index,
               min_confidence=a.min_confidence, harness=a.harness)

    if not a.no_cross_vendor:
        read = a.read_tokens or {Tier.T0: 8_000, Tier.T1: 20_000,
                                 Tier.T2: 60_000, Tier.T3: 120_000}[p.tier]
        try:
            p.substitutes = substitutes(
                p, est_read_tokens=read, context_tokens=ctx, remaining_turns=rem,
                session_model=model, only_cheaper=not a.cross_vendor,
                project=project, task=" ".join(a.task))
        except Exception:
            # A missing or unreadable catalog must never take down a routing
            # decision that does not depend on it.
            p.substitutes = []

    if a.record:
        _record_to_ledger(p, project=project, session=getattr(
            current_session(), "id", "") or "")

    if a.json:
        print(json.dumps({
            "action": p.action, "model": p.model, "effort": p.effort,
            "agent": p.agent, "saving": round(p.saving, 4),
            "overhead": round(p.overhead, 4), "worth_it": p.worth_it,
            "confidence": p.confidence, "p_fail": round(p.p_fail, 3),
            "guarantee": None if p.guarantee is None else {
                "expected": round(p.guarantee.expected, 5),
                "worst": round(p.guarantee.worst, 5),
                "confidence": round(p.guarantee.confidence, 3),
                "safe": p.guarantee.safe, "dominant": p.guarantee.dominant,
                "corner": {k: round(v, 4) for k, v in p.guarantee.corner.items()},
            },
            "reasons": p.reasons, "warnings": p.warnings,
            "context_tokens": ctx, "remaining_turns": rem,
            "ladder": [{
                "tier": r.tier.name, "model": r.model, "effort": r.effort,
                "feasible": r.feasible, "allowed": r.allowed,
                "p_fail": round(r.p_fail, 3), "run": round(r.run, 5),
                "expected": round(r.expected, 5), "note": r.note,
            } for r in p.ladder],
            "substitutes": [{
                "model": s.model, "org": s.org, "rating": s.rating,
                "p_fail": round(s.p_fail, 3), "direct": round(s.direct, 4),
                "baseline": round(s.baseline, 4), "expected": round(s.expected, 4),
                "saving": round(s.saving, 4), "open_weights": s.open_weights,
                "verified": s.verified, "basis": s.basis,
            } for s in p.substitutes],
        }))
    else:
        print(p.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
