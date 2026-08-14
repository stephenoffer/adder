"""Pick a model, or a combination of models, from public data.

What this adds to `policy.py`
-----------------------------
`policy.py` answers *where* work should run -- inline or delegated -- across a
fixed ladder of three Claude tiers. That ladder is hand-written, which means it
is wrong within a week of any launch, and it cannot see a cheaper open-weight
model that would do the job.

This module answers *what* should run it, over whatever the catalog currently
knows: ~500 models from every major lab, priced, with arena ratings. The ladder
becomes a query result instead of a constant.

The three things that make this harder than sorting by price
------------------------------------------------------------
**1. Quality is a proxy, and it is the weakest link here.** Arena Elo measures
human preference on chat and web-dev prompts. It does not measure multi-file
agentic tool use, which is what these sessions actually do. It is used anyway
because it is the only cross-vendor signal that updates within days of a launch
and is not self-reported. Everything derived from it is labelled MODELLED, and
`outcomes.p_fail` overrides it the moment there is measured history for a tier.

**2. The dominant cost term is not the price of the task.** It is the price of
carrying the task's tokens for the rest of the session. A model 5x cheaper per
token that pulls 3x more into the main context is more expensive by turn forty.
So candidates are costed through the session model, not a per-request one.

**3. Cache economics are per-provider and non-transferable.** Anthropic charges
0.10x to read a cached prefix and 1.25x to write it. Others differ, and some
publish nothing. Where the catalog has absolute cache rates, they are used;
where it does not, the entry is flagged rather than assumed, because a
missing cache read rate is the difference between a real saving and a fantasy.

Combinations
------------
The cheapest way to hit a quality bar is often not one model. Three shapes are
priced here, each with its own failure mode:

* **cascade** — cheap model first, strong model on failure. Wins when p_fail is
  low. Loses when failure is not *detected*: an undetected bad answer costs the
  cheap run plus whatever the bad answer breaks downstream.
* **draft-review** — cheap model drafts, strong model reviews a compressed
  diff. Wins when review is much cheaper than generation. Loses when the
  reviewer needs the same context the drafter had.
* **panel** — N cheap runs, one judge picks. Wins on tasks with a verifiable
  answer. Loses on tasks where N wrong answers agree.

Each is reported with the *assumption that decides it* -- detection rate,
compression, agreement -- because those assumptions, not the prices, are where
the recommendation actually comes from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .catalog import Catalog, Entry, load

M = 1_000_000.0

# Anthropic's multipliers, used ONLY as a labelled fallback when a provider
# publishes no cache rates. Applying these to another vendor is a guess.
FALLBACK_CACHE_READ_MULT = 0.10
FALLBACK_CACHE_WRITE_MULT = 1.25

# Elo -> win probability is the standard logistic with a 400-point scale.
ELO_SCALE = 400.0

# Share of preference losses that are bad enough to need the work redone.
#
# This constant exists because Elo does not measure what we need. A model that
# loses 50% of head-to-head comparisons to the frontier does not fail 50% of
# tasks -- most losses are "the other answer was nicer", not "this answer was
# wrong". Collapsing those two into one number is the mistake that makes
# Elo-based routers recommend the frontier model for everything.
#
# So the estimate is decomposed: a public, measured quantity (preference loss,
# from arena Elo) times one named prior (how often a loss is disqualifying).
# The prior is MODELLED and deliberately visible. `outcomes.p_fail` replaces
# the whole thing with measured retry history as soon as there is any.
UNUSABLE_GIVEN_LOSS = 0.35

# Fraction of a weak model's failures that the session actually notices in time
# to escalate. MODELLED: the measured value needs an outcome log with labelled
# retries, which `outcomes.py` collects.
DEFAULT_DETECTION = 0.8

# How much smaller a review pass is than the generation it reviews.
DEFAULT_REVIEW_COMPRESSION = 6.0


def win_probability(rating_a: float, rating_b: float) -> float:
    """P(a is preferred over b), Bradley-Terry on the arena scale."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / ELO_SCALE))


def p_loss_from_elo(candidate: Entry, reference: Entry,
                    *, difficulty: float = 1.0) -> float | None:
    """P(a human prefers the reference's answer), from arena ratings.

    Anchored on the reference rather than an absolute scale, because Elo has no
    absolute meaning: 1450 is only interpretable next to another number.
    `difficulty` stretches the gap -- on a hard task a 50-point deficit matters
    more than on an easy one.
    """
    ra, rb = candidate.rating(), reference.rating()
    if ra is None or rb is None:
        return None
    p_ok = win_probability(ra, rb) ** max(0.1, difficulty)
    return max(0.0, min(1.0, 1.0 - p_ok))


def p_fail_from_elo(candidate: Entry, reference: Entry, *, difficulty: float = 1.0,
                    unusable_given_loss: float = UNUSABLE_GIVEN_LOSS) -> float | None:
    """Modelled probability the answer has to be redone on the strong model.

    Preference loss times the share of losses that are disqualifying. See
    `UNUSABLE_GIVEN_LOSS` for why those are two separate numbers.
    """
    p_loss = p_loss_from_elo(candidate, reference, difficulty=difficulty)
    return None if p_loss is None else p_loss * unusable_given_loss


@dataclass
class Need:
    """What the task requires, and what the session it runs in looks like."""

    context_tokens: int = 100_000       # main-session context right now
    remaining_turns: int = 100          # how long the session still has to run
    est_read_tokens: int = 40_000       # what the task pulls in
    est_out_tokens: int = 1_200
    summary_tokens: int = 0             # what comes back to the main context
    needs_tools: bool = True
    needs_vision: bool = False
    open_weights_only: bool = False
    allow_unverified: bool = True       # trust aggregator prices
    difficulty: float = 1.0
    reference: str = "claude-opus-5"    # the "do it properly" model
    session_model: str = "claude-opus-5"
    # Which harness runs this. Under `claude-code` the main conversation is a
    # Claude model by construction: a GPT or open-weight model can be a
    # subagent, an MCP tool, or an external call, but it cannot *be* the
    # session. Quoting an inline price for one is quoting a placement that
    # does not exist. `any` relaxes this for harnesses that route natively.
    harness: str = "claude-code"

    def __post_init__(self) -> None:
        if not self.summary_tokens:
            self.summary_tokens = max(200, self.est_read_tokens // 10)


@dataclass
class Costed:
    """One placement of one model, priced with its assumptions attached."""

    entry: Entry
    inline: float                # cost if the task runs in the main context
    delegated: float             # cost if it runs in a subagent
    carry: float                 # the part of `inline` that is future turns
    assumed_cache: bool = False  # cache rates were guessed, not published
    inline_feasible: bool = True  # can this model hold the whole session?
    inline_blocked: str = ""      # why not, if not

    @property
    def best(self) -> float:
        """Cheapest *available* placement.

        Not `min(inline, delegated)`: a model whose window cannot hold the
        session cannot run inline at any price, and quoting the inline number
        for it is quoting the cost of a 400.
        """
        return min(self.inline, self.delegated) if self.inline_feasible else self.delegated

    @property
    def placement(self) -> str:
        if not self.inline_feasible:
            return "delegate"
        return "delegate" if self.delegated < self.inline else "inline"


def _cache_rates(e: Entry) -> tuple[float, float, bool]:
    """(read, write) USD per Mtok, and whether they had to be assumed."""
    if e.inp is None:
        return 0.0, 0.0, True
    read = e.cache_read
    write = e.cache_write
    assumed = read is None or write is None
    if read is None:
        read = e.inp * FALLBACK_CACHE_READ_MULT
    if write is None:
        write = e.inp * FALLBACK_CACHE_WRITE_MULT
    return read, write, assumed


def cost_of(e: Entry, need: Need, *, session: Entry | None = None) -> Costed:
    """Price one model two ways: inline in the session, or in a subagent.

    Inline is not `read * rate`. Tokens admitted to the main context are read
    again on every remaining turn, so the carry term is `read * cache_read *
    remaining_turns` -- usually several times the one-off cost, and the reason
    a cheap model can be the expensive choice.

    Delegation pays the read at the subagent's rate once, then admits only the
    summary to the main context, where the session model carries it.
    """
    session = session or e
    s_read, s_write, s_assumed = _cache_rates(session)
    e_read, e_write, e_assumed = _cache_rates(e)
    inp = e.inp or 0.0
    out = e.out or 0.0

    # Inline: the task's tokens land in the main context and stay there.
    # A model switch mid-session also rebuilds the existing prefix, since the
    # prompt cache is model-scoped -- that is the `switch` term.
    switch = 0.0
    if e.key != session.key:
        switch = need.context_tokens * e_write / M
    admit = need.est_read_tokens * e_write / M
    carry = need.est_read_tokens * e_read * need.remaining_turns / M
    generate = need.est_out_tokens * out / M
    out_carry = need.est_out_tokens * e_read * need.remaining_turns / M
    inline = switch + admit + carry + generate + out_carry

    # Delegated: subagent reads at its own rate, main context takes a summary.
    sub = (need.est_read_tokens * inp + need.est_out_tokens * out) / M
    summary_admit = need.summary_tokens * s_write / M
    summary_carry = need.summary_tokens * s_read * need.remaining_turns / M
    delegated = sub + summary_admit + summary_carry

    feasible = e.context is None or e.context >= need.context_tokens
    blocked = ""
    if not feasible:
        blocked = (f"context window {e.context:,} < session context "
                   f"{need.context_tokens:,}")
    elif need.harness == "claude-code" and e.org.lower() != "anthropic":
        feasible = False
        blocked = (f"{e.org or 'this vendor'} cannot be the main Claude Code "
                   "session; reachable as a subagent or tool call only")
    return Costed(e, inline=inline, delegated=delegated, carry=carry,
                  assumed_cache=e_assumed or s_assumed, inline_feasible=feasible,
                  inline_blocked=blocked)


@dataclass
class Pick:
    """A ranked candidate, with everything needed to argue with it."""

    entry: Entry
    cost: float
    placement: str
    rating: float | None
    p_fail: float | None
    p_loss: float | None = None
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def value(self) -> float:
        """Rating points per dollar. Only comparable within one ranking."""
        if self.rating is None or self.cost <= 0:
            return 0.0
        return self.rating / self.cost


def _eligible(cat: Catalog, need: Need) -> list[Entry]:
    """Candidates that clear the hard gates.

    The reference model is not force-included. An `--open-weights` query that
    quietly returns a proprietary model because it happens to be the yardstick
    is answering a different question than the one asked; the reference is
    reported separately, as a comparison, not as a pick.
    """
    ctx_needed = need.est_read_tokens + need.est_out_tokens + 8_000
    out = []
    for e in cat.find(
        needs_tools=need.needs_tools,
        min_context=ctx_needed,
        priced_only=True,
        rated_only=False,
        open_weights=True if need.open_weights_only else None,
        verified_only=not need.allow_unverified,
    ):
        if need.needs_vision and "image" not in e.modalities:
            continue
        out.append(e)
    return out


def rank(
    need: Need,
    *,
    cat: Catalog | None = None,
    quality_floor: float | None = None,
    include_unrated: bool = False,
    limit: int = 10,
) -> list[Pick]:
    """Cheapest models that clear the quality bar, in the session's economics.

    `quality_floor` is an Elo, absolute. Passing None derives one from the
    reference model and the task difficulty: an easy task tolerates a 150-point
    deficit, a hard one about 40.

    Unrated models are excluded by default. This is the single most important
    default in the module: the catalog holds hundreds of models nobody has
    benchmarked, they are disproportionately cheap, and treating "no rating" as
    "passes the bar" makes the ranking a list of the cheapest unknown things on
    the internet. `include_unrated=True` opts back in, and every such row is
    flagged.
    """
    cat = cat or load()
    ref = cat.get(need.reference)
    if ref is None:
        raise KeyError(f"reference model {need.reference!r} is not in the catalog")
    session = cat.get(need.session_model) or ref

    if quality_floor is None:
        r = ref.rating()
        quality_floor = None if r is None else r - (150.0 / max(0.25, need.difficulty))

    picks: list[Pick] = []
    for e in _eligible(cat, need):
        c = cost_of(e, need, session=session)
        rating = e.rating()
        reasons, warnings = [], []

        if rating is None:
            if not include_unrated:
                continue
            warnings.append("no arena rating; quality is unknown, not assumed equal")
        elif quality_floor is not None and rating < quality_floor:
            continue

        if not e.verified:
            warnings.append("price is from an aggregator, not a first-party list")
        if c.assumed_cache:
            warnings.append(
                "provider publishes no cache rates; carried cost assumes "
                "Anthropic-style 0.10x reads and is a guess")
        if not c.inline_feasible and c.inline_blocked:
            reasons.append(f"subagent only: {c.inline_blocked}")

        pf = p_fail_from_elo(e, ref, difficulty=need.difficulty)
        pl = p_loss_from_elo(e, ref, difficulty=need.difficulty)
        if c.placement == "delegate":
            reasons.append(
                f"delegating keeps {need.est_read_tokens:,} tok out of a context "
                f"re-read {need.remaining_turns} more times")
        else:
            reasons.append(f"carry term is ${c.carry:,.3f} of ${c.inline:,.3f}")
        picks.append(Pick(e, c.best, c.placement, rating, pf, pl, reasons, warnings))

    picks.sort(key=lambda p: (p.cost, -(p.rating or 0)))
    return picks[:limit]


# --------------------------------------------------------------------------
# Combinations
# --------------------------------------------------------------------------

@dataclass
class Combo:
    """A multi-model plan, its expected cost, and the assumption that decides it."""

    shape: str                   # single | cascade | draft-review | panel
    models: list[str]
    expected_cost: float
    quality: float | None        # effective rating, MODELLED
    assumption: str
    detail: str = ""

    def render(self) -> str:
        q = f"{self.quality:,.0f}" if self.quality is not None else "  ?  "
        head = f"{self.shape:<12} ${self.expected_cost:>8,.3f}  elo~{q}  {' + '.join(self.models)}"
        return head + (f"\n{' ' * 14}{self.detail}" if self.detail else "")


def combos(
    need: Need,
    *,
    cat: Catalog | None = None,
    cheap: str | None = None,
    strong: str | None = None,
    detection: float = DEFAULT_DETECTION,
    review_compression: float = DEFAULT_REVIEW_COMPRESSION,
    panel_n: int = 3,
    measured_p_fail: float | None = None,
) -> list[Combo]:
    """Price the standard multi-model shapes against doing it once, properly.

    Ranked by expected cost. Quality is the modelled effective rating: for a
    cascade that is the strong model's rating discounted by the failures
    detection misses, not the strong model's rating outright, because an
    undetected failure is exactly the case where the cascade did not work.
    """
    cat = cat or load()
    ref = cat.get(strong or need.reference)
    if ref is None:
        raise KeyError(f"reference model {strong or need.reference!r} not in catalog")
    session = cat.get(need.session_model) or ref

    if cheap:
        low = cat.get(cheap)
        if low is None:
            raise KeyError(f"model {cheap!r} not in catalog")
    else:
        cands = rank(need, cat=cat, quality_floor=None, limit=1)
        low = cands[0].entry if cands else ref

    c_low = cost_of(low, need, session=session)
    c_ref = cost_of(ref, need, session=session)
    r_low, r_ref = low.rating(), ref.rating()

    pf = measured_p_fail
    if pf is None:
        pf = p_fail_from_elo(low, ref, difficulty=need.difficulty)
    if pf is None:
        pf = 0.5

    out: list[Combo] = [
        Combo("single", [ref.id], c_ref.best, r_ref,
              "no assumption beyond the price table",
              f"one pass on the reference model, placed {c_ref.placement}"),
    ]
    if low.key != ref.key:
        out.append(Combo(
            "single", [low.id], c_low.best, r_low,
            "cheap model is good enough on its own",
            f"one pass, placed {c_low.placement}; modelled p_fail {pf:.0%}"))

        # Cascade: pay cheap always, strong on the failures we notice.
        casc = c_low.best + pf * detection * c_ref.best
        missed = pf * (1 - detection)
        q_casc = None
        if r_low is not None and r_ref is not None:
            q_casc = r_ref - missed * max(0.0, r_ref - r_low)
        out.append(Combo(
            "cascade", [low.id, ref.id], casc, q_casc,
            f"{detection:.0%} of failures are detected in time to escalate",
            f"p_fail {pf:.0%}; {missed:.0%} of runs ship the weaker answer unnoticed"))

        # Draft-review: strong model sees a compressed diff, not the whole task.
        review_need = Need(
            context_tokens=need.context_tokens,
            remaining_turns=need.remaining_turns,
            est_read_tokens=max(1_000, int(need.est_read_tokens / review_compression)),
            est_out_tokens=max(200, int(need.est_out_tokens / 3)),
            needs_tools=need.needs_tools, session_model=need.session_model,
        )
        dr = c_low.best + cost_of(ref, review_need, session=session).best
        q_dr = None
        if r_low is not None and r_ref is not None:
            q_dr = r_low + 0.6 * max(0.0, r_ref - r_low)
        out.append(Combo(
            "draft-review", [low.id, ref.id], dr, q_dr,
            f"a review reads {review_compression:.0f}x less than the generation did",
            "reviewer never sees the context the drafter explored; catches "
            "wrong answers, not missing ones"))

        # Panel: N cheap runs plus a judge. Only sane with a verifiable answer.
        judge_need = Need(
            context_tokens=need.context_tokens,
            remaining_turns=need.remaining_turns,
            est_read_tokens=panel_n * max(500, need.est_out_tokens),
            est_out_tokens=max(200, need.est_out_tokens // 3),
            needs_tools=False, session_model=need.session_model,
        )
        panel = panel_n * c_low.best + cost_of(ref, judge_need, session=session).best
        q_panel = None
        if r_low is not None:
            # Best-of-N against an independent-errors model: the effective win
            # rate rises, but only if the N runs fail independently. They do not.
            q_panel = (r_low + ELO_SCALE * math.log10(max(1e-9, 1 - pf ** panel_n)
                                                      / max(1e-9, pf ** panel_n)) * 0.15
                       if 0 < pf < 1 else r_low)
            if r_ref is not None:
                q_panel = min(q_panel, r_ref)
        out.append(Combo(
            "panel", [f"{panel_n}x {low.id}", ref.id], panel, q_panel,
            f"{panel_n} runs fail independently -- they usually do not",
            "only sound when the answer is checkable; N agreeing wrong answers "
            "look like consensus"))

    out.sort(key=lambda c: c.expected_cost)
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        prog="adder.select",
        description="pick a model, or a combination, from the catalog")
    ap.add_argument("task", nargs="*", help="task text (used to size the work)")
    ap.add_argument("--context", type=int, default=None, help="session context tokens")
    ap.add_argument("--remaining", type=int, default=None, help="remaining turns")
    ap.add_argument("--read-tokens", type=int, default=None)
    ap.add_argument("--out-tokens", type=int, default=None)
    ap.add_argument("--reference", default="claude-opus-5",
                    help="the 'do it properly' model to compare against")
    ap.add_argument("--session-model", default=None)
    ap.add_argument("--cheap", default=None, help="pin the cheap leg of a combination")
    ap.add_argument("--open-weights", action="store_true")
    ap.add_argument("--harness", default="claude-code", choices=("claude-code", "any"),
                    help="`claude-code` keeps the main session on a Claude model")
    ap.add_argument("--no-tools", action="store_true",
                    help="drop the tool-use requirement")
    ap.add_argument("--difficulty", type=float, default=1.0)
    ap.add_argument("--floor", type=float, default=None, help="minimum arena rating")
    ap.add_argument("--include-unrated", action="store_true",
                    help="also rank models with no arena rating (off by default)")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--combos", action="store_true", help="rank multi-model plans")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    from .classify import Tier, classify

    ctx, rem, smodel = a.context, a.remaining, a.session_model
    if ctx is None or rem is None or smodel is None:
        try:
            from .live import analyse, current_session
            s = current_session()
            if s is not None:
                r = analyse(s)
                ctx = ctx if ctx is not None else r.context
                rem = rem if rem is not None else r.projected_remaining
                smodel = smodel or r.model
        except Exception:
            pass
    ctx = ctx if ctx is not None else 100_000
    rem = rem if rem is not None else 100
    smodel = smodel or "claude-opus-5"

    task = " ".join(a.task)
    v = classify(task) if task else None
    read = a.read_tokens
    if read is None:
        read = ({Tier.T0: 8_000, Tier.T1: 20_000, Tier.T2: 60_000,
                 Tier.T3: 120_000}[v.tier] if v else 40_000)
    difficulty = a.difficulty
    if v is not None and a.difficulty == 1.0:
        difficulty = {0: 0.4, 1: 0.7, 2: 1.0, 3: 1.4}[int(v.tier)]

    need = Need(
        context_tokens=ctx, remaining_turns=rem, est_read_tokens=read,
        est_out_tokens=a.out_tokens or 1_200,
        needs_tools=not a.no_tools, open_weights_only=a.open_weights,
        difficulty=difficulty, reference=a.reference, session_model=smodel,
        harness=a.harness,
    )

    cat = load()
    stale = cat.age_days()
    if a.combos:
        plans = combos(need, cat=cat, cheap=a.cheap)
        if a.json:
            print(json.dumps([{
                "shape": c.shape, "models": c.models,
                "expected_cost": round(c.expected_cost, 4),
                "quality": c.quality, "assumption": c.assumption,
                "detail": c.detail} for c in plans], indent=1))
            return 0
        print(f"context {ctx:,} tok  |  {rem} turns left  |  difficulty {difficulty:.1f}")
        print()
        for c in plans:
            print(c.render())
            print(f"{' ' * 14}assumes: {c.assumption}")
        print("\nExpected cost is MODELLED. Quality is arena Elo, which measures")
        print("human preference on chat and web-dev, not agentic tool use.")
        return 0

    picks = rank(need, cat=cat, quality_floor=a.floor, limit=a.limit,
                 include_unrated=a.include_unrated)
    if a.json:
        print(json.dumps([{
            "id": p.entry.id, "org": p.entry.org, "cost": round(p.cost, 4),
            "placement": p.placement, "rating": p.rating,
            "p_fail": None if p.p_fail is None else round(p.p_fail, 3),
            "open_weights": p.entry.open_weights, "verified": p.entry.verified,
            "context": p.entry.context, "warnings": p.warnings,
        } for p in picks], indent=1))
        return 0

    if not picks:
        print("no model in the catalog clears the gates.")
        print("try --no-tools, a lower --floor, or `adder models refresh`.")
        return 1
    print(f"context {ctx:,} tok  |  {rem} turns left  |  "
          f"reads ~{read:,} tok  |  vs {a.reference}")
    if stale is not None and stale > 21:
        print(f"! catalog is {stale:.0f} days old; run `adder models refresh`")
    print()
    ref_entry = cat.get(a.reference)
    if ref_entry is not None:
        rc = cost_of(ref_entry, need, session=cat.get(smodel) or ref_entry)
        rr = ref_entry.rating()
        print(f"{'reference: ' + ref_entry.id:<38} {rc.best:>9,.3f} {rc.placement:<9} "
              f"{(f'{rr:,.0f}' if rr else '-'):>6}")
        print()
    print(f"{'model':<38} {'$/task':>9} {'where':<9} {'elo':>6} {'p_loss':>7} {'p_fail':>7}")
    for p in picks:
        elo = f"{p.rating:,.0f}" if p.rating else "-"
        pf = f"{p.p_fail:.0%}" if p.p_fail is not None else "-"
        pl = f"{p.p_loss:.0%}" if p.p_loss is not None else "-"
        print(f"{p.entry.id[:38]:<38} {p.cost:>9,.3f} {p.placement:<9} "
              f"{elo:>6} {pl:>7} {pf:>7}")
    print()
    print("p_loss: modelled share of head-to-head comparisons the reference wins.")
    print("p_fail: the part of that which needs redoing -- see UNUSABLE_GIVEN_LOSS.")
    seen = set()
    for p in picks:
        for w in p.warnings:
            if w not in seen:
                seen.add(w)
                print(f"  ! {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
