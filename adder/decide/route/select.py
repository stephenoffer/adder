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
0.10x to read a cached prefix and 1.25x to write it. OpenAI and Google cache
automatically and charge no write premium at all. Plenty of hosted open-weight
endpoints do not cache, so every re-read is full price. Where the catalog has
absolute cache rates they are used; where it does not, `providers.py` supplies
that vendor's shape and the entry is flagged as an estimate. What is never done
is borrowing one vendor's discount for another: since the carry term is
`cache_read * remaining_turns`, assuming a 0.10x read on a model that has no
cache is not a slightly optimistic estimate, it is an order-of-magnitude error
pointing at the model it should be pointing away from.

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

from dataclasses import dataclass, field, replace

from adder.core import harness as _harness
from adder.core import settings as _settings
from adder.pricing import providers
from adder.pricing.catalog import Catalog, Entry, load
from adder.pricing.cost import Rates, admitted_cost

M = 1_000_000.0

# Anthropic's multipliers, used ONLY as a labelled fallback when a provider
# publishes no cache rates. Applying these to another vendor is a guess.
# Kept as the documented shape of Anthropic's pricing, and as the value the
# fallback used to be for every vendor. `_cache_rates` now asks the provider
# table instead; these remain so the docstring above has something concrete to
# point at and so a reader can see what changed.
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


def ratings_overlap(a: Entry, b: Entry) -> bool:
    """Do the arena's published intervals for these two ratings overlap?

    This matters more than it looks. At the top of the webdev board the 95%
    half-width is around 10 points, so the 17-point "lead" of the first model
    over the second is two overlapping intervals -- a difference the arena
    itself does not claim. Treating it as real and deriving a 52% preference
    loss from it is inventing precision the data does not have.
    """
    ia, ib = a.rating_interval(), b.rating_interval()
    if ia is None or ib is None:
        return False
    return ia[0] <= ib[1] and ib[0] <= ia[1]


def p_loss_from_elo(candidate: Entry, reference: Entry, *, difficulty: float = 1.0,
                    conservative: bool = True) -> float | None:
    """P(a human prefers the reference's answer), from arena ratings.

    Anchored on the reference rather than an absolute scale, because Elo has no
    absolute meaning: 1450 is only interpretable next to another number.
    `difficulty` stretches the gap -- on a hard task a 50-point deficit matters
    more than on an easy one.

    `conservative` compares the candidate at the bottom of its published
    interval against the reference at the top of its own. When the intervals
    overlap the sign of the difference is unresolved, and this repo's rule for
    an unresolved routing question is to route up: the bound never claims a
    substitute is closer to the reference than the evidence supports. On
    current data it widens the estimate by a few points, not by a lot -- which
    is the honest size of the correction, and worth saying rather than
    dramatising.
    """
    ra, rb = candidate.rating(), reference.rating()
    if ra is None or rb is None:
        return None
    if conservative:
        ia, ib = candidate.rating_interval(), reference.rating_interval()
        if ia is not None:
            ra = ia[0]
        if ib is not None:
            rb = ib[1]
    p_ok = win_probability(ra, rb) ** max(0.1, difficulty)
    return max(0.0, min(1.0, 1.0 - p_ok))


def p_fail_from_elo(candidate: Entry, reference: Entry, *, difficulty: float = 1.0,
                    unusable_given_loss: float = UNUSABLE_GIVEN_LOSS,
                    conservative: bool = True) -> float | None:
    """Modelled probability the answer has to be redone on the strong model.

    Preference loss times the share of losses that are disqualifying. See
    `UNUSABLE_GIVEN_LOSS` for why those are two separate numbers.
    """
    p_loss = p_loss_from_elo(candidate, reference, difficulty=difficulty,
                             conservative=conservative)
    return None if p_loss is None else p_loss * unusable_given_loss


# The range the prior is plausibly wrong across. Used by `sensitivity()` to
# answer the only question that matters about an invented constant: does the
# recommendation depend on it?
UNUSABLE_RANGE = (0.15, 0.60)


def calibrate_unusable_given_loss(
    cheap: Entry, reference: Entry, measured_p_fail: float, *,
    difficulty: float = 1.0,
) -> tuple[float, str]:
    """Fit the prior from measured escalation instead of asserting it.

    `UNUSABLE_GIVEN_LOSS` is the weakest link in this module: it converts a
    preference loss into a redo, it scales every cascade cost and every
    substitute verdict linearly, and it was chosen by judgement. But it is not
    unmeasurable. The outcome log records how often a tier actually escalated,
    and the arena says how often that tier's model loses a comparison to the
    escalation target. The ratio of those two *is* the constant:

        unusable_given_loss = measured_escalation_rate / modelled_preference_loss

    So wherever there is enough history, the number stops being a prior and
    becomes a fit with a sample size attached. Where there is not, the prior
    stands and says so. Note this is only defined when the tier's model differs
    from the escalation target -- the top tier has nothing to escalate to, and
    a fit against a zero gap is a division, not a measurement.
    """
    if cheap.key == reference.key:
        # A model compared with itself is not a degenerate gap that produces
        # zero -- Bradley-Terry gives 0.5, a coin flip -- so this has to be
        # rejected by identity rather than by a threshold. Fitting here would
        # silently divide the observed escalation rate by 0.5 and report the
        # result as measured.
        return UNUSABLE_GIVEN_LOSS, "prior (nothing to escalate to from this tier)"
    p_loss = p_loss_from_elo(cheap, reference, difficulty=difficulty,
                             conservative=False)
    if p_loss is None or p_loss < 0.02:
        return UNUSABLE_GIVEN_LOSS, "prior (no resolvable rating gap to fit against)"
    if ratings_overlap(cheap, reference):
        # The gap the fit divides by is inside the arena's own error bars, so
        # the quotient is noise amplified by a division.
        return UNUSABLE_GIVEN_LOSS, "prior (rating gap is inside the arena's error bars)"
    fitted = max(0.01, min(1.0, measured_p_fail / p_loss))
    return fitted, f"fitted from measured escalation ({measured_p_fail:.0%} observed)"


def blend_p_fail(measured: float, elo_gap: float) -> float:
    """Compose a measured tier escalation rate with an Elo-derived deficit.

    The outcome log knows how often a *tier* escalates here. The arena knows
    how much weaker a *substitute* is than the model that tier names. Neither
    answers the question on its own, and picking one throws away the other, so
    they compose as independent failure modes:

        p = measured + (1 - measured) * elo_gap

    The work fails if it would have failed on the Claude tier anyway, or if it
    survives that and the substitute is the weaker model. With no history the
    first term is zero and this is the pure Elo estimate; with a tier that
    escalates constantly it stays high however good the substitute looks.
    """
    measured = max(0.0, min(1.0, measured))
    elo_gap = max(0.0, min(1.0, elo_gap))
    return measured + (1.0 - measured) * elo_gap


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
    # The "do it properly" model, and the model the session itself runs on.
    # Factories rather than literals: a Codex or Gemini CLI user who set the
    # `model` setting was still getting Opus quoted as their reference, which
    # measures the gap to a placement they do not have.
    reference: str = field(default_factory=_settings.session_model)
    session_model: str = field(default_factory=_settings.session_model)
    # Which agent runtime this is for. Some harnesses pin the main
    # conversation to one vendor by construction -- Claude Code to Anthropic,
    # Codex to OpenAI, Gemini CLI to Google. Under those, a model from another
    # vendor can be a subagent, an MCP tool, or an external call, but it cannot
    # *be* the session, and quoting an inline price for one is quoting a
    # placement that does not exist. `adder.core.harness` holds the rule; this
    # is only the name of which one to apply, so the constraint stays data.
    #
    # A factory, not a literal, for the same reason `reference` and
    # `session_model` are: a Codex or Gemini CLI user who set `ADDER_HARNESS`
    # still had Claude Code's placement rules applied to every candidate this
    # dataclass was built without an explicit harness for.
    harness: str = field(default_factory=_harness.default)
    # Correction on the published cache-read rate, as a ratio.
    #
    # A cached prefix is only read at the published rate while it stays warm.
    # Real turns miss -- the TTL expires while you read a diff, a tool result
    # lands past the breakpoint lookback, a fan-out races the first write --
    # and `adder carry` measures the realised multiplier from the transcripts
    # (0.115x against an assumed 0.10x on this machine, so the carry term was
    # under-priced by about 15%).
    #
    # Applied as a ratio, not a rate, on purpose: *how often* a session misses
    # is a property of the workload and transfers across vendors; *what a read
    # costs* is the vendor's published number and does not. 1.0 changes nothing.
    cache_miss_correction: float = 1.0
    # How many times an admitted token is actually re-read before the session
    # ends. `None` means "every remaining turn", the uncorrected assumption. A
    # fitted `carry.Carry` discounts it by the token's chance of surviving each
    # compaction; on this machine no compaction lands inside a typical horizon,
    # so the two agree -- but the plumbing has to exist for the workloads where
    # they do not.
    expected_reads: float | None = None

    def __post_init__(self) -> None:
        if not self.summary_tokens:
            self.summary_tokens = max(200, self.est_read_tokens // 10)


@dataclass
class Costed:
    """One placement of one model, priced with its assumptions attached."""

    entry: Entry
    inline: float                # cost if the task runs in the main context
    delegated: float             # cost if it runs in a subagent
    subagent: float              # the part of `delegated` this model actually does
    carry: float                 # the part of `inline` that is future turns
    assumed_cache: bool = False  # cache rates were guessed, not published
    inline_feasible: bool = True  # can this model hold the whole session?
    inline_blocked: str = ""      # why not, if not
    # Can this harness hand work to a throwaway context at all? `harness.py`
    # carries the answer and calls it the reason the module exists -- "a report
    # that recommends it anyway is recommending a feature the user does not
    # have" -- and then nothing read the field. Aider is one conversation: it
    # has no subagent to delegate to, and every delegated price quoted to an
    # aider user was a placement that does not exist.
    delegate_feasible: bool = True
    delegate_blocked: str = ""

    @property
    def best(self) -> float:
        """Cheapest *available* placement.

        Not `min(inline, delegated)`: a model whose window cannot hold the
        session cannot run inline at any price, and quoting the inline number
        for it is quoting the cost of a 400. The same applies to the other
        placement -- a harness with no subagents cannot delegate at any price.
        """
        if not self.usable:
            # Neither placement exists. Quoting the cheaper of two impossible
            # numbers is how an unusable model reaches the top of a ranking.
            return float("inf")
        if not self.inline_feasible:
            return self.delegated
        if not self.delegate_feasible:
            return self.inline
        return min(self.inline, self.delegated)

    @property
    def placement(self) -> str:
        if not self.inline_feasible:
            return "delegate"
        if not self.delegate_feasible:
            return "inline"
        return "delegate" if self.delegated < self.inline else "inline"

    @property
    def usable(self) -> bool:
        """Is there any placement for this model here at all?"""
        return self.inline_feasible or self.delegate_feasible


class UnpricedEntryError(ValueError):
    """Raised when a cost is asked for from a model nobody published a price for."""


def _cache_rates(e: Entry) -> tuple[float, float, bool]:
    """(read, write) USD per Mtok, and whether they had to be assumed.

    The fallback used to be Anthropic's 0.10x/1.25x applied to every vendor,
    which is the wrong shape twice over: it invents a write premium on the
    automatic-caching providers, and it invents a 10x read discount on the
    endpoints that do not cache at all. Since the carry term is
    `cache_read * remaining_turns`, and that term dominates a long session,
    borrowing Anthropic's discount for a model that has none does not make the
    estimate slightly optimistic -- it makes it wrong by an order of magnitude,
    in the direction that recommends the model.

    The provider table answers it properly now: published rates win, the
    provider's own shape fills the gaps, and an endpoint with no known cache is
    priced at full input rate, which is what it actually charges.
    """
    if e.inp is None:
        # Zero is not an unknown price. Returning it made the carry term free,
        # and the carry term is the dominant one -- an unpriced session model
        # took a delegated cost from $8.07 to $0.05 here, a 175x understatement
        # in the direction that recommends delegating. This module's own
        # docstring is explicit that "a missing cache read rate is the
        # difference between a real saving and a fantasy", and `Entry` keeps
        # prices Optional so that "not published" and "free" stay apart.
        raise UnpricedEntryError(
            f"{e.id or e.key} has no published input price, so its cache rates "
            "cannot be derived; a cost computed from it would be a guess "
            "dressed as a measurement")
    prov = providers.for_model(e.id or e.key, e.org)
    read, write = e.cache_read, e.cache_write
    assumed = read is None or write is None
    if read is None:
        mult = prov.cache_read_mult if prov.caches else 1.0
        read = e.inp * (mult if mult is not None else 1.0)
    if write is None:
        write = e.inp * (prov.write_mult() if prov.caches else 1.0)
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
    if not session.priced:
        raise UnpricedEntryError(
            f"session model {session.id or session.key!r} has no published "
            "price; the summary a delegation returns is carried at the session "
            "model's rate, so without it the dominant term of this comparison "
            "is unknown, not zero")
    s_read, s_write, s_assumed = _cache_rates(session)
    e_read, e_write, e_assumed = _cache_rates(e)
    e_read *= max(0.0, need.cache_miss_correction)
    s_read *= max(0.0, need.cache_miss_correction)
    inp = e.inp or 0.0
    out = e.out or 0.0

    # One expression, shared with `cost.admitted_token_cost`: a token entering a
    # context is a cache write now plus a cache read on every turn that re-reads
    # it. The rates come from the catalog rather than the Claude table, which is
    # the only difference between the two callers.
    e_rates = Rates(inp=inp, out=out, cache_read=e_read, cache_write=e_write)
    s_rates = Rates(inp=session.inp or 0.0, out=session.out or 0.0,
                    cache_read=s_read, cache_write=s_write)
    reads = (need.expected_reads if need.expected_reads is not None
             else float(max(0, need.remaining_turns)))

    # Inline: the task's tokens land in the main context and stay there.
    # A model switch mid-session also rebuilds the existing prefix, since the
    # prompt cache is model-scoped -- that is the `switch` term.
    switch = 0.0
    if e.key != session.key:
        switch = admitted_cost(need.context_tokens, e_rates, reads=0).write
    task = admitted_cost(need.est_read_tokens, e_rates, reads=reads)
    generated = admitted_cost(need.est_out_tokens, e_rates, reads=reads)
    carry = task.reads
    generate = need.est_out_tokens * out / M
    inline = switch + task.total + generate + generated.reads

    # Delegated: subagent reads at its own rate, main context takes a summary.
    sub = (need.est_read_tokens * inp + need.est_out_tokens * out) / M
    summary = admitted_cost(need.summary_tokens, s_rates, reads=reads)
    delegated = sub + summary.total

    # `subagent` is the substitutable leg: what this model charges to do the
    # work. The summary that comes back is admitted and carried at the session
    # model's rate no matter who produced it, so comparing two candidates on
    # `delegated` compares them partly on a term neither one controls.
    harness = _harness.get(need.harness)
    # Unknown context is NOT feasible. `registry.ModelSpec.fits` states the rule
    # this gate exists to apply -- "Unknown context is False, not True. A
    # feasibility gate that passes because it does not know is not a gate" --
    # and this copy of it had the opposite default, so all 53 bundled entries
    # with no published window were quoted as able to hold any session at all.
    feasible = e.context is not None and e.context >= need.context_tokens
    blocked = ""
    if e.context is None:
        blocked = ("no published context window; it cannot be shown to hold "
                   f"the {need.context_tokens:,}-token session")
    elif not feasible:
        blocked = (f"context window {e.context:,} < session context "
                   f"{need.context_tokens:,}")
    elif not harness.allows_main_session(e.org):
        feasible = False
        blocked = harness.why_blocked(e.org)
    return Costed(e, inline=inline, delegated=delegated, subagent=sub, carry=carry,
                  assumed_cache=e_assumed or s_assumed, inline_feasible=feasible,
                  inline_blocked=blocked,
                  delegate_feasible=harness.supports_subagents,
                  delegate_blocked=(
                      "" if harness.supports_subagents else
                      f"{harness.label} runs one conversation; there is no "
                      "subagent to delegate to"))


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
        """Rating points per dollar. Only comparable within one ranking.

        A free model has unbounded value, not none. `cost <= 0` returning 0.0
        conflated "costs nothing" with "cannot be valued" and would have sorted
        every free endpoint last in a ranking meant to reward cheapness -- the
        same zero-is-not-absent error `frontier.build` carried, and reachable
        for the same reason: free rows only survive the catalog round-trip now
        that `to_json` stops erasing a zero price.
        """
        if self.rating is None or self.cost < 0:
            return 0.0
        if self.cost == 0:
            return float("inf")
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


def _session_entry(cat: Catalog, need: Need, ref: Entry) -> Entry:
    """The entry whose rates carry the returned summary, validated.

    Resolved here rather than inline at two call sites so the failure reads the
    same in both. A session model that is missing or unpriced is refused by
    name: it is the model every delegated summary is carried at, so it sets the
    dominant term of the comparison, and falling back to the reference would
    silently answer a different question than the one asked.
    """
    if need.session_model == need.reference:
        return ref
    got = cat.get(need.session_model)
    if got is None:
        raise KeyError(
            f"session model {need.session_model!r} is not in the catalog")
    if not got.priced:
        raise UnpricedEntryError(
            f"session model {need.session_model!r} has no published price; "
            "every delegated summary is carried at its rate, so the comparison "
            "cannot be made without one")
    return got


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
    session = _session_entry(cat, need, ref)

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
            prov = providers.for_model(e.id or e.key, e.org)
            if not prov.caches:
                warnings.append(
                    f"{prov.name} publishes no prompt cache for this model; "
                    "every re-read is priced at full input rate, which is the "
                    "conservative reading and may be pessimistic")
            else:
                warnings.append(
                    f"provider publishes no cache rates; carried cost uses "
                    f"{prov.name}'s published shape and is an estimate")
        if not c.inline_feasible and c.inline_blocked:
            reasons.append(f"subagent only: {c.inline_blocked}")
        if not c.delegate_feasible and c.delegate_blocked:
            reasons.append(f"inline only: {c.delegate_blocked}")
        if not c.usable:
            # Neither placement exists here: not a candidate at any price.
            continue

        pf = p_fail_from_elo(e, ref, difficulty=need.difficulty)
        pl = p_loss_from_elo(e, ref, difficulty=need.difficulty)
        if e.key != ref.key and ratings_overlap(e, ref):
            reasons.append(
                f"arena cannot separate this from {ref.id}: the published "
                f"intervals overlap, so the rating gap is not evidence")
        if e.rating_variant and not e.rating_variant.endswith(e.key):
            # One constant string, so the caller's de-duplication collapses it
            # into a single line; the specific variant goes on the row, where it
            # belongs, instead of repeating the same paragraph per model.
            reasons.append(f"rating earned by the {e.rating_variant!r} variant")
            warnings.append(
                "some ratings were earned at a higher reasoning effort than the "
                "price assumes; the arena ranks efforts separately and the price "
                "table has one price per model (`adder models show <name>`)")
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
    tier_p_fail: float | None = None,
    unusable_given_loss: float = UNUSABLE_GIVEN_LOSS,
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
    session = _session_entry(cat, need, ref)

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

    # `measured_p_fail` replaces the estimate outright; `tier_p_fail` is a
    # measured rate for the tier being replaced and composes with the Elo gap.
    pf = measured_p_fail
    if pf is None:
        pf = p_fail_from_elo(low, ref, difficulty=need.difficulty,
                             unusable_given_loss=unusable_given_loss)
        if pf is not None and tier_p_fail is not None:
            pf = blend_p_fail(tier_p_fail, pf)
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
        #
        # `replace`, not a fresh `Need`. Building one field by field silently
        # dropped everything not restated: the harness (so the review leg was
        # priced under placement rules the caller had ruled out), the reference,
        # the difficulty, and -- worst -- `cache_miss_correction` and
        # `expected_reads`, so `--measured` corrected the two single legs and
        # left the review and judge legs on the uncorrected assumption. Two legs
        # of one comparison priced under different economics is exactly the
        # disagreement this module was written to remove.
        review_need = replace(
            need,
            est_read_tokens=max(1_000, int(need.est_read_tokens / review_compression)),
            est_out_tokens=max(200, int(need.est_out_tokens / 3)),
            summary_tokens=0,
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
        judge_need = replace(
            need,
            est_read_tokens=panel_n * max(500, need.est_out_tokens),
            est_out_tokens=max(200, need.est_out_tokens // 3),
            needs_tools=False,
            summary_tokens=0,
        )
        panel = panel_n * c_low.best + cost_of(ref, judge_need, session=session).best
        # Quality is deliberately not estimated here. The obvious formula --
        # lift the rating by the best-of-N win rate -- requires the N runs to
        # fail independently, and runs of the same model on the same prompt do
        # not. There is no published measurement of that correlation, so any
        # number put here would be a constant chosen to make the row look
        # reasonable. `None` is the honest value, and the shape is still worth
        # pricing because its *cost* is exact.
        q_panel = None
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
    import sys

    ap = argparse.ArgumentParser(
        prog="adder select",
        description="pick a model, or a combination, from the catalog")
    ap.add_argument("task", nargs="*", help="task text (used to size the work)")
    ap.add_argument("--context", type=int, default=None, help="session context tokens")
    ap.add_argument("--remaining", type=int, default=None, help="remaining turns")
    ap.add_argument("--read-tokens", type=int, default=None)
    ap.add_argument("--out-tokens", type=int, default=None)
    ap.add_argument("--reference", default=_settings.session_model(),
                    help="the 'do it properly' model to compare against")
    ap.add_argument("--session-model", default=None)
    ap.add_argument("--cheap", default=None, help="pin the cheap leg of a combination")
    ap.add_argument("--open-weights", action="store_true")
    ap.add_argument("--harness", default=_harness.default(),
                    choices=_harness.names(),
                    help="the agent runtime; ones that pin the main session to "
                         "a vendor (claude-code, codex, gemini-cli) refuse "
                         "inline placement for the others. Set ADDER_HARNESS "
                         "to change the default")
    ap.add_argument("--no-tools", action="store_true",
                    help="drop the tool-use requirement")
    ap.add_argument("--difficulty", type=float, default=1.0)
    ap.add_argument("--project", default=None,
                    help="scope measured escalation history to one project")
    ap.add_argument("--floor", type=float, default=None, help="minimum arena rating")
    ap.add_argument("--measured", action="store_true",
                    help="correct cache-read rates with the miss rate measured "
                         "from your own transcripts (see `adder carry`)")
    ap.add_argument("--include-unrated", action="store_true",
                    help="also rank models with no arena rating (off by default)")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--combos", action="store_true", help="rank multi-model plans")
    ap.add_argument("--sensitivity", action="store_true",
                    help="sweep the unmeasured p_fail prior and report whether "
                         "the winning plan depends on it")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    from adder.decide.route.classify import Tier, classify

    ctx, rem, smodel = a.context, a.remaining, a.session_model
    if ctx is None or rem is None or smodel is None:
        try:
            from adder.measure.session.live import analyse, current_session
            s = current_session()
            if s is not None:
                r = analyse(s)
                ctx = ctx if ctx is not None else r.context
                # The MEAN: `Need.remaining_turns` prices the carry term.
                rem = rem if rem is not None else round(r.carry_turns)
                smodel = smodel or r.model
        except Exception:
            pass
    ctx = ctx if ctx is not None else 100_000
    rem = rem if rem is not None else 100
    smodel = smodel or _settings.session_model()

    task = " ".join(a.task)
    v = classify(task) if task else None
    read = a.read_tokens
    if read is None:
        read = ({Tier.T0: 8_000, Tier.T1: 20_000, Tier.T2: 60_000,
                 Tier.T3: 120_000}[v.tier] if v else 40_000)
    difficulty = a.difficulty
    if v is not None and a.difficulty == 1.0:
        difficulty = {0: 0.4, 1: 0.7, 2: 1.0, 3: 1.4}[int(v.tier)]

    correction, reads = 1.0, None
    correction_note = "published cache rates, uncorrected"
    if a.measured:
        try:
            from adder.core.trace import DEFAULT_ROOT, load_sessions
            from adder.measure.window.carry import Carry
            fitted = Carry.measure(load_sessions(DEFAULT_ROOT))
            # Against the multiplier this workload would otherwise have been
            # priced at, not against Anthropic's constant. On a workload that
            # already runs somewhere without a cache the two differ by 10x, and
            # dividing by the wrong one turns a correction into a new error.
            correction = fitted.read_mult / max(1e-12, fitted.baseline_read_mult)
            # Both halves of the fitted model: how much a re-read really costs
            # (miss rate) and how many re-reads a token actually survives to
            # see (residency across compaction).
            reads = fitted.expected_reads(rem, context_tokens=ctx)
            correction_note = (f"cache reads x{correction:.2f} from a measured "
                               f"{fitted.read_mult:.3f} realised multiplier; "
                               f"{reads:,.0f} expected re-reads of {rem} turns")
        except Exception as e:
            correction_note = f"could not fit a carry model ({e}); using published rates"

    need = Need(
        context_tokens=ctx, remaining_turns=rem, est_read_tokens=read,
        cache_miss_correction=correction, expected_reads=reads,
        est_out_tokens=a.out_tokens or 1_200,
        needs_tools=not a.no_tools, open_weights_only=a.open_weights,
        difficulty=difficulty, reference=a.reference, session_model=smodel,
        harness=a.harness,
    )

    cat = load()
    stale = cat.age_days()

    # Fail on the flags rather than four frames down. A model named on the
    # command line that the catalog has never heard of, or holds no price for,
    # is a typo to correct -- not a traceback, and emphatically not a silent
    # zero for the term that dominates the comparison.
    try:
        _session_entry(cat, need, cat.get(a.reference) or Entry(key="_", id="_"))
    except (KeyError, UnpricedEntryError) as exc:
        print(f"adder pick: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        print("Try `adder models list` to see what the catalog holds.",
              file=sys.stderr)
        return 2

    # Measured escalation history for the classified tier, when there is enough
    # of it to act against the 0.5 prior. This is the one number here that is a
    # measurement rather than a proxy, so it is used wherever it exists.
    tier_pf, tier_basis = None, "arena Elo only"
    if v is not None:
        try:
            # DEFAULT_LOG is read here, not captured as a default argument, so
            # a test can point the log somewhere harmless.
            from adder.decide.track.outcomes import evidence

            ev = evidence(v.tier.name, a.project)
            if ev.informative:
                tier_pf, tier_basis = ev.p_fail, f"blended with {ev.describe()}"
        except Exception:
            pass

    if a.combos:
        # Where there is enough escalation history, the prior stops being a
        # prior: it is the ratio of the measured escalation rate to the
        # modelled preference loss that should have produced it.
        ugl, ugl_basis = UNUSABLE_GIVEN_LOSS, "prior 0.35, unmeasured"
        if tier_pf is not None and a.cheap:
            low_e, ref_e = cat.get(a.cheap), cat.get(a.reference)
            if low_e is not None and ref_e is not None:
                ugl, ugl_basis = calibrate_unusable_given_loss(
                    low_e, ref_e, tier_pf, difficulty=difficulty)
        plans = combos(need, cat=cat, cheap=a.cheap, tier_p_fail=tier_pf,
                       unusable_given_loss=ugl)
        if a.json:
            print(json.dumps([{
                "shape": c.shape, "models": c.models,
                "expected_cost": round(c.expected_cost, 4),
                "quality": c.quality, "assumption": c.assumption,
                "detail": c.detail} for c in plans], indent=1))
            return 0
        print(f"context {ctx:,} tok  |  {rem} turns left  |  "
              f"difficulty {difficulty:.1f}  |  p_fail: {tier_basis}")
        print(f"unusable_given_loss {ugl:.2f} ({ugl_basis})")
        print()
        for c in plans:
            print(c.render())
            print(f"{' ' * 14}assumes: {c.assumption}")
        if a.sensitivity:
            print()
            print(sensitivity(need, cat=cat, cheap=a.cheap,
                              tier_p_fail=tier_pf).render())
        print("\nExpected cost is MODELLED. Quality is arena Elo, which measures")
        print("human preference on chat and web-dev, not agentic tool use.")
        if not a.sensitivity:
            print("Pass --sensitivity to check whether the winner depends on the "
                  "unmeasured prior.")
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
    print(f"{correction_note}")
    if stale is not None and stale > 21:
        print(f"! catalog is {stale:.0f} days old; run `adder models refresh`")
    print()
    ref_entry = cat.get(a.reference)
    if ref_entry is not None:
        rc = cost_of(ref_entry, need, session=_session_entry(cat, need, ref_entry))
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




# --------------------------------------------------------------------------
# Does the answer depend on the number nobody measured?
# --------------------------------------------------------------------------

@dataclass
class Sensitivity:
    """Whether a recommendation survives its own weakest assumption.

    The point of this is not to produce a better estimate of
    `UNUSABLE_GIVEN_LOSS`. It is to find out whether the recommendation even
    depends on it. Most of the time it does not -- the cost gaps between plans
    are wider than any plausible value of the constant can move -- and saying
    so is worth more than another decimal place. When it does, the honest
    output is "this recommendation is not resolvable at the evidence we have",
    which no amount of confident formatting can substitute for.
    """

    parameter: str
    low: float
    high: float
    winner_low: str
    winner_high: str
    flips: list[tuple[float, str, str]] = field(default_factory=list)

    @property
    def stable(self) -> bool:
        return not self.flips

    def render(self) -> str:
        span = f"{self.parameter} over [{self.low:.2f}, {self.high:.2f}]"
        if self.stable:
            return (f"  stable: {self.winner_low} wins across the whole plausible "
                    f"range of {span}")
        lines = [f"  UNSTABLE: the cheapest plan changes inside {span}"]
        for at, before, after in self.flips:
            lines.append(f"    at {self.parameter} = {at:.2f}: {before} -> {after}")
        lines.append("    the recommendation is a coin flip on a number nobody has "
                     "measured; prefer the plan that wins at the pessimistic end")
        return "\n".join(lines)


def sensitivity(
    need: Need,
    *,
    cat: Catalog | None = None,
    cheap: str | None = None,
    span: tuple[float, float] = UNUSABLE_RANGE,
    steps: int = 19,
    **combo_kwargs,
) -> Sensitivity:
    """Sweep the invented constant and report where, if anywhere, the answer flips."""
    cat = cat or load()
    lo, hi = span
    winners: list[tuple[float, str]] = []
    for i in range(steps):
        v = lo + (hi - lo) * i / max(1, steps - 1)
        plans = combos(need, cat=cat, cheap=cheap, unusable_given_loss=v,
                       **combo_kwargs)
        best = plans[0]
        winners.append((v, f"{best.shape} ({' + '.join(best.models)})"))

    flips = [
        (winners[i][0], winners[i - 1][1], winners[i][1])
        for i in range(1, len(winners))
        if winners[i][1] != winners[i - 1][1]
    ]
    return Sensitivity("unusable_given_loss", lo, hi,
                       winners[0][1], winners[-1][1], flips)
if __name__ == "__main__":
    raise SystemExit(main())
