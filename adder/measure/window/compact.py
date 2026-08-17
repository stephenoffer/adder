"""Compaction, priced: what it cost, what it bought, and when to do it.

Compaction is the only lever in this repo that the agent can pull *during* a
session and the only one nobody has ever put a number on. The advice in the
wild is a vibe -- "compact when it feels full" -- and the two ways of being
wrong cost money in opposite directions:

    too late    the context sat at 500K for a hundred turns, every one of them
                re-reading tokens that were about to be thrown away anyway.
    too early   the prefix was rebuilt for a session with thirty turns left,
                and the rebuild cost more than the carry it avoided.

Both are arithmetic, and both are decided by the same comparison.

The comparison
--------------
A compaction pays a **rebuild** -- the surviving context is written back at
1.25x or 2.00x rather than read at 0.10x -- and buys a **smaller prefix** on
every remaining turn:

    cost   = kept_tokens * r * write_mult
    saving = freed_tokens * r * read_mult * remaining_turns

which is positive exactly when

    remaining_turns  >  kept * write_mult / (freed * read_mult)

At the measured multipliers that break-even is usually a few dozen turns, and
it is the number this module reports, because it is the one an agent can act
on: *compact when more turns remain than that, not when the bar looks full.*

What is measured and what is not
--------------------------------
Events are found the same way `carry` finds them -- a near-ceiling context that
loses most of itself -- and their rebuild is read off the **next turn's actual
`cache_creation`**, not modelled. The saving is modelled, because the
counterfactual (the session that did not compact) does not exist.

Two costs are deliberately absent, and both push the same way:

* The summarisation call itself. Claude Code does not always bill it into the
  session transcript, so counting it would sometimes double-count and sometimes
  invent. Its omission makes compaction look slightly better than it is.
* **Quality.** A compaction destroys detail, and detail that gets re-derived is
  paid for twice (`adder reread` finds exactly that). Nothing here prices it.
  A report that says "compact more" without saying this is selling something.

So the verdict is stated as a bound: compaction below the break-even is
*certainly* a loss, above it is a gain *if* nothing important was dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date

from adder.core import settings as _settings
from adder.core.filters import root_of as _root_of
from adder.pricing.cost import Rates
from adder.pricing.registry import context_window, rate

M = 1_000_000.0

# Context below this cannot be worth compacting at any horizon: the rebuild is
# a fixed cost and the freed carry is proportional to what is there.
MIN_CONTEXT = 50_000

# A session that never compacted but spent this many turns above the trigger
# was carrying a full context on purpose or by neglect. Either way it is worth
# a line in the report.
MISSED_TURNS = 40


@dataclass(frozen=True)
class Event:
    """One compaction, as it appears in the transcript."""

    session: str
    project: str
    model: str
    turn: int              # ordinal within the session, 0-based
    before: int            # context on the turn that triggered it
    after: int             # context on the turn that followed
    rebuild_tokens: int    # measured cache_creation on the following turn
    remaining: int         # turns the session had left after the event
    ttl: str = "5m"

    @property
    def freed(self) -> int:
        return max(0, self.before - self.after)

    @property
    def kept(self) -> float:
        return self.after / self.before if self.before else 0.0

    def rebuild_cost(self, *, on: date | None = None) -> float:
        """USD actually paid to write the surviving prefix back."""
        return self.rebuild_tokens * Rates.for_model(
            self.model, ttl=self.ttl, on=on).cache_write / M

    def carry_saved(self, read_mult: float, *, on: date | None = None) -> float:
        """USD of re-reads the smaller prefix avoided over the turns that followed."""
        r = rate(self.model, on).inp
        return self.freed * r * read_mult * self.remaining / M

    def net(self, read_mult: float, *, on: date | None = None) -> float:
        return self.carry_saved(read_mult, on=on) - self.rebuild_cost(on=on)

    def breakeven_turns(self, read_mult: float, *, on: date | None = None) -> int:
        """Turns that had to remain for this compaction to have paid for itself."""
        r = rate(self.model, on).inp
        denom = self.freed * r * read_mult / M
        if denom <= 0:
            return 0
        return int(self.rebuild_cost(on=on) / denom) + 1

    def verdict(self, read_mult: float, *, on: date | None = None) -> str:
        need = self.breakeven_turns(read_mult, on=on)
        if self.remaining >= need:
            return "paid off"
        return "too late" if self.remaining < need / 2 else "marginal"


@dataclass
class Miss:
    """A session that carried a near-full context and never compacted.

    `contexts` is the actual per-turn context over the run above the trigger,
    kept in full because the counterfactual has to be simulated against it
    rather than assumed.
    """

    session: str
    project: str
    model: str
    contexts: tuple[int, ...]
    growth: float
    remaining: int

    @property
    def turns_above(self) -> int:
        return len(self.contexts)

    @property
    def mean_context(self) -> int:
        return int(sum(self.contexts) / len(self.contexts)) if self.contexts else 0

    def saving(self, read_mult: float, *, kept: float = 0.35,
               on: date | None = None) -> float:
        """USD a single compaction at the start of the run would have returned.

        The naive answer -- freed tokens, re-read once per remaining turn -- is
        wrong in a way that only shows up near the ceiling, and near the ceiling
        is where every one of these sessions lives. A compacted context
        **refills**: it regrows at the measured rate while the un-compacted one
        is pinned against the context limit and cannot grow at all. The gap
        closes, and pricing it as constant over 348 turns invents money.

        So the counterfactual is simulated turn by turn against the trajectory
        that actually happened, and the saving is the area between them. It is
        still an estimate -- the session that compacted does not exist -- but it
        is an estimate that cannot exceed what was actually carried.
        """
        if not self.contexts:
            return 0.0
        r = rate(self.model, on).inp
        start = self.contexts[0]
        sim = start * kept
        total = 0.0
        for actual in self.contexts:
            total += max(0.0, actual - sim) * r * read_mult / M
            # Regrowth cannot overtake the real trajectory: the same work was
            # being done either way, so the compacted context refills with the
            # same content and stops where the real one stopped.
            sim = min(float(actual), sim + max(0.0, self.growth))
        rebuild = start * kept * _write_rate(self.model) / M
        return max(0.0, total - rebuild)


@dataclass
class CompactReport:
    events: list[Event] = field(default_factory=list)
    misses: list[Miss] = field(default_factory=list)
    read_mult: float = 0.10
    # The model the break-even rule should be stated for. The rule is a ratio
    # of write premium to read multiplier, and that ratio is 12.5 on Anthropic
    # and 1.0 where writes carry no premium -- so quoting one workload's rule
    # at another's provider is off by more than an order of magnitude.
    model: str = field(default_factory=_settings.session_model)
    survival: float = 0.35
    sessions: int = 0
    source: str = "prior"

    @property
    def n(self) -> int:
        return len(self.events)

    def net_total(self, *, on: date | None = None) -> float:
        return sum(e.net(self.read_mult, on=on) for e in self.events)

    def rebuild_total(self, *, on: date | None = None) -> float:
        return sum(e.rebuild_cost(on=on) for e in self.events)

    def missed_total(self, *, on: date | None = None) -> float:
        return sum(m.saving(self.read_mult, kept=self.survival, on=on)
                   for m in self.misses)

    def by_verdict(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.events:
            v = e.verdict(self.read_mult)
            out[v] = out.get(v, 0) + 1
        return out

    def mean_kept(self) -> float:
        from adder.util.stats import median

        return median([e.kept for e in self.events]) if self.events else self.survival


def find_events(sessions) -> list[Event]:
    """Every compaction on record, with its measured rebuild.

    The detector is `carry.is_compaction`, imported rather than restated: two
    definitions of "that was a compaction" is how `sessions` and `carry` came
    to disagree about how many there were.
    """
    from adder.measure.window.carry import is_compaction

    out: list[Event] = []
    for s in sessions.values():
        # Main chain only, the same walk `Session.compactions` does and for the
        # same reason it gives: a subagent opens its own, much smaller context
        # on a cheaper model, so the step down into one satisfies the detector
        # exactly -- a near-ceiling context that loses most of itself. Walking
        # the combined list invented an event whose `after` was the subagent's
        # context, whose model was the subagent's model, and whose "rebuild"
        # was the subagent's legitimate opening write. `adder compact` and
        # `Session.compactions` then reported different numbers of compactions
        # for the same session, which is the disagreement this repo has already
        # paid for once.
        turns = s.main_turns
        for i in range(1, len(turns)):
            prev, cur = turns[i - 1], turns[i]
            if not is_compaction(prev.context, cur.context, cur.model):
                continue
            out.append(Event(
                session=s.id, project=s.project, model=cur.model, turn=i,
                before=prev.context, after=cur.context,
                rebuild_tokens=cur.cache_write or cur.context,
                remaining=len(turns) - 1 - i, ttl=cur.ttl or "5m",
            ))
    return out


def find_misses(sessions, *, min_turns: int = MISSED_TURNS,
                growth: float = 0.0) -> list[Miss]:
    """Sessions that spent a long run near the ceiling without ever compacting."""
    out: list[Miss] = []
    for s in sessions.values():
        if not s.turns:
            continue
        # Main chain only, for the same reason `find_events` is: a subagent's
        # small context is not this session's context. Including sidechain
        # turns both diluted the run above the trigger and made the
        # main-to-sidechain step read as a compaction, which disqualified the
        # session from being reported as a miss at all.
        turns = s.main_turns
        model = turns[0].model if turns else _settings.session_model()
        try:
            limit = context_window(model)
        except Exception:
            continue
        if not limit:
            continue
        trigger = max(MIN_CONTEXT, 0.60 * limit)
        above = [t for t in turns if t.context >= trigger]
        if len(above) < min_turns:
            continue
        first = next(i for i, t in enumerate(turns) if t.context >= trigger)
        if any(is_compaction_pair(turns, i) for i in range(first, len(turns))):
            continue
        out.append(Miss(
            session=s.id, project=s.project, model=model,
            contexts=tuple(t.context for t in above),
            growth=growth,
            remaining=len(turns) - first,
        ))
    return sorted(out, key=lambda m: -m.turns_above)


def is_compaction_pair(turns, i: int) -> bool:
    from adder.measure.window.carry import is_compaction

    if i <= 0 or i >= len(turns):
        return False
    return is_compaction(turns[i - 1].context, turns[i].context, turns[i].model)


def breakeven_context(model: str, remaining_turns: int, *, read_mult: float = 0.10,
                      kept: float = 0.35, ttl: str = "5m",
                      on: date | None = None) -> int:
    """The context above which compacting now beats carrying it to the end.

    Both sides are proportional to the context, so the size cancels and what is
    left is a condition on `remaining_turns` -- which is why this returns
    `MIN_CONTEXT` when the horizon clears the bar and "unreachable" (0) when it
    does not. Reported as a context anyway because that is the number on the
    screen when the decision is made.
    """
    if remaining_turns <= 0 or read_mult <= 0:
        return 0
    need = kept * _write_mult(model, ttl) / ((1.0 - kept) * read_mult)
    return MIN_CONTEXT if remaining_turns >= need else 0


def breakeven_remaining(*, read_mult: float = 0.10, kept: float = 0.35,
                        ttl: str = "5m", model: str | None = None) -> int:
    """Turns that must remain before a compaction can pay for itself.

    Both sides scale with the input rate, so what survives is the ratio of the
    write premium to the read multiplier. That ratio is 12.5 on Anthropic and
    1.0 on a provider that neither charges to write nor discounts to read --
    which moves the break-even by more than an order of magnitude, and is why
    the model has to be an input rather than an assumption.
    """
    if read_mult <= 0 or kept >= 1.0:
        return 0
    model = model or _settings.session_model()
    return int(kept * _write_mult(model, ttl) / ((1.0 - kept) * read_mult)) + 1


def _write_rate(model: str, ttl: str = "5m", on: date | None = None) -> float:
    """USD per Mtok to write a prefix back on this model's provider."""
    return Rates.for_model(model, ttl=ttl, on=on).cache_write


def _write_mult(model: str, ttl: str = "5m", on: date | None = None) -> float:
    """The write rate as a multiple of the input rate. 1.0 where there is no
    premium, which is every automatic-caching provider."""
    r = Rates.for_model(model, ttl=ttl, on=on)
    return (r.cache_write / r.inp) if r.inp else 1.0


def versus_restart(sessions, *, model: str, context_tokens: int,
                   handoff_tokens: int = 2_000, kept: float = 0.35,
                   ttl: str = "1h", on: date | None = None) -> tuple[str, float, str]:
    """Compact, or start a fresh session? Returns (choice, gap USD, why).

    A restart is not a compaction with extra steps. The floor of a new session
    is mostly still cache-resident (`prefix` measures ~74%), so a restart pays
    the write only on the handoff, while a compaction writes back everything it
    kept -- which at a 500K context is an order of magnitude more.
    """
    from adder.measure.window.prefix import measure as measure_opening

    op = measure_opening(sessions)
    compact_cost = context_tokens * kept * _write_rate(model, ttl, on) / M
    restart_cost = op.cost(model, ttl=ttl, handoff_tokens=handoff_tokens, on=on)
    gap = compact_cost - restart_cost
    if gap > 0:
        why = (f"a restart writes only the {handoff_tokens:,}-token handoff; "
               f"compacting writes back {int(context_tokens * kept):,} tokens")
        return ("restart", gap, why)
    return ("compact", -gap, "the surviving context is smaller than a fresh floor "
                             "plus handoff; there is nothing to gain by leaving")


def _dominant_model(sessions) -> str:
    """The model carrying the most turns, or the default when there are none."""
    tally: dict[str, int] = {}
    for s in sessions.values():
        for t in s.turns:
            tally[t.model] = tally.get(t.model, 0) + 1
    return max(tally, key=lambda m: tally[m]) if tally else _settings.session_model()


def analyse(sessions, *, on: date | None = None) -> CompactReport:
    from adder.measure.window.carry import Carry

    c = Carry.measure(sessions) if sessions else Carry.default()
    rep = CompactReport(
        events=find_events(sessions),
        misses=find_misses(sessions, growth=c.growth),
        read_mult=c.read_mult,
        model=_dominant_model(sessions),
        survival=c.survival,
        sessions=len(sessions),
        source=c.source,
    )
    return rep


def report(rep: CompactReport, sessions, *, top: int = 10,
           on: date | None = None) -> str:
    from adder.util.render import money, table, tokens, warn

    need = breakeven_remaining(read_mult=rep.read_mult, kept=rep.mean_kept(),
                               model=rep.model)
    lines = [
        f"  {rep.n} compactions across {rep.sessions} sessions · "
        f"median kept {rep.mean_kept():.0%} · re-read multiplier "
        f"{rep.read_mult:.3f}x ({rep.source})",
        "",
        f"  The rule: a compaction pays for itself with more than ~{need} turns "
        "left.",
        "    Below that the rebuild costs more than the carry it avoids, so the "
        "cheaper move is to finish.",
        "",
    ]

    if rep.events:
        ranked = sorted(rep.events, key=lambda e: e.net(rep.read_mult, on=on))
        body = []
        for e in ranked[:top]:
            body.append([
                e.session[:8], e.turn, tokens(e.before), tokens(e.after),
                f"{e.kept:.0%}", e.remaining, money(e.rebuild_cost(on=on)),
                money(e.net(rep.read_mult, on=on), sign=True),
                e.verdict(rep.read_mult, on=on),
            ])
        lines += table(body, ["session", "turn", "before", "after", "kept",
                              "left", "rebuild", "net", "verdict"],
                       align="<>>>>>>><")
        counts = ", ".join(f"{v} {k}" for k, v in rep.by_verdict().items())
        lines += ["", f"  {counts} · rebuilds cost {money(rep.rebuild_total(on=on))}, "
                      f"net {money(rep.net_total(on=on), sign=True)}"]
    else:
        lines.append("  No compactions on record. Either sessions end before the "
                     "ceiling, or they are being carried to it.")

    if rep.misses:
        lines += ["", f"  Carried but never compacted — {money(rep.missed_total(on=on))} "
                      "of avoidable carry:", ""]
        body = [[m.session[:8], m.project[-24:], m.turns_above, tokens(m.mean_context),
                 money(m.saving(rep.read_mult, kept=rep.survival, on=on))]
                for m in rep.misses[:top]]
        lines += table(body, ["session", "project", "turns above", "mean ctx",
                              "one compaction is worth"], align="<<>>>")

    lines += ["", "  Not priced: what compaction deletes. A detail that has to be "
                  "re-read afterwards is paid for twice (`adder reread`)."]
    total = rep.missed_total(on=on)
    if total > 1.0:
        lines += ["", warn(f"  {money(total)} was spent carrying context past the "
                           "point where compacting it was cheaper.")]
    return "\n".join(lines)


def _json(rep: CompactReport, *, top: int, on: date | None = None) -> str:
    return json.dumps({
        "sessions": rep.sessions,
        "compactions": rep.n,
        "read_mult": round(rep.read_mult, 5),
        "median_kept": round(rep.mean_kept(), 4),
        "source": rep.source,
        "breakeven_remaining_turns": breakeven_remaining(
            read_mult=rep.read_mult, kept=rep.mean_kept()),
        "rebuild_cost": round(rep.rebuild_total(on=on), 4),
        "net": round(rep.net_total(on=on), 4),
        "by_verdict": rep.by_verdict(),
        "events": [
            {"session": e.session, "project": e.project, "model": e.model,
             "turn": e.turn, "before": e.before, "after": e.after,
             "kept": round(e.kept, 4), "freed": e.freed, "remaining": e.remaining,
             "rebuild_tokens": e.rebuild_tokens,
             "rebuild_cost": round(e.rebuild_cost(on=on), 4),
             "carry_saved": round(e.carry_saved(rep.read_mult, on=on), 4),
             "net": round(e.net(rep.read_mult, on=on), 4),
             "breakeven_turns": e.breakeven_turns(rep.read_mult, on=on),
             "verdict": e.verdict(rep.read_mult, on=on)}
            for e in sorted(rep.events, key=lambda e: e.net(rep.read_mult, on=on))[:top]
        ],
        "missed": [
            {"session": m.session, "project": m.project, "turns_above": m.turns_above,
             "mean_context": m.mean_context,
             "saving": round(m.saving(rep.read_mult, kept=rep.survival, on=on), 4)}
            for m in rep.misses[:top]
        ],
        "missed_total": round(rep.missed_total(on=on), 4),
    })


def main(argv: list[str] | None = None) -> int:
    import argparse

    from adder.core.filters import add_arguments as add_window
    from adder.core.filters import load as load_window
    from adder.util.render import money

    ap = argparse.ArgumentParser(
        prog="adder.compact",
        description="What each compaction cost, what it bought, and the turn "
                    "count above which compacting pays for itself.")
    add_window(ap)
    ap.add_argument("--top", type=int, default=10, metavar="N",
                    help="rows per section (default: %(default)s)")
    ap.add_argument("--vs-restart", type=int, default=0, metavar="TOK",
                    help="at this context, compare compacting with a fresh session")
    ap.add_argument("--handoff", type=int, default=2_000, metavar="TOK",
                    help="tokens carried into a restart (default: %(default)s)")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    a = ap.parse_args(argv)
    # `root_of`: the argument if one was given, else the `root`
    # setting. Resolved here so two commands cannot disagree
    # about which transcript directory `adder config` names.
    a.root = str(_root_of(a))

    sessions, window = load_window(a)
    if not sessions:
        if a.json:
            print(json.dumps({"sessions": 0, "compactions": 0}))
            return 0
        print(f"No sessions under {a.root} matching {window.describe()}.")
        return 1

    rep = analyse(sessions)
    if a.json:
        print(_json(rep, top=a.top))
        return 0

    print()
    print(report(rep, sessions, top=a.top))
    if a.vs_restart:
        model = max(
            ((m, c) for s in sessions.values() for m, c in s.cost_by_model().items()),
            key=lambda kv: kv[1], default=(_settings.session_model(), 0.0))[0]
        choice, gap, why = versus_restart(
            sessions, model=model, context_tokens=a.vs_restart,
            handoff_tokens=a.handoff, kept=rep.mean_kept())
        print()
        print(f"  At {a.vs_restart:,} tokens on {model}: {choice} "
              f"(saves {money(gap)}) — {why}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
