"""The brief you carry into a fresh session, and the largest one worth carrying.

`live` and `compact` both end at the same sentence: restarting this session is
worth more than carrying it. Neither says what has to survive the restart, and
that omission is why the advice does not get taken. A restart with no handoff
is not a saving, it is an amnesia -- the next session re-derives what the last
one knew, and `adder reread` finds the bill for that afterwards.

So the question is not "restart or not". It is **how much may I write down and
still come out ahead**, and that has an exact answer.

The arithmetic
--------------
Restarting frees the working context but pays for a fresh opening plus whatever
is carried:

    freed  = context - H
    saving = freed * r * read_mult * remaining_turns
    cost   = opening(H)          # warm floor + H written at the cache-write rate

The saving falls linearly in `H` and the cost rises linearly in `H`, so they
cross once. `max_handoff` returns that crossing: the brief budget. Above it,
writing more down costs more than the context it replaces -- and a brief that
big is a context, not a summary.

Two properties worth stating, because they are the reason this is not obvious:

* The budget is **large** when the horizon is long and the context is full,
  which is exactly when people write the shortest briefs. At a 500K context
  with 300 turns left it runs to tens of thousands of tokens: there is no
  reason to be terse.
* The budget is **negative** when the session is nearly over. That is not a
  small brief, it is "do not restart" -- and it is the case a rule of thumb
  ("always hand off 2K") gets wrong in the expensive direction.

What a brief has to name
------------------------
The tokens are only half the answer. The other half is what goes in them, and
that is recoverable from the transcript without reading a word of it: the files
this session edited, the ones it read most, and the commands it kept re-running
are all tool *inputs*. This module lists them, ranked, so the brief is written
from what the session did rather than from what the last turn happened to be
about. It never emits message text -- paths and commands only, the same rule
`export` follows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from adder.core import settings as _settings
from adder.core.filters import root_of as _root_of
from adder.pricing.cost import Rates
from adder.pricing.prices import CACHE_READ_MULT
from adder.pricing.registry import rate

M = 1_000_000.0

# A brief below this cannot carry a decision, let alone a plan. When the budget
# comes out under it, the honest output is "do not restart", not "write less".
MIN_USEFUL_BRIEF = 300

# Ranked items to offer. More than this is a context, not a brief.
DEFAULT_ITEMS = 12


@dataclass(frozen=True)
class Budget:
    """How many tokens may be carried before restarting stops paying."""

    tokens: int
    context: int
    remaining: float
    model: str
    read_mult: float
    opening_floor: int
    warm_share: float

    @property
    def viable(self) -> bool:
        return self.tokens >= MIN_USEFUL_BRIEF

    @property
    def binding(self) -> bool:
        """Is cost actually the constraint on the brief?

        When the crossing point sits at most of the context, the answer is no:
        every brief anyone would write is far below it, and reporting "carry up
        to 308,000 tokens" as advice invites someone to try. The number is a
        bound; only sometimes is it a budget.
        """
        return self.share_of_context < 0.5

    @property
    def share_of_context(self) -> float:
        return self.tokens / self.context if self.context else 0.0

    def describe(self) -> str:
        if not self.viable:
            return ("no brief is worth writing: at this horizon a restart costs "
                    "more than carrying the context does")
        if not self.binding:
            return (f"cost is not the constraint — the restart still pays with "
                    f"anything up to {self.tokens:,} tokens carried "
                    f"({self.share_of_context:.0%} of the context). Write what "
                    "the next session needs.")
        return (f"stay under {self.tokens:,} tokens ({self.share_of_context:.0%} "
                "of the context); above that, carrying beats restarting")


def max_handoff(*, context: int, remaining: float, model: str,
                read_mult: float | None = None, opening=None,
                ttl: str = "1h", on: date | None = None) -> int:
    """The largest brief for which restarting still beats carrying on.

    Solved rather than searched: both sides are linear in `H`, so

        (C - H) * r * m * R  =  floor_cost + H * r * w

    has one root, and clamping it to [0, C] is the whole of the edge-case
    handling. Returns 0 when a restart never pays at this horizon.

    `read_mult` is the realised multiple of the input rate a carried token pays,
    as `carry` fits it. Left as `None` it comes from the model's own published
    cache economics rather than from Anthropic's 0.10x: on a provider with no
    prompt cache a re-read costs the full input rate, and defaulting to the
    discount told those sessions that carrying context was ten times cheaper
    than it is, which is the side of this equation that argues against
    restarting. The write side is taken from the provider too, for the same
    reason -- under automatic caching there is no write premium to pay.
    """
    from adder.measure.window.prefix import Opening

    op = opening if opening is not None else Opening.default()
    rates = Rates.for_model(model, ttl=ttl, on=on)
    r = rates.inp
    mult = (rates.cache_read / r) if read_mult is None and r else read_mult
    if mult is None:
        mult = CACHE_READ_MULT
    carry_per_token = r * mult * max(0.0, remaining) / M
    write_per_token = rates.cache_write / M
    # The floor is paid whatever the brief is: it is the opening with no handoff.
    floor = op.cost(model, ttl=ttl, handoff_tokens=0, on=on)
    denom = carry_per_token + write_per_token
    if denom <= 0:
        return 0
    h = (context * carry_per_token - floor) / denom
    return int(max(0, min(context, h)))


def budget(sess, *, remaining: float, read_mult: float | None = None,
           ttl: str = "1h", on: date | None = None) -> Budget:
    """The brief budget for one live session, priced off its own opening."""
    from adder.measure.window.prefix import Opening

    if not getattr(sess, "turns", None):
        return Budget(0, 0, remaining, _settings.session_model(), read_mult, 0, 0.0)
    # The conversation's own last turn. `Opening.from_session` reads the first
    # main-chain turn; taking the last one from the raw list would describe a
    # subagent's context and model when a session ends inside a delegation.
    last = sess.main_turns[-1]
    op = Opening.from_session(sess)
    tokens = max_handoff(context=last.context, remaining=remaining,
                         model=last.model, read_mult=read_mult, opening=op,
                         ttl=ttl, on=on)
    return Budget(tokens=tokens, context=last.context, remaining=remaining,
                  model=last.model, read_mult=read_mult,
                  opening_floor=op.floor_tokens, warm_share=op.warm_share)


@dataclass
class Item:
    """One thing a brief could name, and why it earned the space."""

    kind: str          # "edited" | "read" | "ran"
    name: str
    calls: int = 0
    tokens: int = 0

    @property
    def weight(self) -> float:
        """Rank by what the next session cannot recover for itself.

        The three kinds are not comparable on one scale, and an earlier version
        that tried -- rank everything by tokens seen -- put a one-off `cat` of a
        large file above every edit the session made. Tokens measure what a
        result *cost*, not what the next session *needs*, and those come apart
        exactly where it matters.

        So the order is by kind first:

            edited  state that exists nowhere else. Always named.
            ran     a command run more than once is a loop the next session
                    will re-enter; a command run once is history.
            read    ranked by what re-establishing it would cost.
        """
        if self.kind == "edited":
            return 1e9 + self.calls
        if self.kind == "ran":
            return 1e6 * max(0, self.calls - 1)
        return float(self.tokens)


def items_from(path: Path | str, *, keep_single_commands: bool = False
               ) -> list[Item]:
    """What one session touched, ranked — paths and commands only, never prose.

    Built on `reread.scan` so there is one parser for tool calls in this repo
    rather than two that disagree about which blocks count. Commands run exactly
    once are dropped by default: they are what the session did, and a brief is
    for what the next session has to keep doing.
    """
    from adder.measure.window.reread import scan

    # `min_tokens=0`: an `Edit` answers with "ok" and a `Write` with nothing at
    # all. The re-read report is right to drop those -- they cost nothing to
    # carry -- and a brief that omits every edit the session made is useless.
    rep = scan(path, min_tokens=0)
    # Only tools whose identity is a path or a command. Anything else is
    # identified by a hash of its input (see `reread.identity`), which is
    # meaningless in a brief -- and the tools that fall through are the ones
    # whose input is prose.
    nameable = ("Read", "Edit", "Write", "NotebookEdit", "Bash", "Grep", "Glob")
    edited: dict[str, Item] = {}
    read: dict[str, Item] = {}
    ran: dict[str, Item] = {}
    for r in rep.repeats.values():
        if r.tool not in nameable:
            continue
        target = r.ident.split(":", 1)[1] if ":" in r.ident else r.ident
        tokens = sum(a.tokens for a in r.admissions)
        if r.tool in ("Edit", "Write", "NotebookEdit"):
            it = edited.setdefault(target, Item("edited", target))
        elif r.tool == "Bash":
            it = ran.setdefault(target, Item("ran", target))
        else:
            it = read.setdefault(target, Item("read", target))
        it.calls += r.calls
        it.tokens += tokens
    keep_ran = [i for i in ran.values() if keep_single_commands or i.calls > 1]
    out = list(edited.values()) + keep_ran + list(read.values())
    return sorted(out, key=lambda i: -i.weight)


def plan(sess, path: Path | str | None, *, remaining: float,
         read_mult: float | None = None, top: int = DEFAULT_ITEMS,
         ttl: str = "1h", on: date | None = None) -> tuple[Budget, list[Item]]:
    """The budget and the brief for one session.

    `read_mult` defaults to None, not to `CACHE_READ_MULT`. Defaulting to
    Anthropic's 0.10x is the error `max_handoff` documents at length: on a
    provider with no prompt cache a re-read costs the *full* input rate, and
    quoting the discount tells that session carrying context is ten times
    cheaper than it is -- which is the side of the equation that argues against
    restarting.
    """
    b = budget(sess, remaining=remaining, read_mult=read_mult, ttl=ttl, on=on)
    return b, (items_from(path)[:top] if path is not None else [])


@dataclass
class Measured:
    """What handoffs on this machine have actually been, in tokens."""

    sizes: list[int] = field(default_factory=list)
    floor: int = 0

    @property
    def n(self) -> int:
        return len(self.sizes)

    def median(self) -> int:
        from adder.util.stats import median

        return int(median(self.sizes)) if self.sizes else 0

    def p90(self) -> int:
        from adder.util.stats import quantile

        return int(quantile(self.sizes, 0.9)) if self.sizes else 0


def measured_handoffs(sessions) -> Measured:
    """How much each session opened with above the shared floor.

    The floor is the median opening context: the part every session on this
    machine has in common (system prompt, tool schemas, instruction files).
    What a session opens with *above* that is the only handoff a transcript can
    see -- the first prompt and whatever came attached to it.
    """
    from adder.util.stats import median

    firsts = [s.main_turns[0].context for s in sessions.values() if s.turns]
    if not firsts:
        return Measured()
    floor = int(median(firsts))
    return Measured(sizes=[max(0, c - floor) for c in firsts], floor=floor)


def _elide(name: str, *, head: bool, width: int = 56) -> str:
    """Trim to `width`, keeping the informative end.

    A command is identified by how it starts (`pytest -q …`) and a path by how
    it ends (`…/window/memory.py`). One truncation rule for both makes half the
    rows unreadable.
    """
    if len(name) <= width:
        return name
    return name[: width - 1] + "…" if head else "…" + name[-(width - 1):]


def report(b: Budget, items: list[Item], measured: Measured | None = None) -> str:
    from adder.util.render import money, table, tokens

    lines = [
        f"  Context {tokens(b.context)} on {b.model} · ~{b.remaining:,.0f} turns "
        f"expected to remain · realized re-read {b.read_mult:.3f}x",
        f"  Brief budget: {b.describe()}",
    ]
    if measured and measured.n:
        lines.append(
            f"  For comparison, sessions here opened with a median "
            f"{tokens(measured.median())} above the {tokens(measured.floor)} "
            f"shared floor (p90 {tokens(measured.p90())}).")
    if not b.viable:
        lines += ["", "  Finish the session. Restarting is the more expensive "
                      "option at this horizon, whatever the brief says."]
        return "\n".join(lines)

    if items:
        lines += ["", "  What the brief has to name, ranked by what the next "
                      "session would otherwise re-derive:", ""]
        body = [[i.kind, _elide(i.name, head=i.kind == "ran"), i.calls,
                 tokens(i.tokens)] for i in items]
        lines += table(body, ["kind", "what", "calls", "tokens seen"],
                       align="<<>>")
    lines += ["", f"  Writing the brief costs about {money(_write_cost(b))} once. "
                  "Carrying the context it replaces costs that every turn."]
    return "\n".join(lines)


def _write_cost(b: Budget, *, ttl: str = "1h", on: date | None = None) -> float:
    from adder.pricing.prices import CACHE_WRITE_MULT

    return (b.tokens * rate(b.model, on).inp
            * CACHE_WRITE_MULT.get(ttl, 1.25) / M)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from adder.core.trace import load_sessions
    from adder.measure.session.live import analyse as analyse_live
    from adder.measure.session.live import current_session, current_transcript

    ap = argparse.ArgumentParser(
        prog="adder handoff",
        description="How much may be carried into a fresh session before the "
                    "restart stops paying, and what the brief has to name.")
    ap.add_argument("--cwd", default=None,
                    help="working directory whose session to price")
    ap.add_argument("--root", default=None,
                    help="transcript directory (default: the `root` setting)")
    ap.add_argument("--remaining", type=float, default=0.0, metavar="N",
                    help="turns to assume remain (default: this session's estimate)")
    ap.add_argument("--context", type=int, default=0, metavar="TOK",
                    help="price a hypothetical context instead of the live one")
    ap.add_argument("--model", default=_settings.session_model(),
                    help="model for --context (default: %(default)s)")
    ap.add_argument("--top", type=int, default=DEFAULT_ITEMS, metavar="N",
                    help="items to list (default: %(default)s)")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    a = ap.parse_args(argv)
    # `root_of`: the argument if one was given, else the `root`
    # setting. Resolved here so two commands cannot disagree
    # about which transcript directory `adder config` names.
    a.root = str(_root_of(a))

    # `root_of`: the argument if one was given, else the `root`
    # setting. Resolved here so two commands cannot disagree.
    a.root = str(_root_of(a))
    if a.context:
        tokens_ = max_handoff(context=a.context, remaining=a.remaining or 300,
                              model=a.model)
        if a.json:
            print(json.dumps({"context": a.context, "model": a.model,
                              "remaining": a.remaining or 300,
                              "budget_tokens": tokens_}))
            return 0
        print()
        print(f"  At {a.context:,} tokens on {a.model} with "
              f"{a.remaining or 300:,.0f} turns left: carry up to {tokens_:,} "
              "tokens and the restart still pays.")
        print()
        return 0

    sess = current_session(a.cwd, a.root)
    if sess is None:
        if a.json:
            print(json.dumps({"error": "no transcript for this directory"}))
            return 1
        print("  No transcript found for this directory yet.")
        return 1

    live = analyse_live(sess)
    remaining = a.remaining or live.carry_turns
    b, items = plan(sess, current_transcript(a.cwd, a.root), remaining=remaining,
                    read_mult=live.read_mult, top=a.top)
    measured = measured_handoffs(load_sessions(Path(a.root).expanduser()))

    if a.json:
        print(json.dumps({
            "session": sess.id,
            "context": b.context,
            "model": b.model,
            "remaining": round(b.remaining, 1),
            "read_mult": round(b.read_mult, 5),
            "budget_tokens": b.tokens,
            "viable": b.viable,
            "share_of_context": round(b.share_of_context, 4),
            "write_cost": round(_write_cost(b), 5),
            "measured_median_handoff": measured.median(),
            "measured_p90_handoff": measured.p90(),
            "shared_floor": measured.floor,
            "items": [{"kind": i.kind, "name": i.name, "calls": i.calls,
                       "tokens": i.tokens} for i in items],
        }))
        return 0

    print()
    print(report(b, items, measured))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
