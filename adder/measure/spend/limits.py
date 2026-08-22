"""The carry, denominated in the currency a subscription actually charges.

The gap this closes
-------------------
Every other report in this package is in dollars. For anyone on a Pro or Max
plan that is the wrong unit, and not by a scale factor -- by a change of kind.
They do not pay per token. They pay a flat fee and are metered against a
five-hour rolling window and a weekly cap, so an expensive session does not
produce a bill, it produces a lockout. `adder trace` telling such a user their
week cost $412 is answering a question nobody asked them.

What it does not change is the thesis. It sharpens it. On the API the carry is
money; on a subscription the carry is **the window**, and the window is the
thing that runs out at four in the afternoon with the work unfinished. A token
admitted early is re-read on every turn afterwards, and on a plan each of those
re-reads is drawn against a quota that does not care that you already paid for
that text once. So the same measurement is worth more here, not less: it is the
difference between a window that holds 400 turns and one that holds 120.

The number this exists to print
-------------------------------
Not the block totals -- `ccusage` has reported those for a year and they are the
easy part. The number is **what a turn costs the window as a function of how
late in the window it is**. Within a single block, tokens per turn climbs, and it
climbs for exactly one reason: the context is longer than it was. That ratio is
observable, needs no counterfactual, and is the whole argument in one figure.

Three things are not knowable from a transcript, and are labelled rather than
guessed
---------------------------------------------------------------------------
1. **Where the window boundaries actually fall.** Anthropic does not publish the
   rule. The reconstruction here is the one the community converged on -- a block
   opens at the first turn after a gap of at least the window length, and closes
   `--hours` later -- and it is called a reconstruction everywhere it is printed.
   A block boundary off by an hour moves turns between blocks and changes the
   totals; it does not change the within-block slope, which is why that is the
   headline and the totals are context.

2. **What the window holds.** The cap is not published in tokens, it differs by
   model, and peak-hour throttling drains it faster at some times of day than
   others. So no absolute limit is asserted. The comparison is against **the
   largest block on this machine's own record**, which is a *lower bound* on
   capacity -- it is a window that was survived, not one that was refused -- and
   it is described that way. A projection that says "you will be cut off at
   16:40" would be a fabrication with a clock face on it.

3. **Whether a turn was throttled.** A transcript records tokens, not the
   quota debited for them. Two identical turns at 09:00 and at 21:00 may cost
   the window differently and look identical here.

The one thing that is asserted without qualification is the slope, because it
comes from the same dedup-corrected token counts as every other number here and
does not depend on the boundary rule at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from adder.core.trace import Turn

# The window Claude Code meters against. A flag rather than a constant because
# the value is somebody else's and has changed once already.
WINDOW_HOURS = 5.0

# Quartiles used for the within-block slope. Four buckets over a block that may
# hold a dozen turns is already thin; eight would be noise dressed as detail.
QUARTILES = 4

# Fewest turns a block needs before its slope is reported. Below this the first
# and last quartile are one turn each and the ratio is two samples.
MIN_SLOPE_TURNS = 8


@dataclass
class Block:
    """One reconstructed metering window."""

    start: datetime
    turns: list[Turn] = field(default_factory=list)

    @property
    def end(self) -> datetime:
        """When the window closes, not when work in it stopped."""
        return self.start + timedelta(hours=WINDOW_HOURS)

    @property
    def last(self) -> datetime:
        w = [t.when for t in self.turns if t.when]
        return max(w) if w else self.start

    @property
    def tokens(self) -> int:
        return sum(t.total_tokens for t in self.turns)

    @property
    def context_tokens(self) -> int:
        return sum(t.context for t in self.turns)

    @property
    def carry_tokens(self) -> int:
        """Input that was already in the context: the part re-read, not written.

        Cache reads are the whole of it. A cache *write* is text arriving for the
        first time and is new work by any reading; an uncached input token is
        the same. What makes this the carry is that nobody put it there this
        turn -- it is the price of the conversation existing.
        """
        return sum(t.cache_read for t in self.turns)

    @property
    def new_tokens(self) -> int:
        """Tokens that arrived for the first time: the work, without the re-reads.

        Reported alongside `tokens` because `tokens` on its own does not survive
        contact with a reader. A window showing 776 million tokens over 2,000
        turns looks like an error until you see that 8 million of them were new
        and the rest was the same context read again -- at which point it is not
        an error, it is the finding.
        """
        return sum(t.uncached_in + t.cache_write + t.out for t in self.turns)

    @property
    def carry_share(self) -> float:
        return self.carry_tokens / self.context_tokens if self.context_tokens else 0.0

    def cost(self) -> float:
        """What the same work would have cost on the API, for reference only.

        Printed because a plan user still benefits from knowing which blocks were
        expensive in absolute terms, and because it is the only figure here that
        is comparable with the rest of the tool.
        """
        return sum(t.cost() for t in self.turns)

    @property
    def burn(self) -> float:
        """Tokens per minute of elapsed window, from the block's first turn.

        Divided by elapsed rather than by active time on purpose: the window
        drains on the clock, so a rate that ignores the gaps would project a
        block to exhaust itself several times over.
        """
        mins = max(1.0, (self.last - self.start).total_seconds() / 60.0)
        return self.tokens / mins

    def slope(self) -> tuple[float, int, int] | None:
        """`(ratio, first-quartile tokens/turn, last-quartile tokens/turn)`.

        The cost of being late in a window rather than early in it. None when
        the block is too short for the quartiles to mean anything, which is
        reported as absent rather than as 1.0 -- "no slope measured" and "the
        slope is flat" are opposite findings.

        **A ratio below 1 is not noise, it is a restart.** A window is five
        hours of wall clock and can contain the end of one session and the start
        of another, and the new session's context begins small. So the sub-1
        windows on this report are the ones where the lever the rest of the tool
        argues for was actually pulled, and they are left in the median rather
        than filtered out -- excluding them would report the penalty for staying
        in one conversation while quietly dropping the evidence that leaving it
        works.
        """
        seq = sorted(self.turns, key=lambda t: t.when or self.start)
        if len(seq) < MIN_SLOPE_TURNS:
            return None
        q = len(seq) // QUARTILES
        if q < 1:
            return None
        first = sum(t.total_tokens for t in seq[:q]) / q
        last = sum(t.total_tokens for t in seq[-q:]) / q
        if first <= 0:
            return None
        return last / first, int(first), int(last)

    def to_json(self) -> dict:
        s = self.slope()
        return {
            "start": self.start.isoformat(),
            "closes": self.end.isoformat(),
            "turns": len(self.turns),
            "tokens": self.tokens,
            "new_tokens": self.new_tokens,
            "carry_tokens": self.carry_tokens,
            "carry_share": round(self.carry_share, 4),
            "api_cost": round(self.cost(), 4),
            "burn_tokens_per_min": round(self.burn, 1),
            "slope": None if s is None else {
                "ratio": round(s[0], 2), "first_quartile": s[1], "last_quartile": s[2],
            },
        }


def blocks(sessions: dict, *, hours: float = WINDOW_HOURS) -> list[Block]:
    """Reconstruct the metering windows across every session.

    Sessions are interleaved deliberately. The window is per *account*, not per
    session, so two sessions running side by side drain one quota, and a
    per-session view of a shared meter is the mistake this whole report exists
    to avoid making.

    Turns with no timestamp are dropped, not defaulted. A turn placed in the
    wrong block on a guessed time corrupts two blocks rather than one.
    """
    ts = sorted(
        (t for s in sessions.values() for t in s.turns if t.when),
        key=lambda t: t.when,
    )
    out: list[Block] = []
    span = timedelta(hours=hours)
    for t in ts:
        when = t.when
        if out:
            cur = out[-1]
            # Two ways a block ends: the window elapsed, or nothing happened
            # for long enough that the next turn opens a fresh one.
            if when < cur.start + span and (when - cur.last) < span:
                cur.turns.append(t)
                continue
        # The block opens on the hour, which is the convention the community
        # tooling settled on. It is a guess about somebody else's boundary and
        # it is the only place that guess is made.
        out.append(Block(start=when.replace(minute=0, second=0, microsecond=0),
                         turns=[t]))
    return out


@dataclass
class Report:
    blocks: list[Block]
    hours: float = WINDOW_HOURS
    now: datetime | None = None

    @property
    def envelope(self) -> Block | None:
        """The heaviest block on record: a floor under capacity, not the cap.

        This block was served. That is all it proves. It is used as the
        comparison because it is the only capacity statement the data supports,
        and every place it is printed says so.
        """
        return max(self.blocks, key=lambda b: b.tokens) if self.blocks else None

    @property
    def active(self) -> Block | None:
        """The block still open, if the last turn is inside a live window."""
        if not self.blocks or self.now is None:
            return None
        b = self.blocks[-1]
        return b if self.now < b.end else None

    @property
    def carry_share(self) -> float:
        ctx = sum(b.context_tokens for b in self.blocks)
        return sum(b.carry_tokens for b in self.blocks) / ctx if ctx else 0.0

    def median_slope(self) -> float | None:
        """The typical within-block penalty for being late in the window.

        Median rather than mean: one 900-turn block with a 20x ratio should not
        become the finding. Blocks too short to measure are excluded rather than
        counted as flat, which would drag the figure toward 1.0 by including
        exactly the blocks where the effect has not had room to appear.
        """
        from adder.util.stats import median

        rs = [s[0] for s in (b.slope() for b in self.blocks) if s is not None]
        return median(rs) if rs else None

    def rolling(self, days: float = 7.0) -> tuple[int, datetime] | None:
        """`(tokens, start)` of the heaviest span of `days` on record.

        The weekly cap needs a comparison and a calendar week cannot supply one,
        because the reset day is not published. Anchoring the buckets anywhere
        makes every total depend on the anchor: shift it by a day and the
        heaviest week changes. A sliding window has no anchor, so the peak it
        finds is a property of the workload rather than of a choice this module
        made -- and it is the same floor-under-capacity argument as `envelope`,
        one level up.

        Two things it is not. The published cap is metered in *compute hours*,
        which a transcript does not record, so tokens are a proxy and the ratio
        between them is not constant across models. And a span that was served
        proves only that it was served.
        """
        if not self.blocks:
            return None
        span = timedelta(days=days)
        best, at = 0, self.blocks[0].start
        for i, anchor in enumerate(self.blocks):
            total = 0
            for b in self.blocks[i:]:
                if b.start - anchor.start >= span:
                    break
                total += b.tokens
            if total > best:
                best, at = total, anchor.start
        return best, at

    def trailing(self, days: float = 7.0) -> int | None:
        """Tokens read in the `days` up to `now`. None without a `now`."""
        if self.now is None or not self.blocks:
            return None
        cut = self.now - timedelta(days=days)
        return sum(b.tokens for b in self.blocks if b.start >= cut)

    def projected(self) -> tuple[int, float] | None:
        """`(tokens by close, share of the envelope)` for the open block.

        A straight-line extrapolation of the observed burn rate, which is an
        under-estimate whenever the slope above is greater than 1 -- the later
        turns cost more than the ones the rate was measured on. Stated rather
        than corrected: a projection with two models stacked inside it is harder
        to check than one that is honestly the simpler of the two.
        """
        b = self.active
        env = self.envelope
        if b is None or self.now is None:
            return None
        mins_left = max(0.0, (b.end - self.now).total_seconds() / 60.0)
        total = int(b.tokens + b.burn * mins_left)
        share = total / env.tokens if env and env.tokens else 0.0
        return total, share

    def to_json(self) -> dict:
        env, act, proj = self.envelope, self.active, self.projected()
        return {
            "window_hours": self.hours,
            "blocks": len(self.blocks),
            "carry_share": round(self.carry_share, 4),
            "median_slope": (None if self.median_slope() is None
                             else round(self.median_slope(), 2)),
            "envelope": None if env is None else {
                "start": env.start.isoformat(), "tokens": env.tokens,
                "turns": len(env.turns),
                "note": "largest block observed; a lower bound on capacity, not a cap",
            },
            "active": None if act is None else act.to_json(),
            "projected": None if proj is None else {
                "tokens_by_close": proj[0],
                "share_of_envelope": round(proj[1], 3),
                "note": "straight-line from the observed burn rate; an "
                        "under-estimate while the slope exceeds 1",
            },
            "week": {
                "peak_tokens": None if self.rolling() is None else self.rolling()[0],
                "peak_start": (None if self.rolling() is None
                               else self.rolling()[1].isoformat()),
                "trailing_tokens": self.trailing(),
                "note": "heaviest sliding 7 days, not a calendar week: the reset "
                        "day is not published, so an anchored total would depend "
                        "on the anchor. The cap itself is metered in compute "
                        "hours, which a transcript does not record",
            },
            "recent": [b.to_json() for b in self.blocks[-12:]],
        }


def build(sessions: dict, *, hours: float = WINDOW_HOURS,
          now: datetime | None = None) -> Report:
    return Report(blocks(sessions, hours=hours), hours=hours, now=now)


def render(rep: Report) -> str:
    from adder.util.render import money, table

    out = [f"  Metering windows, reconstructed at {rep.hours:g}h", ""]
    if not rep.blocks:
        out.append("  No timestamped turns in range. Nothing to place in a window.")
        return "\n".join(out) + "\n"

    rows = []
    for b in rep.blocks[-12:]:
        s = b.slope()
        rows.append([
            b.start.strftime("%Y-%m-%d %H:%M"),
            f"{len(b.turns):,}",
            f"{b.tokens:,}",
            f"{b.new_tokens:,}",
            f"{b.carry_share:.0%}",
            f"{b.burn:,.0f}",
            "-" if s is None else f"{s[0]:.1f}x",
            money(b.cost()),
        ])
    out += table(rows, ["window opened", "turns", "read", "new", "carry",
                        "tok/min", "slope", "api $"], align="<>>>>>>>")
    out.append("")
    out.append("  `read` counts every token the model had to take in, cache reads")
    out.append("  included, because that is what the meter sees. `new` is the part")
    out.append("  that had never been read before. A slope under 1.0 is a window")
    out.append("  containing a restart.")
    out.append("")

    out.append(f"  Across {len(rep.blocks):,} windows, {rep.carry_share:.0%} of every "
               f"token read was context")
    out.append("  already read before -- on a plan, that is quota spent on text you "
               "had already")
    out.append("  paid for once.")
    slope = rep.median_slope()
    if slope is not None:
        out.append("")
        out.append(f"  A turn late in a window costs {slope:.1f}x what an early one "
                   f"does (median")
        out.append(f"  across windows with at least {MIN_SLOPE_TURNS} turns). The "
                   f"window does not hold")
        out.append("  a number of turns; it holds a number of turns at the size your "
                   "context")
        out.append("  reaches. Restarting resets the size, which is why `adder "
                   "handoff` is a")
        out.append("  quota decision here and not only a cost one.")

    env = rep.envelope
    if env is not None:
        out.append("")
        out.append(f"  Heaviest window on record: {env.tokens:,} tokens over "
                   f"{len(env.turns):,} turns,")
        out.append(f"  opening {env.start.strftime('%Y-%m-%d %H:%M')}. That window "
                   f"was served, so it is a")
        out.append("  floor under what your plan allows -- not the limit. The limit is "
                   "not published,")
        out.append("  varies by model, and drains faster in peak hours.")

    peak, trail = rep.rolling(), rep.trailing()
    if peak is not None:
        out.append("")
        out.append(f"  Heaviest 7 days on record: {peak[0]:,} tokens, from "
                   f"{peak[1].strftime('%Y-%m-%d')}.")
        if trail is not None:
            share = trail / peak[0] if peak[0] else 0.0
            out.append(f"  The last 7 days: {trail:,} tokens, {share:.0%} of that.")
        out.append("  A sliding span, not a calendar week — the reset day is not "
                   "published, so")
        out.append("  an anchored total would depend on where the anchor was put. The "
                   "weekly cap")
        out.append("  is metered in compute hours, which a transcript does not record, "
                   "so this")
        out.append("  is a proxy for it and not a reading of it.")

    act, proj = rep.active, rep.projected()
    if act is not None and proj is not None and rep.now is not None:
        mins = max(0.0, (act.end - rep.now).total_seconds() / 60.0)
        out.append("")
        out.append(f"  Open now: {act.tokens:,} tokens in, {mins:,.0f} min to close at "
                   f"{act.end.strftime('%H:%M')}.")
        out.append(f"  At the observed {act.burn:,.0f} tok/min that reaches "
                   f"~{proj[0]:,} by close,")
        out.append(f"  {proj[1]:.0%} of the heaviest window you have run. The rate is "
                   f"measured on turns")
        out.append("  smaller than the ones still to come, so read it as a floor.")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    import argparse
    from datetime import timezone

    from adder.core.filters import add_arguments as add_window
    from adder.core.filters import load as load_window

    ap = argparse.ArgumentParser(
        prog="adder limits",
        description="the five-hour metering window, and what the carry costs it")
    add_window(ap)
    ap.add_argument("--hours", type=float, default=WINDOW_HOURS, metavar="H",
                    help="window length to reconstruct (default: %(default)s)")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    a = ap.parse_args(argv)

    sessions, _w = load_window(a)
    rep = build(sessions, hours=a.hours, now=datetime.now(timezone.utc))
    if a.json:
        print(json.dumps(rep.to_json()))
        return 0
    print()
    print(render(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
