"""Current-session cost analysis.

Answers the question that actually changes behaviour mid-session: "what is this
conversation costing me per turn right now, and what will the next big read cost?"

Everything here is priced from the session's own measured turns, so the advice
adapts to the model, cache TTL, and context actually in play rather than to a
global average.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from adder.core import settings as _settings
from adder.core.trace import DEFAULT_ROOT, Session, iter_file
from adder.measure.session.horizon import Horizon
from adder.measure.session.horizon import load as load_horizon
from adder.measure.spend.debt import debt_multiple
from adder.pricing.cost import (
    Rates,
    admitted_token_cost,
    marginal_turn_cost,
    placement_cost,
)
from adder.pricing.prices import MODELS, intro_expiry
from adder.pricing.prices import UnknownModelError as PricesUnknownModelError
from adder.pricing.registry import context_limit, context_window, limit_str, rate, resolve


def _warm_mult(model: str) -> float:
    """What a warm prefix costs on this model, as a multiple of the input rate.

    The number a realised multiplier should be judged against. It is 0.10 on
    Anthropic, 0.20 on the OpenAI 4.x family, and 1.00 wherever there is no
    prompt cache -- and on that last one the "you are rebuilding the cache"
    warning must never fire, because there is no cache to rebuild and no
    action the reader could take.
    """
    r = Rates.for_model(model)
    return (r.cache_read / r.inp) if r.inp else 1.0

# Descriptive session-length stats, measured on deduplicated transcripts. These
# are for reporting only -- remaining turns comes from `horizon`, because a
# countdown from a median is badly wrong on a heavy-tailed length distribution.
# (The pre-deduplication figures were 607/1159; every multi-block turn was being
# counted once per content block. See `trace.iter_file`.)
M = 1_000_000.0

MEDIAN_SESSION_TURNS = 340
P90_SESSION_TURNS = 759

# Above this share of the model's window, compaction is imminent and the next
# turns are the most expensive of the session.
CONTEXT_PRESSURE = 0.75


def slug_for(cwd: Path | str | None = None) -> str:
    """Claude Code's project directory name for a working directory.

    Non-alphanumeric characters all collapse to '-', not just path separators:
    /Users/stephen.offer/Desktop/x -> -Users-stephen-offer-Desktop-x
    (the dot in a username becomes a dash too).
    """
    p = Path(cwd or os.getcwd()).resolve()
    return re.sub(r"[^A-Za-z0-9]", "-", str(p))


def find_project_dir(cwd: Path | str | None = None, root: Path | str = DEFAULT_ROOT) -> Path | None:
    """Locate the transcript directory, falling back to a case-insensitive match."""
    root = Path(root).expanduser()
    exact = root / slug_for(cwd)
    if exact.is_dir():
        return exact
    want = slug_for(cwd).lower()
    for d in root.iterdir() if root.is_dir() else []:
        if d.is_dir() and d.name.lower() == want:
            return d
    return None


def current_transcript(cwd: Path | str | None = None,
                       root: Path | str = DEFAULT_ROOT) -> Path | None:
    """The file this session is being written to, if there is one.

    Split out of `current_session` because two callers need the path itself:
    the re-read check reads the tool results out of it, and a hook that wants
    to watch the session needs something to stat.
    """
    d = find_project_dir(cwd, root)
    if d is None:
        return None
    # `stat` inside the sort key raced the thing being measured: Claude Code is
    # writing this directory while the hook reads it, so a file can be renamed
    # or removed between the glob and the stat, and `sorted` propagates the
    # OSError out of a hook that has no handler. A file that vanished is a file
    # that is not the current transcript.
    stamped = []
    for f in d.glob("*.jsonl"):
        try:
            stamped.append((f.stat().st_mtime, f))
        except OSError:
            continue
    files = [f for _, f in sorted(stamped, key=lambda kv: kv[0], reverse=True)]
    for newest in files[:3]:          # skip empty/unpriced files, don't merge them
        if any(True for _ in iter_file(newest)):
            return newest
    return None


def current_session(cwd: Path | str | None = None, root: Path | str = DEFAULT_ROOT) -> Session | None:
    """Most recently modified transcript for this working directory.

    Reads only that one file. An earlier version fell back to parsing every
    transcript in the directory and treating the union as one session, which
    reported the sum of unrelated conversations as "this session".
    """
    newest = current_transcript(cwd, root)
    if newest is None:
        return None
    s = Session(newest.stem, newest.parent.name)
    s.turns = list(iter_file(newest))
    return s if s.turns else None


@dataclass
class LiveReport:
    turns: int
    context: int
    spent: float
    per_turn: float
    projected_remaining: int
    projected_total: float
    model: str
    out_per_turn: int = 0
    median_gap: float = 0.0
    ttl: str = "5m"
    # Conditional MEAN remaining turns. `projected_remaining` above is the
    # conditional median, which is the right number to show a person and the
    # wrong one to multiply a cost by: carry cost is linear in remaining turns,
    # so its expectation is set by E[R]. Session length is heavy-tailed here, so
    # the two differ by several fold and the median is the smaller one.
    # See `horizon.mean_remaining`.
    expected_remaining: float = 0.0
    # The multiplier this session actually realized on its input, not the 0.10
    # a warm prefix would have cost. A session that keeps missing the cache
    # pays several times that, and every decision below is linear in it.
    read_mult: float = 0.10
    # What this session's own opening cost, so "restart" can be priced from an
    # observation rather than from a global prior.
    opening_cost: float = 0.0

    @property
    def carry_turns(self) -> float:
        """Remaining turns to PRICE with: the conditional mean, not the median.

        Falls back to the median when no mean was supplied, so a caller that
        builds a report by hand still gets the old number rather than zero.
        """
        return self.expected_remaining or float(self.projected_remaining)

    @property
    def context_pressure(self) -> float:
        """How full the model's window is. Past ~0.75 compaction is imminent."""
        # `context_window`, not `context_limit`: 53 bundled catalog entries
        # publish no window, and `max(1, None)` raises. An unknown window is
        # reported as no pressure rather than as a crash -- the number is a
        # warning threshold, and there is nothing to warn against when the
        # ceiling is unknown.
        limit = context_window(self.model)
        return self.context / limit if limit else 0.0

    @property
    def next_turn_cost(self) -> float:
        return marginal_turn_cost(self.context, self.out_per_turn, self.model)

    @property
    def debt_multiple(self) -> float:
        """What a token written now really costs, vs its sticker price."""
        return debt_multiple(self.carry_turns, self.model)

    def compaction_net(self, *, kept: float = 0.35,
                       on: date | None = None) -> float:
        """USD compacting right now returns over the rest of the session.

        Positive means the rebuild is cheaper than carrying what would be
        dropped. It is the same comparison `adder compact` makes after the
        fact, evaluated against this session's own context and horizon --
        which is the only moment the decision is still available.
        """
        rr = Rates.for_model(self.model, ttl=self.ttl, on=on)
        freed = self.context * (1.0 - kept)
        saving = freed * rr.inp * self.read_mult * self.carry_turns / M
        rebuild = self.context * kept * rr.cache_write / M
        return saving - rebuild

    def restart_net(self, *, handoff_tokens: int = 2_000,
                    on: date | None = None) -> float:
        """USD starting a fresh session returns, against carrying this one.

        A restart drops the whole working context rather than a summarised
        share of it, and pays only for a warm floor plus the handoff -- which
        this session already measured on its first turn.
        """
        r = rate(self.model, on).inp
        keep = min(self.context, handoff_tokens)
        freed = max(0, self.context - keep)
        saving = freed * r * self.read_mult * self.carry_turns / M
        return saving - self.opening_cost

    def context_verdict(self, *, kept: float = 0.35,
                        handoff_tokens: int = 2_000) -> tuple[str, float]:
        """(what to do about the context, what it is worth) — carry on by default.

        Ties go to carrying on. Both alternatives destroy information that is
        not priced anywhere in this repo, so a coin-flip margin is not a reason
        to throw a context away.
        """
        comp = self.compaction_net(kept=kept)
        rest = self.restart_net(handoff_tokens=handoff_tokens)
        best, value = max((("compact", comp), ("restart", rest)),
                          key=lambda kv: kv[1])
        return (best, value) if value > 0 else ("carry on", 0.0)

    def read_cost(self, tokens: int,
                  on: date | None = None) -> tuple[float, float, str]:
        """What reading `tokens` inline costs vs delegating it, from here."""
        inline, sub, d = placement_cost(
            tokens_read=tokens,
            summary_tokens=max(200, tokens // 10),
            remaining_turns=self.carry_turns,
            main_model=self.model,
            on=on,
        )
        return inline, sub, ("delegate" if d else "inline")


def analyse(sess: Session, *, horizon: Horizon | None = None) -> LiveReport:
    """Price the session so far and project the rest of it.

    `remaining` comes from the empirical survivor function, not a countdown from
    a median length. Session length is heavy-tailed and close to memoryless, so
    reaching turn 600 is evidence of being in a LONG session, not of being near
    its end -- a countdown says 7 turns left where the data says ~350.
    """
    spent = sess.cost
    n = sess.n_turns
    # The horizon is indexed on the same population it was built from: the
    # conversation's own turns. Querying a 716-record index against a
    # distribution of 207-turn conversations asks where a session sits using a
    # ruler it was not measured with.
    n_main = len(sess.main_turns)
    if not sess.turns:
        # `main_turns[-1]` below. `current_session` never returns an empty
        # session, but `analyse` is public and `policy`, `handoff` and the
        # prompt hook all call it with whatever they were handed -- an
        # IndexError out of any of those is a traceback where a report should
        # have been.
        return LiveReport(turns=0, context=0, spent=0.0, per_turn=0.0,
                          projected_remaining=0, projected_total=0.0,
                          model=_settings.session_model())
    # The conversation's last turn. Everything below reads the context, model
    # and TTL off it, and a session whose final record is a subagent turn would
    # report that subagent's small context on its cheap model as "this session"
    # -- to the report, to `policy.decide`, and to the guard hook.
    last = sess.main_turns[-1]
    h = horizon if horizon is not None else load_horizon()
    remaining = h.remaining(n_main)
    # The median describes the session; the mean prices it. Both are kept
    # because they answer different questions and collapsing them into one
    # number is what under-priced admission by several fold.
    expected = h.mean_remaining(n_main)
    per_turn = spent / max(1, n)
    from adder.measure.window.carry import measured_read_mult
    from adder.measure.window.prefix import Opening

    # Measured on this session alone: `min_turns=1` because there is only ever
    # one session here, and the alternative is a global average that describes
    # somebody else's cache behaviour.
    mult = measured_read_mult({sess.id: sess}, min_turns=1) or _warm_mult(last.model)
    return LiveReport(
        turns=n,
        context=last.context,
        spent=spent,
        per_turn=per_turn,
        projected_remaining=remaining,
        expected_remaining=expected,
        projected_total=spent + per_turn * expected,
        model=last.model,
        out_per_turn=sess.out_tokens // max(1, n),
        median_gap=sess.median_gap(),
        ttl=last.ttl,
        read_mult=mult,
        opening_cost=Opening.from_session(sess).cost(
            last.model, ttl=last.ttl, handoff_tokens=2_000),
    )


def duplicate_reads(path: Path | None) -> tuple[int, int]:
    """(results, tokens) this session admitted while already holding the content.

    Scans one file, not the tree: this runs in front of a person waiting for a
    prompt, and the whole point is that the answer is about the conversation
    they are in.

    Counts the union of the two views `reread` keeps -- the same call made
    twice, and the same *file* read again however it was read. A session whose
    harness reads with `cat` has none of the first and plenty of the second,
    and counting only identities reported it as clean.
    """
    if path is None:
        return (0, 0)
    from adder.measure.window.reread import scan

    rep = scan(path)
    seen: set[int] = set()
    tokens = 0
    groups = [r.redundant for r in rep.with_repeats(min_tokens=1)]
    groups += [p.unchanged for p in rep.with_path_repeats(min_tokens=1)]
    for admissions in groups:
        for a in admissions:
            if a.seq not in seen:
                seen.add(a.seq)
                tokens += a.tokens
    return (len(seen), tokens)


def render(sess: Session | None, *, sizes: list[int] | None = None,
           on: date | None = None, transcript: Path | None = None) -> str:
    """The session, priced. `on` pins "today" so the expiry notice is testable."""
    if sess is None:
        # A transcript appears once Claude Code has written a turn in this
        # directory, so the honest reading of this state is "not yet", not
        # "broken". Saying which of the two it is costs three lines and is the
        # difference between a new user running the next command and stopping.
        return ("  No transcript found for this directory yet.\n\n"
                "  Claude Code writes one per project the first time you use it "
                "here, so this\n  fills in on your next turn. Nothing needs "
                "configuring.\n\n"
                "  Meanwhile `adder auto on --full` installs the part that does "
                "not need\n  history: it prices a tool call before its result "
                "lands in your context.")
    today = on or date.today()
    r = analyse(sess)
    out = [
        f"  This session: {r.turns:,} turns · {r.context:,} tokens in context · "
        f"${r.spent:,.2f} spent (${r.per_turn:.3f}/turn)",
        f"  Model {r.model} · cache TTL {r.ttl} · "
        f"context {r.context_pressure:.0%} of the {limit_str(r.model)} window",
    ]
    if r.projected_remaining:
        out.append(
            f"  Sessions that reach turn {r.turns:,} typically run ~{r.projected_remaining:,} "
            f"more turns (mean {r.expected_remaining:,.0f}) → ~${r.projected_total:,.2f} "
            f"total at the mean"
        )
    out.append(f"  One more turn at this context costs ~${r.next_turn_cost:.3f}.")

    # A rate change is a re-tune, not a footnote: every threshold in this repo
    # is a ratio of two prices. Only warned about when THIS session's model is
    # the one changing, so it stays a fact about the conversation on screen
    # rather than a standing notice nobody reads.
    # `prices.intro_expiry` is the Claude-only table and raises
    # `UnknownModelError` for everything else -- so `adder live` died outright
    # on any session read from an OpenAI, Gemini or LiteLLM log, which is the
    # whole population `core.ingest` exists to support. There is no
    # introductory rate to warn about outside the first-party table, so the
    # honest answer is "no notice", not an exception.
    try:
        expiry = intro_expiry(r.model)
    except PricesUnknownModelError:
        expiry = None
    if expiry is not None:
        left = (expiry - today).days
        if 0 <= left <= 30:
            base = MODELS[resolve(r.model).id].base
            out.append(
                f"  ⚠ {r.model} leaves its introductory rate on {expiry} "
                f"({left} days): input and output go to ${base.inp}/${base.out} "
                f"per MTok. Every figure above rises with it."
            )

    if r.context_pressure >= CONTEXT_PRESSURE:
        # The two multipliers are this model's provider's, not Anthropic's.
        # Under automatic caching a rebuild is billed as ordinary input and
        # there is no premium at all, so the sentence named a penalty the
        # reader does not pay -- and on an endpoint with no cache it named a
        # 0.10x discount that does not exist either.
        rr = Rates.for_model(r.model, ttl=r.ttl, on=today)
        write_x = (rr.cache_write / rr.inp) if rr.inp else 1.0
        read_x = (rr.cache_read / rr.inp) if rr.inp else 1.0
        out.append("")
        out.append("  ⚠ Context is near the window limit. The next turns are the most")
        out.append("    expensive of the session, and compaction is imminent — which")
        out.append(f"    rebuilds the cache at {write_x:.2f}x instead of "
                   f"reading it at {read_x:.2f}x.")

    verdict, worth = r.context_verdict()
    if verdict != "carry on":
        out.append("")
        out.append(
            f"  Context hygiene: {verdict} — worth ~${worth:,.2f} over the "
            f"~{r.carry_turns:,.0f} turns this session is expected to have left."
        )
        other = ("compacting instead" if verdict == "restart" else "restarting instead")
        alt = (r.compaction_net() if verdict == "restart" else r.restart_net())
        out.append(f"    {other}: ~${alt:,.2f}. Neither is priced for what it "
                   "deletes — detail that gets re-read is paid for twice.")
    dup_ids, dup_tokens = duplicate_reads(transcript)
    if dup_ids:
        out.append(
            f"    Already in context: {dup_ids} thing(s) were admitted twice "
            f"({dup_tokens:,} tokens). The first copy never left (`adder reread`)."
        )
    warm = _warm_mult(r.model)
    if r.read_mult > warm * 1.5:
        out.append(
            f"    This session realized {r.read_mult:.3f}x on its input, not the "
            f"{warm:.2f}x a warm prefix costs: it is rebuilding the "
            "cache, not reading it (`adder cache`)."
        )

    out.append("")
    out.append("  Cost of reading a file into THIS context from here:")
    out.append(f"    {'file size':>12}  {'inline':>10}  {'delegated':>10}   verdict")
    for tokens in (sizes or [5_000, 20_000, 50_000, 150_000]):
        inline, sub, verdict = r.read_cost(tokens, today)
        out.append(f"    {tokens:>10,} tok  ${inline:>9,.3f}  ${sub:>9,.3f}   {verdict}")
    # `today`, not the wall clock. `render` takes `on` precisely so the report
    # is testable across a rate change, and these two lines were the ones that
    # ignored it.
    cost10k = admitted_token_cost(10_000, r.model, r.carry_turns, on=today)
    out.append("")
    out.append(
        f"  Every 10K tokens added to this context now costs ~${cost10k:,.2f} "
        f"over the rest of the session."
    )
    out.append(
        f"  An output token written now costs {r.debt_multiple:.1f}x its sticker price, "
        f"once downstream re-reads are counted."
    )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        prog="adder live",
        description="Price the conversation running in this directory, right now.")
    ap.add_argument("--cwd", default=None, help="working directory to look up")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    ap.add_argument("--tokens", type=int, action="append", metavar="N",
                    help="price reading N tokens into this context (repeatable)")
    a = ap.parse_args(argv)

    transcript = current_transcript(a.cwd)
    sess = current_session(a.cwd)
    if a.json:
        if sess is None:
            print(json.dumps({"error": "no transcript for this directory",
                              "cwd": a.cwd or os.getcwd()}))
            return 1
        r = analyse(sess)
        sizes = a.tokens or [5_000, 20_000, 50_000, 150_000]
        print(json.dumps({
            "session": sess.id,
            "project": sess.project,
            "model": r.model,
            "turns": r.turns,
            "context": r.context,
            "context_limit": context_limit(r.model),
            "context_pressure": round(r.context_pressure, 5),
            "spent": round(r.spent, 4),
            "cost_per_turn": round(r.per_turn, 6),
            "next_turn_cost": round(r.next_turn_cost, 6),
            "read_mult": round(r.read_mult, 5),
            "opening_cost": round(r.opening_cost, 6),
            "compaction_net": round(r.compaction_net(), 4),
            "restart_net": round(r.restart_net(), 4),
            "context_verdict": r.context_verdict()[0],
            "context_verdict_worth": round(r.context_verdict()[1], 4),
            "remaining_turns": r.projected_remaining,
            "expected_remaining_turns": round(r.expected_remaining, 2),
            "projected_total": round(r.projected_total, 4),
            "debt_multiple": round(r.debt_multiple, 3),
            "ttl": r.ttl,
            "median_gap_seconds": round(r.median_gap, 1),
            "reads": [
                {"tokens": n, "inline": round(inline, 5),
                 "delegated": round(sub, 5), "verdict": verdict}
                for n in sizes
                for inline, sub, verdict in [r.read_cost(n)]
            ],
        }))
        return 0

    print()
    print(render(sess, sizes=a.tokens, transcript=transcript))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
