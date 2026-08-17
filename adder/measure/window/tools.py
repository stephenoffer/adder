"""Which tool is expensive, and what it costs after the turn that called it.

Every other view in this repo is organised by model, session, or turn. None of
those name the thing a person can actually change on Monday morning. A tool
call is: an unbounded `Bash` that printed 40K tokens of test output, a `Read`
of a 200K-token lockfile, a `Grep` with no `head`. Those are decisions, and
they are the ones that fill a context.

The number this report exists for
---------------------------------
A tool result is not billed when it arrives. It is billed on **every turn after
it**, as part of the prefix, until a compaction drops it. So a 40K-token `Bash`
result on turn 20 of a 400-turn session is not a 40K-token event -- it is 40K
tokens re-read ~380 times. `adder trace --by tool` prices the turn that called
the tool. This prices what the tool left behind, which is the larger number by
an order of magnitude and the one that changes behaviour.

Attribution is bounded, not projected
-------------------------------------
The tempting arithmetic is `result_tokens x read_rate x remaining_turns`. It
over-states by roughly a third, because compaction drops content and the
projection assumes nothing ever leaves. Instead the **measured** accumulated
cache-read spend (`debt.decompose_read_cost`) is apportioned by each tool's
share of estimated context growth. The total across tools can therefore never
exceed what was actually paid -- the same rule the verbosity lever follows, for
the same reason.

Result sizes are estimated from characters; token counts here are only ever
used to compute a *share* of a measured dollar figure, never reported as a
billed quantity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from adder.core.filters import root_of as _root_of
from adder.core.trace import DEFAULT_ROOT, transcripts
from adder.util.records import mapping
from adder.util.text import est_tokens, flatten_text

# A single result at or above this is worth naming on its own line: it is not a
# habit, it is one command that should have been piped through `head`.
BIG_RESULT_TOKENS = 5_000

# Concrete levers, keyed by tool name. Generic advice ("be more concise") is
# not actionable; these are the specific things that bound each tool's output.
LEVERS: dict[str, str] = {
    "Bash": "pipe through `head -50`, `tail`, or `| wc -l`; redirect build logs to a file",
    "Read": "pass `offset`/`limit`, or delegate the read to a subagent",
    "Grep": "use `-l` or `-c` first, then read only the hits that matter",
    "Glob": "narrow the pattern; a repo-wide glob returns the whole tree",
    "Task": "already delegated — check the subagent's summary is bounded",
    "WebFetch": "ask for the specific section, not the page",
    "NotebookEdit": "edit one cell, not the notebook",
    "Edit": "expected to be small; a large Edit result usually means a failed match",
    "Write": "expected to be small",
}


@dataclass
class ToolStat:
    name: str
    calls: int = 0
    errors: int = 0
    result_tokens: int = 0
    # The part of `result_tokens` that came back inside a subagent. Kept apart
    # rather than filtered out, because the two questions this module answers
    # want different populations: "what does this tool return" is about every
    # call, and "what is it costing you to carry" is about the main context
    # only -- which is the context whose re-read cost is being apportioned.
    sidechain_result_tokens: int = 0
    sizes: list[int] = field(default_factory=list)
    biggest: int = 0
    sessions: set[str] = field(default_factory=set)

    @property
    def main_result_tokens(self) -> int:
        """Result tokens that actually entered the main context."""
        return max(0, self.result_tokens - self.sidechain_result_tokens)

    @property
    def error_rate(self) -> float:
        return self.errors / self.calls if self.calls else 0.0

    @property
    def mean_result(self) -> float:
        """Mean size of a result that was actually observed.

        Over `results_seen`, not `calls`, and the distinction is the one
        `results_seen` was written to make: a call that was issued and never
        answered has no size, so dividing by `calls` understates the mean by
        exactly the unanswered rate. It sat in the same table row as `p90`,
        which is taken over `sizes` -- so the two columns described different
        populations and the mean was always the smaller of them for a reason
        that had nothing to do with the tool.
        """
        return self.result_tokens / self.results_seen if self.results_seen else 0.0

    def p90_result(self) -> float:
        from adder.util.stats import quantile

        return quantile(self.sizes, 0.9)

    @property
    def results_seen(self) -> int:
        """Calls whose result was actually observed.

        A call can be issued and never answered -- the session ended, the
        operator interrupted it. Dividing result volume by `calls` in that case
        understates the mean result size, so the two counts are kept apart.
        """
        return len(self.sizes)


@dataclass
class ToolReport:
    by_tool: dict[str, ToolStat] = field(default_factory=dict)
    total_result_tokens: int = 0
    sidechain_result_tokens: int = 0
    user_tokens: int = 0
    sidechain_user_tokens: int = 0
    assistant_tokens: int = 0           # estimated from text; see `growth()`
    files: int = 0

    @property
    def calls(self) -> int:
        return sum(t.calls for t in self.by_tool.values())

    @property
    def errors(self) -> int:
        return sum(t.errors for t in self.by_tool.values())

    def ranked(self) -> list[ToolStat]:
        return sorted(self.by_tool.values(), key=lambda t: -t.result_tokens)

    def growth(self, assistant_tokens: int | None = None) -> int:
        """Everything that enters a context, which is the only honest denominator.

        Assistant output is by far the largest of the three sources -- around
        two thirds of growth on the author's data -- so leaving it out inflates
        every tool's share by roughly 3x, and inflates the dollars apportioned
        by that share along with it. This was wrong for exactly one release and
        it made `adder doctor` rank a $1,000 finding as a $3,000 one.

        `assistant_tokens` overrides the character estimate with the **billed**
        output count when a caller has the session map to supply it, which is
        the same substitution `context.report` makes and for the same reason:
        billed output is measured, the character estimate is not.

        All three terms are **main-chain**. `billed_output` already excluded
        subagent output from the assistant term, and the other two did not, so
        the denominator mixed two populations: 403,581 subagent result tokens
        and 160,057 subagent user tokens were counted as context growth on this
        machine, against an output term that deliberately left the matching
        subagent output out. A subagent's tokens never entered the main
        context -- that is what delegating bought -- so none of the three
        belong here.
        """
        assistant = self.assistant_tokens if assistant_tokens is None else assistant_tokens
        return (self.total_result_tokens - self.sidechain_result_tokens
                + self.user_tokens - self.sidechain_user_tokens
                + assistant)

    def share_of_growth(self, stat: ToolStat,
                        assistant_tokens: int | None = None) -> float:
        total = self.growth(assistant_tokens)
        return stat.main_result_tokens / total if total else 0.0

    @property
    def total_growth_tokens(self) -> int:
        return self.growth()


def scan(root: Path | str = DEFAULT_ROOT, *, window=None) -> ToolReport:
    """Read every transcript under `root` and attribute results to the tool that asked.

    A `tool_result` block names the `tool_use_id` it answers, not the tool. The
    mapping is built from the `tool_use` block that issued it, which may be in
    an earlier record or an earlier file -- so the map is kept across the whole
    scan rather than per file.
    """
    rep = ToolReport()
    pending: dict[str, tuple[str, str]] = {}   # tool_use_id -> (tool name, session)
    # Deduplication is by *block* id, not message id. Claude Code writes one
    # record per content block and repeats the message envelope on each, so a
    # turn that called three tools appears as three records sharing one
    # message id -- and skipping the repeats by message id throws away two of
    # the three `tool_use` blocks. Doing exactly that attributed 56% of all
    # context growth to an unknown tool, because the results came back
    # referencing ids this scan had never seen.
    seen_use: set[str] = set()
    answered: set[str] = set()

    for path in transcripts(root):
        rep.files += 1
        try:
            fh = path.open(encoding="utf-8", errors="replace")
        except OSError:
            continue
        seen_msg: set[str] = set()
        with fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if window is not None and not window.keeps_record(d, path.parent.name):
                    continue
                typ = d.get("type")
                msg = mapping(d, "message")
                content = msg.get("content")
                session = str(d.get("sessionId") or path.stem)
                side = bool(d.get("isSidechain"))
                blocks = content if isinstance(content, list) else []

                if typ == "assistant":
                    mid = str(msg.get("id") or "")
                    first_record = not (mid and mid in seen_msg)
                    if mid:
                        seen_msg.add(mid)
                    if first_record:
                        # Text is per-message; counting it per record would
                        # multiply it by the number of content blocks.
                        for b in blocks:
                            if (not side and isinstance(b, dict)
                                    and isinstance(b.get("text"), str)):
                                rep.assistant_tokens += est_tokens(b["text"])
                    for b in blocks:
                        if not isinstance(b, dict) or b.get("type") != "tool_use":
                            continue
                        use_id = str(b.get("id") or "")
                        if use_id:
                            if use_id in seen_use:
                                continue
                            seen_use.add(use_id)
                        elif not first_record:
                            # No block id to dedup on; fall back to counting
                            # only the message's first record.
                            continue
                        name = str(b.get("name") or "?")
                        st = rep.by_tool.setdefault(name, ToolStat(name))
                        st.calls += 1
                        st.sessions.add(session)
                        if use_id:
                            pending[use_id] = (name, session)

                elif typ == "user":
                    if isinstance(content, str):
                        rep.user_tokens += est_tokens(content)
                        if side:
                            rep.sidechain_user_tokens += est_tokens(content)
                        continue
                    for b in blocks:
                        if not isinstance(b, dict):
                            continue
                        if b.get("type") != "tool_result":
                            if b.get("type") == "text":
                                rep.user_tokens += est_tokens(b.get("text") or "")
                                if side:
                                    rep.sidechain_user_tokens += est_tokens(
                                        b.get("text") or "")
                            continue
                        use_id = str(b.get("tool_use_id") or "")
                        if use_id and use_id in answered:
                            continue          # replayed transcript; already counted
                        if use_id:
                            answered.add(use_id)
                        name, _ = pending.get(use_id, ("?", session))
                        st = rep.by_tool.setdefault(name, ToolStat(name))
                        n = est_tokens(flatten_text(b.get("content")))
                        st.result_tokens += n
                        st.sizes.append(n)
                        st.biggest = max(st.biggest, n)
                        if b.get("is_error"):
                            st.errors += 1
                        rep.total_result_tokens += n
                        if side:
                            st.sidechain_result_tokens += n
                            rep.sidechain_result_tokens += n
    return rep


def billed_output(sessions) -> int:
    """Main-chain output tokens, as billed. The measured half of the denominator.

    Sidechain output is excluded: a subagent's tokens never entered the main
    context, which is the context whose re-read cost is being apportioned.
    """
    return sum(t.out for s in sessions.values() for t in s.turns if not t.sidechain)


def carried_cost(rep: ToolReport, sessions, on=None) -> dict[str, float]:
    """USD of measured re-read spend attributable to each tool's results.

    Bounded by `debt.decompose_read_cost`'s accumulated pool: the money that was
    really spent re-reading content that was not there at the start of the
    session. Apportioned by each tool's share of **all** context growth --
    assistant output included, using the billed count rather than the character
    estimate wherever the session map makes it available.

    Leaving assistant output out of that denominator was the one arithmetic
    error in this module's first release. It is the largest of the three
    sources, so omitting it inflated every tool's share and every dollar
    apportioned by it roughly threefold. If growth cannot be measured at all,
    every tool gets 0.0 rather than a made-up number.
    """
    from adder.measure.spend.debt import decompose_read_cost

    _, _, accumulated = decompose_read_cost(sessions, on)
    total = rep.growth(billed_output(sessions))
    if not total:
        return {t.name: 0.0 for t in rep.by_tool.values()}
    return {
        t.name: accumulated * (t.main_result_tokens / total)
        for t in rep.by_tool.values()
    }


def report(root: Path | str = DEFAULT_ROOT, sessions=None, *, window=None,
           top: int = 12, on=None) -> str:
    from adder.util.render import money, table, tokens

    rep = scan(root, window=window)
    if not rep.calls:
        return "  No tool calls found to attribute."

    costs = carried_cost(rep, sessions, on) if sessions else {}
    # Same denominator as the dollars, or the two columns describe different
    # populations and the reader has no way to tell.
    assistant = billed_output(sessions) if sessions else None
    ranked = rep.ranked()[:top]

    lines = ["  What each tool leaves in your context", ""]
    lines.append(f"  {rep.calls:,} tool calls across {rep.files:,} transcripts · "
                 f"{tokens(rep.total_result_tokens)} of results")
    if window is not None and getattr(window, "filters_records", False):
        lines.append(f"  filter: {window.describe()}")
    if window is not None and getattr(window, "ignores_model", False):
        lines.append("  note: --model-filter is not applied here — a tool result "
                     "carries no model")
    lines.append("")

    rows = []
    for t in ranked:
        rows.append([
            t.name[:22],
            f"{t.calls:,}",
            tokens(t.result_tokens),
            f"{100 * rep.share_of_growth(t, assistant):.1f}%",
            tokens(t.mean_result),
            tokens(t.p90_result()),
            f"{t.error_rate:.0%}" if t.calls else "",
            money(costs[t.name]) if costs else "",
        ])
    header = ["tool", "calls", "results", "of growth", "mean", "p90", "err",
              "carried $" if costs else ""]
    lines += table(rows, header, align="<>>>>>>>")
    lines.append("")
    if costs:
        lines.append("  `carried $` is measured re-read spend apportioned by share of")
        lines.append("  context growth — bounded by what was actually billed, not projected.")
    else:
        lines.append("  Pass a session map to price the carry; sizes are estimated from text.")

    worst = ranked[0] if ranked else None
    if worst and worst.result_tokens:
        lines.append("")
        lines.append(f"  Largest single result: {tokens(worst.biggest)} from {worst.name}.")
        lever = LEVERS.get(worst.name)
        lines.append(f"  {worst.name} is "
                     f"{100 * rep.share_of_growth(worst, assistant):.0f}% of "
                     f"all context growth — assistant output included.")
        if lever:
            lines.append(f"    → {lever}")

    noisy = [t for t in rep.ranked() if t.results_seen and t.p90_result() >= BIG_RESULT_TOKENS]
    if noisy:
        lines.append("")
        lines.append("  Tools whose p90 result is over "
                     f"{BIG_RESULT_TOKENS:,} tokens — one call in ten is a whole file:")
        for t in noisy[:6]:
            lever = LEVERS.get(t.name, "bound the output at the call site")
            lines.append(f"    {t.name:<14}p90 {tokens(t.p90_result()):>7}   {lever}")

    failing = [t for t in rep.ranked() if t.calls >= 20 and t.error_rate > 0.10]
    if failing:
        lines.append("")
        lines.append("  Tools failing more than 10% of the time — a failed call still")
        lines.append("  costs a full turn, and its error text still enters the context:")
        for t in failing[:6]:
            lines.append(f"    {t.name:<14}{t.error_rate:>6.0%}  ({t.errors:,} of {t.calls:,})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from adder.core import settings
    from adder.core.filters import Window
    from adder.core.filters import add_arguments as add_window
    from adder.core.trace import load_sessions

    ap = argparse.ArgumentParser(
        prog="adder tools",
        description="Attribute context growth and carry cost to the tool that caused it.")
    add_window(ap)
    ap.add_argument("--top", type=int, default=12, metavar="N",
                    help="rows to show (default: %(default)s)")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    a = ap.parse_args(argv)
    # `root_of`: the argument if one was given, else the `root`
    # setting. Resolved here so two commands cannot disagree
    # about which transcript directory `adder config` names.
    a.root = str(_root_of(a))

    window = Window.from_args(a)
    sessions = load_sessions(a.root, use_cache=bool(settings.get("cache")))
    if window.active:
        sessions = window.apply(sessions)

    if a.json:
        rep = scan(a.root, window=window)
        costs = carried_cost(rep, sessions)
        assistant = billed_output(sessions)
        print(json.dumps({
            "calls": rep.calls,
            "result_tokens": rep.total_result_tokens,
            "growth_tokens": rep.growth(assistant),
            "assistant_tokens": assistant,
            "user_tokens": rep.user_tokens,
            "tools": [
                {"name": t.name, "calls": t.calls, "errors": t.errors,
                 "error_rate": round(t.error_rate, 4),
                 "result_tokens": t.result_tokens,
                 "mean_result": round(t.mean_result, 1),
                 "p90_result": round(t.p90_result(), 1),
                 "biggest": t.biggest,
                 "share_of_growth": round(rep.share_of_growth(t, assistant), 4),
                 "carried_cost": round(costs.get(t.name, 0.0), 4)}
                for t in rep.ranked()
            ],
        }))
        return 0

    print()
    print(report(a.root, sessions, window=window, top=a.top))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
