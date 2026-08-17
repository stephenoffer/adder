"""What an agent session actually is: mostly speculation, and priced as such.

The framing this report takes seriously
---------------------------------------
Every other view here treats a session as a sequence of billable turns. That is
true and it is not useful, because it describes the *bill* rather than the
*behaviour* that produced it. An agent working on a task does not execute a
plan. It speculates: it fires a burst of probes at the codebase, most of which
are wrong or redundant, keeps what survives, and repeats. The turns are a
by-product of that search.

Reading a session that way turns four properties of the search into things you
can measure off local disk, and each one is a lever with a dollar sign:

* **Scale** -- how many probes per unit of progress, and how many run in
  parallel. High scale is not waste on its own; it is what makes the search
  work at all. It only becomes waste when the next three go wrong.
* **Heterogeneity** -- the mix of exploration, solution formulation, and
  validation. A session that is 90% exploration has not started; one that is
  90% validation is thrashing on a fix that does not hold.
* **Redundancy** -- the share of probes that repeat one already made. This is
  the expensive one, and it is invisible in every per-turn view: re-reading the
  same file for the sixth time costs almost nothing on the turn that does it
  and then gets re-read as prefix on every turn afterwards.
* **Steerability** -- how much a single human sentence collapses the search.
  If a hint cuts probe volume in half, the cheapest intervention available is
  not a smaller model, it is a better first message.

Why redundancy is the number to look at first
---------------------------------------------
Redundancy is high, it is measurable exactly (two probes with the same tool and
the same target are the same probe), and unlike the other three it has no
upside. Exploration that repeats itself has already paid for the answer once.

The dollars attached to it use the same rule as `adder tools`: the measured
accumulated re-read pool from `debt.decompose_read_cost`, apportioned by share
of context growth. So the figure is bounded by money that was really spent and
can never exceed it. It is not a projection of what redundancy "could" cost.

What this does not measure
--------------------------
Whether a probe was *useful*. A read that returns nothing surprising still had
to happen; an agent cannot know the file was irrelevant until it looks. Nothing
here should be read as "this probe was waste" except in the redundancy section,
where the same probe was made twice and the second one certainly was.

Steerability is observational, not an experiment. It compares probe volume
before and after a human turn in sessions that had one. Sessions where a human
interrupts are not a random sample -- people interrupt when the agent is
visibly floundering, which is exactly when probe volume was about to fall
anyway. The report says so where it prints the number, and the honest version
of the experiment is `adder ab`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

from adder.core.filters import root_of as _root_of
from adder.core.trace import DEFAULT_ROOT, load_sessions, transcripts
from adder.util import render
from adder.util.records import mapping
from adder.util.stats import bootstrap_ci, mean, median, quantile, share
from adder.util.text import est_tokens, flatten_text

# Which phase of the search a tool call belongs to. The split is coarse on
# purpose: finer categories would need to know what the call was *for*, which
# the transcript does not record, and a confident wrong phase label is worse
# than a coarse right one.
EXPLORE = "explore"
FORMULATE = "formulate"
VALIDATE = "validate"
PHASES = (EXPLORE, FORMULATE, VALIDATE)

PHASE_OF: dict[str, str] = {
    "Read": EXPLORE,
    "Grep": EXPLORE,
    "Glob": EXPLORE,
    "WebFetch": EXPLORE,
    "WebSearch": EXPLORE,
    "NotebookRead": EXPLORE,
    "Task": EXPLORE,          # delegated exploration is still exploration
    "Edit": FORMULATE,
    "Write": FORMULATE,
    "NotebookEdit": FORMULATE,
    "MultiEdit": FORMULATE,
}

# A `Bash` call is the ambiguous one: it is validation when it runs tests and
# exploration when it lists a directory. Classified by what it runs rather than
# by its name, because the two have very different meanings for the phase mix.
_VALIDATE_CMDS = (
    "pytest", "test", "npm t", "yarn t", "go test", "cargo test", "make check",
    "make test", "ruff", "lint", "mypy", "tsc", "build", "make lint",
)
_EXPLORE_CMDS = ("ls", "cat", "head", "tail", "find", "grep", "rg", "git log",
                 "git status", "git diff", "wc", "tree", "pwd", "which")


def phase_of(tool: str, target: str) -> str:
    """Which phase this probe belongs to. Unknown tools count as exploration."""
    if tool == "Bash":
        cmd = target.strip().lower()
        if any(k in cmd for k in _VALIDATE_CMDS):
            return VALIDATE
        if any(cmd.startswith(k) for k in _EXPLORE_CMDS):
            return EXPLORE
        return VALIDATE if cmd else EXPLORE
    return PHASE_OF.get(tool, EXPLORE)


def target_of(tool: str, inp: dict) -> str:
    """The thing a probe was aimed at, normalised so repeats compare equal.

    Two `Read`s of the same path are the same probe even if one passed an
    offset, because the expensive part -- the result landing in context and
    being re-read on every later turn -- happened both times. A `Bash` is
    identified by its whole command line: `pytest -x tests/test_a.py` and
    `pytest -x tests/test_b.py` are genuinely different probes.
    """
    if not isinstance(inp, dict):
        return ""
    for key in ("file_path", "notebook_path", "path", "url", "command",
                "pattern", "query", "description", "prompt"):
        v = inp.get(key)
        if isinstance(v, str) and v.strip():
            text = " ".join(v.split())
            # A prompt or command can be arbitrarily long; the head of it is
            # identity enough and keeps the dedup map bounded.
            return text[:200]
    return ""


@dataclass
class Probe:
    """One tool call, with what it aimed at and what it dragged back."""

    session: str
    tool: str
    target: str
    phase: str
    turn_index: int
    sidechain: bool
    result_tokens: int = 0


@dataclass
class Scan:
    """Every probe in the window, plus the human turns that steered them."""

    probes: list[Probe] = field(default_factory=list)
    files: int = 0
    # session -> sorted turn indices at which a human said something that was
    # not a tool result. The steer points.
    steers: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    assistant_turns: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def n(self) -> int:
        return len(self.probes)

    @property
    def result_tokens(self) -> int:
        return sum(p.result_tokens for p in self.probes)


def _is_human_text(rec: dict) -> bool:
    """A real human turn, not the transcript's echo of a tool result.

    Claude Code writes tool results as `user` records. Counting those as human
    input would put a "steer" between every single probe and make the
    steerability number meaningless -- which is the bug this predicate exists
    to prevent.
    """
    if rec.get("type") != "user":
        return False
    if rec.get("isMeta") or rec.get("isSidechain"):
        return False
    content = mapping(rec, "message").get("content")
    if isinstance(content, str):
        return bool(content.strip())
    if not isinstance(content, list):
        return False
    kinds = {b.get("type") for b in content if isinstance(b, dict)}
    return bool(kinds - {"tool_result"})


def scan(root: Path | str = DEFAULT_ROOT, *, window=None) -> Scan:
    """Walk the transcripts once, pulling probes, their results, and steers.

    A separate pass from `trace.iter_file` on purpose: that one deduplicates by
    `message.id` because it is pricing turns, and deduplication is exactly wrong
    for the **probes**. Two content blocks in one message are two probes fired
    in parallel, and collapsing them would erase the fan-out this report is
    trying to measure.

    The **turn count** is the opposite case, and conflating the two was a bug.
    Claude Code writes one record per content block, so counting a turn per
    record inflated `assistant_turns` 1.74x here -- 50,145 records against
    28,784 real turns, almost exactly the 1.78x that `trace.iter_file` exists to
    undo. `fan_out` divides probes by that number, so probes-per-turn came out
    42% low. The turn ordinal was per-*file* as well, so two files sharing a
    session id collided and `max_in_one_turn` reported 103 parallel probes that
    were never in the same turn.

    So: probes are deduplicated only by their own **block id**, which keeps the
    fan-out (two blocks in one message are two ids, and two probes) while
    dropping a replay (the same block id restated in a resumed session's new
    transcript is one probe seen twice). Turns are counted per `message.id`,
    per session -- and both of those sets, and the pending map, live for the
    whole scan rather than per file, because a resumed session is two files and
    a sidechain answers a call issued in another one.
    """
    out = Scan()
    pending: dict[str, Probe] = {}
    seen_msg: set[tuple[str, str]] = set()
    seen_probe: set[str] = set()
    for path in transcripts(root):
        out.files += 1
        project = path.parent.name
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not isinstance(rec, dict):
                continue
            if window is not None and not window.keeps_record(rec, project):
                continue
            session = str(rec.get("sessionId") or path.stem)
            if rec.get("type") == "assistant":
                msg = mapping(rec, "message")
                # One turn per message, however many records carry it. The
                # ordinal is the session's running turn count, so every probe
                # fired in the same message shares it and `max_in_one_turn`
                # measures fan-out rather than record layout.
                mid = str(msg.get("id") or "")
                if not mid or (session, mid) not in seen_msg:
                    out.assistant_turns[session] += 1
                    if mid:
                        seen_msg.add((session, mid))
                turn = out.assistant_turns[session]
                for block in (msg.get("content") if isinstance(msg.get("content"), list) else []):
                    if not isinstance(block, dict) or block.get("type") != "tool_use":
                        continue
                    tool = str(block.get("name") or "")
                    if not tool:
                        continue
                    uid = str(block.get("id") or "")
                    if uid:
                        if uid in seen_probe:
                            continue     # a replay, not a second probe
                        seen_probe.add(uid)
                    target = target_of(tool, block.get("input") or {})
                    probe = Probe(
                        session=session,
                        tool=tool,
                        target=target,
                        phase=phase_of(tool, target),
                        turn_index=turn,
                        sidechain=bool(rec.get("isSidechain")),
                    )
                    out.probes.append(probe)
                    if uid:
                        pending[uid] = probe
            elif rec.get("type") == "user":
                if _is_human_text(rec):
                    # The turn this steer follows, in the same per-session
                    # ordinal the probes carry.
                    out.steers[session].append(out.assistant_turns.get(session, 0))
                content = mapping(rec, "message").get("content")
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") != "tool_result":
                            continue
                        probe = pending.pop(str(block.get("tool_use_id") or ""), None)
                        if probe is not None:
                            probe.result_tokens = est_tokens(
                                flatten_text(block.get("content")))
    return out


# --- the four properties ---------------------------------------------------

@dataclass
class Redundancy:
    """Repeats, and what fraction of the re-read bill they account for."""

    probes: int = 0
    distinct: int = 0
    repeats: int = 0
    repeat_tokens: int = 0
    total_tokens: int = 0
    worst: list[tuple[str, int, int]] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return share(self.repeats, self.probes)

    @property
    def token_rate(self) -> float:
        return share(self.repeat_tokens, self.total_tokens)


def redundancy(sc: Scan, *, top: int = 8) -> Redundancy:
    """How much of the search was a probe already made.

    Scoped per session. The same file read in two different sessions is not a
    repeat -- the second session has its own context and genuinely has to look.
    Only a repeat *within* one context is paying twice for one answer.
    """
    seen: set[tuple[str, str, str]] = set()
    counts: Counter[tuple[str, str, str]] = Counter()
    tokens: Counter[tuple[str, str, str]] = Counter()
    rep = Redundancy(probes=sc.n, total_tokens=sc.result_tokens)
    for p in sc.probes:
        if not p.target:
            continue
        key = (p.session, p.tool, p.target)
        counts[key] += 1
        tokens[key] += p.result_tokens
        if key in seen:
            rep.repeats += 1
            rep.repeat_tokens += p.result_tokens
        else:
            seen.add(key)
    rep.distinct = len(seen)
    worst = sorted(
        ((k, c, tokens[k]) for k, c in counts.items() if c > 1),
        key=lambda kv: -kv[2],
    )[:top]
    rep.worst = [(f"{k[1]} {k[2]}", c, t) for k, c, t in worst]
    return rep


def phase_mix(sc: Scan) -> dict[str, float]:
    """Share of probes in each phase. Sums to 1.0, or is empty."""
    counts = Counter(p.phase for p in sc.probes)
    total = sum(counts.values())
    return {ph: share(counts.get(ph, 0), total) for ph in PHASES} if total else {}


def phase_interleaving(sc: Scan) -> float:
    """How much the phases overlap rather than run in sequence, 0 to 1.

    Measured as the rate at which consecutive probes in a session change phase.
    A session that explored, then wrote, then tested scores near 0. One that
    interleaves all three -- which is what agent traces actually look like --
    scores high. It matters because a serialised trace can be cached and
    batched, and an interleaved one cannot.
    """
    by_session: dict[str, list[str]] = defaultdict(list)
    for p in sc.probes:
        by_session[p.session].append(p.phase)
    switches = comparisons = 0
    for phases in by_session.values():
        for a, b in pairwise(phases):
            comparisons += 1
            switches += a != b
    return share(switches, comparisons)


def fan_out(sc: Scan) -> dict[str, float]:
    """Probes per assistant turn, and the share that ran inside a subagent."""
    per_turn: list[float] = []
    by_session: Counter[str] = Counter(p.session for p in sc.probes)
    for session, count in by_session.items():
        turns = sc.assistant_turns.get(session, 0)
        if turns:
            per_turn.append(count / turns)
    parallel = Counter((p.session, p.turn_index) for p in sc.probes)
    return {
        "per_turn_median": median(per_turn),
        "per_turn_p90": quantile(per_turn, 0.9),
        "max_in_one_turn": float(max(parallel.values())) if parallel else 0.0,
        "sidechain_share": share(sum(1 for p in sc.probes if p.sidechain), sc.n),
        "sessions": float(len(by_session)),
    }


@dataclass
class Steer:
    """Probe volume either side of a human interjection."""

    before: float = 0.0
    after: float = 0.0
    n: int = 0
    ci: tuple[float, float] = (0.0, 0.0)

    @property
    def reduction(self) -> float:
        return (self.before - self.after) / self.before if self.before > 0 else 0.0

    @property
    def measured(self) -> bool:
        """False when there were too few steers to say anything."""
        return self.n >= 5


def steerability(sc: Scan, *, span: int = 3) -> Steer:
    """Probes per turn in the `span` turns before a steer, versus after.

    Paired per steer, so the interval is over steers rather than over turns:
    two sessions with wildly different probe rates do not widen it, and one
    session with forty steers does not pretend to be forty independent
    observations of the same size as one session with one.
    """
    by_session: dict[str, Counter[int]] = defaultdict(Counter)
    for p in sc.probes:
        by_session[p.session][p.turn_index] += 1

    deltas: list[float] = []
    befores: list[float] = []
    afters: list[float] = []
    for session, turns in sc.steers.items():
        counts = by_session.get(session)
        if not counts:
            continue
        for t in turns:
            before = mean([counts.get(i, 0) for i in range(t - span, t)])
            after = mean([counts.get(i, 0) for i in range(t + 1, t + 1 + span)])
            befores.append(before)
            afters.append(after)
            deltas.append(before - after)
    st = Steer(before=mean(befores), after=mean(afters), n=len(deltas))
    if deltas:
        st.ci = bootstrap_ci(deltas)
    return st


def redundancy_cost(rep: Redundancy, sessions, on=None) -> float:
    """USD of measured re-read spend attributable to duplicate probe results.

    Bounded by the same accumulated pool `adder tools` apportions, and scaled by
    the duplicate share of *all* context growth -- assistant output included.
    Leaving output out of that denominator is the arithmetic error that once
    tripled every attributed figure in the tools report; it is not repeated
    here.
    """
    if not sessions or not rep.repeat_tokens:
        return 0.0
    from adder.measure.spend.debt import decompose_read_cost
    from adder.measure.window.tools import billed_output

    _, _, accumulated = decompose_read_cost(sessions, on)
    growth = rep.total_tokens + billed_output(sessions)
    return accumulated * share(rep.repeat_tokens, growth)


# --- report ----------------------------------------------------------------

def report(root: Path | str = DEFAULT_ROOT, *, window=None, sessions=None,
           top: int = 8, on=None) -> str:
    sc = scan(root, window=window)
    if not sc.n:
        return "  No tool calls found. Nothing to characterise."
    # See `Window.ignores_model`: a probe is a tool_use block and carries no
    # model, so a model filter cannot reach this scan.
    note = ("  note: --model-filter is not applied here — a tool call carries "
            "no model"
            if window is not None and getattr(window, "ignores_model", False)
            else "")

    rep = redundancy(sc, top=top)
    mix = phase_mix(sc)
    fan = fan_out(sc)
    st = steerability(sc)
    if sessions is None:
        sessions = load_sessions(root, use_cache=True)
    dollars = redundancy_cost(rep, sessions, on)

    out: list[str] = []
    out += render.heading("agentic speculation — what the search cost", rule="=")
    out.append(f"  {sc.n:,} probes across {int(fan['sessions']):,} sessions "
               f"({sc.files:,} transcripts)")
    if note:
        out.append(note)
    out.append("")

    out += render.heading("scale")
    out.append(render.kv("probes per turn",
                         f"median {fan['per_turn_median']:.1f}, "
                         f"p90 {fan['per_turn_p90']:.1f}"))
    out.append(render.kv("widest single turn", f"{int(fan['max_in_one_turn'])} probes"))
    out.append(render.kv("run in a subagent", render.pct(fan["sidechain_share"])))

    out.append("")
    out += render.heading("heterogeneity")
    for ph in PHASES:
        out.append(render.kv(ph, render.bar(mix.get(ph, 0.0), 24) + " " +
                             render.pct(mix.get(ph, 0.0))))
    interleave = phase_interleaving(sc)
    out.append(render.kv("phase switches", render.pct(interleave) +
                         " of consecutive probes"))
    if interleave > 0.5:
        out += render.wrap(
            "The phases interleave rather than run in sequence, which is the "
            "normal shape and the reason a session's context never settles: "
            "every switch pulls in a different kind of content.")

    out.append("")
    out += render.heading("redundancy")
    out.append(render.kv("repeated probes",
                         f"{rep.repeats:,} of {rep.probes:,}  "
                         f"({render.pct(rep.rate)})"))
    out.append(render.kv("distinct targets", f"{rep.distinct:,}"))
    out.append(render.kv("duplicate volume",
                         f"{render.tokens(rep.repeat_tokens)} "
                         f"({render.pct(rep.token_rate)} of all results)"))
    if dollars > 0:
        out.append(render.kv("cost of repeats", render.money(dollars)))
    if rep.worst:
        out.append("")
        out += render.table(
            [[name[:56], f"{count}x", render.tokens(tok)] for name, count, tok in rep.worst],
            ["probe", "times", "tokens"],
            align="<>>",
        )

    out.append("")
    out += render.heading("steerability")
    if st.measured:
        lo, hi = st.ci
        out.append(render.kv("probes/turn before", f"{st.before:.2f}"))
        out.append(render.kv("probes/turn after", f"{st.after:.2f}"))
        out.append(render.kv("change", f"{st.before - st.after:+.2f} "
                                       f"[{lo:+.2f}, {hi:+.2f}] over {st.n} steers"))
        if lo > 0:
            out += render.wrap(
                "A human sentence measurably collapses the search. That is the "
                "cheapest lever in this report and it needs no configuration.")
        else:
            out += render.wrap(
                "No measured change in probe volume around a steer: the "
                "interval spans zero.")
    else:
        out.append(f"  Only {st.n} steers in this window — too few to measure.")
    out += render.wrap(
        "OBSERVATIONAL: people interrupt when an agent is visibly floundering, "
        "which is when probe volume was likely to fall anyway. This is a "
        "correlation. `adder ab` is the controlled version.")
    return "\n".join(out)


def to_json(root: Path | str = DEFAULT_ROOT, *, window=None, sessions=None,
            on=None) -> dict:
    sc = scan(root, window=window)
    rep = redundancy(sc)
    fan = fan_out(sc)
    st = steerability(sc)
    if sessions is None:
        sessions = load_sessions(root, use_cache=True)
    return {
        "probes": sc.n,
        "transcripts": sc.files,
        "scale": fan,
        "heterogeneity": {
            "mix": phase_mix(sc),
            "phase_switch_rate": phase_interleaving(sc),
        },
        "redundancy": {
            "repeats": rep.repeats,
            "distinct": rep.distinct,
            "rate": rep.rate,
            "duplicate_tokens": rep.repeat_tokens,
            "token_rate": rep.token_rate,
            "measured_cost_usd": redundancy_cost(rep, sessions, on),
        },
        "steerability": {
            "before": st.before,
            "after": st.after,
            "steers": st.n,
            "ci95": list(st.ci),
            "measured": st.measured,
            "observational": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    from adder.core import filters

    ap = argparse.ArgumentParser(
        prog="adder spec",
        description="Characterise agent sessions as search: scale, mix, "
                    "redundancy, and how much a hint is worth.",
    )
    ap.add_argument("--top", type=int, default=8,
                    help="how many repeated probes to name (default 8)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    filters.add_arguments(ap)
    args = ap.parse_args(argv)
    # `root_of`: the argument if one was given, else the `root`
    # setting. Resolved here so two commands cannot disagree
    # about which transcript directory `adder config` names.
    args.root = str(_root_of(args))
    window = filters.Window.from_args(args)

    if args.json:
        print(json.dumps(to_json(args.root, window=window), indent=2, sort_keys=True))
        return 0
    print(report(args.root, window=window, top=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
