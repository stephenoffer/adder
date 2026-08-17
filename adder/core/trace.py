"""Parse Claude Code transcripts into per-turn cost records.

Transcripts live at ~/.claude/projects/<slug>/<session-uuid>.jsonl, one JSON
object per line. Assistant records carry `message.usage` with the exact token
accounting we need, so cost here is measured, not estimated.

Three correctness details that change the totals
------------------------------------------------
* **Deduplication.** A transcript can replay the same assistant record -- retries,
  resumed sessions, sidechain files that restate parent turns. Counting them
  twice inflates every downstream figure. Records are keyed by message id.
* **Cache TTL.** `usage.cache_creation` breaks writes into 5m and 1h buckets,
  priced at 1.25x and 2.00x. Assuming 5m everywhere understates any session
  using the 1h cache by up to 60% of its write cost.
* **Fast mode.** Opus 5 fast mode bills at $10/$50, double standard. A
  transcript recorded under `/fast` is twice as expensive as the price table
  says unless the tier is read off the record.

Parsing is cached by (path, mtime, size) so the prompt hook can re-read 169
transcripts without a measurable pause.
"""

from __future__ import annotations

import json
import os
import pickle
import statistics
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, datetime
from itertools import pairwise
from pathlib import Path

from adder.core.filters import day_of
from adder.pricing.cost import Rates, turn_cost
from adder.pricing.prices import is_synthetic
from adder.pricing.registry import context_limit, is_known, rate

DEFAULT_ROOT = Path.home() / ".claude" / "projects"
# The built-in location. `cache_path()` is what the read/write pair uses: it
# lets an explicitly-configured `trace_cache` setting win, which this constant
# alone cannot, because it is read once at import -- so `adder config` reported
# a path the code never opened.
CACHE_PATH = Path(
    os.environ.get("ADDER_TRACE_CACHE", Path.home() / ".claude" / ".adder-trace-cache")
)


def cache_path() -> Path:
    """Where the parse cache lives, resolved per call."""
    try:
        from adder.core.settings import configured_path

        return configured_path("trace_cache", CACHE_PATH)
    except Exception:
        return CACHE_PATH
CACHE_VERSION = 8  # bumped when Turn gained agent_id


# What counts as a compaction rather than a wobble.
#
# Any turn whose context is smaller than the previous turn's is a candidate, and
# most candidates are not compactions: measured here, 122 turns out of 20,524
# show a context drop and only 7 are auto-compactions. The rest are small dips
# from branch resumption and sidechain accounting, clustered between 0.65x and
# 0.98x of the previous context.
#
# Auto-compaction is triggered by the context limit, so the detector keys on
# that: the context has to have been near the model's ceiling AND have lost most
# of itself. The 7 real events all sit at 999.5K-999.9K dropping to 4-6%.
#
# Defined here rather than in `measure.window.carry`, where it was written,
# because `Session.compactions` needs it and `core` may not import `measure`.
# `carry` and `compact` import it from here, so there is one definition: a
# second implementation of "was that a compaction" is exactly the disagreement
# this repo has already paid for.
COMPACT_TRIGGER_FRACTION = 0.60          # of the model's context limit
COMPACT_MAX_SURVIVAL = 0.50              # a compaction loses at least half


def is_compaction(prev_ctx: int, ctx: int, model: str) -> bool:
    """A context drop that is an auto-compaction rather than a wobble."""
    if prev_ctx <= 0 or ctx >= prev_ctx:
        return False
    try:
        limit = context_limit(model)
    except Exception:
        return False
    return (prev_ctx >= COMPACT_TRIGGER_FRACTION * limit
            and ctx / prev_ctx <= COMPACT_MAX_SURVIVAL)


@dataclass
class Turn:
    session: str
    project: str
    model: str
    uncached_in: int
    cache_read: int
    cache_write: int
    out: int
    thinking: int
    sidechain: bool
    ts: str | None = None
    ttl: str = "5m"              # dominant cache-write TTL for this turn
    speed: str = "standard"      # "fast" bills at 2x on Opus 5
    msg_id: str = ""             # dedup key
    tools: tuple[str, ...] = ()  # tool names invoked this turn
    effort: str = ""             # reasoning effort the record was produced at
    # Which subagent produced this turn. Present on every sidechain record and
    # empty on main-chain ones. It is the only field that separates one
    # subagent run from the next: they share the parent's session id, so
    # grouping them any other way merges them.
    agent_id: str = ""

    @property
    def context(self) -> int:
        """Tokens the model had to read this turn."""
        return self.uncached_in + self.cache_read + self.cache_write

    @property
    def total_tokens(self) -> int:
        """Everything billed this turn, both directions."""
        return self.context + self.out

    @property
    def cache_hit_rate(self) -> float:
        """Share of this turn's input that was read from cache rather than paid for.

        The per-turn version of the number `adder cache` reports in aggregate.
        A turn with no input at all is 0.0, not a division error.
        """
        return self.cache_read / self.context if self.context else 0.0

    def pricing_date(self, on: date | None = None) -> date | None:
        """The date to price this turn at: the caller's, or the turn's own.

        A recorded turn was billed on the day it ran, so that is the day it has
        to be priced on. Resolving `None` to *today* -- which is what every rate
        lookup does by default -- makes a measurement of the past change when
        the price list does: Sonnet 5's introductory $2/$10 reverts to $3/$15
        after 2026-08-31, and on 1 September every Sonnet turn already on disk
        would have reported 1.5x what it actually cost, with nothing in the
        repo having changed. Passing `on` explicitly still overrides this, which
        is what `cost_on` is for -- "what would this history cost at today's
        rates" is a different and legitimate question.

        Undated turns fall back to the old behaviour; there is nothing better
        to use, and `filters.Window` already reports how many it dropped.
        """
        if on is not None:
            return on
        w = self.when
        return w.date() if w is not None else None

    def rates(self, on: date | None = None, *, ttl: str | None = None) -> Rates:
        """This turn's provider rates, resolved on the date the turn ran.

        The canonical accessor. Every module that prices a recorded turn should
        come through here rather than calling `Rates.for_model(t.model, ...)`
        with an `on` of None, which silently means *today*: `cost()` below
        resolves the turn's own date, so a direct call would price the same turn
        differently from the same turn's `cost()` the day an introductory rate
        expires. They agree only while every rate in the table is stable, which
        is exactly the assumption `prices.py` exists to refuse.

        `ttl` overrides the turn's recorded TTL, for the cache simulator, which
        asks what this turn *would* have cost under a different setting. The
        date and the speed still come from the turn: those are facts about what
        ran, not parameters of the question.
        """
        return Rates.for_model(self.model, ttl=ttl or self.ttl,
                               on=self.pricing_date(on), speed=self.speed)

    def cost(self, on: date | None = None) -> float:
        return turn_cost(
            self.model,
            uncached_in=self.uncached_in,
            cache_read=self.cache_read,
            cache_write=self.cache_write,
            out=self.out,
            ttl=self.ttl,
            speed=self.speed,
            on=self.pricing_date(on),
        )

    def input_cost(self, on: date | None = None) -> float:
        """Everything this turn paid on the input side, at its provider's rates.

        The three input terms are priced separately because on most providers
        they are three different prices, and on some they are the same price.
        Under automatic caching a cache write is billed as plain input, so a
        report that applied Anthropic's 1.25x premium to an OpenAI transcript
        invented 25% of spend that was never charged.
        """
        r = self.rates(on)
        return (
            self.uncached_in * r.inp
            + self.cache_read * r.cache_read
            + self.cache_write * r.cache_write
        ) / 1_000_000

    def output_cost(self, on: date | None = None) -> float:
        return self.out * rate(self.model, self.pricing_date(on),
                               speed=self.speed).out / 1_000_000

    def thinking_cost(self, on: date | None = None) -> float:
        """Output spend on reasoning tokens. The part `effort` controls."""
        return self.thinking * rate(self.model, self.pricing_date(on),
                                    speed=self.speed).out / 1_000_000

    @property
    def when(self) -> datetime | None:
        return _parse_ts(self.ts)


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError, AttributeError):
        return None


def _ordered(dt: datetime) -> datetime:
    """`dt` made comparable with every other turn's timestamp.

    Claude Code stamps UTC with a `Z`, so within one transcript every turn is
    offset-aware and ordering them is trivial. It stops being trivial the
    moment a foreign log is read: `ingest` accepts OpenAI and OTel exports
    whose timestamps carry no offset at all, and one session assembled from
    both shapes made `sorted()`, `min()` and `max()` raise *"can't compare
    offset-naive and offset-aware datetimes"* -- from `gaps()`, which decides
    the cache TTL, and from `started`/`ended`, which every report prints.

    A naive stamp is read as local, the same reading `filters.day_of` already
    gives it, so the two cannot disagree about which day a turn is from.
    """
    return dt if dt.tzinfo is not None else dt.astimezone()


@dataclass
class Session:
    id: str
    project: str
    turns: list[Turn] = field(default_factory=list)

    @property
    def cost(self) -> float:
        return sum(t.cost() for t in self.turns)

    def cost_on(self, on: date | None = None) -> float:
        """Priced at a specific date, so intro-rate expiry is visible."""
        return sum(t.cost(on) for t in self.turns)

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    @property
    def main_turns(self) -> list[Turn]:
        """The turns of the conversation itself, with subagent turns removed.

        A sidechain turn belongs to a different context: it has its own, much
        smaller prefix, its own model, and its own opening. Five separate
        reports had each grown their own `[t for t in s.turns if not
        t.sidechain]`, and the ones that had not were each wrong in a different
        way -- a subagent's prompt reported as the session's irreducible floor,
        a subagent's opening measured as the session's restart cost, the step
        down into a subagent counted as a compaction of the parent.

        Falls back to every turn when a session is nothing but sidechain
        records, so a caller always has something to describe.
        """
        main = [t for t in self.turns if not t.sidechain]
        return main or list(self.turns)

    @property
    def peak_context(self) -> int:
        return max((t.context for t in self.turns), default=0)

    @property
    def avg_context(self) -> int:
        return sum(t.context for t in self.turns) // max(1, len(self.turns))

    @property
    def base_context(self) -> int:
        """Smallest main-chain context: system prompt, tools, CLAUDE.md. Irreducible.

        Main chain only. A subagent starts with its own, much smaller prompt, so
        a session with any delegation reported that subagent's floor as the
        floor of the main conversation -- a number no main-chain turn ever had.
        On this corpus it affects 4 of 105 sessions and halves the figure in
        each. It is not cosmetic: `debt.decompose_read_cost` multiplies this by
        the turn count to get the irreducible baseline, and everything it does
        not claim as irreducible becomes the "addressable pool" that every
        verbosity saving is scaled by. Understating the floor overstates the
        pool.

        A session that is nothing but sidechain turns falls back to those, since
        there is no main chain to describe.
        """
        return min((t.context for t in self.main_turns), default=0)

    @property
    def models(self) -> set[str]:
        return {t.model for t in self.turns}

    @property
    def out_tokens(self) -> int:
        return sum(t.out for t in self.turns)

    @property
    def thinking_tokens(self) -> int:
        return sum(t.thinking for t in self.turns)

    @property
    def started(self) -> datetime | None:
        times = [_ordered(t.when) for t in self.turns if t.when]
        return min(times) if times else None

    @property
    def ended(self) -> datetime | None:
        times = [_ordered(t.when) for t in self.turns if t.when]
        return max(times) if times else None

    @property
    def wall_seconds(self) -> float:
        """Wall-clock span of the session. 0.0 when the turns carry no timestamps.

        Not the same as time spent computing: most of a long session is a human
        reading. It is the denominator for burn rate, and the thing that decides
        whether a 5m cache TTL could ever have survived between turns.
        """
        a, b = self.started, self.ended
        return (b - a).total_seconds() if a and b else 0.0

    def cost_by_model(self, on: date | None = None) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for t in self.turns:
            out[t.model] += t.cost(on)
        return dict(out)

    def gaps(self) -> list[float]:
        """Seconds between consecutive turns, in time order. Drives the TTL decision.

        Sorted, not filtered. A session's turns are not guaranteed monotonic on
        disk -- a resumed conversation is assembled from two transcript files
        and sidechain records interleave with the parent's -- and the previous
        version dropped any pair that went backwards instead of ordering them.
        Dropping a pair does not remove a gap, it *merges* two: turns at 0, 10,
        5, 15 minutes reported two 600s gaps where the real sequence has three
        of 300s. That lands on the wrong side of the 300s 5m TTL boundary, which
        is the single thing this number is used to decide.
        """
        times = sorted(_ordered(t.when) for t in self.turns if t.when)
        return [(b - a).total_seconds() for a, b in pairwise(times)]

    def median_gap(self) -> float:
        g = self.gaps()
        return statistics.median(g) if g else 0.0

    def cache_misses(self) -> list[Turn]:
        """Turns that rewrote more than they read: an expired or invalidated cache.

        The first turn of a session legitimately writes everything. After that, a
        write-dominant turn means the prefix was rebuilt at 1.25x instead of
        being read at 0.10x.

        Each chain is skipped past its own first turn, not just the session's. A
        subagent opens a fresh context and writes all of it, by construction and
        for the same reason -- and that turn sits at some index past zero in the
        combined list, so it was being reported as a rebuild of a cache that had
        never existed.
        """
        out: list[Turn] = []
        for chain in (self.main_turns, [t for t in self.turns if t.sidechain]):
            for t in chain[1:]:
                if t.cache_write > t.cache_read and t.cache_write > 10_000:
                    out.append(t)
        return out

    def compactions(self) -> int:
        """Auto-compactions observed on this session's main chain.

        Uses the same detector as `carry` and `compact` rather than a private
        "context dropped by 40%" rule. That rule is the one `is_compaction`
        exists to replace: it counts branch-resumption dips, which outnumber
        real compactions 15 to 1, and it counted the step down from a parent
        turn to a subagent turn as a compaction of the parent's context. On this
        machine it reported 13 events where there were 9, one of them nothing
        but a main-to-sidechain boundary.

        Walked on the main chain only, for the same reason: a subagent's context
        is not this session's context, and the drop into one is not a loss.
        """
        return sum(1 for a, b in pairwise(self.main_turns)
                   if is_compaction(a.context, b.context, b.model))


def _count(v) -> int:
    """A non-negative token count from a usage field, whatever it holds.

    `x or 0` handles a missing or null field and nothing else. These records
    come from `~/.claude/projects`, which this tool reads and does not write,
    and a single field holding a string or a dict propagated into `Turn` and
    raised a `TypeError` from inside `cost()` -- taking down the read of the
    whole transcript, not just its own record. `ingest._int` already hardened
    the same coercion for foreign logs; a Claude Code record deserves the same
    treatment for the same reason.
    """
    if v is None or isinstance(v, bool):
        return 0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0
    if f != f or f in (float("inf"), float("-inf")):
        return 0
    try:
        n = int(f)
    except (TypeError, ValueError, OverflowError):
        return 0
    return n if n > 0 else 0


def _dominant_ttl(usage: dict) -> str:
    """Which cache TTL this turn mostly used.

    `usage.cache_creation` splits writes into ephemeral_5m/1h buckets. Absent
    that breakdown, 5m is the Claude Code default.
    """
    cc = usage.get("cache_creation") or {}
    if not isinstance(cc, dict):
        return "5m"
    # `_count`, not `or 0`. The bucket values come off the same untrusted
    # record every other usage field does, and a string or a dict in one of
    # them raised a TypeError from the `>` below -- which is raised while
    # building the Turn, so it took down the read of the whole transcript
    # rather than the one record that was malformed.
    five = _count(cc.get("ephemeral_5m_input_tokens"))
    hour = _count(cc.get("ephemeral_1h_input_tokens"))
    return "1h" if hour > five else "5m"


def _speed(msg: dict, usage: dict) -> str:
    s = usage.get("speed") or msg.get("speed")
    return "fast" if s == "fast" else "standard"


def _tools(msg: dict) -> tuple[str, ...]:
    content = msg.get("content")
    if not isinstance(content, list):
        return ()
    # `str()`, not the raw field. A tool name is a group key in `group_by` and
    # a member of the tuple `iter_file` dedups with `dict.fromkeys`, so a
    # record carrying a dict or a list there raised `unhashable type` from two
    # places that are nowhere near the record that caused it.
    return tuple(
        str(b.get("name", "")) for b in content
        if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name")
    )


def _turn_from_record(d: dict, path_stem: str, project: str,
                      skip_unknown: bool,
                      unknown: dict[str, int] | None = None) -> Turn | None:
    if d.get("type") != "assistant":
        return None
    msg = d.get("message")
    if not isinstance(msg, dict):
        return None
    usage = msg.get("usage")
    model = msg.get("model")
    # Both have to be the shape they claim to be, not merely truthy. A record
    # whose `usage` is a string or whose `model` is a number is not a turn this
    # tool can price, and reaching for `.get`/`.startswith` on one raised out
    # of `iter_file`, which has no handler -- so a single malformed record
    # silently ended the read of every turn after it in that transcript.
    if not isinstance(usage, dict) or not usage:
        return None
    if not isinstance(model, str) or not model:
        return None
    if is_synthetic(model):
        # A client-side placeholder, not a billable turn. Counted, never priced.
        if unknown is not None:
            unknown[str(model)] = unknown.get(str(model), 0) + 1
        return None
    if not is_known(model):
        # Counted, not just dropped. A model missing from `prices.py` -- a launch
        # that shipped after this table was last edited -- makes every total
        # here quietly too small, and a quietly-too-small total is the failure
        # mode this project exists to avoid. `adder trace` prints the tally and
        # `--strict` refuses to report at all.
        if unknown is not None:
            unknown[str(model)] = unknown.get(str(model), 0) + 1
        if skip_unknown:
            return None
    details = usage.get("output_tokens_details")
    if not isinstance(details, dict):
        details = {}
    return Turn(
        session=str(d.get("sessionId") or path_stem),
        project=project,
        model=model,
        uncached_in=_count(usage.get("input_tokens")),
        cache_read=_count(usage.get("cache_read_input_tokens")),
        cache_write=_count(usage.get("cache_creation_input_tokens")),
        out=_count(usage.get("output_tokens")),
        thinking=_count(details.get("thinking_tokens")),
        sidechain=bool(d.get("isSidechain")),
        ts=d.get("timestamp") if isinstance(d.get("timestamp"), str) else None,
        ttl=_dominant_ttl(usage),
        speed=_speed(msg, usage),
        msg_id=str(msg.get("id") or d.get("requestId") or d.get("uuid") or ""),
        tools=_tools(msg),
        # Top-level on the record, not inside `message`. It is the only
        # transcript field that says how hard the model was told to think, and
        # `adder effort` re-fits the output-volume priors from it.
        effort=str(d.get("effort") or ""),
        agent_id=str(d.get("agentId") or ""),
    )


def iter_file(path: Path, *, skip_unknown: bool = True,
              unknown: dict[str, int] | None = None) -> Iterator[Turn]:
    """Yield priced assistant turns from one transcript, deduplicated.

    Claude Code writes **one JSONL record per content block**, and every record
    repeats the whole message's `usage`. A turn with a thinking block and two
    tool calls is three records, each reporting the same token counts. Summing
    lines therefore multi-counts most turns: on this machine's 50 transcripts it
    inflates 18,144 real turns to 32,251 and $4,442 of spend to $7,507.

    Records are grouped by `message.id` and the one with the **highest**
    `output_tokens` wins. Partially-streamed records carry a running count that
    only the final record completes -- keeping the first instead of the max
    undercounts output by 2.6% here.
    """
    project = path.parent.name
    try:
        # Explicit UTF-8. Claude Code writes transcripts as UTF-8 regardless of
        # locale; opening with the platform default decoded them as cp1252 on a
        # Windows checkout and as ASCII under `LC_ALL=C`, which silently
        # mangles every non-Latin path and prompt the reports quote back.
        fh = path.open(encoding="utf-8", errors="replace")
    except OSError:
        return
    best: dict[str, Turn] = {}
    # Position of each message's FIRST record, in the same counter space the
    # anonymous records use. An earlier version kept a separate `order` list and
    # looked positions up with `order.index()`, which was both quadratic and
    # wrong: the index into `order` is not comparable with the global counter,
    # so a record with no message id sorted next to an unrelated turn.
    pos: dict[str, int] = {}
    anonymous: list[tuple[int, Turn]] = []
    idx = 0
    with fh:
        for line in fh:
            # Cheap prefilter: most lines are not assistant records.
            if '"assistant"' not in line:
                continue
            try:
                d = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            try:
                t = _turn_from_record(d, path.stem, project, skip_unknown, unknown)
            except Exception:
                # A record this reader cannot make sense of costs one turn, not
                # the rest of the file. There is no handler above this: the
                # generator is consumed inside `load_sessions`, so an exception
                # here ends the read of every later turn in the transcript and
                # reports the shortfall as a smaller bill. The named coercions
                # in `_turn_from_record` cover the shapes seen so far; this
                # covers the ones nobody has seen yet.
                continue
            if t is None:
                continue
            if not t.msg_id:
                anonymous.append((idx, t))
                idx += 1
                continue
            prev = best.get(t.msg_id)
            if prev is None:
                best[t.msg_id] = t
                pos[t.msg_id] = idx
                idx += 1
            elif t.out > prev.out:
                # Later record completed the stream; keep the full accounting
                # but merge the tool calls seen across every block record.
                t.tools = tuple(dict.fromkeys(prev.tools + t.tools))
                best[t.msg_id] = t
            else:
                prev.tools = tuple(dict.fromkeys(prev.tools + t.tools))

    merged = [(pos[m], t) for m, t in best.items()]
    merged.extend(anonymous)
    if not merged:
        # Nothing here looked like a Claude Code assistant record. Before
        # reporting an empty file, try the foreign-format adapters: an OpenAI
        # agent loop, a Gemini session, a LiteLLM proxy log and an OTel export
        # all carry the same usage numbers under different names, and this
        # parser was the only thing stopping adder from analysing them.
        #
        # Tried second rather than first so the Claude Code path -- which also
        # does the per-content-block deduplication the adapters have no view of
        # -- always wins on a file that is genuinely Claude Code.
        from adder.core.ingest import iter_turns as _iter_foreign

        yield from _iter_foreign(path, skip_unknown=skip_unknown, unknown=unknown)
        return
    for _, t in sorted(merged, key=lambda kv: kv[0]):
        yield t


def iter_turns(root: Path | str = DEFAULT_ROOT, *,
               skip_unknown: bool = True,
               unknown: dict[str, int] | None = None) -> Iterator[Turn]:
    """Yield every priced assistant turn under `root`."""
    for path in transcripts(root):
        yield from iter_file(path, skip_unknown=skip_unknown, unknown=unknown)


def transcripts(root: Path | str = DEFAULT_ROOT) -> list[Path]:
    """Every transcript under `root`, or `root` itself if it is one file.

    One definition, because four call sites had written this three ways and one
    of them forgot that a caller may point at a single `.jsonl`.
    """
    root = Path(root).expanduser()
    if root.is_file():
        return [root]
    try:
        found = list(root.rglob("*.jsonl"))
        # Foreign logs are not all named `.jsonl`: an OpenAI or OTel export is
        # commonly a `.json` array. Only picked up when the caller pointed
        # somewhere other than the Claude Code transcript directory, because
        # under that directory a `.json` file is configuration rather than a
        # transcript and reading it as one would invent turns.
        if root != Path(DEFAULT_ROOT):
            found += list(root.rglob("*.json"))
        return sorted(found)
    except OSError:
        return []


# --------------------------------------------------------------------------
# Parse cache. The prompt hook runs on every keystroke-to-submit; re-parsing
# 169 transcripts each time is the difference between 20ms and 3s.
# --------------------------------------------------------------------------

def _cache_load() -> dict:
    try:
        with cache_path().open("rb") as fh:
            blob = pickle.load(fh)
        if blob.get("v") == CACHE_VERSION:
            return blob.get("files", {})
    except Exception:
        # Deliberately broad. The cache is an optimisation and never an input to
        # a number, so no failure to read it may fail the tool. The case that
        # forced this: pickled `Turn` objects name the module they came from, so
        # moving this module made every existing cache unloadable with a
        # `ModuleNotFoundError` -- an exception no narrower clause listed.
        pass
    return {}


def _cache_store(files: dict) -> None:
    """Write the parse cache atomically.

    The temp name carries the pid: several Claude Code sessions share one
    machine, and a fixed `.tmp` path lets two of them clobber each other's
    partial write. `replace` is atomic, so the loser is simply overwritten
    rather than producing a torn file.
    """
    try:
        target = cache_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        try:
            with tmp.open("wb") as fh:
                pickle.dump({"v": CACHE_VERSION, "files": files}, fh, protocol=4)
            tmp.replace(target)
        finally:
            tmp.unlink(missing_ok=True)
    except (OSError, pickle.PicklingError, RecursionError):
        pass


def load_sessions(root: Path | str = DEFAULT_ROOT, *,
                  use_cache: bool | None = None,
                  unknown: dict[str, int] | None = None) -> dict[str, Session]:
    """Group every priced turn into sessions, deduplicated across files.

    The cache memoizes per-file parses by (mtime, size), so only files that
    changed are re-read. Measured on 222 local transcripts that is **2,339ms
    cold against 81ms warm** -- 29x, and the difference between a hook that is
    invisible and one that is uninstalled.

    `use_cache` defaults to the `cache` setting rather than to `False`, and
    that is a correction rather than a preference. The parameter defaulted to
    off while the setting defaulted to on, so caching only happened where a
    caller had remembered to ask for it -- and the paths that most needed it
    had not. `horizon.load` is the one that mattered: it is reached from
    `live.analyse`, which both hooks call, so every prompt submission and every
    guarded read was paying a full 2.3-second re-parse of every transcript on
    the machine to fit a distribution that changes by one session a day.

    Pass `use_cache=False` to force a cold parse; the tests that assert on
    parsing itself do.

    Deduplication happens twice and both passes are load-bearing. `iter_file`
    removes the one-record-per-content-block repetition inside a single
    transcript. This function removes the *cross-file* repetition: a resumed
    session writes a new `.jsonl` that replays earlier turns, and a sidechain
    file restates the parent turn it branched from. Both carry the original
    `message.id`, so both were being counted a second time -- with the same
    inflation mechanism, and in the same direction, as the per-block bug that
    cost 1.78x.

    A message id is only compared within its own session id, because ids are
    only unique per conversation.
    """
    paths = transcripts(root)

    if use_cache is None:
        try:
            from adder.core.settings import get as _setting

            use_cache = bool(_setting("cache"))
        except Exception:
            use_cache = True
    cache = _cache_load() if use_cache else {}
    dirty = False
    sessions: dict[str, Session] = {}
    seen: set[tuple[str, str]] = set()

    for path in paths:
        key = str(path)
        try:
            st = path.stat()
            stamp = (st.st_mtime_ns, st.st_size)
        except OSError:
            continue
        hit = cache.get(key)
        if use_cache and hit and tuple(hit[0]) == stamp and len(hit) >= 3:
            turns, seen_unknown = hit[1], hit[2]
        else:
            seen_unknown: dict[str, int] = {}
            turns = list(iter_file(path, unknown=seen_unknown))
            if use_cache:
                cache[key] = (stamp, turns, seen_unknown)
                dirty = True
        if unknown is not None:
            for m, n in seen_unknown.items():
                unknown[m] = unknown.get(m, 0) + n
        for t in turns:
            if t.msg_id:
                mark = (t.session, t.msg_id)
                if mark in seen:
                    continue
                seen.add(mark)
            s = sessions.get(t.session)
            if s is None:
                s = sessions[t.session] = Session(t.session, t.project)
            s.turns.append(t)

    if use_cache and dirty:
        live = {str(p) for p in paths}
        _cache_store({k: v for k, v in cache.items() if k in live})
    return sessions


@dataclass
class Summary:
    total: float = 0.0
    input_side: float = 0.0
    output_side: float = 0.0
    cache_read_cost: float = 0.0
    cache_write_cost: float = 0.0
    thinking_cost: float = 0.0
    by_model: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    turns_by_model: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    sidechain_cost: float = 0.0
    sidechain_turns: int = 0
    fast_cost: float = 0.0
    fast_turns: int = 0
    n_turns: int = 0
    n_sessions: int = 0
    out_tokens: int = 0
    thinking_tokens: int = 0
    in_tokens: int = 0
    unknown_models: dict[str, int] = field(default_factory=dict)
    synthetic_turns: int = 0

    @property
    def unknown_turns(self) -> int:
        return sum(self.unknown_models.values())

    @property
    def cost_per_turn(self) -> float:
        return self.total / self.n_turns if self.n_turns else 0.0

    @property
    def cost_per_session(self) -> float:
        return self.total / self.n_sessions if self.n_sessions else 0.0


def summarize(root: Path | str = DEFAULT_ROOT, *,
              use_cache: bool = False) -> tuple[Summary, dict[str, Session]]:
    unknown: dict[str, int] = {}
    sessions = load_sessions(root, use_cache=use_cache, unknown=unknown)
    return summarize_sessions(sessions, unknown=unknown), sessions


def summarize_sessions(sessions: dict[str, Session], *,
                       unknown: dict[str, int] | None = None) -> Summary:
    """Aggregate an already-loaded session map.

    Split out from `summarize` so a filtered view -- one project, one week --
    can be summed with exactly the same arithmetic as the whole corpus. When
    the two disagreed, it was always because a caller had reimplemented this
    loop.
    """
    # `unknown` arrives holding both genuinely unpriced models and the client's
    # own `<synthetic>` placeholders. They mean opposite things -- one says the
    # total is too small, the other says nothing at all -- so they are split
    # here rather than reported together.
    raw = dict(unknown or {})
    synthetic = sum(n for m, n in raw.items() if is_synthetic(m))
    s = Summary(
        n_sessions=len(sessions),
        unknown_models={m: n for m, n in raw.items() if not is_synthetic(m)},
        synthetic_turns=synthetic,
    )
    for sess in sessions.values():
        for t in sess.turns:
            c = t.cost()
            r = t.rates()
            s.total += c
            s.input_side += t.input_cost()
            s.output_side += t.output_cost()
            s.cache_read_cost += t.cache_read * r.cache_read / 1_000_000
            s.cache_write_cost += t.cache_write * r.cache_write / 1_000_000
            s.thinking_cost += t.thinking_cost()
            s.by_model[t.model] += c
            s.turns_by_model[t.model] += 1
            s.n_turns += 1
            s.out_tokens += t.out
            s.thinking_tokens += t.thinking
            if t.sidechain:
                s.sidechain_cost += c
                s.sidechain_turns += 1
            s.in_tokens += t.context
            if t.speed == "fast":
                s.fast_cost += c
                s.fast_turns += 1
    return s


# --------------------------------------------------------------------------
# Grouping. "Where did the money go" has four useful answers, and the answer
# that matters depends on what you are about to change: by model if you are
# picking one, by project if you are budgeting, by day if you are checking
# whether last week's change landed, by session if you are hunting an outlier.
# --------------------------------------------------------------------------

GROUPINGS = ("model", "project", "session", "day", "tool", "speed", "ttl")


def _turn_keys(t: Turn, by: str) -> tuple[str, ...]:
    if by == "model":
        return (t.model,)
    if by == "project":
        return (t.project,)
    if by == "session":
        return (t.session,)
    if by == "speed":
        return (t.speed,)
    if by == "ttl":
        return (t.ttl,)
    if by == "day":
        w = t.when
        return (day_of(w).isoformat(),) if w else ("undated",)
    if by == "tool":
        # A turn that calls three tools is attributed to all three, so the
        # column sums above the total. Said in the report rather than hidden,
        # because splitting a turn's cost evenly between its tools would invent
        # an attribution the transcript does not support.
        return t.tools or ("(no tool call)",)
    raise ValueError(f"unknown grouping {by!r}; known: {', '.join(GROUPINGS)}")


@dataclass
class Group:
    key: str
    cost: float = 0.0
    turns: int = 0
    out_tokens: int = 0
    in_tokens: int = 0
    sessions: set[str] = field(default_factory=set)

    @property
    def cost_per_turn(self) -> float:
        return self.cost / self.turns if self.turns else 0.0


def group_by(sessions: dict[str, Session], by: str,
             on: date | None = None) -> list[Group]:
    """Cost broken down by one dimension, most expensive first."""
    out: dict[str, Group] = {}
    for sess in sessions.values():
        for t in sess.turns:
            c = t.cost(on)
            for key in _turn_keys(t, by):
                g = out.get(key)
                if g is None:
                    g = out[key] = Group(key)
                g.cost += c
                g.turns += 1
                g.out_tokens += t.out
                g.in_tokens += t.context
                g.sessions.add(t.session)
    return sorted(out.values(), key=lambda g: -g.cost)
