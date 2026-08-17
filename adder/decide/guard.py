"""The decision the PreToolUse hook makes, extracted so it can be tested.

The guard is the only component in this repository that can *prevent* spend.
Everything else reports on spend that has already happened. That asymmetry is
the reason this module exists as a library rather than as a hundred lines
inside `.claude/hooks/pretooluse_read_guard.py`: the one component whose
failure is silent had the least testable shape in the project.

What changed when it moved here
-------------------------------
The old decision was `size >= 2000 and cost >= $0.25 and delegating is cheaper`,
where `size` came from a list of substrings. Three things were missing, and
each of them is a way for a cost tool to cost money.

**The size was fabricated.** Any command containing `cat ` or `git log` was
assumed to return 15,000 tokens. Measured across 222 local transcripts, the
calls that matched produce a median of 143 -- and the eighteen largest results
in the corpus matched nothing. `adder.core.shapes` replaces the patterns with what
commands of that shape actually returned here.

**The advice was free.** A fire injects `additionalContext` into the
conversation, and that text is admitted to the context like any other token:
written once, re-read on every remaining turn. The old guard fired 903 times at
a median real result size of 143 tokens, so it was spending carry to warn about
reads that were never going to be expensive. `decide` now prices its own
message and refuses to fire unless the expected saving covers it -- the
solvency test `adder ledger` applies to the tool as a whole, applied to the one
mechanism that runs unattended.

**Some savings are certain and were never taken.** Re-reading a file already in
the context, unchanged, admits every one of its tokens a second time for no new
information. It is 19.2% of unbounded reads of text files on this machine. The
guard could not see it because it had no memory between calls, and it is the
cheapest saving in this project: no delegation to model, no horizon to
forecast, nothing to trade off. `GuardState` supplies exactly enough memory for
it, and no more.

The figure was first quoted as 27.4% across all unbounded reads, which counted
138 screenshots. An image is capped near 1,600 tokens whatever its byte size,
so those are cents rather than dollars -- true and misleading, which for a
measurement tool is the same as wrong.

Everything here fails open. A guard that raises is a guard that has stopped
guarding, and the failure is invisible because the tool call still succeeds.
"""
from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from adder.core.filters import root_of as _root_of
from adder.core.shapes import Estimate, SizeModel, bound_lines, empty_model, is_bounded, shape
from adder.pricing.cost import admitted_token_cost, placement_cost
from adder.util.text import est_tokens

GUARDED: tuple[str, ...] = ('Read', 'Bash', 'Grep', 'Glob', 'WebFetch', 'WebSearch', 'Task', 'Agent')
# Watched, but never advised about. A `Write` is not a read and costs nothing to
# admit -- its content is already in the context, because the content *is* the
# tool input. That is exactly why it has to be remembered: reading the file back
# afterwards admits every one of those tokens a second time.
#
# `Edit` is deliberately absent. An edit puts a hunk in the context, not a file,
# so re-reading an edited file can legitimately be the only way to see the rest
# of it. Counting that as waste would be advising against the correct move.
OBSERVED: tuple[str, ...] = (*GUARDED, 'Write')
MAX_SESSIONS = 200
MAX_STATE_AGE_S = 14 * 86400.0
MAX_REMEMBERED_READS = 400
# What a subagent's return to the main context should aim for. A delegated read
# exists to keep tokens *out* of the main context, so a `Task` that hands back
# ten thousand of them has spent the subagent and kept the cost. Used as the
# comparison point for `Task`/`Agent`, where "delegate it" is not available
# advice because delegating is what is already happening.
BRIEF_TARGET_TOKENS = 1000
MAX_MESSAGE_TOKENS = 90
# Cumulative tokens one command shape may admit in a session before the guard
# says so once. Derived rather than picked: across 222 local transcripts, 32
# session-and-shape pairs exceed this, and together they account for **19.7% of
# every Bash result token in the corpus** (1.38M of 7.0M).
#
# This exists because the per-call view is structurally blind to them. The
# largest single channel is `sed -n 'A,Bp'` -- a *bounded* read, correctly waved
# through every time, 246 calls at a 513-token average admitting 126,222 tokens
# into one session. No per-call rule can see that, however well calibrated:
# every one of those calls really was small.
AGGREGATE_TOKENS = 20000
# A file whose mtime lands within this many seconds of a `Write` we saw is
# taken to be that write. Wider than it needs to be for clock granularity, and
# the failure direction is safe: too wide only means the guard stays quiet.
WRITE_SETTLE_S = 5.0

@dataclass(frozen=True)
class Settings:
    """The thresholds, resolved when the guard runs rather than when it imports.

    `render.color_enabled` already learned this lesson in this repo: a constant
    read at import time is one no test can change and no `.adder.json` can
    override, and the guard is where a silently-ignored setting costs most --
    it looks installed either way.

    Resolution goes through `adder.core.settings`, so a project can tune its own guard
    in `.adder.json` without exporting anything, and `adder config` can say
    which layer set each value.
    """
    min_tokens: int = 2000
    min_cost: float = 0.25
    hard_tokens: int = 60000
    block: bool = False
    advice_taken: float = 0.5
    max_fires: int = 15
    state_path: Path = Path.home() / '.claude' / '.adder-guard.json'

    @classmethod
    def resolve(cls, *, cwd=None, env: dict[str, str] | None=None) -> Settings:
        """Effective settings. Never raises: a broken config falls back to defaults.

        The `ROUTER_*` fallback is the unfinished half of the rename, kept
        because someone who configured this tool under its old name and then
        upgraded would otherwise get a guard that reads none of their settings
        and reports no error at all.
        """
        env = os.environ if env is None else env
        values: dict = {}
        sources: dict = {}
        try:
            from adder.core.settings import resolve as _resolve
            for k, v in _resolve(cwd=cwd, env=env).items():
                values[k], sources[k] = (v.value, v.source)
        except Exception:
            pass

        def pick(name: str, default, cast):
            value = values.get(name, default)
            if sources.get(name, 'default') == 'default':
                legacy = env.get(f'ROUTER_{name.upper()}')
                if legacy:
                    try:
                        return cast(legacy)
                    except (TypeError, ValueError):
                        return value
            try:
                return cast(value)
            except (TypeError, ValueError):
                return default

        def as_bool(v) -> bool:
            return v if isinstance(v, bool) else str(v).strip().lower() in ('1', 'true', 'yes', 'on')
        return cls(min_tokens=pick('guard_min_tokens', 2000, int), min_cost=pick('guard_min_cost', 0.25, float), hard_tokens=pick('guard_hard', 60000, int), block=pick('guard_block', False, as_bool), advice_taken=pick('guard_advice_taken', 0.5, float), max_fires=pick('guard_max_fires', 15, int), state_path=Path(str(pick('guard_state', str(Path.home() / '.claude' / '.adder-guard.json'), str))))

@dataclass(frozen=True)
class Verdict:
    """Why the guard did or did not speak. Never a bare bool.

    `reason` is populated on a pass as well as on a fire, because "why was the
    guard silent on that 40K read" is the question asked when someone suspects
    it is not installed, and the answer has to exist.
    """
    fire: bool
    reason: str
    kind: str = ''
    message: str = ''
    ask: bool = False
    tokens: int = 0
    inline: float = 0.0
    delegated: float = 0.0
    saving: float = 0.0
    overhead: float = 0.0
    advice_taken: float = 0.5
    estimate: Estimate | None = None

    @property
    def net(self) -> float:
        """Expected value of firing: the saving, discounted, less the overhead."""
        return self.saving * self.advice_taken - self.overhead

    def clipped(self) -> Verdict:
        """This verdict with its message held to `MAX_MESSAGE_TOKENS`."""
        limit = MAX_MESSAGE_TOKENS * 4
        if len(self.message) <= limit:
            return self
        cut = self.message[:limit - 1].rsplit(' ', 1)[0]
        return replace(self, message=cut + '…')

    def payload(self) -> dict:
        """The hook's stdout, or `{}` when there is nothing to say."""
        if not self.fire:
            return {}
        out: dict = {'hookEventName': 'PreToolUse'}
        self = self.clipped()
        if self.ask:
            out['permissionDecision'] = 'ask'
            out['permissionDecisionReason'] = self.message
        else:
            out['additionalContext'] = self.message
        return {'hookSpecificOutput': out}

@dataclass
class GuardState:
    """What the guard remembers within one session, and nothing beyond it.

    Four facts, each earning its place by preventing a specific waste: `reads`
    catches the re-read of an unchanged file, `advised` stops the guard
    repeating itself about a command shape, and the two running totals keep it
    solvent -- it must not spend more on advice than the advice is worth.

    Deliberately not a cache of anything expensive. Delete this file mid-session
    and the guard degrades to its stateless behaviour, which is the behaviour it
    had before.
    """
    reads: dict[str, float] = field(default_factory=dict)
    wrote: dict[str, float] = field(default_factory=dict)
    admitted: dict[str, int] = field(default_factory=dict)
    shape_calls: dict[str, int] = field(default_factory=dict)
    advised: list[str] = field(default_factory=list)
    fires: int = 0
    saving: float = 0.0
    overhead: float = 0.0
    touched: float = 0.0

    def to_json(self) -> dict:
        return {'reads': self.reads, 'wrote': self.wrote, 'admitted': self.admitted, 'shape_calls': self.shape_calls, 'advised': self.advised[-64:], 'fires': self.fires, 'saving': round(self.saving, 6), 'overhead': round(self.overhead, 6), 'touched': self.touched}

    @classmethod
    def from_json(cls, d: dict) -> GuardState:
        if not isinstance(d, dict):
            return cls()
        reads = d.get('reads')
        wrote = d.get('wrote')
        admitted = d.get('admitted')
        shape_calls = d.get('shape_calls')
        advised = d.get('advised')
        return cls(reads={str(k): float(v) for k, v in reads.items()} if isinstance(reads, dict) else {}, wrote={str(k): float(v) for k, v in wrote.items()} if isinstance(wrote, dict) else {}, admitted={str(k): int(v) for k, v in admitted.items()} if isinstance(admitted, dict) else {}, shape_calls={str(k): int(v) for k, v in shape_calls.items()} if isinstance(shape_calls, dict) else {}, advised=[str(x) for x in advised] if isinstance(advised, list) else [], fires=int(d.get('fires') or 0), saving=float(d.get('saving') or 0.0), overhead=float(d.get('overhead') or 0.0), touched=float(d.get('touched') or 0.0))

    def solvent(self, advice_taken: float=0.5) -> bool:
        """Has the advice been worth more than the advice has cost?"""
        return self.saving * advice_taken >= self.overhead

def load_state(session: str, path: Path | None=None) -> GuardState:
    """This session's state. Never raises; a missing or corrupt file is empty."""
    p = Path(path) if path is not None else Settings.resolve().state_path
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
        return GuardState.from_json((blob or {}).get(session) or {})
    except (OSError, ValueError, TypeError, AttributeError):
        return GuardState()

def save_state(session: str, state: GuardState, path: Path | None=None) -> None:
    """Persist one session's state, atomically, keeping the file bounded.

    Written on every guarded call, so it is one small JSON document rather than
    a log. The replace is atomic because a second hook process may be reading
    it: PreToolUse hooks overlap whenever a turn issues parallel tool calls.
    """
    p = Path(path) if path is not None else Settings.resolve().state_path
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(blob, dict):
            blob = {}
    except (OSError, ValueError):
        blob = {}
    state.touched = time.time()
    blob[session] = state.to_json()
    cutoff = state.touched - MAX_STATE_AGE_S
    blob = {k: v for k, v in blob.items() if k == session or _touched(v) >= cutoff}
    if len(blob) > MAX_SESSIONS:
        ranked = sorted(blob.items(), key=lambda kv: -_touched(kv[1]))
        # Unique per writer. Several Claude Code sessions share one machine
        # and run this from a hook, so a fixed `.tmp` name is a shared
        # mutable path: one writer's `replace` moves the file out from
        # under another's, and the loser raises FileNotFoundError into an
        # `except OSError` that drops it. Measured at 45% of writes lost
        # under three concurrent writers.
        blob = dict(ranked[:MAX_SESSIONS // 2])
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f'{p.name}.{os.getpid()}.tmp')
        try:
            tmp.write_text(json.dumps(blob), encoding="utf-8")
            tmp.replace(p)
        finally:
            tmp.unlink(missing_ok=True)
    except OSError:
        pass

def _num(v) -> float:
    """A finite float from an untrusted field, or 0.0. Never raises."""
    if v is None or isinstance(v, bool):
        return 0.0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if f == f and f not in (float('inf'), float('-inf')) else 0.0

def _touched(entry) -> float:
    """When a stored session was last written, or 0.0 for anything unreadable.

    `blob` is checked for being a dict; its *values* were not. A state file
    whose entries are strings -- a half-written file, a hand-edit, an older
    schema -- made `(v or {}).get('touched')` raise `AttributeError`, and a
    non-numeric `touched` made `float()` raise `ValueError`. Neither is inside
    the `try` around the read, so both escaped `save_state` into a PreToolUse
    hook whose only handler is a blanket `except`: the guard stops remembering
    anything and reports nothing, which is the failure mode this module calls
    its worst because it is invisible.
    """
    return _num(entry.get('touched')) if isinstance(entry, dict) else 0.0

def _mtime(path: str) -> float:
    """Modification time, or -1.0 for anything this cannot stat.

    `ValueError` as well as `OSError`: a path carrying an embedded null raises
    the former, and a tool input is whatever the model emitted. The guard runs
    in a hook whose only handler is a blanket `except`, so an uncaught
    exception here does not surface as an error -- it surfaces as the guard
    quietly not running.
    """
    try:
        return Path(str(path)).stat().st_mtime
    except (OSError, ValueError):
        return -1.0

def _affordable_lines(model: str, remaining_turns: int, cfg: Settings, carry, context_tokens: int) -> int:
    """How many lines this session can read before the carry clears the floor.

    "Bound it" is advice nobody can act on without doing this arithmetic
    themselves, and the arithmetic depends on where in the session they are: at
    turn 20 of a long run it is a few dozen lines, and at the end it is the
    whole file. Solved by search rather than in closed form because
    `admitted_token_cost` may be backed by a fitted carry model whose read
    count is not linear in the horizon.
    """
    from adder.core.shapes import READ_TOKENS_PER_LINE
    per_line = READ_TOKENS_PER_LINE[1]
    lo, hi = (1, 20000)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        cost = admitted_token_cost(int(mid * per_line), model, remaining_turns, carry=carry, context_tokens=context_tokens)
        if cost < cfg.min_cost:
            lo = mid
        else:
            hi = mid - 1
    return lo

def _bounded_hint(tool: str) -> str:
    """The cheaper shape of this exact call, when there is an obvious one."""
    return {'Read': 'read it with offset/limit, or delegate the read', 'Bash': 'pipe it through `head -50`, `wc -l`, or redirect to a file', 'Grep': 'use `-l` or `-c` first, then read only the hits that matter', 'Glob': 'narrow the pattern', 'WebFetch': 'ask for the section you need, not the page', 'WebSearch': 'narrow the query, or fetch the one result you need', 'Task': 'tell the subagent what to return, and how briefly', 'Agent': 'tell the subagent what to return, and how briefly'}.get(tool, 'bound the output')

def _already_known(path: str, state: GuardState, *, now: float | None=None) -> str:
    """Why this file's content is already in the context, or "" if it is not.

    Two ways in, and they are not the same claim. A previous whole read put the
    file in the context and is only still valid if the file has not changed
    since -- re-reading after an edit is the correct thing to do. A previous
    `Write` put the content there as the tool's own input, so the file is known
    unless something outside this session has touched it since.
    """
    seen = state.reads.get(path)
    mtime = _mtime(path)
    if mtime < 0:
        return ''
    if seen is not None and abs(seen - mtime) < 1e-06:
        return 'is already in this context and has not changed on disk since it was read'
    written = state.wrote.get(path)
    if written is not None and mtime <= written + WRITE_SETTLE_S:
        return 'was written by this session, so its content is already in the context'
    return ''

def needs_pricing(tool: str, tool_input: dict, *, sizes: SizeModel | None=None, state: GuardState | None=None, min_tokens: int=2000) -> bool:
    """Is this call worth parsing a transcript for?

    Split out of `decide` because it is the only part that may run on every
    single tool call. Pricing needs the session's model and remaining turns,
    and getting those means reading a transcript -- affordable a few times a
    session, not the 22,761 times this corpus called Bash. Everything here is a
    `stat` at worst.
    """
    tool_input = tool_input or {}
    if tool not in GUARDED:
        return False
    state = state or GuardState()
    sizes = sizes or empty_model()
    if tool == 'Read':
        fp = tool_input.get('file_path')
        if fp and (not tool_input.get('limit')) and (str(fp) in state.reads or str(fp) in state.wrote):
            return True
    if tool == 'Bash':
        cmd = tool_input.get('command') or ''
        sh = shape(cmd)
        if sh in state.advised:
            return False
        if state.admitted.get(sh, 0) >= AGGREGATE_TOKENS:
            return True
        if is_bounded(cmd) and bound_lines(cmd) is None:
            return False
    if tool == 'Grep':
        if (tool_input.get('output_mode') or 'files_with_matches') != 'content':
            return False
        if tool_input.get('head_limit'):
            return False
    return sizes.predict_tool(tool, tool_input).p90 >= min_tokens

def decide(tool: str, tool_input: dict, *, model: str, remaining_turns: int, cfg: Settings | None=None, sizes: SizeModel | None=None, state: GuardState | None=None, carry=None, context_tokens: int=0, p_redo: float=0.0) -> Verdict:
    """Should the guard speak about this call, and what should it say?

    Pure: every input that varies is an argument. The hook supplies the session
    lookup, the file mtimes and the state file; this supplies the judgement,
    which is the part that was previously impossible to test.
    """
    tool_input = tool_input or {}
    cfg = cfg or Settings.resolve()
    state = state or GuardState()
    sizes = sizes or empty_model()
    if tool not in GUARDED:
        return Verdict(False, f'{tool} is not a tool this guard can size')
    if tool == 'Read':
        fp = tool_input.get('file_path')
        if fp and (not tool_input.get('limit')):
            why = _already_known(str(fp), state)
            if why:
                est = sizes.predict_tool('Read', tool_input)
                if est.p90 >= cfg.min_tokens:
                    saving = admitted_token_cost(est.p90, model, remaining_turns, carry=carry, context_tokens=context_tokens)
                    msg = f'[read guard] {Path(str(fp)).name} {why}. Reading it admits ~{est.p90:,} tokens again for ~${saving:,.2f} and no new information.'
                    over = _advice_cost(msg, model, remaining_turns, carry, context_tokens)
                    return _ledger_gate(Verdict(True, 'duplicate read of an unchanged file', kind='duplicate', message=msg, tokens=est.p90, inline=saving, saving=saving, overhead=over, advice_taken=cfg.advice_taken, estimate=est), state, cfg)
    if tool == 'Bash':
        cmd = tool_input.get('command') or ''
        sh = shape(cmd)
        running = state.admitted.get(sh, 0)
        if running >= AGGREGATE_TOKENS and sh not in state.advised:
            return _aggregate_gate(sh, running, state.shape_calls.get(sh, 0), model, remaining_turns, cfg, state, carry, context_tokens)
        if is_bounded(cmd) and bound_lines(cmd) is None:
            return Verdict(False, 'output is bounded by construction')
        if sh in state.advised:
            return Verdict(False, f'already advised about `{sh}` this session')
    if tool == 'Grep':
        if (tool_input.get('output_mode') or 'files_with_matches') != 'content':
            return Verdict(False, 'grep mode returns paths or counts, not content')
        if tool_input.get('head_limit'):
            return Verdict(False, 'the caller already bounded it')
    est = sizes.predict_tool(tool, tool_input)
    tokens = est.p90
    if tool in ('Task', 'Agent'):
        return _brief_gate(tool, tokens, est, model, remaining_turns, cfg, state, carry, context_tokens)
    if tokens < cfg.min_tokens:
        return Verdict(False, f'predicted {tokens:,} tok, below the {cfg.min_tokens:,} floor', tokens=tokens, estimate=est)
    inline = admitted_token_cost(tokens, model, remaining_turns, carry=carry, context_tokens=context_tokens)
    if inline < cfg.min_cost:
        return Verdict(False, f'${inline:,.2f} to carry, below the ${cfg.min_cost:.2f} floor', kind='size', tokens=tokens, inline=inline, estimate=est)
    _, sub, placement = placement_cost(tokens_read=tokens, summary_tokens=max(200, tokens // 10), remaining_turns=remaining_turns, main_model=model, p_redo=p_redo, carry=carry, context_tokens=context_tokens)
    if not placement.ok and placement.saving == 0.0:
        return Verdict(False, placement.reason, kind='size', tokens=tokens, inline=inline, delegated=sub, estimate=est)
    saving = inline - sub
    if saving <= 0:
        return Verdict(False, 'delegating this one costs more than carrying it', kind='size', tokens=tokens, inline=inline, delegated=sub, saving=saving, estimate=est)
    if tool == 'Read':
        lines = _affordable_lines(model, remaining_turns, cfg, carry, context_tokens)
        how = f'read at most ~{lines:,} lines of it (`limit: {lines}`), or delegate the read'
    else:
        how = _bounded_hint(tool)
    msg = f'[read guard] This {tool} admits {est.describe()} to a context re-read ~{remaining_turns:,} more times: ~${inline:,.2f} to carry, vs ~${sub:,.2f} delegated to a subagent. If you need one fact from it, {how}.'
    over = _advice_cost(msg, model, remaining_turns, carry, context_tokens)
    return _ledger_gate(Verdict(True, 'predicted carry exceeds the cost of saying so', kind='size', message=msg, ask=cfg.block and tokens >= cfg.hard_tokens, tokens=tokens, inline=inline, delegated=sub, saving=saving, overhead=over, advice_taken=cfg.advice_taken, estimate=est), state, cfg)

def _brief_gate(tool: str, tokens: int, est, model: str, remaining_turns: int, cfg: Settings, state: GuardState, carry, context_tokens: int) -> Verdict:
    """Price a subagent's *return*, which is the only part that reaches context.

    The subagent's own reads are already outside the main window -- that is
    what delegating bought. What is left to decide is how much it hands back,
    and a return is compared against a brief rather than against an alternative
    placement.
    """
    if tokens <= BRIEF_TARGET_TOKENS:
        return Verdict(False, f'predicted {tokens:,} tok back, already within a brief', kind='brief', tokens=tokens, estimate=est)
    inline = admitted_token_cost(tokens, model, remaining_turns, carry=carry, context_tokens=context_tokens)
    if inline < cfg.min_cost:
        return Verdict(False, f'${inline:,.2f} to carry, below the ${cfg.min_cost:.2f} floor', kind='brief', tokens=tokens, inline=inline, estimate=est)
    bounded = admitted_token_cost(BRIEF_TARGET_TOKENS, model, remaining_turns, carry=carry, context_tokens=context_tokens)
    saving = inline - bounded
    msg = f'[read guard] Subagent returns here have run {est.describe()}, admitted to a context re-read ~{remaining_turns:,} more times: ~${inline:,.2f} to carry. Ask it for under {BRIEF_TARGET_TOKENS:,} tokens — the findings, not the transcript — and that becomes ~${bounded:,.2f}.'
    over = _advice_cost(msg, model, remaining_turns, carry, context_tokens)
    return _ledger_gate(Verdict(True, 'a subagent return larger than a brief', kind='brief', message=msg, tokens=tokens, inline=inline, delegated=bounded, saving=saving, overhead=over, advice_taken=cfg.advice_taken, estimate=est), state, cfg)

def _aggregate_gate(sh: str, running: int, calls: int, model: str, remaining_turns: int, cfg: Settings, state: GuardState, carry, context_tokens: int) -> Verdict:
    """Price what one command shape has admitted across the whole session.

    Deliberately not a per-call judgement, because every one of these calls was
    correctly judged small. `sed -n '1,200p'` is a bounded read and the guard is
    right to say nothing about any single one of them; it is 246 of them, at a
    513-token average, that put 126K tokens into a context.

    The advice is about the *habit* rather than about this call, so it fires
    once per shape per session and never again.
    """
    inline = admitted_token_cost(running, model, remaining_turns, carry=carry, context_tokens=context_tokens)
    if inline < cfg.min_cost:
        return Verdict(False, f'`{sh}` has admitted {running:,} tok, worth ${inline:,.2f}', kind='aggregate', tokens=running, inline=inline)
    per_call = running // max(1, calls)
    msg = f'[read guard] `{sh}` has run {calls:,} times this session and admitted ~{running:,} tokens ({per_call:,} a call) into a context re-read ~{remaining_turns:,} more times — ~${inline:,.2f} of carry. Each call was small; the total is not. Read it once, delegate it, or keep the results out of the main context.'
    over = _advice_cost(msg, model, remaining_turns, carry, context_tokens)
    saving = inline * 0.5
    return _ledger_gate(Verdict(True, 'one command shape has admitted more than the floor', kind='aggregate', message=msg, tokens=running, inline=inline, saving=saving, overhead=over, advice_taken=cfg.advice_taken), state, cfg)

def _advice_cost(message: str, model: str, remaining_turns: int, carry, context_tokens: int) -> float:
    """What injecting this sentence costs over the rest of the session.

    The message is admitted to the context exactly like a tool result, so it is
    priced exactly like one. Roughly 55 tokens carried across 300 remaining
    turns is real money at Opus rates, and it is money the guard spends whether
    or not its advice is taken.
    """
    return admitted_token_cost(est_tokens(message), model, remaining_turns, carry=carry, context_tokens=context_tokens)

def _ledger_gate(v: Verdict, state: GuardState, cfg: Settings) -> Verdict:
    """The guard's own solvency test, applied before it is allowed to speak."""
    if state.fires >= cfg.max_fires:
        return Verdict(False, f'already advised {state.fires} times this session', kind=v.kind, tokens=v.tokens, inline=v.inline, delegated=v.delegated, saving=v.saving, overhead=v.overhead, advice_taken=cfg.advice_taken, estimate=v.estimate)
    if v.net <= 0:
        return Verdict(False, f'saying so costs ${v.overhead:,.4f} to carry and is worth ${v.saving * cfg.advice_taken:,.4f}', kind=v.kind, tokens=v.tokens, inline=v.inline, delegated=v.delegated, saving=v.saving, overhead=v.overhead, advice_taken=cfg.advice_taken, estimate=v.estimate)
    return v

def observe(tool: str, tool_input: dict, state: GuardState, verdict: Verdict, *, now: float | None=None, sizes: SizeModel | None=None) -> GuardState:
    """Fold this call into the session's memory. Returns the same state object.

    Bash is accumulated whether or not the guard had anything to say about it,
    because the aggregate rule only works if the small bounded calls are
    counted -- they are the ones that add up to most of it.
    """
    tool_input = tool_input or {}
    if tool == 'Bash' and sizes is not None:
        cmd = tool_input.get('command') or ''
        if cmd:
            sh = shape(cmd)
            state.admitted[sh] = state.admitted.get(sh, 0) + sizes.predict_tool('Bash', tool_input).p50
            state.shape_calls[sh] = state.shape_calls.get(sh, 0) + 1
            if len(state.admitted) > MAX_REMEMBERED_READS:
                keep = MAX_REMEMBERED_READS // 2
                hot = sorted(state.admitted.items(), key=lambda kv: -kv[1])[:keep]
                state.admitted = dict(hot)
                state.shape_calls = {k: state.shape_calls.get(k, 0) for k, _ in hot}
    if tool == 'Write':
        fp = tool_input.get('file_path')
        if fp:
            # Recorded at issue time, because a PreToolUse hook runs before the
            # write lands and cannot read the resulting mtime.
            state.wrote[str(fp)] = time.time() if now is None else now
            state.reads.pop(str(fp), None)
            if len(state.wrote) > MAX_REMEMBERED_READS:
                keep = MAX_REMEMBERED_READS // 2
                state.wrote = dict(list(state.wrote.items())[-keep:])
    if tool == 'Read':
        fp = tool_input.get('file_path')
        # Only a *whole* read puts the whole file in the context. A read with
        # `limit` or `offset` admitted a slice, and remembering it as a full
        # read made the guard tell the model that a later complete read was
        # `already in this context and has not changed on disk` -- advice
        # against the one move that would actually have got the rest of the
        # file. The guard is the only component here that changes behaviour,
        # so a confidently wrong sentence from it is the expensive kind.
        if fp and not tool_input.get('limit') and not tool_input.get('offset'):
            state.reads[str(fp)] = _mtime(str(fp))
            if len(state.reads) > MAX_REMEMBERED_READS:
                keep = MAX_REMEMBERED_READS // 2
                state.reads = dict(list(state.reads.items())[-keep:])
    if verdict.fire:
        state.fires += 1
        state.saving += verdict.saving
        state.overhead += verdict.overhead
        if tool == 'Bash':
            sh = shape(tool_input.get('command') or '')
            if sh not in state.advised:
                state.advised.append(sh)
    return state

def _leading(tool_input: dict | None) -> str:
    """The first program in a command, which survives being bounded."""
    from adder.core.shapes import segments
    segs = segments((tool_input or {}).get('command') or '')
    return segs[0][0] if segs else ''

def fires_log() -> Path:
    try:
        from adder.core.settings import get as _setting
        return Path(str(_setting('home'))) / 'adder-guard-fires.jsonl'
    except Exception:
        return Path.home() / '.claude' / 'adder-guard-fires.jsonl'
MAX_FIRE_RECORDS = 5000

def record_fire(session: str, tool: str, tool_input: dict, v: Verdict, *, path: Path | None=None, now: float | None=None) -> None:
    """Append one fire, so `uptake` can ask later whether it changed anything.

    Identities only -- a shape, never a command; a basename, never a path --
    for the same reason `GuardState` holds no contents. Never raises: a guard
    that cannot write its log is still a guard.
    """
    # Everything, including building the row, is inside the try. The promise
    # above is unconditional and the row is built from a tool input, which is
    # whatever the model emitted: `Path(str(...))` raises ValueError on an
    # embedded null and `json.dumps` raises TypeError on anything it cannot
    # serialise. Neither is an OSError, and this runs in a hook whose only
    # other handler is a blanket except -- so the symptom would be the guard
    # firing and no longer recording that it did, which is how the uptake
    # measurement quietly becomes an assumption again.
    try:
        p = Path(path) if path is not None else fires_log()
        row = {'ts': time.time() if now is None else now, 'session': session, 'tool': tool, 'kind': v.kind, 'shape': shape((tool_input or {}).get('command') or '') if tool == 'Bash' else '', 'prog': _leading(tool_input) if tool == 'Bash' else '', 'name': Path(str((tool_input or {}).get('file_path') or '')).name, 'tokens': v.tokens}
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + '\n')
    except (OSError, TypeError, ValueError):
        pass

def load_fires(path: Path | None=None) -> list[dict]:
    """Every recorded fire, newest last. A corrupt line is skipped, not fatal."""
    p = Path(path) if path is not None else fires_log()
    out: list[dict] = []
    try:
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(row, dict) and row.get('session'):
                    out.append(row)
    except OSError:
        return []
    return out[-MAX_FIRE_RECORDS:]

@dataclass(frozen=True)
class Uptake:
    """How often the guard's advice was followed by the behaviour it asked for.

    Not a mind-reading exercise and not presented as one. For a `Bash` finding
    the question is whether later calls of that same shape, in that same
    session, were bounded more often than earlier ones were. For a duplicate
    read it is whether the file was read again. Both are observable in the
    transcript, and neither proves causation -- the model may have bounded the
    next call for its own reasons. It is an estimate of the term
    `guard_advice_taken` assumes, measured on the machine that assumes it.
    """
    fires: int = 0
    changed: int = 0
    before: float = 0.0
    after: float = 0.0

    @property
    def rate(self) -> float:
        return self.changed / self.fires if self.fires else 0.0

    @property
    def measured(self) -> bool:
        """Enough fires to be worth preferring to the assumption."""
        return self.fires >= 10

    def describe(self) -> str:
        if not self.fires:
            return 'no fires recorded yet — the assumption stands'
        return f'{self.changed:,} of {self.fires:,} findings were followed by the behaviour asked for ({self.rate:.0%}); bounded share of those shapes {self.before:.0%} → {self.after:.0%}'

def uptake(root=None, *, log: Path | None=None) -> Uptake:
    """Measure how often a fire was followed by the change it asked for.

    Reads the fires this guard recorded and the transcripts around them. Both
    halves are this project's own formats -- nothing here parses a shape of
    record it did not write -- which is why this is a measurement and the
    0.5 default is still labelled an assumption until it has data.
    """
    from adder.core.shapes import DEFAULT_ROOT, is_bounded, iter_calls
    fires = load_fires(log)
    if not fires:
        return Uptake()
    calls: dict[str, list[tuple[str, str, str, bool, str]]] = {}
    for session, _model, tool, inp, ts in iter_calls(root or DEFAULT_ROOT):
        cmd = (inp or {}).get('command') or ''
        calls.setdefault(session, []).append((ts, tool, _leading(inp) if tool == 'Bash' else '', is_bounded(cmd) if tool == 'Bash' else bool((inp or {}).get('limit')), Path(str((inp or {}).get('file_path') or '')).name))
    judged = changed = 0
    before_hits = before_n = after_hits = after_n = 0
    for f in fires:
        rows = calls.get(str(f.get('session') or ''))
        if not rows:
            continue
        stamp = _iso(f.get('ts'))
        if f.get('kind') == 'duplicate':
            name = str(f.get('name') or '')
            if not name:
                continue
            later = [r for r in rows if r[0] >= stamp and r[4] == name and (r[1] == 'Read')]
            judged += 1
            if not later:
                changed += 1
            continue
        sh = str(f.get('prog') or '')
        if not sh:
            continue
        earlier = [r for r in rows if r[0] <= stamp and r[2] == sh]
        later = [r for r in rows if r[0] > stamp and r[2] == sh]
        if not later:
            continue
        judged += 1
        before_n += len(earlier)
        before_hits += sum(1 for r in earlier if r[3])
        after_n += len(later)
        after_hits += sum(1 for r in later if r[3])
        if sum(1 for r in later if r[3]) / len(later) > (sum(1 for r in earlier if r[3]) / len(earlier) if earlier else 0.0):
            changed += 1
    return Uptake(judged, changed, before_hits / before_n if before_n else 0.0, after_hits / after_n if after_n else 0.0)

def _iso(ts) -> str:
    """A float epoch as the ISO string transcripts are stamped with."""
    from datetime import datetime, timezone
    try:
        return datetime.fromtimestamp(float(ts), timezone.utc).isoformat().replace('+00:00', 'Z')
    except (TypeError, ValueError, OSError, OverflowError):
        # OverflowError is `datetime.fromtimestamp(1e30)`. It is not a
        # ValueError, and a fires log is a file this tool appends to from a
        # hook -- a torn write puts anything at all in that field.
        return ''

@dataclass
class Replay:
    """What a replay of the guard over recorded sessions found."""
    calls: int = 0
    priced: int = 0
    fires: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    saving: float = 0.0
    overhead: float = 0.0
    sessions: int = 0
    biggest: list[tuple[float, str]] = field(default_factory=list)

    @property
    def net(self) -> float:
        return self.saving - self.overhead

    @property
    def fire_rate(self) -> float:
        return self.fires / self.calls if self.calls else 0.0

    @property
    def lookup_rate(self) -> float:
        """Share of calls that cost a transcript parse. The latency budget."""
        return self.priced / self.calls if self.calls else 0.0

def replay(root=None, *, cfg: Settings | None=None, sizes: SizeModel | None=None, advice_taken: float | None=None, top: int=8) -> Replay:
    """Run the guard over recorded transcripts and report what it would have done.

    Deliberately not a saving anyone should bank. Three things make it an upper
    bound: the horizon is the one the guard would have projected rather than
    the turns that really remained, the saving assumes the advice is acted on,
    and a call the guard talked someone out of would have changed everything
    after it. What it *is* good for is the shape of the thing -- how often it
    would speak, on what, and whether the advice would have cost more than it
    was worth on this workload rather than on the author's.

    Read-only: it never writes state, so replaying does not disturb a live
    session's memory.
    """
    from adder.core.shapes import DEFAULT_ROOT, iter_calls, load_model
    from adder.measure.session.horizon import load as load_horizon
    cfg = cfg or Settings.resolve()
    sizes = sizes or load_model()
    taken = cfg.advice_taken if advice_taken is None else advice_taken
    # Fitted to the corpus being replayed, not to the default one. The replay
    # walks `iter_calls(root)` and priced every one of those calls with a
    # horizon built from `~/.claude/projects` -- two different workloads in one
    # number, and now that the horizon cache is keyed by root the mismatch is
    # exact rather than merely likely.
    horizon = load_horizon(root or DEFAULT_ROOT)
    rep = Replay()
    states: dict[str, GuardState] = {}
    seen_turns: dict[str, int] = {}
    for session, model, tool, inp, _ts in iter_calls(root or DEFAULT_ROOT):
        rep.calls += 1
        state = states.get(session)
        if state is None:
            state = states[session] = GuardState()
            rep.sessions += 1
        i = seen_turns[session] = seen_turns.get(session, 0) + 1
        remaining = horizon.mean_remaining(i)
        if not needs_pricing(tool, inp, sizes=sizes, state=state, min_tokens=cfg.min_tokens):
            observe(tool, inp, state, Verdict(False, 'below floor'), sizes=sizes)
            continue
        rep.priced += 1
        v = decide(tool, inp, model=model, remaining_turns=int(remaining), cfg=cfg, sizes=sizes, state=state)
        observe(tool, inp, state, v, sizes=sizes)
        if not v.fire:
            continue
        rep.fires += 1
        rep.by_kind[v.kind] = rep.by_kind.get(v.kind, 0) + 1
        rep.saving += v.saving * taken
        rep.overhead += v.overhead
        rep.biggest.append((v.saving * taken, f'{v.kind}: {v.message[13:110]}'))
    rep.biggest.sort(reverse=True)
    rep.biggest = rep.biggest[:top]
    return rep
HOOK_RELPATH = '.claude/hooks/pretooluse_read_guard.py'

def hook_path(repo: Path | None=None) -> Path:
    """Absolute path to the shipped hook, for the install snippet."""
    here = Path(__file__).resolve()
    root = repo or here.parents[2]
    return root / HOOK_RELPATH

def settings_files(cwd=None) -> list[Path]:
    """Every settings file a hook could be declared in, nearest last."""
    home = Path.home() / '.claude' / 'settings.json'
    proj = Path(cwd or os.getcwd()).resolve() / '.claude'
    return [home, proj / 'settings.json', proj / 'settings.local.json']

def installed_in(cwd=None) -> list[Path]:
    """Which settings files declare this hook. Read-only, and never raises.

    The reason this exists at all: an uninstalled guard, a broken guard and a
    correctly quiet guard produce exactly the same experience. Everything else
    in `adder guard` reports what the guard decided; this reports whether it is
    in a position to decide anything.
    """
    found = []
    try:
        paths = settings_files(cwd)
    except OSError:
        # `os.getcwd()` raises when the working directory has been deleted out
        # from under the process, which happens to anyone running this from a
        # branch they just switched away from. The promise above is
        # unconditional.
        return found
    for path in paths:
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        try:
            text = json.dumps(blob)
        except (TypeError, ValueError):
            continue
        if 'pretooluse_read_guard' in text:
            found.append(path)
    return found

def install_snippet(cwd=None) -> str:
    """The settings.json block that installs the hooks, ready to merge.

    Two entries. The guard is the one that decides; the PreCompact learner just
    keeps the size model current at a moment the session is already paused, so
    that the guard is predicting from this machine rather than from the shipped
    prior. Installing the first without the second still works -- it is simply
    less accurate until somebody runs `adder guard --learn`.
    """
    learner = hook_path().parent / 'precompact_learn.py'
    return json.dumps({'hooks': {'PreToolUse': [{'matcher': '|'.join(OBSERVED), 'hooks': [{'type': 'command', 'command': f'python3 {hook_path()}'}]}], 'PreCompact': [{'hooks': [{'type': 'command', 'command': f'python3 {learner}'}]}]}}, indent=2)

def ledger(path: Path | None=None) -> dict:
    """What the guard has promised and what saying it has cost, across sessions.

    The guard is the one mechanism here that runs without being asked, so it is
    the one whose own solvency nobody would otherwise check. This is the same
    arithmetic `adder ledger` applies to routing recommendations, over the
    guard's own state file.
    """
    p = Path(path) if path is not None else Settings.resolve().state_path
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(blob, dict):
            blob = {}
    except (OSError, ValueError):
        blob = {}
    fires = saving = overhead = 0.0
    sessions = 0
    for row in blob.values():
        if not isinstance(row, dict):
            continue
        sessions += 1
        # `_num`, not bare `float()`. This file is written by a hook that can be
        # killed mid-write and is small enough that people edit it; a
        # non-numeric `fires` raised `ValueError` straight out of `adder guard`,
        # from the section whose whole purpose is to say whether the guard is
        # solvent.
        fires += _num(row.get('fires'))
        saving += _num(row.get('saving'))
        overhead += _num(row.get('overhead'))
    return {'sessions': sessions, 'fires': int(fires), 'saving': saving, 'overhead': overhead, 'path': str(p)}

def _as_call(target: str) -> tuple[str, dict]:
    """Read `--explain` input as a tool call.

    A path is a `Read`, anything else is a shell command. Guessing this rather
    than adding a `--tool` flag because the two are unambiguous in practice and
    the question people bring to `--explain` is "why did it say nothing about
    *this*", where *this* is whatever they typed or read.
    """
    target = (target or '').strip()
    if target in ('Task', 'Agent'):
        return (target, {'description': 'a delegated step'})
    if target.startswith(('/', './', '~/')) and ' ' not in target:
        return ('Read', {'file_path': str(Path(target).expanduser())})
    return ('Bash', {'command': target})

def _explain(target: str, sizes: SizeModel, cfg: Settings, *, model: str, remaining: int) -> list[str]:
    """Why the guard would or would not speak about one specific call."""
    from adder.util.render import bullet, kv, money
    tool, inp = _as_call(target)
    est = sizes.predict_tool(tool, inp)
    v = decide(tool, inp, model=model, remaining_turns=remaining, sizes=sizes, cfg=cfg)
    command = inp.get('command') or inp.get('file_path') or target
    out = ['', kv('call', f'{tool} {command}'), kv('shape', shape(command) if tool == 'Bash' else 'n/a — sized from disk'), kv('bounded', 'yes — output is capped by construction' if tool == 'Bash' and is_bounded(command) else 'no'), kv('predicted size', f'{est.p50:,} tok median, {est.p90:,} tok p90 ({est.source}, n={est.n:,})'), kv('verdict', 'SPEAK' if v.fire else 'quiet'), kv('reason', v.reason)]
    if v.inline:
        out.append(kv('carry cost', money(v.inline)))
    if v.delegated:
        out.append(kv('delegated', money(v.delegated)))
    if v.fire:
        out.append(kv('advice costs', money(v.overhead)))
        out.append(kv('expected net', money(v.net)))
        out += ['', *(bullet(line) for line in [v.message])]
    return out

def _clip(text: str, width: int=74) -> str:
    text = ' '.join(str(text).split())
    return text if len(text) <= width else text[:width - 1] + '…'

def report(root=None, *, learn: bool=False, explain: str | None=None, replay_it: bool=False, cwd=None) -> str:
    """Everything about the guard that is otherwise invisible."""
    from adder.core.shapes import DEFAULT_ROOT, PRIOR, SizeModel, model_path, refresh
    from adder.util.render import heading, kv, money, table, tokens, wrap
    cfg = Settings.resolve(cwd=cwd)
    root = root or DEFAULT_ROOT
    if learn:
        sizes = SizeModel.learn(root)
        with contextlib.suppress(OSError):
            sizes.save(model_path())
    else:
        sizes = refresh(root)
    out: list[str] = []
    out += heading('Installed?', rule='=')
    where = installed_in(cwd)
    if where:
        out += [kv('status', 'yes'), *[kv('declared in', str(w)) for w in where]]
    else:
        out += [kv('status', 'NO — nothing is preventing spend'), '', '  Every other report here measures money already gone. This is the', '  only component that runs while the decision is still reversible,', '  and it is not in any settings.json this command can see.', '', '  Run `adder guard --install` for the block to merge.']
    out += ['', *heading('Size model — what the guard predicts from', rule='=')]
    if not sizes.calls:
        out += [kv('state', 'not learned yet'), kv('effect', 'every prediction falls back to the shipped prior'), '', '  Run `adder guard --learn` to derive it from your own transcripts.']
    else:
        age = 'just now' if learn else f'{sizes.age_s / 3600:.1f}h ago'
        out += [kv('built', age), kv('observations', f'{sizes.calls:,} answered tool calls'), kv('shapes', f'{len(sizes.shapes):,} command shapes, {len(sizes.heads):,} programs')]
        ranked = sorted(sizes.shapes.items(), key=lambda kv_: -kv_[1][1])[:10]
        if ranked:
            out += ['', *heading('Largest shapes, by p90 result size')]
            out += table([(k, tokens(v[0]), tokens(v[1]), f'{v[2]:,}') for k, v in ranked], headers=('shape', 'median', 'p90', 'calls'))
        rows = []
        for tool in sorted(sizes.tools):
            local = sizes.tools[tool]
            if local[2] < 3 or tool not in PRIOR:
                continue
            shipped = PRIOR[tool][1]
            ratio = shipped / local[1] if local[1] else float('inf')
            rows.append((tool, tokens(shipped), tokens(local[1]), f'{local[2]:,}', '—' if 0.5 <= ratio <= 2 else f'{ratio:,.1f}x out'))
        if rows:
            out += ['', *heading('Shipped prior vs this machine (p90)')]
            out += table(rows, headers=('tool', 'prior', 'yours', 'calls', 'verdict'))
    out += ['', *heading('Guard ledger — has advising been worth the advice?', rule='=')]
    led = ledger(cfg.state_path)
    if not led['fires']:
        out += [kv('fires', 'none recorded yet')]
    else:
        net = led['saving'] * cfg.advice_taken - led['overhead']
        out += [kv('sessions seen', f"{led['sessions']:,}"), kv('times it spoke', f"{led['fires']:,}"), kv('saving promised', money(led['saving'])), kv('cost of saying it', money(led['overhead'])), kv(f'net at {cfg.advice_taken:.0%} uptake', money(net)), kv('solvent', 'yes' if net >= 0 else 'NO — the guard is costing money')]
    if replay_it:
        out += ['', *heading('Replay — what it would have done here', rule='=')]
        r = replay(root, cfg=cfg, sizes=sizes)
        if not r.calls:
            out += [kv('calls', 'none found')]
        else:
            out += [kv('tool calls replayed', f'{r.calls:,} across {r.sessions:,} sessions'), kv('cost a transcript parse', f'{r.lookup_rate:.2%} of calls'), kv('times it would speak', f'{r.fires:,} ({r.fire_rate:.2%} of calls)'), kv('by kind', ', '.join((f'{k} {n:,}' for k, n in sorted(r.by_kind.items()))) or '—'), kv(f'saving at {cfg.advice_taken:.0%} uptake', money(r.saving)), kv('cost of saying it', money(r.overhead)), kv('net', money(r.net))]
            if r.biggest:
                out += ['', *heading('Largest findings')]
                out += table([(money(v), _clip(t)) for v, t in r.biggest], headers=('worth', 'finding'), align='><')
            out += ['']
            out += wrap('An upper bound, not a saving to bank: the horizon is the one the guard would have projected rather than the turns that really remained, the saving assumes the advice is acted on, and a call it talked someone out of would have changed everything after it.')
    out += ['', *heading('Uptake — is the assumption holding?', rule='=')]
    u = uptake(root)
    out += [kv('measured', u.describe())]
    if u.measured:
        out += [kv('assumed', f'{cfg.advice_taken:.0%}'), kv('verdict', 'the assumption is conservative' if u.rate >= cfg.advice_taken else 'the assumption is optimistic — lower guard_advice_taken')]
    else:
        out += [kv('assumed', f'{cfg.advice_taken:.0%} (guard_advice_taken)'), kv('note', 'an assumption until there are 10 findings to judge')]
    out += ['', *heading('Settings in effect', rule='=')]
    out += [kv('floor', f'{cfg.min_tokens:,} tok predicted'), kv('worth interrupting', money(cfg.min_cost) + ' of carry'), kv('mode', 'ask for confirmation' if cfg.block else 'advise only'), kv('assumed uptake', f'{cfg.advice_taken:.0%} of advice acted on'), kv('ceiling', f'{cfg.max_fires} fires per session'), kv('state', str(cfg.state_path))]
    if explain:
        out += ['', *heading('Explain', rule='=')]
        from adder.core import settings as _settings
        out += _explain(explain, sizes, cfg, model=str(_settings.get('model')), remaining=300)
    return '\n'.join(out)

def main(argv: list[str] | None=None) -> int:
    import argparse

    from adder.core.shapes import DEFAULT_ROOT
    ap = argparse.ArgumentParser(prog='adder guard', description='What the PreToolUse guard predicts, decides, and has cost.')
    ap.add_argument('root', nargs='?', default=DEFAULT_ROOT, help='transcript directory (default: %(default)s)')
    ap.add_argument('--learn', action='store_true', help='re-derive the size model from local transcripts')
    ap.add_argument('--explain', metavar='CMD', help='show what the guard would do with one shell command')
    ap.add_argument('--install', action='store_true', help='print the settings.json block that installs the hook')
    ap.add_argument('--replay', action='store_true', help='replay the guard over your transcripts and price what it would have said')
    ap.add_argument('--json', action='store_true', help='machine-readable')
    a = ap.parse_args(argv)
    # `root_of`: the argument if one was given, else the `root`
    # setting. Resolved here so two commands cannot disagree
    # about which transcript directory `adder config` names.
    a.root = str(_root_of(a))
    # `root_of`: the argument if one was given, else the `root`
    # setting. Resolved here so two commands cannot disagree.
    a.root = str(_root_of(a))
    if a.install:
        print('Merge this into ~/.claude/settings.json (user-wide) or .claude/settings.json (this project):')
        print()
        print(install_snippet())
        print()
        print('Then run `adder guard --learn` once, so it predicts result sizes from your\ntranscripts rather than from the shipped prior.')
        return 0
    if a.json:
        from adder.core.shapes import SizeModel, load_model
        sizes = SizeModel.learn(a.root) if a.learn else load_model()
        cfg = Settings.resolve()
        out = {'installed': [str(w) for w in installed_in()], 'model': {'calls': sizes.calls, 'shapes': len(sizes.shapes), 'age_s': None if sizes.age_s == float('inf') else sizes.age_s}, 'ledger': ledger(cfg.state_path), 'uptake': {'fires': uptake(a.root).fires, 'rate': round(uptake(a.root).rate, 4), 'measured': uptake(a.root).measured}, 'settings': {'min_tokens': cfg.min_tokens, 'min_cost': cfg.min_cost, 'block': cfg.block, 'advice_taken': cfg.advice_taken, 'max_fires': cfg.max_fires}}
        if a.replay:
            r = replay(a.root, cfg=cfg, sizes=sizes)
            out['replay'] = {'calls': r.calls, 'sessions': r.sessions, 'priced': r.priced, 'fires': r.fires, 'by_kind': r.by_kind, 'fire_rate': round(r.fire_rate, 5), 'lookup_rate': round(r.lookup_rate, 5), 'saving': round(r.saving, 4), 'overhead': round(r.overhead, 4), 'net': round(r.net, 4)}
        print(json.dumps(out, indent=2))
        return 0
    print(report(a.root, learn=a.learn, explain=a.explain, replay_it=a.replay))
    return 0
