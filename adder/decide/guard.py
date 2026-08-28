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

That memory was then keyed on `Read`'s `file_path`, which is a second way to
see nothing. Under `bypassPermissions` -- how harnesses run unattended -- the
guidance routes file access to the shell, so `cat` and `sed -n` do the reading,
nothing populates the key, and the rule reports zero. Zero and "this
instrumentation cannot observe the reads on this machine" printed identically,
and on one 8-session corpus what printed as $0.00 was 25.8% of every Bash
result token -- content the session had already read. `core.reads` names what a
call read for both tools, so which one the harness happened to pick stops
deciding whether the saving exists.

Everything here fails open. A guard that raises is a guard that has stopped
guarding, and the failure is invisible because the tool call still succeeds.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shlex
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from adder.core.filters import root_of as _root_of
from adder.core.reads import ReadTarget, tool_targets, whole_reads
from adder.core.reads import resolve as _resolve_path
from adder.core.shapes import (
    Estimate,
    SizeModel,
    bound_lines,
    empty_model,
    is_bounded,
    read_estimate,
    shape,
)
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
# How far the guard may go, in order. `off` is the historical behaviour and is
# still the default for anyone who has not activated anything.
#
# The split is not a preference dial, it is the line between two different
# claims. `certain` covers the calls where the saving needs no model and no
# forecast: the content is already in this context, so the call buys nothing at
# all and refusing it cannot cost information. `full` also refuses a large read
# that has a strictly cheaper equal -- true on the measurement, but it rests on
# a horizon estimate and on a subagent returning a brief, so it is a separate
# opt-in.
# `shadow` sits between the two claims rather than beside them, and it is the
# level anybody sceptical should start on. Everything downstream of a refusal
# rests on `guard_advice_taken`, which this project's own docs call its weakest
# number: 0.5, assumed, on every machine that has not measured it. Enforcement
# asks a user to hand refusal authority to a component on the strength of that
# assumption. `shadow` runs the whole decision, records what it *would* have
# refused, and refuses nothing -- so the assumption becomes a measurement, on
# this machine, before anything is denied. It models `certain`, because
# `certain` is what `adder auto on` installs.
ENFORCE_LEVELS: tuple[str, ...] = ('off', 'shadow', 'certain', 'full')
# A refusal is offered once per target and never twice. If the model asks for
# the same thing again it has a reason the guard cannot see -- it may have been
# compacted since, or the first refusal may simply have been wrong -- and a
# guard that refuses in a loop is a guard that has broken the session it was
# supposed to make cheaper. Second ask always wins.
MAX_REMEMBERED_DENIALS = 200

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
    # Per-tool overrides for the two numbers above, empty by default. One floor
    # and one ceiling were serving every tool, and the tools are not alike:
    # measured here, `Bash` returns a p90 of 1,200 tokens over 2,490 calls in a
    # session and `Read` 5,900 over 58. Against a single 2,000-token floor the
    # first is almost never priced and the second routinely is, and against a
    # single 15-fire ceiling the first can spend the whole budget before the
    # second has said anything. Different distributions doing different jobs.
    #
    # Shipped empty rather than with a table, because a table would be one
    # machine's workload asserted as everyone's -- the mistake this module
    # already made once with the size prior. `adder guard --floors` derives the
    # numbers from the reader's own transcripts instead.
    min_tokens_by_tool: dict = field(default_factory=dict)
    max_fires_by_tool: dict = field(default_factory=dict)
    state_path: Path = Path.home() / '.claude' / '.adder-guard.json'
    enforce: str = 'off'
    # Whether a `Task` also gets told which tier to run on. Defaults on because
    # it is advice rather than a refusal and it is gated by the same solvency
    # test as everything else here; `guard_route=false` turns it off for anyone
    # who would rather the guard only talked about size.
    route: bool = True
    # Whether a refusal may become a substitution: the same bounded call the
    # message was already asking for, made for the model instead of demanded of
    # it. Off by default because a rewrite travels with an approval and can
    # therefore suppress a permission prompt -- see `decide/narrow.py`. It only
    # ever applies where the guard was going to refuse anyway, so enabling it
    # relaxes a denial rather than permitting anything new.
    narrow: bool = False

    def min_tokens_for(self, tool: str) -> int:
        """The size floor for one tool. The global one unless it was overridden."""
        return int(self.min_tokens_by_tool.get(tool, self.min_tokens))

    def max_fires_for(self, tool: str) -> int:
        """How often the guard may speak about one tool in one session.

        Both this and `max_fires` have to pass. A per-tool ceiling exists to
        stop one loud tool starving the others, not to raise the total, and a
        per-tool number that could exceed the global one would do the second
        while claiming to do the first.
        """
        return min(self.max_fires, int(self.max_fires_by_tool.get(tool, self.max_fires)))

    @property
    def enforcing(self) -> bool:
        """Is the guard allowed to refuse anything at all?

        `shadow` is not enforcing, and the distinction is the whole point of
        it: it computes the same refusal and then does not make it. Written as
        an allowlist rather than as `!= 'off'` so that a level added later has
        to say which side of the line it is on.
        """
        return self.enforce in ('certain', 'full')

    @property
    def shadowing(self) -> bool:
        """Is it recording the refusals it is not making?"""
        return self.enforce == 'shadow'

    @property
    def would_refuse(self) -> bool:
        """Is a refusal being computed, whether or not it is carried out?"""
        return self.enforcing or self.shadowing

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

        def taken() -> float:
            """The uptake discount: measured if it has been, assumed if not.

            This is the number every advisory saving in the tool is multiplied
            by, and until now it was 0.5 on every machine -- including the ones
            that had measured their own. `uptake()` has existed and been
            reported by `adder auto status` the whole time; nothing consumed it.
            A measurement nobody acts on is the same failure as a router nobody
            invokes.

            An explicitly configured `guard_advice_taken` still wins. Somebody
            who wrote a number into `.adder.json` has said something the
            estimator does not know, and silently overriding it would make the
            setting decorative -- which is the exact bug `Settings` was written
            to avoid.
            """
            picked = pick('guard_advice_taken', 0.5, float)
            if sources.get('guard_advice_taken', 'default') != 'default':
                return picked
            rate, measured, _age = load_uptake()
            return max(UPTAKE_FLOOR, min(1.0, rate)) if measured else picked

        def by_tool(name: str, cast) -> dict:
            """`Bash=800,Read=6000` as a dict. A typo is ignored, not fatal.

            Same shape and same failure rule as `classify.ladder`: an
            unparseable entry is dropped rather than raising, because a broken
            config file must not be what takes the guard down -- and a guard
            that raises is a guard that has stopped guarding, invisibly.
            """
            out: dict[str, int] = {}
            for part in str(pick(name, '', str) or '').split(','):
                tool, _, value = part.partition('=')
                tool = tool.strip()
                if not tool or tool not in OBSERVED:
                    continue
                try:
                    out[tool] = cast(value.strip())
                except (TypeError, ValueError):
                    continue
            return out

        def as_bool(v) -> bool:
            return v if isinstance(v, bool) else str(v).strip().lower() in ('1', 'true', 'yes', 'on')
        def as_level(v) -> str:
            # An unrecognised level reads as `off`. The failure direction has to
            # be "advise" rather than "refuse": a typo in a config file must not
            # be what starts denying tool calls.
            got = str(v).strip().lower()
            return got if got in ENFORCE_LEVELS else 'off'
        return cls(min_tokens_by_tool=by_tool('guard_min_tokens_by_tool', int), max_fires_by_tool=by_tool('guard_max_fires_by_tool', int), min_tokens=pick('guard_min_tokens', 2000, int), min_cost=pick('guard_min_cost', 0.25, float), hard_tokens=pick('guard_hard', 60000, int), block=pick('guard_block', False, as_bool), advice_taken=taken(), max_fires=pick('guard_max_fires', 15, int), state_path=Path(str(pick('guard_state', str(Path.home() / '.claude' / '.adder-guard.json'), str))), enforce=as_level(pick('guard_enforce', 'off', str)), route=pick('guard_route', True, as_bool), narrow=pick('guard_narrow', False, as_bool))

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
    deny: bool = False
    certain: bool = False
    target: str = ''
    # How many independent things this message asks for. Only the clipping
    # budget uses it: a two-claim message is not a rambling one, and each claim
    # was separately priced against the cost of carrying it.
    claims: int = 1
    # The tier clause's own key, kept apart from `target` because a `Task`
    # verdict can carry both: `target` is what a refusal would be recorded
    # against, and this is what stops the routing sentence being said twice.
    tier_target: str = ''
    # The bounded call to run in place of this one, when the guard was going to
    # refuse and `guard_narrow` is on. `None` means "refuse or advise as
    # before" -- the field is additive and its absence is the old behaviour.
    narrowed: dict | None = None
    # True when this is the refusal `shadow` computed and did not make. The
    # verdict still carries the whole counterfactual -- message, tokens,
    # saving, target -- because the message is what the reader has to be able
    # to inspect before they agree to let it run for real.
    shadow: bool = False

    @property
    def action(self) -> str:
        """What this verdict does, as one word.

        `narrow` sits where `deny` would: it is the same judgement about the
        same call, carried out by substitution instead of refusal. Kept as its
        own word rather than folded into `deny` because the ledger has to be
        able to tell them apart -- a refusal saves the whole read, a
        substitution saves the difference, and reporting one as the other would
        overstate the saving.
        """
        if self.shadow:
            return 'shadow'
        if self.narrowed is not None:
            return 'narrow'
        return 'deny' if self.deny else ('ask' if self.ask else 'advise')

    @property
    def uptake(self) -> float:
        """The share of this verdict's saving that is actually realised.

        The 0.5 default is an assumption about whether a sentence changes what
        a model does next, and it is the right assumption for a sentence. A
        refusal is not a sentence: the call does not happen, so the tokens are
        not admitted, and discounting that by an advice-uptake prior would be
        booking half a saving that is whole.
        """
        # A shadow verdict realises nothing at all: the call runs, the tokens
        # are admitted, and the only thing that happened is that a note was
        # made. Booking any part of its saving would put back exactly the
        # assumption shadow mode exists to replace.
        if self.shadow:
            return 0.0
        # A substitution is enforced too: the bounded call is what runs, so the
        # difference is realised whether or not anything was persuaded.
        return 1.0 if (self.deny or self.narrowed is not None) else self.advice_taken

    @property
    def net(self) -> float:
        """Expected value of firing: the saving, discounted, less the overhead."""
        return self.saving * self.uptake - self.overhead

    ESCAPE = 'Re-issue this exact call if you need'

    def clipped(self) -> Verdict:
        """This verdict with its message held to `MAX_MESSAGE_TOKENS`.

        A refusal gets a wider budget, and it is not generosity. The clause
        that tells the model how to get through is at the end of the sentence,
        and clipping it off would turn "refused once, ask again" into a wall
        the model has no stated way past -- the exact failure the refuse-once
        rule exists to prevent. It is still bounded; it is bounded higher.
        """
        # The same reasoning covers a message carrying two claims: the second
        # one is at the end, clipping it off leaves advice nobody can act on,
        # and it only got joined at all after paying for its own words.
        limit = MAX_MESSAGE_TOKENS * (6 if self.deny or self.claims > 1 else 4)
        if len(self.message) <= limit:
            return self
        cut = self.message[:limit - 1].rsplit(' ', 1)[0]
        out = replace(self, message=cut + '…')
        if self.deny and self.ESCAPE not in out.message:
            out = replace(out, message=out.message + f' {self.ESCAPE} it anyway.')
        return out

    def payload(self) -> dict:
        """The hook's stdout, or `{}` when there is nothing to say.

        A `deny` is the only output here that changes what happens rather than
        describing it, and it always carries its reason: the model is told what
        it already has and what to do instead, so the refusal is a redirection
        and not a wall. There is no path to a silent denial.
        """
        if not self.fire or self.shadow:
            # Shadow says nothing. Not a quieter sentence, not an
            # `additionalContext` note -- nothing, because a message is
            # admitted to the context and carried, and a mode whose purpose is
            # to cost nothing while it measures may not charge for itself.
            return {}
        out: dict = {'hookEventName': 'PreToolUse'}
        self = self.clipped()
        if self.narrowed is not None:
            # `allow` is required for the substitution to be taken: the shipped
            # client honours `updatedInput` on the approval path. The reason
            # travels with it so the model is told the call it made is not the
            # call that ran -- a silent truncation would have it treat a slice
            # as the whole file.
            out['permissionDecision'] = 'allow'
            out['permissionDecisionReason'] = self.message
            out['updatedInput'] = self.narrowed
        elif self.deny:
            out['permissionDecision'] = 'deny'
            out['permissionDecisionReason'] = self.message
        elif self.ask:
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
    denied: dict[str, float] = field(default_factory=dict)
    fires: int = 0
    saving: float = 0.0
    overhead: float = 0.0
    prevented: float = 0.0
    touched: float = 0.0
    # Shadow mode's whole record, kept apart from every field above so that a
    # machine running `shadow` reports a guard that has promised nothing, cost
    # nothing and refused nothing -- which is the truth about it.
    #
    # `contradicted` is the half that matters. A shadow refusal is a claim that
    # a call was worth nothing; the session asking for the same target again
    # afterwards is the closest thing to evidence that the claim was wrong,
    # and it is the number that decides whether enforcement is safe here. It is
    # counted rather than flagged, because a target asked for three more times
    # is a worse refusal than one asked for twice.
    shadowed: dict[str, float] = field(default_factory=dict)
    contradicted: dict[str, int] = field(default_factory=dict)
    shadow_fires: int = 0
    shadow_saving: float = 0.0
    # How often the guard has spoken about each tool. The global counter alone
    # let the tool called two thousand times a session spend the whole budget
    # before the tool called fifty times had said anything.
    fires_by_tool: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {'reads': self.reads, 'wrote': self.wrote, 'admitted': self.admitted, 'shape_calls': self.shape_calls, 'advised': self.advised[-64:], 'denied': self.denied, 'fires': self.fires, 'saving': round(self.saving, 6), 'overhead': round(self.overhead, 6), 'prevented': round(self.prevented, 6), 'touched': self.touched, 'shadowed': self.shadowed, 'contradicted': self.contradicted, 'shadow_fires': self.shadow_fires, 'shadow_saving': round(self.shadow_saving, 6), 'fires_by_tool': self.fires_by_tool}

    def forget_context(self) -> GuardState:
        """Drop every claim that something is already in the context.

        Called from the PreCompact hook. Compaction is precisely the event that
        makes `reads` and `wrote` false: the tokens they refer to are about to
        leave the window, so a refusal justified by "you already have this"
        would be refusing a read of something the model no longer has. The
        running totals survive, because they are a record of what was spent and
        compaction does not refund it.
        """
        self.reads = {}
        self.wrote = {}
        self.denied = {}
        # Shadow refusals go with them, and for the same reason: after
        # compaction the tokens they claimed were redundant are gone, so a
        # later read of the same target is the session recovering what it lost
        # rather than the guard having been wrong. Counting it as a
        # contradiction would libel the mode's own measurement.
        self.shadowed = {}
        return self

    @classmethod
    def from_json(cls, d: dict) -> GuardState:
        if not isinstance(d, dict):
            return cls()
        reads = d.get('reads')
        wrote = d.get('wrote')
        admitted = d.get('admitted')
        shape_calls = d.get('shape_calls')
        advised = d.get('advised')
        denied = d.get('denied')
        shadowed = d.get('shadowed')
        contradicted = d.get('contradicted')
        fires_by_tool = d.get('fires_by_tool')
        return cls(reads={str(k): float(v) for k, v in reads.items()} if isinstance(reads, dict) else {}, wrote={str(k): float(v) for k, v in wrote.items()} if isinstance(wrote, dict) else {}, admitted={str(k): int(v) for k, v in admitted.items()} if isinstance(admitted, dict) else {}, shape_calls={str(k): int(v) for k, v in shape_calls.items()} if isinstance(shape_calls, dict) else {}, advised=[str(x) for x in advised] if isinstance(advised, list) else [], denied={str(k): _num(v) for k, v in denied.items()} if isinstance(denied, dict) else {}, fires=int(d.get('fires') or 0), saving=float(d.get('saving') or 0.0), overhead=float(d.get('overhead') or 0.0), prevented=float(d.get('prevented') or 0.0), touched=float(d.get('touched') or 0.0), shadowed={str(k): _num(v) for k, v in shadowed.items()} if isinstance(shadowed, dict) else {}, contradicted={str(k): int(_num(v)) for k, v in contradicted.items()} if isinstance(contradicted, dict) else {}, shadow_fires=int(_num(d.get('shadow_fires'))), shadow_saving=_num(d.get('shadow_saving')), fires_by_tool={str(k): int(_num(v)) for k, v in fires_by_tool.items()} if isinstance(fires_by_tool, dict) else {})

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

# Tools whose call has a strictly-narrower valid form. Mirrors
# `narrow.BOUNDS`, imported lazily there so the guard's own import stays cheap.
NARROWABLE = ('Read', 'Grep')


def _line_tokens(lines: int) -> int:
    """Tokens a bounded result of `lines` lines admits, on the read model."""
    from adder.core.shapes import READ_TOKENS_PER_LINE
    return max(1, int(lines * READ_TOKENS_PER_LINE[1]))


def _narrowed(tool: str, tool_input: dict, lines: int) -> dict | None:
    """The bounded call to substitute, or None. Never raises.

    Wrapped because the guard must degrade to its previous behaviour rather
    than fail: a hook that throws on a tool input shape nobody anticipated
    would take the whole tool call down, and the whole point of this path is
    that it is optional.
    """
    try:
        from adder.decide.narrow import narrow
        return narrow(tool, tool_input, lines=lines)
    except Exception:
        return None


def _bounded_hint(tool: str) -> str:
    """The cheaper shape of this exact call, when there is an obvious one."""
    return {'Read': 'read it with offset/limit, or delegate the read', 'Bash': 'pipe it through `head -50`, `wc -l`, or redirect to a file', 'Grep': 'use `-l` or `-c` first, then read only the hits that matter', 'Glob': 'narrow the pattern', 'WebFetch': 'ask for the section you need, not the page', 'WebSearch': 'narrow the query, or fetch the one result you need', 'Task': 'tell the subagent what to return, and how briefly', 'Agent': 'tell the subagent what to return, and how briefly'}.get(tool, 'bound the output')

def _read_key(path, cwd: str | None=None) -> str:
    """One spelling of a path, so `Read` and `cat` name the same file.

    Without this the two halves of the duplicate rule keep separate books:
    `Read` records `/repo/pyproject.toml` and `cat pyproject.toml` looks up
    `pyproject.toml`, and a file read by one tool and re-read by the other is
    two identities that never meet. Falls back to the string as written, since
    an unresolvable path is still a usable key -- just a narrower one.
    """
    return _resolve_path(str(path), cwd) or str(path)

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

def _bash_duplicate(command: str, state: GuardState, *, cwd: str | None=None) -> list[tuple[ReadTarget, str]]:
    """Files this command reads that the context already holds in full, or [].

    Empty unless *every* file it reads is already known. A command that reads
    one known file and one new one brings something, and refusing it would cost
    the new half to save the old.

    A slice counts as a hit, which looks asymmetric next to `observe` refusing
    to *record* one. It is not: `state.reads` only ever holds whole reads, so a
    hit means the entire file is in the context and `sed -n '40,80p'` of it is
    asking for lines that are already there. Recording a slice would claim
    something that is not true; refusing one rests on a claim that is.
    """
    targets = tool_targets('Bash', {'command': command}, cwd=cwd)
    if not targets:
        return []
    hits = []
    for t in targets:
        why = _already_known(t.path, state)
        if not why:
            return []
        hits.append((t, why))
    return hits

def _target(tool: str, tool_input: dict) -> str:
    """A stable name for what a call is about, for the refuse-once ledger.

    A path for a read, a command *shape* for a shell call -- the same reduction
    the state file uses everywhere else, so nothing that reaches disk carries
    an argument. Shape rather than the literal command on purpose: refusing
    `cat a.log` and then refusing `cat b.log` is refusing the same habit twice,
    and once is the limit.
    """
    tool_input = tool_input or {}
    if tool == 'Read':
        # A path, in the clear, because `state.reads` already holds paths and
        # the guard cannot match a re-read without them. Normalised, so that a
        # refusal of `cat f` and a refusal of `Read f` are one entry in the
        # refuse-once ledger rather than two chances to say no about one file.
        fp = tool_input.get('file_path')
        return f'Read:{_read_key(fp)}' if fp else ''
    if tool == 'Bash':
        return f'Bash:{shape(str(tool_input.get("command") or ""))}'
    # Everything else is identified by a digest rather than by its text. A grep
    # pattern or a URL is an argument, and arguments do not reach this file --
    # `shape()` exists to strip them from commands, and a hash is the same
    # promise kept for the tools that have no shape.
    raw = str(tool_input.get('pattern') or tool_input.get('url') or tool_input.get('query') or tool_input.get('description') or '')
    return f'{tool}:{hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:12]}' if raw else ''

def _refusable(target: str, state: GuardState) -> bool:
    """May the guard refuse this target, or has it already refused it once?

    The one rule that makes refusing safe. A guard that says no to the same
    call every time it is tried is not saving money, it is a session that
    cannot finish -- and the model has no way to tell the guard it knows
    something the guard does not. Refusing once and then standing aside means
    the worst case of a wrong refusal is one wasted turn, and the model always
    has a way through.
    """
    return bool(target) and target not in state.denied

def _refused(target: str, state: GuardState, *, now: float | None=None) -> GuardState:
    """Record a refusal, so the next attempt at the same target goes through."""
    if target:
        state.denied[target] = time.time() if now is None else now
        if len(state.denied) > MAX_REMEMBERED_DENIALS:
            keep = MAX_REMEMBERED_DENIALS // 2
            state.denied = dict(sorted(state.denied.items(), key=lambda kv: -kv[1])[:keep])
    return state

def needs_pricing(tool: str, tool_input: dict, *, sizes: SizeModel | None=None, state: GuardState | None=None, min_tokens: int=2000, cfg: Settings | None=None, cwd: str | None=None) -> bool:
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
        # Ahead of the `advised` short-circuit on purpose. Having said
        # something once about a command *shape* is a reason not to say it
        # again about size; it is not a reason to stop noticing that this
        # particular file is already in the context.
        if _bash_duplicate(cmd, state, cwd=cwd):
            return True
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
    # `cfg` wins when it is given, because the floor is per tool now and
    # `min_tokens` cannot express that. The bare argument stays for callers
    # that genuinely mean one number -- the tests that sweep it, mostly.
    floor = cfg.min_tokens_for(tool) if cfg is not None else min_tokens
    return sizes.predict_tool(tool, tool_input).p90 >= floor

def decide(tool: str, tool_input: dict, *, model: str, remaining_turns: int, cfg: Settings | None=None, sizes: SizeModel | None=None, state: GuardState | None=None, carry=None, context_tokens: int=0, p_redo: float=0.0, cwd: str | None=None) -> Verdict:
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
            why = _already_known(_read_key(fp, cwd), state)
            if why:
                est = sizes.predict_tool('Read', tool_input)
                # The size floor exists to stop the guard interrupting about
                # reads that were never going to be expensive. A refusal is not
                # an interruption of that kind -- nothing is being weighed, the
                # call returns what the context already holds -- so the only
                # test it has to pass is the ledger one below: is the refusal
                # worth more than the sentence that carries it.
                dup_target = _target('Read', tool_input)
                would = cfg.would_refuse and _refusable(dup_target, state)
                refusing, shadowing = would and cfg.enforcing, would and cfg.shadowing
                if est.p90 >= cfg.min_tokens_for('Read') or would:
                    saving = admitted_token_cost(est.p90, model, remaining_turns, carry=carry, context_tokens=context_tokens)
                    name = Path(str(fp)).name
                    if refusing or shadowing:
                        msg = f'[adder] Not re-read: {name} {why}. Use the copy you have. Re-issue this exact call if you need it anyway.'
                    else:
                        msg = f'[read guard] {name} {why}. Reading it admits ~{est.p90:,} tokens again for ~${saving:,.2f} and no new information.'
                    over = 0.0 if shadowing else _advice_cost(msg, model, remaining_turns, carry, context_tokens)
                    return _ledger_gate(Verdict(True, 'duplicate read of an unchanged file', kind='duplicate', message=msg, tokens=est.p90, inline=saving, saving=saving, overhead=over, advice_taken=cfg.advice_taken, estimate=est, deny=refusing, certain=True, target=dup_target, shadow=shadowing), state, cfg, 'Read')
    if tool == 'Bash':
        cmd = tool_input.get('command') or ''
        hits = _bash_duplicate(cmd, state, cwd=cwd)
        if hits:
            v = _duplicate_gate(tool_input, hits, model, remaining_turns, cfg, state, sizes, carry, context_tokens)
            if v is not None:
                return v
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
        return _brief_gate(tool, tool_input, tokens, est, model, remaining_turns, cfg, state, carry, context_tokens)
    floor = cfg.min_tokens_for(tool)
    if tokens < floor:
        return Verdict(False, f'predicted {tokens:,} tok, below the {floor:,} floor for {tool}', tokens=tokens, estimate=est)
    inline = admitted_token_cost(tokens, model, remaining_turns, carry=carry, context_tokens=context_tokens)
    if inline < cfg.min_cost:
        return Verdict(False, f'${inline:,.2f} to carry, below the ${cfg.min_cost:.2f} floor', kind='size', tokens=tokens, inline=inline, estimate=est)
    _, sub, placement = placement_cost(tokens_read=tokens, summary_tokens=max(200, tokens // 10), remaining_turns=remaining_turns, main_model=model, p_redo=p_redo, carry=carry, context_tokens=context_tokens)
    if not placement.ok and placement.saving == 0.0:
        return Verdict(False, placement.reason, kind='size', tokens=tokens, inline=inline, delegated=sub, estimate=est)
    saving = inline - sub
    if saving <= 0:
        return Verdict(False, 'delegating this one costs more than carrying it', kind='size', tokens=tokens, inline=inline, delegated=sub, saving=saving, estimate=est)
    lines = 0
    if tool in NARROWABLE:
        # The same budget for both: a grep match and a source line are the same
        # unit of admitted text, so pricing them apart would be inventing a
        # second number for one quantity.
        lines = _affordable_lines(model, remaining_turns, cfg, carry, context_tokens)
    if tool == 'Read':
        how = f'read at most ~{lines:,} lines of it (`limit: {lines}`), or delegate the read'
    else:
        how = _bounded_hint(tool)
    target = _target(tool, tool_input)
    # `full` refuses what `certain` only prices. The claim is weaker than the
    # duplicate one -- it rests on a horizon estimate and on a subagent
    # actually returning a brief -- which is why it is a separate level, and
    # why the refusal still names the cheaper call rather than just saying no.
    refusing = cfg.enforce == 'full' and saving > 0 and _refusable(target, state)
    # A refusal the guard can carry out itself. Only reachable where it was
    # going to refuse anyway, so this strictly relaxes the denial -- and it is
    # priced against what the bounded call actually admits, not against the
    # whole read, because the bounded call does run.
    sub_input = _narrowed(tool, tool_input, lines) if (refusing and cfg.narrow) else None
    if sub_input is not None:
        from adder.decide.narrow import describe as _describe
        kept = admitted_token_cost(_line_tokens(lines), model, remaining_turns, carry=carry, context_tokens=context_tokens)
        msg = f'[adder] Run {_describe(tool, tool_input, sub_input)}. Unbounded this {tool} admits {est.describe()} at ~${inline:,.2f} of carry; this way ~${kept:,.2f}.'
        over = _advice_cost(msg, model, remaining_turns, carry, context_tokens)
        return _ledger_gate(Verdict(True, 'the bounded call was substituted for the one written', kind='size', message=msg, tokens=tokens, inline=inline, delegated=kept, saving=max(0.0, inline - kept), overhead=over, advice_taken=cfg.advice_taken, estimate=est, deny=False, target=target, narrowed=sub_input), state, cfg, tool)
    if refusing:
        msg = f'[adder] Not run as written: this {tool} admits {est.describe()} at ~${inline:,.2f} of carry, against ~${sub:,.2f} delegated. Instead, {how}. Re-issue this exact call if you need all of it.'
    else:
        msg = f'[read guard] This {tool} admits {est.describe()} to a context re-read ~{remaining_turns:,.0f} more times: ~${inline:,.2f} to carry, vs ~${sub:,.2f} delegated to a subagent. If you need one fact from it, {how}.'
    over = _advice_cost(msg, model, remaining_turns, carry, context_tokens)
    return _ledger_gate(Verdict(True, 'predicted carry exceeds the cost of saying so', kind='size', message=msg, ask=cfg.block and tokens >= cfg.hard_tokens, tokens=tokens, inline=inline, delegated=sub, saving=saving, overhead=over, advice_taken=cfg.advice_taken, estimate=est, deny=refusing, target=target), state, cfg, tool)

def _brief_gate(tool: str, tool_input: dict, tokens: int, est, model: str, remaining_turns: int, cfg: Settings, state: GuardState, carry, context_tokens: int) -> Verdict:
    """Price a subagent's *return*, and the tier it is about to run on.

    The subagent's own reads are already outside the main window -- that is
    what delegating bought. What is left to decide is how much it hands back,
    and a return is compared against a brief rather than against an alternative
    placement.

    The second question is what it runs on, and it is answered here rather than
    in a second hook for one reason: two sentences injected about one call are
    carried for the rest of the session twice. `_tier_clause` is folded into
    whichever message this gate was going to send anyway, and when the return
    size has nothing to say it becomes the message.
    """
    tier = _tier_advice(tool_input, model, remaining_turns, cfg, state, carry, context_tokens)
    if tokens <= BRIEF_TARGET_TOKENS:
        return _tier_only(tier, state, cfg, tool) or Verdict(False, f'predicted {tokens:,} tok back, already within a brief', kind='brief', tokens=tokens, estimate=est)
    inline = admitted_token_cost(tokens, model, remaining_turns, carry=carry, context_tokens=context_tokens)
    if inline < cfg.min_cost:
        return _tier_only(tier, state, cfg, tool) or Verdict(False, f'${inline:,.2f} to carry, below the ${cfg.min_cost:.2f} floor', kind='brief', tokens=tokens, inline=inline, estimate=est)
    bounded = admitted_token_cost(BRIEF_TARGET_TOKENS, model, remaining_turns, carry=carry, context_tokens=context_tokens)
    saving = inline - bounded
    # Keyed on the tool and not on the task, so this is once per session rather
    # than once per description. A `Task` description is different every time,
    # so a per-target ledger would never stop a second refusal -- and refusing
    # delegation twice is refusing the lever the whole tool is arguing for.
    target = f'{tool}:brief'
    refusing = cfg.enforce == 'full' and _refusable(target, state)
    if refusing:
        msg = f'[adder] Not delegated as written: returns from here have run {est.describe()}, which costs ~${inline:,.2f} to carry. Re-issue it asking for under {BRIEF_TARGET_TOKENS:,} tokens — the findings, not the transcript — for ~${bounded:,.2f}.'
    else:
        msg = f'[read guard] Subagent returns here have run {est.describe()}, admitted to a context re-read ~{remaining_turns:,.0f} more times: ~${inline:,.2f} to carry. Ask it for under {BRIEF_TARGET_TOKENS:,} tokens — the findings, not the transcript — and that becomes ~${bounded:,.2f}.'
    reason, said, claims = 'a subagent return larger than a brief', '', 1
    if tier.fire:
        # Priced on the joined text, not on the two halves: what the session
        # carries is one message. The saving adds because the two claims are
        # independent -- how much comes back, and what produced it.
        msg = f'{msg} {tier.clause or tier.message}'
        saving += tier.saving * cfg.advice_taken
        reason += ', and a cheaper tier to run it on'
        said, claims = tier.target, 2
    over = _advice_cost(msg, model, remaining_turns, carry, context_tokens)
    return _ledger_gate(Verdict(True, reason, kind='brief', message=msg, tokens=tokens, inline=inline, delegated=bounded, saving=saving, overhead=over, advice_taken=cfg.advice_taken, estimate=est, deny=refusing, target=target, tier_target=said, claims=claims), state, cfg, tool)

def _tier_advice(tool_input: dict, model: str, remaining_turns: int, cfg: Settings, state: GuardState, carry, context_tokens: int):
    """Ask `delegate.advise` where this delegated step should run.

    Wrapped so the guard never grows a second failure mode: routing needs the
    classifier, the outcome log and the ladder, and none of those are worth a
    tool call failing over. A guard that dies on a `Task` is worse than a guard
    with nothing to say about it.
    """
    from adder.decide.delegate import Advice, advise
    if not cfg.route:
        return Advice(False, 'tier advice is off')
    try:
        # No dollar floor here, only the solvency test. `min_cost` is the
        # threshold for *interrupting*, and a clause appended to a message the
        # guard was already sending interrupts nothing -- it costs the words and
        # nothing else. The floor is applied in `_tier_only`, which is the path
        # where this becomes an interruption of its own.
        got = advise(tool_input or {}, session_model=model, remaining_turns=remaining_turns,
                     context_tokens=context_tokens, carry=carry,
                     advice_taken=cfg.advice_taken, min_cost=0.0)
    except Exception:
        return Advice(False, 'could not price the ladder')
    # Once per tier per session. The ladder does not change between two `Task`
    # calls, so a second sentence saying the same thing is pure overhead --
    # and this one is charged to the context whether or not anybody acts on it.
    if got.fire and got.target in state.advised:
        return Advice(False, f'already said {got.target} this session')
    return got

def _tier_only(tier, state: GuardState, cfg: Settings, tool: str='Task') -> Verdict | None:
    """The tier clause as a verdict of its own, when the return size is quiet.

    Two differences from the ride-along path. It carries the dollar floor,
    because a message sent for this reason alone is an interruption and the
    floor is what stops the guard interrupting over small change. And it is
    never a denial: the guard may refuse a read, but refusing a delegation would
    refuse the largest lever this tool has, on the strength of a classifier that
    abstains by design.
    """
    if not tier.fire or tier.saving < cfg.min_cost:
        return None
    return _ledger_gate(Verdict(True, 'a cheaper tier for this delegated step', kind='tier', message=tier.message, saving=tier.saving, overhead=tier.overhead, advice_taken=cfg.advice_taken, tier_target=tier.target), state, cfg, tool)

def _duplicate_gate(tool_input: dict, hits: list[tuple[ReadTarget, str]], model: str, remaining_turns: int, cfg: Settings, state: GuardState, sizes: SizeModel, carry, context_tokens: int) -> Verdict | None:
    """The shell half of the duplicate-read rule: `cat` of a file already held.

    Same claim as the `Read` branch and the same refusal, but the way through
    is written in the language the caller is working in. A model reading with
    `cat` cannot act on advice about `limit:`, and telling it to use `Read`
    instead is advice about how the harness is configured rather than about
    this call.

    `None` rather than a non-firing `Verdict`, because this runs *before* the
    aggregate rule. Returning a verdict that says nothing would also stop the
    aggregate rule looking, and a session whose every `cat` is a duplicate is
    exactly the session whose running total is worth reporting.
    """
    if all(t.whole for t, _ in hits):
        # Every file is on disk and can be measured, which beats a distribution
        # over other commands that happened to have the same shape.
        p90 = sum(read_estimate({'file_path': t.path}).p90 for t, _ in hits)
        est = Estimate(p90, p90, 0, 'stat')
    else:
        # A slice. What this admits is bounded by how the command was written,
        # not by how big the file it names is.
        est = sizes.predict_tool('Bash', tool_input)
    target = _target('Read', {'file_path': hits[0][0].path})
    why = hits[0][1]
    would = cfg.would_refuse and _refusable(target, state)
    refusing, shadowing = would and cfg.enforcing, would and cfg.shadowing
    if est.p90 < cfg.min_tokens_for('Bash') and not would:
        return None
    saving = admitted_token_cost(est.p90, model, remaining_turns, carry=carry, context_tokens=context_tokens)
    name = Path(hits[0][0].path).name
    more = f', as are the {len(hits) - 1} other files it reads' if len(hits) > 1 else ''
    if refusing or shadowing:
        msg = f'[adder] Not run: {name} {why}{more}. The copy this context already holds is the same bytes — use it. Re-issue this exact call if you need it anyway.'
    else:
        msg = f'[read guard] {name} {why}{more}. Running this admits ~{est.p90:,} tokens again for ~${saving:,.2f} and no new information.'
    over = 0.0 if shadowing else _advice_cost(msg, model, remaining_turns, carry, context_tokens)
    v = _ledger_gate(Verdict(True, 'shell re-read of a file the context already holds', kind='duplicate', message=msg, tokens=est.p90, inline=saving, saving=saving, overhead=over, advice_taken=cfg.advice_taken, estimate=est, deny=refusing, certain=True, target=target, shadow=shadowing), state, cfg, 'Bash')
    return v if v.fire else None

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
    # The one class where refusing is *more* proportionate than advising. Every
    # individual call here was small and correctly waved through; what is being
    # refused is the two-hundred-and-forty-seventh of them, and a sentence
    # about a habit is the easiest kind of sentence to read past. Once per
    # shape per session either way, so the next call goes through.
    target = f'Bash:{sh}'
    refusing = cfg.enforce == 'full' and _refusable(target, state)
    if refusing:
        msg = f'[adder] Not run: `{sh}` has already admitted ~{running:,} tokens over {calls:,} calls this session, ~${inline:,.2f} of carry. Each was small; the total is not. Read it once, delegate it, or keep the results out of context — or re-issue to run it anyway.'
    else:
        msg = f'[read guard] `{sh}` has run {calls:,} times this session and admitted ~{running:,} tokens ({per_call:,} a call) into a context re-read ~{remaining_turns:,.0f} more times — ~${inline:,.2f} of carry. Each call was small; the total is not. Read it once, delegate it, or keep the results out of the main context.'
    over = _advice_cost(msg, model, remaining_turns, carry, context_tokens)
    saving = inline * 0.5
    return _ledger_gate(Verdict(True, 'one command shape has admitted more than the floor', kind='aggregate', message=msg, tokens=running, inline=inline, saving=saving, overhead=over, advice_taken=cfg.advice_taken, deny=refusing, target=target), state, cfg, 'Bash')

def _advice_cost(message: str, model: str, remaining_turns: int, carry, context_tokens: int) -> float:
    """What injecting this sentence costs over the rest of the session.

    The message is admitted to the context exactly like a tool result, so it is
    priced exactly like one. Roughly 55 tokens carried across 300 remaining
    turns is real money at Opus rates, and it is money the guard spends whether
    or not its advice is taken.
    """
    return admitted_token_cost(est_tokens(message), model, remaining_turns, carry=carry, context_tokens=context_tokens)

def _ledger_gate(v: Verdict, state: GuardState, cfg: Settings, tool: str='') -> Verdict:
    """The guard's own solvency test, applied before it is allowed to speak.

    Shadow verdicts skip both halves of it, and neither exemption is a
    loophole. The fire ceiling bounds how often the guard may *interrupt*, and
    a shadow verdict interrupts nothing; the solvency test asks whether a
    sentence is worth what carrying it costs, and a shadow verdict has no
    sentence and costs nothing. Applying either would silently truncate the
    measurement -- at `guard_max_fires` findings a session -- and a shadow
    report that stops counting partway through is worse than none, because it
    reads as a complete one.
    """
    if v.shadow:
        return v
    if state.fires >= cfg.max_fires:
        return Verdict(False, f'already advised {state.fires} times this session', kind=v.kind, tokens=v.tokens, inline=v.inline, delegated=v.delegated, saving=v.saving, overhead=v.overhead, advice_taken=cfg.advice_taken, estimate=v.estimate)
    if tool and state.fires_by_tool.get(tool, 0) >= cfg.max_fires_for(tool):
        return Verdict(False, f'already advised {state.fires_by_tool.get(tool, 0)} times about {tool} this session', kind=v.kind, tokens=v.tokens, inline=v.inline, delegated=v.delegated, saving=v.saving, overhead=v.overhead, advice_taken=cfg.advice_taken, estimate=v.estimate)
    if v.net <= 0:
        return Verdict(False, f'saying so costs ${v.overhead:,.4f} to carry and is worth ${v.saving * cfg.advice_taken:,.4f}', kind=v.kind, tokens=v.tokens, inline=v.inline, delegated=v.delegated, saving=v.saving, overhead=v.overhead, advice_taken=cfg.advice_taken, estimate=v.estimate)
    return v

def _remember_read(state: GuardState, path: str, mtime: float) -> None:
    """Record that `path` is in the context, keeping `reads` bounded."""
    state.reads[path] = mtime
    if len(state.reads) > MAX_REMEMBERED_READS:
        keep = MAX_REMEMBERED_READS // 2
        state.reads = dict(list(state.reads.items())[-keep:])

def observe(tool: str, tool_input: dict, state: GuardState, verdict: Verdict, *, now: float | None=None, sizes: SizeModel | None=None, cwd: str | None=None) -> GuardState:
    """Fold this call into the session's memory. Returns the same state object.

    Bash is accumulated whether or not the guard had anything to say about it,
    because the aggregate rule only works if the small bounded calls are
    counted -- they are the ones that add up to most of it. It is also the tool
    that does the *reading* under a harness told to prefer the shell, so it
    populates `reads` on exactly the same terms `Read` does: whole files only.
    """
    tool_input = tool_input or {}
    if tool == 'Bash' and not verdict.deny:
        # `whole_reads` refuses a slice, a glob, a redirect and a file larger
        # than the harness truncates at -- each of which would put a path in
        # here that is not actually in the context, and the guard is the one
        # component whose confidently wrong sentence changes what happens.
        for path in whole_reads('Bash', tool_input, cwd=cwd):
            mtime = _mtime(path)
            if mtime >= 0:
                _remember_read(state, path, mtime)
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
            state.wrote[_read_key(fp, cwd)] = time.time() if now is None else now
            state.reads.pop(_read_key(fp, cwd), None)
            if len(state.wrote) > MAX_REMEMBERED_READS:
                keep = MAX_REMEMBERED_READS // 2
                state.wrote = dict(list(state.wrote.items())[-keep:])
    if tool == 'Read' and verdict.deny:
        # A denied read never happens, so nothing was admitted and there is
        # nothing to remember. Recording it would also make the guard's own
        # refusal the evidence for the next one.
        pass
    elif tool == 'Read':
        fp = tool_input.get('file_path')
        # Only a *whole* read puts the whole file in the context. A read with
        # `limit` or `offset` admitted a slice, and remembering it as a full
        # read made the guard tell the model that a later complete read was
        # `already in this context and has not changed on disk` -- advice
        # against the one move that would actually have got the rest of the
        # file. The guard is the only component here that changes behaviour,
        # so a confidently wrong sentence from it is the expensive kind.
        if fp and not tool_input.get('limit') and not tool_input.get('offset'):
            key = _read_key(fp, cwd)
            _remember_read(state, key, _mtime(key))
    _note_contradiction(tool, tool_input, state, verdict, cwd=cwd)
    if verdict.shadow:
        # Booked in its own fields and nowhere else. A machine in shadow mode
        # has to be able to report, truthfully, that its guard has spoken zero
        # times, promised nothing and cost nothing -- because it has.
        state.shadow_fires += 1
        state.shadow_saving += verdict.saving
        if verdict.target:
            state.shadowed[verdict.target] = time.time() if now is None else now
            if len(state.shadowed) > MAX_REMEMBERED_DENIALS:
                keep = MAX_REMEMBERED_DENIALS // 2
                state.shadowed = dict(sorted(state.shadowed.items(),
                                             key=lambda kv: -kv[1])[:keep])
        return state
    if verdict.fire:
        state.fires += 1
        state.fires_by_tool[tool] = state.fires_by_tool.get(tool, 0) + 1
        state.saving += verdict.saving
        state.overhead += verdict.overhead
        if verdict.deny:
            # Kept apart from `saving` on purpose. `saving` is what the guard
            # argued for and half-believes; `prevented` is a call that did not
            # happen. Only the second one can be reported without an uptake
            # assumption attached, and mixing them would put the assumption
            # back into the one number that does not need it.
            state.prevented += verdict.saving
            _refused(verdict.target, state, now=now)
        if verdict.tier_target and verdict.tier_target not in state.advised:
            state.advised.append(verdict.tier_target)
        if tool == 'Bash':
            sh = shape(tool_input.get('command') or '')
            if sh not in state.advised:
                state.advised.append(sh)
    return state

def _contradicts(tool: str, tool_input: dict, state: GuardState, *, cwd: str | None=None) -> str:
    """Which shadow refusal, if any, this call would have been blocked by.

    Two shapes count, and they are the two the report names. The obvious one is
    the same target asked for again: under real enforcement that is the escape
    hatch firing, which only happens when the model had a reason the guard
    could not see. The second is a duplicate `Read` refusal followed by the
    file arriving through the shell instead -- `Read:` and `Bash:` are
    different targets, so nothing above would match them, and it is exactly the
    case `core.reads` exists for.
    """
    direct = _target(tool, tool_input)
    if direct and direct in state.shadowed:
        return direct
    if tool == 'Bash':
        for path in tool_targets('Bash', tool_input, cwd=cwd):
            key = f'Read:{_read_key(path.path)}'
            if key in state.shadowed:
                return key
    return ''


def _note_contradiction(tool: str, tool_input: dict, state: GuardState, verdict: Verdict, *, cwd: str | None=None) -> None:
    """Record that the session went and did what a shadow refusal would have stopped.

    This is the half of shadow mode that can say no. Everything else it
    measures is a saving it is confident about; this is the evidence against.
    A refusal the session immediately worked around did not save anything -- it
    would have cost a turn -- and the ratio between the two is what somebody
    needs before they agree to be refused for real.

    Not counted for the call that created the shadow entry, and not counted
    twice for one call.
    """
    if verdict.shadow:
        return
    hit = _contradicts(tool, tool_input, state, cwd=cwd)
    if hit:
        state.contradicted[hit] = state.contradicted.get(hit, 0) + 1


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
        row = {'ts': time.time() if now is None else now, 'session': session, 'tool': tool, 'kind': v.kind, 'shape': shape((tool_input or {}).get('command') or '') if tool == 'Bash' else '', 'prog': _leading(tool_input) if tool == 'Bash' else '', 'name': Path(str((tool_input or {}).get('file_path') or '')).name, 'tokens': v.tokens, 'action': v.action, 'saving': round(v.saving, 6), 'target': v.target, 'reason': v.reason}
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

# Where the measured uptake is cached, and how stale it may get before a
# `--learn` re-derives it. Mirrors `size_model` / `size_max_age` deliberately:
# the same problem (a scan too expensive for the hook's hot path, a number the
# hook needs on every call) already has an answer in this repo, and a second
# shape for it would be a second thing to reason about.
UPTAKE_PATH = Path.home() / '.claude' / '.adder-uptake.json'

# The measured rate is never allowed below this, and the reason is not caution.
# `advice_taken` gates whether advice is worth saying at all, so a measured 0
# stops the guard speaking -- and a guard that does not speak records no fires,
# so nothing can ever measure it again. The estimator would seal itself shut on
# one bad week and there would be no path back that did not involve editing a
# config file nobody knows exists. The floor is what keeps the loop open.
UPTAKE_FLOOR = 0.1


def uptake_path() -> Path:
    """The cache location in effect: the `uptake_cache` setting, or the default."""
    try:
        from adder.core.settings import configured_path
        return configured_path('uptake_cache', UPTAKE_PATH)
    except Exception:
        return UPTAKE_PATH


def save_uptake(u: Uptake, path: Path | None=None) -> Path:
    """Write the measurement. Never raises: this is a cache, not a record."""
    p = Path(path) if path is not None else uptake_path()
    with contextlib.suppress(OSError, TypeError, ValueError):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({'rate': round(u.rate, 4), 'fires': u.fires, 'changed': u.changed, 'measured': u.measured, 'ts': time.time()}), encoding='utf-8')
    return p


def load_uptake(path: Path | None=None) -> tuple[float, bool, float]:
    """`(rate, measured, age_seconds)` from the cache; `(0.5, False, inf)` if none.

    The fallback is the assumption the whole thing exists to replace, which is
    the correct thing to fall back to: an unreadable cache must leave the guard
    exactly as it behaved before this cache existed.
    """
    # Resolving the path is inside the try, not before it. `uptake_path` reads
    # the settings layer and touches `Path.home()`, either of which can raise on
    # a machine with no home directory or an unreadable config -- and this runs
    # inside `Settings.resolve`, which runs before every tool call. An exception
    # escaping here does not degrade the guard, it takes the tool call with it.
    try:
        p = Path(path) if path is not None else uptake_path()
        d = json.loads(p.read_text(encoding='utf-8'))
        rate = float(d['rate'])
        if not 0.0 <= rate <= 1.0:
            return (0.5, False, float('inf'))
        return (rate, bool(d.get('measured')), max(0.0, time.time() - float(d.get('ts') or 0.0)))
    except Exception:
        return (0.5, False, float('inf'))


def refresh_uptake(root=None, *, path: Path | None=None, log: Path | None=None) -> Uptake:
    """Re-measure and cache. Called by `adder guard --learn`, never by the hook.

    Measuring reads every transcript under `root`, which is a second or two --
    fine once, and out of the question on a path that runs before every tool
    call. So the hook reads the cached number and the scan happens when somebody
    asks for it, exactly as the size model does.
    """
    u = uptake(root, log=log)
    save_uptake(u, path)
    return u


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
        # A shadow fire injected no sentence, so there was nothing for anybody
        # to act on. Counting it here would fold a mode that says nothing into
        # the measurement of whether saying something works, and would push the
        # rate towards zero on exactly the machines running shadow to find out
        # what the rate is.
        if f.get('action') == 'shadow':
            continue
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
    refusals: int = 0
    prevented: float = 0.0

    @property
    def net(self) -> float:
        return self.saving + self.prevented - self.overhead

    @property
    def assumed_share(self) -> float:
        """How much of the headline rests on the uptake assumption.

        The number worth looking at when deciding whether to believe a replay.
        A result that is mostly `prevented` is a result about calls that would
        not have happened; a result that is mostly `saving` is a result about
        sentences somebody may or may not have acted on.
        """
        total = self.saving + self.prevented
        return self.saving / total if total > 0 else 0.0

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
    # The tier clause is switched off for the replay, and this is not a detail.
    # `bench` already models what routing delegated work to a cheaper tier is
    # worth, as its own rung ("+ the tier agents"); counting the sentence that
    # argues for it here as well would book one dollar twice and inflate every
    # ratio taken against the null configuration. What this function measures is
    # the read guard.
    cfg = replace(cfg, route=False)
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
        if not needs_pricing(tool, inp, sizes=sizes, state=state, cfg=cfg):
            observe(tool, inp, state, Verdict(False, 'below floor'), sizes=sizes)
            continue
        rep.priced += 1
        v = decide(tool, inp, model=model, remaining_turns=int(remaining), cfg=cfg, sizes=sizes, state=state)
        observe(tool, inp, state, v, sizes=sizes)
        if not v.fire:
            continue
        rep.fires += 1
        rep.by_kind[v.kind] = rep.by_kind.get(v.kind, 0) + 1
        # A refusal is booked whole and a sentence is booked discounted, which
        # is the same rule `Verdict.uptake` applies live. Keeping the two in
        # separate fields rather than summing them here is the point: it is
        # what lets the report say how much of its own headline is an
        # assumption.
        if v.deny:
            rep.refusals += 1
            rep.prevented += v.saving
            credited = v.saving
        else:
            rep.saving += v.saving * taken
            credited = v.saving * taken
        rep.overhead += v.overhead
        rep.biggest.append((credited, f'{v.kind}: {v.message[13:110]}'))
    rep.biggest.sort(reverse=True)
    rep.biggest = rep.biggest[:top]
    return rep
# Inside the package, not under `.claude/`. The wheel prunes `.claude/`, so for
# four releases the hook this path names existed only in a git checkout and every
# `pip install` user was handed an install snippet pointing at nothing.
HOOK_RELPATH = 'adder/decide/hooks/pretooluse_read_guard.py'
# Every spelling of "the read guard is declared here" that this tool has ever
# written. Three, because the command form changed twice: an absolute script
# path, `-m <module>`, and `adder hook read-guard` -- the last so a
# project-scope install names something that resolves on somebody else's
# machine. `installed_in` matching only the first meant that after the change
# an installed guard reported as absent, which is the same as reporting that
# nothing is preventing spend when something is.
HOOK_MARKERS: tuple[str, ...] = ('pretooluse_read_guard', 'hook read-guard')

def hook_path(repo: Path | None=None) -> Path:
    """Absolute path to the shipped hook, for the install snippet.

    `repo` is the directory that *holds* the `adder` package -- a checkout root
    or a `site-packages`. Both resolve the same way, which is the point of
    keeping the hooks inside the package.
    """
    here = Path(__file__).resolve()
    root = repo or here.parents[2]
    return root / HOOK_RELPATH

def interpreter() -> str:
    """The python the hook should be run with, quoted for a shell if it needs it.

    `sys.executable`, not the string `python3`. A hook command runs through the
    user's shell, where `python3` is whatever is first on PATH -- on macOS that
    is routinely 3.9, which cannot import this package at all, and the failure
    surfaces as a hook error on every tool call rather than as anything about
    versions. The interpreter that installed adder is the one that can import it.
    """
    return shlex.quote(sys.executable) if sys.executable else 'python3'

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
        if any(m in text for m in HOOK_MARKERS):
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
    from adder.decide.auto import HOOK_TIMEOUT_S, HOOKS, hook_command
    by_name = {h['name']: h for h in HOOKS}
    guard_cmd = hook_command(by_name['read-guard'])
    learn_cmd = hook_command(by_name['compact-learn'])
    return json.dumps({'hooks': {'PreToolUse': [{'matcher': '|'.join(OBSERVED), 'hooks': [{'type': 'command', 'command': guard_cmd, 'timeout': HOOK_TIMEOUT_S}]}], 'PreCompact': [{'hooks': [{'type': 'command', 'command': learn_cmd, 'timeout': HOOK_TIMEOUT_S}]}]}}, indent=2)

@dataclass(frozen=True)
class Shadow:
    """What shadow mode found, across every session in the state file.

    The point of this dataclass is the pair of numbers at the top. `would_save`
    is the counterfactual saving of refusals that were never made; `contradicted`
    is how many of those refusals the session went on to work around. The first
    is what enforcement offers, the second is what it costs when it is wrong,
    and until now the only thing standing between a user and that trade was an
    assumed 0.5.
    """
    sessions: int = 0
    fires: int = 0
    would_save: float = 0.0
    contradicted: int = 0
    contradictions: int = 0
    targets: list[tuple[str, int]] = field(default_factory=list)

    @property
    def contradiction_rate(self) -> float:
        """Share of shadow refusals the session went on to contradict."""
        return self.contradicted / self.fires if self.fires else 0.0

    @property
    def realised(self) -> float:
        """The saving left once every contradicted refusal is written off whole.

        Deliberately harsh. A contradicted refusal did not merely fail to save
        its tokens -- under enforcement it would have cost a turn to work
        around -- so crediting it with any part of its modelled saving would
        make the pessimistic case read better than it is. This is the number to
        quote, and it is a lower bound.
        """
        if not self.fires:
            return 0.0
        return self.would_save * (1.0 - self.contradiction_rate)

    @property
    def measured(self) -> bool:
        """Enough findings to say anything. Below this it is an anecdote."""
        return self.fires >= 10


def shadow(path: Path | None=None) -> Shadow:
    """Read every session's shadow record out of the guard's state file.

    The state file rather than the fires log, because the contradiction is a
    property of a *session* -- the same target asked for again after we would
    have refused it -- and the state file is the only thing that holds it. It
    keeps 200 sessions and 14 days, which is the window this can speak about.
    """
    p = Path(path) if path is not None else Settings.resolve().state_path
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Shadow()
    if not isinstance(blob, dict):
        return Shadow()
    sessions = fires = contradicted = contradictions = 0
    total = 0.0
    per_target: dict[str, int] = {}
    for entry in blob.values():
        st = GuardState.from_json(entry if isinstance(entry, dict) else {})
        if not st.shadow_fires and not st.contradicted:
            continue
        sessions += 1
        fires += st.shadow_fires
        total += st.shadow_saving
        contradicted += len(st.contradicted)
        for target, n in st.contradicted.items():
            contradictions += n
            per_target[target] = per_target.get(target, 0) + n
    ranked = sorted(per_target.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
    return Shadow(sessions=sessions, fires=fires, would_save=total,
                  contradicted=contradicted, contradictions=contradictions,
                  targets=ranked)


def last_session(path: Path | None=None) -> tuple[str, list[dict]]:
    """The newest session in the fires log, and everything the guard did in it.

    Exists because a refusal the user cannot inspect is a refusal they will
    turn off the first time they suspect it. Every other report here is an
    aggregate over weeks; this one answers "what did it just do to me", which
    is the question actually being asked at the moment somebody reaches for
    `guard_enforce=off`.
    """
    rows = load_fires(path)
    if not rows:
        return '', []
    newest = max(rows, key=lambda r: _num(r.get('ts')))
    session = str(newest.get('session') or '')
    mine = [r for r in rows if str(r.get('session') or '') == session]
    mine.sort(key=lambda r: _num(r.get('ts')))
    return session, mine


def _identity(row: dict) -> str:
    """What one logged fire was about, in the terms the log actually keeps.

    Identities only -- a shape, a basename, a digest. `record_fire` promises
    that nothing here carries an argument, and a report that reconstructed one
    would break the promise on the way out.
    """
    for key in ('name', 'shape', 'target'):
        value = str(row.get(key) or '')
        if value:
            return value
    return str(row.get('tool') or '?')


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
    fires = saving = overhead = prevented = 0.0
    sessions = 0
    for row in blob.values():
        if not isinstance(row, dict):
            continue
        sessions += 1
        prevented += _num(row.get('prevented'))
        # `_num`, not bare `float()`. This file is written by a hook that can be
        # killed mid-write and is small enough that people edit it; a
        # non-numeric `fires` raised `ValueError` straight out of `adder guard`,
        # from the section whose whole purpose is to say whether the guard is
        # solvent.
        fires += _num(row.get('fires'))
        saving += _num(row.get('saving'))
        overhead += _num(row.get('overhead'))
    return {'sessions': sessions, 'fires': int(fires), 'saving': saving, 'overhead': overhead, 'prevented': prevented, 'path': str(p)}

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

def _uptake_line(cfg: Settings) -> str:
    """The uptake discount, and where the number came from.

    Printing `50% of advice acted on` without saying whether anything measured
    that is how the assumption survived so long: it reads like a finding.
    """
    rate, measured, age = load_uptake()
    if not measured:
        return f'{cfg.advice_taken:.0%} of advice acted on — ASSUMED; run `adder guard --learn` to measure it'
    hours = age / 3600.0
    when = 'just now' if hours < 0.5 else f'{hours:.0f}h ago'
    floored = ' (at the floor)' if rate < UPTAKE_FLOOR else ''
    return f'{cfg.advice_taken:.0%} of advice acted on — measured {when}{floored}'


def _clip(text: str, width: int=74) -> str:
    text = ' '.join(str(text).split())
    return text if len(text) <= width else text[:width - 1] + '…'

# How recently another session must have written the shared state file for the
# two to be treated as running side by side. Wide, because the hazard is not a
# collision at one instant -- it is a tree of agents editing files over minutes.
CONCURRENT_WINDOW_S = 900.0


def concurrent_sessions(path: Path | None=None, *, window_s: float=CONCURRENT_WINDOW_S, now: float | None=None) -> int:
    """How many sessions have written the shared guard state recently.

    The duplicate rule -- the one saving here that needs no model -- keys on a
    path and its mtime, and asks "is this the same bytes I already have". That
    is the right design and it has one failure mode, which is other people. In
    a tree where several agents are working at once the mtime moves for reasons
    that have nothing to do with this session: another agent formats the file,
    a build writes it back, a sibling worktree touches it. The guard then sees
    a changed file, correctly declines to call the read a duplicate, and the
    lever silently reports less than it is worth.

    It fails in the safe direction -- a missed saving, never a wrong refusal --
    which is exactly why nothing would ever have surfaced it. This counts the
    condition so the report can say the number is a floor.
    """
    p = Path(path) if path is not None else Settings.resolve().state_path
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    if not isinstance(blob, dict):
        return 0
    cutoff = (time.time() if now is None else now) - window_s
    return sum(1 for v in blob.values() if _touched(v) >= cutoff)


def capture_gap(cfg: Settings | None=None) -> list[str]:
    """Why the guard captures less of a lever than the lever is worth.

    Two commands price the duplicate-read lever and they disagree by about 4x
    on the same corpus. Both are right, and printing two correct numbers in two
    different reports with nothing between them is the same as printing one
    wrong one: the reader has no way to know they are not the same quantity.

    `adder savings` measures the **pool** -- every token admitted to a context
    that already held it, over the whole corpus. `adder bench` prices the
    **guard**, which is a hook with a memory, a budget and a rule about not
    refusing twice. The list below is the difference, item by item, with this
    machine's own settings in it. None of these are defects; three of them are
    the safety properties that make refusing survivable at all.
    """
    cfg = cfg or Settings.resolve()
    out = [
        'once per target: the second and later re-reads of the same file go '
        'through. A guard that says no every time is a session that cannot '
        'finish, and the model has no way to tell it what it knows',
        f'at most {cfg.max_fires} findings a session, and no more than '
        f'{cfg.max_fires} about any one tool',
        'compaction clears the memory: after it, a re-read is the session '
        'recovering what it lost rather than a duplicate',
        'the rule asks by mtime, so in a tree several agents are working in it '
        'declines whenever somebody else has touched the file',
    ]
    if not cfg.enforcing:
        out.insert(0, f'nothing predicted under {cfg.min_tokens:,} tokens is '
                      f'priced at all, and nothing worth under '
                      f'${cfg.min_cost:.2f} of carry is worth interrupting for')
        out.append(f'and every dollar left is multiplied by {cfg.advice_taken:.0%} '
                   f'— the share of advice acted on. `guard_enforce=certain` '
                   f'refuses instead, which removes this term')
    return out


def floors_report(sizes=None) -> list[str]:
    """What one floor and one ceiling are doing to tools that are nothing alike.

    `guard_min_tokens` is an I/O gate: below it the guard does not parse a
    transcript, so a call below it is invisible to every rule that needs a
    price. One number served every tool, and the tools differ by an order of
    magnitude -- on the machine this was written for, `Bash` returns a p90 of
    1.2K over 2,490 calls a session and `Read` 5.9K over 58. Against a 2,000
    floor the first is almost never priced and the second usually is.

    The suggestion is derived rather than chosen: a floor set at a tool's own
    p90 prices the top decile of that tool's calls, by definition of p90. That
    is a statement about the reader's transcripts, which is the only kind this
    file is allowed to make.
    """
    from adder.core.shapes import load_model
    from adder.util.render import heading, kv, table, tokens, wrap
    cfg = Settings.resolve()
    sizes = load_model() if sizes is None else sizes
    out = heading('Floors and ceilings — one number, several distributions', rule='=')
    if not sizes.calls or not sizes.tools:
        out += [kv('state', 'no learned size model — run `adder guard --learn`')]
        return out
    rows = []
    for tool in sorted(sizes.tools):
        # Only the tools this guard can act on. The size model learns every
        # tool in the transcript, and a floor for `TodoWrite` is a row about
        # nothing -- the guard returns on `tool not in GUARDED` long before any
        # floor is consulted.
        if tool not in GUARDED:
            continue
        p50, p90, calls = sizes.tools[tool][0], sizes.tools[tool][1], sizes.tools[tool][2]
        if calls < 3:
            continue
        floor = cfg.min_tokens_for(tool)
        if floor > p90:
            share = 'under a tenth'
        elif floor > p50:
            share = 'a tenth to a half'
        else:
            share = 'over half'
        rows.append((tool, f'{calls:,}', tokens(p50), tokens(p90), tokens(floor),
                     share, f'{cfg.max_fires_for(tool)}'))
    if not rows:
        out += [kv('state', 'not enough calls per tool to say anything')]
        return out
    out += table(rows, headers=('tool', 'calls', 'p50', 'p90', 'floor',
                                'priced', 'ceiling'), align='<>>>>><')
    out += ['']
    suggest = ','.join(f'{t}={sizes.tools[t][1]}' for t, *_ in rows
                       if sizes.tools[t][1] > 0)
    out += wrap('`priced` is the share of that tool\'s calls the floor lets '
                'through to be priced at all; below the floor the guard returns '
                'before reading anything, so those calls are invisible to every '
                'rule that needs a price. A floor at a tool\'s own p90 prices '
                'its top decile, by construction:')
    out += ['', f'  guard_min_tokens_by_tool = "{suggest}"', '']
    out += wrap('The ceiling is shared as well as global: both `guard_max_fires` '
                'and the per-tool one have to pass, so raising a tool\'s entry '
                'cannot buy it more than the session allows — it can only stop '
                'the tool called two thousand times spending the budget before '
                'the tool called fifty times has said anything.')
    return out


def shadow_report(path: Path | None=None) -> list[str]:
    """What shadow mode has recorded, and what it is not entitled to say yet."""
    from adder.util.render import heading, kv, money, table, wrap
    sh = shadow(path)
    out = heading('Shadow — what it would have refused, and did not', rule='=')
    cfg = Settings.resolve()
    if not sh.fires:
        out += [kv('recorded', 'nothing yet')]
        if cfg.shadowing:
            out += ['', *wrap('Shadow mode is on and has not found a refusable '
                              'call yet. That is the expected state until a '
                              'session re-reads something it already holds.')]
        else:
            out += ['', *wrap(f'The guard is at `{cfg.enforce}`. `adder auto on '
                              '--shadow` runs the whole refusal decision and '
                              'records it without refusing anything, so the '
                              'trade below is measured on this machine before '
                              'anything is denied.')]
        return out
    out += [kv('sessions', f'{sh.sessions:,}'),
            kv('would have refused', f'{sh.fires:,} calls'),
            kv('worth', money(sh.would_save) + '   no uptake assumption: a '
               'refused call does not happen'),
            kv('contradicted', f'{sh.contradicted:,} of them '
               f'({sh.contradiction_rate:.0%}), {sh.contradictions:,} times'),
            kv('realised, worst case', money(sh.realised) +
               '   every contradicted refusal written off whole')]
    if sh.targets:
        out += ['', *heading('Refusals the session went around')]
        out += table([(_clip(t, 60), f'{n:,}') for t, n in sh.targets],
                     headers=('target', 'asked again'), align='<>')
    out += ['']
    if not sh.measured:
        out += wrap(f'{sh.fires} findings is an anecdote, not a rate. Ten is the '
                    'floor this reports a contradiction rate against, for the '
                    'same reason `uptake` has one.')
    else:
        out += wrap('A contradicted refusal is the closest thing to evidence '
                    'that a refusal was wrong: the session asked for the same '
                    'target again, which under enforcement is the escape hatch '
                    'firing and costs a turn. If that rate is low here, '
                    '`adder auto on` is being asked to do something this '
                    'machine has already watched it do.')
    return out


def last_report(path: Path | None=None) -> list[str]:
    """Everything the guard did in the most recent session it recorded."""
    from adder.util.render import heading, kv, money, table, wrap
    session, rows = last_session(path)
    out = heading('Last session — what it actually did', rule='=')
    if not rows:
        out += [kv('recorded', 'nothing yet')]
        return out
    out += [kv('session', session[:12] or '—'), kv('entries', f'{len(rows):,}')]
    out += ['', *table([(str(r.get('action') or '?'), str(r.get('tool') or '?'),
                         str(r.get('kind') or ''), _clip(_identity(r), 40),
                         f"{int(_num(r.get('tokens'))):,}",
                         money(_num(r.get('saving'))))
                        for r in rows[-40:]],
                       headers=('action', 'tool', 'kind', 'what', 'tok', 'worth'),
                       align='<<<<>>')]
    denied = sum(1 for r in rows if r.get('action') in ('deny', 'shadow'))
    if denied:
        out += ['', *wrap('`deny` is a call that did not happen; `shadow` is one '
                          'that did, recorded as if it had not. Either way the '
                          'guard offers a target at most once a session — if '
                          'the same call is made again it goes through.')]
    return out


def report(root=None, *, learn: bool=False, explain: str | None=None, replay_it: bool=False, cwd=None) -> str:
    """Everything about the guard that is otherwise invisible."""
    from adder.core.shapes import DEFAULT_ROOT, PRIOR, SizeModel, model_path, refresh
    from adder.util.render import heading, kv, money, table, tokens, wrap
    cfg = Settings.resolve(cwd=cwd)
    root = root or DEFAULT_ROOT
    if learn:
        # Both learned things, in the one command anybody actually runs. The
        # uptake measurement was previously derivable and never derived, which
        # is why it sat unused: nothing wrote the cache the gate would read.
        refresh_uptake(root)
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
    near = concurrent_sessions(cfg.state_path)
    if near > 1:
        out += ['', *heading('Concurrency', rule='=')]
        out += [kv('sessions active', f'{near} in the last '
                   f'{CONCURRENT_WINDOW_S / 60:.0f} minutes')]
        out += ['', *wrap('The duplicate rule asks whether a file is the same '
                          'bytes it already read, and it asks by mtime. In a '
                          'tree several agents are working in, the mtime moves '
                          'for reasons that have nothing to do with this '
                          'session, so the rule correctly declines to call the '
                          'read a duplicate and the saving below is a floor '
                          'rather than a total. It fails towards saying '
                          'nothing, never towards a wrong refusal.')]
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
            out += [kv('tool calls replayed', f'{r.calls:,} across {r.sessions:,} sessions'), kv('cost a parse', f'{r.lookup_rate:.2%} of calls'), kv('times it would speak', f'{r.fires:,} ({r.fire_rate:.2%} of calls)'), kv('by kind', ', '.join((f'{k} {n:,}' for k, n in sorted(r.by_kind.items()))) or '—'), kv(f'saving at {cfg.advice_taken:.0%} uptake', money(r.saving)), kv('cost of saying it', money(r.overhead)), kv('net', money(r.net))]
            if r.refusals:
                out += [kv('of which refused', f'{r.refusals:,} calls that would not have happened'), kv('prevented outright', money(r.prevented) + '   no uptake assumption'), kv('assumed share', f'{r.assumed_share:.0%} of the total rests on the uptake prior')]
            if r.biggest:
                out += ['', *heading('Largest findings')]
                out += table([(money(v), _clip(t)) for v, t in r.biggest], headers=('worth', 'finding'), align='><')
            out += ['']
            out += wrap('An upper bound, not a saving to bank: the horizon is the one the guard would have projected rather than the turns that really remained, the saving assumes the advice is acted on, and a call it talked someone out of would have changed everything after it.')
    # Only when there is something to say. The section exists for two
    # readers -- somebody running shadow, and somebody deciding whether to --
    # and for everybody else it is a heading over a blank. The one-line pointer
    # in `Settings in effect` covers the second reader instead.
    if shadow(cfg.state_path).fires or cfg.shadowing:
        out += ['', *shadow_report(cfg.state_path)]
    out += ['', *heading('Uptake — is the assumption holding?', rule='=')]
    u = uptake(root)
    out += [kv('measured', u.describe())]
    if u.measured:
        # Telling the reader to go and lower a setting was the right advice
        # while nothing consumed the measurement. It is now wrong: the gate
        # picks the measured rate up on its own, and the only reason it would
        # not is that somebody set the value explicitly. So the line reports
        # which of the two is in force rather than issuing an instruction that
        # has already been carried out.
        applied = abs(cfg.advice_taken - max(UPTAKE_FLOOR, min(1.0, u.rate))) < 1e-09
        out += [kv('in force', f'{cfg.advice_taken:.0%}'),
                kv('source', 'the measurement above' if applied
                   else 'guard_advice_taken, set explicitly — the measurement is reported, not applied')]
        if applied and u.rate < UPTAKE_FLOOR:
            out += [kv('note', f'the measured {u.rate:.0%} is below the {UPTAKE_FLOOR:.0%} floor. The floor is not caution: at an uptake near zero no advice clears its own cost, the guard falls silent, and a silent guard records no fires for anything to re-measure')]
    else:
        out += [kv('in force', f'{cfg.advice_taken:.0%} (guard_advice_taken)'), kv('note', 'an assumption until there are 10 findings to judge')]
    out += ['', *heading('Settings in effect', rule='=')]
    out += [kv('floor', f'{cfg.min_tokens:,} tok predicted'), kv('worth interrupting', money(cfg.min_cost) + ' of carry'), kv('mode', 'ask for confirmation' if cfg.block else 'advise only'), kv('enforce', cfg.enforce), kv('uptake', _uptake_line(cfg)), kv('ceiling', f'{cfg.max_fires} fires per session'), kv('state', str(cfg.state_path))]
    if not cfg.enforcing and not cfg.shadowing:
        out += ['', *wrap('Every dollar above is multiplied by '
                          f'{cfg.advice_taken:.0%} — an assumption about whether '
                          'a sentence changes what a model does next, and the '
                          'weakest number in this project. `adder auto on '
                          '--shadow` replaces it with a measurement: it runs the '
                          'whole refusal decision, records what it would have '
                          'refused and how often the session went round it, and '
                          'refuses nothing. `adder guard --shadow` reads it back.')]
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
    ap.add_argument('--shadow', action='store_true', help='what shadow mode would have refused, and how often the session went round it. Read-only: `adder auto on --shadow` is what turns it on')
    ap.add_argument('--last', action='store_true', help='every refusal and finding from the most recent recorded session')
    ap.add_argument('--floors', action='store_true', help='per-tool result-size distributions, what the shared floor prices, and the per-tool floor your own transcripts imply')
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
    if a.shadow or a.last or a.floors:
        from adder.core.shapes import load_model as load_sizes
        cfg = Settings.resolve()
        out: list[str] = []
        if a.shadow:
            out += shadow_report(cfg.state_path)
        if a.last:
            out += (['', ''] if out else []) + last_report()
        if a.floors:
            out += (['', ''] if out else []) + floors_report()
        if a.json:
            sh = shadow(cfg.state_path)
            session, rows = last_session()
            blob: dict = {}
            if a.shadow:
                blob['shadow'] = {'sessions': sh.sessions, 'fires': sh.fires, 'would_save': round(sh.would_save, 4), 'contradicted': sh.contradicted, 'contradictions': sh.contradictions, 'contradiction_rate': round(sh.contradiction_rate, 4), 'realised': round(sh.realised, 4), 'measured': sh.measured, 'targets': [{'target': t, 'asked_again': n} for t, n in sh.targets]}
            if a.floors:
                blob['floors'] = {t: {'p50': v[0], 'p90': v[1], 'calls': v[2], 'floor': cfg.min_tokens_for(t), 'ceiling': cfg.max_fires_for(t)} for t, v in sorted(load_sizes().tools.items()) if t in GUARDED}
            if a.last:
                blob['last'] = {'session': session, 'entries': [{'ts': _num(r.get('ts')), 'action': r.get('action'), 'tool': r.get('tool'), 'kind': r.get('kind'), 'what': _identity(r), 'tokens': int(_num(r.get('tokens'))), 'saving': round(_num(r.get('saving')), 4)} for r in rows]}
            print(json.dumps(blob, indent=2))
            return 0
        print('\n'.join(out))
        return 0
    if a.json:
        from adder.core.shapes import SizeModel, load_model
        sizes = SizeModel.learn(a.root) if a.learn else load_model()
        cfg = Settings.resolve()
        out = {'installed': [str(w) for w in installed_in()], 'model': {'calls': sizes.calls, 'shapes': len(sizes.shapes), 'age_s': None if sizes.age_s == float('inf') else sizes.age_s}, 'ledger': ledger(cfg.state_path), 'uptake': {'fires': uptake(a.root).fires, 'rate': round(uptake(a.root).rate, 4), 'measured': uptake(a.root).measured}, 'settings': {'min_tokens': cfg.min_tokens, 'min_cost': cfg.min_cost, 'block': cfg.block, 'advice_taken': cfg.advice_taken, 'max_fires': cfg.max_fires}}
        if a.replay:
            r = replay(a.root, cfg=cfg, sizes=sizes)
            out['replay'] = {'calls': r.calls, 'sessions': r.sessions, 'priced': r.priced, 'fires': r.fires, 'by_kind': r.by_kind, 'fire_rate': round(r.fire_rate, 5), 'lookup_rate': round(r.lookup_rate, 5), 'saving': round(r.saving, 4), 'overhead': round(r.overhead, 4), 'net': round(r.net, 4), 'refusals': r.refusals, 'prevented': round(r.prevented, 4), 'assumed_share': round(r.assumed_share, 4)}
        print(json.dumps(out, indent=2))
        return 0
    print(report(a.root, learn=a.learn, explain=a.explain, replay_it=a.replay))
    return 0
