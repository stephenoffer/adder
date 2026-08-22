"""`adder auto`: turn the thing on, in one command, and say what it now does.

Everything else in this repository is a report you have to run. The parts that
act without being asked -- the PreToolUse guard, the PreCompact learner, the
prompt advisor -- were shipped as a JSON block you were expected to merge into
`settings.json` yourself, plus a `--learn` you were expected to remember. The
tool's own benchmark measured the result: **1.6x installed, 6.4x if you follow
what the reports say**, and it said in as many words that nothing enforced the
gap. A saving that depends on the user pasting configuration is a saving most
users do not get.

So this writes the configuration, and it is the only module here that writes
anything outside a path the user named. Three rules follow from that, and each
one is a test:

* **It shows the change before it makes it.** `plan()` is pure and returns the
  exact before/after; `apply()` is the only function that touches disk.
* **It backs up what it edits**, once, next to the original.
* **It is reversible.** `adder auto off` removes precisely what `on` added and
  leaves anything it did not write alone. An activation you cannot undo is one
  people will not try.

What it deliberately does not do is start a background process. Nothing here
runs on a timer or holds a socket: "in the background" means the hooks the
harness already calls, on events it already fires, costing nothing when there
is nothing to say. A daemon would be a second thing to trust, and the whole
argument of this tool is that the cheapest work is the work that does not
happen.
"""

from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from adder.core.settings import PROJECT_FILE, USER_FILE, project_file
from adder.decide import guard

# The three hooks, in the order they matter. `event` is the Claude Code hook
# event; `matcher` is only meaningful for PreToolUse.
#
# The advisor is included because the largest measured lever -- splitting a
# session -- is the one no hook can take on its own. It is the only lever here
# that still ends in a human deciding, so the least the tool can do is put the
# price in front of them at the moment it applies.
HOOKS: tuple[dict, ...] = (
    {'script': 'pretooluse_read_guard.py', 'event': 'PreToolUse',
     'matcher': '|'.join(guard.OBSERVED),
     'does': 'prices, and can refuse, a call before its result lands in context'},
    {'script': 'precompact_learn.py', 'event': 'PreCompact', 'matcher': '',
     'does': 'forgets what compaction drops, and re-learns result sizes'},
    {'script': 'session_cost_advisor.py', 'event': 'UserPromptSubmit', 'matcher': '',
     'does': 'prices compaction against a restart, once a session is expensive'},
)
BACKUP_SUFFIX = '.adder.bak'
# The agent definitions, which are the other half of "installed and changed
# nothing". `adder bench` prices the hooks and these together and the pair is
# what its headline multiple means -- on the author's history the hooks alone
# are 2.5x and the tier files take it to 3.1x, so activation that installed one
# and not the other would be quoting a number it does not deliver.
#
# `Explore` is first because it is the one that needs no routing decision from
# anybody: it overrides a built-in that already runs on every exploration, and
# exploration is read-heavy work in a throwaway context, which is the one place
# a cheap model costs nothing in cache.
AGENTS: tuple[str, ...] = ('Explore.md', 'route-t0.md', 'route-t1.md', 'route-t2.md')

# The thresholds an enforcing guard runs at, and why they are not the advisory
# ones. All three were swept with `guard.replay` over 34,144 recorded tool
# calls on the author's machine; `adder auto on --tune` re-derives them from
# yours, which is the only reason it is honest to ship them as constants.
#
#   floor   fires    gate   refusals   prevented   overhead    net   parses
#   2,000      15   $0.25        278        $166      $3.17   $182     1.4%
#     800      60   $0.25        757        $283      $9.80   $303    15.7%
#     800      60   $0.10      2,242        $490     $25.40   $487       8%
#     800     200   $0.10      2,421        $517     $27.45   $513       8%
#     800   1,000   $0.10      2,448        $523     $27.79   $519       8%
#     300     200   $0.10      4,590        $677     $51.54   $651      39%
#
# Three things that table settles. The $0.25 gate is not a threshold on this
# lever at all -- it exists to stop the guard *interrupting* over small change,
# and a refusal is not an interruption, so under enforcement it comes down to
# $0.10 and finds $200 more. The 15-fire ceiling was sized for a guard that
# talks; a guard that redirects can afford 200, which is worth $26 and costs no
# latency. Past 200 the curve is flat.
#
# The floor is the one real trade, and it is a trade of latency for money
# rather than a free choice: 300 finds $138 more and takes the share of tool
# calls that stop to parse a transcript from 8% to 39%. 800 is shipped because
# a hook people uninstall saves nothing, and because that ratio is a property
# of one workload -- `--tune` is how somebody whose reads are smaller finds out
# that 300 is right for them.
ENFORCING_THRESHOLDS: dict[str, float | int] = {
    'guard_min_tokens': 800,
    'guard_min_cost': 0.10,
    'guard_max_fires': 200,
}
# What `--tune` sweeps. Deliberately small: each point is a full replay of
# every recorded tool call, and the shape of the answer is a plateau rather
# than a peak, so four well-chosen points find it. The last one is the
# latency trade, included so the report can show what it would buy.
TUNE_GRID: tuple[tuple[int, float], ...] = (
    (2000, 0.25), (800, 0.25), (800, 0.10), (300, 0.10))
# How much better a point has to be before it is worth tripling the share of
# tool calls that stop to parse a transcript. Within this margin the sweep
# prefers the quieter setting, because the difference is inside the precision
# of the size model that produced it.
MATERIAL_MARGIN = 0.05


def hooks_dir(repo: Path | None = None) -> Path:
    """Where the hook scripts live -- inside the package, wherever it is installed."""
    return guard.hook_path(repo).parent


def agents_dir(repo: Path | None = None) -> Path:
    """Where the agent definitions live. Beside the hooks, and for the same reason.

    Both used to be read out of `.claude/` in the checkout root, which meant
    activation from a `pip install` found neither: it wrote three hooks pointing
    at absent files and copied zero agents, while reporting a plan that looked
    complete. `agent_plan` skips a source that is not there, so the missing half
    of the headline multiple was silent as well as absent.
    """
    return hooks_dir(repo).parent / 'agents'


def agent_plan(target: Path, *, repo: Path | None = None) -> tuple[list[str], list[str]]:
    """Which agent files would be written, and which are left alone.

    Never overwrites. A user may have their own `Explore` they rely on, and
    silently replacing it during a command whose subject is cost would be the
    worst kind of surprise -- it changes what a built-in does, in a file they
    did not know we knew about. An existing file is reported and skipped, and
    the report says so rather than pretending the install was complete.
    """
    write, skip = [], []
    for name in AGENTS:
        src = agents_dir(repo) / name
        if not src.is_file():
            continue
        dst = target / name
        if not dst.exists():
            write.append(name)
        elif dst.read_text(encoding='utf-8', errors='replace') != \
                src.read_text(encoding='utf-8', errors='replace'):
            skip.append(name)
    return write, skip


def _command(script: str, repo: Path | None = None) -> str:
    """The shell command that runs one hook. See `guard.interpreter`."""
    return f'{guard.interpreter()} {shlex.quote(str(hooks_dir(repo) / script))}'


def _is_ours(command: str) -> bool:
    """Does this hook command point at one of our scripts?

    Matched on the script basename rather than the whole path, so a checkout
    that has moved is still recognised as installed -- and, more importantly,
    so `off` removes an entry `on` wrote from a different directory instead of
    leaving a stale hook that fails on every tool call.
    """
    return any(h['script'] in str(command) for h in HOOKS)


def _read_json(path: Path) -> dict:
    """A settings file as a dict. Anything unreadable is treated as absent.

    Not `{}` on a *parse* failure -- see `merge`. Returning empty for a file
    that exists but is malformed would mean overwriting somebody's settings
    with only ours, which is the worst thing this module could do.
    """
    try:
        blob = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    return blob if isinstance(blob, dict) else {}


def _parses(path: Path) -> bool:
    """Is this file safe to edit -- absent, or present and valid JSON?"""
    if not path.exists():
        return True
    try:
        return isinstance(json.loads(path.read_text(encoding='utf-8')), dict)
    except (OSError, ValueError):
        return False


def merge(settings: dict, *, repo: Path | None = None) -> tuple[dict, list[str]]:
    """Settings with our hooks added. Returns the new dict and what changed.

    Pure, and it never reorders or drops anything it did not add: the file
    belongs to the user and may hold hooks from three other tools. Adding a
    hook that is already declared is a no-op, so running `on` twice is the same
    as running it once.
    """
    out = json.loads(json.dumps(settings)) if settings else {}
    changed: list[str] = []
    hooks = out.setdefault('hooks', {})
    if not isinstance(hooks, dict):
        return settings, []
    for h in HOOKS:
        event = str(h['event'])
        groups = hooks.setdefault(event, [])
        if not isinstance(groups, list):
            continue
        if any(_is_ours(entry.get('command', ''))
               for g in groups if isinstance(g, dict)
               for entry in (g.get('hooks') or []) if isinstance(entry, dict)):
            continue
        group: dict = {'hooks': [{'type': 'command',
                                  'command': _command(str(h['script']), repo)}]}
        if h['matcher']:
            group = {'matcher': h['matcher'], **group}
        groups.append(group)
        changed.append(f"{event}: {h['script']}")
    return out, changed


def unmerge(settings: dict) -> tuple[dict, list[str]]:
    """Settings with our hooks removed, and nothing else touched."""
    out = json.loads(json.dumps(settings)) if settings else {}
    changed: list[str] = []
    hooks = out.get('hooks')
    if not isinstance(hooks, dict):
        return out, []
    for event, groups in list(hooks.items()):
        if not isinstance(groups, list):
            continue
        kept_groups = []
        for g in groups:
            if not isinstance(g, dict):
                kept_groups.append(g)
                continue
            entries = g.get('hooks')
            if not isinstance(entries, list):
                kept_groups.append(g)
                continue
            kept = [e for e in entries
                    if not (isinstance(e, dict) and _is_ours(e.get('command', '')))]
            for e in entries:
                if isinstance(e, dict) and _is_ours(e.get('command', '')):
                    changed.append(f"{event}: {Path(str(e.get('command'))).name}")
            if not kept:
                continue                      # the group held nothing else
            kept_groups.append({**g, 'hooks': kept})
        if kept_groups:
            hooks[event] = kept_groups
        else:
            hooks.pop(event)                  # do not leave an empty event behind
    if not hooks:
        out.pop('hooks', None)
    return out, changed


@dataclass
class Tuning:
    """One point of the threshold sweep, priced against recorded calls."""
    min_tokens: int
    min_cost: float
    refusals: int
    prevented: float
    promised: float
    overhead: float
    parse_rate: float

    @property
    def net(self) -> float:
        return self.prevented + self.promised - self.overhead


def tune(root=None, *, level: str = 'full', grid=TUNE_GRID) -> list[Tuning]:
    """Re-derive the enforcing thresholds from this machine's own transcripts.

    The shipped constants are a measurement of one workload. Somebody whose
    sessions are short, or who reads few large files, has a different answer --
    and the guard's whole argument against the version that preceded it was
    that it assumed a number instead of measuring one. This is that argument
    applied to its own configuration.

    Expensive on purpose: every point replays every recorded tool call, so it
    is a flag on activation and never something a report does behind your back.
    """
    out = []
    for min_tokens, min_cost in grid:
        cfg = guard.Settings.resolve()
        cfg = type(cfg)(min_tokens=min_tokens, min_cost=min_cost,
                        hard_tokens=cfg.hard_tokens, block=cfg.block,
                        advice_taken=cfg.advice_taken,
                        max_fires=int(ENFORCING_THRESHOLDS['guard_max_fires']),
                        state_path=cfg.state_path, enforce=level)
        r = guard.replay(root, cfg=cfg)
        out.append(Tuning(min_tokens=min_tokens, min_cost=min_cost,
                          refusals=r.refusals, prevented=r.prevented,
                          promised=r.saving, overhead=r.overhead,
                          parse_rate=r.lookup_rate))
    return out


def best(points: list[Tuning]) -> Tuning | None:
    """The highest net, unless a quieter setting is within a rounding error of it.

    Not cosmetic. The lowest floor in the grid always finds at least as much
    money as the one above it and always parses more transcripts to do it, so
    a pure argmax over net would pick the noisiest setting every time, however
    small the margin. `MATERIAL_MARGIN` is the answer to "how much more money
    is worth tripling the latency", and inside it the difference is smaller
    than the precision of the size model that produced it anyway.

    Latency is not dollars. It matters because a hook people uninstall saves
    nothing, which is the same reason the guard prices its own sentences.
    """
    if not points:
        return None
    top = max(points, key=lambda t: t.net)
    if top.net <= 0:
        return top
    contenders = [t for t in points if t.net >= top.net * (1 - MATERIAL_MARGIN)]
    return sorted(contenders, key=lambda t: (t.parse_rate, -t.net))[0]


@dataclass
class Plan:
    """What activation would change, computed before anything is written."""
    settings_path: Path
    config_path: Path
    hook_changes: list[str] = field(default_factory=list)
    agents_path: Path = Path()
    agent_writes: list[str] = field(default_factory=list)
    agent_skips: list[str] = field(default_factory=list)
    repo: Path | None = None
    level: str = 'certain'
    was_level: str = 'off'
    settings_after: dict = field(default_factory=dict)
    config_after: dict = field(default_factory=dict)
    config_before: dict = field(default_factory=dict)
    blocked: str = ''
    warnings: list[str] = field(default_factory=list)

    @property
    def config_changes(self) -> list[str]:
        """Every setting whose value this plan would change, as `k: old -> new`."""
        return [f'{k}: {self.config_before.get(k, "unset")} -> {v}'
                for k, v in sorted(self.config_after.items())
                if self.config_before.get(k) != v]

    @property
    def empty(self) -> bool:
        return not self.hook_changes and not self.config_changes \
            and not self.agent_writes


def _override_warnings(env: dict[str, str] | None = None) -> list[str]:
    """Environment that would quietly defeat what we are about to install.

    `CLAUDE_CODE_SUBAGENT_MODEL` outranks both agent frontmatter and the
    per-invocation model, so with it set the tier agents route nothing and the
    delegation half of the saving silently does not happen. Worth a sentence at
    activation, which is the only moment anyone is looking.
    """
    env = os.environ if env is None else env
    out = []
    for name in ('CLAUDE_CODE_SUBAGENT_MODEL', 'ANTHROPIC_MODEL'):
        if env.get(name):
            out.append(f'{name} is set to {env[name]!r}; it outranks agent '
                       f'frontmatter, so delegated steps will not be routed.')
    if str(env.get('ADDER_GUARD_ENFORCE', '')).strip().lower() in ('off', '0', 'false'):
        out.append('ADDER_GUARD_ENFORCE is set to off in the environment, which '
                   'beats the config file this writes.')
    return out


def plan(*, cwd: Path | str | None = None, level: str = 'certain', user: bool = False,
         repo: Path | None = None, env: dict[str, str] | None = None,
         thresholds: dict | None = None) -> Plan:
    """What `on` would do. Pure with respect to disk: it reads, never writes."""
    base = Path(cwd or os.getcwd()).resolve()
    settings_path = ((Path.home() / '.claude' / 'settings.json') if user
                     else base / '.claude' / 'settings.json')
    config_path = USER_FILE if user else (project_file(base) or base / PROJECT_FILE)
    current = _read_json(settings_path)
    after, changes = merge(current, repo=repo)
    config = _read_json(config_path)
    was = str(config.get('guard_enforce', 'off'))
    # `full` moves the thresholds as well as the level. `certain` deliberately
    # does not: it refuses only the calls that admit nothing new, and how large
    # a call has to be before the guard *speaks* is a separate question that
    # the user's existing configuration already answers.
    tuned = dict(config)
    if level == 'full':
        tuned.update(ENFORCING_THRESHOLDS if thresholds is None else thresholds)
    agents_path = settings_path.parent / 'agents'
    writes, skips = agent_plan(agents_path, repo=repo)
    p = Plan(settings_path=settings_path, config_path=config_path, hook_changes=changes,
             agents_path=agents_path, agent_writes=writes, agent_skips=skips, repo=repo,
             level=level, was_level=was, settings_after=after,
             config_after={**tuned, 'guard_enforce': level}, config_before=config,
             warnings=_override_warnings(env))
    if not _parses(settings_path):
        p.blocked = f'{settings_path} exists but is not valid JSON; fix it first'
    elif not _parses(config_path):
        p.blocked = f'{config_path} exists but is not valid JSON; fix it first'
    return p


def plan_off(*, cwd: Path | str | None = None, user: bool = False) -> Plan:
    """What `off` would do."""
    base = Path(cwd or os.getcwd()).resolve()
    settings_path = ((Path.home() / '.claude' / 'settings.json') if user
                     else base / '.claude' / 'settings.json')
    config_path = USER_FILE if user else (project_file(base) or base / PROJECT_FILE)
    after, changes = unmerge(_read_json(settings_path))
    config = _read_json(config_path)
    # Only the level is reset. The thresholds `--full` wrote are left where
    # they are: they are measurements of this workload, they are meaningful to
    # an advisory guard too, and silently reverting a number the user may since
    # have tuned by hand is not what "off" should mean.
    # The agent files are not removed. They are ordinary configuration a user
    # may have come to rely on, they cost nothing when the hooks are gone, and
    # deleting a file somebody may have edited is not something `off` should
    # do quietly. The report says they are staying.
    agents_path = settings_path.parent / 'agents'
    return Plan(settings_path=settings_path, config_path=config_path,
                hook_changes=changes, agents_path=agents_path, level='off',
                was_level=str(config.get('guard_enforce', 'off')),
                settings_after=after, config_after={**config, 'guard_enforce': 'off'},
                config_before=config)


def _write(path: Path, blob: dict) -> None:
    """Write JSON, keeping one backup of whatever was there before.

    Atomic, and unique per writer, for the same reason every other write in
    this project is: several sessions share one machine. The backup is written
    once and never overwritten -- the point of it is the state before adder
    touched anything, and a second activation must not destroy that.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if path.exists() and not backup.exists():
        backup.write_text(path.read_text(encoding='utf-8'), encoding='utf-8')
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    try:
        tmp.write_text(json.dumps(blob, indent=2) + '\n', encoding='utf-8')
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def apply(p: Plan) -> list[str]:
    """Carry out a plan. The only function in this module that writes."""
    if p.blocked:
        raise ValueError(p.blocked)
    done: list[str] = []
    if p.hook_changes:
        _write(p.settings_path, p.settings_after)
        done.append(f'{p.settings_path}: {len(p.hook_changes)} hook(s)')
    if p.config_changes:
        _write(p.config_path, p.config_after)
        done.append(f'{p.config_path}: {", ".join(p.config_changes)}')
    if p.agent_writes:
        p.agents_path.mkdir(parents=True, exist_ok=True)
        for name in p.agent_writes:
            src = agents_dir(p.repo) / name
            (p.agents_path / name).write_text(
                src.read_text(encoding='utf-8'), encoding='utf-8')
        done.append(f'{p.agents_path}: {len(p.agent_writes)} agent(s)')
    return done


@dataclass
class Status:
    """What is actually on, and what it has been worth.

    `prevented` is the only figure here with no assumption behind it: it is the
    carry cost of calls that did not happen because the guard refused them.
    `promised` is what the advisory half argued for, and it is reported
    separately and discounted, because nothing in a transcript says whether a
    sentence changed what a model did next.
    """
    installed: list[Path]
    level: str
    hooks_present: list[str]
    hooks_missing: list[str]
    sessions: int
    fires: int
    prevented: float
    promised: float
    overhead: float
    uptake: float
    measured_uptake: bool
    model_calls: int
    model_age_s: float

    @property
    def realised(self) -> float:
        """Everything the guard is credited with, refusals at par and advice
        discounted by whatever uptake is currently believed."""
        return self.prevented + self.promised * self.uptake

    @property
    def ratio(self) -> float:
        """Returned per dollar of context the guard spent to return it."""
        return self.realised / self.overhead if self.overhead > 0 else 0.0

    @property
    def active(self) -> bool:
        return bool(self.installed) and not self.hooks_missing


def status(*, cwd: Path | str | None = None) -> Status:
    """What is installed and what it has done. Read-only; never raises."""
    cfg = guard.Settings.resolve(cwd=cwd)
    led = guard.ledger(cfg.state_path)
    present, missing = [], []
    declared = ''
    for path in guard.settings_files(cwd):
        try:
            declared += json.dumps(_read_json(path))
        except (TypeError, ValueError):
            continue
    for h in HOOKS:
        (present if str(h['script']) in declared else missing).append(str(h['script']))
    try:
        up = guard.uptake()
        rate, measured = up.rate, up.measured
    except Exception:
        rate, measured = cfg.advice_taken, False
    try:
        from adder.core.shapes import load_model
        sizes = load_model()
        calls, age = sizes.calls, sizes.age_s
    except Exception:
        calls, age = 0, float('inf')
    return Status(installed=guard.installed_in(cwd), level=cfg.enforce,
                  hooks_present=present, hooks_missing=missing,
                  sessions=int(led.get('sessions') or 0),
                  fires=int(led.get('fires') or 0),
                  prevented=float(led.get('prevented') or 0.0),
                  promised=float(led.get('saving') or 0.0),
                  overhead=float(led.get('overhead') or 0.0),
                  uptake=rate if measured else cfg.advice_taken,
                  measured_uptake=measured, model_calls=calls, model_age_s=age)


def _render_plan(p: Plan, *, off: bool = False) -> list[str]:
    from adder.util.render import kv
    verb = 'remove' if off else 'add'
    out = ['', f'  This will {verb}:', '']
    if p.hook_changes:
        for c in p.hook_changes:
            out.append(f'    {verb:<6} {c}')
        out.append(f'    in     {p.settings_path}')
    else:
        out.append(f'    (hooks already {"absent" if off else "installed"} in '
                   f'{p.settings_path})')
    out.append('')
    if p.config_changes:
        for c in p.config_changes:
            out.append(f'    set    {c}')
        out.append(f'    in     {p.config_path}')
    else:
        out.append(f'    (guard_enforce is already {p.level})')
    if not off and (p.agent_writes or p.agent_skips):
        out.append('')
        for name in p.agent_writes:
            out.append(f'    copy   {name}')
        for name in p.agent_skips:
            out.append(f'    keep   {name}  (yours differs — left alone)')
        out.append(f'    in     {p.agents_path}')
    if off:
        out += ['', '    (agent files are left in place; they cost nothing '
                'without the hooks)']
    if not off:
        out += ['', '  What each hook does:', '']
        for h in HOOKS:
            out.append(f"    {h['event']:<18}{h['does']}")
        out.append(f"    {'agents':<18}what a delegated step runs on — Explore on "
                   "Haiku, three tiers")
        out += ['', kv('refusals', 'certain: only calls that admit nothing new'
                       if p.level == 'certain'
                       else 'full: also a large read with a cheaper equal')]
    for w in p.warnings:
        out += ['', f'  ! {w}']
    return out


def render_status(s: Status) -> str:
    from adder.util.render import kv, money
    out = ['', '  adder auto', '']
    if not s.installed:
        out += [kv('status', 'OFF — nothing is running between your turns'), '',
                '  Every report here measures money already gone. The hooks are',
                '  the only part that runs while the decision is still',
                '  reversible, and they are not in any settings.json.', '',
                '  Run `adder auto on`.', '']
        return '\n'.join(out)
    out += [kv('status', f'ON — enforcing `{s.level}`' if s.level != 'off'
               else 'ADVISORY — installed, but refusing nothing'),
            kv('declared in', ', '.join(str(p) for p in s.installed))]
    if s.hooks_missing:
        out.append(kv('missing', ', '.join(s.hooks_missing)))
    out.append(kv('size model', f'{s.model_calls:,} calls learned'
                  if s.model_calls else 'the shipped prior — run `adder guard --learn`'))
    out += ['', '  What it has been worth', '']
    out += [kv('sessions seen', f'{s.sessions:,}'),
            kv('times it acted', f'{s.fires:,}'),
            kv('prevented', money(s.prevented) + '   calls that did not happen'),
            kv('argued for', money(s.promised) + f'   x {s.uptake:.0%} '
               f'{"measured" if s.measured_uptake else "assumed"} uptake'),
            kv('cost of saying it', money(s.overhead))]
    if s.overhead > 0:
        out += ['', kv('returned per $1 spent', f'{s.ratio:,.1f}x')]
    elif s.fires == 0:
        out += ['', '  It has not had to act yet. That is the expected state in a',
                '  short session: nothing here fires unless it is worth more',
                '  than the sentence it costs to say it.']
    out.append('')
    return '\n'.join(out)


def _confirm(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in ('y', 'yes')
    except (EOFError, KeyboardInterrupt):
        return False


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        prog='adder auto',
        description='Turn on the parts of adder that run without being asked.')
    ap.add_argument('action', nargs='?', default='status',
                    choices=('on', 'off', 'status'),
                    help='on: install the hooks; off: remove them; status: what is running')
    ap.add_argument('--full', action='store_true',
                    help='also refuse a large read that has a cheaper equal '
                         '(default: refuse only calls that admit nothing new)')
    ap.add_argument('--user', action='store_true',
                    help='write to ~/.claude rather than this project')
    ap.add_argument('--yes', '-y', action='store_true', help='do not ask')
    ap.add_argument('--dry-run', action='store_true', help='print the change and stop')
    ap.add_argument('--tune', action='store_true',
                    help='re-derive the enforcing thresholds from your own '
                         'transcripts instead of the shipped ones (slow: it '
                         'replays every recorded tool call, three times)')
    ap.add_argument('--json', action='store_true', help='machine-readable status')
    a = ap.parse_args(argv)

    if a.action == 'status':
        s = status()
        if a.json:
            print(json.dumps({'installed': [str(p) for p in s.installed],
                              'level': s.level, 'active': s.active,
                              'hooks_missing': s.hooks_missing,
                              'sessions': s.sessions, 'fires': s.fires,
                              'prevented': round(s.prevented, 4),
                              'promised': round(s.promised, 4),
                              'overhead': round(s.overhead, 4),
                              'uptake': round(s.uptake, 4),
                              'measured_uptake': s.measured_uptake,
                              'ratio': round(s.ratio, 4)}, indent=2))
        else:
            print(render_status(s))
        return 0

    off = a.action == 'off'
    thresholds = None
    if a.tune and not off:
        if not a.full:
            print('  --tune only has thresholds to tune at --full.', file=sys.stderr)
            return 2
        print('\n  Replaying your recorded tool calls at three settings. '
              'This takes a minute.\n')
        points = tune()
        from adder.util.render import money, table
        rows = [[f'{t.min_tokens:,}', f'${t.min_cost:.2f}', f'{t.refusals:,}',
                 money(t.prevented), money(t.overhead), money(t.net),
                 f'{t.parse_rate:.0%}'] for t in points]
        print('\n'.join(table(rows, ['floor', 'gate', 'refusals', 'prevented',
                                     'overhead', 'net', 'parses'],
                              align='>>>>>>>')))
        pick = best(points)
        if pick is None:
            print('\n  Nothing to learn from — falling back to the shipped '
                  'thresholds.')
        else:
            thresholds = {**ENFORCING_THRESHOLDS, 'guard_min_tokens': pick.min_tokens,
                          'guard_min_cost': pick.min_cost}
            print(f'\n  Best on your data: floor {pick.min_tokens:,} tok, gate '
                  f'${pick.min_cost:.2f}.')
    p = (plan_off() if off else
         plan(level='full' if a.full else 'certain', user=a.user,
              thresholds=thresholds))
    print('\n'.join(_render_plan(p, off=off)))
    if p.blocked:
        print(f'\n  ! {p.blocked}\n', file=sys.stderr)
        return 1
    if p.empty:
        print('\n  Nothing to do.\n')
        return 0
    if a.dry_run:
        print('\n  --dry-run: nothing written.\n')
        return 0
    if not a.yes and not _confirm('\n  Write these changes? [y/N] '):
        print('\n  Nothing written.\n')
        return 1
    try:
        done = apply(p)
    except (OSError, ValueError) as e:
        print(f'\n  ! {e}\n', file=sys.stderr)
        return 1
    print('')
    for d in done:
        print(f'  wrote  {d}')
    if not off:
        # Learned now rather than on the first tool call: the model is what
        # turns "this Bash call is probably big" from a guess into a
        # measurement, and the first session after activation is as entitled to
        # it as the tenth.
        try:
            from adder.core.shapes import refresh
            m = refresh(force=True)
            if m.calls:
                print(f'  learned result sizes from {m.calls:,} local tool calls')
            else:
                # A fresh machine has no transcripts to learn from, and "learned
                # from 0 calls" reads as a failure. It is not one: the shipped
                # prior is what the guard uses until there is history, and the
                # PreCompact hook re-learns without being asked.
                print('  no local history yet — using the shipped size prior, '
                      'and re-learning as you work')
        except Exception:
            print('  (could not learn result sizes; the shipped prior is in use)')
        print('\n  Active in every new session. `adder auto status` for what it '
              'has saved.\n')
    else:
        print('\n  Removed. Your original files are beside them, suffixed '
              f'`{BACKUP_SUFFIX}`.\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
