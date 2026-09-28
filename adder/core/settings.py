"""One registry for every setting, and one command that shows what is in effect.

Before this, eight modules each read their own environment variable at import
time and documented it in a comment. `ADDER_LOG`, `ADDER_LEDGER`, `ADDER_HOME`,
`ADDER_CATALOG`, `ADDER_TRACE_CACHE`, `ADDER_OFFLINE`, `ADDER_GUARD_BLOCK`,
`ADDER_WARN_SPEND` -- all real, all load-bearing, none discoverable without
grepping the source. A tool whose behaviour depends on invisible state is a tool
whose numbers cannot be reproduced by the person reading them.

So: every setting is declared once, here, with its type, default, and the
reason it exists. `adder config` prints the resolved value **and where it came
from**, which is the half that matters when a report disagrees with another
machine.

Precedence, lowest to highest
-----------------------------
    built-in default  <  ~/.claude/adder.json  <  ./.adder.json  <  ADDER_* env

The project file is searched upward from the working directory, the way git
finds `.git`, so a repo-level setting applies from any subdirectory of it.

Reading only
------------
Nothing here writes a config file. `adder config --init` prints a template to
stdout for the user to redirect where they want it; the tool does not decide
where a person's dotfiles live, and CLAUDE.md's "no mutation of user data" rule
does not have a carve-out for files we would find convenient.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adder.util.homepath import HomeDefault

_USER = HomeDefault(".claude", "adder.json")
USER_FILE = _USER.at_import          # repointed by tests; read it via `user_file()`


def user_file() -> Path:
    """`~/.claude/adder.json` under the home directory as of now."""
    return _USER.live(USER_FILE)
PROJECT_FILE = ".adder.json"


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _as_path(v: Any) -> str:
    return str(Path(str(v)).expanduser())


def _home(*parts: str) -> Callable[[], str]:
    """A default naming a path under the user's home, resolved when asked.

    `Path.home()` evaluated at import is a value no test can move: a
    `monkeypatch.setenv("HOME", ...)` runs after the module is imported and
    reaches nothing. That is not hypothetical isolation hygiene -- `auto on`
    learns a size model into `~/.claude`, so activating this tool made its own
    suite fail, and the only difference between a red run and a green one was a
    file the fixture believed it had redirected.
    """
    return lambda: str(Path.home().joinpath(*parts))


def _default_root() -> str:
    """Claude Code's transcript directory, unless only another agent has sessions.

    The default used to be `~/.claude/projects` unconditionally, so somebody
    who runs Codex and not Claude Code got "no priced turns found" from every
    command on first run, with their sessions sitting in `~/.codex/sessions`.
    Claude Code keeps first place whenever it has anything, so the answer for
    the people this tool was built on cannot change underneath them.
    """
    from adder.core.native import CLAUDE_CODE, discover, has_sessions

    found = discover()
    claude = found.get(CLAUDE_CODE)
    if claude is not None and has_sessions(claude):
        return str(claude)
    for name, path in found.items():
        if name != CLAUDE_CODE and has_sessions(path):
            return str(path)
    return str(Path.home() / ".claude" / "projects")


@dataclass(frozen=True)
class Setting:
    name: str
    # A value, or a zero-argument callable returning one. Callable for anything
    # under the user's home; see `_home`.
    default: Any
    cast: Callable[[Any], Any]
    help: str
    env: str = ""
    # Read only from the environment, never from a config file.
    #
    # Three settings are consumed by `util` and `pricing`, which sit BELOW
    # `core` and may not import this module (`tests/repo/test_structure.py`
    # enforces it). They read their environment variable directly, so a value
    # written into `.adder.json` reaches nothing -- and `adder config` was
    # reporting it as the effective value anyway, which is precisely the
    # invisible state this module's docstring says it exists to remove.
    #
    # Marking them means `resolve` does not claim a file layer set them, and
    # `adder config` can say why in one line instead of the reader discovering
    # it by watching a setting have no effect.
    env_only: bool = False
    # Never read from a project's `.adder.json`, only from the user's file and
    # the environment. The project file is found by walking up from the working
    # directory, so it is whatever the repository someone just cloned says it
    # is. A path here is a file the hook *writes* on every Bash call, and a
    # project that set `guard_state` to `~/.zshrc` replaced it with guard JSON;
    # `guard_narrow` turns a refusal into an approval that skips the permission
    # prompt. `trace_cache` is unpickled, so choosing it is choosing code to
    # run. None of these is a repository's to decide.
    user_only: bool = False

    @property
    def env_var(self) -> str:
        return self.env or f"ADDER_{self.name.upper()}"

    @property
    def initial(self) -> Any:
        """The default, computed now rather than when this module imported."""
        return self.default() if callable(self.default) else self.default


# Ordered for display: the ones a person changes first come first.
SETTINGS: tuple[Setting, ...] = (
    Setting("root", lambda: _default_root(), _as_path,
            "transcript directory every report reads. Unset, it is Claude Code's, "
            "or else whichever of Codex, Gemini CLI or OpenCode has sessions here"),
    Setting("model", "claude-opus-5", str,
            "model assumed for the session when a report cannot read one. Unset, "
            "it is the model your newest session under `root` ran on"),
    Setting("harness", "claude-code", str,
            "agent runtime driving the session: claude-code, codex, gemini-cli, "
            "aider, openhands, opencode, custom, or any. Harnesses that pin the "
            "main session to one vendor make other vendors subagent-only. Unset, "
            "it follows `root`: a Codex root means codex"),
    Setting("classify_terms", "", str,
            "vocabulary this project has that the shipped classifier does not, "
            "as `cheap=map_batches,placement group; hard=autoscaler,preemption`. "
            "A `cheap` term names one findable thing here, so a search for it is "
            "bounded and can route down; a `hard` term names open-ended work. "
            "Belongs in the project's `.adder.json`, because it is a fact about "
            "this repository and not about the machine. Empty means the shipped "
            "vocabulary alone, which on a domain codebase abstains on nearly "
            "everything and spends routing overhead to say `no change`"),
    Setting("ladder", "", str,
            "dispatch ladder as `T0=model,T1=model,...`, overriding the pinned "
            "Claude default. Empty keeps the built-in; the catalog reports "
            "drift but never repoints dispatch on its own"),
    Setting("ttl", "5m", str,
            "cache TTL assumed when a transcript does not say (5m or 1h)"),
    Setting("budget", 0.0, float,
            "monthly spend target in USD; 0 disables the burn-down"),
    Setting("handoff_tokens", 2_000, int,
            "tokens a session restart carries forward, for `prefix` and `plan`"),
    Setting("target", 10.0, float,
            "default percentage reduction `plan` solves for"),
    Setting("cache", True, _as_bool,
            "memoize transcript parsing by (mtime, size)"),
    Setting("color", "auto", str,
            "auto, always, or never; NO_COLOR always wins", env="ADDER_COLOR",
            env_only=True),
    Setting("offline", False, _as_bool,
            "refuse every network fetch, including `models refresh`",
            env="ADDER_OFFLINE", env_only=True),
    Setting("guard_min_cost", 0.25, float,
            "USD at which the PreToolUse read guard speaks up",
            env="ADDER_GUARD_MIN_COST"),
    Setting("guard_block", False, _as_bool,
            "escalate the read guard from advice to a confirmation prompt",
            env="ADDER_GUARD_BLOCK"),
    Setting("guard_min_tokens", 2_000, int,
            "predicted result size below which the guard does not price a call"),
    Setting("guard_hard", 60_000, int,
            "tokens above which the guard asks for confirmation, when blocking"),
    Setting("uptake_cache", _home(".claude", ".adder-uptake.json"), _as_path,
            "cached measurement of how often guard advice was followed; the "
            "hook reads it, `adder guard --learn` writes it", user_only=True),
    Setting("guard_advice_taken", 0.5, float,
            "share of guard advice that is acted on; discounts the saving "
            "before it is weighed against the cost of saying it. Only a "
            "fallback: once `adder guard --learn` has measured the rate on this "
            "machine, the measurement is used instead -- unless this is set "
            "explicitly, in which case what you set wins"),
    Setting("guard_max_fires", 15, int,
            "most times the guard may speak in one session"),
    Setting("guard_min_tokens_by_tool", "", str,
            "per-tool overrides for guard_min_tokens, as `Bash=800,Read=6000`. "
            "One floor cannot serve two distributions: on the machine this was "
            "written for, Bash returns a p90 of 1.2K over 2,490 calls and Read "
            "5.9K over 58, so the shipped 2,000 is almost never reached by one "
            "and routinely by the other. `adder guard --floors` prints the "
            "distributions and the floor each implies. Empty keeps the global "
            "value for every tool, which is the previous behaviour exactly"),
    Setting("guard_max_fires_by_tool", "", str,
            "per-tool overrides for guard_max_fires, as `Bash=8,Read=8`. The "
            "global ceiling is shared, so the tool called two thousand times a "
            "session can spend all of it before the tool called fifty times "
            "says anything. Both ceilings apply; neither raises the other"),
    Setting("guard_enforce", "off", str,
            "how far the guard may go: off (advise only), shadow (compute the "
            "refusals, record them, refuse nothing), certain (refuse the calls "
            "that admit nothing new), full (also refuse a large read that has a "
            "cheaper equal). `shadow` is where to start: it turns the assumed "
            "uptake term into a measurement on this machine before anything is "
            "denied",
            env="ADDER_GUARD_ENFORCE"),
    Setting("guard_state", _home(".claude", ".adder-guard.json"), _as_path,
            "per-session guard memory: files read, shapes already advised",
            user_only=True),
    Setting("guard_narrow", False, _as_bool,
            "where the guard would refuse a large Read or Grep, substitute the "
            "bounded call instead of demanding it -- saves the turn spent "
            "re-issuing, but a substitution travels with an approval and can "
            "suppress a permission prompt, so it is off until you say otherwise",
            env="ADDER_GUARD_NARROW", user_only=True),
    Setting("guard_route", True, _as_bool,
            "on a delegated step, also name the cheapest tier that clears the "
            "task -- advice only, and never a refusal",
            env="ADDER_GUARD_ROUTE"),
    Setting("size_model", _home(".claude", ".adder-sizes.json"), _as_path,
            "learned result-size quantiles the guard predicts from",
            user_only=True),
    Setting("size_max_age", 86_400.0, float,
            "seconds before the learned size model is re-derived"),
    Setting("warn_spend", 15.0, float,
            "session spend in USD at which the prompt hook warns",
            env="ADDER_WARN_SPEND"),
    Setting("warn_context", 400_000, int,
            "context size in tokens at which the prompt hook warns",
            env="ADDER_WARN_CONTEXT"),
    Setting("catalog_max_age_days", 21.0, float,
            "age past which the model catalog is reported as stale"),
    Setting("log", _home(".claude", "adder-outcomes.jsonl"), _as_path,
            "dispatch outcome log that calibrates p_fail", env="ADDER_LOG",
            user_only=True),
    Setting("ledger", _home(".claude", "adder-ledger.jsonl"), _as_path,
            "ledger of recommendations made and verified", env="ADDER_LEDGER",
            user_only=True),
    Setting("home", _home(".claude"), _as_path,
            "base directory for caches and logs", env="ADDER_HOME",
            user_only=True),
    Setting("trace_cache", _home(".claude", ".adder-trace-cache"), _as_path,
            "parse cache file", env="ADDER_TRACE_CACHE",
            user_only=True),
    Setting("catalog", "", str,
            "pin the whole model catalog to one file", env="ADDER_CATALOG",
            env_only=True),
)

BY_NAME: dict[str, Setting] = {s.name: s for s in SETTINGS}


def _read_json(path: Path) -> dict[str, Any]:
    """A config file that does not parse is reported, never silently ignored.

    Silently falling back to defaults on a typo is how someone spends an hour
    wondering why their budget is not applied.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    try:
        d = json.loads(text)
    except (json.JSONDecodeError, ValueError) as e:
        raise ConfigError(f"{path}: {e}") from e
    if not isinstance(d, dict):
        raise ConfigError(f"{path}: top level must be an object, got {type(d).__name__}")
    return d


class ConfigError(ValueError):
    pass


def project_file(start: Path | str | None = None) -> Path | None:
    """Nearest `.adder.json` at or above `start`. None if there is none."""
    p = Path(start or os.getcwd()).expanduser().resolve()
    if p.is_file():
        p = p.parent
    for d in [p, *p.parents]:
        candidate = d / PROJECT_FILE
        if candidate.is_file():
            return candidate
    return None


@dataclass(frozen=True)
class Resolved:
    """One setting's effective value and the layer that supplied it."""

    setting: Setting
    value: Any
    source: str

    @property
    def name(self) -> str:
        return self.setting.name

    @property
    def overridden(self) -> bool:
        return self.source != "default"


# Values the command line supplied, the layer above the environment. Only
# `filters.root_of` writes here: a report run as `adder doctor codex` has said
# which directory it reads, and every setting derived from the root -- the
# harness, the session model -- has to follow that argument rather than the
# configured root the argument replaced. Process-scoped, because a CLI run is
# one process; the test suite clears it between tests.
_ARGUMENTS: dict[str, Any] = {}


def set_argument(name: str, value: Any) -> None:
    """Record a value the command line gave for setting `name`."""
    if name not in BY_NAME:
        raise KeyError(f"unknown setting {name!r}")
    _ARGUMENTS[name] = value


def clear_arguments() -> None:
    _ARGUMENTS.clear()


# Settings whose derived value reads another one. `_derive_harness` and
# `_derive_model` both follow `root`, so asking for either alone resolves it too.
_NEEDS: dict[str, frozenset[str]] = {"harness": frozenset({"root"}),
                                     "model": frozenset({"root"})}


def resolve(*, cwd: Path | str | None = None,
            env: dict[str, str] | None = None,
            derive_model: bool = False,
            names: frozenset[str] | None = None) -> dict[str, Resolved]:
    """Every setting, with its effective value and where it came from.

    `env` is injectable so the tests do not have to mutate `os.environ` -- a
    mutation that leaks into whatever test runs next.

    `names` limits the work to those settings and what they derive from. `get`
    passes it: the `root` default walks every agent's transcript directory,
    and computing all thirty-odd defaults to answer one lookup is what made a
    per-turn `Tier.model` cost a millisecond.
    """
    env = os.environ if env is None else env
    uf = user_file()
    user = _read_json(uf) if uf.is_file() else {}
    pf = project_file(cwd)
    proj = _read_json(pf) if pf else {}

    out: dict[str, Resolved] = {}
    for s in SETTINGS:
        if names is not None and s.name not in names:
            continue
        value, source = s.initial, "default"
        # An env-only setting skips the file layers rather than reporting a
        # value nothing will read. See `Setting.env_only`.
        if not s.env_only:
            if s.name in user:
                value, source = user[s.name], str(user_file())
            if s.name in proj and not s.user_only:
                value, source = proj[s.name], str(pf)
        raw = env.get(s.env_var)
        if raw is not None and raw != "":
            value, source = raw, f"${s.env_var}"
        if s.name in _ARGUMENTS:
            value, source = _ARGUMENTS[s.name], "command line"
        try:
            value = s.cast(value)
        except (TypeError, ValueError) as e:
            raise ConfigError(f"{s.name} from {source}: {e}") from e
        out[s.name] = Resolved(s, value, source)
    _derive_harness(out)
    _derive_enforce(out, cwd)
    if derive_model:
        _derive_model(out)
    return out


def _derive_model(out: dict[str, Resolved]) -> None:
    """An unset `model` is the model the newest session under `root` ran on.

    The `model` setting answers "what does this machine run?", and it used to
    be answered by whoever remembered to write it down -- which nobody did, so
    a Codex user was priced as `claude-opus-5` everywhere a report had no turn
    to read the model from. The transcripts already say. A configured value
    still wins; this only replaces the shipped default.

    Opt-in per call (`get("model")` asks for it) because it reads the disk,
    and every other setting is resolved dozens of times per command.
    """
    m, root = out.get("model"), out.get("root")
    if m is None or root is None or m.source != "default":
        return
    from adder.core.native import latest_model

    found = latest_model(Path(str(root.value)))
    if found:
        out["model"] = Resolved(m.setting, found, "your newest session")


def _derive_harness(out: dict[str, Resolved]) -> None:
    """An unset `harness` follows the transcript directory it is reading.

    Someone who sets `root` to `~/.codex/sessions` has said which agent they
    run; making them say it a second time in `harness` is a step nobody knows
    to take, and the cost of missing it is the placement gate offering Claude
    models to a Codex session and refusing OpenAI ones. An explicit `harness`
    always wins, and a root that is no agent's own leaves the default alone.
    """
    h, root = out.get("harness"), out.get("root")
    if h is None or root is None or h.source != "default":
        return
    from adder.core import native

    name = native.agent_of_root(Path(str(root.value)))
    agent = native.get(name) if name else None
    if agent is not None and agent.harness != h.value:
        out["harness"] = Resolved(h.setting, agent.harness,
                                  f"derived from root ({root.source})")


# What an unset `guard_enforce` means once the plugin is installed. `certain`
# refuses only a call that admits nothing new -- a re-read of an unchanged file
# the context already holds, or of one this session wrote -- and never the same
# call twice, so a wrong refusal costs one turn. Installing a cost plugin is
# asking for it to cut cost; left at `off`, every dollar it could save was
# multiplied by a guess about whether its advice would be taken.
PLUGIN_ENFORCE = "certain"


def _derive_enforce(out: dict[str, Resolved], cwd: Path | str | None) -> None:
    """An unset `guard_enforce` is `certain` when the adder plugin is enabled.

    Only the default moves. Any value in a config file or the environment
    still wins, which is how `adder auto off` turns the plugin's guard back to
    advice: it writes `off` explicitly.
    """
    g = out.get("guard_enforce")
    if g is None or g.source != "default":
        return
    from adder.core.claude import plugin_enabled

    if plugin_enabled(cwd):
        out["guard_enforce"] = Resolved(g.setting, PLUGIN_ENFORCE, "the adder plugin")


def ignored_in_files(*, cwd: Path | str | None = None,
                     env: dict[str, str] | None = None) -> list[str]:
    """Env-only settings a config file tries to set. Named, not silently dropped.

    A key written into `.adder.json` that nothing will ever read is worse than
    a missing one: it looks configured.
    """
    uf = user_file()
    user = _read_json(uf) if uf.is_file() else {}
    pf = project_file(cwd)
    proj = _read_json(pf) if pf else {}
    written = set(user) | set(proj)
    return sorted(s.name for s in SETTINGS if s.env_only and s.name in written)


def ignored_in_project(*, cwd: Path | str | None = None) -> list[str]:
    """User-only settings the project file tries to set, which it may not."""
    pf = project_file(cwd)
    proj = _read_json(pf) if pf else {}
    return sorted(s.name for s in SETTINGS if s.user_only and s.name in proj)


def get(name: str, *, cwd: Path | str | None = None,
        env: dict[str, str] | None = None) -> Any:
    """Effective value of one setting.

    Deliberately not cached. Resolution reads at most two small JSON files, and
    a cached config is a config that ignores the environment a test just set.
    """
    if name not in BY_NAME:
        raise KeyError(f"unknown setting {name!r}; known: {sorted(BY_NAME)}")
    wanted = frozenset({name}) | _NEEDS.get(name, frozenset())
    return resolve(cwd=cwd, env=env, derive_model=name == "model",
                   names=wanted)[name].value


# --------------------------------------------------------------------------
# Derived defaults
# --------------------------------------------------------------------------
#
# Two dozen function signatures used to spell `"claude-opus-5"` or
# `"claude-haiku-4-5"` as their default. Each one was individually harmless and
# collectively they made the tool Claude-only in a way no single edit could
# fix: a Codex user changing the `model` setting still got Haiku quoted as the
# subagent in `placement_cost`, `delegate_threshold`, `savings` and `plan`.
#
# These read the settings instead. Callers keep passing an explicit model
# whenever they know one -- nothing here overrides an argument -- so the only
# behaviour that changes is what happens when nobody said, which is exactly
# where a hardcoded vendor does the damage.


def configured_path(name: str, fallback: Path) -> Path:
    """A path setting the user actually set, or `fallback`.

    Three modules name a file the user is invited to move -- the outcome log,
    the recommendation ledger, the transcript parse cache -- and all three read
    their environment variable **at import time** into a module constant. So
    `adder config` reported the value from `.adder.json`, and the code went on
    using the one from `~/.claude`: a documented setting that did nothing, which
    is the exact failure this module's docstring says it exists to prevent.

    Only an *overridden* value wins. Resolving unconditionally would replace the
    constant with a value equal to it, and those constants are how the tests
    (and `isolated_home`) point a log somewhere harmless; a resolver that
    ignored them would read the developer's real files during a test run.
    """
    try:
        r = resolve()[name]
    except (KeyError, OSError, ValueError):
        return fallback
    return Path(str(r.value)) if r.overridden else fallback


def session_model() -> str:
    """The model to assume for the main conversation when none was read."""
    try:
        return str(get("model")) or "claude-opus-5"
    except (KeyError, OSError, ValueError):
        return "claude-opus-5"


def sub_model() -> str:
    """The model to assume for a delegated subagent when none was named.

    Read off rung T0 of the configured ladder, because that rung is *defined*
    as the cheap read-only tier -- which is precisely what a delegation
    estimate wants. Parsed here rather than imported from `classify` so this
    module stays free of a dependency on the routing layer.
    """
    try:
        raw = str(get("ladder") or "")
    except (KeyError, OSError, ValueError):
        raw = ""
    for part in raw.split(","):
        rung, _, model = part.partition("=")
        if rung.strip().upper() == "T0" and model.strip():
            return model.strip()
    # No ladder written: the T0 a vendor-pinned harness can reach, the same
    # one `classify.ladder` derives, so a Codex estimate is not priced on Haiku.
    from adder.core import harness as _harness

    derived = _harness.vendor_ladder(_harness.get(harness()))
    return derived["T0"] if derived else "claude-haiku-4-5"


def harness() -> str:
    try:
        return str(get("harness")) or "claude-code"
    except (KeyError, OSError, ValueError):
        return "claude-code"


# Settings whose unset value is worked out, not fixed: writing the default down
# is not the same as leaving it unset. `guard_enforce` is `certain` under the
# plugin and `off` only when nothing says otherwise.
_DERIVED: frozenset[str] = frozenset({"root", "model", "harness", "guard_enforce"})


def template() -> str:
    """A starting `.adder.json` that changes nothing about how the tool behaves.

    It used to dump every default, and `adder config` tells people to save it
    as their project file. That pinned `model`, `harness`, `root` and
    `guard_enforce` from the project file, which switched off every derived
    default -- the model stopped following the newest session and the plugin's
    `certain` enforcement dropped to `off` -- and wrote this machine's absolute
    home paths into a file meant to be committed. The next `adder config` then
    warned about ten keys in its own template that a project file may not set.
    """
    body = {s.name: s.initial for s in SETTINGS
            if s.name not in _DERIVED and not s.env_only and not s.user_only
            and not callable(s.default) and s.initial not in ("", None)}
    return json.dumps(body, indent=2, sort_keys=True)
