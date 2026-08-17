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

USER_FILE = Path.home() / ".claude" / "adder.json"
PROJECT_FILE = ".adder.json"


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _as_path(v: Any) -> str:
    return str(Path(str(v)).expanduser())


@dataclass(frozen=True)
class Setting:
    name: str
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

    @property
    def env_var(self) -> str:
        return self.env or f"ADDER_{self.name.upper()}"


# Ordered for display: the ones a person changes first come first.
SETTINGS: tuple[Setting, ...] = (
    Setting("root", str(Path.home() / ".claude" / "projects"), _as_path,
            "transcript directory every report reads"),
    Setting("model", "claude-opus-5", str,
            "model assumed for the session when a report cannot read one"),
    Setting("harness", "claude-code", str,
            "agent runtime driving the session: claude-code, codex, gemini-cli, "
            "aider, openhands, custom, or any. Harnesses that pin the main "
            "session to one vendor make other vendors subagent-only"),
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
    Setting("guard_advice_taken", 0.5, float,
            "assumed share of guard advice that is acted on; discounts the "
            "saving before it is weighed against the cost of saying it"),
    Setting("guard_max_fires", 15, int,
            "most times the guard may speak in one session"),
    Setting("guard_state", str(Path.home() / ".claude" / ".adder-guard.json"), _as_path,
            "per-session guard memory: files read, shapes already advised"),
    Setting("size_model", str(Path.home() / ".claude" / ".adder-sizes.json"), _as_path,
            "learned result-size quantiles the guard predicts from"),
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
    Setting("log", str(Path.home() / ".claude" / "adder-outcomes.jsonl"), _as_path,
            "dispatch outcome log that calibrates p_fail", env="ADDER_LOG"),
    Setting("ledger", str(Path.home() / ".claude" / "adder-ledger.jsonl"), _as_path,
            "ledger of recommendations made and verified", env="ADDER_LEDGER"),
    Setting("home", str(Path.home() / ".claude"), _as_path,
            "base directory for caches and logs", env="ADDER_HOME"),
    Setting("trace_cache", str(Path.home() / ".claude" / ".adder-trace-cache"), _as_path,
            "parse cache file", env="ADDER_TRACE_CACHE"),
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


def resolve(*, cwd: Path | str | None = None,
            env: dict[str, str] | None = None) -> dict[str, Resolved]:
    """Every setting, with its effective value and where it came from.

    `env` is injectable so the tests do not have to mutate `os.environ` -- a
    mutation that leaks into whatever test runs next.
    """
    env = os.environ if env is None else env
    user = _read_json(USER_FILE) if USER_FILE.is_file() else {}
    pf = project_file(cwd)
    proj = _read_json(pf) if pf else {}

    out: dict[str, Resolved] = {}
    for s in SETTINGS:
        value, source = s.default, "default"
        # An env-only setting skips the file layers rather than reporting a
        # value nothing will read. See `Setting.env_only`.
        if not s.env_only:
            if s.name in user:
                value, source = user[s.name], str(USER_FILE)
            if s.name in proj:
                value, source = proj[s.name], str(pf)
        raw = env.get(s.env_var)
        if raw is not None and raw != "":
            value, source = raw, f"${s.env_var}"
        try:
            value = s.cast(value)
        except (TypeError, ValueError) as e:
            raise ConfigError(f"{s.name} from {source}: {e}") from e
        out[s.name] = Resolved(s, value, source)
    return out


def ignored_in_files(*, cwd: Path | str | None = None,
                     env: dict[str, str] | None = None) -> list[str]:
    """Env-only settings a config file tries to set. Named, not silently dropped.

    A key written into `.adder.json` that nothing will ever read is worse than
    a missing one: it looks configured.
    """
    user = _read_json(USER_FILE) if USER_FILE.is_file() else {}
    pf = project_file(cwd)
    proj = _read_json(pf) if pf else {}
    written = set(user) | set(proj)
    return sorted(s.name for s in SETTINGS if s.env_only and s.name in written)


def get(name: str, *, cwd: Path | str | None = None,
        env: dict[str, str] | None = None) -> Any:
    """Effective value of one setting.

    Deliberately not cached. Resolution reads at most two small JSON files, and
    a cached config is a config that ignores the environment a test just set.
    """
    if name not in BY_NAME:
        raise KeyError(f"unknown setting {name!r}; known: {sorted(BY_NAME)}")
    return resolve(cwd=cwd, env=env)[name].value


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
    return "claude-haiku-4-5"


def harness() -> str:
    try:
        return str(get("harness")) or "claude-code"
    except (KeyError, OSError, ValueError):
        return "claude-code"


def template() -> str:
    """A commented-by-example config file, with every setting at its default."""
    body = {s.name: s.default for s in SETTINGS if s.default not in ("", None)}
    return json.dumps(body, indent=2, sort_keys=True)
