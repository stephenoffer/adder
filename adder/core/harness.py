"""Which agent runtime is driving, and what it can therefore be told to do.

Why a routing tool needs this at all
------------------------------------
`select.py` prices ~500 models and ranks them. Most of that ranking is
meaningless if the answer cannot be acted on. Under Claude Code the main
conversation is a Claude model by construction: a GPT or open-weight model can
be a subagent, an MCP tool, or an external call, but it cannot *be* the
session. Quoting an inline price for one is quoting a placement that does not
exist, and a recommendation nobody can take is worse than no recommendation --
it costs the reader the time to discover it is impossible.

That constraint used to be a string comparison against the literal
`"claude-code"`, with the only alternative being `"any"`. Two problems, both of
which this module exists to fix:

1. **It named one vendor's harness in library code.** Codex pins the main
   session to OpenAI for exactly the same structural reason, Gemini CLI to
   Google. The rule is not "Claude Code is special", it is "some harnesses pin
   the main session to their vendor". Written the old way, a Codex user got
   OpenAI models refused as main-session candidates and Claude models offered,
   which is precisely backwards.
2. **`any` was doing two jobs.** A harness that routes freely and a harness
   nobody has described yet are different states. The first should relax the
   gate; the second should say so, because silently relaxing a gate is how a
   report ends up recommending a placement that does not exist.

What is deliberately *not* here
-------------------------------
Anything about price, quality, or context. Those belong to the model, not to
the thing running it. A harness record answers one question -- what placements
are physically available -- and it answers it the same way whichever model wins
the ranking.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

# The harness that imposes no constraint, for callers that route natively or
# are just asking the pricing question in the abstract.
ANY = "any"


@dataclass(frozen=True)
class Harness:
    """One agent runtime, described by what it makes possible."""

    name: str
    # Vendor the main conversation is pinned to, lowercase, or "" for none.
    # This is the whole reason the module exists.
    main_session_org: str = ""
    # Can work be handed to a separate, throwaway context? Without this the
    # dominant lever in this repo -- delegation -- is not available, and a
    # report that recommends it anyway is recommending a feature the user does
    # not have.
    supports_subagents: bool = True
    # Can a reasoning-effort level be set per call?
    supports_effort: bool = True
    # Can a model be switched mid-session? Where it cannot, the "downgrade the
    # warm conversation" question is moot and should not be raised.
    supports_model_switch: bool = True
    # Does the harness expose the prompt cache as something to place
    # breakpoints in, or is it handled for you?
    exposes_cache_control: bool = True
    notes: str = ""

    @property
    def pins_main_session(self) -> bool:
        return bool(self.main_session_org)

    def allows_main_session(self, org: str) -> bool:
        """Can a model from `org` be the main conversation here?

        Unknown vendor plus a pinning harness is False, not True. The gate
        exists to stop a recommendation that cannot be acted on, and one that
        passes because it could not identify the vendor is not a gate.
        """
        if not self.pins_main_session:
            return True
        return (org or "").strip().lower() == self.main_session_org

    def why_blocked(self, org: str) -> str:
        return (f"{org or 'this vendor'} cannot be the main {self.label} "
                f"session, which runs on {self.main_session_org}; reachable as "
                f"a subagent or tool call only")

    @property
    def label(self) -> str:
        return self.name

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "main_session_org": self.main_session_org,
            "supports_subagents": self.supports_subagents,
            "supports_effort": self.supports_effort,
            "supports_model_switch": self.supports_model_switch,
            "exposes_cache_control": self.exposes_cache_control,
            "notes": self.notes,
        }

    @staticmethod
    def from_json(d: dict[str, Any]) -> Harness:
        return Harness(
            name=str(d.get("name", "")),
            main_session_org=str(d.get("main_session_org", "") or "").lower(),
            supports_subagents=bool(d.get("supports_subagents", True)),
            supports_effort=bool(d.get("supports_effort", True)),
            supports_model_switch=bool(d.get("supports_model_switch", True)),
            exposes_cache_control=bool(d.get("exposes_cache_control", True)),
            notes=str(d.get("notes", "")),
        )


CLAUDE_CODE = Harness(
    name="claude-code",
    main_session_org="anthropic",
    supports_subagents=True,
    supports_effort=True,
    supports_model_switch=True,
    exposes_cache_control=True,
    notes="main session is a Claude model; other vendors reachable as subagents "
          "or MCP tools",
)

CODEX = Harness(
    name="codex",
    main_session_org="openai",
    supports_subagents=True,
    supports_effort=True,
    exposes_cache_control=False,   # OpenAI caching is automatic
    notes="main session is an OpenAI model; prompt caching is automatic, so "
          "breakpoint placement is not a lever here",
)

GEMINI_CLI = Harness(
    name="gemini-cli",
    main_session_org="google",
    supports_subagents=True,
    supports_effort=True,
    exposes_cache_control=False,
    notes="main session is a Gemini model; implicit caching is automatic",
)

# Harnesses that genuinely route across vendors.
AIDER = Harness(name="aider", supports_subagents=False,
                notes="single conversation, no subagent placement to recommend")
OPENHANDS = Harness(name="openhands", notes="routes across vendors")
CUSTOM = Harness(name="custom", notes="a loop you wrote; nothing is assumed")

ANY_HARNESS = Harness(
    name=ANY,
    notes="no constraint applied; every model is treated as placeable anywhere",
)

_BUILTIN: dict[str, Harness] = {
    h.name: h for h in (CLAUDE_CODE, CODEX, GEMINI_CLI, AIDER, OPENHANDS,
                        CUSTOM, ANY_HARNESS)
}

_ALIASES = {
    "claude": "claude-code",
    "claude-cli": "claude-code",
    "cc": "claude-code",
    "openai-codex": "codex",
    "codex-cli": "codex",
    "gemini": "gemini-cli",
    "": ANY,
    "none": ANY,
}


def _overrides() -> dict[str, Harness]:
    """`ADDER_HARNESSES=<path>` adds or amends harnesses without a code change.

    Same layering rule as the provider table: a file that mentions one field
    amends that field and leaves the rest of the record alone.
    """
    path = os.environ.get("ADDER_HARNESSES")
    if not path:
        return {}
    try:
        raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    entries = raw.get("harnesses", raw) if isinstance(raw, dict) else {}
    if not isinstance(entries, dict):
        return {}
    out: dict[str, Harness] = {}
    for name, d in entries.items():
        if not isinstance(d, dict):
            continue
        key = str(name).lower()
        over = Harness.from_json({**d, "name": key})
        base = _BUILTIN.get(key)
        if base is None:
            out[key] = over
            continue
        fields = {f: getattr(over, f) for f in d if hasattr(over, f) and f != "name"}
        out[key] = replace(base, **fields)
    return out


def all_harnesses() -> dict[str, Harness]:
    return {**_BUILTIN, **_overrides()}


def get(name: str | None) -> Harness:
    """Harness by name or alias.

    An unrecognised name resolves to `any` rather than raising, because the
    alternative is a cost report that refuses to run over a spelling. It does
    not silently pin anything, so the worst case is a gate that does not fire.
    """
    key = (name or "").strip().lower()
    key = _ALIASES.get(key, key)
    return all_harnesses().get(key, ANY_HARNESS)


def names() -> list[str]:
    """Every selectable harness, for `--harness` choices and completion."""
    return sorted(all_harnesses())


def default() -> str:
    """The harness to assume when nobody said.

    `ADDER_HARNESS` first, so a Codex or Gemini CLI user can set it once and
    have every report stop assuming Claude Code. The fallback stays
    `claude-code` because that is what this tool reads transcripts from by
    default, and changing that default would silently relax a gate for the
    users who have it hardest.
    """
    env = os.environ.get("ADDER_HARNESS")
    if env and get(env) is not ANY_HARNESS:
        return get(env).name
    if env:
        return ANY
    return "claude-code"


def infer_from_models(models: list[str]) -> Harness:
    """Guess the harness from the models a workload actually ran on.

    Not a substitute for being told, and used only where nothing was. A
    workload whose main-chain turns are all one vendor's is running on a
    harness pinned to that vendor, or on one that happens to be configured
    that way -- and either way, applying that vendor's pin produces the right
    placement advice.
    """
    from adder.pricing.registry import provider_for

    orgs = {provider_for(m).name for m in models if m}
    orgs.discard("unknown")
    if len(orgs) != 1:
        return ANY_HARNESS
    org = orgs.pop()
    for h in _BUILTIN.values():
        if h.main_session_org == org:
            return h
    return ANY_HARNESS
