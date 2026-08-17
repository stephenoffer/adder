"""What the always-loaded prefix costs, per file, on every turn of every session.

Every other report here prices something that happened *during* a session. This
one prices what was already true before it started: the instruction files, the
memory index, and the skill and agent descriptions that are pasted into the
context of every session on this machine, whether or not anything reads them.

The mistake this module exists to correct
-----------------------------------------
`prefix` measures a 27,953-token opening context and reports that ~74% of it
arrives as a **cache read**, so a restart is cheap. That is a true statement
about restarts, and it was quietly read as a statement about the floor: if the
floor is served at 0.10x, why would its size matter?

Because a floor token is not read once per session. It is read once per
**turn**. At the measured re-read multiplier and the cost-weighted median
session length, 1,000 tokens of `CLAUDE.md` is not a 1,000-token cost -- it is
1,000 tokens re-read several hundred times, in every session, indefinitely.

And unlike a tool result, memory has **no residency decay**. `carry` prices a
tool result against the compaction that will eventually evict it. Compaction
cannot evict `CLAUDE.md`: the prefix is rebuilt from the same file, so the
survival term is 1.0 forever. Memory is the only content in a context whose
carry never ends, and the only content whose size is set by a file you can edit
in ten seconds.

What counts as resident
-----------------------
Resident means "in the prefix of a turn that read none of it".

    CLAUDE.md (user + project)   all of it, every turn          resident
    MEMORY.md index              all of it, every turn          resident
    memory/*.md                  only when recalled             on demand
    .claude/skills/*/SKILL.md    name + description only        resident
    .claude/agents/*.md          name + description only        resident
    nested CLAUDE.md             only when that dir is read     on demand

The split matters more than the total. A 40,000-token skill library costs
almost nothing resident and a 4,000-token `CLAUDE.md` costs real money, which
is the opposite of what file sizes suggest.

What is not counted
-------------------
The system prompt and the tool schemas are the rest of the floor and are not
on disk, so they cannot be read here. They are reported as `unaccounted`
against the measured opening context rather than silently folded into a file's
share -- a residual named as a residual is honest; one attributed to
`CLAUDE.md` is a wrong number about a file someone is about to edit.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from adder.core import settings as _settings
from adder.core.filters import root_of as _root_of
from adder.pricing.cost import Rates
from adder.pricing.prices import CACHE_READ_MULT
from adder.pricing.registry import rate
from adder.util.text import est_tokens

M = 1_000_000.0

# Per-entry overhead for a skill or agent listed in the prefix: the wrapper
# around the name and description (separators, the "name:" label, the tool
# framing). Small, but multiplied by every skill installed, and a report that
# counts only the description under-states a 60-skill machine by ~600 tokens.
ENTRY_OVERHEAD_TOKENS = 10

# Directories a nested-CLAUDE.md walk must never descend into. Without this the
# walk is O(repo) and finds vendored instruction files nobody wrote.
SKIP_DIRS = frozenset({
    ".git", ".hg", "node_modules", "__pycache__", ".venv", "venv", "env",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    "site-packages", ".next", "target", "vendor",
})

# How deep to look for nested CLAUDE.md files. Deeper than this and the file is
# unlikely to be loaded often enough to price.
MAX_DEPTH = 4

# A skill or agent description longer than this is being used as documentation.
# The body is the place for documentation; the description is in every prefix.
DESCRIPTION_BUDGET_CHARS = 500

# Lines shorter than this are boilerplate ("## Why", "---", "```") and repeat
# across documents for reasons that are not duplication.
DUP_MIN_CHARS = 60

KINDS = ("claude-md", "memory-index", "memory", "skill", "agent")


@dataclass(frozen=True)
class Doc:
    """One file that contributes to the prefix, and how much of it does."""

    path: Path
    kind: str
    scope: str          # "user" | "project"
    load: str           # "always" | "on-demand"
    bytes_: int
    tokens: int         # the whole file
    resident: int       # of it, present in every prefix
    title: str = ""

    @property
    def name(self) -> str:
        return self.title or self.path.name

    @property
    def on_demand_tokens(self) -> int:
        return max(0, self.tokens - self.resident)


@dataclass(frozen=True)
class Finding:
    """Something to fix, with the resident tokens it would return."""

    kind: str
    detail: str
    tokens: int = 0
    path: Path | None = None


@dataclass(frozen=True)
class Pricing:
    """The measured terms that turn resident tokens into dollars.

    `source` is load-bearing for the same reason it is in `Carry` and
    `Opening`: a per-session figure fitted to 200 sessions and one taken from a
    prior print identically, and the reader deciding whether to delete half
    their instruction file needs to know which they are holding.
    """

    model: str = field(default_factory=_settings.session_model)
    turns: float = 100.0
    sessions: int = 0
    # Sessions that actually load this repository's files. A project
    # `CLAUDE.md` is resident in *its* project's sessions and nowhere else;
    # pricing it across every session on the machine over-states it by the
    # ratio of the two, which on a machine with several repos is several fold.
    project_sessions: int = 0
    # Anthropic-shaped prior, correct for the default model above. For any
    # other model build the record with `Pricing.prior(model)`, which reads the
    # multiplier off that model's provider instead of assuming this one.
    read_mult: float = CACHE_READ_MULT
    warm_share: float = 0.0
    ttl: str = "1h"
    source: str = "prior"
    on: date | None = None

    @property
    def measured(self) -> bool:
        return self.source == "measured"

    @property
    def _rates(self) -> Rates:
        return Rates.for_model(self.model, ttl=self.ttl, on=self.on)

    @property
    def write_mult(self) -> float:
        """Write premium as a multiple of input, for this model's provider.

        1.0 under automatic caching, where laying down a prefix is billed as
        ordinary input. Quoting Anthropic's 1.25x there overstates the cost of
        every restart, which makes long sessions look better than they are --
        the exact bias this report exists to remove.
        """
        r = self._rates
        return (r.cache_write / r.inp) if r.inp else 1.0

    def open_cost(self, tokens: float) -> float:
        """USD to put `tokens` into the prefix once, at the measured warmth.

        A restart does not rewrite the whole floor: the part identical to the
        last session is still resident and is served at 0.10x. Only the cold
        share is written.
        """
        r = self._rates
        cold, warm = 1.0 - self.warm_share, self.warm_share
        return tokens * (cold * r.cache_write + warm * r.cache_read) / M

    def carry_cost(self, tokens: float) -> float:
        """USD to re-read `tokens` on every turn after the opening one."""
        r = rate(self.model, self.on).inp
        return tokens * r * self.read_mult * max(0.0, self.turns - 1.0) / M

    def session_cost(self, tokens: float) -> float:
        return self.open_cost(tokens) + self.carry_cost(tokens)

    def window_cost(self, tokens: float, *, scope: str = "user") -> float:
        """The same tokens across the sessions that actually load them.

        `scope="project"` counts only this repository's sessions; `"user"`
        counts every session on record, which is right for `~/.claude/CLAUDE.md`
        and wrong for everything under a repo.
        """
        n = self.project_sessions if scope == "project" else self.sessions
        return self.session_cost(tokens) * max(0, n)

    def sessions_for(self, scope: str) -> int:
        return self.project_sessions if scope == "project" else self.sessions

    def per_1k(self) -> float:
        """USD per 1,000 resident tokens per session — the editing unit."""
        return self.session_cost(1_000)

    def describe(self) -> str:
        if not self.measured:
            return ("pricing: prior (no local sessions); assuming "
                    f"{self.turns:.0f} turns and the {self.read_mult:.2f}x "
                    "re-read multiplier")
        return (f"pricing from {self.sessions} sessions on {self.model}: "
                f"{self.turns:.0f} turns each, re-read multiplier "
                f"{self.read_mult:.3f}x, opening {self.warm_share:.0%} warm")

    @classmethod
    def prior(cls, model: str, *, ttl: str = "1h", turns: float = 100.0,
              on: date | None = None) -> Pricing:
        """An unmeasured record whose multiplier still matches the provider.

        The dataclass default is Anthropic's 0.10x, which is right for the
        default model and wrong for every other one. Constructing a prior for
        `gpt-5` with 0.10x understates its re-reads by half; for a model with
        no cache at all it understates them tenfold.
        """
        r = Rates.for_model(model, ttl=ttl, on=on)
        return cls(model=model, ttl=ttl, turns=turns, on=on,
                   read_mult=(r.cache_read / r.inp) if r.inp else 1.0)

    @classmethod
    def measure(cls, sessions, *, model: str | None = None, ttl: str = "1h",
                on: date | None = None, project_sessions: int | None = None
                ) -> Pricing:
        """Fit the pricing terms to recorded transcripts.

        Session length is the **cost-weighted** median, not the plain one: a
        resident token is re-read once per turn, so the sessions that carry it
        most are the long ones, and the plain median of a heavy-tailed length
        distribution under-states that by a factor of several.
        """
        from adder.measure.window.carry import Carry
        from adder.measure.window.prefix import measure as measure_opening
        from adder.measure.window.prefix import weighted_median_turns

        n_project = len(sessions) if project_sessions is None else project_sessions
        if not sessions:
            return cls.prior(model or _settings.session_model(), ttl=ttl, on=on)

        c = Carry.measure(sessions)
        op = measure_opening(sessions)
        turns = weighted_median_turns(sessions)
        if model is None:
            model = _dominant_model(sessions)
        return cls(
            model=model,
            turns=float(max(1, turns)),
            sessions=len(sessions),
            project_sessions=n_project,
            read_mult=c.read_mult,
            warm_share=op.warm_share if op.measured else 0.0,
            ttl=ttl,
            source="measured" if c.measured or op.measured else "prior",
            on=on,
        )


def _dominant_model(sessions) -> str:
    """The model that carries the most spend — the one the floor is priced at."""
    spend: dict[str, float] = {}
    for s in sessions.values():
        for model, cost in s.cost_by_model().items():
            spend[model] = spend.get(model, 0.0) + cost
    if not spend:
        return _settings.session_model()
    return max(spend.items(), key=lambda kv: kv[1])[0]


_FM = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.S)


def frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split a leading `---` block into scalar keys and the remaining body.

    Deliberately not a YAML parser: no dependency is allowed here, and the only
    keys that matter for prefix cost are the scalar `name` and `description`
    that Claude Code lists. A block-scalar or nested value is kept as its raw
    text so its *size* is still counted; nothing downstream interprets it.
    """
    m = _FM.match(text)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    key = None
    for line in m.group(1).splitlines():
        if not line.strip():
            continue
        if line[:1] not in (" ", "\t") and ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            meta[key] = val.strip().strip("'\"")
        elif key is not None:
            meta[key] = (meta[key] + " " + line.strip()).strip()
    return meta, text[m.end():]


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _doc(path: Path, kind: str, scope: str, load: str, *,
         resident: int | None = None, title: str = "") -> Doc:
    text = _read(path)
    tok = est_tokens(text)
    return Doc(path=path, kind=kind, scope=scope, load=load,
               bytes_=len(text.encode("utf-8")), tokens=tok,
               resident=tok if resident is None else max(0, resident),
               title=title)


def _entry_doc(path: Path, kind: str, scope: str) -> Doc:
    """A skill or agent: only its name and description are resident."""
    meta, _ = frontmatter(_read(path))
    listed = f"{meta.get('name', path.stem)} {meta.get('description', '')}"
    resident = est_tokens(listed) + ENTRY_OVERHEAD_TOKENS
    return _doc(path, kind, scope, "always", resident=resident,
                title=meta.get("name", path.stem))


def _nested_claude_md(project: Path) -> list[Path]:
    """Nested instruction files, bounded in depth and pruned of build output."""
    out: list[Path] = []
    root_depth = len(project.resolve().parts)
    stack = [project]
    # Resolved directories already walked. Without this the walk does not
    # terminate: the depth test is applied to `e.resolve()`, so a symlink
    # pointing back at an ancestor resolves to a *shorter* path, passes the
    # test forever, and `adder memory` hangs on any repository containing one.
    # A `node_modules`-style self-link is the common case and `SKIP_DIRS` only
    # covers the names somebody thought of.
    seen: set[Path] = set()
    while stack:
        d = stack.pop()
        real = d.resolve()
        if real in seen:
            continue
        seen.add(real)
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for e in entries:
            if e.is_dir():
                if e.name in SKIP_DIRS or e.name.startswith("."):
                    continue
                if len(e.resolve().parts) - root_depth < MAX_DEPTH:
                    stack.append(e)
            elif e.name == "CLAUDE.md" and e.parent != project:
                out.append(e)
    return sorted(out)


def memory_dir(project: Path, root: Path) -> Path | None:
    """Where this project's memory files live, if they exist.

    Memory is keyed by the same path slug the transcripts are, so it is found
    the same way `live` finds a session rather than by a second convention that
    can drift from the first.
    """
    from adder.measure.session.live import find_project_dir

    d = find_project_dir(project, root)
    if d is None:
        return None
    mem = d / "memory"
    return mem if mem.is_dir() else None


def discover(project: Path | str | None = None, *, home: Path | str | None = None,
             root: Path | str | None = None) -> list[Doc]:
    """Every file that contributes to a prefix, with its resident share.

    `home` is the `~/.claude` directory and `root` the transcript root beneath
    it. Both are parameters rather than constants so a test can point them at a
    temp directory; nothing here ever falls back to reading a real one that was
    not asked for.
    """
    project = Path(project or Path.cwd()).resolve()
    home = Path(home) if home is not None else Path.home() / ".claude"
    root = Path(root) if root is not None else home / "projects"

    docs: list[Doc] = []

    for path, scope in ((home / "CLAUDE.md", "user"),
                        (project / "CLAUDE.md", "project"),
                        (project / "CLAUDE.local.md", "project"),
                        (project / ".claude" / "CLAUDE.md", "project")):
        if path.is_file():
            docs.append(_doc(path, "claude-md", scope, "always"))

    for path in _nested_claude_md(project):
        docs.append(_doc(path, "claude-md", "project", "on-demand"))

    mem = memory_dir(project, root)
    if mem is not None:
        index = mem / "MEMORY.md"
        if index.is_file():
            docs.append(_doc(index, "memory-index", "project", "always"))
        for path in sorted(mem.glob("*.md")):
            if path.name == "MEMORY.md":
                continue
            meta, _ = frontmatter(_read(path))
            docs.append(_doc(path, "memory", "project", "on-demand", resident=0,
                             title=meta.get("name", path.stem)))

    for base, scope in ((home, "user"), (project / ".claude", "project")):
        for path in sorted((base / "skills").glob("*/SKILL.md")):
            docs.append(_entry_doc(path, "skill", scope))
        for path in sorted((base / "agents").glob("*.md")):
            docs.append(_entry_doc(path, "agent", scope))

    return docs


def project_sessions(sessions, project: Path | str) -> dict:
    """The sessions that ran in `project`, matched the way `live` matches them.

    Claude Code names a transcript directory after the working directory's path
    slug. Reusing `live.slug_for` rather than re-deriving it keeps one
    definition of "this project" in the repo; a second one drifts, and the
    symptom is a per-file cost that is wrong by the ratio of two session counts.
    """
    from adder.measure.session.live import slug_for

    want = slug_for(project).lower()
    return {sid: s for sid, s in sessions.items()
            if (s.project or "").lower() == want}


def resident_tokens(docs) -> int:
    return sum(d.resident for d in docs)


def by_kind(docs) -> dict[str, tuple[int, int]]:
    """kind -> (files, resident tokens), ordered by resident cost."""
    out: dict[str, list[int]] = {}
    for d in docs:
        row = out.setdefault(d.kind, [0, 0])
        row[0] += 1
        row[1] += d.resident
    return {k: (v[0], v[1]) for k, v in
            sorted(out.items(), key=lambda kv: -kv[1][1])}


def unaccounted(docs, floor_tokens: int) -> int:
    """Floor tokens not explained by any file: system prompt and tool schemas."""
    return max(0, int(floor_tokens) - resident_tokens(docs))


_LINK = re.compile(r"\[\[([^\]]+)\]\]")
_INDEX_ROW = re.compile(r"\[[^\]]+\]\(([^)]+\.md)\)")


def stale_links(docs) -> list[Finding]:
    """`[[name]]` references with no memory file behind them.

    A dangling link is not a cosmetic problem: the agent that reads the index
    spends a recall on a file that is not there, and the next agent repeats it.
    """
    names = {d.name for d in docs if d.kind == "memory"}
    names |= {d.path.stem for d in docs if d.kind == "memory"}
    out: list[Finding] = []
    for d in docs:
        if d.kind not in ("memory", "memory-index"):
            continue
        missing = sorted({m for m in _LINK.findall(_read(d.path)) if m not in names})
        for name in missing:
            out.append(Finding("stale-link",
                               f"{d.path.name} links to [[{name}]], which does not exist",
                               path=d.path))
    return out


def index_drift(docs) -> list[Finding]:
    """Memory files the index does not list, and rows pointing at nothing."""
    index = next((d for d in docs if d.kind == "memory-index"), None)
    files = {d.path.name for d in docs if d.kind == "memory"}
    if index is None:
        return ([Finding("no-index",
                         f"{len(files)} memory files and no MEMORY.md index; "
                         "nothing tells the agent they exist", tokens=0)]
                if files else [])
    text = _read(index.path)
    listed = {Path(t).name for t in _INDEX_ROW.findall(text)}
    out: list[Finding] = []
    for name in sorted(files - listed):
        out.append(Finding("unindexed",
                           f"{name} is never listed in MEMORY.md, so it is only "
                           "found by accident", path=index.path))
    for name in sorted(listed - files):
        out.append(Finding("dangling-row",
                           f"MEMORY.md lists {name}, which does not exist",
                           path=index.path))
    return out


def _norm(line: str) -> str:
    return " ".join(line.split()).lower()


def duplicates(docs) -> list[Finding]:
    """Substantial lines that are resident in more than one document.

    Duplicated instruction text is paid for twice on every turn, and it is the
    cheapest thing on this report to fix, because deleting one copy changes no
    behaviour.
    """
    seen: dict[str, list[Doc]] = {}
    for d in docs:
        if d.load != "always":
            continue
        for line in dict.fromkeys(_read(d.path).splitlines()):
            n = _norm(line)
            if len(n) < DUP_MIN_CHARS:
                continue
            seen.setdefault(n, []).append(d)
    groups: dict[tuple[str, ...], int] = {}
    for line, ds in seen.items():
        names = tuple(sorted({str(x.path) for x in ds}))
        if len(names) < 2:
            continue
        groups[names] = groups.get(names, 0) + est_tokens(line)
    out = []
    for names, size in sorted(groups.items(), key=lambda kv: -kv[1]):
        # `Finding.tokens` is what deleting this returns to the prefix, and a
        # line living in three files is paid for three times: two copies come
        # back, not one. Reporting the size of a single copy was right for the
        # two-file case and understated every wider one -- and the widest group
        # is the likeliest to be boilerplate worth deleting.
        recoverable = size * (len(names) - 1)
        where = " and ".join(_label(n) for n in names)
        extra = f", paid for {len(names)}x" if len(names) > 2 else ""
        out.append(Finding(
            "duplicate",
            f"{size:,} tokens of identical text in {where}{extra}",
            tokens=recoverable))
    return sorted(out, key=lambda f: -f.tokens)


def _label(path: str) -> str:
    """`SKILL.md and SKILL.md` names nothing. Keep the parent when the leaf
    is a convention rather than a name."""
    p = Path(path)
    return f"{p.parent.name}/{p.name}" if p.name in {
        "SKILL.md", "CLAUDE.md", "MEMORY.md", "index.md"} else p.name


def oversize_descriptions(docs, *, budget: int = DESCRIPTION_BUDGET_CHARS
                          ) -> list[Finding]:
    """Skill and agent descriptions long enough to be documentation."""
    out: list[Finding] = []
    for d in docs:
        if d.kind not in ("skill", "agent"):
            continue
        meta, _ = frontmatter(_read(d.path))
        desc = meta.get("description", "")
        if len(desc) > budget:
            over = est_tokens(desc[budget:])
            out.append(Finding(
                "long-description",
                f"{d.name} has a {len(desc):,}-char description "
                f"({budget:,} is plenty); it is resident in every prefix",
                tokens=over, path=d.path))
    return sorted(out, key=lambda f: -f.tokens)


def _repo_paths(project: Path) -> set[str]:
    """Every file in the repo, as POSIX suffixes, for reference checking.

    Built once and matched by suffix because an instruction file names paths
    from wherever the writer was standing: `core/trace.py`, `adder/core/trace.py`
    and `./core/trace.py` are the same claim about the same file. Resolving only
    against the repo root reported all three of this repo's real layout
    references as stale, which is worse than not checking -- a checker that
    cries wolf gets its whole report ignored.
    """
    out: set[str] = set()
    stack = [project]
    root_depth = len(project.resolve().parts)
    seen: set[Path] = set()          # same symlink loop as `_nested_claude_md`
    while stack:
        d = stack.pop()
        real = d.resolve()
        if real in seen:
            continue
        seen.add(real)
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for e in entries:
            if e.is_dir():
                if e.name in SKIP_DIRS:
                    continue
                if len(e.resolve().parts) - root_depth < MAX_DEPTH + 2:
                    stack.append(e)
            else:
                try:
                    out.add(e.resolve().relative_to(project).as_posix())
                except ValueError:
                    continue
    return out


def stale_paths(docs, project: Path) -> list[Finding]:
    """Backticked paths in resident docs that no longer exist anywhere.

    An instruction file that names a moved module teaches every future session
    a wrong fact, and pays to teach it on every turn.
    """
    pat = re.compile(r"`([\w./-]+\.(?:py|md|json|toml|ya?ml|sh|ts|js))`")
    known = _repo_paths(project)
    out: list[Finding] = []
    for d in docs:
        # Only instruction files: a skill body is not resident, and its prose
        # names files that belong to whatever repo the skill is used in.
        if d.kind not in ("claude-md", "memory"):
            continue
        missing = []
        for ref in dict.fromkeys(pat.findall(_read(d.path))):
            # A bare filename is usually a generic mention ("settings.json"),
            # not a claim about this repo. Only a path can be stale.
            if "/" not in ref:
                continue
            rel = ref.lstrip("./")
            if any(k == rel or k.endswith("/" + rel) for k in known):
                continue
            if (project / ref).exists() or (d.path.parent / ref).exists():
                continue
            missing.append(ref)
        if missing:
            out.append(Finding(
                "stale-path",
                f"{d.path.name} names {len(missing)} path(s) that do not exist: "
                + ", ".join(missing[:4]) + ("\u2026" if len(missing) > 4 else ""),
                path=d.path))
    return out


# Findings that are wrong rather than merely expensive. A dangling link or a
# path that moved teaches every future session a false fact, and it does so at
# the same price as a true one -- so these fail `--strict` at any size, while a
# few duplicated tokens do not.
CORRECTNESS_KINDS = frozenset({"stale-link", "dangling-row", "stale-path"})

# Resident tokens of duplication tolerated before `--strict` fails a run.
DEFAULT_MAX_WASTE = 200


@dataclass
class MemoryReport:
    docs: list[Doc]
    pricing: Pricing
    findings: list[Finding] = field(default_factory=list)
    floor_tokens: int = 0
    project: Path | None = None

    @property
    def resident(self) -> int:
        return resident_tokens(self.docs)

    @property
    def unaccounted_tokens(self) -> int:
        return unaccounted(self.docs, self.floor_tokens)

    @property
    def controllable_share(self) -> float:
        """Share of the measured floor that is a file the user can edit."""
        return self.resident / self.floor_tokens if self.floor_tokens else 0.0

    @property
    def recoverable_tokens(self) -> int:
        return sum(f.tokens for f in self.findings)

    @property
    def wrong(self) -> list[Finding]:
        """Findings that are false statements, not just expensive ones."""
        return [f for f in self.findings if f.kind in CORRECTNESS_KINDS]

    def fails(self, *, max_waste: int = DEFAULT_MAX_WASTE) -> bool:
        return bool(self.wrong) or self.recoverable_tokens >= max_waste

    def cost(self, doc: Doc) -> float:
        return self.pricing.window_cost(doc.resident, scope=doc.scope)

    def ranked(self) -> list[Doc]:
        return sorted(self.docs, key=lambda d: (-d.resident, str(d.path)))


def analyse(sessions, project: Path | str | None = None, *,
            home: Path | str | None = None, root: Path | str | None = None,
            model: str | None = None, ttl: str = "1h",
            on: date | None = None) -> MemoryReport:
    from adder.measure.window.prefix import measure as measure_opening

    project = Path(project or Path.cwd()).resolve()
    docs = discover(project, home=home, root=root)
    pricing = Pricing.measure(sessions, model=model, ttl=ttl, on=on,
                              project_sessions=len(project_sessions(sessions, project)))
    op = measure_opening(sessions) if sessions else None
    floor = op.floor_tokens if op is not None else 0
    findings = (duplicates(docs) + oversize_descriptions(docs)
                + index_drift(docs) + stale_links(docs) + stale_paths(docs, project))
    return MemoryReport(docs=docs, pricing=pricing, findings=findings,
                        floor_tokens=floor, project=project)


def report(rep: MemoryReport, *, top: int = 12) -> str:
    from adder.util.render import money, table, tokens, warn

    p = rep.pricing
    lines = [
        f"  {len(rep.docs)} files feed the prefix · "
        f"{tokens(rep.resident)} resident in every turn",
        f"  {p.describe()}",
        "",
    ]

    if not rep.docs:
        return "\n".join([*lines, "  Nothing found. Pass --project to point at a repo."])

    body = []
    for d in rep.ranked()[:top]:
        body.append([
            d.name[:34],
            d.kind,
            d.scope,
            d.load,
            tokens(d.tokens),
            tokens(d.resident) if d.resident else "—",
            money(p.session_cost(d.resident)) if d.resident else "—",
            money(rep.cost(d)) if d.resident else "—",
            p.sessions_for(d.scope) if d.resident else "—",
        ])
    lines += table(body, ["file", "kind", "scope", "loaded", "size", "resident",
                          "$/session", "$ so far", "sessions"],
                   align="<<<<>>>>>")
    if len(rep.docs) > top:
        lines.append(f"    … {len(rep.docs) - top} more")

    lines += ["", f"  Editing unit: {money(p.per_1k())} per 1,000 resident tokens "
                  "per session "
                  f"({money(p.window_cost(1_000, scope='project'))} across the "
                  f"{p.project_sessions} sessions this project has on record; "
                  f"{money(p.window_cost(1_000))} if it were in your user-level "
                  f"file, which all {p.sessions} load)."]

    if rep.floor_tokens:
        lines.append(
            f"  Measured opening context {tokens(rep.floor_tokens)}; "
            f"{tokens(rep.resident)} of it is yours ({rep.controllable_share:.0%}), "
            f"{tokens(rep.unaccounted_tokens)} is system prompt and tool schemas.")

    if rep.findings:
        lines += ["", "  Findings"]
        for f in rep.findings[:10]:
            # Findings are priced at project scope: it is the smaller of the
            # two counts, and a recoverable figure should never be the
            # flattering one.
            saved = (f"  (+{tokens(f.tokens)}, "
                     f"{money(p.window_cost(f.tokens, scope='project'))})"
                     if f.tokens else "")
            lines.append(f"    {f.kind:<18} {f.detail}{saved}")
        if rep.recoverable_tokens:
            lines.append("")
            lines.append(warn(
                f"  {tokens(rep.recoverable_tokens)} resident tokens are duplicated "
                "or over-long: "
                f"{money(p.window_cost(rep.recoverable_tokens, scope='project'))} "
                "across the sessions this project has on record."))
    return "\n".join(lines)


def _json(rep: MemoryReport, what_if: int) -> str:
    p = rep.pricing
    return json.dumps({
        "project": str(rep.project) if rep.project else None,
        "resident_tokens": rep.resident,
        "floor_tokens": rep.floor_tokens,
        "unaccounted_tokens": rep.unaccounted_tokens,
        "controllable_share": round(rep.controllable_share, 4),
        "recoverable_tokens": rep.recoverable_tokens,
        "wrong": len(rep.wrong),
        "pricing": {
            "model": p.model, "sessions": p.sessions, "turns": round(p.turns, 1),
            "read_mult": round(p.read_mult, 5), "warm_share": round(p.warm_share, 4),
            "source": p.source,
            "project_sessions": p.project_sessions,
            "per_1k_per_session": round(p.per_1k(), 6),
            "per_1k_window": round(p.window_cost(1_000), 4),
            "per_1k_window_project": round(
                p.window_cost(1_000, scope="project"), 4),
        },
        "by_kind": {k: {"files": n, "resident_tokens": t}
                    for k, (n, t) in by_kind(rep.docs).items()},
        "docs": [
            {"path": str(d.path), "name": d.name, "kind": d.kind, "scope": d.scope,
             "load": d.load, "bytes": d.bytes_, "tokens": d.tokens,
             "resident_tokens": d.resident,
             "session_cost": round(p.session_cost(d.resident), 6),
             "window_cost": round(p.window_cost(d.resident, scope=d.scope), 4),
             "window_sessions": p.sessions_for(d.scope)}
            for d in rep.ranked()
        ],
        "findings": [
            {"kind": f.kind, "detail": f.detail, "tokens": f.tokens,
             "window_cost": round(p.window_cost(f.tokens, scope="project"), 4),
             "path": str(f.path) if f.path else None}
            for f in rep.findings
        ],
        "what_if": ({"tokens": what_if,
                     "session_cost": round(p.session_cost(what_if), 6),
                     "window_cost": round(p.window_cost(what_if), 4)}
                    if what_if else None),
    })


def main(argv: list[str] | None = None) -> int:
    import argparse

    from adder.core.filters import add_arguments as add_window
    from adder.core.filters import load as load_window
    from adder.util.render import money, tokens

    ap = argparse.ArgumentParser(
        prog="adder memory",
        description="What the always-loaded prefix — CLAUDE.md, memory, skill "
                    "and agent descriptions — costs on every turn of every session.")
    add_window(ap)
    ap.add_argument("--repo", default=None, metavar="DIR",
                    help="repository to audit (default: the working directory)")
    ap.add_argument("--home", default=None, metavar="DIR",
                    help="Claude home (default: the parent of the transcript root)")
    ap.add_argument("--model", default=None,
                    help="price the floor on this model (default: your costliest)")
    ap.add_argument("--ttl", choices=("5m", "1h"), default="1h",
                    help="cache TTL to price the opening write at (default: %(default)s)")
    ap.add_argument("--top", type=int, default=12, metavar="N",
                    help="files to list (default: %(default)s)")
    ap.add_argument("--what-if", type=int, default=0, metavar="TOK",
                    help="price adding (or removing) this many resident tokens")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero on a stale reference or on more than "
                         "--max-waste tokens of duplicated resident text")
    ap.add_argument("--max-waste", type=int, default=DEFAULT_MAX_WASTE,
                    metavar="TOK",
                    help="duplication tolerated under --strict (default: %(default)s)")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    a = ap.parse_args(argv)
    # `root_of`: the argument if one was given, else the `root`
    # setting. Resolved here so two commands cannot disagree
    # about which transcript directory `adder config` names.
    a.root = str(_root_of(a))

    sessions, _ = load_window(a)
    # The `home` setting, not the transcript root's parent. Deriving it that
    # way is only right while `root` is `~/.claude/projects`; point `root` at
    # `/data/transcripts` -- which the setting exists to allow -- and `home`
    # became `/data`, so the user-level `CLAUDE.md`, the skills and the agents
    # were looked for in a directory that has none of them and reported as
    # absent. `home` is a declared setting with its own default.
    home = Path(a.home) if a.home else Path(str(_settings.get("home")))
    rep = analyse(sessions, a.repo, home=home, root=a.root, model=a.model,
                  ttl=a.ttl)

    if a.json:
        print(_json(rep, a.what_if))
        return 1 if (a.strict and rep.fails(max_waste=a.max_waste)) else 0

    print()
    print(report(rep, top=a.top))
    if a.what_if:
        p = rep.pricing
        verb = "Adding" if a.what_if > 0 else "Removing"
        print()
        print(f"  {verb} {tokens(abs(a.what_if))} of resident memory: "
              f"{money(abs(p.session_cost(a.what_if)))} per session, "
              f"{money(abs(p.window_cost(a.what_if)))} across "
              f"{p.sessions} sessions.")
    print()
    return 1 if (a.strict and rep.fails(max_waste=a.max_waste)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
