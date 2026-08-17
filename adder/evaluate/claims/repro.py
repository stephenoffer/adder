"""A manifest of everything a number depended on, so it can be re-derived.

The failure this exists to stop
-------------------------------
Someone reports a 6.1x improvement. Three weeks later it is 4.2x. Nobody can
say why, and there are four candidates: the transcripts grew, the price table
changed, the catalog was refreshed, or the code changed. Without a record of
which of those was true when the first number was produced, the disagreement is
unresolvable and both numbers become untrustworthy.

Every claim this tool makes is a function of four inputs and nothing else:

1. **the transcripts** -- what was measured;
2. **the price table** -- hand-maintained, first-party, date-aware;
3. **the catalog** -- refreshed from public sources, and stale by some number
   of days;
4. **the code** -- this package.

So a manifest hashes all four. Re-run later, diff the manifests, and the reason
a number moved is a line of output rather than an afternoon.

Reproducible means byte-identical, not "about the same"
--------------------------------------------------------
Two properties are enforced rather than hoped for:

* **No wall clock in the fingerprint.** The manifest records when it was made,
  but that field is outside the digest. A digest containing a timestamp differs
  from itself one second later, which makes the whole exercise decorative.
* **Deterministic ordering.** Files are hashed in sorted order and dictionaries
  are serialised with sorted keys, so the same inputs give the same digest on
  any machine and any filesystem.

Two ways to fingerprint the transcripts, and the difference matters
--------------------------------------------------------------------
The default digests each file's `(relative path, size)`. It is fast enough to
run on every report and it catches the thing that actually happens: files
appended to, added, or removed.

`--deep` digests the bytes. It is the one to use when comparing two machines or
two checkouts, because size collides more often than you would like -- an edit
that replaces one model name with another of the same length does not change
any file's size, and the metadata digest will call that identical.

Modification times are deliberately **not** in either digest. They are not
preserved by copying, cloning, or archiving, so including them would report
drift between two byte-identical copies of the same data -- a false alarm that
would train people to ignore the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path

from adder import __version__
from adder.core.filters import root_of as _root_of
from adder.core.trace import DEFAULT_ROOT, transcripts
from adder.util import render

# Read in chunks: a transcript directory is routinely gigabytes, and reading a
# 400MB JSONL into memory to hash it is a way to make a diagnostic tool the
# reason a machine swapped.
CHUNK = 1 << 20


def _digest(parts) -> str:
    """SHA-256 over an ordered sequence of strings. Sorted by the caller."""
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8", "replace"))
        h.update(b"\0")
    return h.hexdigest()


def hash_file(path: Path) -> str:
    """SHA-256 of a file's bytes, or `""` if it cannot be read.

    Unreadable is not fatal. A manifest that refuses to be produced because one
    file is locked is a manifest nobody generates.
    """
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            while chunk := fh.read(CHUNK):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


@dataclass
class Fingerprint:
    """One input's identity, with enough detail to say how it differs."""

    name: str
    digest: str
    files: int = 0
    bytes: int = 0
    detail: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        out = {"digest": self.digest, "files": self.files, "bytes": self.bytes}
        out.update(self.detail)
        return out


def fingerprint_transcripts(root: Path | str = DEFAULT_ROOT, *,
                            deep: bool = False) -> Fingerprint:
    """Identify the measured data. Content-hashed only when asked."""
    root = Path(root)
    files = sorted(transcripts(root))
    parts: list[str] = []
    total = 0
    for p in files:
        try:
            size = p.stat().st_size
        except OSError:
            continue
        total += size
        rel = p.relative_to(root) if p.is_relative_to(root) else p.name
        parts.append(f"{rel}|{size}" + (f"|{hash_file(p)}" if deep else ""))
    return Fingerprint("transcripts", _digest(parts), len(files), total,
                       {"deep": deep, "root": str(root)})


def fingerprint_code() -> Fingerprint:
    """Identify the package that produced the number.

    Hashes the source rather than trusting `__version__`: the version is bumped
    at release, and the number in question was very likely produced by a working
    tree that sits between two releases.
    """
    pkg = Path(__file__).resolve().parent.parent.parent
    files = sorted(pkg.rglob("*.py"))
    parts = [f"{p.relative_to(pkg)}|{hash_file(p)}" for p in files
             if "__pycache__" not in p.parts]
    return Fingerprint("code", _digest(parts), len(parts), 0,
                       {"version": __version__})


def fingerprint_prices() -> Fingerprint:
    """Identify the hand-maintained rate table.

    Hashed as data, not as a file: the digest is over the resolved rates, so a
    comment change does not read as a price change and a reformat does not
    invalidate a comparison.
    """
    from adder.pricing import prices

    table = getattr(prices, "MODELS", None)
    if table is None:
        return Fingerprint("prices", _digest([hash_file(Path(prices.__file__))]))
    parts = [json.dumps(_plain(table), sort_keys=True, default=str)]
    return Fingerprint("prices", _digest(parts), 1, 0,
                       {"models": _count_keys(table)})


def fingerprint_catalog() -> Fingerprint:
    """Identify the catalog, and how stale it is.

    Staleness is recorded as a number rather than a vibe: a catalog nobody has
    refreshed in three months should visibly degrade a recommendation, and a
    manifest that says only "the catalog changed" cannot tell you whether the
    old one was already out of date.
    """
    from adder.pricing.catalog import load

    cat = load()
    entries = sorted(cat, key=lambda e: e.key)
    parts = [
        f"{e.key}|{e.inp}|{e.out}|{e.cache_read}|{e.cache_write}|"
        f"{sorted(e.elo.items())}|{e.context}"
        for e in entries
    ]
    ages = [a for a in (e.age_days() for e in entries) if isinstance(a, (int, float))]
    return Fingerprint("catalog", _digest(parts), len(entries), 0,
                       {"oldest_days": max(ages) if ages else None,
                        "newest_days": min(ages) if ages else None})


def _plain(obj):
    """Make a nested structure JSON-serialisable without losing ordering."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def _count_keys(obj) -> int:
    return len(obj) if hasattr(obj, "__len__") else 0


def manifest(root: Path | str = DEFAULT_ROOT, *, deep: bool = False,
             command: str = "") -> dict:
    """Every input's fingerprint, plus an environment block outside the digest.

    The split is the point. `inputs` is what the numbers are a function of, and
    its digest is what two runs are compared on. `environment` records the
    machine and the moment, which are useful when explaining a difference and
    must never *cause* one.
    """
    prints = [
        fingerprint_transcripts(root, deep=deep),
        fingerprint_prices(),
        fingerprint_catalog(),
        fingerprint_code(),
    ]
    inputs = {f.name: f.to_json() for f in prints}
    return {
        "schema": 1,
        "inputs": inputs,
        "digest": _digest([f"{f.name}:{f.digest}" for f in prints]),
        "environment": {
            "adder": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "command": command,
        },
    }


@dataclass(frozen=True)
class Drift:
    """One input that changed between two manifests."""

    name: str
    was: str
    now: str
    note: str = ""


def compare(old: dict, new: dict) -> list[Drift]:
    """What differs between two manifests, in the order worth reading.

    Ordered so the most likely explanation comes first. Code changing explains
    a moved number completely; transcripts growing explains it partly; a catalog
    refresh usually explains a routing recommendation rather than a measurement.
    """
    order = ("code", "prices", "catalog", "transcripts")
    old_in = old.get("inputs", {})
    new_in = new.get("inputs", {})
    out: list[Drift] = []
    for name in order:
        a = old_in.get(name, {})
        b = new_in.get(name, {})
        if a.get("digest") == b.get("digest"):
            continue
        note = ""
        if name == "transcripts":
            da = int(a.get("files", 0) or 0)
            db = int(b.get("files", 0) or 0)
            ba = int(a.get("bytes", 0) or 0)
            bb = int(b.get("bytes", 0) or 0)
            note = f"{db - da:+,} files, {bb - ba:+,} bytes"
            if not a.get("deep") or not b.get("deep"):
                note += " (metadata digest; use --deep to compare bytes)"
        elif name == "code":
            note = f"{a.get('version', '?')} → {b.get('version', '?')}"
        out.append(Drift(name, str(a.get("digest", ""))[:12],
                         str(b.get("digest", ""))[:12], note))
    return out


def report(man: dict, *, against: dict | None = None) -> str:
    out: list[str] = []
    out += render.heading("reproducibility manifest", rule="=")
    out.append(render.kv("inputs digest", man.get("digest", "")[:16]))
    for name, block in sorted(man.get("inputs", {}).items()):
        detail = ""
        if name == "transcripts":
            detail = (f"{block.get('files', 0):,} files, "
                      f"{block.get('bytes', 0) / 1e6:,.1f} MB"
                      + ("" if block.get("deep") else ", metadata only"))
        elif name == "code":
            detail = f"{block.get('files', 0)} modules, v{block.get('version', '?')}"
        elif name == "catalog":
            oldest = block.get("oldest_days")
            detail = f"{block.get('files', 0)} models"
            if isinstance(oldest, (int, float)):
                detail += f", oldest entry {oldest:.0f}d"
        out.append(render.kv(f"  {name}", f"{block.get('digest', '')[:12]}  {detail}"))

    env = man.get("environment", {})
    out.append("")
    out.append(render.kv("python", env.get("python", "")))
    out.append(render.kv("platform", str(env.get("platform", ""))[:48]))

    if against is not None:
        out.append("")
        out += render.heading("against the recorded manifest")
        drifts = compare(against, man)
        if not drifts:
            out.append("  Identical. Any number that moved did not move because "
                       "of an input.")
        else:
            out += render.table(
                [[d.name, d.was, d.now, d.note] for d in drifts],
                ["input", "was", "now", "how"],
                align="<<<<",
            )
            out.append("")
            out += render.wrap(
                f"{drifts[0].name} changed, and it is listed first because it is "
                "the most complete explanation available for a number that moved.")
    out.append("")
    out += render.wrap(
        "The digest covers inputs only. Time, hostname and platform are recorded "
        "but excluded, so two runs over the same data agree byte for byte.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="adder repro",
        description="Fingerprint everything a report depends on, and diff it "
                    "against a manifest recorded earlier.",
    )
    ap.add_argument("root", nargs="?", default=None,
                    help="transcript directory (default: %(default)s)")
    ap.add_argument("--deep", action="store_true",
                    help="hash transcript bytes, not just names and sizes")
    ap.add_argument("--check", type=Path, default=None, metavar="PATH",
                    help="compare against a manifest written earlier")
    ap.add_argument("--write", type=Path, default=None, metavar="PATH",
                    help="write the manifest to PATH")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    # `root_of`: the argument if one was given, else the `root`
    # setting. Resolved here so two commands cannot disagree
    # about which transcript directory `adder config` names.
    args.root = str(_root_of(args))

    man = manifest(args.root, deep=args.deep,
                   command=" ".join(sys.argv[1:]) if len(sys.argv) > 1 else "")

    recorded = None
    if args.check is not None:
        try:
            recorded = json.loads(args.check.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"adder repro: cannot read {args.check}: {exc}", file=sys.stderr)
            return 2

    if args.write is not None:
        try:
            args.write.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")
        except OSError as exc:
            print(f"adder repro: cannot write {args.write}: {exc}", file=sys.stderr)
            return 2

    if args.json:
        payload = dict(man)
        if recorded is not None:
            payload["drift"] = [
                {"input": d.name, "was": d.was, "now": d.now, "note": d.note}
                for d in compare(recorded, man)
            ]
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(report(man, against=recorded))

    # A drift check that always exits 0 is a check nobody wires into CI.
    if recorded is not None and compare(recorded, man):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
