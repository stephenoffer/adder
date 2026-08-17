"""Does the tool pay for itself? -- the accounting that makes that answerable.

adder is not free to run. Every recommendation it emits costs a routing turn,
and a routing turn at 500K of context on Opus is about $0.25 before anything
useful has happened. A tool that hands out advice worth $0.10 a time, and
charges $0.25 a time to hand it out, is a more expensive way to work than not
having it -- and nothing in the reports would say so, because each report only
ever prices the advice, never the asking.

So the claim "using this is cheaper than not using it" has to be an invariant,
not a hope. Write the bill out:

    cost_with_adder = baseline - savings + overhead

which is below `baseline` exactly when `savings >= overhead`. Two mechanisms
here keep that true rather than assuming it.

**Solvency.** A recommendation is only emitted when its *worst-case* saving --
the pessimistic vertex from `risk.guarantee`, not its expected value -- exceeds
the overhead of emitting it. Every emitted recommendation therefore banks a
non-negative margin under any inputs the estimates admit, and a sum of
non-negative terms cannot go negative. The ledger records both sides so the
invariant is checkable after the fact instead of merely argued for.

**Calibration drift.** Worst-case bounds protect against the parameters being
wrong. They do not protect against the *model* being wrong -- a systematic bias
that shifts every corner of the box at once. That shows up only as realized
savings falling short of predicted ones, so it is measured directly and applied
as a haircut: if the last N verified recommendations delivered 60% of what they
promised, every future prediction is multiplied by 0.6 before it meets its gate.
A model that over-promises therefore raises its own bar until it stops, without
anyone editing a constant.

The haircut is capped at 1.0 on purpose. A model that under-promises does not
get to award itself credit for it; the error that costs money is one direction
only, and an estimator allowed to inflate itself on good news is an estimator
that will eventually inflate itself on noise.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

# The built-in location. `ledger_path()` is what callers should use: it lets
# an explicitly-configured `ledger` setting win, which this constant alone
# cannot, because it is read once at import.
DEFAULT_LEDGER = Path(
    os.environ.get("ADDER_LEDGER", Path.home() / ".claude" / "adder-ledger.jsonl")
)


def ledger_path(log: Path | str | None = None) -> Path:
    """The ledger in effect: the caller's, the `ledger` setting, or the default."""
    if log is not None:
        return Path(log)
    from adder.core.settings import configured_path

    return configured_path("ledger", DEFAULT_LEDGER)

# Verified entries needed before the haircut is allowed to move off 1.0. Below
# this, a single unlucky delegation would throttle every later recommendation.
MIN_VERIFIED = 8

# Floor on the haircut. A model measured to deliver a tenth of what it promises
# is broken, and the right response is to say so rather than to keep scaling a
# number that has stopped meaning anything.
MIN_HAIRCUT = 0.20

# Older evidence counts for less, matching `outcomes.HALF_LIFE_DAYS`.
HALF_LIFE_DAYS = 30.0

MAX_ROWS = 20_000


@dataclass
class Entry:
    """One recommendation, priced when it was made and again when it landed.

    `predicted` is what the model said the action would save. `worst` is what it
    guaranteed under the pessimistic corner. `overhead` is what emitting it
    cost. `realized` is filled in later by whatever verified it, and stays None
    for advice nobody checked -- which is most of it, and which is why the
    haircut is computed only over the entries that have it.
    """

    action: str                     # "delegate" | "downgrade" | "inline" | ...
    predicted: float                # modelled saving at the point estimates
    worst: float                    # saving at the pessimistic corner
    overhead: float                 # cost of the routing turn that emitted it
    realized: float | None = None   # measured saving, once known
    accepted: bool = True           # False when the gate declined to emit
    project: str = ""
    session: str = ""
    note: str = ""
    ts: float = field(default_factory=time.time)


_FIELDS = {f.name for f in fields(Entry)}

# Fields the arithmetic sums, sorts and divides by. Anything else in the row is
# carried as-is.
_NUMERIC = ("predicted", "worst", "overhead", "ts")


def _num(v, default: float = 0.0) -> float:
    """A finite float from a log field, or `default`. Never raises."""
    if v is None or isinstance(v, bool):
        return default
    if isinstance(v, (int, float)):
        f = float(v)
        return f if f == f and f not in (float("inf"), float("-inf")) else default
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            pass
        # Every other timestamp in this repo is an ISO string, so a caller
        # writing one into `ts` is a matter of time. `outcomes._coerce_ts`
        # already makes this correction and explains why; the ledger carries
        # the identical field and did not, so one hand-edited row made
        # `prune` raise on a sort and `promised` raise on a sum -- both from
        # inside handlers that swallow the exception, so the symptom was the
        # ledger silently ceasing to influence anything.
        from datetime import datetime

        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return default
    return default


def _coerce(d: dict) -> dict:
    """One log row, with the numbers made numbers."""
    out = {k: v for k, v in d.items() if k in _FIELDS}
    for k in _NUMERIC:
        if k in out:
            out[k] = _num(out[k])
    if out.get("realized") is not None:
        out["realized"] = _num(out["realized"])
    for k in ("action", "project", "session", "note"):
        if k in out:
            out[k] = str(out[k] or "")
    if "accepted" in out:
        out["accepted"] = bool(out["accepted"])
    return out


def record(entry: Entry, log: Path | str | None = None) -> None:
    """Append one entry. Never raises: accounting must not break routing."""
    try:
        p = ledger_path(log)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(entry), separators=(",", ":")) + "\n")
    except (OSError, TypeError, ValueError):
        pass


def load(log: Path | str | None = None) -> list[Entry]:
    """Read the ledger, skipping anything malformed and tolerating new fields."""
    p = ledger_path(log)
    if not p.exists():
        return []
    out: list[Entry] = []
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            if not isinstance(d, dict):
                continue
            out.append(Entry(**_coerce(d)))
        except (ValueError, TypeError):
            continue
    return out


def _weight(ts: float, now: float) -> float:
    return 0.5 ** (max(0.0, (now - ts) / 86400.0) / HALF_LIFE_DAYS)


@dataclass(frozen=True)
class Ledger:
    """The running account, and the two numbers a gate should ask it for."""

    entries: list[Entry] = field(default_factory=list)

    @classmethod
    def load(cls, log: Path | str | None = None) -> Ledger:
        return cls(load(log))

    @property
    def accepted(self) -> list[Entry]:
        return [e for e in self.entries if e.accepted]

    @property
    def verified(self) -> list[Entry]:
        return [e for e in self.accepted if e.realized is not None]

    @property
    def banked(self) -> float:
        """Guaranteed saving across every recommendation acted on.

        The worst-case number, not the expected one. This is the quantity the
        solvency invariant is stated over, because it is the only one that
        cannot be wrong in the direction that costs money.
        """
        return sum(e.worst for e in self.accepted)

    @property
    def promised(self) -> float:
        return sum(e.predicted for e in self.accepted)

    @property
    def spent(self) -> float:
        return sum(e.overhead for e in self.accepted)

    @property
    def delivered(self) -> float:
        return sum(e.realized or 0.0 for e in self.verified)

    @property
    def solvent(self) -> bool:
        """Has the advice been worth more than the asking, guaranteed?"""
        return self.banked >= self.spent

    @property
    def margin(self) -> float:
        return self.banked - self.spent

    def haircut(self, *, now: float | None = None) -> float:
        """Multiplier applied to a prediction before it meets its gate.

        `sum(realized) / sum(predicted)` over verified entries, recency-weighted,
        capped at 1.0 and floored at `MIN_HAIRCUT`. Returns 1.0 -- no haircut --
        until there is enough verified history for the ratio to mean anything,
        because throttling every recommendation on the strength of two
        observations is its own kind of expensive.
        """
        v = self.verified
        if len(v) < MIN_VERIFIED:
            return 1.0
        now = now if now is not None else time.time()
        num = sum(_weight(e.ts, now) * (e.realized or 0.0) for e in v)
        den = sum(_weight(e.ts, now) * e.predicted for e in v)
        if den <= 0:
            return 1.0
        return max(MIN_HAIRCUT, min(1.0, num / den))

    def describe(self) -> str:
        if not self.accepted:
            return "no recommendations acted on yet; nothing has been spent or saved"
        state = "solvent" if self.solvent else "INSOLVENT"
        h = self.haircut()
        tail = "" if h >= 1.0 else f"; predictions haircut to {h:.0%} of face value"
        return (f"{len(self.accepted)} recommendations: guaranteed ${self.banked:,.2f} "
                f"against ${self.spent:,.2f} of routing overhead -- {state} "
                f"by ${self.margin:,.2f}{tail}")


def prune(log: Path | str | None = None, keep: int = MAX_ROWS) -> int:
    """Trim the ledger to its most recent `keep` rows. Returns rows dropped."""
    rows = load(log)
    if len(rows) <= keep:
        return 0
    rows.sort(key=lambda e: e.ts)
    kept = rows[-keep:]
    try:
        p = ledger_path(log)
        # Unique per writer. Several Claude Code sessions share one machine
        # and run this from a hook, so a fixed `.tmp` name is a shared
        # mutable path: one writer's `replace` moves the file out from
        # under another's, and the loser raises FileNotFoundError into an
        # `except OSError` that drops it. Measured at 45% of writes lost
        # under three concurrent writers. `trace._cache_store` already
        # carries the pid for exactly this reason.
        tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                for e in kept:
                    fh.write(json.dumps(asdict(e), separators=(",", ":")) + "\n")
            tmp.replace(p)
        finally:
            tmp.unlink(missing_ok=True)
    except OSError:
        return 0
    return len(rows) - len(kept)


def current(log: Path | str | None = None) -> Ledger:
    """The ledger, or an empty one. Never raises -- routing must not depend on it."""
    try:
        return Ledger.load(log)
    except Exception:                                     # pragma: no cover
        return Ledger([])


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="adder ledger",
        description="Has adder's advice been worth more than the asking?")
    ap.add_argument("--log", default=None)
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    if a.prune:
        print(f"  dropped {prune(a.log)} old rows")
        return 0

    led = current(a.log)
    if a.json:
        print(json.dumps({
            "recommendations": len(led.accepted),
            "verified": len(led.verified),
            "banked": round(led.banked, 4),
            "promised": round(led.promised, 4),
            "delivered": round(led.delivered, 4),
            "spent": round(led.spent, 4),
            "margin": round(led.margin, 4),
            "solvent": led.solvent,
            "haircut": round(led.haircut(), 4),
        }))
        return 0

    print()
    if not led.accepted:
        print(f"  Nothing recorded yet ({ledger_path(a.log)}).")
        print("  Until adder has emitted a recommendation there is no overhead to")
        print("  have earned back, and the invariant holds trivially.\n")
        return 0

    print(f"  {'recommendations acted on':<34}{len(led.accepted):>12,}")
    print(f"  {'guaranteed saving (worst case)':<34}${led.banked:>11,.2f}")
    print(f"  {'modelled saving (expected)':<34}${led.promised:>11,.2f}")
    print(f"  {'routing overhead paid':<34}${led.spent:>11,.2f}")
    print(f"  {'margin':<34}${led.margin:>11,.2f}")
    print()
    if led.verified:
        h = led.haircut()
        print(f"  {len(led.verified)} verified: delivered ${led.delivered:,.2f} against "
              f"${sum(e.predicted for e in led.verified):,.2f} predicted")
        print(f"  Calibration haircut: {h:.0%} -- every future prediction is scaled by")
        print("  this before it has to clear its own overhead.")
    else:
        print("  Nothing verified yet, so predictions run at face value.")
    print()
    print(f"  {led.describe()}")
    if not led.solvent:
        print("  The gate should be refusing to emit; if it is not, that is a bug.")
    print()
    return 0 if led.solvent else 1


if __name__ == "__main__":
    raise SystemExit(main())
