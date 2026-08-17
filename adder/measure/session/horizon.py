"""How many turns are left? -- the unobservable at the centre of the cost model.

Every placement decision multiplies by `remaining_turns`, and that number is a
forecast, not a fact. Getting it wrong is not symmetric: under-estimating makes
adder skip delegation in exactly the long sessions where debt compounds.

The obvious estimator is a countdown from a typical session length,
`median_length - turn_index`. Measured against real session lengths it is badly
wrong, and wrong in the expensive direction:

    at turn  400: countdown says 207, actual median 456   (2.2x under)
    at turn  600: countdown says   7, actual median 350   ( 50x under)
    at turn 1000: countdown says   0, actual median 309   (infinitely under)

The reason is that session length is heavy-tailed and close to memoryless: mean
remaining turns stays near 500-670 regardless of how far in you are. Reaching
turn 600 is evidence you are in a long session, not evidence you are near the end.

So this estimates remaining turns from the empirical survivor function --
`median(L - N for L in observed_lengths if L > N)` -- and falls back to a flat
prior where data is thin.

Median for display, mean for money
----------------------------------
`remaining()` returns the conditional *median*, which is the right number to
show a person: it answers "how much longer will this probably run". It is the
wrong number to multiply a cost by.

Carry cost is linear in remaining turns, so its expectation is `E[cost] =
c * E[R]`, and `E[R]` is the conditional **mean**. Session length is heavy-
tailed -- a few very long sessions hold most of the spend -- so the mean sits
well above the median. On this repo's own history the conditional mean is
roughly 1.5-2x the conditional median, and every dollar of that gap was being
left out of the carry term, which under-priced admission and under-recommended
delegation in exactly the sessions where it matters most.

So `mean_remaining()` exists and is what the cost model should call.
`remaining()` stays as it is, because a report that quotes a mean session length
to a user is quoting a number no session has.
"""

from __future__ import annotations

import statistics
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

# Fallback prior when no local history is available. Measured mean remaining
# turns is roughly flat in N, so a constant beats a countdown.
DEFAULT_REMAINING = 450

# How long a fitted horizon stays usable. An hour, because the fit is over ~100
# sessions and a session is not finished in an hour -- refitting more often
# spends 2.3 seconds to move the estimate by nothing.
MAX_AGE_S = 3_600.0
CACHE_VERSION = 1
MIN_SAMPLES = 5

# How far a conservative horizon backs off when there is nothing to measure.
# A gate using the lower bound should not be handed the same number as a gate
# using the point estimate, or the bound is decorative.
PRIOR_LOWER_FRACTION = 0.25


@dataclass
class Horizon:
    """Empirical survivor-function estimator for remaining turns."""

    lengths: list[int]

    @classmethod
    def from_sessions(cls, sessions, min_turns: int = 5) -> Horizon:
        """Session lengths, counted on the main chain.

        The number this produces is multiplied by the carry rate to price how
        many times a token admitted now will be re-read. A subagent turn does
        not re-read the main context -- that is the whole of what delegating
        bought -- so counting it inflates the re-read count for work that never
        touched the prefix.

        Four of eighty sessions here carry subagent turns, one of them at 3.5x
        (716 records for a 207-turn conversation). It moves the median remaining
        at turn 0 from 302 to 277 and the mean from 362 to 354, and the mean is
        what prices admission.
        """
        return cls(sorted(len(s.main_turns) for s in sessions.values()
                          if len(s.main_turns) >= min_turns))

    @classmethod
    def default(cls) -> Horizon:
        return cls([])

    def remaining(self, turn_index: int) -> int:
        """Median additional turns, conditioned on having reached `turn_index`."""
        if not self.lengths:
            return DEFAULT_REMAINING
        i = bisect_right(self.lengths, turn_index)
        alive = self.lengths[i:]
        if len(alive) < MIN_SAMPLES:
            # Thin tail: fall back to the prior rather than pretending precision.
            return DEFAULT_REMAINING
        return int(statistics.median(L - turn_index for L in alive))

    def survivors(self, turn_index: int) -> list[int]:
        """Observed remaining lengths, conditioned on having reached `turn_index`.

        The raw conditional sample. Everything else here is a statistic of it,
        and `risk.p_cheaper` needs the sample itself rather than a summary.
        """
        i = bisect_right(self.lengths, turn_index)
        return [L - turn_index for L in self.lengths[i:]]

    def mean_remaining(self, turn_index: int) -> float:
        """Conditional MEAN additional turns -- the one that prices carry cost.

        Cost is linear in remaining turns, so the expected cost of admitting a
        token is set by E[R], not by the median. With a heavy right tail those
        two numbers are far apart and the median is the smaller one, so using it
        under-prices admission. See the module docstring.
        """
        alive = self.survivors(turn_index)
        if len(alive) < MIN_SAMPLES:
            return float(DEFAULT_REMAINING)
        return sum(alive) / len(alive)

    def quantile_remaining(self, turn_index: int, q: float) -> float:
        """Quantile of the conditional remaining-turns distribution."""
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"quantile must be in [0,1], got {q}")
        from adder.util.stats import quantile

        alive = self.survivors(turn_index)
        if len(alive) < MIN_SAMPLES:
            return DEFAULT_REMAINING * (PRIOR_LOWER_FRACTION if q < 0.5 else 1.0)
        # `stats.quantile`, not a fifth transcription of the same interpolation.
        # Every quantile in this package is meant to be the one estimator, so
        # that `bounds()` here and `risk.empirical_bounds` cannot disagree about
        # the same sample.
        return quantile(alive, q)

    def bounds(self, turn_index: int, *, alpha: float = 0.10):
        """`risk.Interval` over remaining turns: low quantile, mean, high quantile.

        The low end is what a gate should use when a *longer* session makes the
        recommendation look better, which is every placement decision here: a
        delegation that only pays if the session runs another 400 turns is not a
        recommendation, it is a bet on the horizon estimate.
        """
        from adder.util.risk import Interval

        lo = self.quantile_remaining(turn_index, alpha / 2.0)
        hi = self.quantile_remaining(turn_index, 1.0 - alpha / 2.0)
        mean = self.mean_remaining(turn_index)
        return Interval(min(lo, mean), mean, max(hi, mean))

    def countdown(self, turn_index: int) -> int:
        """The naive estimator, kept so the error can be shown rather than asserted."""
        if not self.lengths:
            return max(0, DEFAULT_REMAINING - turn_index)
        return max(0, int(statistics.median(self.lengths)) - turn_index)

    def error_table(self, points=(10, 100, 400, 600, 1000)) -> list[tuple[int, int, int]]:
        """(turn_index, countdown_estimate, empirical_estimate)."""
        return [(n, self.countdown(n), self.remaining(n)) for n in points]


def cache_path() -> Path:
    """Where the fitted horizon is kept. Under `ADDER_HOME`, so tests redirect it."""
    try:
        from adder.core.settings import get as _setting

        return Path(str(_setting("home"))) / ".adder-horizon.json"
    except Exception:
        return Path.home() / ".claude" / ".adder-horizon.json"


def _root_key(root) -> str:
    """The transcript directory a fit was built from, as a stable string."""
    from adder.core.trace import DEFAULT_ROOT

    return str(Path(root or DEFAULT_ROOT).expanduser())


def _cached(max_age_s: float, root=None) -> Horizon | None:
    """The stored fit, if it is fresh enough AND from the same root. Never raises.

    Keyed on the root because it was not: one cache file held one fit, so
    `adder carry <some-other-directory>` -- which passes its root explicitly --
    read back a distribution fitted to whatever directory had been analysed
    first, and every carry and placement number downstream was computed from
    somebody else's session lengths. Verified: two different corpora returned
    byte-identical `lengths`.
    """
    import json
    import time

    try:
        blob = json.loads(cache_path().read_text(encoding="utf-8"))
        if blob.get("v") != CACHE_VERSION:
            return None
        if blob.get("root") != _root_key(root):
            return None
        if time.time() - float(blob.get("built") or 0) > max_age_s:
            return None
        lengths = blob.get("lengths")
        if not isinstance(lengths, list):
            return None
        return Horizon(sorted(int(x) for x in lengths))
    except (OSError, ValueError, TypeError):
        return None


def _store(h: Horizon, root=None) -> None:
    import json
    import os
    import time

    p = cache_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        # Unique per writer. Several Claude Code sessions share one machine
        # and run this from a hook, so a fixed `.tmp` name is a shared
        # mutable path: one writer's `replace` moves the file out from
        # under another's, and the loser raises FileNotFoundError into an
        # `except OSError` that drops it. Measured at 45% of writes lost
        # under three concurrent writers. `trace._cache_store` already
        # carries the pid for exactly this reason.
        tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps({"v": CACHE_VERSION, "built": time.time(),
                                       "root": _root_key(root),
                                       "lengths": h.lengths}), encoding="utf-8")
            tmp.replace(p)              # atomic: a second hook may be reading it
        finally:
            tmp.unlink(missing_ok=True)
    except OSError:
        pass


def load(root: Path | str | None = None, *, use_cache: bool = True,
         max_age_s: float | None = None) -> Horizon:
    """Build a horizon from local transcripts, or the flat prior if unavailable.

    Cached on disk, because of who calls it. `live.analyse` calls this, and
    both hooks call `analyse` -- so every prompt submission and every guarded
    read was re-fitting a distribution over every session on the machine. That
    is 2.3 seconds against 81ms warm, to move an estimate built from ~100
    sessions by at most one session.

    Staleness is bounded by time rather than by a content fingerprint on
    purpose. The transcript tree changes on every single turn, so a
    content-keyed cache would invalidate constantly and never hit, while the
    statistic it holds changes by a session a day.
    """
    if use_cache:
        hit = _cached(MAX_AGE_S if max_age_s is None else max_age_s, root)
        if hit is not None:
            return hit
    try:
        from adder.core.trace import DEFAULT_ROOT, load_sessions

        h = Horizon.from_sessions(load_sessions(root or DEFAULT_ROOT))
    except Exception:
        return Horizon.default()
    if use_cache and h.lengths:
        _store(h, root)
    return h


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="adder horizon")
    ap.add_argument("root", nargs="?", default=None)
    ap.add_argument("--json", action="store_true", help="machine-readable")
    ap.add_argument("--at", type=int, action="append", metavar="N",
                    help="turn index to report (repeatable; default 10/100/400/600/1000)")
    from adder.core.filters import root_of as _root_of

    a = ap.parse_args(argv)
    # `root_of`: the argument if one was given, else the `root`
    # setting. Resolved here so two commands cannot disagree
    # about which transcript directory `adder config` names.
    a.root = str(_root_of(a))

    h = load(a.root)
    points = tuple(a.at) if a.at else (10, 100, 400, 600, 1000)
    if a.json:
        import json

        print(json.dumps({
            "sessions": len(h.lengths),
            "median_length": int(statistics.median(h.lengths)) if h.lengths else None,
            "default_prior": DEFAULT_REMAINING,
            "at": {
                str(n): {
                    "median_remaining": h.remaining(n),
                    "mean_remaining": round(h.mean_remaining(n), 2),
                    "countdown": h.countdown(n),
                    "p05": round(h.quantile_remaining(n, 0.05), 2),
                    "p95": round(h.quantile_remaining(n, 0.95), 2),
                    "survivors": len(h.survivors(n)),
                }
                for n in points
            },
        }))
        return 0
    print(f"\n  {len(h.lengths)} sessions observed"
          f"{f'; median length {int(statistics.median(h.lengths)):,}' if h.lengths else ''}\n")
    print(f"  {'at turn':>9}{'median left':>14}{'mean left':>12}"
          f"{'ratio':>9}   the mean is what prices carry cost")
    for n in points:
        med, mean = h.remaining(n), h.mean_remaining(n)
        ratio = f"{mean / med:.2f}x" if med else "n/a"
        print(f"  {n:>9,}{med:>14,}{mean:>12,.0f}{ratio:>9}")
    print()
    print(f"  {'at turn':>9}{'countdown':>12}{'empirical':>12}{'countdown error':>18}")
    for n, cd, emp in h.error_table(points):
        if emp <= 0:
            err = "n/a"
        elif cd <= 0:
            err = "infinite"
        else:
            err = f"{emp / cd:.1f}x under" if emp > cd else f"{cd / emp:.1f}x over"
        print(f"  {n:>9,}{cd:>12,}{emp:>12,}{err:>18}")
    print("\n  Reaching a high turn count is evidence of being in a LONG session,")
    print("  not evidence of being near its end.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
