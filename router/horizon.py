"""How many turns are left? -- the unobservable at the centre of the cost model.

Every placement decision multiplies by `remaining_turns`, and that number is a
forecast, not a fact. Getting it wrong is not symmetric: under-estimating makes
the router skip delegation in exactly the long sessions where debt compounds.

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
"""

from __future__ import annotations

import statistics
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

# Fallback prior when no local history is available. Measured mean remaining
# turns is roughly flat in N, so a constant beats a countdown.
DEFAULT_REMAINING = 450
MIN_SAMPLES = 5


@dataclass
class Horizon:
    """Empirical survivor-function estimator for remaining turns."""

    lengths: list[int]

    @classmethod
    def from_sessions(cls, sessions, min_turns: int = 5) -> "Horizon":
        return cls(sorted(len(s.turns) for s in sessions.values()
                          if len(s.turns) >= min_turns))

    @classmethod
    def default(cls) -> "Horizon":
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

    def countdown(self, turn_index: int) -> int:
        """The naive estimator, kept so the error can be shown rather than asserted."""
        if not self.lengths:
            return max(0, DEFAULT_REMAINING - turn_index)
        return max(0, int(statistics.median(self.lengths)) - turn_index)

    def error_table(self, points=(10, 100, 400, 600, 1000)) -> list[tuple[int, int, int]]:
        """(turn_index, countdown_estimate, empirical_estimate)."""
        return [(n, self.countdown(n), self.remaining(n)) for n in points]


def load(root: Path | str | None = None) -> Horizon:
    """Build a horizon from local transcripts, or the flat prior if unavailable."""
    try:
        from .trace import DEFAULT_ROOT, load_sessions

        return Horizon.from_sessions(load_sessions(root or DEFAULT_ROOT))
    except Exception:
        return Horizon.default()


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="router.horizon")
    ap.add_argument("root", nargs="?", default=None)
    a = ap.parse_args(argv)

    h = load(a.root)
    print(f"\n  {len(h.lengths)} sessions observed"
          f"{f'; median length {int(statistics.median(h.lengths)):,}' if h.lengths else ''}\n")
    print(f"  {'at turn':>9}{'countdown':>12}{'empirical':>12}{'countdown error':>18}")
    for n, cd, emp in h.error_table():
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
