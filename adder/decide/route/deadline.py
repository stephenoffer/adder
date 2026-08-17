"""When the cheap-but-slow path is worth using, given a deadline.

The discount that is not a discount
------------------------------------
Batch processing is half price. That is a real, published, first-party
discount, and it is the largest single price lever available -- larger than any
model swap this tool will ever recommend. `adder` has known about it since the
cost model was written and has never recommended it, for one reason: the
discount is paid for in latency, and nothing here knew what a deadline was.

The trade is not "cheaper but slower". It is **cheaper but uncertain**. The
cheap path returns work at a rate you do not control and cannot predict, and it
may return nothing at all for a stretch. Against a deadline that turns a
discount into a risk, and a risk needs a policy rather than a preference.

The policy
----------
The obvious two are both bad. *Always batch* is cheapest and misses deadlines.
*Always interactive* never misses and pays full price for work nobody needed
this hour.

The greedy middle -- batch until the slack runs out, then sprint -- is the one
worth being careful about, because **it is optimal under one common assumption
and bad under another**, and which you are in decides the answer:

* If the guaranteed path can absorb the entire remaining queue at once (true of
  an API you can fan out against), greedy wins outright. It collects the whole
  discount and the last-step sprint always rescues it. Measured on a 200-unit
  queue over 24 steps, greedy costs $100.52 against the proportional policy's
  $130.35, and both meet every deadline.
* If the guaranteed path is rate-limited -- a quota, a concurrency cap, a human
  in the loop -- greedy concentrates every expensive unit into the window where
  it has the least capacity to place them, and it starts missing deadlines.

So this module does not pick a favourite. It prices all four and names the
cheapest that meets the deadline, which is the only recommendation that
survives both cases.

The policy implemented here keeps **progress proportional to elapsed time**. At
each step, compare the work completed against the work that should have been
completed by now if it were spread evenly across the window:

    required(t) = total_units * t / horizon

Ahead of that line, use the cheap path and accept the uncertainty. Behind it,
use the guaranteed path until back on the line. It has no tuning parameter --
no slack fraction, no risk tolerance, no threshold to fit -- which matters
because a parameter is a thing that gets set once from one workload and is then
wrong everywhere else.

One addition to that rule is load-bearing and was not obvious until the tests
caught it. Falling behind the line is recoverable; running out of the steps in
which the guaranteed path could still finish is not. So the switch has to
happen *before* that point, not at it:

    if guaranteed_steps_needed(remaining) >= steps_left: use the guaranteed path

Without that override the policy is a heuristic that usually finishes. With it,
the deadline is a guarantee whenever the guaranteed path could have met it
alone -- which is the only version worth recommending, because the entire
purpose of the cheap path is to be abandoned safely.

Its weakness is worth stating, since it decides when not to use it: when the
cheap path is available only in short bursts, the rule keeps switching to catch
each one, and switching is not free. A workload of that shape should set
`--min-batch-run` and accept a smaller discount for fewer changeovers.

What this is not
----------------
It is not a claim about how fast the cheap path actually is on your account.
Throughput and stall rate are inputs. The defaults are deliberately pessimistic,
and the report prints the break-even values so you can see how wrong the guess
has to be before the answer changes. If you have run batch work, measure the
two numbers and pass them; if you have not, treat the output as a bound rather
than a forecast.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field

from adder.util import render
from adder.util.stats import DEFAULT_SEED, mean, quantile

# Deliberately pessimistic defaults, so a workload that clears the bar here
# clears it on any account. A cheap path that delivers a third of the queue in
# an hour, and stalls one hour in three, is worse than what is typically
# observed -- which is the point of a default nobody measured.
DEFAULT_BATCH_THROUGHPUT = 0.34     # share of the remaining queue per step
DEFAULT_STALL_RATE = 0.33           # share of steps that return nothing
DEFAULT_TRIALS = 400


@dataclass(frozen=True)
class Workload:
    """A queue of deferrable work, and the window it has to finish in."""

    units: int = 100
    horizon: int = 12                    # steps (hours, typically) until the deadline
    cost_cheap: float = 0.50             # USD per unit on the batch path
    cost_guaranteed: float = 1.00        # USD per unit on the interactive path
    batch_throughput: float = DEFAULT_BATCH_THROUGHPUT
    stall_rate: float = DEFAULT_STALL_RATE
    # Interactive throughput is a share of the remaining queue per step. 1.0
    # means "the guaranteed path can always finish the rest right now", which
    # is true for API work and false for anything rate-limited.
    guaranteed_throughput: float = 1.0
    # Minimum consecutive steps to stay on the cheap path once switched to it.
    # Switching is not free when a task carries context; this is the knob for
    # workloads where the cheap path arrives in short bursts.
    min_batch_run: int = 1

    def __post_init__(self) -> None:
        if self.units < 0:
            raise ValueError(f"units must be >= 0, got {self.units}")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        for name in ("batch_throughput", "stall_rate", "guaranteed_throughput"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0,1], got {v}")
        if self.cost_cheap < 0 or self.cost_guaranteed < 0:
            raise ValueError("costs must be non-negative")

    @property
    def discount(self) -> float:
        """How much cheaper the uncertain path is, as a share."""
        if self.cost_guaranteed <= 0:
            return 0.0
        return max(0.0, 1.0 - self.cost_cheap / self.cost_guaranteed)

    @property
    def all_guaranteed_cost(self) -> float:
        return self.units * self.cost_guaranteed

    @property
    def floor_cost(self) -> float:
        """What the work would cost if the cheap path never stalled."""
        return self.units * self.cost_cheap


@dataclass
class Run:
    """One simulated pass through the window."""

    cheap_units: int = 0
    guaranteed_units: int = 0
    unfinished: int = 0
    switches: int = 0
    cost: float = 0.0

    @property
    def met(self) -> bool:
        return self.unfinished == 0


def _step_cheap(remaining: int, w: Workload, rng: random.Random) -> int:
    """Units the cheap path returns this step. Zero on a stall."""
    if rng.random() < w.stall_rate:
        return 0
    return min(remaining, round(remaining * w.batch_throughput))


def _step_guaranteed(remaining: int, w: Workload) -> int:
    return min(remaining, max(1, round(remaining * w.guaranteed_throughput)))


def _guaranteed_steps_needed(remaining: int, w: Workload) -> int:
    """Steps the guaranteed path needs to clear `remaining` on its own.

    One step when it can take the whole queue at once, more when it is rate
    limited. This is what makes the deadline a guarantee rather than a hope:
    the policy has to switch *before* the guaranteed path can no longer finish,
    not at the moment it notices it is behind.
    """
    steps = 0
    left = remaining
    while left > 0 and steps <= w.horizon:
        left -= _step_guaranteed(left, w)
        steps += 1
    return steps


def simulate(w: Workload, policy: str, *, seed: int = DEFAULT_SEED) -> Run:
    """Run the window once under `policy`.

    Policies:

    * `cheap` -- always the uncertain path. Cheapest, misses deadlines.
    * `guaranteed` -- always the certain path. Never misses, never discounts.
    * `greedy` -- cheap until the remaining work can only just be finished by
      the guaranteed path, then sprint. Concentrates every expensive unit at the
      end, which is where a stall hurts most.
    * `uniform` -- keep completed work proportional to elapsed time. Spreads the
      guaranteed work evenly, which costs more than greedy when the guaranteed
      path is instant and less when it is rate-limited.
    """
    if policy not in ("cheap", "guaranteed", "greedy", "uniform"):
        raise ValueError(f"unknown policy {policy!r}")
    rng = random.Random(seed)
    run = Run()
    remaining = w.units
    # `None` until the first step chooses, so the minimum-run guard cannot fire
    # before there is a run to protect -- which otherwise forces every policy,
    # including the guaranteed one, onto the cheap path for its first step.
    on_cheap: bool | None = None
    run_length = 0

    for t in range(1, w.horizon + 1):
        if remaining <= 0:
            break
        steps_left = w.horizon - t + 1
        must_sprint = _guaranteed_steps_needed(remaining, w) >= steps_left

        if policy == "cheap":
            want_cheap = True
        elif policy == "guaranteed":
            want_cheap = False
        elif policy == "greedy":
            # Stay cheap while the guaranteed path could still finish the rest
            # inside the steps that are left.
            want_cheap = not must_sprint
        else:
            done = w.units - remaining
            required = w.units * (t - 1) / w.horizon
            # The safety override is what makes this a guarantee. Falling
            # behind the line is recoverable; running out of steps in which the
            # guaranteed path could still finish is not, so the switch has to
            # happen before that point rather than at it.
            want_cheap = (done >= required) and not must_sprint

        # Honour a minimum run on the cheap path, so a bursty supply does not
        # produce a switch every step. Never at the expense of the deadline.
        if (on_cheap and not want_cheap and run_length < w.min_batch_run
                and not must_sprint):
            want_cheap = True
        if on_cheap is None:
            on_cheap = want_cheap
        elif want_cheap != on_cheap:
            run.switches += 1
            on_cheap = want_cheap
            run_length = 0
        run_length += 1

        if on_cheap:
            got = _step_cheap(remaining, w, rng)
            run.cheap_units += got
            run.cost += got * w.cost_cheap
        else:
            got = _step_guaranteed(remaining, w)
            run.guaranteed_units += got
            run.cost += got * w.cost_guaranteed
        remaining -= got

    run.unfinished = remaining
    # Work that missed the deadline still has to be done, and the honest way to
    # price a miss is to charge for finishing it at full rate. Otherwise the
    # cheapest policy is always the one that gives up.
    run.cost += remaining * w.cost_guaranteed
    return run


@dataclass
class Outcome:
    """What a policy does across many windows, not in one lucky one."""

    policy: str
    cost_mean: float = 0.0
    cost_p90: float = 0.0
    met_rate: float = 0.0
    cheap_share: float = 0.0
    switches: float = 0.0
    runs: list[Run] = field(default_factory=list)

    @property
    def saving_vs_guaranteed(self) -> float:
        return 0.0

    def to_json(self, all_guaranteed: float) -> dict:
        return {
            "policy": self.policy,
            "cost_mean_usd": self.cost_mean,
            "cost_p90_usd": self.cost_p90,
            "deadline_met_rate": self.met_rate,
            "cheap_share": self.cheap_share,
            "switches_mean": self.switches,
            "saving_vs_guaranteed_usd": all_guaranteed - self.cost_mean,
        }


def evaluate(w: Workload, policy: str, *, trials: int = DEFAULT_TRIALS,
             seed: int = DEFAULT_SEED) -> Outcome:
    """Average a policy over many independent windows.

    One window tells you nothing: the cheap path's stalls are the whole
    question, and a single seed either hits them or does not. Trials are seeded
    from a fixed base so two runs of this report agree exactly.
    """
    runs = [simulate(w, policy, seed=seed + i) for i in range(max(1, trials))]
    costs = [r.cost for r in runs]
    total_units = max(1, w.units)
    return Outcome(
        policy=policy,
        cost_mean=mean(costs),
        cost_p90=quantile(costs, 0.9),
        met_rate=sum(1 for r in runs if r.met) / len(runs),
        cheap_share=mean([r.cheap_units / total_units for r in runs]),
        switches=mean([float(r.switches) for r in runs]),
        runs=runs,
    )


POLICIES = ("guaranteed", "cheap", "greedy", "uniform")


def compare(w: Workload, *, trials: int = DEFAULT_TRIALS,
            seed: int = DEFAULT_SEED) -> list[Outcome]:
    return [evaluate(w, p, trials=trials, seed=seed) for p in POLICIES]


def breakeven_horizon(w: Workload, *, trials: int = 120,
                      seed: int = DEFAULT_SEED) -> int:
    """Shortest window in which the policy still beats paying full price.

    Below this, the deadline is too tight for the cheap path to contribute and
    the correct answer is "just run it": a discount you cannot collect is not a
    discount, and recommending one is how a cost tool loses a deadline for
    someone.
    """
    from dataclasses import replace

    for h in range(1, w.horizon + 1):
        trial = replace(w, horizon=h)
        out = evaluate(trial, "uniform", trials=trials, seed=seed)
        if out.met_rate >= 0.99 and out.cost_mean < trial.all_guaranteed_cost * 0.99:
            return h
    return w.horizon + 1


def report(w: Workload, *, trials: int = DEFAULT_TRIALS,
           seed: int = DEFAULT_SEED) -> str:
    outs = compare(w, trials=trials, seed=seed)
    out: list[str] = []
    out += render.heading("deadline — when the cheap path is worth the risk",
                          rule="=")
    out.append(render.kv("queue", f"{w.units:,} units in {w.horizon} steps"))
    out.append(render.kv("price", f"{render.money(w.cost_cheap)} cheap / "
                                  f"{render.money(w.cost_guaranteed)} guaranteed "
                                  f"({render.pct(w.discount)} off)"))
    out.append(render.kv("cheap path", f"{render.pct(w.batch_throughput)} of the queue "
                                       f"per step, stalls {render.pct(w.stall_rate)}"))
    out.append("")

    out += render.table(
        [[o.policy, render.money(o.cost_mean), render.money(o.cost_p90),
          render.pct(o.met_rate), render.pct(o.cheap_share), f"{o.switches:.1f}"]
         for o in outs],
        ["policy", "cost", "p90 cost", "deadline met", "on cheap", "switches"],
        align="<>>>>>",
    )

    guaranteed = next(o for o in outs if o.policy == "guaranteed")
    cheap = next(o for o in outs if o.policy == "cheap")
    qualifying = [o for o in outs if o.met_rate >= 0.99]
    winner = min(qualifying, key=lambda o: o.cost_mean) if qualifying else guaranteed
    saved = guaranteed.cost_mean - winner.cost_mean

    out.append("")
    if winner.policy == "guaranteed":
        out += render.wrap(
            "Nothing beats paying full price here while still meeting the "
            "deadline. The window is too tight for the cheap path to contribute, "
            "so run it and stop optimising.")
    else:
        out += render.wrap(
            f"Cheapest policy that meets every deadline: **{winner.policy}**, at "
            f"{render.money(winner.cost_mean)} against "
            f"{render.money(guaranteed.cost_mean)} — a saving of "
            f"{render.money(saved)} ({render.pct(saved / guaranteed.cost_mean)}), "
            f"with {render.pct(winner.cheap_share)} of the work on the cheap path.")

    runner_up = [o for o in qualifying if o.policy != winner.policy]
    if runner_up:
        second = min(runner_up, key=lambda o: o.cost_mean)
        gap = second.cost_mean - winner.cost_mean
        if gap > 0:
            out += render.wrap(
                f"{second.policy} also meets every deadline and costs "
                f"{render.money(gap)} more. Which of the two wins depends on "
                "whether the guaranteed path can absorb the whole remaining "
                "queue at once: if it is rate-limited, prefer the proportional "
                "policy, because greedy leaves all its expensive work until the "
                "step with the least capacity to place it.")

    if cheap.met_rate < 0.99:
        out += render.wrap(
            f"Going all-in on the cheap path finishes only "
            f"{render.pct(cheap.met_rate)} of windows, and once the unfinished "
            "work is priced at full rate it costs "
            f"{render.money(cheap.cost_mean)} — the naive discount is not the "
            "one you collect.")

    be = breakeven_horizon(w, seed=seed)
    if be <= w.horizon:
        out.append("")
        out.append(render.kv("shortest window", f"{be} steps"))
    else:
        out.append("")
        out += render.wrap(
            "No window inside this horizon collects the discount safely.")

    out.append("")
    out += render.wrap(
        "MODELLED: throughput and stall rate are inputs, not measurements of "
        "your account. The defaults are pessimistic on purpose — a workload "
        "that clears the bar here clears it anywhere.")
    return "\n".join(out)


def to_json(w: Workload, *, trials: int = DEFAULT_TRIALS,
            seed: int = DEFAULT_SEED) -> dict:
    outs = compare(w, trials=trials, seed=seed)
    return {
        "workload": {
            "units": w.units, "horizon": w.horizon,
            "cost_cheap": w.cost_cheap, "cost_guaranteed": w.cost_guaranteed,
            "discount": w.discount,
            "batch_throughput": w.batch_throughput, "stall_rate": w.stall_rate,
        },
        "policies": [o.to_json(w.all_guaranteed_cost) for o in outs],
        "best": min(
            (o for o in outs if o.met_rate >= 0.99),
            key=lambda o: o.cost_mean, default=outs[0]).policy,
        "breakeven_horizon": breakeven_horizon(w, seed=seed),
        "modelled": True,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="adder deadline",
        description="Should deferrable work go to the cheap path? Compares "
                    "policies against a deadline, including the misses.",
    )
    ap.add_argument("--units", type=int, default=100, help="tasks in the queue")
    ap.add_argument("--horizon", type=int, default=12,
                    help="steps (typically hours) until the deadline")
    ap.add_argument("--cheap", type=float, default=0.50,
                    help="USD per unit on the cheap path")
    ap.add_argument("--guaranteed", type=float, default=1.00,
                    help="USD per unit on the guaranteed path")
    ap.add_argument("--throughput", type=float, default=DEFAULT_BATCH_THROUGHPUT,
                    help="share of the remaining queue the cheap path returns per step")
    ap.add_argument("--stall-rate", type=float, default=DEFAULT_STALL_RATE,
                    help="share of steps the cheap path returns nothing")
    ap.add_argument("--min-batch-run", type=int, default=1,
                    help="minimum consecutive steps to stay on the cheap path")
    ap.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    try:
        w = Workload(
            units=args.units, horizon=args.horizon,
            cost_cheap=args.cheap, cost_guaranteed=args.guaranteed,
            batch_throughput=args.throughput, stall_rate=args.stall_rate,
            min_batch_run=max(1, args.min_batch_run),
        )
    except ValueError as exc:
        print(f"adder deadline: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(to_json(w, trials=args.trials, seed=args.seed),
                         indent=2, sort_keys=True))
    else:
        print(report(w, trials=args.trials, seed=args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
