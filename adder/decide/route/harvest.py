"""How much of this workload could survive being interrupted, and what that buys.

Why the first pass rejected this, and why that was wrong
---------------------------------------------------------
Harvesting preemptible capacity -- running work on cheap resources that can be
taken away, and recovering when they are -- is a hardware argument, and this
tool never rents hardware. That was the reason given for skipping it, and it
skipped the transferable half.

The transferable half is that **preemptibility is a property of the work, not of
the machine.** Cheap capacity is only cheap if your work can absorb being
interrupted, and whether it can is decided by how much progress is lost when it
is. That question has an exact answer here, because a transcript records
precisely how much context a session had accumulated when each turn ran, and
context is the thing that is lost.

The number that decides it
--------------------------
For an interruption at turn `k` of a session, the work lost is the context that
has to be rebuilt to carry on:

* with **no checkpoint**, everything -- the whole session restarts cold;
* with a **handoff** of `h` tokens, only the difference between the context and
  what the handoff carried.

Averaged over where an interruption actually lands, that gives the expected loss
per interruption. Multiply by an interruption rate, compare against the discount
on the cheap path, and the answer is a number rather than a temperament.

    worth_it  when  discount * spend  >  interruptions * expected_loss

What the local data actually said, against expectation
------------------------------------------------------
The expected finding was that checkpointing decides it, as it does for the
hardware version. Measured on 87 real sessions, it does not, and the reason is
worth keeping:

    spend per session      $84.59
    loss per interruption   $3.19  with no checkpoint at all
                            $3.16  with a 2,000-token handoff
    breakeven               13.3 interruptions per session

Two things fall out. The **discount dwarfs the rebuild**: half of $84.59 buys
roughly thirteen interruptions before it stops paying, so on this workload
interruptibility is close to free and no checkpoint is needed to justify it.
And the **handoff barely helps** -- it removes 1% of the loss, not most of it,
because two thousand tokens against a context of several hundred thousand is a
rounding error.

The general claim -- "checkpointing is what makes preemptible capacity pay" --
is therefore true only where the checkpoint is large relative to the state, and
a handoff summary is not. Where sessions are long and handoffs are small, the
deciding term is the ratio of the discount to a single rebuild, which is a much
simpler quantity and one this report leads with.

The module prints whichever of the two is true for your data rather than the
one that makes a better story.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

from adder.util import render
from adder.util.stats import mean, share

M = 1_000_000.0


@dataclass
class Loss:
    """What an interruption costs, for one session."""

    session: str
    turns: int
    mean_context: float
    expected_loss_cold: float = 0.0
    expected_loss_handoff: float = 0.0

    @property
    def protected(self) -> float:
        """Share of the loss a handoff avoids."""
        if self.expected_loss_cold <= 0:
            return 0.0
        return 1.0 - (self.expected_loss_handoff / self.expected_loss_cold)


def losses(sessions: dict, *, handoff_tokens: int = 2_000,
           model: str | None = None) -> list[Loss]:
    """Expected cost of one interruption, per session, with and without a handoff.

    The expectation is over a uniformly random interruption point, because
    nothing in a transcript says when a preemption would have landed and
    pretending otherwise would be inventing a distribution. Uniform is the
    honest default and it is stated wherever the number is printed.
    """
    from adder.pricing.cost import Rates

    out: list[Loss] = []
    for sid, s in sessions.items():
        turns = s.turns
        if len(turns) < 2:
            continue
        r = Rates.for_model(model or turns[0].model)
        # Rebuilding a context costs one cold read plus one prefix write.
        rebuild = (r.inp + r.cache_write) / M
        cold = [t.context * rebuild for t in turns]
        warm = [max(0, t.context - handoff_tokens) * rebuild for t in turns]
        out.append(Loss(
            session=sid,
            turns=len(turns),
            mean_context=mean([float(t.context) for t in turns]),
            expected_loss_cold=mean(cold),
            expected_loss_handoff=mean(warm),
        ))
    return out


@dataclass
class Report:
    rows: list[Loss] = field(default_factory=list)
    spend: float = 0.0
    handoff_tokens: int = 2_000
    discount: float = 0.5
    interruptions: float = 1.0

    @property
    def sessions(self) -> int:
        return len(self.rows)

    @property
    def expected_loss_cold(self) -> float:
        return mean([r.expected_loss_cold for r in self.rows])

    @property
    def expected_loss_handoff(self) -> float:
        return mean([r.expected_loss_handoff for r in self.rows])

    @property
    def protected(self) -> float:
        return share(self.expected_loss_cold - self.expected_loss_handoff,
                     self.expected_loss_cold)

    def gain(self, *, checkpointed: bool) -> float:
        """Discount earned minus interruptions paid for, per session."""
        per_session = share(self.spend, max(1, self.sessions))
        loss = self.expected_loss_handoff if checkpointed else self.expected_loss_cold
        return self.discount * per_session - self.interruptions * loss

    @property
    def worth_it_cold(self) -> bool:
        return self.gain(checkpointed=False) > 0

    @property
    def worth_it_checkpointed(self) -> bool:
        return self.gain(checkpointed=True) > 0

    def breakeven_rate(self, *, checkpointed: bool) -> float:
        """Interruptions per session the discount can absorb before it stops paying."""
        loss = self.expected_loss_handoff if checkpointed else self.expected_loss_cold
        if loss <= 0:
            return float("inf")
        per_session = share(self.spend, max(1, self.sessions))
        return (self.discount * per_session) / loss

    def to_json(self) -> dict:
        return {
            "sessions": self.sessions,
            "handoff_tokens": self.handoff_tokens,
            "discount": self.discount,
            "interruptions_per_session": self.interruptions,
            "expected_loss_cold_usd": self.expected_loss_cold,
            "expected_loss_handoff_usd": self.expected_loss_handoff,
            "protected_share": self.protected,
            "gain_cold_usd": self.gain(checkpointed=False),
            "gain_checkpointed_usd": self.gain(checkpointed=True),
            "worth_it_cold": self.worth_it_cold,
            "worth_it_checkpointed": self.worth_it_checkpointed,
            "breakeven_interruptions_cold": self.breakeven_rate(checkpointed=False),
            "breakeven_interruptions_checkpointed":
                self.breakeven_rate(checkpointed=True),
            "uniform_interruption_assumption": True,
        }


def analyse(sessions: dict, *, handoff_tokens: int = 2_000,
            discount: float = 0.5, interruptions: float = 1.0,
            model: str | None = None) -> Report:
    rows = losses(sessions, handoff_tokens=handoff_tokens, model=model)
    return Report(
        rows=rows,
        spend=sum(s.cost for s in sessions.values() if s.turns),
        handoff_tokens=handoff_tokens,
        discount=discount,
        interruptions=interruptions,
    )


def format_report(rep: Report) -> str:
    out: list[str] = []
    out += render.heading("harvest — could this work survive being interrupted?",
                          rule="=")
    if not rep.rows:
        out.append("  No session long enough to lose anything. Nothing to model.")
        return "\n".join(out)

    per_session = share(rep.spend, max(1, rep.sessions))
    out.append(render.kv("sessions", f"{rep.sessions:,}"))
    out.append(render.kv("spend per session", render.money(per_session)))
    out.append(render.kv("assumed discount", render.pct(rep.discount)))
    out.append(render.kv("interruptions", f"{rep.interruptions:g} per session"))
    out.append("")

    out += render.table(
        [["no checkpoint", render.money(rep.expected_loss_cold),
          render.money(rep.gain(checkpointed=False)),
          f"{rep.breakeven_rate(checkpointed=False):.2f}"],
         [f"{rep.handoff_tokens:,}-token handoff",
          render.money(rep.expected_loss_handoff),
          render.money(rep.gain(checkpointed=True)),
          f"{rep.breakeven_rate(checkpointed=True):.2f}"]],
        ["recovery", "loss per interruption", "net per session",
         "breakeven interruptions"],
        align="<>>>",
    )

    out.append("")
    if rep.worth_it_checkpointed and not rep.worth_it_cold:
        out += render.wrap(
            f"The checkpoint is what makes this work. A "
            f"{rep.handoff_tokens:,}-token handoff removes "
            f"{render.pct(rep.protected)} of what an interruption costs, which "
            "is the difference between the discount paying for itself and not. "
            "`adder prefix` prices the handoff itself.")
    elif rep.worth_it_cold:
        out += render.wrap(
            "Even with no checkpoint at all the discount covers the expected "
            "loss. These sessions are short enough that being interrupted is "
            "cheap.")
    else:
        out += render.wrap(
            f"Not worth it at this interruption rate. The discount is worth "
            f"{render.money(rep.discount * per_session)} per session and an "
            f"interruption costs {render.money(rep.expected_loss_handoff)} even "
            f"with a handoff, so it stops paying above "
            f"{rep.breakeven_rate(checkpointed=True):.2f} interruptions.")

    out.append("")
    out += render.wrap(
        "MODELLED, and the assumption worth arguing with is uniform: an "
        "interruption is taken to be equally likely at any point in a session, "
        "because nothing in a transcript says otherwise. Losses grow with "
        "context, so if interruptions cluster late this understates them.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    from adder.core import filters

    ap = argparse.ArgumentParser(
        prog="adder harvest",
        description="Whether cheap-but-interruptible capacity would pay, given "
                    "how much context an interruption would destroy.",
    )
    ap.add_argument("--handoff", type=int, default=2_000,
                    help="tokens a restart can carry forward (default 2000)")
    ap.add_argument("--discount", type=float, default=0.5,
                    help="share off the price of the interruptible path")
    ap.add_argument("--interruptions", type=float, default=1.0,
                    help="expected interruptions per session")
    ap.add_argument("--json", action="store_true")
    filters.add_arguments(ap)
    args = ap.parse_args(argv)

    sessions, _w = filters.load(args, use_cache=True)
    if not sessions:
        print(json.dumps({"sessions": 0}, indent=2) if args.json
              else "  No sessions found.")
        return 1

    rep = analyse(sessions, handoff_tokens=max(0, args.handoff),
                  discount=max(0.0, min(1.0, args.discount)),
                  interruptions=max(0.0, args.interruptions))
    if args.json:
        print(json.dumps(rep.to_json(), indent=2, sort_keys=True))
    else:
        print(format_report(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
