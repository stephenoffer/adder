"""What the verbose model's rating advantage costs you per answer.

Reads a log of head-to-head comparisons with the responses attached, fits the
style-controlled Bradley-Terry model from `adder.pricing.style`, and turns the
result into the number that matters here: **the dollars per answer you are
paying for the tokens that bought the rating rather than the capability.**

The estimator's caveats are in `adder/pricing/style.py` and are repeated in the
output rather than left in the source. The short version: this is
observational, length correlates with substance, and the controlled strength is
a lower bound on capability rather than a corrected measurement of it. Paired
with an exact cost, a lower bound is the conservative direction for a spending
decision, which is why it is still the right number to route on.

Input format, one JSON object per line:

    {"a": "model-x", "b": "model-y", "winner": "a",
     "a_text": "...", "b_text": "..."}

or, when you have the counts but not the text:

    {"a": "model-x", "b": "model-y", "winner": "tie",
     "a_style": {"tokens": 900, "headers": 3, "lists": 4, "bold": 2},
     "b_style": {"tokens": 300, "headers": 0, "lists": 0, "bold": 0}}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from adder.pricing.bt import Battle
from adder.pricing.style import FEATURES, Style, fit_controlled, mean_style, premium_cost
from adder.util import render

DEFAULT_RESAMPLES = 40


def _style_from(d: dict, text_key: str, style_key: str) -> Style:
    text = d.get(text_key)
    if isinstance(text, str) and text:
        from adder.pricing.style import measure

        return measure(text)
    raw = d.get(style_key)
    if isinstance(raw, dict):
        return Style(tokens=int(raw.get("tokens", 0)),
                     headers=int(raw.get("headers", 0)),
                     lists=int(raw.get("lists", 0)),
                     bold=int(raw.get("bold", 0)))
    return Style()


def load(path: Path) -> tuple[list[Battle], list[tuple[Style, Style]]]:
    """Read comparisons and their response styles. Malformed lines are fatal.

    Fatal rather than skipped, for the same reason as everywhere else here: a
    silently skipped line means the fit ran on whatever happened to parse, and
    nobody finds out.
    """
    battles: list[Battle] = []
    styles: list[tuple[Style, Style]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: not JSON ({exc.msg})") from exc
            missing = [k for k in ("a", "b") if k not in d]
            if missing:
                raise ValueError(f"{path}:{lineno}: missing {', '.join(missing)}")
            battles.append(Battle(str(d["a"]), str(d["b"]),
                                  str(d.get("winner", "tie"))))
            styles.append((_style_from(d, "a_text", "a_style"),
                           _style_from(d, "b_text", "b_style")))
    return battles, styles


def per_model_style(battles, styles) -> dict[str, Style]:
    """Average response style per model, for pricing the premium."""
    seen: dict[str, list[Style]] = {}
    for b, (sa, sb) in zip(battles, styles, strict=True):
        seen.setdefault(b.a, []).append(sa)
        seen.setdefault(b.b, []).append(sb)
    return {m: mean_style(v) for m, v in seen.items()}


def report(battles, styles, *, remaining_turns: int = 100,
           resamples: int = DEFAULT_RESAMPLES, model_for_rates: str = "claude-opus-5",
           top: int = 10) -> str:
    from adder.pricing.cost import Rates

    out: list[str] = []
    out += render.heading("verbosity — what the extra tokens bought", rule="=")
    if not battles:
        out.append("  No comparisons to fit. Supply a log with responses attached.")
        return "\n".join(out)

    fitted = fit_controlled(battles, styles, resamples=resamples)
    out.append(render.kv("comparisons", f"{len(battles):,}"))
    out.append(render.kv("models", str(len(fitted.strength))))

    if not fitted.identified:
        out.append("")
        out += render.wrap(
            "Style never varied within a matchup in this log, so style and skill "
            "are collinear and no fit can separate them. The coefficients below "
            "are not evidence of anything. Collect comparisons where the same "
            "pair produced answers of different lengths.")
        return "\n".join(out)

    out.append("")
    out += render.heading("what the judges paid for")
    rows = []
    for f in FEATURES:
        lo, hi = fitted.beta_ci.get(f, (0.0, 0.0))
        rows.append([f, f"{fitted.beta.get(f, 0.0):+.3f}",
                     f"[{lo:+.3f}, {hi:+.3f}]" if fitted.beta_ci else "—",
                     "yes" if lo > 0 else ""])
    out += render.table(rows, ["feature", "coefficient", "95% CI", "rewarded"],
                        align="<>><")

    out.append("")
    out += render.heading("per model")
    rates = Rates.for_model(model_for_rates)
    mean_by_model = per_model_style(battles, styles)
    baseline = min((s.tokens for s in mean_by_model.values()), default=0)
    body = []
    for m in sorted(fitted.strength, key=lambda k: -fitted.premium(k))[:top]:
        premium = fitted.premium(m)
        extra = max(0, mean_by_model.get(m, Style()).tokens - baseline)
        dollars = premium_cost(premium, extra_tokens=extra, out_rate=rates.out,
                               cache_read_rate=rates.cache_read,
                               remaining_turns=remaining_turns)
        body.append([m[:26], f"{fitted.uncontrolled.get(m, 0):.0f}",
                     f"{fitted.strength.get(m, 0):.0f}", f"{premium:+.0f}",
                     f"{extra:,}", render.money(dollars)])
    out += render.table(
        body, ["model", "raw", "controlled", "premium", "extra tok", "$/answer"],
        align="<>>>>>")

    out.append("")
    if fitted.length_matters:
        out += render.wrap(
            "Length is rewarded in this data: its coefficient's interval clears "
            "zero. Any ranking built on it is partly a length contest, and the "
            "dollars column is what that costs you per answer over "
            f"{remaining_turns} remaining turns — once as output, then again as "
            "prefix on every turn after.")
    else:
        out += render.wrap(
            "Length is not measurably rewarded here, so the raw ranking is not "
            "obviously a length contest. The controlled column should still be "
            "the one you route on, because it cannot be worse.")

    out += render.wrap(
        "OBSERVATIONAL: a model may write more because it is doing more work, "
        "and the control strips that out along with the padding. The controlled "
        "strength is a lower bound on capability, not a corrected measurement.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="adder verbosity",
        description="Separate a model's capability from how much it writes, "
                    "and price the difference.",
    )
    ap.add_argument("path", nargs="?", type=Path,
                    help="JSONL of comparisons with responses or style counts")
    ap.add_argument("--turns", type=int, default=100,
                    help="remaining turns, for the carry half of the cost")
    ap.add_argument("--rates-from", default="claude-opus-5",
                    help="model whose rates price the premium")
    ap.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    battles: list[Battle] = []
    styles: list[tuple[Style, Style]] = []
    if args.path is not None:
        if not args.path.exists():
            print(f"adder verbosity: no such file: {args.path}", file=sys.stderr)
            return 1
        try:
            battles, styles = load(args.path)
        except ValueError as exc:
            print(f"adder verbosity: {exc}", file=sys.stderr)
            return 2

    if args.json:
        fitted = fit_controlled(battles, styles, resamples=args.resamples)
        payload = fitted.to_json()
        payload["mean_style"] = {
            m: {"tokens": s.tokens, "headers": s.headers,
                "lists": s.lists, "bold": s.bold}
            for m, s in sorted(per_model_style(battles, styles).items())
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(report(battles, styles, remaining_turns=max(0, args.turns),
                     resamples=args.resamples, model_for_rates=args.rates_from,
                     top=max(1, args.top)))
    return 0 if battles else 1


if __name__ == "__main__":
    raise SystemExit(main())
