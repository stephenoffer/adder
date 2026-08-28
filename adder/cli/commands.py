"""The command table: the one place a subcommand is declared.

Every other part of the CLI is generated from this tuple -- the help text, the
shell completions, the smoke target in the Makefile, and the test that asserts
each entry actually imports. That is the point: a command that is added here is
reachable and documented everywhere at once, and a command that is not here does
not exist, however importable its module is.

The `module` field is a dotted path resolved lazily at dispatch time, so
`adder live` never pays to import the A/B harness.
"""

from __future__ import annotations

from typing import NamedTuple


class Command(NamedTuple):
    name: str
    module: str
    group: str
    usage: str
    summary: str


COMMANDS: tuple[Command, ...] = (
    # Measure — read-only reports over transcript files.
    Command("live", "adder.measure.session.live", "Measure", "[--cwd DIR]",
            "this session: cost/turn, next-turn cost, pressure"),
    Command("trace", "adder.measure.spend.trace", "Measure", "[root] [--json] [--verify]",
            "total spend, by model and session"),
    Command("debt", "adder.measure.spend.debt", "Measure", "[root]",
            "what an output token really costs"),
    Command("context", "adder.measure.window.context", "Measure", "[root]",
            "where context growth comes from"),
    Command("cache", "adder.measure.window.cache", "Measure", "[root]",
            "cache hit rate and rebuild waste, by cause"),
    Command("spec", "adder.measure.session.speculation", "Measure",
            "[root] [--top N] [--json]",
            "agent sessions as search: scale, mix, redundancy, steerability"),
    Command("cachesim", "adder.measure.window.cachesim", "Measure",
            "[root] [--ttl S] [--json]",
            "simulated prefix cache: hit rate vs capacity and block size"),
    Command("speed", "adder.measure.session.speed", "Measure",
            "[root] [--max-gap S] [--json]",
            "the fast path bills at 2x — did the speed arrive?"),
    Command("sched", "adder.measure.session.sched", "Measure",
            "[root] [--json]",
            "does how far a session has run predict what is left?"),
    Command("quality", "adder.measure.session.quality", "Measure", "[root] [--since DATE]",
            "agent-performance proxies"),
    Command("horizon", "adder.measure.session.horizon", "Measure", "[root]",
            "remaining-turns estimate vs the naive countdown"),
    Command("carry", "adder.measure.window.carry", "Measure", "[root] [--model M]",
            "what carrying a token in context really costs, measured"),
    Command("prefix", "adder.measure.window.prefix", "Measure", "[root] [--handoff TOK]",
            "what a session restart really costs, measured"),
    Command("tools", "adder.measure.window.tools", "Measure", "[root] [--top N] [--json]",
            "which tool fills your context, and what that carry costs"),
    Command("compact", "adder.measure.window.compact", "Measure",
            "[root] [--top N] [--vs-restart TOK] [--json]",
            "what each compaction cost, and the turn count where it pays"),
    Command("reread", "adder.measure.window.reread", "Measure",
            "[root] [--top N] [--min-sessions N] [--json]",
            "content admitted to context twice, and what a note would cost instead"),
    Command("memory", "adder.measure.window.memory", "Measure",
            "[root] [--repo DIR] [--what-if TOK] [--json]",
            "what CLAUDE.md, memory, and skill descriptions cost every turn"),
    Command("sessions", "adder.measure.spend.sessions", "Measure", "[root] [--sort K] [--top N]",
            "one row per session: cost, $/turn, peak context, cache damage"),
    Command("limits", "adder.measure.spend.limits", "Measure",
            "[root] [--hours H] [--json]",
            "the 5-hour metering window, and what the carry costs it"),
    Command("budget", "adder.measure.spend.budget", "Measure", "[root] [--limit USD] [--period P]",
            "burn-down and projection against a spend target"),
    Command("export", "adder.measure.spend.export", "Measure", "[root] [--format F] [--grain G]",
            "priced turns out as CSV or JSON; no message content"),
    Command("anomaly", "adder.measure.spend.anomaly", "Measure", "[root] [--z N] [--top N]",
            "the turns that cost far more than the rest, and why"),
    Command("agents", "adder.measure.spend.agents", "Measure", "[root] [--top N] [--json]",
            "delegation as measured: subagent spend, and what was not delegated"),
    Command("effort", "adder.measure.session.effort", "Measure", "[root] [--model M] [--json]",
            "re-fit the effort→output priors against local transcripts"),
    # Decide — turn a measurement into a routing choice.
    Command("policy", "adder.decide.route.policy", "Decide", '"<task>" [--json]',
            "route a task: inline vs delegate"),
    Command("outcomes", "adder.decide.track.outcomes", "Decide", "[--log PATH]",
            "escalation calibration (p_fail)"),
    Command("guard", "adder.decide.guard", "Decide",
            "[root] [--learn] [--explain CMD] [--json]",
            "what the PreToolUse guard predicts, decides, and has cost"),
    Command("ledger", "adder.decide.track.ledger", "Decide", "[--log PATH] [--json]",
            "has the advice been worth more than the asking?"),
    Command("handoff", "adder.decide.handoff", "Decide",
            "[--cwd DIR] [--context TOK] [--json]",
            "how much may cross a restart, and what the brief must name"),
    Command("classify", "adder.decide.route.classify", "Decide", '"<task>"',
            "task-complexity classification, on its own"),
    Command("similar", "adder.decide.track.similar", "Decide",
            '"<task>" [--floor R] [--top K] [--json]',
            "what happened last time on tasks like this one"),
    Command("pick", "adder.decide.route.select", "Decide", '"<task>" [--combos] [--json]',
            "cheapest model, or combination, that clears the quality bar"),
    Command("harvest", "adder.decide.route.harvest", "Decide",
            "[root] [--handoff TOK] [--discount R] [--json]",
            "could this work survive interruption, and what would that buy?"),
    Command("place", "adder.decide.route.place", "Decide",
            "[--model M] [--context TOK] [--turns N] [--json]",
            "stay warm, or move this session somewhere cheaper?"),
    Command("blend", "adder.decide.route.blend", "Decide",
            "[queue.jsonl] [--ttl S] [--json]",
            "order a queue so shared prefixes stay warm"),
    Command("deadline", "adder.decide.route.deadline", "Decide",
            "[--units N] [--horizon N] [--stall-rate R] [--json]",
            "is the cheap slow path worth it, given a deadline?"),
    Command("cascade", "adder.decide.route.cascade", "Decide",
            "[--weak M] [--strong M] [--p-fail P] [--json]",
            "try cheap and check, or go straight to the big model?"),
    Command("frontier", "adder.decide.route.frontier", "Decide",
            '["<task>"] [--board B] [--context TOK] [--json]',
            "the cost-quality frontier, with domination decided by intervals"),
    Command("models", "adder.decide.route.models", "Decide", "[list|show|ladder|refresh]",
            "the cross-provider catalog: what exists, at what price and rating"),
    # Evaluate — check that a lever is real before trusting it.
    Command("savings", "adder.evaluate.claims.savings", "Evaluate", "[root] [--max-turns N]",
            "what each lever is worth"),
    Command("verify", "adder.evaluate.claims.verify", "Evaluate", "--since DATE [root]",
            "did a change actually land?"),
    Command("validate", "adder.evaluate.claims.validate", "Evaluate", "[root]",
            "re-test the claims everything rests on"),
    Command("regret", "adder.evaluate.claims.regret", "Evaluate", "[root]",
            "dollar regret of the horizon estimator"),
    Command("simulate", "adder.evaluate.replay.simulate", "Evaluate", "[root]",
            "replay sessions under interventions; test lever composition"),
    Command("plan", "adder.evaluate.replay.plan", "Evaluate", "[root] [--target N]",
            "price the whole workload under one followable regime"),
    Command("bench", "adder.evaluate.replay.bench", "Evaluate", "[root] [--guard-cost USD] [--json]",
            "cost with adder vs without, on the same recorded turns"),
    Command("routereval", "adder.evaluate.replay.routereval", "Evaluate",
            "[episodes.jsonl] [--split S] [--json]",
            "score the router: PGR, APGR, CPT against a random baseline"),
    Command("calib", "adder.evaluate.claims.calib", "Evaluate",
            "[--log PATH] [--global-rate] [--json]",
            "is p_fail a probability? scored out of sample"),
    Command("verbosity", "adder.evaluate.claims.verbosity", "Evaluate",
            "[battles.jsonl] [--turns N] [--json]",
            "how much of a rating is length, and what that costs"),
    Command("design", "adder.evaluate.claims.design", "Evaluate",
            "[battles.jsonl] [--budget N] [--cost USD] [--json]",
            "which comparison to run next, given a fixed budget"),
    Command("ab", "adder.evaluate.replay.ab", "Evaluate", "--help",
            "controlled A/B on answer quality"),
    Command("repro", "adder.evaluate.claims.repro", "Evaluate",
            "[root] [--deep] [--write PATH] [--check PATH] [--json]",
            "fingerprint every input a number depended on, and diff it"),
    Command("doctor", "adder.evaluate.doctor", "Evaluate", "[root] [--strict] [--json]",
            "run every check and rank the findings by dollars at stake"),
    # Setup — inspect the machine's own configuration, and the one command
    # that changes it.
    Command("auto", "adder.decide.auto", "Setup",
            "[on|off|status] [--full] [--project] [--yes]",
            "run adder between your turns: install the hooks, enforce the levers"),
    Command("hook", "adder.decide.hooks.run", "Setup", "NAME",
            "run one harness hook; Claude Code calls this, you do not"),
    Command("config", "adder.cli.config", "Setup", "[name] [--json] [--init]",
            "settings in effect, and which layer set each one"),
    Command("completion", "adder.cli.completion", "Setup", "[bash|zsh|fish]",
            "shell completion, generated from this table"),
)

BY_NAME: dict[str, Command] = {c.name: c for c in COMMANDS}

# Display order. A command whose group is not listed here is unreachable from
# `adder help`, which is exactly the silent failure `tests/test_cli.py` guards.
GROUPS: tuple[str, ...] = ("Measure", "Decide", "Evaluate", "Setup")

GROUP_BLURB = {
    "Measure": "read-only, no API calls, no network",
    # `models refresh` is the single exception to the no-network rule, and it
    # only ever runs when typed.
    "Decide": "offline except `models refresh`",
    "Evaluate": "",
    "Setup": "what is configured, and the one command that changes it",
}
