# Command reference

Every report is computed locally from transcript files. No API key, no model
calls, and no network, with exactly one exception: `adder models refresh`, which
is opt-in and named as such below.

Generated from the command table in `adder/cli/commands.py`;
`tests/repo/test_invariants.py` fails if a command is missing here.

## Measure

Read-only reports over transcript files. None of these write anything.

| Command | What it does |
|---|---|
| `adder live [--cwd DIR]` | this session: cost/turn, next-turn cost, pressure |
| `adder trace [root] [--json] [--verify]` | total spend, by model and session |
| `adder debt [root]` | what an output token really costs ([cost-model.md](cost-model.md)) |
| `adder context [root]` | where context growth comes from |
| `adder cache [root]` | cache hit rate and rebuild waste, by cause |
| `adder speed [root] [--max-gap S] [--json]` | the fast serving path bills at 2x; audit whether the speed arrived, paired within model, wall clock only ([research-map.md](research-map.md)) |
| `adder sched [root] [--json]` | the mean-residual-life curve: whether how far a session has run predicts how much is left, against the equal-length reference of -0.50 turns/turn  ([systems.md](systems.md))|
| `adder spec [root] [--top N] [--json]` | agent sessions read as search: probe scale, the explore/formulate/validate mix, repeated probes and what they cost, and how much a human steer collapses the search |
| `adder cachesim [root] [--ttl S] [--json]` | replay the workload against a simulated prefix cache: hit rate against capacity, block size and TTL (SIMULATED) |
| `adder quality [root] [--since DATE]` | agent-performance proxies ([quality.md](quality.md)) |
| `adder horizon [root]` | remaining-turns estimate vs the naive countdown |
| `adder carry [root] [--model M]` | what carrying a token in context really costs, measured |
| `adder prefix [root] [--model M] [--ttl T] [--handoff TOK]` | what a session restart really costs, measured ([levers.md](levers.md)) |
| `adder tools [root] [--top N] [--json]` | which tool fills your context, and what carrying its results costs |
| `adder compact [root] [--top N] [--vs-restart TOK] [--handoff TOK] [--json]` | every compaction on record with its measured rebuild and net verdict, the turns-remaining threshold above which compacting pays, the sessions that carried a full context and never compacted, and compact-vs-restart at a given context |
| `adder reread [root] [--top N] [--min-tokens TOK] [--min-sessions N] [--json]` | content the agent admitted to the context more than once (separating a redundant copy from a justified refresh), the files re-read however the harness read them — `Read`, `cat` or a `sed -n` range of one already held — plus the reads that recur across sessions and the largest resident note that would still beat them |
| `adder memory [root] [--repo DIR] [--home DIR] [--model M] [--top N] [--what-if TOK] [--json]` | what the always-loaded prefix (CLAUDE.md, the memory index, skill and agent descriptions) costs on every turn of every session, per file, plus what is duplicated, stale, or unindexed |
| `adder sessions [root] [--sort K] [--top N] [--json]` | one row per session: cost, $/turn, peak context, compactions, cache rebuilds |
| `adder agents [root] [--top N] [--json]` | delegation as measured: subagent spend, subagent model choice, and the large reads that went inline |
| `adder anomaly [root] [--z N] [--top N] [--json]` | the turns that cost far more than the rest, each with the mechanism that explains it |
| `adder effort [root] [--model M] [--json]` | re-fit the effort→output-volume priors against local transcripts |
| `adder limits [root] [--hours H] [--json]` | the five-hour metering window rebuilt from timestamps: what each window read, how much of it was carry, and what a turn late in a window costs against one early in it, plus the heaviest sliding 7 days as a proxy for the weekly cap. For plan users, where the constraint is a lockout rather than a bill |
| `adder budget [root] [--limit USD] [--period P] [--strict]` | burn-down and projection against a spend target |
| `adder export [root] [--format F] [--grain G] [-o PATH]` | priced turns, sessions, or days as CSV/JSON/JSONL; never any message content |

## Decide

Turn a measurement into a choice.

| Command | What it does |
|---|---|
| `adder policy "<task>" [--json] [--cross-vendor] [--record]` | route a task: inline vs delegate, and whether another vendor should run the subagent ([models.md](models.md)). `--record` books the recommendation in the ledger |
| `adder outcomes [--log PATH] [--project P] [--context TOK] [--json]` | escalation calibration (p_fail), and how far each tier is from being allowed to take work |
| `adder outcomes record --tier T [--model M] [--project P] [--escalated] ...` | append one dispatch outcome by hand |
| `adder outcomes import [root] [--write] [--json]` | backfill that log from transcripts: every `Agent` dispatch and whether it escalated. Dry run unless `--write`; idempotent, so re-running adds only what is new |
| `adder guard [root] [--learn] [--replay] [--shadow] [--last] [--explain CMD] [--install] [--json]` | what the PreToolUse guard predicts, decides, and has cost. `--shadow` reads back what shadow mode would have refused and how often the session went round it; `--last` lists every finding and refusal from the most recent recorded session. Both are read-only — `adder auto on --shadow` is what turns shadow mode on |
| `adder ledger [--log PATH] [--json]` | has the advice been worth more than the asking? |
| `adder guard [root] [--learn] [--replay] [--shadow] [--last] [--explain CMD] [--json]` | what the PreToolUse guard predicts, decides, and has cost ([guard.md](guard.md)) |
| `adder handoff [--cwd DIR] [--context TOK] [--remaining N] [--top N] [--json]` | the largest brief that can cross a restart before restarting stops paying, what the brief has to name (files edited, commands re-run, reads by re-establishment cost), and what handoffs on this machine have actually been |
| `adder classify "<task>" [--json]` | task-complexity classification, on its own |
| `adder classify --terms [--json]` | the project vocabulary in effect, and how to set it. The shipped vocabulary is English about software in general; on a domain codebase it abstains on nearly every task, which spends routing overhead to arrive at "no change". `classify_terms` in the project's `.adder.json` teaches it this repository's nouns |
| `adder similar "<task>" [--floor R] [--top K] [--json]` | what happened last time on tasks whose vocabulary resembles this one: the per-tier escalation rate over the nearest recorded runs, against the tier-wide rate the router used before ([tiers.md](tiers.md)) |
| `adder pick "<task>" [--combos] [--json]` | cheapest model, or combination, that clears the quality bar ([models.md](models.md)) |
| `adder harvest [root] [--handoff TOK] [--discount R] [--interruptions N] [--json]` | whether cheap-but-interruptible capacity would pay, given how much context an interruption destroys ([research-map.md](research-map.md)) |
| `adder place [--model M] [--context TOK] [--turns N] [--rated-only] [--json]` | should this warm session move to a cheaper model? prices the resident prefix you would discard and the turns a move needs to repay itself  ([systems.md](systems.md))|
| `adder blend [queue.jsonl] [--ttl S] [--task-seconds S] [--json]` | order a queue of deferrable work so shared prefixes stay warm; sweeps the TTL, because the saving peaks rather than rising with it ([research-map.md](research-map.md)) |
| `adder deadline [--units N] [--horizon N] [--stall-rate R] [--json]` | should deferrable work go to the cheap-but-slow path? compares policies against a deadline and prices the misses  ([systems.md](systems.md))|
| `adder cascade [--weak M] [--strong M] [--p-fail P] [--miss R] [--turns N] [--json]` | price try-cheap-then-check against going straight to the big model, including the carry cost of a failed attempt left in the context |
| `adder frontier ["<task>"] [--board B] [--context TOK] [--turns N] [--json]` | the cost-quality Pareto frontier for a task; a model only outranks a cheaper one when its rating interval clears it, so the models whose lead is noise drop out ([routing.md](routing.md)) |
| `adder models [list|show|ladder|refresh]` | the cross-provider catalog: what exists, at what price and rating ([models.md](models.md)) |

## Evaluate

Check that a lever is real before trusting it.

| Command | What it does |
|---|---|
| `adder savings [root] [--max-turns N]` | what each lever is worth ([levers.md](levers.md)) |
| `adder verify --since DATE [root]` | did a change actually land? |
| `adder validate [root]` | re-test the claims everything rests on; each is `PASS`, `FAIL`, or `N/A` when this corpus has nothing to test it against, and only a `FAIL` sets the exit code ([measurement.md](measurement.md)) |
| `adder regret [root]` | dollar regret of the horizon estimator |
| `adder simulate [root]` | replay sessions under interventions; test lever composition |
| `adder plan [root] [--target N] [--delegate-above TOK] [--split-turns N] [--effort E] [--session-model M] [--session-rework F]` | price the whole workload under one followable regime, and solve for the mildest one that hits a target reduction |
| `adder bench [root] [--guard-cost USD] [--handoff TOK] [--json]` | cost with adder vs without, on the same recorded turns ([benchmark.md](benchmark.md)) |
| `adder routereval [episodes.jsonl] [--log PATH] [--split S] [--targets 50,80,95] [--json]` | score the router itself: PGR, APGR and CPT against a random-ordering baseline, on both a call-count and a dollar axis ([routing.md](routing.md)) |
| `adder calib [--log PATH] [--global-rate] [--json]` | score `p_fail` out of sample, prequentially: Brier, skill against the base rate, reliability by bin, and the dollars miscalibration moved |
| `adder verbosity [battles.jsonl] [--turns N] [--json]` | fit the style-controlled Bradley-Terry model: how much of a model's rating is length rather than capability, and what those extra tokens cost per answer ([research-map.md](research-map.md)) |
| `adder design [battles.jsonl] [--budget N] [--cost USD] [--json]` | allocate a fixed comparison budget to the model pairs that would actually reduce uncertainty about the ranking, instead of spreading it evenly ([routing.md](routing.md)) |
| `adder ab [--models A,B] [--backend cli\|sdk] [--repeats N] [--run]` | controlled A/B on answer quality: identical prompts, identical context, different models ([quality.md](quality.md)) |
| `adder ab --recall [--models A,B] [--run]`, `adder ab --print-seed` | score each model on a fixture with a known number of planted defects, and name the ones it missed. The only quality signal here that shares no code with the cost model, so it is the only one that cannot agree with a saving by construction. `--print-seed` prints the fixture and the prompt so the same measurement can be run by hand |
| `adder repro [root] [--deep] [--write PATH] [--check PATH] [--json]` | hash the four things every number depends on (transcripts, prices, catalog, code) and diff against a manifest recorded earlier; exits 1 on drift |
| `adder doctor [root] [--strict] [--json]` | run every check and rank the findings by dollars at stake |

## Setup

Inspect what the tool is configured to do, and turn on the parts that run
without being asked. `adder auto on` is the only command in this tool that
writes a file you did not name: it says what it will change before it changes
it, keeps a `.adder.bak` of whatever was there, and `adder auto off` removes
exactly what it added. It writes to `~/.claude` unless you pass `--project`:
a `.claude/settings.json` is commonly tracked in git, and a hooks block there
is configuration every contributor inherits without asking for it.

| Command | What it does |
|---|---|
| `adder auto [status]` | is anything running between your turns, and what has it been worth: refusals at par, advice discounted by measured uptake |
| `adder auto on [--shadow\|--full] [--project] [--yes] [--dry-run]` | install the three hooks (with an explicit 5s timeout each) and start enforcing. Writes to `~/.claude` by default; `--project` writes to this repository's `.claude/` instead and says what it is putting inside a tracked tree. `certain` (default) refuses only calls that admit nothing new; `--full` also refuses a large read that has a cheaper equal; `--shadow` refuses nothing at all and records what it would have refused, which is how the assumed uptake term becomes a measurement before anything is denied |
| `adder auto off [--project] [--yes]` | remove the hooks and stop enforcing; foreign hooks in the same file are left alone |
| `adder hook NAME` | run one harness hook (`read-guard`, `compact-learn`, `cost-advisor`). Claude Code calls this; you do not. It exists so a project-scope install can name a command that resolves on every contributor's machine instead of one absolute path from the machine that ran `auto on` |
| `adder config [name] [--json] [--explain]` | every setting in effect, its value, and which layer set it |
| `adder config --init` | print a config-file template to stdout |
| `adder completion [bash\|zsh\|fish]` | shell completion, generated from the command table and each module's own parser |

Precedence is `built-in default < ~/.claude/adder.json < ./.adder.json < ADDER_* environment`.
`adder config` prints the source of each value, which is the half that matters
when two machines produce different numbers from the same transcripts.

## Conventions

- `root` defaults to your local Claude Code projects directory
  (`~/.claude/projects`), or to whatever `adder config root` reports. Pass a
  path to analyse a copy instead.
- Every command that reads transcripts accepts the same **window flags**:
  `--since DATE`, `--until DATE`, `--project SUBSTR`, `--model-filter PREFIX`,
  `--session ID`, `--min-turns N`, and `--only-subagents` / `--no-subagents`.
  Dates may be absolute (`2026-08-01`) or relative (`7d`, `2w`, `today`,
  `yesterday`). The window is half-open (`--since` is inclusive, `--until` is
  exclusive), so two adjacent windows partition the data exactly.
- `adder` with no arguments, or `adder help`, prints the command list.
- `adder help <command>` and `adder <command> --help` both show that command's flags.
  Each module owns its own parser, so the flags shown are always the real ones.
- `adder version` prints the installed version.
- Every command exits `0` on success, `1` on a reported failure, and `2` on a
  usage error.
- Every report takes `--json` and prints exactly one document on stdout, with no
  banner line before it and no bare `NaN`/`Infinity` inside it.
  `tests/repo/test_json_surface.py` discovers the flag from the source and asserts
  both, so a new command cannot ship a JSON mode that breaks a pipeline.
  `adder export` is the exception by design: its format is `--format json`.
- `--strict` on `trace`, `budget`, `doctor`, and `validate` turns a finding into
  a non-zero exit, for use from a hook or from CI.
- Settings resolve as `built-in default < ~/.claude/adder.json < ./.adder.json <
  ADDER_* environment`. `adder config` prints the effective value and the layer
  that set it.

## The one networked command

`adder models refresh` is the only command that opens a socket. It pulls the public
model catalog, writes it to your user cache, and touches nothing else. It never
runs implicitly, and `ADDER_OFFLINE=1` makes it refuse outright. Every other
command works from a machine with no network at all.

Replay a saved capture instead of fetching with
`adder models refresh --from openrouter=models.json`.

## What ships

- **`adder/`**: pricing with context limits and cache minimums, a cache-aware
  cost model, transcript parsing with deduplication, growth attribution, cache
  analysis, quality proxies, the model catalog, and the routing policy.
- **`adder/cli/commands.py`**: the dispatcher. Adding a command means one row
  here plus a module, a test, and a line in this file.
- **`adder/decide/agents/`**: Explore on Haiku plus three routing tiers, each
  with output-bounding rules. Inside the package, so a `pip install` carries
  them; `adder auto on` copies them to `.claude/agents/`.
- **`adder/decide/hooks/`**: a prompt hook that prices the session, and a
  **PreToolUse guard** that prices a large read *before* it lands in context.
  The guard is the only thing here that prevents cost rather than reporting it.
  It fires on a **cost**, not a token count (`ADDER_GUARD_MIN_COST`, default
  $0.25), because the same read is worth interrupting for with 400 turns left
  and not worth mentioning with three. `ADDER_GUARD_BLOCK=1` escalates from
  advice to a confirmation prompt; it never denies, and never blocks silently.
- **`.claude/skills/`**: `/adder`, `/adder-doctor`, `/adder-context`,
  `/adder-init`. Checkout-only; they are conveniences, not the mechanism.
- **`scripts/adder`**: launcher for a git checkout; `pip install` provides
  `adder` directly.
