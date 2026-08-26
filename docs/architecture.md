# Architecture

One paragraph: transcripts on disk are parsed into per-turn records, priced by a
cache-aware cost model, and then read by a set of independent report modules. A
dispatcher maps a command name to one of those modules. Nothing else happens.

```
~/.claude/projects/**/*.jsonl        any usage log: OpenAI, Gemini,
          |                          OTel, LiteLLM, generic .json/.jsonl
          |                                    |
          v                                    v
   trace.iter_file          parse + DEDUPLICATE by message.id
          |                 falls back to core/ingest.py, which sniffs the
          |                 format PER RECORD and normalizes it
          v
   Turn / Session           the only shared data model
          |
    +-----+---------------------------+
    |                                 |
    v                                 v
 pricing/registry.py               pricing/cost.py
 one resolution point for any      what a turn costs, given the cache
 model: prices.py (first-party,    economics of the provider that
 dated) < catalog.py (~500         actually served it
 models) < providers.py (per-
 vendor cache mechanics)
    |                                 |
    +-----+---------------------------+
          |
          v
   report modules            live, debt, context, cache, quality, horizon,
   (one per command)         tools, sessions, agents, anomaly, effort, budget,
          |                  export, savings, verify, validate, regret,
          |                  simulate, plan, bench, ab, doctor, policy,
          |                  outcomes, classify, select, models
          v
       cli.py                name -> module, lazily imported
```

Two tables sit underneath the pricing layer and are pure data, both overridable
from a file so a new vendor never needs a fork. `pricing/providers.py` holds how
each provider *bills* caching: explicit or automatic, what a write costs, what
a read costs, whether a TTL is even selectable. That shape, not the per-token
price, is what the carry term turns on. `core/harness.py` holds what
each agent runtime makes *possible*: Claude Code pins the main session to
Anthropic, Codex to OpenAI, Gemini CLI to Google, and quoting an inline price
for a model the harness cannot run as its session is quoting a placement that
does not exist. See `docs/providers.md`.

Three small modules sit beside that pipeline rather than inside it, and every
report may use them: `stats.py` (quantiles and robust scale), `render.py`
(money, tokens, tables, colour), and `filters.py` (the `--since`/`--project`
window). They exist because each replaced several disagreeing copies: three definitions
of a p90, fifteen dollar formats, two incompatible date boundaries.
None of them import anything from the rest of the package, so nothing in the
pipeline depends on the order they load.

## The layers

**Parsing (`trace.py`).** The one place that reads transcript files. Its most
important behaviour is deduplication: Claude Code writes one JSONL record per
content block and each record repeats the whole message's `usage`, so summing
raw lines multi-counts nearly every turn. Records are grouped by `message.id`
and the highest `output_tokens` record wins, because partial streamed records
carry a running count. Getting this wrong inflated the project's original
numbers by 1.78x. A parse cache keyed on `(mtime, size)` makes repeated runs
cheap, is written atomically, and is namespaced per pid so concurrent sessions
cannot clobber each other.

**Pricing (`prices.py`, `catalog.py`, `sources.py`).** `prices.py` is a
hand-maintained, date-aware table of first-party Claude rates plus per-model
context limits and cache minimums. Date-aware because introductory pricing
expires, and a threshold tuned against an intro rate is silently wrong the day
it reverts. `catalog.py` generalises this across vendors as *data* rather than
code, merged in layers (bundled snapshot < user cache < project override <
first-party table), with provenance and staleness on every record.
`sources.py` refreshes that catalog and is the only module in the package
allowed to open a socket.

**Cost model (`cost.py`).** The piece everything else calls, and the one that
differs from a conventional LLM router. A stateless API prices a request as
`in*rate_in + out*rate_out`. In a persistent agent session that is wrong,
because the prefix is re-sent every turn: a token admitted to the main context
is billed once as a cache write and then again, as a cache read, on every
remaining turn. That term is what makes an output token cost roughly 7.8x its
sticker price over a long session, and it is why "use a cheaper model" is often
a losing move: the prompt cache is model-scoped, so switching invalidates the
prefix.

**Reports (one module per command).** Each owns its own `argparse` parser and a
`main(argv) -> int`. They share the data model and the cost model and nothing
else. This is why adding a command is a one-row change: there is no shared
report framework to extend.

**Utilities (`stats.py`, `render.py`, `filters.py`, `config.py`).** Leaves of
the dependency graph, deliberately. `stats.py` owns every quantile in the
project: one estimator, linear-interpolated, so `median()` and `quantile(0.5)`
agree by construction and the "p90" is never quietly the maximum. `render.py`
owns formatting, including the rule that a cost below a cent must not print as
`$0.00`. `filters.py` owns the half-open `--since`/`--until` window, so two
adjacent windows partition the data exactly. `config.py` is the single registry
of every setting and environment variable, and `adder config` prints not just
the value but the layer that set it, which is the half that matters when two
machines disagree about the same transcripts.

**Aggregation (`doctor.py`).** The one module that calls other reports. It runs
each check, prices the finding, and sorts by dollars at stake. It computes
nothing itself: every measurement comes from the module that owns it, because a
summary command that reimplements the cost model is a second answer waiting to
disagree with the first.

**Dispatch (`cli.py`).** A table of `Command` rows mapping a name to a module,
imported lazily so `adder live` does not pay to import the A/B harness. The
dispatcher never re-declares a command's flags, so `adder <cmd> --help` is always
the module's real parser.

## Load-bearing invariants

Each of these is enforced by a test in `tests/repo/test_invariants.py` or by a module's
own tests, not just by convention. They are listed in `CLAUDE.md` as rules; here is why
they exist.

- **Deduplicate by `message.id`.** Without it, every figure in the project is
  ~1.78x too high.
- **No network outside `sources.py`.** A cost report that silently depends on a
  third party breaks in CI at the worst moment. `tests/repo/test_invariants.py`
  walks the AST of every module and fails on a networking import.
- **No runtime dependencies.** The tool must run from a bare checkout on any
  machine with Python 3.10+, including one with no reachable package index.
- **Read-only over user data.** The tool never writes under `~/.claude`. Output
  goes to stdout or to a path the user named.
- **Feasibility gates before profitability.** A model that cannot hold the
  context is not an option at any price, so the window check runs before the
  break-even math.
- **A recommendation must clear its own overhead.** A routing turn re-reads the
  whole context, which at 500K tokens on Opus is ~$0.25. If the modelled saving
  is smaller than that, the honest output is "just do it".

## Why it is shaped this way

The alternative design — one big analysis object that computes everything and a
thin CLI over it — was rejected because the reports disagree with each other on
purpose. `context.py` exists to *falsify* a claim that `debt.py` was built on;
`regret.py` exists to evaluate whether `horizon.py`'s estimator is worth its
complexity; `simulate.py` exists to check whether `savings.py`'s composition
model is an approximation that holds. Keeping them independent means a module
can contradict another one and the disagreement shows up as a failing test
rather than as a merge conflict inside a shared object.

## Where things live

The package is a tree, not a directory. Seven layers, and an import may only
point *down* the list, and `tests/repo/test_structure.py` fails the build if
one points up. See `docs/structure.md` for the rules and how to add to it.

```
adder/
  __init__.py    version only
  __main__.py    `python -m adder`
  util/          no domain at all: render, stats, risk, text
  pricing/       what a token costs: prices, catalog, providers, registry,
                 cost, bt, sources (the only networked module), data/
  core/          reading a session off disk: trace, filters, settings, shapes
  measure/       read-only reports, grouped by subject
    spend/       where the money went: trace, sessions, debt, export, …
    window/      what fills the context: context, cache, carry, prefix, tools
    session/     how one session behaves: live, horizon, quality, effort
  decide/        measurement -> choice
    route/       where a task runs: classify, policy, select, cascade, frontier
    track/       the record of past choices: outcomes, ledger, dispatch
    guard.py     what a tool call is allowed to admit into the context
    handoff.py   what is worth carrying into a fresh session
  evaluate/      did the advice hold up
    replay/      re-run recorded work under a counterfactual: simulate, bench
    claims/      re-derive a published number: validate, savings, verify
    doctor.py    runs every check and ranks the findings by dollars
  cli/           dispatcher, command table, help, completion, config
tests/           mirrors the package tree, directory for directory
docs/            the reasoning; the README summarises it
scripts/adder    launcher for a checkout; delegates to adder.cli
.claude/         agents, hooks, skills — part of the product, tracked
```

The split that matters most is `core/trace.py` against
`measure/spend/trace.py`. The first reads and deduplicates transcripts and is
imported by nearly every other module; the second is the `adder trace` report.
They were one file, which meant the PreToolUse hook (which runs on every
submit) imported an argparse parser and a printing routine to ask what a
session had cost.
