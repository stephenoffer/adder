# Architecture

One paragraph: transcripts on disk are parsed into per-turn records, priced by a
cache-aware cost model, and then read by a set of independent report modules. A
dispatcher maps a command name to one of those modules. Nothing else happens.

```
~/.claude/projects/**/*.jsonl
          |
          v
   trace.iter_file          parse + DEDUPLICATE by message.id
          |
          v
   Turn / Session           the only shared data model
          |
    +-----+---------------------------+
    |                                 |
    v                                 v
 prices.py / catalog.py            cost.py
 what a token costs, by date       what a turn costs, given the cache
    |                                 |
    +-----+---------------------------+
          |
          v
   report modules            live, debt, context, cache, quality, horizon,
   (one per command)         savings, verify, validate, regret, simulate, ab,
          |                  policy, outcomes, classify, select, models
          v
       cli.py                name -> module, lazily imported
```

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
code, merged in layers — bundled snapshot < user cache < project override <
first-party table — with provenance and staleness on every record.
`sources.py` refreshes that catalog and is the only module in the package
allowed to open a socket.

**Cost model (`cost.py`).** The piece everything else calls, and the one that
differs from a conventional LLM router. A stateless API prices a request as
`in*rate_in + out*rate_out`. In a persistent agent session that is wrong,
because the prefix is re-sent every turn: a token admitted to the main context
is billed once as a cache write and then again, as a cache read, on every
remaining turn. That term is what makes an output token cost roughly 7.8x its
sticker price over a long session, and it is why "use a cheaper model" is often
a losing move — the prompt cache is model-scoped, so switching invalidates the
prefix.

**Reports (one module per command).** Each owns its own `argparse` parser and a
`main(argv) -> int`. They share the data model and the cost model and nothing
else. This is why adding a command is a one-row change: there is no shared
report framework to extend.

**Dispatch (`cli.py`).** A table of `Command` rows mapping a name to a module,
imported lazily so `adder live` does not pay to import the A/B harness. The
dispatcher never re-declares a command's flags, so `adder <cmd> --help` is always
the module's real parser.

## Load-bearing invariants

Each of these is enforced by a test in `tests/test_cli.py` or by a module's own
tests, not just by convention. They are listed in `CLAUDE.md` as rules; here is
why they exist.

- **Deduplicate by `message.id`.** Without it, every figure in the project is
  ~1.78x too high.
- **No network outside `sources.py`.** A cost report that silently depends on a
  third party breaks in CI at the worst moment. `tests/test_cli.py` walks the
  AST of every module and fails on a networking import.
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

```
adder/
  cli.py         dispatcher; the command table
  trace.py       parsing, dedup, Turn/Session, parse cache
  cost.py        the cache- and context-aware cost model
  prices.py      first-party Claude rates, date-aware
  catalog.py     provider-agnostic model data, layered and provenanced
  sources.py     catalog refresh — the only networked module
  data/          bundled catalog snapshot, shipped in the wheel
  <report>.py    one per command; each owns its own parser
tests/           one module per adder module
docs/            the reasoning; the README summarises it
scripts/adder    launcher for a checkout; delegates to adder.cli
.claude/         agents, hooks, skills — part of the product, tracked
```
