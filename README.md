<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/stephenoffer/adder/main/docs/assets/adder-logo-dark.png">
    <img src="https://raw.githubusercontent.com/stephenoffer/adder/main/docs/assets/adder-logo.png" alt="" width="112" height="112">
  </picture>
</p>

<h1 align="center">adder</h1>

<p align="center">
  <a href="https://github.com/stephenoffer/adder/actions/workflows/ci.yml"><img src="https://github.com/stephenoffer/adder/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/adder-cli/"><img src="https://img.shields.io/pypi/v/adder-cli.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/adder-cli/"><img src="https://img.shields.io/pypi/pyversions/adder-cli.svg" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
</p>

**Your coding agent's bill is bigger than your dashboard says, and most of it is
avoidable.**

adder reads the transcripts your coding agent already writes to disk and prices
what your context is costing you. On Claude Code it also refuses the tool calls
that waste the money and routes the work that is left to the cheapest model
that can do it.

## Install

In Claude Code, as a plugin, with nothing else to install:

```bash
claude plugin marketplace add stephenoffer/adder
claude plugin install adder@adder
```

That gives you `/adder:doctor`, `/adder:context` and `/adder:route`, the tier
agents, and the read guard at `certain` once the plugin loads: it refuses a
re-read of a file the context already holds unchanged, and nothing that would
admit anything new. Run `/adder:init` once for the rest. It offers `--full`
and installs the Haiku `Explore` override, which a plugin cannot ship because
plugin agents are namespaced and cannot replace a built-in.

From PyPI, for the command line, or if you would rather not use plugins:

```bash
uv tool install adder-cli    # or pipx; the distribution is `adder-cli`, since
adder auto on --full         # bare `adder` on PyPI is an unrelated 2014 package
```

`auto on` installs the hooks, the agents and the same skills (as `/adder`,
`/adder-doctor`, `/adder-context`, `/adder-init`), prints every change first,
and `adder auto off` undoes it. If the plugin is also installed, `auto on`
notices and writes no second copy of the hooks.

Either way, no account, no API key, no model calls, no network, and no runtime
dependencies. The line that changes the bill is the one that turns on
enforcement.

Not ready to let a hook refuse a tool call? `adder auto on --shadow` runs the
whole decision, records what it would have refused, and refuses nothing.
[Getting started](docs/getting-started.md) walks through the first run.

**Not on Claude Code?** Codex CLI, Gemini CLI and OpenCode transcripts are
found where those agents keep them, so `adder doctor` needs no setup there
either; `adder trace codex` picks one. Aider, an OpenTelemetry export or any
log carrying token counts works too — point it at the file. To ask adder from
inside one of those agents, register its MCP server:
`adder mcp --print-config codex` (or `gemini`, `cursor`) prints the snippet.
Enforcement is the part that needs a hook able to refuse a tool call, so
`adder auto on` declines rather than installing something inert.
[agents.md](docs/agents.md) has the split.

## What it is worth

Replaying 46,687 recorded turns across 103 real sessions (September 2026):

| | spend | |
|---|---|---|
| as it actually ran | $8,757 | |
| after `adder auto on --full` | $7,897 | 1.11x cheaper, hands off |
| if you also restart when it says to | **$3,131** | 2.8x cheaper |

The middle row asks nothing of you, and it is worth about a tenth of the bill:
refused re-reads of files the context already holds, and large tool results
sent to a subagent where that is cheaper than carrying them. The third row is
where the money is. It needs one thing from you, a restart every few dozen
turns, and the tool says when. At the pessimistic corner of the three inputs no
transcript can settle, 2.8x becomes 2.4x.

These are re-priced replays of turns that really happened, not projections.
Replayed with nothing changed, they reproduce the measured bill to +0.0%, which
shows the bookkeeping is sound and says nothing about the counterfactuals; the
counterfactuals are what [benchmark.md](docs/benchmark.md) spells out. An
earlier version of this table read 3.1x and 6.4x. That replay let a refused
read hand the model's own output to a subagent and priced each subagent at a
fraction of a cent, and both errors ran in the tool's favour.

On a Pro or Max plan you are not billed these dollars; they are what the same
tokens cost at API list price. The saving arrives as headroom instead: 99% of
the input tokens in each five-hour window here were context already read once,
so a shorter context is more work before the window closes. `adder limits`
reports your windows in tokens rather than dollars.

Every dollar here comes from one machine's history, dominated by one workload.
Your absolute numbers will differ. Subagent output is also a lower bound here:
from mid-September Claude Code stopped writing the final usage record for most
subagent messages, and `adder trace` counts how many. Run `adder savings` before you believe any
number on this page — re-checking it on your own transcripts is the entire
point of the tool.

## Start here

| You want to know | Run |
|---|---|
| Just tell me what's wrong | `adder doctor` |
| Which sessions cost the most? | `adder sessions` |
| Stop telling me and start doing it | `adder auto on --full` · `adder auto status` |
| What is this session costing me right now? | `adder live` |
| Where has all my money gone? | `adder trace` · `adder savings` |
| Do this here or delegate it, and to what? | `adder policy "<task>"` |
| What model should run it, across every vendor? | `adder pick "<task>"` |
| Did last week's change actually work? | `adder verify --since DATE` |

Run `adder` on its own for a short version of this list, `adder help <command>`
for one command's flags, and `adder help --all` for the other fifty-odd.
Reports over your history share the same window flags (`--since DATE`,
`--project NAME`, `--session ID`), and most take `--json`.

## The one idea

You don't pay for a piece of text once. You pay for it on every turn after it
appears.

When the agent writes 1,000 tokens, you're billed for writing them. Then they
join the context, so you're billed again to re-read them on turn 2. And turn 3.
Write something with 340 turns left and you pay for it 341 times.

| turns remaining | 0 | 50 | 200 | 340 | 759 |
|---|---|---|---|---|---|
| what a token really cost, Opus 5 | 1.0x | 2.0x | 5.0x | **7.8x** | 16.2x |
| on Opus 5.5, whose cache reads are half price | 1.0x | 1.5x | 3.0x | 4.4x | 8.6x |

Your usage dashboard shows you the 1.0x column. Two things follow, and they are
the whole tool: the expensive decision is rarely which model you used but what
you let into the context and how early, and a 600-turn session is not one long
session. It is the same context, re-read 600 times.

[How it works](docs/how-it-works.md) is the ten-minute version of that
argument, from the arithmetic through to what the guard and the router do with
it.

## Documentation

| | |
|---|---|
| **[Getting started](docs/getting-started.md)** | install, first run, what each report answers |
| **[How it works](docs/how-it-works.md)** | the carry math, the guard, the router, in one read |
| **[Commands](docs/commands.md)** | every command, grouped, with flags |
| [docs/](docs/README.md) | the reasoning behind every number, page by page |

## The name

A full adder has two outputs: the sum, and the carry — the bit that doesn't fit
in this column and has to be paid in the next one. Every cost tool reports the
sum. This one reports the carry, which on the history above was 5.4x the sum.
It's also a snake, the entry fee for a Python project.
([naming.md](docs/naming.md))

## Contributing, security, license

`pip install -e ".[dev]"` then `make check`: ruff and pytest, exactly what CI
runs. Read [CONTRIBUTING.md](CONTRIBUTING.md) first; agents working in this repo
should read [CLAUDE.md](CLAUDE.md), the binding version of those rules. The bar
is about numbers rather than style: anything that moves a reported figure needs
the measurement behind it in the PR.

adder reads your transcripts, which contain your source code and prompts. It
writes nothing under `~/.claude/projects`, sends nothing anywhere, and holds no
credentials; [SECURITY.md](SECURITY.md) has the threat model.
[MIT](LICENSE) · [CHANGELOG.md](CHANGELOG.md)
