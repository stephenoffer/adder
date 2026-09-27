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

adder reads the Claude Code transcripts already on your disk and prices what
your context is costing you. Turned on, it also refuses the tool calls that
waste the money and routes the work that is left to the cheapest model that can
do it.

## Install

```bash
pip install adder-cli        # the distribution is `adder-cli`; bare `adder`
adder auto on --full         # on PyPI is an unrelated 2014 package
```

Two lines, and the second is the one that changes the bill. No account, no API
key, no model calls, no network, no runtime dependencies.

Not ready to let a hook refuse a tool call? `adder auto on --shadow` runs the
whole decision, records what it would have refused, and refuses nothing.
[Getting started](docs/getting-started.md) walks through the first run.

## What it is worth

Replaying 33,192 recorded turns across 118 real sessions:

| | spend | |
|---|---|---|
| as it actually ran | $7,888 | |
| after `adder auto on --full` | **$2,567** | 3.1x cheaper, hands off |
| if you also restart when it says to | $1,233 | 6.4x cheaper |

Nothing in the middle row asks you to work differently. The third needs one
thing from you, and the tool is explicit about which. These are re-priced
replays of turns that really happened, not projections; the null configuration
reproduces the measured bill to within 0.0% before any ratio against it means
anything. ([benchmark.md](docs/benchmark.md))

Every dollar here comes from one machine's history, dominated by one workload.
Your absolute numbers will differ. Run `adder savings` before you believe any
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
| what a token really cost | 1.0x | 2.0x | 5.0x | **7.8x** | 16.2x |

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
sum. This one reports the carry, which on the history above was 5.7x the sum.
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
