# Using adder with something other than Claude Code

adder does two different jobs, and they have different reach. Reporting works
on any agent that records token counts. Enforcement needs a hook that can
refuse a tool call, and today that means Claude Code.

Saying so up front is the point of this page. The alternative is what the tool
used to do: write three Claude Code hooks onto a Codex machine, print a success
message, and deliver none of the saving it had just promised.

| | works on |
|---|---|
| every report (`doctor`, `trace`, `sessions`, `savings`, `carry`, `context`, `cache`, `anomaly`, `compact`, `debt`) | any format below |
| placement advice (`policy`, `pick`, `place`, `classify`) | any, once `harness` and `ladder` are set |
| enforcement (`auto on`, the read guard, the routing clause) | Claude Code |
| asking from inside the agent | Claude Code through the skills; anything that speaks MCP through `adder mcp` |

## Asking adder from inside the agent

In Claude Code the skills do this: `/adder:doctor` from the plugin, or
`/adder-doctor` after `adder auto on`. Everywhere else, adder runs as an MCP
server over stdio. Install it where the agent can start it, then print the
snippet for that agent and paste it into the file the first line names:

```bash
uv tool install adder-cli
adder mcp --print-config codex     # ~/.codex/config.toml
adder mcp --print-config gemini    # ~/.gemini/settings.json
adder mcp --print-config cursor    # ~/.cursor/mcp.json
```

It prints the snippet and writes nothing. `auto on` is still the only command
that edits a file you did not name, and a server able to reach it would be a
second one.

The server exposes nine tools: `doctor`, `live`, `trace`, `sessions`,
`savings`, `context`, `policy`, `pick` and `handoff`. That is every report an
agent is likely to want mid-task, and nothing that writes. Nine is also a
budget. An MCP tool definition sits in the agent's context on every turn of
every session, which is the charge `adder memory` prices, so each description
is kept to one line and the other fifty-odd reports stay one shell command
away.

Each call runs in a fresh process. Run in process, one memoised lookup, the
model your newest session ran on, would last as long as the server did, so a
server started in the morning would still price the afternoon's sessions at
the morning's model.


## Agents it finds on its own

Four agents keep their transcripts in a known place, and adder looks there
without being told. If you run one of them, `adder doctor` works the same way
it does on Claude Code.

| agent | name to pass | where it looks |
|---|---|---|
| Claude Code | `claude` | `~/.claude/projects` |
| Codex CLI | `codex` | `~/.codex/sessions` |
| Gemini CLI | `gemini` | `~/.gemini/tmp` |
| OpenCode | `opencode` | `~/.local/share/opencode/storage/message` |

With nothing configured, adder reads Claude Code's directory if it holds any
sessions, and otherwise the first of the others that does. To pick one, give
its name wherever a report takes a directory:

```bash
adder trace codex
adder sessions gemini --since 7d
```

An agent's own transcript is read by `adder/core/native.py`, not the
per-record adapters below. The model and the token counts are written to
different lines, so no single line can be priced by itself. Codex also writes
each `token_count` event twice. Summing them all prices a session at about
double what it cost: on the five Codex sessions this was checked against, the
naive sum came to 1.96x to 1.99x Codex's own running total. adder counts a call
only when that running total moves, and its sum matches the total exactly.

OpenCode records reasoning tokens next to output tokens and does not say
whether the output count already includes them. adder bills the output count
alone and keeps reasoning in a separate field. Where it is wrong, it
under-counts.

## Other logs it reads

`adder/core/ingest.py` normalises a usage record into the same `Turn` the
Claude Code parser produces, so a foreign log gets the same carry analysis and
the same reports.

| format | recognised by | example source |
|---|---|---|
| `claude-code` | `type: assistant` with a `message.usage` | `~/.claude/projects/**.jsonl` |
| `anthropic-api` | `type: message` plus `usage` and `model` | an agent built on the Anthropic SDK |
| `openai-chat` | `usage.prompt_tokens` | Chat Completions; Aider |
| `openai-responses` | `usage.input_tokens` with no Anthropic cache field | the Responses API |
| `gemini` | `usageMetadata` | the Gemini API |
| `otel` | any `gen_ai.usage.*` attribute | an OpenTelemetry export |
| `generic` | bare `input_tokens` / `output_tokens` | a LiteLLM proxy log, a loop you wrote |

Point a report at the file or directory and it reads whatever is there:

```bash
adder trace ~/logs/agent.jsonl --by model
```

`adder doctor` counts the files it read of each shape in its `formats` check,
and names any it could not read at all. Check that line before trusting
anything else on the page, because a mis-read is silent:
the adapters disagree about whether the cached prefix is counted inside the
input total, and reading an OpenAI record with Anthropic's convention roughly
doubles the input side while every number still looks plausible.

A directory adder cannot parse reports that fact rather than reporting zero.
"No priced turns" and "there are eleven files here and none of them carry a
token count" are different problems, and only one of them is fixed by waiting.

## Tell it what is driving

If adder is reading one agent's own directory (it found it by itself, you set
`root` to it, or you named the agent on the command line, as in `adder doctor
codex`), the harness follows: a Codex root means `codex` unless you say
otherwise. `adder config harness` prints `derived from root` when that
happened. Anywhere else, set it:

```bash
export ADDER_HARNESS=codex            # or gemini-cli, opencode, aider, openhands, custom
```

or, to keep it, the `harness` key in `~/.claude/adder.json` for the machine or
`.adder.json` for one repository. `adder config --init` prints a template and
`adder config harness` shows what is in effect and where it came from. There is
no command that writes a setting for you: `adder auto on` is the only thing in
the tool that edits a file you did not name, and it stays that way.

This is not cosmetic. Some harnesses pin the main conversation to one vendor:
under Claude Code the session is a Claude model, under Codex an OpenAI one.
Left at the default, `adder policy` on a Codex machine offers Claude models as
main-session candidates and refuses OpenAI ones, which is exactly backwards.
`adder doctor` flags the mismatch when the logs it just read disagree with the
setting.

| harness | main session | subagents | pre-tool-call hook |
|---|---|---|---|
| `claude-code` | Anthropic | yes | yes |
| `codex` | OpenAI | yes | no |
| `gemini-cli` | Google | yes | no |
| `aider` | any | no | no |
| `openhands` | any | yes | no |
| `opencode` | any | yes | no |
| `custom` | any | yes | no |

## The model and the ladder are read, not configured

You do not tell adder which model you run. The `model` setting, when unset, is
the model your newest session under `root` ran on. adder reads it from the end
of the newest few transcripts, skipping subagent files, once per command.
`adder config model` prints `your newest session` as its source when that is
where the answer came from.

The dispatch ladder follows from that. The shipped ladder is Claude, because
that is what the measurements were taken on, and a harness pinned to another
vendor cannot dispatch to any rung of it. So on Codex or Gemini CLI, with no
`ladder` set, adder derives one from the model catalog:

| rung | model |
|---|---|
| T2, T3 | the model your sessions run, or that vendor's best-rated model if they run another vendor's |
| T1 | the best-rated model from that vendor with input under $3.50/Mtok that is cheaper than T2 |
| T0 | the same under $1.50/Mtok, cheaper than T1 |

Each rung only ever gets cheaper going down, and a rung with nothing cheaper
below it repeats the one above. The price bands are the ones `adder models
ladder` reports against. On this machine's Codex sessions it chose
`T0=gpt-5, T1=gpt-5.6-luna, T2=gpt-5.3-codex`. `adder doctor` prints the
ladder in effect and says when adder chose it.

Ratings come from public arenas, not from your workload, so a derived ladder
is a starting point. Override any rung and the rest stay on the vendor:

```bash
export ADDER_LADDER=T0=gpt-5-nano
```

Effort levels follow the model rather than the ladder: OpenAI stops at
`high`, so a T3 rung on an OpenAI model is dispatched without the `xhigh`
Anthropic would have taken.

A harness adder has never heard of is describable without a code change:

```bash
cat > harnesses.json <<'JSON'
{"harnesses": {"myagent": {"main_session_org": "openai", "supports_hooks": true}}}
JSON
export ADDER_HARNESSES=harnesses.json
export ADDER_HARNESS=myagent
```

`supports_hooks` is what `adder auto on` checks before it writes anything. It
defaults to false for anything undescribed, because installing a hook that
cannot fire is worse than declining to install one: the reports keep working
and read as proof that enforcement is on.

## What you give up without the guard

The measured 1.1x in [benchmark.md](benchmark.md) is activation: the guard
refusing reads that admit nothing new, plus the routing clause. None of that is
available without a pre-tool-call hook, so on another harness adder is a
measurement tool that makes recommendations, and acting on them is yours.

That is not nothing. `adder savings` on the same history attributes $5,840 of a
$7,888 bill to context already paid for once, and the three largest levers in
[levers.md](levers.md) are all things a person does rather than things a hook
does: session length, effort, compaction. The honest split is that the levers
adder can pull for you are Claude Code only, and the larger ones were always
yours.

## Pricing follows the provider, not the harness

A hosted endpoint with no prompt cache re-reads your prefix at full input rate
rather than a tenth of it, which makes the carry multiple roughly ten times
larger and moves the break-even from turn 50 to almost immediately.
[providers.md](providers.md) has the table; `adder debt --model <id>` prices
whichever model you name.
