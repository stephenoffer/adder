# Security Policy

## Reporting a vulnerability

Do not open a public issue. Report privately through GitHub's
[Report a vulnerability](https://github.com/stephenoffer/llm-router/security/advisories/new)
form, or email the maintainer listed in `pyproject.toml`.

Expect an acknowledgement within 3 business days and an assessment within 10.
If a fix is warranted, it ships in a patch release with a GitHub Security
Advisory and a CHANGELOG entry crediting you, unless you would rather not be
named.

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | yes |
| < 0.1 | no |

Until 1.0, security fixes land on the latest minor only.

## Threat model

This is an offline analysis tool, which shapes what counts as a vulnerability
here.

**In scope:**

- Anything that causes the tool to **write to, modify, or delete** files under
  `~/.claude/` or elsewhere outside a path the user explicitly named. The tool
  is read-only over user data by design; a violation of that is a bug of the
  highest severity.
- Any **network call outside `router/sources.py`**, or any call inside it that
  fires without an explicit user command, or that ignores `LLM_ROUTER_OFFLINE=1`.
- **Transcript content leaking off the machine** by any path — a report that
  posts data, a log that writes prompts somewhere world-readable, an error
  message that dumps source code into a third-party service.
- **Path traversal** through a `root`, `--log`, `--out`, or `--from` argument
  that escapes the intended directory.
- Code execution from parsing an untrusted transcript, catalog, or captured
  source file (`--from name=path`).
- Dependency-supply-chain issues in the release pipeline.

**Out of scope:**

- A malformed transcript causing a crash or traceback. Annoying; not a
  vulnerability. File a normal bug.
- Inaccurate cost figures. File a normal bug with the measurement.
- Denial of service from pointing the tool at an enormous directory.
- Anything requiring an attacker to already have write access to your
  `~/.claude` directory or your checkout — at that point they have your source
  code and your prompts regardless.

## What the tool touches

For anyone auditing before running it:

- **Reads:** `~/.claude/projects/**/*.jsonl` (or a `root` you pass), plus its
  own catalog cache.
- **Writes:** stdout, and only paths you name explicitly (`--out`, `--log`).
  The bundled snapshot in `router/data/` is never written at runtime.
- **Network:** only `rt models refresh` / `router.sources`, only when invoked
  directly, and never when `LLM_ROUTER_OFFLINE=1` is set.
- **Credentials:** none. The tool has no API key, reads no keychain, and sends
  no authenticated request.

## Handling transcript data

Transcripts contain your source code, file paths, and prompts. When filing an
issue, redact or synthesise — never paste raw transcript content into a public
issue, and never commit a real transcript as a test fixture.
