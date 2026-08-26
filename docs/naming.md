# The name

## What it means

A full adder has two outputs. One is the sum, which is the answer you asked for.
The other is the carry: the bit that does not fit in this column and has to be
paid in the next one.

Every cost tool in this space reports the sum. `adder` reports the carry.

That is the whole thesis of the project stated in one word. An output token is
billed once when it is written, then again as cached input on every turn that
follows it. The write is the sum. The 340 re-reads are the carry. On the
transcripts this was built against, the carry was **5.7x** the sum, and no
dashboard anywhere showed it.

The name is also a snake, which is the traditional entry fee for a Python
project.

## What it replaced, and why

The project was called `llm-router` until 2026-08-14.

That name was wrong in three ways, and each one cost something:

1. **It named the least interesting output.** Routing is one thing this tool
   emits, at the end, after the measurement. `policy.decide` refuses to emit a
   recommendation at all when the modelled saving does not clear the cost of the
   routing turn itself. A tool whose headline feature declines to fire is not a
   router. It is a measurement tool that occasionally recommends something.

2. **It implied a request path.** "Router" describes middleware: something
   between your process and the model, seeing traffic, making a call. This
   reads JSONL files off local disk after the fact. It has never held a request,
   never held an API key, and cannot. `docs/cost-model.md` opens with a section
   titled "Why this is not a model router". When the docs need a section
   disclaiming the project's own name, the name is the problem.

3. **It sat in the most crowded shelf on PyPI.** Every LLM-adjacent gateway,
   proxy, and load balancer is called some variant of router. The name conveyed
   membership in a category the tool is not in.

`adder` names the operation instead of the category, and the operation is the
part nobody else does.

## The distribution name

```bash
pip install adder-cli
```

The PyPI name `adder` is taken by an unrelated package: a single release, 13KB,
uploaded 2014-08-28, described as "An AI library", untouched since. It is a
candidate for a [PEP 541](https://peps.python.org/pep-0541/) name request, and
one may be filed. Until that resolves (if it ever does), the distribution ships
as `adder-cli`.

Everything else is plain `adder`:

| | |
|---|---|
| `pip install adder-cli` | the distribution name, and the only place the suffix appears |
| `import adder` | the import name |
| `adder live` | the console script |
| `python -m adder` | the module entry point |
| `./scripts/adder` | the checkout launcher |

A distribution name that differs from its import name is common enough
(`pillow`/`PIL`, `beautifulsoup4`/`bs4`) that it is not worth contorting the
codebase to avoid.

## What did not get renamed

- **`route-t0`, `route-t1`, `route-t2`.** The tier subagents in
  `.claude/agents/`. These are named for what they do, which is routing, and
  they are referenced by name in users' existing project configs. Renaming them
  would break installs to no benefit.
- **Generic uses of "router" in prose.** Where the docs say "a router that ranks
  on the text board picks the wrong model", the word means the category, not
  this project. Those stayed.
