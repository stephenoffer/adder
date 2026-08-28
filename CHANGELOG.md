# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Because this is a measurement tool, one extra rule applies: **any entry that
changes a reported figure names the measurement behind it.** A number that moved
without a stated reason is a regression, not a change.

## [Unreleased]

## [0.2.0] - 2026-08-27

- **Activation wrote a third-party hook into a repository by default, with a
  path from the machine that ran it.** `adder auto on` defaulted to project
  scope and wrote absolute paths for both the interpreter and the script:
  `/home/ray/anaconda3/bin/python3 /mnt/cluster_storage/.../pretooluse_read_guard.py`.
  `.claude/settings.json` is commonly tracked in git and under
  `bypassPermissions` the hooks *are* the guardrail, so the default put a
  third-party hook inside somebody's committed security perimeter -- and the
  command it wrote worked on exactly one machine. It also dropped `.adder.json`
  and a `settings.json.adder.bak` into the tree with no `.gitignore` entry
  anywhere.

  The default is now `--user`; `--project` is opt-in and prints what it is
  putting inside a tracked tree, including which of the files nothing ignores.
  The command form changed with it: a user-scope install writes
  `<sys.executable> -m adder.decide.hooks.<module>`, which drops the absolute
  *script* path so a moved checkout no longer strands the hooks, and a
  project-scope one writes `adder hook <name>` -- a new subcommand that resolves
  through PATH to each contributor's own console script, which carries its own
  interpreter in its shebang. Every entry now carries an explicit `"timeout": 5`
  instead of inheriting the 60s default on a hook that matches `Bash`: measured
  latency is 159ms flat, so the bound costs nothing and turns a hang into a
  skip. `guard.installed_in` matches all three command forms, because after the
  change it would otherwise have reported a working install as absent -- which
  reads as "nothing is preventing spend" while something is.

- **`guard_enforce=shadow`: the refusal decision, run in full, carried out on
  nothing.** Every advisory dollar in this tool is multiplied by
  `guard_advice_taken`, which is 0.5 and assumed on any machine that has not
  measured it, and enforcement then asks a user to hand refusal authority to the
  guard on the strength of that assumption. Shadow mode computes exactly the
  refusal `certain` would make, records it, and refuses nothing, so the trade is
  measured on this machine before anything is denied. It injects no message, so
  its overhead is zero and it does not spend the fire budget -- a ceiling
  applied here would truncate the measurement at `guard_max_fires` findings a
  session and still print as complete.

  It also records the evidence *against*: a shadow refusal the session went
  round -- the same target asked for again, or a duplicate `Read` refusal
  followed by the file arriving through the shell -- is the closest thing to
  proof that the refusal was wrong, since under enforcement that is the escape
  hatch firing and it costs a turn. `adder guard --shadow` prints the
  counterfactual saving, the contradiction rate, and a realised figure that
  writes off every contradicted refusal whole. `adder auto on --shadow` turns it
  on.

- **`adder guard --last`: what the guard actually did in this session.** A
  refusal the user cannot inspect is a refusal they will turn off the first time
  they suspect it, and `guard_enforce=off` is one line. Every other report here
  is an aggregate over weeks; this one lists the findings and refusals of the
  most recent recorded session, identities only.

- **The classifier's defect wordlist leaked, so the failure direction
  inverted.** `_DEFECT` is a list of nouns, and six probes phrased as ordinary
  defect hunts named classes that were not on it: `find the security flaw`,
  `find the data corruption`, `locate the privilege escalation`, `find the auth
  bypass` and `locate the crash` all reached T0 at 0.85 confidence -- a
  whole-tree audit priced as a lookup, with exactly the silent failure the
  defect rule exists to prevent. Adding six words would have bought six probes.

  So the default inverted instead: an enumeration over a *singular* definite
  noun phrase abstains unless the sentence bounds it -- a path, a named symbol,
  a quoted string, or a noun that names one findable thing. The wordlist stays
  as an accelerator. Ordinary lookups are unaffected (`find the config file`,
  `locate the definition of process_batch`), and a plural target keeps the T1
  rule it already had.

- **`classify_terms`: the vocabulary a project has and the classifier does
  not.** On a real domain codebase the classifier abstained on twelve of twelve
  task phrasings from the repository's own tracker, every one at confidence 0.3
  -- so the routing decision cost its own overhead, twelve times, to arrive at
  "no change". A `cheap` term names something findable in this repository, which
  bounds a search over it; a `hard` term names work that is open-ended here.
  Declared in the project's `.adder.json` rather than learned, because
  `outcomes.Outcome` stores `task_hash` and never the task text and
  `track/similar.py` builds a MinHash sketch specifically so the terms cannot be
  recovered -- learning a vocabulary out of that would undo it. Every term that
  decides names itself in the verdict's reasons. `adder classify --terms`.

- **`adder ab --recall`: a quality signal that shares no code with the cost
  model.** `adder quality` and `adder verify` read the same transcripts, through
  the same parser, priced by the same cost model as the saving they are
  checking; if that model is wrong they are wrong in the same direction and
  agree with themselves. Cost is measured five ways here and quality -- the
  thing routing to a cheaper tier would actually lose -- was measured by the
  cost machinery. `adder/evaluate/replay/seeded.py` ships a source file with
  nine planted defects and scores a reply against them by string match;
  `tests/evaluate/replay/test_seeded.py` asserts the module never imports from
  `pricing`, `core.trace`, `core.shapes` or `measure`. The prompt does not say
  how many defects there are, because a model told to find nine reports nine.
  The misses are named, not summarised: "Haiku found 6 of 9" is a number, and
  "Haiku missed the unbounded retry" is a decision about what to route to it.

- **Per-tool floors and ceilings.** One `guard_min_tokens` and one
  `guard_max_fires` served every tool the guard watches, and the tools are not
  alike: measured here, `Bash` returns a p90 of 1.2K tokens over 2,490 calls in
  a session and `Read` 5.9K over 58. Against a shared 2,000-token floor the
  first is almost never priced and the second usually is; against a shared
  15-fire ceiling the first can spend the whole budget before the second has
  said anything. `guard_min_tokens_by_tool` and `guard_max_fires_by_tool`
  override per tool, both default to empty -- so behaviour is unchanged until
  something is set -- and a per-tool ceiling can only lower the global one, not
  raise it. `adder guard --floors` prints each tool's own distribution and the
  floor it implies, derived rather than shipped: a floor at a tool's p90 prices
  its top decile by construction.

- **`adder doctor` says which numbers are not yours.** Everything here that
  adapts -- the size model, `p_fail`, the uptake term, savings read as a trend
  -- needs weeks of transcripts, and below that each silently falls back to a
  prior measured on one workload while the report reads identically. Two new
  checks: `history` names the days of transcripts and lists the features
  currently running on the shipped prior, and `prior` promotes the
  prior-vs-yours table out of `adder guard` into a finding, naming every tool
  the shipped size prior is more than 2x out on. On the machine this was written
  for that table showed `Agent` 13.5x out, in a report nobody opens unless they
  already suspect the guard.

- **`adder savings` and `adder bench` disagreed 4x on one lever with nothing
  between them.** Both were right: `savings` measures the pool -- every token
  admitted to a context that already held it -- and `bench` prices the guard,
  which is a hook with a memory, a budget and a rule against refusing twice.
  Two correct numbers in two reports read as one wrong number. `guard.capture_gap`
  states the difference item by item with this machine's own settings in it, and
  both reports print it. `bench` also names the `guard_enforce` level it priced
  against, and says so when that is not the level `adder auto on` installs --
  which was the whole gap: the report described the advisory guard while the
  install command writes `certain`.

- **Concurrency is named rather than left to be discovered.** The duplicate rule
  keys on a path and its mtime, and in a tree several agents are working in, the
  mtime moves for reasons this session had no part in -- so the rule correctly
  declines and the lever silently reports less than it is worth. It fails
  towards saying nothing, never towards a wrong refusal, which is exactly why
  nothing would have surfaced it. `guard.concurrent_sessions` counts the
  condition and `adder guard` says the figure is a floor when it holds.

- **The overflow note named the weaker of two equal models.** When nothing on
  the ladder holds the read, `policy.decide` reports which model has the largest
  window. Sonnet and Opus both hold 1M tokens and `max()` returns the first
  maximum, so the note said `claude-sonnet-5` on every overflow -- and that
  sentence is the one a reader follows when they go and split the read by hand.
  Ties now break toward the more capable rung.

- **Two test modules read the developer's real `~/.claude`.** CLAUDE.md forbids
  it and it bit during this change: `Settings.resolve` reads
  `~/.claude/adder.json`, so on a machine that had run `adder auto on` the guard
  was enforcing throughout `tests/decide/hooks/test_hooks.py`, two assertions
  about `additionalContext` failed for a reason that exists nowhere in CI, and
  the suite's own fires went to the real `adder-guard-fires.jsonl` where they
  skewed the uptake measurement. Setting `HOME` is not enough --
  `settings.USER_FILE` is `Path.home() / ...` evaluated at import -- so both
  modules now take `isolated_home`. `tests/decide/test_auto.py` gained an
  autouse fixture of its own: with `plan()` defaulting to user scope, a test
  that forgets to say which scope it means writes into whoever ran the suite.

- **The declared build floor could not build the package, and CI never
  installed what it built.** `build-system.requires` said `setuptools>=68`
  while `[project]` declared `license = "MIT"` and `license-files`, which are
  PEP 639 and need setuptools 77. Building with 76.1.0 dies on `project.license
  must be valid exactly by one definition`; 77.0.0 is the first that works, and
  the floor is now 77. Nothing caught it because pip builds in an isolated
  environment and resolves the newest setuptools there -- the floor is only
  exercised by `--no-build-isolation` or a pinned toolchain, and until then it
  was a false claim about what the package needs.
  `tests/repo/test_invariants.py` pins the floor against the metadata actually
  declared, in both directions.

  The `build` job checked filenames inside the wheel and stopped there. A
  subpackage missing from `packages.find`, a broken entry point, or a data file
  the code cannot locate once it is in `site-packages` all pass a name check and
  fail the user. Both workflows now install the wheel *and* the sdist into clean
  virtualenvs, run the installed console script from outside the checkout, drive
  every subcommand's `--help`, and assert the bundled catalog and the activation
  payload are readable where `pip` put them. Verified against a wheel with those
  files deliberately stripped: the old check passed it, the new one fails with
  the list of what is missing. `adder models list` is not that check -- without
  the snapshot it falls back to nine built-in models and still exits 0.

- **CI was red on two jobs, and one of them was a real bug in the tool.**
  The `build` job asserted the wheel carried `adder/cli.py`; the layer split
  turned that into `adder/cli/`, so the check failed on every build after the
  rename. It now names the package layout, and lists the two data files
  setuptools only ships because `package-data` says so -- a wheel missing
  `pricing/data/catalog.json` or `decide/agents/*.md` works from a checkout and
  fails after `pip install`, which is the failure worth catching here.

  The Windows leg of the matrix was failing on the CLI smoke step for a reason
  users hit too. Reports print `→`, `⚠` and `█`; a *redirected* stdout on
  Windows encodes in the ANSI code page, cp1252 has none of the three, and
  `adder help > out.txt` raised UnicodeEncodeError before printing a line.
  `run` now widens stdout and stderr to UTF-8 with `errors="replace"`, next to
  the BrokenPipeError handler and for the same reason: one place, one
  behaviour, every command. Reproducible off Windows with
  `PYTHONIOENCODING=cp1252`.

  Three tests were Unix-only rather than wrong. `test_memory_walks` bounded its
  hang with `signal.alarm`, which does not exist on Windows, so the file
  asserted nothing there -- the watchdog is a thread now. `test_filters`
  compared a `Path` against the string `/tmp/somewhere`. And the write-loss
  race now counts `FileNotFoundError` (the bug: another writer took the scratch
  path) separately from a blocked replace (Windows refusing a move while the
  target is held), because only the first one is what that test is about.

- **The measured advice-uptake now reaches the gate that assumed it.**
  `guard.uptake()` has estimated this for a long time and `adder auto status`
  has printed it, but nothing consumed it: every advisory saving in the tool was
  discounted by a flat 0.5, including on machines that had measured their own
  rate and were displaying it on screen. A measurement nobody acts on is the
  same failure as a router nobody invokes.

  `adder guard --learn` now caches the measurement to
  `~/.claude/.adder-uptake.json` and `Settings.resolve` reads it, so the
  solvency gate strengthens or weakens on evidence instead of staying anchored
  to a default nobody checked. The hook reads the cache rather than re-scanning
  transcripts before every tool call — the same split the size model already
  uses. An explicitly set `guard_advice_taken` still wins.

  **A floor of 10% keeps the loop open, and it is not caution.**
  `advice_taken` decides whether advice is worth saying at all, so a measured
  rate near zero silences the guard — and a silent guard records no fires, so
  nothing can re-measure it. Unfloored, the estimator seals itself shut on one
  bad week with no way back short of editing a config file nobody knows exists.

  Two smaller fixes came with it. `load_uptake` resolves its path *inside* the
  try: it runs inside `Settings.resolve`, which runs before every tool call, and
  an exception escaping there does not degrade the guard, it takes the tool call
  with it. And `adder guard` no longer tells the reader to lower
  `guard_advice_taken` by hand — correct advice while nothing consumed the
  measurement, and an instruction to redo something already done now that the
  gate picks it up itself. ([guard.md](docs/guard.md))

- **`adder routereval` now reports the comparison PGR cannot make.** PGR is 1.0
  at the all-strong endpoint by construction, and that endpoint is a single
  model, so no PGR-derived number can say whether the router beat simply picking
  one model and never routing. The published benchmarks report exactly that, and
  it is where several well-known routers fail. Added `gain vs best single` and
  `cost saved at equal quality`, named as those benchmarks name them so a figure
  here is quotable next to one from a paper. The baseline is the *better* of the
  two arms, not assumed to be the strong one, because a mix where the weak model
  wins would otherwise flatter every router scored against it. ParetoDist was
  deliberately not added: the frontier here is two fixed points plus an oracle,
  so the distance to it is the oracle regret already printed, under a second
  name. ([routing.md](docs/routing.md))

- **`adder limits` also reports the week.** The five-hour window was the only
  span covered, and the weekly cap is the other half of the constraint. The
  comparison is the heaviest *sliding* seven days rather than a calendar week:
  the reset day is not published, so an anchored total would depend on where the
  anchor was put, while a sliding peak is a property of the workload. Labelled a
  proxy where it is printed — the cap itself is metered in compute hours, which a
  transcript does not record.

- **The guard can now carry out its own advice instead of asking for it**
  (`guard_narrow`, off by default). At `full`, a refusal costs a turn: the guard
  says "read at most 116 lines of it (`limit: 116`)", the model reads that,
  agrees, and issues the bounded call. The outcome is the bounded read plus one
  round trip spent arriving at advice that was already priced.

  A `PreToolUse` hook may return `updatedInput`, so the guard now substitutes the
  bounded call rather than demanding it. `Read` gains a `limit`, `Grep` a
  `head_limit`; both are read-only and both substitutions are a strict subset of
  what was asked for. `Bash` is still refused rather than rewritten — appending
  `| head` to a command this tool did not write can change its exit status or cut
  a `&&` chain, and rewriting somebody's shell is not a bounded operation.

  The field was verified against the shipped client rather than the docs, which
  do not state whether it exists on this event, what happens when it is absent,
  or what it travels with. Claude Code 2.1.238 answers all three, and the third
  answer is why this is opt-in: `Hook satisfied user interaction for <tool> via
  updatedInput, bypassing permission prompt`. A substitution can suppress a
  prompt, and a smaller result than the one requested is not the same as an
  authorised one. It is reachable only where the guard was going to refuse
  outright, so enabling it relaxes a denial and can never permit something the
  guard would have been silent about — asserted in
  `tests/decide/test_narrow.py`, which fails if a substitution appears at `off`
  or `certain`.

  No reported figure moves. The saving is booked against what actually ran, the
  bounded read, not against the whole read a refusal would have prevented, and
  `Verdict.action` reports `narrow` rather than `deny` so the ledger cannot
  confuse the two. ([guard.md](docs/guard.md))

- **The router now conditions on the task, not just the tier.** `p_fail` was
  scoped per (project, tier), which averages over task kinds that share nothing:
  one T0 number covering both "where is the retry logic" and "make the scheduler
  preemptible" is too timid for the first and too bold for the second, and which
  error you get depends on that week's mix. It is now also conditioned on the
  task, via the recorded runs whose vocabulary resembles it.

  Similarity is a MinHash sketch of terms and adjacent bigrams with Jaccard
  between sketches — no model, no dependency, no network, and the task text is
  not stored (each slot is a minimum over the whole term set, so the terms are
  not recoverable). `adder similar "<task>"` shows the working: per-tier
  escalation rate over the nearest runs, against the tier-wide rate the gate
  used before.

  The asymmetry the ladder already runs on is applied again, one level deeper,
  because a rate over four neighbours is far easier to push around than one over
  four hundred runs. A neighbour estimate with mass behind it replaces the
  tier-wide rate in either direction; a thin one may only *raise* `p_fail`, which
  can at worst decline a downgrade; a thin optimistic one is discarded. Where
  there are too few neighbours, or the log predates sketches, the gate uses
  exactly the estimate it used before — the neighbour half is additive and its
  absence costs nothing. No reported figure moves as a result: this changes which
  rung gets chosen on future dispatches, and `adder calib` is what scores whether
  the sharper estimate is better calibrated out of sample.

  `adder outcomes import` writes a sketch for every delegation it recovers, and
  `adder outcomes record` takes `--task` to sketch a hand-filed row.

- **`adder limits`: the carry, in the unit a subscription actually charges.**
  Every figure in the package was in dollars, which is the wrong unit for a Pro
  or Max user by a change of kind rather than a scale factor — they do not get a
  bill, they get a lockout. This reconstructs the five-hour metering window from
  turn timestamps and reports, per window, tokens read against tokens that were
  new, the carry share, the burn rate, and the within-window slope: what a turn
  late in a window costs against one early in it.

  Three things a transcript cannot know are labelled rather than guessed. The
  boundary rule is not published, so the reconstruction is called one everywhere
  it is printed. The cap is not published in tokens, differs by model, and drains
  faster in peak hours, so no capacity is asserted — the comparison is against
  the heaviest window on this machine's own record, which is a floor under
  capacity because it was served, not a limit. And a straight-line projection of
  the burn rate is an under-estimate while the slope exceeds 1, which is said
  rather than corrected. The slope itself is the one number quoted without
  qualification: it comes from the same dedup-corrected counts as everything
  else and does not depend on the boundary rule at all.

- **`adder auto`: the tool now acts, and the headline moved 1.6x → 3.1x.**
  `adder bench` has said for months that installing this saves 1.6x while
  following its reports saves 6.4x, and that "nothing in this repo enforces
  it". That gap was the product. It is now three things.

  *Activation is one command.* `adder auto on [--full]` writes the three hooks
  into `settings.json` and the thresholds into `.adder.json`, prints every
  change before making it, keeps a `.adder.bak`, and is reversed exactly by
  `adder auto off`. The previous route was a JSON block you were expected to
  merge by hand plus a `--learn` you were expected to remember; a saving gated
  on a copy-and-paste is a saving most people never get. It starts no
  background process — the hooks run on events the harness already fires.

  *The guard can refuse.* `guard_enforce` is `off`, `certain` or `full`.
  `certain` refuses only what admits nothing new — a read of a file already in
  this context, or one this session wrote — which cannot lose information at
  any price. `full` also refuses a large read that has a strictly cheaper
  equal, and names it. Replayed over 34,144 recorded tool calls, net saving
  goes **$93 advisory → $182 at `full` → $513 at the shipped thresholds**, and
  the share resting on the uptake assumption goes **100% → 4%**. That second
  number is the point: a refusal is not discounted by a guess about whether
  advice is taken, because the call does not happen.

  *Refusing is survivable by construction.* Never twice for the same target, so
  the worst case of a wrong refusal is one wasted turn; always with the reason
  and the cheaper call; and the PreCompact hook now clears the read and write
  memory, because "already in this context" stops being true the moment the
  context is rebuilt. Without that last one an enforcing guard refuses reads of
  content the model no longer has, which is the one way it could cost more than
  it saves.

  The enforcing thresholds are swept, not picked: **floor 800 tok, gate $0.10,
  ceiling 200 fires**. The $0.25 gate exists to stop the guard *interrupting*
  over small change and a refusal is not an interruption, so dropping it finds
  $200 more; the 15-fire ceiling was sized for a guard that talks, and 200
  costs no latency and is worth $26. The floor is the one real trade — 300
  tokens finds $138 more and takes the share of tool calls that stop to parse a
  transcript from 8% to 39% — so 800 ships and
  `adder auto on --full --tune` re-derives it against your transcripts,
  preferring the quieter setting whenever the noisier one is within 5%.

  Two new claims in `validate.py`, because two README numbers moved:
  `activating_it_pays_more_than_installing_it` (3.1x, and it fails if
  enforcement ever stops beating advice) and
  `enforcement_removes_the_assumption` (fails if the prevented share falls
  below half). The first one earned its place immediately by failing on its
  own first draft: told to enforce but left at the advisory thresholds, the
  ladder answers 1.6x, because activation is the level *and* the thresholds
  and neither half does anything alone.

  *Activation installs the agent files too.* `adder bench` prices the hooks and
  the tier definitions together — the hooks alone are 2.5x and the pair is
  3.1x — so an `adder auto on` that wrote hooks and not `.claude/agents/` was
  quoting a multiple it did not deliver. It now copies `Explore.md` and the
  three tier agents, and never overwrites one you already have: a file that
  differs is listed and left alone, because silently replacing somebody's
  `Explore` during a command about cost changes what a built-in does in a file
  they did not know we knew about.

  The aggregate and subagent-brief findings refuse under `full` as well, for
  consistency rather than for money: measured, they add about a dollar, because
  at an 800-token floor the size gate has usually already spoken about the same
  shape. The inconsistency was the reason to fix it — a guard that refuses a
  single large read but only *mentions* a command shape on its two-hundredth
  call is hard to predict.

  `bench` splits its bottom rung in two, because the delegation threshold and
  the restart cadence are enforceable by different things — a hook can refuse a
  read and nothing here can restart a session. Only the cadence keeps the
  asterisk now.


- **Fixed: `pip install adder-cli && adder auto on` installed nothing that
  worked.** The hooks and the tier agents lived under `.claude/`, which
  `MANIFEST.in` prunes, so the wheel carried none of them. Activation from a
  PyPI install therefore wrote three hook entries pointing at files that did not
  exist inside `site-packages`, copied zero agent definitions, and reported a
  plan that looked complete — `agent_plan` skips a source it cannot find, so the
  missing half was silent as well as absent. Everything measured in the entry
  above was reachable only from a git checkout, which is not who the README's
  install line is written for.

  The payload moved into the package: `adder/decide/hooks/` (modules, so
  `packages.find` ships them without anybody remembering a data glob) and
  `adder/decide/agents/` (named by `package-data`). Each hook resolves the
  directory holding `adder` four levels up from itself, which is the same
  arithmetic in a checkout and in `site-packages`. `.claude/hooks/*.py` remain as
  forwarding shims so a `settings.json` written before this keeps working, and a
  test caps them at 30 lines so no decision moves back into a file the wheel does
  not carry.

  Activation also writes `sys.executable` rather than the string `python3`. A
  hook command runs through the user's shell, where `python3` is whatever is
  first on PATH — on macOS routinely 3.9, which cannot import this package at
  all, and which fails as a hook error on every tool call rather than as anything
  about versions.

  Four assertions in `tests/repo/test_invariants.py` now stand where this failed
  silently: the payload lives inside the package, every file activation names
  exists where activation looks for it, the hooks are a package rather than a
  directory, and anything in the payload that is not a module is covered by a
  `package-data` glob.

- **The router now runs at the moment of delegation, not only when asked.**
  `adder policy` has always combined the classifier, the measured failure rate
  per tier and the session cost model into "what should this run on", and it has
  always been reachable only by typing it. A router nobody invokes routes
  nothing: the tier agents `auto on` installs are a *fixed* assignment, so
  whatever the caller names is what runs, and if it names nothing the work runs
  on the session model.

  The PreToolUse guard already sees every `Task`, so it now asks. One hook, one
  message, one ledger entry — the tier clause joins whatever the guard was going
  to say about the size of the subagent's return, because two sentences about
  one call are carried for the rest of the session twice. `adder/decide/delegate.py`
  owns the decision; `guard_route=false` turns it off.

  It is advice and never a refusal. The guard may refuse a read; refusing a
  `Task` would refuse the largest lever in the tool on the strength of a
  classifier that abstains by design. It also stays quiet when the call already
  names a routed agent, when the two rungs are one model spelled two ways
  (`claude-opus-5` against `claude-opus-5[1m]` — a "switch" that throws away a
  model-scoped cache to buy nothing), when nothing measured justifies going down
  a rung, when the sentence costs more than the switch saves, and after it has
  said it once.

  Cross-vendor substitutes are deliberately absent from that sentence. `pick`
  ranks ~500 models against LMArena Elo and `policy` reports the cheaper ones,
  but a Claude Code `Task` cannot be dispatched to Qwen, and naming a model at
  the moment nobody can act on it is how a router stops being read. The arena
  signal reaches this decision through the ladder, which `models ladder` diffs
  against the live catalog.

  `guard.replay` switches the clause off, and that is not a detail: `bench`
  already prices what routing delegated work to a cheaper tier is worth, as its
  own rung, so counting the sentence arguing for it as well would book one dollar
  twice and inflate every ratio taken against the null. Re-measured after the
  change, the null still reproduces the measured bill to -0.0% and
  `adder validate` holds every claim, activation included at 3.0x on a corpus
  that has grown to 124 sessions since the 3.1x in the README was taken.

- **The README leads with the mechanism instead of the measurement.** It opened
  with what the numbers were and worked backwards to what the tool does, which
  answers "is this real" before "why would I run it" — the wrong order for a
  page most people read once. It now goes why the bill is bigger than the
  dashboard says, what the hooks refuse while you work, how the router picks a
  tier, and only then what each of those was worth, in 284 lines rather than 328
  with a section on the router that did not exist before.

- **The first thing a new install prints is no longer a dead end.** On a machine
  with no transcripts `adder doctor` printed `No sessions under <root>.` and
  stopped, which reads as a broken tool rather than an empty one, and `adder
  live` said only "No transcript found for this directory yet." Both now say
  which of the two it is: every report here reads history you have already paid
  for, the guard needs none, and `adder auto on --full` is useful before the
  first session. `auto on` on a fresh machine also stopped claiming it "learned
  result sizes from 0 local tool calls", which is a success message describing a
  fallback.

  The four skills and the `adder-init` install instructions no longer hardcode
  the author's home directory, which was in `allowed-tools` and in every command
  they printed.

- **The guard now sees the aggregate, which is where most of the money is.**
  Its per-call view was structurally blind to the largest single admission
  channel, and no amount of calibration could fix that: across 222 local
  transcripts, **32 session-and-shape pairs exceed 20K cumulative tokens and
  together account for 19.7% of every Bash result token in the corpus** (1.38M
  of 7.0M). The biggest is `sed -n 'A,Bp'` -- a *bounded* read the guard is
  right to wave through every single time -- at **246 calls, a 513-token
  average, and 126,222 tokens into one session**. Every one of those calls
  really was small. The guard now counts what each command shape has admitted,
  bounded calls included, and says so once when the running total is worth
  more than saying it. It claims half the carry rather than all of it: the
  tokens already admitted cannot be un-admitted, and only the calls still to
  come can be avoided. `adder validate` checks the premise rather than quoting
  it: aggregated by shape, the ones that clear the threshold hold **47% of all
  Bash result tokens against 4%** in calls big enough for a per-call gate to
  see. A workload of a few big calls will fail that claim, and should — there
  the aggregate rule is not earning its state.
- **A subagent cannot be advised to use a subagent.** `Task` and `Agent` were
  priced through `placement_cost`, so the guard would tell a subagent call
  "~$0.20 delegated to a subagent" -- modelling delegating a delegation, and
  quoting a saving from an option that had already been taken. They are now
  priced against a **brief**: the subagent's reads are already outside the main
  window, so the only lever left is the size of what comes back, and the
  comparison is against a 1,000-token return.
- **`adder guard` leads with whether the guard is installed at all**, and
  `doctor` fails when it is not. An uninstalled hook, a broken hook and a
  correctly quiet one are indistinguishable from the outside, which makes this
  the most valuable line in the report -- and the first thing it said on the
  author's own machine was that nothing had ever been preventing spend here.
  `--install` prints the settings block rather than writing it, the precedent
  `adder config --init` set and one that matters more for a hook.
- **`--explain` takes a path as well as a command**, and a `Read` sized from
  the filesystem is no longer described as a prior. A byte count is a stronger
  measurement than any quantile over past calls, and labelling it "no local
  history for this shape" reported a measurement as a guess.
- **The size model's path and staleness resolve at call time**, like
  `guard.Settings` and `render.color_enabled` before them. Captured at import
  they named a file under the user's home that no test could redirect, so the
  suite read and wrote the developer's own model.


- **`adder guard --replay` -- what this guard, at these settings, would have
  done to transcripts already paid for.** Replayed over 29,464 tool calls in 80
  local sessions it would speak **236 times (0.80% of calls)**, cost a
  transcript parse on **1.44%**, and be worth **$85 against $2.58 of advice**.
  Quoted as an upper bound and labelled as one: the horizon is the projected
  one rather than the turns that really remained, the saving assumes the advice
  is acted on, and a call it talked someone out of would have changed
  everything after it.
- **Building the replay immediately found a bug that had inflated the guard's
  own value 12x.** Its eight largest findings were duplicate reads of PNG
  screenshots worth $25-$31 each, because `read_estimate` sized every file as
  `bytes / 4`. An image is billed by its dimensions and capped near 1,600
  tokens however many megabytes it is on disk, so a 1MB screenshot was being
  priced at 250,000 tokens. Corrected, the guard's modelled value on this
  workload falls from **$1,053 to $85** -- which is the number worth having,
  because the first one was a confident wrong number and those are the only
  kind this project treats as a defect. Archives, fonts, media and databases
  now get no estimate at all rather than a guess.
- **The routing tiers and the guard now state one return budget**, and a test
  fails if they drift. The guard prices a `Task` against 1,000 tokens; the
  tier files told a subagent "a few lines" and Explore told it "500 words". A
  measurement sets it: returns here run 193 tokens at the median and 3,723 at
  p90, and the budget goes where the tail is.
- **The advice names a number.** "Bound it" cannot be acted on without doing
  the arithmetic, and the arithmetic depends on where in the session you are:
  the guard now solves for the largest read that stays under its own floor and
  says so -- 583 lines with 50 turns left, 39 with 900. The same reason the
  trigger is a cost and not a token count, applied to the advice.

### Added

- **The guard's last assumption is now measurable.** `guard_advice_taken` is
  the one term the solvency gate rests on and the only one nothing measured.
  Fires are recorded (a shape, never a command; a basename, never a path) and
  `uptake()` asks the transcript what happened next: were later calls of that
  program bounded more often than earlier ones, was the duplicate file read
  again. Both halves are formats this project writes itself -- nothing parses a
  record shape it did not author. Below ten judged findings the report says the
  assumption stands; above it, `validate` and `doctor` switch to the measured
  rate on their own, so the solvency claim moves on evidence rather than
  staying anchored to a default. The first version matched later calls on the
  full command *shape*, which can never see the improvement it measures --
  `cat f` is what gets advised about and `cat f | head -20` is what compliance
  looks like.
- **A PreCompact hook that re-learns result sizes while the session is already
  paused.** The size model is refreshed when somebody remembers to run
  `--learn`; compaction is the moment that fixes it, since it only happens in a
  long session, a stale model costs most there, and the context is already
  stopping. It **prints nothing** -- no `hookSpecificOutput` at all -- so it
  cannot inject tokens mid-compaction and does not depend on what a PreCompact
  hook is allowed to return. A no-op unless the model is actually stale.
  `adder guard --install` now emits both hooks.

### Added
### Added

- **`adder memory`: the always-loaded prefix, priced per file.** `adder prefix`
  measured that a session opening is ~74% cache read and the conclusion —
  restarts are cheap — was over-read into "the size of the floor does not
  matter". A floor token is read once per *turn*, not once per session, and
  unlike a tool result it is never compacted away: compaction rebuilds the
  prefix from the same file, so its survival term is 1.0 forever. Measured
  here, 1,000 resident tokens cost **$0.31 per session** — $5.59 across the 18
  sessions this project has on record, $32.64 in a user-level file that all 105
  load. Scope decides which count applies, and pricing a project file against
  every session on the machine over-states it several fold. That is the unit an
  edit to `CLAUDE.md` should be priced in (`--what-if`). The report separates resident text from
  load-on-demand text — a skill's `description` is resident, its body is not,
  so a 40,000-token skill library costs less than a 3,000-token instruction
  file — and names what is duplicated, stale, unindexed, or over-long. What it
  cannot see (system prompt, tool schemas: 93% of the opening context here) is
  reported as `unaccounted` rather than attributed to a file someone is about
  to edit.
- **`adder reread`: content admitted to a context that already held it.** A
  file read on turn 8 is still resident on turn 140; reading it again buys
  nothing and pays its whole carry a second time. **$6.88 across 31 identities**
  on this machine. The report refuses to conflate two cases that look identical
  in a transcript: a *redundant* copy (byte-identical to one already resident,
  recoverable in full) and a *refresh* (the result changed, the call was
  justified) — reporting them together would tell someone their test runs are
  waste. For things re-learned across many sessions it prints a **note budget**:
  the largest resident note that still beats re-reading them, which comes out at
  4–46 tokens here. That is the arithmetic behind advice this repo does not
  otherwise give — *do not write it down* — and it exists because "just put it
  in CLAUDE.md" is free only if nobody prices the note.
- **`adder compact`: compaction as a trade with a threshold.** It pays a rebuild
  and buys a smaller prefix on every remaining turn, so it is worth doing when
  `remaining_turns > kept * write_mult / (freed * read_mult)` — a few dozen
  turns at the measured multipliers. The common failure is therefore not
  compacting too often but carrying a full context for hundreds of turns: 9
  compactions on record (median survival **6%**, not the 35% the model
  conservatively assumes) netted **+$1,857**, against 18 sessions that never
  compacted at all, worth **$718**. That second figure is simulated turn by turn
  against what actually happened, because a compacted context *refills* while an
  un-compacted one is pinned against the ceiling; pricing freed tokens as a
  constant saving over 348 turns over-stated it by 16%.
- **`adder handoff`: what may cross a restart.** The objection to the restart
  lever has always been "I would lose the context", and nothing here answered
  it. The crossing point is exact — at a 500K context with 300 turns left it is
  **467,000 tokens**, so the constraint on a brief is what you can usefully say,
  not what you can afford to carry. The report says that in those words instead
  of printing a budget someone might try to fill, and goes *negative* near the
  end of a session, where the honest reading is "do not restart" rather than
  "write less". It also lists what the brief has to name — files edited,
  commands re-run, reads ranked by re-establishment cost — recovered from tool
  inputs alone, never from message text.
- **`adder live` ends with a verdict, not a warning.** It now prices compacting
  and restarting against this session's own horizon, cache behaviour, and
  measured opening, and prints the better of the two only when it beats carrying
  on (`Context hygiene: restart — worth ~$55 over the ~350 turns expected to
  remain`). Ties go to carrying on: both alternatives destroy information that
  nothing in this repo prices. The same verdict replaced the generic "starting
  fresh is the largest saving" line in the `UserPromptSubmit` hook, which was
  true, unpriced, and therefore ignored.
- **`adder doctor` gained `memory`, `reread`, and `compact` checks**, each
  delegating to the module that owns the measurement. On this machine `compact`
  is now the largest single finding.
- **`adder validate` gained three claims**: memory is carried rather than
  written (measured 537x), compaction keeps less than the 35% the model assumes
  (measured 6%, so the assumption is conservative in the safe direction), and a
  brief can cross a restart (281K tokens at the median context).
- **`adder-context` skill**: the compact/restart/carry-on decision, the re-read
  rule, and what memory costs, written for an agent to act on. Its description
  costs 76 resident tokens per session by the measure `adder memory` applies to
  everything else.
- **[docs/context.md](docs/context.md)**: why these four reports belong together.


- **The parse cache defaulted off on the one path built to use it.**
  `load_sessions(use_cache=...)` defaulted to `False` while the `cache` setting
  defaulted to `True`, so memoization only happened where a caller had
  remembered to ask -- and the callers that most needed it had not.
  `horizon.load` is the one that mattered: it is reached from `live.analyse`,
  which **both hooks call**, so every prompt submission and every guarded read
  re-parsed every transcript on the machine to fit a distribution that moves by
  one session a day. Measured over 222 local transcripts: **2,339ms cold
  against 81ms warm, 29x**. The parameter now defaults to the setting;
  `use_cache=False` still forces a cold parse for the tests that assert on
  parsing.
- **The fitted horizon is cached on disk (2,089ms to fit, 4ms to read back).**
  Bounded by time rather than by a content fingerprint, deliberately: the
  transcript tree changes on every turn, so a content-keyed cache would
  invalidate constantly and never hit, while the statistic it holds is a
  distribution over ~100 sessions. An empty fit is never cached -- doing so
  would hide the first real session for an hour.
- **The PreToolUse guard went from 2,136ms to 139ms on a guarded read**, and
  from 74.7ms to 43.7ms on a tool it has no opinion about -- against a 32.5ms
  floor that is the Python interpreter starting. Three changes: the two caches
  above, a literal tool-name check *before* importing `adder` (the import costs
  27ms and the hook runs on every single call), and `os.path` instead of
  `pathlib` for the one path it computes at module scope. Latency is not
  dollars, but a two-second hook is one people uninstall, and an uninstalled
  guard saves nothing.


- **`adder validate` ran the benchmark replay twice, and it is the most
  expensive computation in the file by two orders of magnitude.** Two claims
  each called `bench.run`, which takes ~250s -- 206s of that is `corner_sweep`,
  which evaluates the worst case by replaying the whole workload at each vertex
  of the uncertainty box. The command whose entire job is to let someone
  re-check the foundations took ten minutes. Memoized on the session map:
  **249s then 0.0s** for the second claim.


- **A bound that names a number is a size, not a free pass.** `is_bounded`
  answers "capped by construction", and the guard was treating a `yes` as a
  reason to stay silent -- so `sed -n '1,600p'`, bounded to six hundred lines
  and about six thousand tokens, was waved through and returned 6,079. **45
  supposedly-bounded calls in the corpus returned over 3,000 tokens.** Measured
  over 16,727 calls carrying an explicit line bound, output runs **11.4 tokens
  per line at the median and 35.6 at p90**, and `lines x 11.4` predicts the
  real result to a median absolute error of 83 tokens. A bound now also *caps*
  a learned estimate: `cat huge.log | head -50` inherits `cat`'s 40K-token
  history through the program backoff, and fifty lines is fifty lines. Median
  prediction error over the holdout falls from 85 to **68 tokens**.
- **The shipped `PRIOR` was invented, and wrong in the expensive direction on
  every line.** `WebFetch` was quoted at 12,000 tokens p90 against a measured
  **595 -- a 20x over-statement** -- and the generic fallback at 3,000 against a
  measured 322. That is the same failure the size model was written to remove;
  it had simply moved from the hook into the default, where a machine that has
  learned nothing yet would be interrupted constantly. Re-derived from the
  population the guard actually predicts (unbounded calls only) across 222
  transcripts, and `Grep`, `Glob` and `Task` -- which have no observations here
  -- now inherit the pooled fallback rather than a number someone liked the
  look of. The consequence is deliberate: on a fresh machine nothing without
  local evidence clears the floor, so only `Read` is guarded, and `Read` is
  sized off the filesystem rather than predicted.
- **`SizeModel.learn` counted a different population than `PRIOR` describes**,
  so `adder guard` compared a prior derived from unbounded calls against an
  average over all of them, and the two disagreed for reasons that had nothing
  to do with the machine being different.

### Fixed
### Fixed
### Fixed
### Fixed

- **An unknown tool's input is no longer quoted anywhere.** `reread.identity`
  fell back to embedding a tool's JSON input for tools it had no shape for, and
  those inputs can be prose written for a human — which then appeared in report
  rows and, through the advisor hook, in a context. Unknown tools are now
  identified by a hash of their input, and the brief builder skips them
  entirely: a hash is noise in a brief and a leak at worst.

- **`adder blend`: order a queue so shared prefixes stay warm.** Resource-aware
  batching groups work by resource profile before running it. This was rejected
  once on the grounds that the engine belongs to somebody else -- which skipped
  the point, because the submission order is yours and the prefix cache is
  billed to you by the token. Building it surfaced a result that runs against
  the intuition: the ordering saving is **not monotone in the cache TTL**. Too
  short and even a grouped run goes cold between its own members; too long and
  the prefix survives the interleaving without help. It peaks between the two,
  so the report sweeps the TTL instead of quoting one number.
- **`adder harvest`: could this work survive being interrupted?** Harvesting
  preemptible capacity was rejected once as a hardware argument. The
  transferable half is that preemptibility is a property of the *work*: cheap
  capacity is only cheap if an interruption is survivable, and a transcript
  records exactly how much context one would destroy. Prices the expected loss
  with and without a handoff, against the discount, and reports the interruption
  rate at which it stops paying. The conclusion matches the hardware version and
  for the same reason -- checkpointing is the deciding term, and here the
  checkpoint is a handoff summary rather than a file.
- **`adder verbosity`: how much of a rating is length, and what that costs.**
  Head-to-head preference data rewards long, heavily formatted answers somewhat
  independently of whether they are better. The established fix is to put style
  features into the Bradley-Terry regression as covariates. On a leaderboard
  that is a fairness correction; here it is a **cost** correction and a sharper
  one, because the property that inflated the rating is the property this tool
  bills you for -- once as output, then again as prefix on every later turn. The
  command fits the controlled model and prices the gap between the controlled
  and uncontrolled strengths per answer. `adder/pricing/style.py` holds the
  estimator, which recovers a planted length coefficient of 0.9 to within 0.03.
- **`adder speed`: the fast path bills at 2x -- did the speed arrive?** The
  multiplier has been in the price table since it was written and no report had
  ever asked what it bought. Audited from the transcripts, paired within model
  so a mix shift cannot fake a result. The wall-clock caveat is printed with
  every number: transcripts carry one timestamp per turn, so the only available
  clock includes tool execution and reading time, which means absolute
  throughput is understated for both paths and only the ratio is meaningful. On
  the machine this was written against, 0 of 29,612 timed turns had ever used
  it, so the report is the prospective calculation instead: $4.22 per median
  session, which is the output half of the bill rather than the whole of it.
- **`adder place`: locality against price, for a warm session.** Its context is
  cached against one model, so moving to a cheaper one discards the prefix and
  pays a cold read plus a fresh write. The command sweeps the whole catalog for
  the crossover -- `migration / (cost_per_turn_here - cost_per_turn_there)` --
  and prices the affinity being discarded as a number in its own right. Two
  gates come before price: a window that cannot hold the context is not a cheap
  option, and a provider that publishes no cache rate is assumed to have no
  cache, so its prefix is re-read at full input rate rather than a tenth of it.
- **`adder deadline`: whether the cheap, slow path is worth it.** Batch is half
  price and this tool had never recommended it, because nothing here knew what
  a deadline was. Compares four policies -- always cheap, always guaranteed,
  greedy, and progress-proportional -- with unfinished work charged at full
  rate, since otherwise the cheapest strategy is always the one that gives up.
- **`adder sched`: does how far a session has run predict what is left?** The
  mean-residual-life curve, summarised in turns per turn against an
  equal-length reference of -0.50. Reported as a two-way verdict on purpose;
  see Fixed below.
- **`adder design`: where to spend the next measurement dollar.** `adder ab` is
  the only thing here that spends money to produce data, and it was being
  allocated evenly across pairs -- which buys a ranking whose weakest link is
  the pair nobody sampled. Allocation is now proportional to information per
  unit of remaining uncertainty, and the report says when a pair would need
  more comparisons than the routing decision is worth, which is the answer that
  ends an experiment rather than extending it.
- **`adder routereval`: the router is now scored, not asserted.** Every claim in
  this repo was about cost, and none of them answered whether the routing was
  any good — a saving is trivial to produce by sending work to a cheaper model,
  and the question is what it cost in quality. The command computes PGR, APGR
  and CPT against a random-ordering baseline, which is the standard the routing
  literature settled on, so the numbers are comparable to published ones. Two
  additions to that standard: an **oracle ceiling** printed next to the score
  (APGR's maximum is set by the task mix, not by 1.0, and reading 0.75 as a C
  grade when it is a perfect score is the obvious misreading), and a **dollar
  axis** alongside the call-count axis. The second is not cosmetic for this
  workload: a strong call on a 190K context costs ~40x a weak call on an 8K one,
  so a router sending 30% of *calls* to the strong model can be spending 95% of
  the budget. The report names the gap when the two axes disagree.
- **`adder calib`: `p_fail` is scored out of sample.** Every escalation gate
  multiplies by this estimate and it had only ever been inspected — `outcomes
  calibration` printed the rate next to the data it was fitted on, which is a
  tautology. Scoring is now **prequential**: walk the log in timestamp order,
  predict each row from only the rows before it, reveal the outcome. The
  headline is the Brier **skill score against predicting the base rate**, so an
  estimator that adds nothing over a constant is reported as adding nothing
  rather than as a respectable-looking 0.18.
- **`adder frontier`: the cost-quality frontier, with domination decided by
  confidence intervals.** A model only outranks a cheaper one when its rating
  interval clears that model's. This makes the frontier *narrower* than one
  drawn on point estimates — the models it drops are the ones whose lead sits
  inside the noise, which are exactly the ones a price-and-rating table talks
  you into buying.
- **`adder cascade`: try-cheap-then-check, priced with the term the batch
  analyses omit.** Published cascade economics assume a failed attempt is
  discarded. In a session it is not: it stays in the context and is re-read on
  every remaining turn. On a 300-turn session that carry term exceeds the failed
  attempt's own generation cost, which is why the recommendation usually comes
  out as "cascade, but in a subagent". Verifier false-negative and
  false-positive rates are modelled separately because they cost different
  things — one ships broken answers, the other just wastes money.
- **`adder spec`: agent sessions read as search rather than as turns.** Probe
  scale and fan-out, the explore/formulate/validate mix and how much it
  interleaves, redundancy (probes repeating one already made, priced against the
  measured re-read pool), and how much a human interjection collapses probe
  volume. The redundancy figure is the actionable one: it is exact, it has no
  upside, and it is invisible in every per-turn view.
- **`adder cachesim`: replay the workload against a simulated prefix cache.**
  `adder cache` reports what the cache did; this answers what a differently
  configured one *would* do — hit rate against capacity, block size and TTL,
  with cold and capacity misses counted apart. Matching is prefix-anchored, as
  the hardware requires; relaxing that is the easiest way to write a simulator
  that reports a number nobody can reproduce. Everything it prints is labelled
  SIMULATED and nothing else consumes it as a saving.
- **`adder repro`: a manifest of everything a number depended on.** Hashes the
  four inputs — transcripts, price table, catalog, code — so "it was 6.1x and
  now it is 4.2x" is a diff instead of an afternoon. The digest deliberately
  excludes the wall clock and modification times, so two runs over identical
  data agree byte for byte and copying the data does not read as drift.
- **`adder/pricing/bt.py`: Bradley-Terry ratings that carry an interval.** The
  public boards are a batch MLE, not the Elo update rule people assume; Elo
  depends on the order games arrived in, so re-running it on a shuffled log
  gives different ratings, which is disqualifying for a tool whose whole claim
  is "re-run the measurement". Includes optimal k-tier partitioning by dynamic
  programming (pairwise data is under 0.1% dense, so tiers are estimable where
  per-pair comparisons are not) and an `indistinguishable()` query, which is the
  answer most of the time and one a scalar comparison cannot give.
- **Uncertainty machinery in `adder/util/stats.py`**: seeded percentile and
  paired bootstraps, permutation tests, Wilson and Newcombe intervals, an
  anytime-valid confidence sequence for the peek-until-it-looks-good failure
  mode, Benjamini-Hochberg (twenty `doctor` checks at alpha=0.05 is a 64% chance
  of one false finding), Hedges' g, a power calculation, and rank correlations.
  Every resampler takes an explicit seed defaulted to a constant, so two runs
  produce the same interval and a report can be diffed in CI.

### Fixed

- **`adder deadline`'s progress rule now guarantees the deadline it exists to
  protect.** Comparing progress against the elapsed-time line is not sufficient
  on its own: falling behind is recoverable, but running out of the steps in
  which the guaranteed path could still finish is not, so the switch has to
  happen before that point rather than at it. Without the override the policy
  missed roughly a fifth of windows it was supposed to guarantee. A second bug
  in the same simulation forced every policy -- including the always-guaranteed
  one -- onto the cheap path for its first step, because the minimum-run guard
  fired before there was a run to protect.
- **`adder sched` reports a two-way verdict instead of a third one it cannot
  support.** Three successive versions of the statistic each produced a
  confident answer that was an artefact. Correlating attained against remaining
  over pooled positions is mechanically anti-correlated -- inside one session
  the two sum to a constant -- and returned -0.75 on a workload built to be
  heavy-tailed, which would have advised the exact opposite of the truth. A rank
  correlation over thresholds reads "bounded" for every workload, because the
  truncated tail decides the ordering. Least squares over the full range hands
  the same decision to the deepest threshold through sheer leverage. What
  survives is a slope in turns per turn over the supported range, read against
  an equal-length reference of -0.50 -- and no heavy-tailed category, because
  every finite workload's curve turns down past the median length regardless of
  its tail. Four synthetic long-tailed workloads all summarised negative.
- **`blend.saving` snaps float noise.** Two orderings that are arithmetically
  identical can differ by 1e-16 through summation order, which printed as
  `-0.0` and read as "grouping made it worse".
- **The style fit recovers a coefficient it was given.** The first version
  alternated between the strength and coefficient blocks, which meant rounding a
  continuous residual back into a discrete winner to reuse the plain fit -- it
  discarded exactly the information the coefficients are estimated from and
  returned a length coefficient of 0.000 on data generated with one of 0.9. It
  is now a joint Newton fit with a ridge penalty, recovering 0.927.
- **`length_matters` is decided on an interval, not a threshold.** A cutoff of
  0.05 on the point estimate reported a length effect from a judge that had
  none, because noise put the coefficient at 0.061. Coefficients now carry a
  bootstrap interval and the claim requires it to clear zero. Collinear logs --
  where style never varied within a matchup -- are reported as unidentified
  rather than as "no effect", because those are different answers.
- **`adder speed` prices the premium against output only.** It multiplied the
  whole median session bill, which overstated the lever by 5.6x on local data
  ($23.84 against the correct $4.22): cached input is billed at the same rate on
  either path, so only the output half moves.
- **`adder place` no longer silently drops models the registry does not know.**
  It priced exclusively through the first-party registry, so any catalog-only
  entry -- a project override, a provider added last week, a hand-pinned row --
  was skipped without a word, and a user-supplied catalog produced an empty
  field and no explanation. Entries now fall back to their own published rates,
  pessimistically: no published cache rate is treated as no cache at all.
- **The Bradley-Terry interval no longer reports zero width on a swept log.**
  Resampling rows — what the public boards do, and correct on a large log —
  fails silently on a small one: six battles a single model swept have no
  outcome variation to resample, every resample refits to the same value, and
  the interval comes out at ±0. A tool whose purpose is to stop confident wrong
  numbers cannot answer "how sure are you" with "completely" after six
  observations. `fit_with_ci` now holds the matchup schedule fixed and redraws
  winners from the fitted probabilities; the row-resampling method is still
  available and still documented as degenerate on small logs.
- **`outcomes.evidence()` and `p_fail()` take an explicit `now`.** They decayed
  their recency weights toward the wall clock, so replaying any log older than a
  few half-lives collapsed the evidence mass and returned the 0.5 prior for
  every row — which looks like a calibrated coin and is really an admission that
  no data was used. It also made the function untestable to a fixed value, in a
  repo whose testing rules forbid wall-clock dependence. Found by building
  `adder calib`.
- **The cache simulator stores only whole blocks.** A trailing partial block was
  rounded up and treated as resident, which credited the next request with up to
  a full block of tokens that were never cached. The error scales with the block
  size, and it surfaced as a 1024-token block reporting a *higher* hit rate than
  a 16-token one -- backwards, since bigger blocks match less often and waste
  more at the boundary. Floor, not ceiling; the remainder is always a miss. On
  three days of local transcripts the block-size sweep now increases in cost
  monotonically ($3,691.92 at 16 tokens to $3,738.19 at 1024) instead of
  inverting.
- **`stats.wilson_interval` snaps its exact endpoints.** A clean run reported a
  lower bound of 2.8e-17 rather than 0, which the formatter rendered as
  "0.0000%" with no way to explain it.

### Changed

- **The guard's own message is capped at 90 tokens**, and its state is pruned by
  age as well as by count. Every fire is already charged against its saving, but
  a long path or command could turn one sentence into a paragraph the session
  then pays to re-read; and a count-only prune keeps two hundred dead sessions
  on a quiet machine while dropping live ones on a busy afternoon.
- **`awk 'NR<=N'` and `head -c N` are recognised as bounds.** The byte form is
  converted directly, skipping the tokens-per-line term, which is the weakest
  assumption in the estimate.
- **The latency defect is pinned by tests, not by memory**: that a tool the
  guard has no opinion on imports no `adder` module at all, that the hook keeps
  `pathlib` off its hot path, that `load_sessions` defaults its cache to the
  setting, and that `horizon.load` does not re-fit per call.


- **The prompt hook now charges for its own advice too.** It injects ~155
  tokens that are then carried for the rest of the session, and it had never
  been weighed against what it was arguing for. It clears easily and is
  expected to -- it fires at most twice a session and advocates a lever worth
  hundreds -- which is precisely why checking beats assuming: if it ever stops
  clearing, the message has grown too long or the threshold is too low. Both
  hooks now read one uptake assumption (`guard_advice_taken`), so the same
  sentence cannot be priced two ways.
- **Every routing tier states the same return budget as the guard prices
  against** (1,000 tokens), with a test that fails on drift. The tiers said "a
  few lines", Explore said "500 words", and the guard priced a `Task` against a
  third number.
- **`Read` honours `offset`.** A read from an offset with no limit runs to the
  end of the file, and the whole-file size over-states it by everything already
  skipped.



- **The package is a tree of seven layers instead of fifty modules in one
  directory, and the layout is enforced rather than described.** `adder/` had
  reached 50 top-level modules with `trace.py` -- the transcript reader every
  report depends on -- sitting between `tools.py` and `validate.py` with nothing
  marking it as the foundation. Modules now live in `util` < `pricing` < `core`
  < `measure` < `decide` < `evaluate` < `cli`, and an import may only point down
  that list. `tests/repo/test_structure.py` fails the build on an upward import,
  on a directory holding more than 12 Python files or 10 subdirectories, and on
  a test file that does not mirror the module it covers. No number moved:
  `adder trace --json` over the same 26,614 turns reports $6,307.96 before and
  after, with every scalar and the whole per-model breakdown identical, and
  `adder trace --verify` still passes all seven structural checks.
- **`trace` and `config` are each two modules now, split along the line between
  computation and command.** `core/trace.py` reads and deduplicates transcripts;
  `measure/spend/trace.py` is the `adder trace` report. `core/settings.py`
  resolves settings for the fifteen modules that read one; `cli/config.py` is
  the `adder config` report. The PreToolUse hook runs on every submit and was
  importing an argparse parser and a printing routine to ask what a session had
  cost. The foundation now carries no commands, and a test asserts it.
- **Imports inside the package are absolute.** In a tree this deep the dots in
  `from ...pricing.cost import turn_cost` are load-bearing and invisible, and
  they break when a file moves. Ruff's `TID252` enforces the absolute form.
- **Ruff gained the rules the style section had only asked for**: `TID` (import
  form), `N` (naming -- exceptions end in `Error`), `D100` (every module has a
  docstring), `ISC`, `PIE`, and `I002`, which requires `from __future__ import
  annotations`. `N806` is ignored on purpose: single-capital locals in the cost
  model are the symbols from the derivation in the docstring above them.

### Fixed

- **A stale parse cache can no longer fail a report.** `trace._cache_load`
  caught five specific exceptions, none of which was `ModuleNotFoundError` --
  and a pickled `Turn` names the module it came from, so moving that module made
  every existing cache unloadable and took the tool down with it. The cache is
  an optimisation and never an input to a number, so no failure to read it may
  fail the tool; the handler is now deliberately broad and the cache version is
  bumped to 6.

### Added

- **The guard stopped guessing. `adder/core/shapes.py` and
  `adder/decide/guard.py`.** The PreToolUse hook is the only thing here that can
  prevent spend rather than report it, and it decided using a list of substrings
  and one fabricated constant: any command containing `cat ` or `git log` was
  assumed to admit 15,000 tokens. Measured across 222 local transcripts (27,698
  answered tool calls), the calls it fired on return a **median of 143 tokens**
  -- 105x less -- and **89% of them came in under the guard's own 2,000-token
  floor**. It saw 9.7% of all Bash result tokens and matched **none of the 18
  largest results in the corpus**, because `for f in ...; do cat $f; done` and
  `wc -l a.ts b.ts` contain none of its patterns. Two entries in its
  "already bounded" list were actively wrong: `-n ` was there for `grep -n` and
  waved through every `sed -n '1,600p'`, and the list was matched against the
  whole string, so `head -1 f && cat huge.log` counted as bounded.
- **Sizes now come from what commands of that shape actually returned here.**
  Held out even-vs-odd over 23,228 Bash calls, the learned model's median
  absolute error is **85 tokens against the constant's 14,871**; it fires on
  **155 calls instead of 489**, at a median real size of **1,309 tokens instead
  of 143**, and flags **20 of the 26 results over 5,000 tokens** where the old
  matcher flagged 15. One third the interruptions, and more of the large calls
  caught. `adder validate` re-derives the comparison rather than quoting it.
- **Bounding is decided by parsing, not by searching.** Within a pipeline the
  last stage decides; across a `;` sequence every command must be bounded. A
  filter is not a limit. The parser is quote-aware and hand-written because
  `shlex` raises on the unterminated quotes real transcripts contain, and a
  parser that raises inside a PreToolUse hook is a guard that has silently
  stopped guarding. Quoting was worth real accuracy: a regex split cut
  `grep -vE "^warning|^\s+-->"` in half at the alternation inside its own
  pattern, producing **12,208 shapes from 27,643 calls** -- nearly all
  singletons, all below the evidence floor, so the guard fell back to the prior
  for almost everything. Quote- and heredoc-aware splitting brings that to
  **7,027 shapes over 167 programs**.
- **The guard charges for its own advice.** A fire injects `additionalContext`,
  which is admitted to the context and re-read on every remaining turn exactly
  like a tool result. It now refuses to speak unless `saving x P(taken)` exceeds
  the cost of carrying the message, where `P(taken)` defaults to 0.5 and is
  declared as the assumption it is (`guard_advice_taken`). Setting it to 0
  silences the guard, which is the correct behaviour for anyone who believes
  advice is never acted on. With it come one fire per command shape per session,
  a ceiling of 15, and a running per-session ledger. A new `validate` claim
  sweeps 240 combinations of size, horizon and model and fails if any emitted
  fire has non-positive expected value.
- **The certain saving nobody was taking: 19.2% of unbounded `Read` calls on
  text files re-read something already in the context** (44 of 229). The guard
  could not see it because it had no memory between calls. `GuardState`
  remembers each path with its mtime, so a re-read after an edit -- the correct
  thing to do -- is not flagged and an unchanged re-read is. No delegation to
  model, no horizon to forecast, nothing to trade off.

  Quoted at 27.4% across *all* unbounded reads until the image fix below: 138
  of the 182 duplicates in this corpus are screenshots, and an image is capped
  near 1,600 tokens whatever its byte size, so those are cents rather than
  dollars. The first number was true and misleading, which for a measurement
  tool is the same as being wrong.
- **`adder guard`** reports what the model predicts, what the guard decided,
  and what it has cost; `--learn` re-derives the size model, `--explain CMD`
  answers for one command including why it would stay quiet. A `doctor` check
  fails when the model was never learned or when the guard is not solvent, and
  `ADDER_GUARD_DEBUG=1` prints the tracebacks that every fail-open path
  otherwise swallows -- the failure mode where a working guard, an uninstalled
  one and a broken one all look identical. See [docs/guard.md](docs/guard.md).
- **The decision is testable at all now.** It lived in the hook file, which is
  loaded by path and had no unit tests behind it; the component whose failure is
  silent had the least testable shape in the project. It is a library function
  taking every varying input as an argument, with 57 tests against it and 21
  more through the hook.


- **`bench.py` -- what installing the tool is worth, separated from what obeying
  it is worth.** `adder plan` prices the cheapest regime an optimiser can find
  and reaches 10.7x, which is the right number for setting a target and the
  wrong one to quote to somebody deciding whether to install anything: it
  assumes they will do everything the tool says. `adder bench` replays the same
  turns across a ladder split at the line between what the software does
  unprompted -- the PreToolUse guard and the tier files in `.claude/agents/` --
  and what it can only recommend. Measured over 23,922 turns and 90 sessions:
  **1.57x installed with no behaviour change, 6.7x following the solved
  threshold and cadence**, against a $5,846 baseline the replay reproduces to
  +0.0%. Both numbers are quoted because quoting only the second would describe
  the orchestrator pattern rather than the tool.
- **The guard's token threshold is derived rather than assumed, and the
  derivation found that the dollar gate is not the binding constraint.** The
  hook fires on a cost ($0.25), so turning it into a read size is one division
  against the expected re-read count. On this workload that resolves to 1,500
  tokens, below the hook's own 2,000-token I/O floor -- so tuning
  `ADDER_GUARD_MIN_COST` here would be tuning a gate that never decides
  anything. `bench.guard_threshold` applies the floor after the gate, the way
  the hook does, and `tests/test_bench.py` parses the hook and fails if the two
  constants drift.
- **The 6.7x is quoted against a corner sweep of the three inputs no transcript
  can settle** -- summary ratio, `p_fail`, handoff size. The pessimistic corner
  is 3.4x, and the floor is set by the summary ratio: a delegated read that
  hands back 30% instead of 10% puts most of the avoided carry back in the
  context. `adder ab` remains the only test of that assumption.
- **Two `validate` claims so the README figures are re-measured rather than
  remembered**: *installing it pays before you obey it* (enforced rungs clear
  1.3x) and *the advice reaches 5x* (solved regime clears 5x at nominal
  assumptions). Both are workload-dependent and expected to fail where sessions
  stay short enough that there is little carry to remove.
- **[docs/benchmark.md](docs/benchmark.md)**, and a README section stating both
  multiples side by side.

- **`risk.py` -- the uncertainty layer, and the reason advice can now be
  declined for being uncertain rather than for being unprofitable.** Every gate
  in this repo compared two modelled costs at their midpoints. Three of the
  inputs to that comparison are estimates with real spread: remaining turns is a
  forecast off a heavy-tailed distribution, `p_fail` is a rate off a handful of
  logged outcomes, and the summary a delegated read hands back is a modelled
  ratio. The module supplies a regularized incomplete beta and Beta quantile
  written out by hand (no dependency, pinned against closed forms and a binomial
  identity in `tests/test_risk.py`), credible intervals over those inputs, an
  exact worst-case evaluation, and `p_cheaper`, the probability a recommendation
  is cheaper than not taking it.
- **The worst case is exact, not sampled.** Every cost function here is
  multilinear in its uncertain arguments, and a multilinear function on a box
  attains its extrema at a vertex, so enumerating 2^k corners *is* the
  minimisation. The claim is tested against a dense interior sweep rather than
  asserted.
- **`carry.py` -- what carrying a token in context actually costs, measured.**
  `admitted_token_cost` priced a token as one write plus `remaining_turns` reads
  at 0.10x. Two of those three terms were wrong. The realized multiplier is
  recoverable from the transcripts (`(uncached + 0.10*read + 1.25*write) /
  context`, token-weighted, first turns excluded) and measures **0.115x on this
  machine -- 1.15x the assumption**, so carry, already ~76% of spend, was being
  under-priced. Against that, a token does not survive to the end of the
  session: compaction evicts it, and `expected_reads` sums the per-epoch
  survival series instead of assuming it never happens.
- **A compaction detector that is not fooled by wobble.** 122 turns out of
  20,524 show a context drop; only 7 are auto-compactions. Counting all of them
  fitted a 4-turn compaction period, which would have priced a token admitted
  now at 16 re-reads instead of 348 and switched delegation off across the
  board. A compaction now requires the context to have been at 60% of the
  model's ceiling *and* to have lost half of itself; the 7 real events sit at
  999.5K-999.9K dropping to 4-6%.
- **`horizon.mean_remaining` -- the conditional mean, because cost is linear in
  it.** `remaining()` returns the median, which answers "how much longer will
  this run" and is the wrong number to multiply a cost by. `E[cost] = c * E[R]`,
  and on this machine the conditional mean is **1.15x the median (351 vs 305
  turns at turn 0)**. The median stays for display. Added with it:
  `quantile_remaining`, `survivors`, and `bounds`.
- **`ledger.py` and `adder ledger` -- the solvency invariant.**
  `cost_with_adder = baseline - savings + overhead`, so the tool is cheaper than
  not having it exactly when savings cover overhead. The ledger records the
  guaranteed saving of every recommendation acted on against the routing
  overhead it cost, and a **calibration haircut** measures the part bounds
  cannot: if verified recommendations delivered 60% of what they promised, every
  future prediction is scaled by 0.6 before it meets its gate. Capped at 1.0 --
  a model that under-promises earns no credit for it.
- **`policy.schedule` and `adder policy --batch` -- one routing turn for several
  decisions.** Overhead is charged per turn, not per recommendation, so asking
  once about five steps costs one turn. Two consequences: steps whose saving is
  real but below a whole routing turn now clear it together, and the batch is a
  *more certain* bet than its members, because the horizon risk is shared while
  the idiosyncratic redo risk averages down.
- **`carry.optimal_split` -- how long to run a session, in closed form.** Average
  per-turn cost on a `k`-turn cycle is `m*r*F + m*r*g*(k+1)/2 + W/k`, so
  `k* = sqrt(2W/(m*r*g))`. The square root is the result: being wrong about the
  handoff cost by 4x moves the answer by 2x, which is why the number survives
  being derived from the one input nothing measures. Reported as a sweep over
  handoff size rather than as a single number.
- **`carry.delegate_threshold` -- the read size above which delegating pays,
  in closed form.** Both sides are affine in the read size, so the break-even is
  one division. It matters because a threshold is the only advice here that is
  free to apply: no routing turn to pay for, so it cannot cost more than not
  asking.
- **`adder carry` and `adder ledger`** join the command table.
- Four claims in `adder validate`: the measured carry multiplier sits above the
  0.10x assumption, the horizon mean exceeds its median, **every recommendation
  the router emits clears its own overhead** (a 240-case sweep), and the ledger
  is solvent.


- **`adder prefix` -- what a session restart actually costs, measured.** Both
  models of a restart in this repo were wrong, in opposite directions:
  `adder plan` charged nothing for one, and `carry.optimal_split` charged a full
  prefix rebuild. Every session records what its own opening turn was billed, so
  neither had to be assumed. Measured over the 46 openings that followed a turn
  inside the 5m TTL: an opening is 27,953 tokens of which **74% arrives as a
  cache read**, because the expensive part of the floor -- system prompt, tool
  schemas, `CLAUDE.md` -- is identical across sessions and still resident. A
  restart carrying a 2,000-token handoff costs $0.103 against the $0.300 a
  rebuild would cost: **2.9x cheaper**. Openings measure warm after gaps of days
  too, which no TTL explains; that observation is reported and deliberately not
  relied on.

- **A cross-provider model catalog, refreshed from public data.** adder
  previously knew nine hardcoded Claude models. It now carries ~500 models from
  every major lab -- Anthropic, OpenAI, Google, and the open-weight families --
  with price, context window, cache rates, tool support, licence, and arena
  rating, as *data* rather than code. `adder/catalog.py` layers it
  bundled snapshot < user cache < project override < first-party Claude table,
  merging field by field so a failed refresh cannot blank a price we already
  had and a project can pin one rate without forking the file.
- `adder models` -- browse the catalog, `adder models show <name>` for one model,
  `adder models refresh` to pull the sources, and `adder models ladder`, which
  re-derives each rung of the T0/T1/T2 ladder from the catalog and prints the
  drift against the constants in `classify.py`. The ladder reports; it never
  silently repoints dispatch.
- `adder pick "<task>"` -- rank every model that clears the gates by cost in *this*
  session's economics, not by sticker price, and `adder pick --combos`, which
  prices four multi-model plans (single, cascade, draft-review, panel) and
  states the assumption that decides each one.
- `adder/sources.py` -- the only module in the package that opens a socket.
  Opt-in, honours `ADDER_OFFLINE=1`, fails soft when a source is down, and
  replays saved captures via `adder models refresh --from lmarena=page.html` so the
  parsers are testable without a network.
- `ADDER_CATALOG=<path>` pins the whole catalog to one file, so a
  recommendation can be reproduced on another machine.
- **`adder pick` and `adder policy` now use measured escalation history.**
  `docs/models.md` claimed the outcome log overrode the Elo estimate; nothing
  called it. It does now, and it composes rather than replaces:
  `select.blend_p_fail(measured, elo_gap) = measured + (1 - measured) * elo_gap`.
  The log knows how often a *tier* escalates here; the arena knows how much
  weaker a *substitute* is than the model that tier names. Each row reports
  which basis it used.
- **Cross-vendor substitution in `adder policy`.** When a plan delegates -- and
  only then, because a subagent starts cold and has no model-scoped cache to
  rebuild -- the catalog is asked whether another vendor's model could run it
  for less. The candidate is priced as a cascade (`run + p_fail x redo on the
  Claude tier`), held to a per-tier Elo tolerance (120 points at T0, 40 at T2),
  and shown only if the saving clears the routing overhead. On this machine's
  numbers it usually does not, and the plan says so in one line rather than
  offering a four-cent recommendation. `--cross-vendor` shows the candidates
  anyway; `--no-cross-vendor` skips the lookup.
- **`adder plan`** -- prices a whole workload under a *regime*: a followable
  operating configuration, replayed turn by turn against the recorded
  transcripts, with both sides of every delegation on the books. It reproduces
  the measured bill to within 0.1% before quoting any multiple, and
  `--target N` searches a stated grid for the mildest configuration that
  reaches an N-fold reduction, or reports the floor when nothing does. On this
  machine: 10.5x at the default regime.
- **Delegability is measured rather than assumed.** Every earlier estimate here
  used "assume 25% of turns are delegable". `adder plan` triggers on how many
  tokens a step would admit to context, which the transcript records exactly, so
  the rule is one a hook can apply and the 23% that matches is a measurement.
- **The session model is now a lever.** `adder plan --session-model` prices the
  work as if the session had *started* on a cheaper model. This is not the
  mid-session switch the tool has always refused: a session that starts on
  Sonnet never built an Opus prefix, so no cache is invalidated. Measured on the
  same transcripts, switching a warm conversation is worth 0.5% of spend and
  starting cheap is worth 60% before any rework allowance. The capability cost
  is charged as an explicit `--session-rework` fraction (default 20%) and
  labelled MODELLED, because a transcript that only ran on one model cannot
  settle it.
- Two new claims in `adder validate`: `plan replay reproduces measured spend`
  (the baseline every multiple is a ratio against) and `starting cheap beats
  switching cheap` (so the two model-choice results can be checked against each
  other rather than confused for each other).
- `outcomes.evidence()` returns the escalation rate *with* its scope and sample
  mass, so a caller can tell "0.5 measured over 200 runs" from "0.5 because
  nothing was ever recorded". `p_fail()` is now a thin wrapper over it.
- **The unmeasured prior is now fitted where it can be, and stress-tested where
  it cannot.** `UNUSABLE_GIVEN_LOSS` converts a preference loss into a redo and
  scales every cascade cost and substitute verdict linearly; it was chosen by
  judgement. `select.calibrate_unusable_given_loss` fits it as
  `measured escalation / modelled preference loss` wherever the outcome log has
  enough history, refusing the fit for a tier with nothing to escalate to, an
  unrated model, or a gap inside the arena's error bars. `adder pick --combos
  --sensitivity` sweeps it across [0.15, 0.60] and reports whether the winning
  plan depends on it at all — on current data it does not, which is the useful
  answer.
- `adder pick --measured` corrects published cache-read rates by the realised
  miss multiplier that `adder carry` measures from your own transcripts (0.115x
  against an assumed 0.10x here, so the carry term was under-priced by ~15%).
  Applied as a ratio rather than a rate: how often a session misses is a
  property of the workload and transfers across vendors, what a read costs is
  the vendor's published number and does not.
- **`adder outcomes record`** -- a command-line write path for the outcome log.
  Recording a dispatch previously meant pasting a Python snippet out of a skill
  file, which is the entire adaptive half of the tool sitting behind a step
  nobody performs. The empty log on every machine was the evidence: `p_fail`
  never left its prior, so the router never learned that a cheaper tier works on
  a given project and never stopped sending the work up. One line per dispatch
  now does it, and 15 recorded runs is enough to move an abstained task off Opus.
- **`adder outcomes` says what it is waiting for.** Alongside the calibration
  table it now reports, per tier, how much more recent history it needs before
  the router may route below the classifier's tier, and what failure rate that
  tier has to beat at the current context size. "p_fail 0.50" is not an
  explanation of why everything goes to Opus; "needs 12 more runs, and has to
  come in under 26%" is.
- **`adder policy --record`** books a recommendation in the ledger as it is
  emitted -- the one moment when both the prediction and the overhead are
  actually known. `decide` already read the ledger to haircut its own
  predictions, but nothing wrote one, so that correction was a branch that could
  never be taken: the tool consulted its own record of whether it was worth
  using and always found the page blank. Opt-in, because a command people run in
  a loop should not start writing because it was run.
- **The read guard decides on a cost, not a token count.** `pretooluse_read_guard`
  is the only component in the tool that can *prevent* spend rather than report
  it, and it was gating on a hardcoded 15,000 tokens before it looked at the
  session at all -- three times looser than the threshold `adder plan` derives
  from the same transcripts. A 6,000-token read with 400 turns left costs $1.24
  to carry and $0.13 delegated, and the guard said nothing. The token floor is
  now only an I/O guard (2,000, to avoid parsing a transcript on every trivial
  read); the decision is the dollar comparison it was already computing, tunable
  as `ADDER_GUARD_MIN_COST`.
- 28 tests for `.claude/hooks/`, which had none. `.claude/` is tracked and
  shipped, so it is testable and now tested -- including that the guard advises
  rather than blocks by default, that blocking asks rather than denies, and that
  a broken transcript lookup can never break the turn.
- **Two claims in `adder validate` that pin the two things this work was for.**
  Neither headline behaviour was checkable before: a constant could move and
  every unit test would still pass.
  - `a prior never buys a downgrade` sweeps the router and fails if it ever
    routes below the classifier's tier without the outcome log having been
    informative there. This is the silent failure the right-sizing change had
    to avoid: under a no-evidence prior the cheapest rung genuinely does have
    the lowest expected cost, so a router that minimises expected cost without
    a permission gate sends real work to the cheapest model it can hold and
    reports a saving for it. Nothing in the output would look wrong.
  - `a regime exists that reaches 10x` replays the frontier -- every lever at
    the end of its range -- and reports the multiple. Checked against the bound
    rather than by searching, so it costs one replay instead of four hundred,
    and it is expected to fail on a workload with no long sessions, because
    there the honest answer is that a 10x is not available.
- `adder models refresh --if-stale [--max-age DAYS]` checks the local catalog's age
  and returns before opening a socket if it is current, so the refresh can be
  put on a timer without hammering two public endpoints.
- 90 tests across `tests/test_catalog.py`, `test_sources.py`, `test_select.py`,
  and `test_models.py`.

- `adder/cli.py`: a single dispatcher for every subcommand, with grouped help,
  `--version`, and a did-you-mean suggestion on an unknown command. Modules
  still own their own `argparse` parsers, so `adder <cmd> --help` stays accurate.
- `python -m adder` as an equivalent of the `adder` console script.
- `adder` and `adder` console entry points, so a `pip install` puts the tool
  on `PATH` without the `scripts/` launcher.
- `CLAUDE.md`: the binding working agreement for agents and humans editing this
  repo — invariants, layout, style, testing rules, and what an agent must not do.
- `CONTRIBUTING.md`, `SECURITY.md` (with an explicit threat model for an offline
  tool that reads transcripts), `CODE_OF_CONDUCT.md`, and `LICENSE` (MIT).
- GitHub Actions CI: `ruff` plus a test matrix on Python 3.10–3.14, a macOS and
  Windows spot check, a CLI smoke test over all 16 subcommands, and a build job
  that asserts the wheel carries its package data.
- **An offline-guarantee check in CI.** It parses every module under `adder/`
  and fails the build if anything except `sources.py` imports a networking
  module. The no-network property is now enforced, not just documented.
- Release workflow: a tag push verifies that the tag, `adder.__version__`, and
  the CHANGELOG agree before anything is built, then publishes a GitHub Release
  with notes extracted from this file and uploads to PyPI via trusted publishing.
- Issue templates (bug, wrong number, feature), a PR template whose checklist is
  the CONTRIBUTING checklist, `CODEOWNERS`, and Dependabot for dev tooling and
  Actions.
- `.pre-commit-config.yaml`, including two project-specific hooks: one refuses to
  commit `.jsonl` files (real transcripts contain source code and prompts), and
  one fails if `[project.dependencies]` stops being empty.
- `Makefile` with `help`, `test`, `cov`, `lint`, `fmt`, `check`, `smoke`,
  `build`, `verify-dist`, `clean`, `hooks`, and `release-check`.
- `adder/py.typed` (PEP 561) and `MANIFEST.in`.
- `.gitattributes` for line-ending normalisation and export rules.

### Added — the adaptive half, finally running

- **`adder outcomes import` -- backfill the dispatch log from transcripts.**
  The escalation gate needs a measured `p_fail`. The only way to write that log
  was `adder outcomes record` after every delegation, by hand. Nobody does that,
  so the log is empty on every machine, `p_fail` sits on its 0.5 prior forever,
  and the router is permanently forbidden from preferring a cheaper tier. A
  feature that requires a discipline nobody keeps is a feature that does not
  exist. The evidence was on disk the whole time: a delegation is an `Agent`
  tool_use block naming a `subagent_type`, and its outcome is the `tool_result`
  that answers it. Dry-run by default, idempotent by `task_hash`, so re-running
  adds only what is new. On the author's transcripts it recovers **15
  dispatches, 7 usable, 1 escalation** where the log held nothing.
- **Tiers match on price, not just on model id.** A run on `claude-opus-4-8` is
  $5/$25, the same arithmetic and the same decision as `claude-opus-5`, so it
  belongs on the same rung. Matching ids alone left every run on a previous
  generation untiered and therefore uncounted. Base rates are compared, never
  dated ones: an introductory price makes a model temporarily cheaper without
  moving it to a different rung, and a tier map that reshuffles itself on an
  expiry date is worse than no tier map.
- **What the import cannot see is stated where it is used.** An escalation is an
  error result or an `ESCALATE:` reply. A subagent that returned a confident
  wrong answer and was believed is invisible -- and is equally invisible to a
  human filing a report afterwards, so importing is not worse than
  hand-recording. It does mean the rate is a **lower bound**, which matters
  because under-estimating `p_fail` is the expensive direction. Rows carry a
  `source` field for provenance; both sources are weighed the same, because both
  are blind to the same thing.
- **A quality check in `adder doctor`.** Every lever in this repo trades tokens
  for something, and a degraded agent often looks *cheaper* per turn while
  taking more turns to finish -- so a health check that only looked at money was
  recommending exactly the changes it could not evaluate. Only the tool error
  rate is gated, because it is the one proxy with a defensible absolute
  threshold: a failed call still costs a full turn and leaves its error in the
  context. The rest are reported without a verdict, because they only mean
  something compared against themselves before and after a change.
- **The read guard now honours the `Grep` matcher its own install snippet has
  always advertised.** The hook returned early for anything that was not `Read`
  or `Bash`, so anybody who followed the documented `"matcher": "Read|Bash|Grep"`
  got a guard that looked installed and did nothing for a third of it. A
  content-mode `Grep` with no `head_limit` returns every matching line in the
  repository; `files_with_matches` and `count` are bounded by construction and
  are still waved through. A test now asserts the install snippet and the
  guarded set cannot drift apart again.

### Added — reporting surface

- **`adder tools` -- attribution by the thing a person can actually change.**
  Every other view is organised by model, session, or turn; none of those name
  a decision. A tool call is one: an unbounded `Bash`, a `Read` of a lockfile, a
  `Grep` with no `head`. The report prices what each tool *leaves behind* rather
  than the turn that called it, because a 40K-token result on turn 20 of a
  400-turn session is 40K tokens re-read ~380 times. Attribution apportions the
  **measured** accumulated cache-read pool by share of context growth, so the
  column can never sum above what was billed. On the author's transcripts:
  **`Bash` is 23% of all context growth and carries $1,073 of $6,212** — the
  single largest addressable finding on that workload.

  (An earlier draft of this entry quoted 73% and $3,168. That denominator left
  out assistant output, which is two thirds of growth; the corrected figures are
  above, and `tests/test_tools.py` now pins the decomposition against
  `adder context` so the two cannot drift.)
- **`adder sessions` -- one row per session, sortable.** `trace` reports that
  the top quarter of sessions hold three quarters of the spend and then prints
  three of them. `--sort per-turn` separates a session that was expensive
  because it was long from one that was expensive per turn; they have different
  fixes. Also reports compactions and cache rebuilds per session.
- **`adder agents` -- delegation as measured, not as recommended.** Joins what
  `policy` advises against what the transcripts did. Three findings per run:
  the share of spend that is subagent work (0.5% here), subagent runs whose
  peak context would have fitted a cheaper model (a subagent starts cold, so
  the cache-invalidation argument that protects a live session does not apply),
  and main-chain turns that admitted more than 20K tokens at once, each priced
  against delegating it with the horizon estimator **at that turn's own index**.
- **`adder anomaly` -- the expensive turns, each with the mechanism that
  explains it.** Uses the median and MAD rather than mean and sigma: one $40
  turn inflates sigma enough to score itself as ordinary, which is the detector
  failing on the case it exists for. Each finding is labelled `prefix rebuild`,
  `context jump`, `long output`, `fast mode`, or `big context`, in that
  precedence order. On this workload: **89 unusual turns, $365 above the median
  turn, 80 of them prefix rebuilds.**
- **`adder effort` -- the re-fit `cost.py` has claimed for months.** The
  docstring on `EFFORT_OUTPUT_MULT` said "`adder.effort` re-fits them from a
  transcript"; no such module existed. It does now. Transcripts carry an
  `effort` field per record, so output volume per level is measurable. It
  **refuses** to fit a level with under 50 turns or when only one level appears
  (which is the case on the author's machine), and says so rather than
  producing a multiplier from nine turns.
- **`adder budget` -- burn-down against a spend target.** Two projections are
  reported, not one: the mean over elapsed days, and the median of *active*
  days in the last 14, which is robust to one expensive day and to weekends.
  The higher of the two is what the verdict uses, because under-projecting a
  budget is the expensive error. `--strict` exits non-zero for a hook or CI.
- **`adder doctor` -- one command that says what to do next, ranked by
  dollars.** Runs every check, delegating each measurement to the module that
  owns it, and orders findings by money at stake rather than by severity: a 12%
  cache hit rate on a $30 workload deserves less attention than a delegation
  gap on a $6,000 one. `--strict` fails only on findings above a materiality
  floor, so the exit code means something stable.
- **`adder export` -- the priced turns, at turn, session, or day grain, as CSV,
  JSONL, or JSON.** Field names are identical across formats. **No message
  content is ever exported** -- transcripts hold source code and prompts, and a
  cost export needs none of it; `tests/test_export.py` asserts a known secret
  string cannot appear in the output.
- **`adder config` -- every setting, its value, and the layer that set it.**
  Eight modules each read their own environment variable and documented it in a
  comment. `ADDER_LOG`, `ADDER_LEDGER`, `ADDER_HOME`, `ADDER_CATALOG`,
  `ADDER_TRACE_CACHE`, `ADDER_OFFLINE`, `ADDER_GUARD_BLOCK`, `ADDER_WARN_SPEND`
  -- all load-bearing, none discoverable without grepping. Precedence is
  `default < ~/.claude/adder.json < ./.adder.json < ADDER_*`, and the report
  prints the source of each value, which is the half that matters when two
  machines disagree. Nothing here writes a config file.
- **Shared window flags on every report that reads transcripts**: `--since`,
  `--until`, `--project`, `--model-filter`, `--session`, `--min-turns`,
  `--only-subagents`/`--no-subagents`. Dates accept `2026-08-01`, `7d`, `2w`,
  `today`, `yesterday`. The window is half-open (`--since` inclusive, `--until`
  exclusive) so two adjacent windows partition the data exactly -- previously
  the two commands that had date filters disagreed on the boundary, one
  comparing dates and the other datetimes.
- **`--json` on every report that lacked it**: `live`, `debt`, `context`,
  `cache`, `quality`, `horizon`, `carry`, `prefix`, `savings`, `verify`,
  `validate`, `regret`, `simulate`, `plan`. `tests/test_json_surface.py`
  discovers them from the source and asserts each emits exactly one parseable
  document with no bare `NaN`/`Infinity` -- Python's own loader accepts those,
  so a round-trip test passes while every other parser rejects the file.
- **`trace --by model|project|session|day|tool|speed|ttl`, `--top`, and
  `--strict`.** The `tool` grouping attributes a turn to each tool it called
  and says so, rather than inventing a split the transcript does not support.
  `--by day` sorts chronologically and draws a bar per day: every other
  grouping is a ranking, but a date axis sorted by cost is a bar chart with the
  x-axis shuffled.
- **`adder completion bash|zsh|fish`**, generated from the command table and
  from each module's own argparse parser at print time. A hand-written
  completion file is a second copy of the command list, and a second copy is
  the one that goes stale; `tests/test_completion.py` asserts the generated
  script covers every live command and every live flag.
- **An API-error proxy in `adder quality`.** `<synthetic>` records are counted
  as client-side failures rather than turns. Deliberately excluded from the
  before/after regression check: most are network flakiness, and failing a cost
  change because someone was on hotel wifi is a false positive that teaches
  people to ignore the check.
- **`effort` is captured per turn** and exported. It is a top-level field on the
  record, not inside `message`, and it is the only thing in a transcript that
  says how hard the model was told to think.
- **`stats.py`, `render.py`, `filters.py`** -- one definition each of a
  quantile, a formatted dollar, and a date window, replacing three, fifteen, and
  two respectively.

### Fixed — claims about the tool itself

- **The README said adder "never writes anything under `~/.claude`". It does.**
  The parse cache lives at `~/.claude/.adder-trace-cache`, the outcome log and
  the ledger beside it, and the catalog snapshot under `~/.claude/adder/`. The
  guarantee that is actually true and actually load-bearing is narrower:
  **nothing under `~/.claude/projects` is ever written, renamed, or deleted.**
  The README now says that instead, and lists every file adder does create with
  the command that creates it. A measurement tool that misstates its own side
  effects has no standing to correct anybody else's numbers.
- **That guarantee is now a test, not a promise.** Every command is run over a
  fixture transcript tree and the tree is compared byte-for-byte before and
  after -- sizes, mtimes, and hashes. A static check cannot prove read-only;
  this can.
- **Every command is now executed under test, not merely asked for `--help`.**
  CI smoke-tested `--help` for each command, which proves the parser builds and
  nothing else. It would not have caught a report that raises on the first real
  record, or a JSON branch referencing a field its dataclass does not have --
  both of which happened while this change was being written.

### Fixed — filters that were accepted and ignored

- **`--session` and `--project` were silently dropped by every raw-record
  scanner.** `tools`, `context`, `quality`, and the new dispatch scan read
  message *content* rather than billed usage, so they never build a `Turn` and
  could not use the turn-level predicate. Each was applying only the date part
  of the window, which meant `adder tools --session abc` reported the whole
  corpus. A filter that is accepted and ignored is worse than one that is
  rejected, because the number looks like an answer. `Window.keeps_record`
  now applies date, project, session, and sidechain to raw records.
- **`adder context --since` printed filtered billing beside unfiltered
  attribution** — two numbers drawn from different populations, laid out as
  though one described the other. Introduced and fixed inside this release.
- **`--model-filter` is reported as inapplicable rather than ignored.** A
  `tool_result` block carries no model, so a tool report genuinely cannot honour
  it; it now says so on screen instead of quietly widening.

### Fixed — measurement

- **Cross-file deduplication.** `iter_file` removed the one-record-per-content-
  block repetition inside a transcript; nothing removed it *across* transcripts.
  A resumed session writes a new `.jsonl` replaying earlier turns, and a
  sidechain file restates the parent turn it branched from -- both carry the
  original `message.id`, and both were counted twice, by the same mechanism and
  in the same direction as the per-block bug that cost 1.78x. `load_sessions`
  now dedups by `(session id, message id)`; ids are only unique per
  conversation, so the session id is part of the key.
- **Records with no message id sorted to the wrong place.** The merge compared
  an index into the id list against a global record counter -- two different
  counter spaces -- so an id-less record landed next to an unrelated turn. The
  same expression was `order.index()` inside a loop, i.e. quadratic; on a
  1,854-turn transcript that is the parse.
- **`<synthetic>` records are no longer counted as either turns or unpriced
  spend.** Claude Code writes an assistant record with that model id when the
  *client* produced the message: "API Error: Connection closed mid-response", an
  interrupted stream. Their usage block is all zeros. Counted as turns they
  depress every per-turn average; counted as an unknown model they raise a
  "this report is a lower bound" warning about spend that does not exist. They
  are now reported as what they are -- a count of client-side failures.
- **A model missing from `prices.py` is reported instead of dropped in
  silence.** Any turn whose model has no price was skipped, so the total was a
  lower bound that did not say so. `trace` now prints the tally and `--strict`
  exits non-zero.
- **The p90 was the maximum.** `trace` computed percentiles by indexing a
  sorted list at `int(len * p)` -- the nearest-rank estimator with no
  interpolation, which on ten sessions returns the tenth value as the "p90".
  Every quantile now goes through `stats.quantile`, the linear-interpolation
  estimator, so `median()` and `quantile(0.5)` agree by construction.
- **`robust_z` scored everything zero when MAD was zero.** MAD is zero whenever
  more than half a sample is identical, and the fallback of "no dispersion, so
  no outliers" makes the detector miss the one obviously extreme value. The
  median now serves as the scale in that case; constant data still scores zero
  throughout, so nothing is invented.
- **`adder anomaly` took 105 seconds.** `robust_z(x, xs)` re-sorts `xs` on every
  call, so scoring a series with it is O(n² log n) -- 24,000 turns took a minute
  and a half. `stats.robust_z_series` computes the median and scale once:
  **105s → 0.2s**, and `adder doctor`, which runs it, **105s → 1.9s**.
- **`budget.exhausted_on` raised `OverflowError`** on a small rate against a
  large budget, because the projected day count overflowed `date`. The answer
  there is "not in this period", not a traceback.
- **A subagent's context was being compared against the main chain's.** Sidechain
  turns run in their own window, so carrying one forward as the previous turn
  makes the next main-chain turn look like a large admission when nothing was
  admitted. `agents.missed` and `anomaly.scan` now track each chain separately.
- **The growth denominator in `adder tools` left out assistant output.** It is
  the largest of the three sources — around two thirds — so every tool's share
  and every dollar apportioned by that share was inflated roughly threefold, and
  `adder doctor` ranked a $1,073 finding as a $3,168 one. The denominator is now
  tool results + user text + assistant output, using the **billed** output count
  rather than the character estimate wherever the session map is available. Both
  columns of the report use the same denominator, or they would describe
  different populations with no way for a reader to tell.
- **Tool attribution was orphaning results from the tool that asked for them.**
  Deduplicating assistant records by *message* id discards every `tool_use`
  block after the first, because Claude Code writes one record per content
  block. The results then reference ids the scan never saw. Deduplication is by
  block id; before the fix, 56% of context growth was attributed to a tool
  called `?`.

### Fixed

- **Placement was priced as though delegation could not fail.** A delegated read
  that comes back missing what the session needed costs the subagent run, the
  turn that noticed, *and* the inline read anyway -- strictly worse than never
  having delegated. `escalation_is_profitable` has carried the equivalent term
  for tiers since the beginning; placement, the larger lever, had none.
  `placement_cost` now takes `p_redo` and `redo_overhead`, and the router feeds
  it the tier's own measured escalation rate. It changes real verdicts: an 8K
  read at a 900K context with three turns left used to come back as a
  `delegate` whose saving sat below its own overhead.
- **The downgrade branch never checked its own overhead.** Found by the new
  `emitted advice clears its own overhead` sweep on the day it was written:
  three cases out of 240 emitted a $0.011 downgrade from a turn that cost
  $0.015 to spend. It is now gated like everything else, against a band on the
  output estimate the whole decision turns on.
- **`routing_overhead` assumed a warm cache.** It is the bar every
  recommendation clears, so understating it made adder emit advice too
  eagerly. It now takes the measured carry multiplier, which raises the bar by
  15% on this machine.
- **A plan that declined to delegate still argued for delegating.** The reason
  list on an `inline` plan carried the delegation case it had just rejected.


- **An aggregator's alias routes were treated as a different vendor.**
  `~anthropic/claude-haiku-latest` is an Anthropic model on a floating alias.
  The tilde survived into the organisation field, so the Claude Code harness
  gate refused it inline placement and told the user that "~anthropic cannot be
  the main session". Organisations are normalised on the way in.
- **A hand-edited catalog file could crash a cost report.** Pinning one price in
  a project override is an advertised feature, so the types in that file are
  whatever a human typed. A price written `"5"` instead of `5` loaded fine and
  then raised a bare `TypeError` from inside the cost model. `Entry.from_json`
  now coerces what is recoverable and treats the rest as unknown — never as
  free.
- **Arena ratings were compared without their published error bars.** The
  source ships a 95% interval per rating and the catalog discarded it, so a
  17-point gap between two overlapping intervals became a confident 52%
  preference loss. Intervals are now stored, comparisons take the candidate at
  the bottom of its interval against the reference at the top of its own, and a
  ranking states when the arena cannot separate two models. Worth about three
  points of `p_loss` on current data.
- **The panel combination reported a fabricated quality number.** Its formula
  assumed N runs of one model fail independently, with a bare `0.15` constant
  tuning the result. Best-of-N quality is now reported as unknown; the cost,
  which is exact, is still priced.
- The cross-vendor substitute block never warned about a stale catalog, though
  `adder pick` did. A year-old snapshot could drive a recommendation silently.
- **The outcome log silently stopped influencing routing when a row carried an
  ISO timestamp.** `Outcome.ts` is epoch seconds, but every other timestamp in
  the repo is an ISO string, so writing one here is a matter of time. The row
  loaded cleanly and then raised inside the recency weighting — where both
  callers swallow the exception, so the symptom was not an error but a
  calibration path quietly going dead. `outcomes.load()` now coerces a
  recoverable timestamp and drops an unrecoverable row.
- A cross-vendor substitute was compared against the Claude tier on the *full*
  delegated cost, which includes admitting and carrying the returned summary at
  the session model's rate. That term is identical for every candidate and
  large enough on a long session to swamp the difference the choice is about,
  and the escalation multiplier was being applied to it. The comparison is now
  on the subagent leg alone (`Costed.subagent`).

### Changed

- **One admitted-cost expression instead of two.** `select.py` had grown its own
  copy of "a cache write now, then a cache read on every remaining turn",
  because it prices ~500 catalog models whose cache rates are published
  absolutely rather than as multiples of the input rate. The duplicate is how
  the two came to disagree about whether the carry term could be corrected:
  `cost.py` grew a fitted-carry hook and `select.py` could not accept one. Both
  now call `cost.admitted_cost(n, Rates, reads=...)`, with the rates filled in
  from the Claude table on one side and the catalog on the other, and a test
  asserts the two entry points agree to within floating point on the same
  inputs. `adder pick --measured` consequently uses both halves of the fitted
  model — the realised miss multiplier *and* the expected re-read count after
  compaction survival.

- **`adder plan` charges for restarts, and solves for the cadence instead of
  taking one.** A split used to reset the context and cost nothing, which is a
  lever with no price term -- and the grid duly pushed it to the end of its
  range for free. Each restart is now charged what that session's own opening
  was billed, plus the handoff written in at the cache-write rate. With the
  price measured rather than assumed, `k* = sqrt(2W/(m*r*g))` can be evaluated
  against this workload: **19 turns**, not the round 300 that was there before,
  and against the 536-turn sessions the spend actually sits in that is 6.1x
  cheaper per turn on the input side. `--split-turns` still pins it; `--handoff`
  sweeps the one input nobody can measure.
- **The replay scales cache writes by admissions, not by context.** It used to
  scale a turn's whole input bill by how much context the regime left it
  carrying. Reads work that way; writes do not -- measured here, cache writes
  run at 2.1x admitted tokens, and splitting a session changes what is re-read
  without changing what is written. The old scaling made every split-heavy
  regime look cheaper than it is: correcting it moved the default ladder from
  11.7x to **9.1x**, which is the honest number.
- **The delegation threshold is solved too, and it was 6x too conservative.**
  `--delegate-above` defaulted to a round 5,000 tokens. `carry.delegate_threshold`
  gets the break-even in one division, and evaluated at the solved cadence it
  answers ~285. The direction is not the intuitive one: a 19-turn cycle leaves
  only ~9 re-reads to avoid, which raises the threshold, but it stays small
  because admitting a token to an Opus context costs 2.00x its input rate as a
  cache write while reading it once on Haiku costs 1.00x of a rate five times
  lower. Delegation is not only a carry play.
- **A delegated step's own output is charged.** It was charged at zero, which
  made delegation a free way to delete the session's generation cost -- and with
  the threshold solved down to 300 tokens the optimiser found it: 99% of admitted
  tokens delegated and main-session output at 1% of the bill. It is a rate
  substitution, not a deletion, and is now priced on the subagent's model. This
  cost 1.2x of the headline multiple, which is the point of finding it.
- **`carry.optimal_split` takes a measured restart cost.** `restart_cost=` uses
  it; without one the pessimistic cold rebuild is still assumed, because with no
  data the safe direction is to recommend restarting less, not more.
- Two claims added to `adder validate`: that a session opening is mostly a cache
  read (>=40%, measures 74%), and that the solved cadence is shorter than the
  sessions this workload actually runs (19 vs 536). The restart cadence is now
  the largest single lever `adder plan` recommends, and it rests on both.

- **Gate 3 of `policy.decide` is a ladder search, not a one-way escalation.**
  It used to ask one question -- "is the classifier's tier better than Opus?" --
  which could only ever escalate, so an abstention meant Opus permanently no
  matter how much history said otherwise. The tier is now the rung with the
  lowest expected cost, `run + p_fail x (finishing on T2 + the turn that catches
  the failure)`, with asymmetric permissions: moving up needs no evidence, and
  moving below the classifier's tier needs an abstention, an informative outcome
  log, and a measured failure rate under that rung's break-even. Cheapness alone
  never buys a downgrade. The whole ladder, losers included, is printed and in
  `--json`.
- Failure rates are clamped monotone up the ladder. Without it a rung nobody had
  logged sat at the prior while the rung below it sat at a measured rate, and the
  table reported T3 (Opus at xhigh) as seven times likelier to fail than T2 (the
  same model at high) purely because of which label the log carried.
- The escalation prior is read off the classifier's confidence rather than being
  a flat 0.5. Beta(1,1) is right for "no information", but the classifier *is*
  information; a flat prior on an empty log -- which is every user's first
  session -- made a bounded lookup look like a coin flip and sent it to Opus.
- **Renamed the project from `llm-router` to `adder`.** The old name advertised
  the least interesting output — routing is emitted last, and `policy.decide`
  declines to emit it when the modelled saving does not clear the cost of the
  routing turn — and implied a request path the tool has never had. `adder` names
  the operation instead: a full adder's second output is the carry, and the carry
  is what this measures. Reasoning in [docs/naming.md](docs/naming.md). No
  reported figure moves; this is a rename, not a measurement change.

  Breaking, and there is no compatibility shim, because nothing was ever
  published under the old names:

  | was | is |
  |---|---|
  | `rt <cmd>` | `adder <cmd>` |
  | `llm-router` console script | removed |
  | `pip install llm-router` | `pip install adder-cli` |
  | `import router` / `python -m router` | `import adder` / `python -m adder` |
  | `LLM_ROUTER_OFFLINE`, `LLM_ROUTER_CATALOG`, `LLM_ROUTER_HOME` | `ADDER_OFFLINE`, `ADDER_CATALOG`, `ADDER_HOME` |
  | `ROUTER_LOG`, `ROUTER_STATE`, `ROUTER_TRACE_CACHE`, `ROUTER_GUARD_BLOCK`, `ROUTER_WARN_*` | `ADDER_LOG`, `ADDER_STATE`, `ADDER_TRACE_CACHE`, `ADDER_GUARD_BLOCK`, `ADDER_WARN_*` |
  | `RT_PYTHON` | `ADDER_PYTHON` |
  | `~/.claude/llm-router/`, `./.llm-router/` | `~/.claude/adder/`, `./.adder/` |
  | `~/.claude/router-outcomes.jsonl` | `~/.claude/adder-outcomes.jsonl` |
  | `/route`, `/route-init`, `/route-doctor` | `/adder`, `/adder-init`, `/adder-doctor` |

  The outcome log is the only one of these holding data worth keeping:
  `mv ~/.claude/router-outcomes.jsonl ~/.claude/adder-outcomes.jsonl`. The trace
  cache and the advisor state rebuild themselves. The `route-t0` / `route-t1` /
  `route-t2` subagents keep their names — they are named for what they do, and
  they are referenced from users' project configs.

- `pyproject.toml`: real project metadata, classifiers, URLs, `dev` extras, a
  version read dynamically from `adder.__version__`, coverage config, ruff
  config, and `--strict-markers --strict-config` with warnings as errors in
  pytest. `package-data` now ships `adder/data/*.json`, which the wheel
  previously dropped at install time.
- `scripts/adder` no longer dispatches; it resolves an interpreter, checks for
  Python 3.10+ with a readable error, and hands off to `adder.cli`.
- `.gitignore` expanded from 5 lines to cover build artifacts, coverage and type
  caches, editor and OS junk, secrets, and tool scratch output — while keeping
  `.claude/` tracked, since the agents, hooks, and skills are part of the
  product.
- Piping a report into `head` now exits cleanly instead of printing a
  `BrokenPipeError` traceback.

### Fixed

- **The read guard read the wrong environment variables.** The rename to `adder`
  documented `ROUTER_GUARD_*` -> `ADDER_GUARD_*` in this file and then did not
  finish the job: the hook went on reading `ROUTER_GUARD_WARN` and
  `ROUTER_GUARD_HARD`, so anyone who followed the changelog configured a guard
  that silently ignored them. That is the worst failure available to a guard --
  it still looks installed. Both prefixes are now honoured, `ADDER_` winning.
- **`escalation_is_profitable` charged the cheap run twice.** The failure branch
  was priced `cheap + p_fail x (cheap + expensive)`, billing the cheap attempt
  again in a branch where it is not re-run. There is no reading under which that
  is true -- escalating is the opposite of running the cheap model a second time
  -- and it made every cheap tier look worse than it is. Now
  `cheap + p_fail x (expensive + retry_overhead)`. The gate's tolerance for
  failure at 9,200 tokens of task context rises from 67% to 80% before the new
  overhead term, and to 19% after it.
- **The turn that catches a failure was free.** A subagent that returns something
  wrong does not announce it: a main-session turn has to read the result, judge
  it, and dispatch again, and that turn re-reads the whole context. At 400K
  tokens on Opus that is $0.21 the gate was not charging. `retry_overhead` is
  now passed by `policy.decide` and exposed on `max_tolerable_p_fail`.
- 17 lint findings across `adder/`, `tests/`, and `.claude/hooks/`: ambiguous
  `l` identifiers, unused unpacked values, redundant `int(round(...))` and list
  comprehensions, collapsible nested conditionals, and `zip()` over successive
  pairs replaced with `itertools.pairwise`. No behaviour changed; all 278 tests
  pass unchanged.

### Housekeeping

- Removed committed cache directories and OS junk from the working tree.

## [0.1.0] - 2026-08-14

The initial public cut. Grouped by area; every item is covered by tests unless
noted.

### The correction that reframes everything (1–7)

1. **Deduplicate assistant records by `message.id`.** Claude Code writes one
   JSONL record per content block, each repeating the whole message's `usage`.
   Summing lines multi-counted most turns: 32,251 reported vs 18,163 actual,
   $7,507 vs $4,456.
2. **Keep the highest-`output_tokens` record per message**, not the first —
   partial streamed records carry a running count; keeping the first undercounts
   output by 2.6%.
3. Merge tool names across a message's block records.
4. Preserve original turn order after grouping.
5. Records with no `message.id` are never collapsed together.
6. **Corrected the headline finding.** "Assistant output is ~105% of context
   growth" was an artifact of (1): duplicates inflated output ~1.78x while
   leaving context deltas untouched. Measured share is ~50%.
7. Updated `validate.py`'s claim and expected range to match, with the reason
   recorded in-place rather than the measurement explained away.

### Feasibility: savings that were impossible (8–17)

8. Per-model context limits in the price table.
9. `fits()` / `context_limit()` predicates.
10. `switch_is_profitable` refuses a target model the context cannot fit — Haiku
    holds 200K, the measured median peak context is 544K.
11. `check_context=False` escape hatch so pure break-even math stays testable.
12. `placement_cost` refuses a subagent whose window cannot hold the read.
13. `escalation_is_profitable` gates on the cheap tier's window.
14. `policy.decide` escalates a tier for *feasibility* and says so in `warnings`.
15. `cheapest_that_fits()` with a capability floor.
16. Longest-prefix model resolution — `claude-sonnet-4-6-*` no longer resolves
    as `claude-sonnet-5`.
17. Resolve Claude Code's `[1m]` context-variant suffix.

### Cache mechanics (18–34)

18. Per-model **cache minimums** (512 / 1024 / 2048 / 4096) — non-monotonic
    across generations; a prefix below the minimum silently does not cache.
19. `caches()` / `cache_min()` predicates.
20. Per-turn **cache TTL detection** from `usage.cache_creation`.
21. TTL-aware `turn_cost` and `input_cost` — 1h writes bill at 2.00x, not 1.25x.
22. `TTL_SECONDS`, `CACHE_LOOKBACK_BLOCKS`, `BATCH_MULT` constants.
23. `cache_write_cost` / `cache_read_cost` / `cache_miss_cost`.
24. `choose_ttl()` — picks 5m vs 1h from measured idle gaps.
25. `fanout_cost()` — N parallel calls over a shared prefix all miss the cache;
    staggering the first turns N writes into 1 write + (N−1) reads.
26. **New `adder/cache.py`**: cache efficiency and rebuild waste.
27. Per-rebuild **cause attribution**: model switch, idle expiry, post-compaction,
    growth.
28. Recoverable vs unrecoverable classification — an idle gap beyond 1h is not
    fixable by any TTL, and is reported as a session-boundary problem instead.
29. Mix-aware TTL recommendation — a workload already on 1h is not told to
    "switch to 1h".
30. Cache hit-rate metric (99.1% measured here).
31. Fast-mode detection and pricing (Opus 5 fast bills at $10/$50, double).
32. `UnsupportedSpeed` for models without fast mode.
33. Batch API 50% multiplier and `batch_saving()`.
34. `marginal_turn_cost()` — what one more turn costs right now.

### Where context actually comes from (35–45)

35. **New `adder/context.py`**: attribute growth to its sources.
36. Model-authored volume taken from **billed** output tokens, never estimated
    from text.
37. Handles two traps that made estimation wrong: Opus 5 returns thinking text
    *empty* while billing it, and `tool_use` JSON arguments are output tokens in
    no `text` block. Text-based estimation undercounts ~60x.
38. Per-tool attribution — `Bash` alone is ~4.1M tokens, more than every other
    tool combined.
39. `output_share_of_growth()` — the measured ceiling on any verbosity claim.
40. `measured_growth()` from billed token deltas.
41. Read-vs-written verdict that inverts the advice when reads dominate.
42. **`verbosity_saving` now scales by the measured output share** — assuming
    1.0 over-claimed terseness roughly twofold.
43. Debt module docstrings corrected (105%→50%, 607→340 turns, 13x→7.8x).
44. `decompose_read_cost` docstring corrected.
45. True-cost-of-output line scaled by the measured share.

### New levers (46–55)

46. **`tool_output_discipline`** — the read half of the pool, which no
    writing-style instruction can reach.
47. **`effort_reduction`** — the only output-side lever that does *not*
    invalidate the prompt cache, so unlike a downgrade it is free to apply
    mid-session. Ranks third.
48. **`cache_discipline`** — MEASURED recoverable rebuild waste.
49. `effort_saving()` in the cost model, with downstream re-reads included.
50. `EFFORT_OUTPUT_MULT`, documented as priors rather than measurements.
51. Terseness `pool_fraction` scaled by measured share.
52. Pool documented as two halves so lever ranking follows the data.
53. Removed the `callable(getattr(...))` hack via `Session.cost_on(date)`.
54. Savings report ends by pointing at `adder quality`.
55. Every modelled lever states its assumption inline.

### Maintaining agent performance (56–66)

56. **New `adder/quality.py`**: performance proxies from transcripts.
57. Tool error rate (`is_error` tool results).
58. Correction rate (operator redirect phrasing).
59. Interrupt rate.
60. Turns per human prompt.
61. Rework ratio (edits per distinct file).
62. Tool replies and injected meta-messages excluded from "human prompts".
63. `regressions()` with a noise tolerance.
64. Before/after windowing by date.
65. **`adder verify` refuses to claim a clean saving when a proxy regressed** — a
    cheaper agent that needs more turns is not cheaper.
66. `adder quality --since DATE` as a standalone guard.

### Routing policy (67–77)

67. **`p_fail` wired from the measured outcome log** — it was computed but never
    used by `decide()`.
68. `escalation_is_profitable` integrated into the decision path.
69. Per-project `p_fail` scoping.
70. `choose_effort()` per tier.
71. Falls back to the model default where `effort` is rejected (Haiku 4.5).
72. Cache-safe inline effort downgrade when a switch is not worth it.
73. `Plan.warnings` for feasibility escalations.
74. `Plan.p_fail` surfaced.
75. `--project` and richer `--json` output.
76. Four gates documented in order of veto.
77. Restored the horizon survivor-function estimator after an accidental revert.

### Session analysis (78–86)

78. **Fixed a real bug**: `current_session` fell back to parsing *every*
    transcript in the directory and reporting the union as "this session".
79. Skips empty transcripts instead of merging them.
80. `context_pressure` vs the model's window.
81. Compaction-imminent warning.
82. `next_turn_cost` on the live report.
83. `debt_multiple` on the live report.
84. `out_per_turn`, `median_gap`, `ttl` surfaced.
85. Session-length constants corrected to deduplicated values.
86. `Session.gaps/median_gap/cache_misses/compactions/base_context/out_tokens`.

### Performance and robustness (87–96)

87. **mtime+size keyed parse cache** — the prompt hook re-reads 171 transcripts
    with no perceptible pause.
88. Atomic cache write via `replace`.
89. **Per-pid temp name** — several Claude Code sessions share one machine and a
    fixed `.tmp` path let them clobber each other.
90. Corrupt or stale-version caches are ignored, not fatal.
91. Cache pruned to files that still exist.
92. Outcome log: recency weighting (30-day half-life).
93. Outcome log: `prune()` with a row cap.
94. Outcome log: tolerates fields written by a newer version.
95. Outcome log: `effort` / `duration_s` fields, `--json`, `--prune`.
96. Advisor state file bounded and written atomically.

### Hooks: preventing cost, not reporting it (97–104)

97. **New `PreToolUse` read guard** — prices a large read *before* it lands in
    context. The only component here that prevents spend.
98. Prices against the *current* session, not a global average.
99. Detects already-bounded commands (`head`, `wc`, `-n`) and stays silent.
100. Flags unbounded verbose commands (`cat`, `find`, `git log`, recursive greps).
101. Advisory by default; `ADDER_GUARD_BLOCK=1` escalates to a confirmation.
102. Never fires below a cost threshold, and never breaks a turn.
103. Session advisor now uses the parse cache and reports context pressure.
104. Advisor message is horizon-aware rather than quoting a countdown.

### Agents, docs, tooling (105–115)

105. `Explore` gains explicit output-bounding rules targeting the measured #1
     source of admitted context.
106. `route-t0` states its 200K window and escalates rather than truncating.
107. `route-t1` / `route-t2` gain bounded-output and targeted-read rules.
108. `route-t2` gains cost discipline as the expensive tier.
109. `route-doctor` rewritten: leads with the measured split, warns that terseness
     only reaches half of it, prefers effort over downgrade, requires the
     feasibility check.
110. `adder help` usage screen; new commands registered.
111. `adder trace --json` and `--no-cache`.
112. `--verify` replaced pinned dollar figures with structural invariants,
     including an input+output reconciliation check.
113. README rewritten around the corrected numbers, and leads with the
     measurement bug rather than burying it.
114. README caveat that these are one workload's numbers.
115. `CHANGELOG.md` (this file).

### Tests (116)

116. **143 new tests, 263 total** (from 120), covering deduplication, TTL
     detection, fast-mode pricing, feasibility gates, cache-miss attribution,
     growth attribution, quality proxies, policy gates, and the new levers.

[Unreleased]: https://github.com/stephenoffer/adder/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/stephenoffer/adder/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/stephenoffer/adder/releases/tag/v0.1.0
