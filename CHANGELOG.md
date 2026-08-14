# Changelog

Improvements to make the router cut cost more reliably, without degrading the
agent. Grouped by area; every item is in the working tree and covered by tests
unless noted.

## The correction that reframes everything (1–7)

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

## Feasibility: savings that were impossible (8–17)

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

## Cache mechanics (18–34)

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
26. **New `router/cache.py`**: cache efficiency and rebuild waste.
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

## Where context actually comes from (35–45)

35. **New `router/context.py`**: attribute growth to its sources.
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

## New levers (46–55)

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
54. Savings report ends by pointing at `rt quality`.
55. Every modelled lever states its assumption inline.

## Maintaining agent performance (56–66)

56. **New `router/quality.py`**: performance proxies from transcripts.
57. Tool error rate (`is_error` tool results).
58. Correction rate (operator redirect phrasing).
59. Interrupt rate.
60. Turns per human prompt.
61. Rework ratio (edits per distinct file).
62. Tool replies and injected meta-messages excluded from "human prompts".
63. `regressions()` with a noise tolerance.
64. Before/after windowing by date.
65. **`rt verify` refuses to claim a clean saving when a proxy regressed** — a
    cheaper agent that needs more turns is not cheaper.
66. `rt quality --since DATE` as a standalone guard.

## Routing policy (67–77)

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

## Session analysis (78–86)

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

## Performance and robustness (87–96)

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

## Hooks: preventing cost, not reporting it (97–104)

97. **New `PreToolUse` read guard** — prices a large read *before* it lands in
    context. The only component here that prevents spend.
98. Prices against the *current* session, not a global average.
99. Detects already-bounded commands (`head`, `wc`, `-n`) and stays silent.
100. Flags unbounded verbose commands (`cat`, `find`, `git log`, recursive greps).
101. Advisory by default; `ROUTER_GUARD_BLOCK=1` escalates to a confirmation.
102. Never fires below a cost threshold, and never breaks a turn.
103. Session advisor now uses the parse cache and reports context pressure.
104. Advisor message is horizon-aware rather than quoting a countdown.

## Agents, docs, tooling (105–115)

105. `Explore` gains explicit output-bounding rules targeting the measured #1
     source of admitted context.
106. `route-t0` states its 200K window and escalates rather than truncating.
107. `route-t1` / `route-t2` gain bounded-output and targeted-read rules.
108. `route-t2` gains cost discipline as the expensive tier.
109. `route-doctor` rewritten: leads with the measured split, warns that terseness
     only reaches half of it, prefers effort over downgrade, requires the
     feasibility check.
110. `rt help` usage screen; new commands registered.
111. `rt trace --json` and `--no-cache`.
112. `--verify` replaced pinned dollar figures with structural invariants,
     including an input+output reconciliation check.
113. README rewritten around the corrected numbers, and leads with the
     measurement bug rather than burying it.
114. README caveat that these are one workload's numbers.
115. `CHANGELOG.md` (this file).

## Tests (116)

116. **143 new tests, 263 total** (from 120), covering deduplication, TTL
     detection, fast-mode pricing, feasibility gates, cache-miss attribution,
     growth attribution, quality proxies, policy gates, and the new levers.
