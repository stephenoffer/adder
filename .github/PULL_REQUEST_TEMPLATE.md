## What and why

<!-- What changes, and what decision or measurement motivated it.
     If this moves a reported number, the measurement goes here. -->

## Measurement

<!-- Required if any figure changes. Paste the before/after, and say what data
     it ran against (how many sessions, turns, over what period).
     "This looks more correct" is not a measurement. Delete if not applicable. -->

## Checklist

- [ ] `make check` passes (`ruff check .` + `pytest`)
- [ ] New behaviour has a test in this PR
- [ ] No new runtime dependency
- [ ] No network call outside `adder/sources.py`
- [ ] Nothing writes under `~/.claude/`
- [ ] Tests use `tmp_path`, not the real transcript directory
- [ ] `CHANGELOG.md` updated under `## [Unreleased]` if user-visible
- [ ] `docs/commands.md` and the README table updated if a command changed
- [ ] No untracked junk in `git status`

## Notes for the reviewer

<!-- Anything non-obvious: a tradeoff you made, a thing you chose not to do,
     an invariant in CLAUDE.md you had to work around. -->
