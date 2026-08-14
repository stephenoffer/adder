# Releasing

Releases are cut from `main` by pushing a tag. CI does the rest. The pipeline is
built so that a mistake fails *before* anything is published rather than after.

## Versioning

[Semantic Versioning](https://semver.org). For a tool whose output is numbers,
read the categories this way:

| Bump | When |
|---|---|
| **major** | A command is removed or renamed, output JSON changes shape, or a default changes in a way that flips a recommendation. |
| **minor** | A new command, a new lever, a new flag, a new field in `--json`. |
| **patch** | A bug fix, a documentation fix, a corrected figure that does not change a recommendation. |

One extra rule: **a corrected measurement that changes a headline number is at
minimum a minor bump, even if no code signature changed.** Someone made a
decision on the old number.

`adder/__version__` in `adder/__init__.py` is the single source of truth.
`pyproject.toml` reads it dynamically — never restate a version there, and
`tests/test_cli.py` fails if you do.

## Steps

```bash
# 1. Everything must be green, including the built artifacts.
make release-check

# 2. Bump the version.
$EDITOR adder/__init__.py          # __version__ = "0.2.0"

# 3. Promote the Unreleased section in CHANGELOG.md.
#    "## [Unreleased]" -> "## [0.2.0] - YYYY-MM-DD", then add a fresh
#    empty "## [Unreleased]" above it and update the link refs at the bottom.
$EDITOR CHANGELOG.md

# 4. Commit, PR, merge. CI must be green on main.
git commit -am "chore: release 0.2.0"

# 5. Tag and push. The tag drives everything after this point.
git tag -a v0.2.0 -m "v0.2.0"
git push origin v0.2.0
```

## What the tag triggers

`.github/workflows/release.yml` runs four jobs in order. Each one is a gate.

1. **guard** — refuses to continue unless `v<tag>` equals `adder.__version__`
   *and* `CHANGELOG.md` contains a `## [<version>]` section. This is the check
   that catches the most common release mistake: tagging before bumping.
2. **test** — the full matrix, Python 3.10 through 3.14, plus `ruff`. `fail-fast`
   is on; a release does not proceed on a partial pass.
3. **build** — `python -m build` and `twine check`.
4. **github-release** and **pypi** — the release notes are extracted from this
   version's CHANGELOG section, so what users read is what you wrote. PyPI
   publishing uses [trusted publishing](https://docs.pypi.org/trusted-publishers/)
   over OIDC; there is no API token stored in this repository.

## One-time PyPI setup

The `pypi` job fails until this exists. That failure does not block the GitHub
Release, which is deliberate — you can ship before PyPI is wired up.

1. On PyPI, add a trusted publisher for the project **`adder-cli`** (the bare
   name `adder` is held by an unrelated 2014 package — see
   [naming.md](naming.md)): owner `stephenoffer`, repository `adder`,
   workflow `release.yml`, environment `pypi`.
2. In GitHub repository settings, create an environment named `pypi`. Add
   required reviewers if you want a human approval step before upload.

Nothing else. No token to rotate, no secret to leak.

## Dry run

To exercise the build and publish path without cutting a version, use the manual
trigger: **Actions -> Release -> Run workflow**, leaving "Publish to PyPI"
unticked. The guard job skips its tag comparison on a manual run, so this
validates the build without needing a tag.

## Hotfix

Branch from the release tag, not from `main`:

```bash
git checkout -b hotfix/0.2.1 v0.2.0
# fix, test, bump to 0.2.1, add a CHANGELOG section
git tag -a v0.2.1 -m "v0.2.1" && git push origin v0.2.1
```

Then merge the fix forward into `main`. A hotfix that only exists on the tag is
a regression waiting for the next release.

## If a release goes wrong

- **Caught before the tag is pushed:** delete the local tag and start over.
- **Caught after the tag but before PyPI upload:** delete the tag and the draft
  GitHub Release, fix, re-tag the same version. Nothing external consumed it yet.
- **Caught after PyPI upload:** do not delete the release. PyPI does not allow
  re-uploading a version, and yanking breaks anyone who already pinned it.
  Ship a patch version with the fix and yank the bad one only if it is actively
  harmful.
