# Task runner. Every target works from a bare checkout; PY can point anywhere.
PY ?= python3
PKG := adder

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Editable install with dev extras
	$(PY) -m pip install -e ".[dev]"

.PHONY: test
test: ## Run the test suite
	$(PY) -m pytest

.PHONY: cov
cov: ## Run tests with a coverage report
	$(PY) -m pytest --cov=$(PKG) --cov-report=term-missing --cov-report=xml

.PHONY: lint
lint: ## Lint (no changes written)
	$(PY) -m ruff check .

.PHONY: fmt
# `ruff format` is deliberately NOT a gate here: several modules hand-align
# price and catalog tables into columns, and the formatter destroys that
# alignment for no readability gain. Style is enforced by `ruff check` + review.
fmt: ## Auto-fix lint findings
	$(PY) -m ruff check --fix .

.PHONY: structure
# The layout rules from docs/structure.md, on their own so the failure is
# readable: breadth caps, layer direction, and the test-tree mirror.
structure: ## Check the package layout rules
	$(PY) -m pytest -q tests/repo/test_structure.py

.PHONY: check
check: lint test ## What CI runs: lint + tests

.PHONY: smoke
# The command list is read from adder/cli/commands.py rather than restated here, so a
# new command is covered by this target the moment it is registered.
smoke: ## Every subcommand answers --help without importing the world
	./scripts/adder help >/dev/null
	./scripts/adder version
	@$(PY) -c "from adder.cli import COMMANDS; print(' '.join(c.name for c in COMMANDS))" \
	  | tr ' ' '\n' | while read -r c; do \
	      ./scripts/adder "$$c" --help >/dev/null 2>&1 \
	        || { echo "FAIL: adder $$c --help"; exit 1; }; \
	    done
	@echo "all subcommands respond to --help"

.PHONY: doctor
doctor: ## Run every check against your own transcripts, ranked by dollars
	./scripts/adder doctor

.PHONY: guard
guard: ## Learn result sizes from your transcripts, then price what the guard would say
	./scripts/adder guard --learn --replay

.PHONY: gate
# Exit non-zero when something material is wrong. Intended for a pre-push hook
# or a scheduled job, not for CI on this repo -- it reads the machine's own
# transcripts, which a CI runner does not have.
gate: ## Fail if any check finds something material
	./scripts/adder doctor --strict

.PHONY: build
build: clean ## Build sdist + wheel into dist/
	$(PY) -m build

.PHONY: verify-dist
verify-dist: build ## Check the built artifacts are PyPI-valid and carry data files
	$(PY) -m twine check dist/*
	@$(PY) -c "import zipfile,glob,sys; \
w=glob.glob('dist/*.whl')[0]; n=zipfile.ZipFile(w).namelist(); \
print('wheel:', w); \
missing=[p for p in ('adder/cli/__init__.py','adder/cli/commands.py','adder/py.typed','adder/pricing/data/catalog.json') if p not in n]; \
sys.exit('missing from wheel: %s' % missing) if missing else print('wheel contents ok')"

.PHONY: clean
clean: ## Remove caches, build artifacts, and OS junk
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage coverage.xml htmlcov
	find . -path ./.git -prune -o -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	find . -path ./.git -prune -o -name '*.py[cod]' -delete 2>/dev/null || true
	find . -path ./.git -prune -o -name '.DS_Store' -delete 2>/dev/null || true
	@echo "clean"

.PHONY: hooks
hooks: ## Install the pre-commit and pre-push hooks
	$(PY) -m pip install pre-commit
	pre-commit install
	pre-commit install -t pre-push

.PHONY: release-check
release-check: check verify-dist ## Everything that must pass before tagging a release
	@$(PY) -c "import adder; print('version:', adder.__version__)"
	@grep -q '## \[Unreleased\]' CHANGELOG.md && echo 'CHANGELOG has an Unreleased section' || true
	@echo "release-check passed"
