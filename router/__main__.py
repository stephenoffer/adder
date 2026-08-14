"""Allow `python -m router` as an equivalent of the `rt` console script."""

from __future__ import annotations

import sys

from router.cli import run

if __name__ == "__main__":
    sys.exit(run())
