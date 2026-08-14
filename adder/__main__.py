"""Allow `python -m adder` as an equivalent of the `adder` console script."""

from __future__ import annotations

import sys

from adder.cli import run

if __name__ == "__main__":
    sys.exit(run())
