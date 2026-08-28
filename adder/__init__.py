"""Cache- and context-aware cost tooling for Claude Code agent sessions.

Everything in this package reads local transcript files and computes. There is
no network access, no API key, and no model call anywhere in the library.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.2.0"
