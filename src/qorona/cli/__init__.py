"""Command-line entry points.

The ``qorona`` click group (``build`` / ``render`` / ``run`` / ``info``) over the cached
viewpoint-independent Q⊥ volume, with Rich progress and a polished end-of-run metrics summary.
"""

from __future__ import annotations

from qorona.cli.main import main

__all__ = ["main"]
