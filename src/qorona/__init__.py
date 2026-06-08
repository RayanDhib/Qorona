"""Qorona: synthetic coronal imagery from global MHD solutions.

Qorona post-processes global coronal MHD solutions (COCONUT, MAS, and others) into
eclipse-like renderings, primarily a line-of-sight integral of the magnetic
squashing factor Q-perp.
"""

from qorona.io import NativeSolution, read_solution

__version__ = "0.1.0"

__all__ = ["NativeSolution", "__version__", "read_solution"]
