"""Qorona: synthetic coronal imagery from global MHD solutions.

Qorona post-processes global coronal MHD solutions into eclipse-like renderings,
primarily a line-of-sight integral of the magnetic squashing factor Q-perp.
COCONUT (``.CFmesh``) is the supported input; further models plug in behind the
same reader interface.
"""

from qorona.io import NativeSolution, read_solution

__version__ = "0.4.0"

__all__ = ["NativeSolution", "__version__", "read_solution"]
