"""Qorona: synthetic coronal imagery from global MHD solutions.

Qorona post-processes global coronal and heliospheric MHD solutions into synthetic images:
mainly a line-of-sight integral of the magnetic squashing factor Q-perp, and also white light,
field lines and Q-maps. The supported models are COCONUT (``.CFmesh`` and Tecplot ``.plt``), MAS
(HDF4) and Icarus (``.vtu``). Further models plug in behind the same reader interface.
"""

from qorona.io import NativeSolution, read_solution

__version__ = "0.5.0"

__all__ = ["NativeSolution", "__version__", "read_solution"]
