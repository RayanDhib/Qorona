"""COCONUT readers: the COOLFluiD corona model, in its native and Tecplot formats.

One model, one-or-more formats: :class:`~qorona.io.readers.coconut.cfmesh.CFmeshReader` (native
``.CFmesh``) and :class:`~qorona.io.readers.coconut.tecplot.CoconutTecplotReader` (``.plt``). Both
produce the same :class:`~qorona.io.native.NativeSolution`; the Tecplot container parse they could
share with other models lives in :mod:`qorona.io.formats`.
"""

from __future__ import annotations

from qorona.io.readers.coconut.cfmesh import CFmeshReader
from qorona.io.readers.coconut.tecplot import CoconutTecplotReader

__all__ = ["CFmeshReader", "CoconutTecplotReader"]
