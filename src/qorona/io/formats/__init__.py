"""Container-format parsers, independent of any model.

A format parser turns a file container (e.g. COOLFluiD Tecplot ``.plt``) into a neutral,
model-agnostic structure carrying the raw mesh and named variable arrays. A model reader
in :mod:`qorona.io.readers` then attaches meaning (canonical variable names, normalization,
units, boundaries) and builds the :class:`~qorona.io.native.NativeSolution`. Splitting the two
lets one container format serve several models (the same ``.plt`` is written by more than one
COOLFluiD model).
"""

from __future__ import annotations

from qorona.io.formats.tecplot import ParsedTecplot, parse_tecplot

__all__ = ["ParsedTecplot", "parse_tecplot"]
