"""MAS solution readers (one reader per file format)."""

from qorona.io.readers.mas.hdf import MasHdf5Reader, MasHdfReader

__all__ = ["MasHdf5Reader", "MasHdfReader"]
