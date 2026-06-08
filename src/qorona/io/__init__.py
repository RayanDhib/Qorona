"""Solution reading: readers and the model-agnostic ``NativeSolution`` container."""

from __future__ import annotations

from qorona.io.native import Boundary, NativeSolution, SolutionMetadata
from qorona.io.readers import available_models, read_solution, register_reader

__all__ = [
    "Boundary",
    "NativeSolution",
    "SolutionMetadata",
    "available_models",
    "read_solution",
    "register_reader",
]
