"""Reader interface for reading MHD solutions.

A reader converts one model/format into a :class:`~qorona.io.native.NativeSolution`.
Supporting a new model means adding one ``SolutionReader`` subclass; nothing
downstream changes. A model may own more than one format (one reader each); the container
parsing they share lives in :mod:`qorona.io.formats`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from astropy import units as u

from qorona.io.native import Boundary, NativeSolution
from qorona.io.textio import open_solution_text, payload_suffix

__all__ = [
    "SolutionReader",
    "open_solution_text",
    "payload_suffix",
    "synthesize_shell_boundaries",
]


class SolutionReader(ABC):
    """Base class for model-specific solution readers.

    Subclasses set the class attributes below and implement :meth:`read`.

    Attributes
    ----------
    model
        Short model identifier (e.g. ``"coconut"``).
    file_format
        Short format identifier (e.g. ``"cfmesh"``).
    extensions
        File extensions this reader recognises (including the leading dot).
    category
        The solution category this reader ingests: ``"corona"`` (domain rooted at the
        solar surface; the locked coronal defaults apply) or ``"heliosphere"`` (domain far
        above the surface; the internal shell adopts the native radial extent and the
        scale-coupled defaults derive from it).
    """

    model: str
    file_format: str
    extensions: tuple[str, ...]
    category: str = "corona"

    @abstractmethod
    def read(self, path: str | Path, *, show_progress: bool = True) -> NativeSolution:
        """Read a solution file into a :class:`NativeSolution`."""

    @classmethod
    def handles(cls, path: str | Path) -> bool:
        """Return whether this reader recognises ``path`` by its (payload) extension."""
        return payload_suffix(path) in cls.extensions

    @classmethod
    def identifies(cls, path: str | Path) -> bool:
        """Return whether the file's *content* identifies it as this reader's model.

        The disambiguation seam for a format that more than one model can write (e.g. a
        Tecplot ``.plt``): when several readers share an extension, the registry consults
        this content check to pick one. The default returns ``False`` (no claim); a reader
        overrides it only when it must be told apart from a sibling on the same extension.
        Unused today (each extension has a single claimant); kept as the documented hook.
        """
        return False


def synthesize_shell_boundaries(inner: float, outer: float) -> dict[str, Boundary]:
    """Face-less inner/outer shell boundaries for meshes whose file carries no tags.

    Structured models (MAS, Icarus) store no boundary surfaces; only a ``mean_radius`` and a
    descriptive ``source_name`` are consumed downstream, so the shells carry empty face and
    adjacency arrays.

    Parameters
    ----------
    inner, outer
        Radii of the inner and outer shells, in R_sun.

    Returns
    -------
    dict[str, Boundary]
        The ``"inner"`` and ``"outer"`` shell boundaries keyed by canonical role.
    """

    def shell(role: str, radius: float) -> Boundary:
        return Boundary(
            name=role,
            source_name=f"r={radius:.3f} shell (synthesized)",
            faces=np.empty((0, 4), dtype=np.int32),
            adjacent_cells=np.empty((0,), dtype=np.int32),
            mean_radius=radius * u.R_sun,
        )

    return {"inner": shell("inner", inner), "outer": shell("outer", outer)}
