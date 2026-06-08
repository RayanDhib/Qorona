"""Model-agnostic container for an MHD solution.

Every reader, regardless of source model or file format, produces a
``NativeSolution``: the raw mesh and cell-centred fields on their native grid,
in the model's own units. All format- and model-specific knowledge lives in the
readers; everything downstream (resampling onto Qorona's internal mesh, tracing,
squashing-factor computation, rendering) consumes only this structure.

Field values are kept in their native normalization (recorded in the metadata);
converting them to physical units is a separate, explicit step. Coordinates are
unambiguous and are carried as astropy quantities in solar radii.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.units import Quantity


@dataclass
class Boundary:
    """A tagged boundary surface of the mesh (e.g. the inner/outer sphere).

    Attributes
    ----------
    name
        Canonical role of the surface: ``"inner"`` or ``"outer"``.
    source_name
        The boundary's name in the source file (e.g. ``"Inlet"``, ``"Outlet"``).
    faces
        ``(n_faces, n_nodes_per_face)`` node indices into ``NativeSolution.nodes``.
    adjacent_cells
        ``(n_faces,)`` index of the cell adjacent to each face.
    mean_radius
        Mean radial distance of the surface's nodes.
    """

    name: str
    source_name: str
    faces: np.ndarray
    adjacent_cells: np.ndarray
    mean_radius: Quantity

    @property
    def n_faces(self) -> int:
        """Number of boundary faces."""
        return int(self.faces.shape[0])


@dataclass
class SolutionMetadata:
    """Provenance and descriptive metadata for a solution."""

    model: str
    file_format: str
    source_path: Path
    normalization: str
    dimension: int
    n_equations: int
    element_type: str
    length_unit: str = "R_sun"
    extra: dict[str, str] = field(default_factory=dict)


@dataclass
class NativeSolution:
    """An MHD solution on its native mesh, in its native units.

    Attributes
    ----------
    nodes
        ``(n_nodes, 3)`` Cartesian node coordinates.
    connectivity
        ``(n_cells, n_nodes_per_cell)`` node indices defining each cell.
    cell_centers
        ``(n_cells, 3)`` Cartesian cell-centre coordinates.
    variables
        Cell-centred field values, keyed by name (native normalization).
    boundaries
        Tagged boundary surfaces, keyed by canonical role (``"inner"``/``"outer"``).
    metadata
        Provenance and grid description.
    """

    nodes: Quantity
    connectivity: np.ndarray
    cell_centers: Quantity
    variables: dict[str, np.ndarray]
    boundaries: dict[str, Boundary]
    metadata: SolutionMetadata

    @property
    def n_nodes(self) -> int:
        """Number of mesh nodes."""
        return int(self.nodes.shape[0])

    @property
    def n_cells(self) -> int:
        """Number of mesh cells."""
        return int(self.cell_centers.shape[0])

    @property
    def variable_names(self) -> list[str]:
        """Names of the available cell-centred fields."""
        return list(self.variables)

    def vector(self, *components: str) -> np.ndarray:
        """Assemble a vector field from its named scalar components.

        Parameters
        ----------
        *components
            Variable names of the components, in order (e.g. ``"Bx", "By", "Bz"``).

        Returns
        -------
        numpy.ndarray
            ``(n_cells, len(components))`` array of the stacked components.
        """
        return np.column_stack([self.variables[name] for name in components])

    @property
    def magnetic_field(self) -> np.ndarray:
        """``(n_cells, 3)`` cell-centred magnetic field (native normalization)."""
        return self.vector("Bx", "By", "Bz")

    def __repr__(self) -> str:
        return (
            f"<NativeSolution model={self.metadata.model!r} "
            f"cells={self.n_cells} nodes={self.n_nodes} "
            f"variables={self.variable_names} "
            f"boundaries={list(self.boundaries)}>"
        )


def radial_distance(points: Quantity) -> Quantity:
    """Return the radial distance ``|r|`` of Cartesian ``(..., 3)`` points."""
    return np.linalg.norm(points.to_value(u.R_sun), axis=-1) * u.R_sun
