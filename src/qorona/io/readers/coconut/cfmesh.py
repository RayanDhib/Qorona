"""Reader for COOLFluiD ``.CFmesh`` solutions (the COCONUT coronal MHD model).

CFmesh is COOLFluiD's native ASCII mesh+solution format: a keyword-delimited
header (``!NB_NODES``, ``!NB_ELEM``, ...) followed by element connectivity
(``!LIST_ELEM``), tagged boundary surfaces (``!TRS_NAME`` / ``!LIST_GEOM_ENT``),
node coordinates (``!LIST_NODE``), and cell-centred state vectors (``!LIST_STATE``).

COCONUT discretises the corona as a geodesic icosahedron radially extruded into
triangular prisms (6 nodes per cell), with one piecewise-constant state per cell.
For full MHD the state holds nine variables in COOLFluiD "corona" normalization
(dimensionless): density, the three velocity components, the three magnetic-field
components, pressure, and the GLM divergence-cleaning scalar.

The parse streams the file and averages each prism's six node coordinates to a cell
centre, reading the tagged inner/outer boundaries and validating the variable count
against the file's own ``!NB_EQ``.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import TextIO

import numpy as np
from astropy import units as u

from qorona.console import status
from qorona.io.native import Boundary, NativeSolution, SolutionMetadata
from qorona.io.readers.base import SolutionReader
from qorona.io.textio import open_solution_text


class CFmeshReader(SolutionReader):
    """Read a COCONUT ``.CFmesh`` solution into a :class:`NativeSolution`."""

    model = "coconut"
    file_format = "cfmesh"
    extensions = (".CFmesh",)

    #: Cell-centred state layout for COCONUT full MHD (COOLFluiD corona normalization).
    DEFAULT_VARIABLES: tuple[str, ...] = (
        "rho",
        "vx",
        "vy",
        "vz",
        "Bx",
        "By",
        "Bz",
        "p",
        "psi",
    )

    def __init__(self, variables: tuple[str, ...] | None = None) -> None:
        """Initialise the reader.

        Parameters
        ----------
        variables
            Names for the state-vector columns, in file order. Must match the
            file's ``!NB_EQ``. Defaults to the COCONUT full-MHD layout.
        """
        self.variables = tuple(variables) if variables is not None else self.DEFAULT_VARIABLES

    def read(self, path: str | Path, *, show_progress: bool = True) -> NativeSolution:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CFmesh file not found: {path}")

        header: dict[str, str] = {}
        nodes: np.ndarray | None = None
        connectivity: np.ndarray | None = None
        state_index: np.ndarray | None = None
        states: np.ndarray | None = None
        raw_boundaries: list[dict[str, object]] = []
        current_name = ""
        current_n_faces = 0

        with (
            status(f"Reading {path.name}", enabled=show_progress),
            open_solution_text(path) as handle,
        ):
            for line in handle:
                if not line.startswith("!"):
                    continue
                key, _, rest = line.strip().partition(" ")
                rest = rest.strip()

                if key == "!LIST_ELEM":
                    n_nodes_per_cell = int(header["!NB_NODES_PER_TYPE"])
                    block = _read_block(handle, int(header["!NB_ELEM"]), dtype=np.int64)
                    connectivity = block[:, :n_nodes_per_cell].astype(np.int32)
                    state_index = block[:, n_nodes_per_cell]
                elif key == "!LIST_NODE":
                    nodes = _read_block(handle, _count(header["!NB_NODES"]), dtype=np.float64)
                elif key == "!LIST_STATE":
                    states = _read_block(handle, _count(header["!NB_STATES"]), dtype=np.float64)
                elif key == "!TRS_NAME":
                    current_name = rest
                elif key == "!NB_GEOM_ENTS":
                    current_n_faces = int(rest)
                elif key == "!LIST_GEOM_ENT":
                    block = _read_block(handle, current_n_faces, dtype=np.int64)
                    raw_boundaries.append({"name": current_name, "entities": block})
                elif key == "!END":
                    break
                else:
                    header[key] = rest

        if nodes is None or connectivity is None or state_index is None or states is None:
            raise ValueError(f"Incomplete CFmesh file (missing a required section): {path}")

        n_equations = int(header["!NB_EQ"])
        if len(self.variables) != n_equations:
            raise ValueError(
                f"File has NB_EQ={n_equations} state variables but "
                f"{len(self.variables)} names were given ({list(self.variables)}). "
                f"Pass a matching `variables` tuple."
            )

        # Align states to cells (one piecewise-constant state per cell) and split
        # the state vector into named cell-centred fields.
        cell_states = states[state_index]
        variables = {name: cell_states[:, i] for i, name in enumerate(self.variables)}

        cell_centers = nodes[connectivity].mean(axis=1)
        boundaries = _build_boundaries(raw_boundaries, nodes)

        metadata = SolutionMetadata(
            model=self.model,
            file_format=self.file_format,
            source_path=path,
            normalization="corona",
            dimension=int(header["!NB_DIM"]),
            n_equations=n_equations,
            element_type=header.get("!ELEM_TYPES", "Prism"),
            extra={"coolfluid_version": header.get("!COOLFLUID_VERSION", "")},
        )

        return NativeSolution(
            nodes=nodes * u.R_sun,
            connectivity=connectivity,
            cell_centers=cell_centers * u.R_sun,
            variables=variables,
            boundaries=boundaries,
            metadata=metadata,
        )


def _count(value: str) -> int:
    """Parse a count from a header value such as ``"768150 0"``."""
    return int(value.split()[0])


def _read_block(handle: TextIO, n_rows: int, *, dtype: type) -> np.ndarray:
    """Read exactly ``n_rows`` whitespace-delimited numeric rows from ``handle``."""
    return np.loadtxt(itertools.islice(handle, n_rows), dtype=dtype, ndmin=2)


def _build_boundaries(
    raw_boundaries: list[dict[str, object]], nodes: np.ndarray
) -> dict[str, Boundary]:
    """Turn parsed boundary entities into canonical inner/outer surfaces.

    Each ``!LIST_GEOM_ENT`` row is ``[n_face_nodes, n_face_states, node ids...,
    cell id]``. Boundaries are tagged by mean radius: the innermost becomes
    ``"inner"`` (the seeding surface), the outermost ``"outer"``.
    """
    surfaces: list[tuple[float, str, np.ndarray, np.ndarray]] = []
    for raw in raw_boundaries:
        entities = np.asarray(raw["entities"])
        n_face_nodes = int(entities[0, 0])
        faces = entities[:, 2 : 2 + n_face_nodes]
        adjacent_cells = entities[:, 2 + n_face_nodes]
        mean_radius = float(np.linalg.norm(nodes[faces.ravel()], axis=1).mean())
        surfaces.append((mean_radius, str(raw["name"]), faces, adjacent_cells))

    surfaces.sort(key=lambda surface: surface[0])
    roles = _assign_roles(len(surfaces))

    boundaries: dict[str, Boundary] = {}
    for role, (mean_radius, source_name, faces, adjacent_cells) in zip(
        roles, surfaces, strict=True
    ):
        boundaries[role] = Boundary(
            name=role,
            source_name=source_name,
            faces=faces.astype(np.int32),
            adjacent_cells=adjacent_cells.astype(np.int32),
            mean_radius=mean_radius * u.R_sun,
        )
    return boundaries


def _assign_roles(n_surfaces: int) -> list[str]:
    """Canonical role names for radius-sorted surfaces (innermost first)."""
    if n_surfaces == 2:
        return ["inner", "outer"]
    return [f"boundary_{i}" for i in range(n_surfaces)]
