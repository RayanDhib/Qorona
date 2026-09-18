"""Reader for Icarus solutions exported as VTK unstructured files (``.vtu``).

Icarus is an inner-heliosphere MHD model built on MPI-AMRVAC. Its ``.vtu`` snapshots
carry primitive variables on a spherical block grid whose point coordinates are stored
unprojected as (x, y, z) = (r [R_sun], theta [rad], phi [rad]); cell components
``b1..b3`` are the spherical (B_r, B_theta, B_phi) in nT and ``rho`` is the mass density
in kg/m^3, both kept native (they are scale-free for Qorona's products).

Icarus: Verbeke et al. (2022), A&A 662, A50; Baratashvili et al. (2025), A&A 694, A306.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy import units as u

from qorona.geometry.coordinates import spherical_to_cartesian, spherical_to_cartesian_vectors
from qorona.io.formats.vtu import ParsedVtu, parse_vtu
from qorona.io.native import NativeSolution, SolutionMetadata, StructuredMesh
from qorona.io.readers.base import SolutionReader, synthesize_shell_boundaries
from qorona.io.readers.mas.hdf import spherical_mesh_points

_REQUIRED = ("rho", "b1", "b2", "b3")


class IcarusVtuReader(SolutionReader):
    """Reader for Icarus heliospheric ``.vtu`` snapshots."""

    model = "icarus"
    file_format = "vtu"
    extensions = (".vtu",)
    category = "heliosphere"

    def __init__(self, variables: tuple[str, ...] | None = None) -> None:
        """Initialise the reader.

        Parameters
        ----------
        variables
            Not supported: an Icarus ``.vtu`` is self-describing (``rho`` plus the spherical
            field components ``b1, b2, b3``), so passing a value raises :class:`ValueError`.
        """
        if variables is not None:
            raise ValueError(
                "The Icarus VTU reader reads a self-describing file; "
                "`variables` cannot be overridden."
            )

    def read(self, path: str | Path, *, show_progress: bool = True) -> NativeSolution:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Solution file not found: {path}")
        if path.suffix != ".vtu":
            raise ValueError(f"Compressed .vtu files are not supported ({path.name}).")
        parsed = parse_vtu(path, cell_arrays=_REQUIRED, show_progress=show_progress)

        found = set(parsed.pieces[0].cell_data)
        missing = sorted(set(_REQUIRED) - found)
        if missing:
            raise ValueError(f"{path.name} lacks required cell arrays {missing}.")

        centers, columns, radial_edges = _gather_cells(parsed)
        axes, cubes = _to_product_grid(centers, columns, path)
        mesh = StructuredMesh(radii=axes[0], colatitudes=axes[1], azimuths=axes[2])

        points = spherical_to_cartesian(spherical_mesh_points(mesh))
        b_spherical = np.column_stack(
            [cubes["b1"].ravel(), cubes["b2"].ravel(), cubes["b3"].ravel()]
        )
        b_cartesian = spherical_to_cartesian_vectors(b_spherical, points)
        variables = {
            "rho": cubes["rho"].ravel(),
            "Bx": b_cartesian[:, 0],
            "By": b_cartesian[:, 1],
            "Bz": b_cartesian[:, 2],
        }

        metadata = SolutionMetadata(
            model=self.model,
            file_format=self.file_format,
            source_path=path,
            normalization="icarus-si",
            dimension=3,
            n_equations=len(variables),
            element_type="point",
            category=self.category,
            extra={
                "native_mesh": "x".join(str(len(a)) for a in axes),
                "radial_range": f"{radial_edges[0]:.3f}-{radial_edges[1]:.3f}",
                "blocks": str(len(parsed.pieces)),
                "native_time": repr(parsed.field_data.get("TIME")),
            },
        )
        nodes = points * u.R_sun
        return NativeSolution(
            nodes=nodes,
            connectivity=np.empty((0, 8), dtype=np.int32),
            cell_centers=nodes,
            variables=variables,
            boundaries=synthesize_shell_boundaries(radial_edges[0], radial_edges[1]),
            metadata=metadata,
            structured=mesh,
        )


def _gather_cells(
    parsed: ParsedVtu,
) -> tuple[np.ndarray, dict[str, np.ndarray], tuple[float, float]]:
    """Concatenate all pieces into spherical cell centres and per-variable columns.

    The third return is the domain-edge radial extent (min/max node radius across all pieces);
    the model boundaries anchor on these physical domain edges, not the cell-centre axis ends.
    """
    centers = [piece.points[piece.connectivity].mean(axis=1) for piece in parsed.pieces]
    columns = {
        name: np.concatenate([piece.cell_data[name] for piece in parsed.pieces])
        for name in _REQUIRED
    }
    r_min = min(float(piece.points[:, 0].min()) for piece in parsed.pieces)
    r_max = max(float(piece.points[:, 0].max()) for piece in parsed.pieces)
    return np.concatenate(centers), columns, (r_min, r_max)


def _cluster_axis(values: np.ndarray) -> np.ndarray:
    """Merge near-identical coordinates (block-face duplicates) into one increasing axis."""
    order = np.unique(values)
    if len(order) < 2:
        return order
    tolerance = 1.0e-6 * max(float(order[-1] - order[0]), 1.0)
    breaks = np.nonzero(np.diff(order) > tolerance)[0]
    return np.array([float(group.mean()) for group in np.split(order, breaks + 1)])


def _nearest_index(axis: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Index of the nearest axis value for each coordinate."""
    idx = np.clip(np.searchsorted(axis, coords), 1, len(axis) - 1)
    return idx - ((coords - axis[idx - 1]) < (axis[idx] - coords))


def _to_product_grid(
    centers: np.ndarray, columns: dict[str, np.ndarray], path: Path
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, np.ndarray]]:
    """Reconstruct the global (r, theta, phi) axes and scatter cells into cubes.

    Verifies the grid is a single-level separable product (every cell maps to a unique
    (i, j, k)); AMR files fail this and get a clear error.
    """
    axes = (
        _cluster_axis(centers[:, 0]),
        _cluster_axis(centers[:, 1]),
        _cluster_axis(centers[:, 2]),
    )
    n_r, n_theta, n_phi = (len(a) for a in axes)
    indices = tuple(_nearest_index(axes[k], centers[:, k]) for k in range(3))
    flat = (indices[0] * n_theta + indices[1]) * n_phi + indices[2]
    n_cells = len(flat)
    if n_r * n_theta * n_phi != n_cells or len(np.unique(flat)) != n_cells:
        raise ValueError(
            f"{path.name} is not a single-level separable spherical grid "
            f"({n_r}x{n_theta}x{n_phi} axes for {n_cells} cells); "
            f"AMR VTU files are not supported."
        )
    cubes: dict[str, np.ndarray] = {}
    for name, values in columns.items():
        cube = np.empty(n_cells)
        cube[flat] = values
        cubes[name] = cube.reshape(n_r, n_theta, n_phi)
    return axes, cubes
