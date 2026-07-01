"""Reader for COCONUT Tecplot ``.plt`` solutions (the COOLFluiD corona model, Tecplot export).

The Tecplot sibling of :class:`~qorona.io.readers.coconut.cfmesh.CFmeshReader`: it parses the
model-agnostic Tecplot container (:func:`qorona.io.formats.parse_tecplot`) and attaches COCONUT
meaning. COCONUT writes a single ``FEBRICK`` volume zone holding nodal coordinates ``x0,x1,x2`` and
the cell-centred corona MHD state ``rho,u,v,w,Bx,By,Bz,p,phi`` (with optional extra fields), in
COOLFluiD "corona" normalization. Unlike ``.CFmesh`` the export carries no tagged boundary surfaces,
so the inner/outer shells are synthesised from the node radial extremes.

The variable set varies between exports (the nine MHD fields are always present; extras are not), so
the names come from the file's own ``VARIABLES`` line; the velocity ``u,v,w`` and the GLM scalar
``phi`` become Qorona's canonical ``vx,vy,vz,psi`` and any extra fields are kept as written.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
from astropy import units as u

from qorona.io.formats import parse_tecplot
from qorona.io.formats.tecplot import PRISM_COLUMNS
from qorona.io.native import Boundary, NativeSolution, SolutionMetadata
from qorona.io.readers.base import SolutionReader

#: Fraction of the radial span within which a node counts as on the inner/outer shell.
_SHELL_TOLERANCE = 1.0e-3


class CoconutTecplotReader(SolutionReader):
    """Read a COCONUT Tecplot ``.plt`` solution into a :class:`NativeSolution`."""

    model = "coconut"
    file_format = "tecplot"
    extensions = (".plt",)

    #: Cartesian nodal coordinate names, in order.
    COORDINATE_NAMES: ClassVar[tuple[str, ...]] = ("x0", "x1", "x2")

    #: Tecplot cell-centred field names mapped to Qorona's canonical names; others kept as written.
    RENAME: ClassVar[dict[str, str]] = {"u": "vx", "v": "vy", "w": "vz", "phi": "psi"}

    def __init__(self, variables: tuple[str, ...] | None = None) -> None:
        """Initialise the reader.

        Parameters
        ----------
        variables
            Not supported: a Tecplot file is self-describing (the names come from its
            ``VARIABLES`` line), so passing a value raises :class:`ValueError`.
        """
        if variables is not None:
            raise ValueError(
                "CoconutTecplotReader takes its variable names from the file's VARIABLES line; "
                "the `variables` override applies to the CFmesh reader only."
            )

    def read(self, path: str | Path, *, show_progress: bool = True) -> NativeSolution:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tecplot file not found: {path}")

        parsed = parse_tecplot(path, show_progress=show_progress)

        try:
            nodes = np.column_stack([parsed.nodal[name] for name in self.COORDINATE_NAMES])
        except KeyError as missing:
            raise ValueError(
                f"Tecplot file lacks a nodal coordinate among {self.COORDINATE_NAMES}: {missing}."
            ) from None

        variables = {self.RENAME.get(name, name): values for name, values in parsed.cell.items()}

        prism = parsed.connectivity[:, PRISM_COLUMNS]
        cell_centers = nodes[prism].mean(axis=1)
        boundaries = _synthesize_boundaries(nodes)

        metadata = SolutionMetadata(
            model=self.model,
            file_format=self.file_format,
            source_path=path,
            normalization="corona",
            dimension=3,
            n_equations=len(parsed.cell),
            element_type="Prism",
            extra={"tecplot_title": parsed.title, "solution_time": repr(parsed.solution_time)},
        )

        return NativeSolution(
            nodes=nodes * u.R_sun,
            connectivity=prism.astype(np.int32),
            cell_centers=cell_centers * u.R_sun,
            variables=variables,
            boundaries=boundaries,
            metadata=metadata,
        )


def _synthesize_boundaries(nodes: np.ndarray) -> dict[str, Boundary]:
    """Build inner/outer boundaries from the node radial extremes.

    The ``.plt`` export carries no tagged surfaces, so the inner and outer shells are the
    minimum- and maximum-radius node sets. Only ``mean_radius`` (and a descriptive ``source_name``)
    is recorded; ``faces`` / ``adjacent_cells`` are left empty, matching what downstream consumes
    from a boundary (the domain radii come from the run's grid configuration, not the solution).
    """
    radius = np.linalg.norm(nodes, axis=1)
    r_min, r_max = float(radius.min()), float(radius.max())
    tolerance = _SHELL_TOLERANCE * max(r_max - r_min, r_max)

    def shell(mask: np.ndarray, role: str, radius_value: float) -> Boundary:
        return Boundary(
            name=role,
            source_name=f"r={radius_value:.3f} shell (synthesized)",
            faces=np.empty((0, 3), dtype=np.int32),
            adjacent_cells=np.empty((0,), dtype=np.int32),
            mean_radius=float(radius[mask].mean()) * u.R_sun,
        )

    return {
        "inner": shell(radius <= r_min + tolerance, "inner", r_min),
        "outer": shell(radius >= r_max - tolerance, "outer", r_max),
    }
