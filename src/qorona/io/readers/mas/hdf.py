"""Reader for MAS coronal solutions (one HDF4 file per variable).

MAS solves the global corona on a structured spherical (r, θ, φ) mesh and writes each variable
to its own HDF4 file (``rho002.hdf``, ``br002.hdf``, ...), each holding one 3-D dataset with the
file's own 1-D coordinate axes; the per-variable meshes are mutually staggered. The reader
accepts any one file of the set, discovers the siblings it needs next to it, interpolates every
variable onto the density mesh, converts the spherical field components to Cartesian there, and
emits a point-value :class:`NativeSolution` that also carries the structured axes for the
structured resampler. MAS: Mikic et al. (1999), Phys. Plasmas 6, 2217; the thermodynamic
corona: Lionello et al. (2009), ApJ 690, 902.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import units as u

from qorona.console import status
from qorona.geometry import spherical_to_cartesian
from qorona.io.native import Boundary, NativeSolution, SolutionMetadata, StructuredMesh
from qorona.io.readers.base import SolutionReader

#: Variable-name tokens a MAS run writes; used to locate the anchor file's token.
_MAS_VARIABLES = ("rho", "br", "bt", "bp", "vr", "vt", "vp", "t", "p")

#: Spherical magnetic-field components the reader requires (converted to Cartesian B).
_B_COMPONENTS = ("br", "bt", "bp")

#: HDF4 dataset name each MAS file stores its variable under.
_DATASET_NAME = "Data-Set-2"

#: Friendly guidance shown when a MAS file is read without ``pyhdf`` installed.
PYHDF_MISSING_HINT = (
    "reading MAS solutions needs pyhdf; install it with `pip install pyhdf` "
    "or `conda install -c conda-forge pyhdf`"
)


@dataclass
class _MasVariable:
    """One variable as read from its file: its own mesh and the (n_r, n_theta, n_phi) values."""

    mesh: StructuredMesh
    values: np.ndarray


class MasHdfReader(SolutionReader):
    """Read a MAS one-file-per-variable HDF4 solution set into a :class:`NativeSolution`."""

    model = "mas"
    file_format = "hdf4"
    extensions = (".hdf",)

    def __init__(self, variables: tuple[str, ...] | None = None) -> None:
        """Initialise the reader.

        Parameters
        ----------
        variables
            Not supported: the MAS variable set is fixed (``br, bt, bp`` plus optional
            ``rho``) and each file is self-describing, so passing a value raises
            :class:`ValueError`.
        """
        if variables is not None:
            raise ValueError(
                "MasHdfReader reads a fixed variable set from self-describing files; "
                "the `variables` override applies to the CFmesh reader only."
            )

    def read(self, path: str | Path, *, show_progress: bool = True) -> NativeSolution:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"MAS HDF file not found: {path}")
        files = _sibling_paths(path)

        fields: dict[str, _MasVariable] = {}
        with status(f"Reading {len(files)} MAS variable files", enabled=show_progress):
            for name, file in files.items():
                fields[name] = _read_variable(file)

        mesh = (fields["rho"] if "rho" in fields else fields["br"]).mesh
        target = spherical_mesh_points(mesh)
        with status("Aligning staggered variable meshes", enabled=show_progress):
            aligned = {name: _onto_mesh(field, mesh, target) for name, field in fields.items()}

        points = spherical_to_cartesian(target)
        b_spherical = np.stack([aligned[name] for name in _B_COMPONENTS], axis=-1)
        b_cartesian = _b_to_cartesian(b_spherical, target)

        variables = {axis: b_cartesian[:, i] for i, axis in enumerate(("Bx", "By", "Bz"))}
        if "rho" in aligned:
            variables = {"rho": aligned["rho"], **variables}

        metadata = SolutionMetadata(
            model=self.model,
            file_format=self.file_format,
            source_path=path,
            normalization="mas-code",
            dimension=3,
            n_equations=len(variables),
            element_type="point",
            extra={
                "files": ",".join(sorted(file.name for file in files.values())),
                "native_mesh": "x".join(str(n) for n in mesh.shape),
                "radial_range": f"{mesh.radii[0]:.3f}-{mesh.radii[-1]:.3f}",
            },
        )

        nodes = points * u.R_sun
        return NativeSolution(
            nodes=nodes,
            connectivity=np.empty((0, 8), dtype=np.int32),
            cell_centers=nodes,
            variables=variables,
            boundaries=_synthesize_boundaries(mesh),
            metadata=metadata,
            structured=mesh,
        )


def _sibling_paths(anchor: Path) -> dict[str, Path]:
    """Map the needed MAS variable names to files next to ``anchor``.

    The variable name is the unique alphabetic token of the anchor's stem matching a MAS
    variable; substituting it names the siblings. ``br, bt, bp`` are required, ``rho`` optional.
    """
    tokens = re.split(r"([a-zA-Z]+)", anchor.stem)
    matches = [i for i, token in enumerate(tokens) if token in _MAS_VARIABLES]
    if len(matches) != 1:
        raise ValueError(
            f"Cannot locate the MAS variable name in {anchor.name!r}: expected exactly one "
            f"name token among {sorted(_MAS_VARIABLES)} (e.g. rho002.hdf, run_br.hdf)."
        )

    def sibling(variable: str) -> Path:
        parts = list(tokens)
        parts[matches[0]] = variable
        return anchor.with_name("".join(parts) + anchor.suffix)

    paths = {variable: sibling(variable) for variable in (*_B_COMPONENTS, "rho")}
    missing = [paths[v].name for v in _B_COMPONENTS if not paths[v].exists()]
    if missing:
        raise ValueError(f"MAS solution incomplete next to {anchor.name!r}: missing {missing}.")
    if not paths["rho"].exists():
        del paths["rho"]
    return paths


def _read_variable(path: Path) -> _MasVariable:
    """Read one MAS file: the 3-D dataset (stored as φ, θ, r; transposed to r, θ, φ) and its
    coordinate axes."""
    try:
        from pyhdf.SD import SD, SDC
    except ImportError as error:
        raise ImportError(PYHDF_MISSING_HINT) from error
    store = SD(str(path), SDC.READ)
    try:
        dataset = store.select(_DATASET_NAME)
        values = np.asarray(dataset.get(), dtype=np.float64)
        azimuths, colatitudes, radii = (
            np.asarray(dataset.dim(i).getscale(), dtype=np.float64) for i in range(3)
        )
    finally:
        store.end()
    mesh = StructuredMesh(radii=radii, colatitudes=colatitudes, azimuths=azimuths)
    return _MasVariable(mesh=mesh, values=values.transpose(2, 1, 0))


def spherical_mesh_points(mesh: StructuredMesh) -> np.ndarray:
    """Return the mesh's points as ``(n, 3)`` spherical ``(r, θ, φ)`` coordinates."""
    r, theta, phi = np.meshgrid(mesh.radii, mesh.colatitudes, mesh.azimuths, indexing="ij")
    return np.column_stack([r.ravel(), theta.ravel(), phi.ravel()])


def _onto_mesh(variable: _MasVariable, mesh: StructuredMesh, target: np.ndarray) -> np.ndarray:
    """Return the variable's flat values on ``mesh`` (its own values if already there, else a
    linear interpolation from its own axes onto ``target``, the mesh's spherical points)."""
    same = variable.mesh.shape == mesh.shape and all(
        np.array_equal(a, b)
        for a, b in (
            (variable.mesh.radii, mesh.radii),
            (variable.mesh.colatitudes, mesh.colatitudes),
            (variable.mesh.azimuths, mesh.azimuths),
        )
    )
    if same:
        return variable.values.ravel()
    return variable.mesh.interpolator(variable.values)(target)


def _b_to_cartesian(b_spherical: np.ndarray, spherical: np.ndarray) -> np.ndarray:
    """Rotate spherical ``(br, bt, bp)`` into Cartesian at the mesh's own ``(θ, φ)`` angles.

    The basis is formed from the file's angle values, not from the Cartesian points: on the
    pole-overhang rows (θ outside ``[0, π]``) the two bases differ by a sign in the θ and φ
    directions, and the file stores its components in the extended-angle basis.
    """
    theta = spherical[:, 1]
    phi = spherical[:, 2]
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    b_r, b_theta, b_phi = b_spherical[:, 0], b_spherical[:, 1], b_spherical[:, 2]
    return np.stack(
        [
            b_r * sin_theta * cos_phi + b_theta * cos_theta * cos_phi - b_phi * sin_phi,
            b_r * sin_theta * sin_phi + b_theta * cos_theta * sin_phi + b_phi * cos_phi,
            b_r * cos_theta - b_theta * sin_theta,
        ],
        axis=-1,
    )


def _synthesize_boundaries(mesh: StructuredMesh) -> dict[str, Boundary]:
    """Inner/outer shells from the radial axis extremes (no tagged surfaces in MAS files; only
    ``mean_radius`` and a descriptive ``source_name`` are consumed downstream)."""

    def shell(role: str, radius: float) -> Boundary:
        return Boundary(
            name=role,
            source_name=f"r={radius:.3f} shell (synthesized)",
            faces=np.empty((0, 4), dtype=np.int32),
            adjacent_cells=np.empty((0,), dtype=np.int32),
            mean_radius=radius * u.R_sun,
        )

    return {
        "inner": shell("inner", float(mesh.radii[0])),
        "outer": shell("outer", float(mesh.radii[-1])),
    }
