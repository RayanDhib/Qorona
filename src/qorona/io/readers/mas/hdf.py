"""Readers for MAS coronal solutions (one HDF4 or HDF5 file per variable).

MAS solves the global corona on a structured spherical (r, θ, φ) mesh and writes each variable
to its own file (for example ``rho002.hdf`` in HDF4 or ``rho002363.h5`` in HDF5), each holding
one 3-D dataset with the file's own 1-D coordinate axes; the per-variable meshes
are mutually staggered. The two formats store the same layout: the dataset is ordered
``(φ, θ, r)`` and its axes are ``r``, ``θ`` and ``φ``. The reader
accepts any one file of the set, discovers the siblings it needs next to it, interpolates every
variable onto the density mesh, converts the spherical field components to Cartesian there, and
emits a point-value :class:`NativeSolution` that also carries the structured axes for the
structured resampler. MAS: Mikic et al. (1999), Phys. Plasmas 6, 2217; the thermodynamic
corona: Lionello et al. (2009), ApJ 690, 902.

MAS writes ``rho = m_p nₑ`` and ``p = 2 nₑ k T`` (``μ = 1``), and one code density unit is
``nₑ = 10⁸ cm⁻³``. The readers declare these, so the white-light render is absolute. The
temperature ``t`` and pressure ``p`` files are read when present, with their code units, for the
EUV render. When the run's ``omas`` log sits next to the files, the reader also takes the run's
transition-region broadening temperature (``T_CUTOFF1``) from it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import constants
from astropy import units as u

from qorona.console import status
from qorona.geometry import spherical_to_cartesian
from qorona.io.native import NativeSolution, SolutionMetadata, StructuredMesh
from qorona.io.readers.base import SolutionReader, synthesize_shell_boundaries

#: Variable-name tokens a MAS run writes; used to locate the anchor file's token.
_MAS_VARIABLES = ("rho", "br", "bt", "bp", "vr", "vt", "vp", "t", "p")

#: Spherical magnetic-field components the reader requires (converted to Cartesian B).
_B_COMPONENTS = ("br", "bt", "bp")

#: Optional variables read when their files are present.
_OPTIONAL = ("rho", "t", "p")

#: HDF4 dataset name each MAS file stores its variable under.
_DATASET_NAME = "Data-Set-2"

#: HDF5 dataset name and the names of its ``r``, ``θ`` and ``φ`` axes.
_H5_DATASET_NAME = "Data"
_H5_AXES = ("dim1", "dim2", "dim3")

#: The physical mass density of one code density unit: ``nₑ = 10⁸ cm⁻³`` of protons.
_DENSITY_UNIT = u.Unit((1.0e8 * u.cm**-3 * constants.m_p).to(u.kg / u.m**3))

#: The temperature of one code temperature unit.
_TEMPERATURE_UNIT = u.Unit(2.807066716734894e7 * u.K)

#: The pressure of one code pressure unit.
_PRESSURE_UNIT = u.Unit((0.387571701579587 * u.erg / u.cm**3).to(u.Pa))

#: Friendly guidance shown when a MAS file is read without ``pyhdf`` installed.
PYHDF_MISSING_HINT = (
    "reading MAS solutions needs pyhdf; install it with `pip install pyhdf` "
    "or `conda install -c conda-forge pyhdf`"
)

#: Friendly guidance shown when a MAS HDF5 file is read without ``h5py`` installed.
H5PY_MISSING_HINT = (
    "reading MAS HDF5 solutions needs h5py; install it with `pip install h5py` "
    "or `conda install -c conda-forge h5py`"
)
#: The MAS run log, which echoes the run's input parameters.
_RUN_LOG = "omas"
#: The broadening-temperature parameter in the run log (namelist echo or input line).
_TR_TEMPERATURE = re.compile(r"^\s*T_CUTOFF1\s*=\s*([0-9.+\-eEdD]+)", re.IGNORECASE | re.MULTILINE)


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

    @staticmethod
    def _read_variable(path: Path) -> _MasVariable:
        return _read_hdf4_variable(path)

    def __init__(self, variables: tuple[str, ...] | None = None) -> None:
        """Initialise the reader.

        Parameters
        ----------
        variables
            Not supported: the MAS variable set is fixed (``br, bt, bp`` plus the optional
            ``rho``, ``t`` and ``p``) and each file is self-describing, so passing a value raises
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
                fields[name] = self._read_variable(file)

        mesh = (fields["rho"] if "rho" in fields else fields["br"]).mesh
        target = spherical_mesh_points(mesh)
        with status("Aligning staggered variable meshes", enabled=show_progress):
            aligned = {name: _onto_mesh(field, mesh, target) for name, field in fields.items()}

        points = spherical_to_cartesian(target)
        b_spherical = np.stack([aligned[name] for name in _B_COMPONENTS], axis=-1)
        b_cartesian = _b_to_cartesian(b_spherical, target)

        variables = {"rho": aligned["rho"]} if "rho" in aligned else {}
        variables.update({axis: b_cartesian[:, i] for i, axis in enumerate(("Bx", "By", "Bz"))})
        variables.update({name: aligned[name] for name in ("t", "p") if name in aligned})

        metadata = SolutionMetadata(
            model=self.model,
            file_format=self.file_format,
            source_path=path,
            normalization="mas-code",
            density_unit=_DENSITY_UNIT,
            mean_molecular_weight=1.0,
            pressure_unit=_PRESSURE_UNIT,
            temperature_unit=_TEMPERATURE_UNIT,
            tr_temperature=_run_tr_temperature(path.parent),
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
            boundaries=synthesize_shell_boundaries(float(mesh.radii[0]), float(mesh.radii[-1])),
            metadata=metadata,
            structured=mesh,
        )


class MasHdf5Reader(MasHdfReader):
    """Read a MAS one-file-per-variable HDF5 solution set into a :class:`NativeSolution`."""

    file_format = "hdf5"
    extensions = (".h5",)

    @staticmethod
    def _read_variable(path: Path) -> _MasVariable:
        return _read_hdf5_variable(path)


def _run_tr_temperature(directory: Path) -> float | None:
    """The run's transition-region broadening temperature (K) from its ``omas`` log, if present.

    MAS echoes its input parameters into the ``omas`` log, among them ``T_CUTOFF1``, the
    temperature below which the run broadens its transition region.
    """
    log = directory / _RUN_LOG
    if not log.is_file():
        return None
    match = _TR_TEMPERATURE.search(log.read_text(errors="replace"))
    if match is None:
        return None
    value = float(match.group(1).replace("D", "E").replace("d", "e"))
    return value if value > 0.0 else None


def _sibling_paths(anchor: Path) -> dict[str, Path]:
    """Map the needed MAS variable names to files next to ``anchor``.

    The variable name is the unique alphabetic token of the anchor's stem matching a MAS
    variable; substituting it names the siblings. ``br, bt, bp`` are required; ``rho``, ``t``
    and ``p`` are optional.
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

    paths = {variable: sibling(variable) for variable in (*_B_COMPONENTS, *_OPTIONAL)}
    missing = [paths[v].name for v in _B_COMPONENTS if not paths[v].exists()]
    if missing:
        raise ValueError(f"MAS solution incomplete next to {anchor.name!r}: missing {missing}.")
    return {variable: path for variable, path in paths.items() if path.exists()}


def _read_hdf4_variable(path: Path) -> _MasVariable:
    """Read one MAS HDF4 file: the 3-D dataset (stored as φ, θ, r; transposed to r, θ, φ) and its
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


def _read_hdf5_variable(path: Path) -> _MasVariable:
    """Read one MAS HDF5 file: the ``Data`` set (stored as φ, θ, r; transposed to r, θ, φ) and its
    ``dim1``, ``dim2``, ``dim3`` axes (r, θ, φ)."""
    try:
        import h5py
    except ImportError as error:
        raise ImportError(H5PY_MISSING_HINT) from error
    with h5py.File(path, "r") as store:
        values = np.asarray(store[_H5_DATASET_NAME], dtype=np.float64)
        radii, colatitudes, azimuths = (
            np.asarray(store[name], dtype=np.float64) for name in _H5_AXES
        )
    if values.shape != (azimuths.size, colatitudes.size, radii.size):
        raise ValueError(
            f"{path.name}: dataset shape {values.shape} does not match its axes "
            f"(φ {azimuths.size}, θ {colatitudes.size}, r {radii.size})."
        )
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
