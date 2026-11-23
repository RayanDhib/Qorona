"""``PlasmaVolume``: electron density and temperature for the EUV render.

The EUV emission of the plasma goes as ``nₑ² K(T, nₑ)``, so the EUV render needs the absolute
electron density and the temperature. This volume holds ``log₁₀ nₑ`` (cm⁻³) and ``log₁₀ T`` (K)
on its own spherical grid, which can be finer at the base, where the transition region is.

The values are decoded at the native cells with the model's equation of state,
``p = 2 rho k T / (μ m_H)``:

    nₑ = rho / (μ m_p),        T = p / (2 nₑ k),

or ``T`` is taken from the stored temperature when the model writes one (MAS). ``μ`` is the
reader's mean molecular weight, the same value the white-light render uses. The logarithms are
then resampled onto the grid and sampled trilinearly. Logarithms keep the values positive across
the steep density jump of the transition region, where a linear interpolant can overshoot below
zero.

A solution whose base layer is mostly below :data:`TRANSITION_REGION_TEMPERATURE` includes the
transition region: its inner boundary is in the chromosphere. The EUV render corrects that layer
for its artificial broadening at the run's broadening temperature.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from astropy import constants
from astropy import units as u

from qorona.field.density import MEAN_MOLECULAR_WEIGHT
from qorona.io.native import NativeSolution
from qorona.resample.grid import GHOST, SphericalGrid, pad_field
from qorona.resample.resampler import Resampler

__all__ = ["PlasmaDecode", "PlasmaVolume", "decode_plasma"]

#: Electron density (cm⁻³) given to cells whose decoded density is not positive or not finite.
DENSITY_FLOOR = 1.0
#: Temperature (K) given to cells whose decoded temperature is not positive or not finite.
TEMPERATURE_FLOOR = 1.0
#: A solution whose base layer has a median temperature (K) below this value includes the
#: transition region.
TRANSITION_REGION_TEMPERATURE = 1.0e5
#: The base layer: the cells closer to the Sun than this factor times the innermost cell radius.
#: A local pocket of cold cells elsewhere does not make a transition region.
_BASE_LAYER = 1.005

_M_P_KG = constants.m_p.to_value(u.kg)
_K_B = constants.k_B.to_value(u.J / u.K)


@dataclass(frozen=True)
class PlasmaDecode:
    """The decoded plasma at the native cells.

    Attributes
    ----------
    log_ne, log_t
        ``(n_cells,)`` ``log₁₀ nₑ`` (cm⁻³) and ``log₁₀ T`` (K).
    mean_molecular_weight
        The ``μ`` used for the decode.
    temperature_source
        ``"temperature"`` when the model stores ``T``, ``"pressure"`` when ``T`` comes from ``p``.
    floored
        Number of cells whose density or temperature was not positive and finite, and was set to
        the floor value.
    transition_region
        Whether the solution includes the transition region (see
        :data:`TRANSITION_REGION_TEMPERATURE`).
    """

    log_ne: np.ndarray
    log_t: np.ndarray
    mean_molecular_weight: float
    temperature_source: str
    floored: int
    transition_region: bool


def decode_plasma(solution: NativeSolution) -> PlasmaDecode:
    """Decode ``log₁₀ nₑ`` and ``log₁₀ T`` at the native cells of ``solution``.

    Parameters
    ----------
    solution
        A solution whose reader declares ``density_unit`` and either ``temperature_unit`` (with a
        stored ``t``) or ``pressure_unit`` (with a stored ``p``).

    Raises
    ------
    ValueError
        If the solution has no density, no physical density unit, or no way to find the
        temperature.
    """
    metadata = solution.metadata
    model = metadata.model.upper()
    if "rho" not in solution.variables:
        raise ValueError(f"the {model} solution carries no density (rho); EUV needs it")
    if metadata.density_unit is None:
        raise ValueError(
            f"the {model} reader declares no physical density unit, so the electron density "
            "cannot be made absolute; EUV needs it"
        )
    mu = metadata.mean_molecular_weight or MEAN_MOLECULAR_WEIGHT
    rho = np.asarray(solution.variables["rho"], dtype=np.float64)
    density_si = (1.0 * metadata.density_unit).to_value(u.kg / u.m**3)
    electron_density = rho * density_si / (mu * _M_P_KG)  # m⁻³

    if "t" in solution.variables and metadata.temperature_unit is not None:
        temperature = np.asarray(solution.variables["t"], dtype=np.float64) * (
            (1.0 * metadata.temperature_unit).to_value(u.K)
        )
        source = "temperature"
    elif "p" in solution.variables and metadata.pressure_unit is not None:
        pressure = np.asarray(solution.variables["p"], dtype=np.float64) * (
            (1.0 * metadata.pressure_unit).to_value(u.Pa)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            temperature = pressure / (2.0 * electron_density * _K_B)
        source = "pressure"
    else:
        raise ValueError(
            f"the {model} solution has neither a temperature nor a pressure with a known unit; "
            "EUV needs one of them"
        )

    electron_density_cgs = electron_density * 1e-6
    bad = ~(np.isfinite(electron_density_cgs) & (electron_density_cgs > 0.0))
    bad |= ~(np.isfinite(temperature) & (temperature > 0.0))
    electron_density_cgs = np.where(bad, DENSITY_FLOOR, electron_density_cgs)
    temperature = np.where(bad, TEMPERATURE_FLOOR, temperature)
    radius = np.linalg.norm(np.asarray(solution.cell_centers, dtype=np.float64), axis=1)
    base = ~bad & (radius < _BASE_LAYER * radius.min())
    transition_region = bool(
        np.any(base) and np.median(temperature[base]) < TRANSITION_REGION_TEMPERATURE
    )
    return PlasmaDecode(
        log_ne=np.log10(electron_density_cgs),
        log_t=np.log10(temperature),
        mean_molecular_weight=float(mu),
        temperature_source=source,
        floored=int(np.count_nonzero(bad)),
        transition_region=transition_region,
    )


@dataclass(frozen=True)
class PlasmaVolume:
    """``log₁₀ nₑ`` and ``log₁₀ T`` on a spherical grid, sampled trilinearly.

    Attributes
    ----------
    grid
        The grid the values sit on.
    payload
        ``(n_r + 2·GHOST, n_theta + 2·GHOST, n_phi + 2·GHOST, 2)`` ghost-padded values; channel 0
        is ``log₁₀ nₑ`` (cm⁻³), channel 1 is ``log₁₀ T`` (K).
    length_unit
        Physical length of one grid coordinate unit (the model's own length unit).
    mean_molecular_weight
        The ``μ`` of the decode.
    temperature_source
        Where ``T`` came from (see :class:`PlasmaDecode`).
    floored
        Number of native cells set to the floor values during the decode.
    transition_region
        Whether the solution includes the transition region.
    tr_temperature
        The run's transition-region broadening temperature (K), when the solution records it.
    """

    grid: SphericalGrid
    payload: np.ndarray
    length_unit: u.UnitBase
    mean_molecular_weight: float
    temperature_source: str
    floored: int
    transition_region: bool = False
    tr_temperature: float | None = None

    @classmethod
    def from_solution(
        cls,
        solution: NativeSolution,
        grid: SphericalGrid,
        resampler: Resampler,
        *,
        decoded: PlasmaDecode | None = None,
        show_progress: bool = True,
    ) -> PlasmaVolume:
        """Decode ``solution`` at its native cells and resample the logarithms onto ``grid``.

        Parameters
        ----------
        solution
            The native solution (see :func:`decode_plasma` for what it must declare).
        grid
            The target plasma grid.
        resampler
            The resampler that maps native cells onto ``grid``.
        decoded
            The decode of ``solution``, when the caller already has it.
        show_progress
            Whether to show resampling progress.
        """
        if decoded is None:
            decoded = decode_plasma(solution)
        derived = replace(solution, variables={"log_ne": decoded.log_ne, "log_t": decoded.log_t})
        values = resampler.resample(derived, grid, ("log_ne", "log_t"), show_progress=show_progress)
        stacked = np.stack([values["log_ne"], values["log_t"]], axis=-1)
        return cls(
            grid=grid,
            payload=pad_field(stacked),
            length_unit=solution.metadata.length_unit,
            mean_molecular_weight=decoded.mean_molecular_weight,
            temperature_source=decoded.temperature_source,
            floored=decoded.floored,
            transition_region=decoded.transition_region,
            tr_temperature=solution.metadata.tr_temperature,
        )

    def sample(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``log₁₀ nₑ`` and ``log₁₀ T`` at Cartesian ``points`` (R☉), trilinearly.

        Radii outside the grid shell are clamped to the shell; the render keeps its samples
        inside it.

        Parameters
        ----------
        points
            ``(n, 3)`` Cartesian coordinates.

        Returns
        -------
        log_ne, log_t : numpy.ndarray
            ``(n,)`` each.
        """
        index, _ = self.grid.index_coordinates(np.asarray(points, dtype=np.float64))
        index[:, 0] = np.clip(index[:, 0], 0.0, self.grid.n_r - 1.0)
        index += GHOST
        lower = np.floor(index).astype(np.int64)
        weight = index - lower
        i, j, k = lower[:, 0], lower[:, 1], lower[:, 2]
        wr, wt, wp = weight[:, 0:1], weight[:, 1:2], weight[:, 2:3]
        p = self.payload
        result = (
            (1.0 - wr) * (1.0 - wt) * (1.0 - wp) * p[i, j, k]
            + (1.0 - wr) * (1.0 - wt) * wp * p[i, j, k + 1]
            + (1.0 - wr) * wt * (1.0 - wp) * p[i, j + 1, k]
            + (1.0 - wr) * wt * wp * p[i, j + 1, k + 1]
            + wr * (1.0 - wt) * (1.0 - wp) * p[i + 1, j, k]
            + wr * (1.0 - wt) * wp * p[i + 1, j, k + 1]
            + wr * wt * (1.0 - wp) * p[i + 1, j + 1, k]
            + wr * wt * wp * p[i + 1, j + 1, k + 1]
        )
        return result[:, 0], result[:, 1]
