"""``DensityVolume``: the electron density on the internal spherical grid, for the Thomson branch.

Electron density is the only MHD-derived input to the white-light / polarized-brightness physics
(the optional Q⊥ Thomson weighting and the standalone brightness render). It is a dense scalar,
defined everywhere the solution is, so unlike the Q⊥ volume it carries no coverage gaps and needs
no NaN-tolerant interpolation: a plain tricubic on a ghost-padded payload, the same
:class:`~qorona.resample.grid.SphericalGrid` machinery the field and Q⊥ volume use.

It is built from the resampler's density output (the canonical ``rho`` variable), so it rides
the resample the field already runs (B and density fit together). The stored values are the
model's mass density divided by the fixed :data:`MEAN_MOLECULAR_WEIGHT`. This relative shape is all
that the Q⊥ weighting and the relative pB need. A constant prefactor cancels in the
weight-normalised average and in the polarization ratio.

When the reader knows the density's physical unit, the volume also records
:attr:`DensityVolume.unit`. This is the electron number density of one stored value,
``Nₑ = rho / (μ m_p)``, with ``μ`` the mean molecular weight of the model's equation of state,
``p = 2 rho k T / (μ m_H)``. The absolute white-light brightness multiplies by it at the end, so the
stored values and every relative product stay the same.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import constants
from astropy import units as u

from qorona.field.interpolation import tricubic
from qorona.resample.grid import GHOST, SphericalGrid, pad_field

#: The default mean molecular weight ``μ``, used when the reader declares none. It is also the
#: fixed divisor of the stored relative shape, a constant that cancels in every relative product.
MEAN_MOLECULAR_WEIGHT = 1.27


@dataclass(frozen=True, slots=True)
class DensityVolume:
    """Scalar electron density ``Nₑ`` on the internal spherical grid (dense; plain tricubic).

    The payload is ghost-padded once at construction so :meth:`sample` is a plain scalar tricubic.
    Values are the relative density shape (see the module header). A constant prefactor does not
    affect the Q⊥ weighting or the polarization ratio.

    Attributes
    ----------
    grid
        The spherical grid the density is stored on (its shell radii set the Thomson coefficient
        table's range).
    density
        ``(n_r + 2·GHOST, n_theta + 2·GHOST, n_phi + 2·GHOST, 1)`` ghost-padded ``Nₑ`` for the
        edge-agnostic tricubic with grid indices offset by :data:`~qorona.resample.grid.GHOST`.
    unit
        The electron number density of one stored value (a scaled ``m⁻³`` unit). ``None`` when the
        physical density is unknown and only the relative shape is available.
    length_unit
        The physical length of one grid coordinate unit (the model's own length unit; R☉ unless the
        reader declares another).
    mean_molecular_weight
        The ``μ`` that :attr:`unit` uses; ``None`` when :attr:`unit` is ``None``.
    """

    grid: SphericalGrid
    density: np.ndarray
    unit: u.UnitBase | None = None
    length_unit: u.UnitBase = u.R_sun
    mean_molecular_weight: float | None = None

    @classmethod
    def from_grid_values(
        cls,
        grid: SphericalGrid,
        mass_density: np.ndarray,
        *,
        mu: float = MEAN_MOLECULAR_WEIGHT,
        density_unit: u.UnitBase | None = None,
        length_unit: u.UnitBase = u.R_sun,
        mean_molecular_weight: float = MEAN_MOLECULAR_WEIGHT,
    ) -> DensityVolume:
        """Build a :class:`DensityVolume` from resampled ``(n_r, n_theta, n_phi)`` mass density.

        Divides the model-normalised mass density by the fixed divisor ``mu`` (the relative shape;
        see the module header) and ghost-pads the result. With a known ``density_unit``
        it also records :attr:`unit`, so one stored value converts to ``Nₑ`` in ``m⁻³``.

        Parameters
        ----------
        grid
            The spherical grid the density was resampled onto.
        mass_density
            ``(n_r, n_theta, n_phi)`` resampled model-normalised mass density on the grid nodes.
        mu
            The fixed divisor of the stored shape (default :data:`MEAN_MOLECULAR_WEIGHT`). It is
            not the model's composition, which is ``mean_molecular_weight``.
        density_unit
            The physical mass density of one ``mass_density`` unit, or ``None`` if unknown.
        length_unit
            The physical length of one grid coordinate unit.
        mean_molecular_weight
            The ``μ`` of the model's equation of state, which converts the physical mass density to
            ``Nₑ = rho / (μ m_p)`` (default :data:`MEAN_MOLECULAR_WEIGHT`).
        """
        shape = np.asarray(mass_density, dtype=np.float64) / mu
        padded = pad_field(shape.reshape(grid.n_r, grid.n_theta, grid.n_phi, 1))
        if density_unit is None:
            return cls(grid=grid, density=padded, length_unit=length_unit)
        per_value = (1.0 * density_unit) * mu / (mean_molecular_weight * constants.m_p)
        return cls(
            grid=grid,
            density=padded,
            unit=u.Unit(per_value.to(u.m**-3)),
            length_unit=length_unit,
            mean_molecular_weight=mean_molecular_weight,
        )

    def sample(self, points: np.ndarray) -> np.ndarray:
        """Return ``(n,)`` interpolated ``Nₑ`` at ``points``; ``0`` outside the shell.

        A plain scalar tricubic on the padded payload (no NaN handling; the field is dense). Points
        outside the radial shell ``[R_inner, R_outer]`` are masked rather than interpolated
        (interpolating there would extrapolate off the ghost padding), so they return ``0``: no
        plasma is sampled there, contributing nothing to a brightness integral or weight.

        Parameters
        ----------
        points
            ``(n, 3)`` Cartesian coordinates in R☉.

        Returns
        -------
        numpy.ndarray
            ``(n,)`` electron density; ``0`` outside the shell.
        """
        points = np.asarray(points, dtype=np.float64)
        radius = np.sqrt(np.sum(points * points, axis=-1))
        inside = (radius >= self.grid.radii[0]) & (radius <= self.grid.radii[-1])

        values = np.zeros(points.shape[0])
        if inside.any():
            index, _ = self.grid.index_coordinates(points[inside])
            interpolated, _ = tricubic(self.density, index + GHOST, gradient=False)
            values[inside] = interpolated[:, 0]
        return values
