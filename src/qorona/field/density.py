"""``DensityVolume``: the electron density on the internal spherical grid, for the Thomson branch.

Electron density is the only MHD-derived input to the white-light / polarized-brightness physics
(the optional Q⊥ Thomson weighting and the standalone brightness render). It is a dense scalar,
defined everywhere the solution is, so unlike the Q⊥ volume it carries no coverage gaps and needs
no NaN-tolerant interpolation: a plain tricubic on a ghost-padded payload, the same
:class:`~qorona.resample.grid.SphericalGrid` machinery the field and Q⊥ volume use.

It is built from the resampler's electron-density output (the COCONUT ``rho`` column), so it rides
the resample the field already runs (one k-d tree, B and density fit together). The COCONUT
corona-normalised mass density is converted toward physical electron number density by the mean
molecular weight ``μ`` (10 % helium ⇒ ``μ = 1.27``); the *absolute* normalisation constant is
deferred: the Q⊥ weighting and relative pB are scale-free (a constant prefactor cancels in the
weight-normalised average and in the polarization ratio), so this volume carries the relative shape
and an absolute factor is folded in only when a calibrated brightness is wanted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qorona.field.interpolation import tricubic
from qorona.resample.grid import GHOST, SphericalGrid, pad_field

#: Mean molecular weight per electron for a 10 % helium corona; the COCONUT corona-normalised mass
#: density is divided by this toward electron number density. A parameter, locked at this default.
MEAN_MOLECULAR_WEIGHT = 1.27


@dataclass(frozen=True, slots=True)
class DensityVolume:
    """Scalar electron density ``Nₑ`` on the internal spherical grid (dense; plain tricubic).

    The payload is ghost-padded once at construction so :meth:`sample` is a plain scalar tricubic.
    Values are the relative density shape (the absolute calibration is deferred; see the module
    header); a constant prefactor does not affect the Q⊥ weighting or the polarization ratio.

    Attributes
    ----------
    grid
        The spherical grid the density is stored on (its shell radii set the Thomson coefficient
        table's range).
    density
        ``(n_r + 2·GHOST, n_theta + 2·GHOST, n_phi + 2·GHOST, 1)`` ghost-padded ``Nₑ`` for the
        edge-agnostic tricubic with grid indices offset by :data:`~qorona.resample.grid.GHOST`.
    """

    grid: SphericalGrid
    density: np.ndarray

    @classmethod
    def from_grid_values(
        cls, grid: SphericalGrid, mass_density: np.ndarray, *, mu: float = MEAN_MOLECULAR_WEIGHT
    ) -> DensityVolume:
        """Build a :class:`DensityVolume` from resampled ``(n_r, n_theta, n_phi)`` mass density.

        Converts the corona-normalised mass density toward electron number density by dividing by
        the mean molecular weight ``mu`` (the absolute factor is deferred; relative shape only) and
        ghost-pads the result.

        Parameters
        ----------
        grid
            The spherical grid the density was resampled onto.
        mass_density
            ``(n_r, n_theta, n_phi)`` resampled corona-normalised mass density on the grid nodes.
        mu
            Mean molecular weight per electron (default :data:`MEAN_MOLECULAR_WEIGHT`).
        """
        electron_density = np.asarray(mass_density, dtype=np.float64) / mu
        padded = pad_field(electron_density.reshape(grid.n_r, grid.n_theta, grid.n_phi, 1))
        return cls(grid=grid, density=padded)

    def sample(self, points: np.ndarray) -> np.ndarray:
        """Return ``(n,)`` interpolated ``Nₑ`` at ``points``; ``0`` outside the shell.

        A plain scalar tricubic on the padded payload (no NaN handling; the field is dense). Points
        outside the radial shell ``[R_inner, R_outer]`` are not interpolated (``index_coordinates``
        extrapolates off the ghost padding past the grid), so they return ``0``: no plasma is
        sampled there, contributing nothing to a brightness integral or weight.

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
