"""Analytic gate for the Thomson-brightness engine: a power-law corona ``Nₑ ∝ r^(-gamma)`` has
the asymptotic polarization ``P = pB / K_tot = (gamma+1)/(gamma+3)`` far from the Sun (Inhester
2015, Eq. 4.7). Exercises the coefficient table and the line-of-sight quadrature end to end with
no MHD input, on whichever path (kernel or NumPy) is installed; both are gated by the same
analytic value.
"""

from __future__ import annotations

import numpy as np
from astropy.units import R_sun

from qorona.field.density import DensityVolume
from qorona.geometry.camera import OrthographicCamera
from qorona.radiation.brightness import render_brightness
from qorona.resample.grid import LogarithmicSpacing, SphericalGrid


def test_power_law_polarization_matches_analytic() -> None:
    gamma = 6.0
    grid = SphericalGrid(
        spacing=LogarithmicSpacing(inner=1.0, outer=30.0), n_r=192, n_theta=24, n_phi=48
    )
    values = np.broadcast_to(
        (grid.radii**-gamma)[:, None, None], (grid.n_r, grid.n_theta, grid.n_phi)
    )
    density = DensityVolume.from_grid_values(grid, np.ascontiguousarray(values), mu=1.0)
    # A single equatorial pixel row: rho = |x|, the polarization depends on rho alone. u = 0 (the
    # analytic result is for the bare coefficients) and a wide shell so the truncated tail beyond
    # r_outer is negligible against the steep power law.
    camera = OrthographicCamera.from_sub_observer(
        longitude=0.0, latitude=0.0, fov=24.0 * R_sun, pixels=(1, 32)
    )
    result = render_brightness(
        density, camera, u=0.0, step=0.01, occult="none", show_progress=False
    )
    rho = np.abs(result.x_rsun)
    polarization = result.polarization()[0]
    window = (rho > 6.0) & (rho < 10.0)
    expected = (gamma + 1.0) / (gamma + 3.0)
    assert window.any()
    assert np.all(np.abs(polarization[window] - expected) < 0.02)
