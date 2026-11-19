"""Validation study: the absolute white-light brightness against the point-source limit.

Far from the Sun the photosphere acts as a point source of irradiance ``π B☉ (R☉/r)²``, and an
electron scatters its polarized part with the cross-section ``(r_e²/2) sin²χ``, with no limb
darkening or finite-disk coefficients. For a power-law corona ``Nₑ = N₀ r⁻ⁿ`` (``r`` in R☉) seen
along a full line of sight with impact parameter ``rho``, this gives the closed form

    pB / B☉ = (π r_e² / 2) · N₀ R☉ · rho^-(n+1) · B(1/2, (n+3)/2),

with ``B`` the Beta function. The study renders such coronae in absolute units (MSB) through the
full pipeline path (a declared density unit, the stored relative shape, the length unit, the
finite-disk coefficients, the limb-darkening normalisation) and compares them at large impact
parameters. A match for two limb-darkening values also confirms the ``1 - u/3`` normalisation,
which must cancel in this limit.

Report-only: prints the agreement and asserts nothing. Run on demand:

    python validation/absolute_brightness.py
"""

from __future__ import annotations

import numpy as np
from astropy import constants
from astropy import units as u
from scipy.special import beta

from qorona.field.density import MEAN_MOLECULAR_WEIGHT, DensityVolume
from qorona.geometry.camera import OrthographicCamera
from qorona.radiation.brightness import render_brightness
from qorona.resample.grid import LogarithmicSpacing, SphericalGrid

#: Mass-density unit declared for the synthetic corona (an arbitrary physical scale).
_DENSITY_UNIT = u.Unit(1.0e-20 * u.kg / u.m**3)


def point_source() -> None:
    """Rendered MSB pB against the point-source closed form, for two exponents and two ``u``."""
    grid = SphericalGrid(LogarithmicSpacing(1.0, 2000.0), n_r=900, n_theta=6, n_phi=12)
    nodes = grid.node_points().reshape(-1, 3)
    radius = np.linalg.norm(nodes, axis=1).reshape(grid.n_r, grid.n_theta, grid.n_phi)
    # N₀ in m⁻³: one stored mass-density unit over μ proton masses (the default μ).
    n_0 = float((1.0 * _DENSITY_UNIT / (MEAN_MOLECULAR_WEIGHT * constants.m_p)).to_value(u.m**-3))
    r_sun = float(constants.R_sun.to_value(u.m))
    sigma = float(constants.sigma_T.to_value(u.m**2))
    camera = OrthographicCamera.from_sub_observer(
        longitude=0.0, latitude=0.0, fov=160.0 * u.R_sun, pixels=(9, 9)
    )
    print("point-source limit: rendered MSB pB vs (π r_e²/2) N₀ R☉ rho^-(n+1) B(1/2, (n+3)/2)")
    for exponent in (2.0, 3.0):
        density = DensityVolume.from_grid_values(
            grid, radius**-exponent, density_unit=_DENSITY_UNIT
        )
        for limb in (0.0, 0.6):
            result = render_brightness(
                density, camera, u=limb, step=0.05, occult="none", show_progress=False
            )
            far = result.impact > 30.0
            rho = result.impact[far]
            closed = (3.0 * sigma / 16.0) * n_0 * r_sun * rho ** -(exponent + 1.0)
            closed *= beta(0.5, (exponent + 3.0) / 2.0)
            ratio = result.polarized[far] / closed
            print(
                f"    n = {exponent:g}, u = {limb:g} ({result.unit}): rendered / closed form "
                f"{ratio.min():.5f} to {ratio.max():.5f} "
                f"over rho {rho.min():.0f}-{rho.max():.0f} R_sun"
            )


def main() -> None:
    point_source()
    print("report-only: no assertions")


if __name__ == "__main__":
    main()
