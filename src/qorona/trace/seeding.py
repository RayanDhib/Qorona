"""Seed-point builders for the field-line tracer.

Camera-dependent seeding (the limb ring of the field-line render) lives with the render; this
module holds the camera-free builders, usable from any script that wants raw traced lines.
"""

from __future__ import annotations

import numpy as np

from qorona.geometry.coordinates import spherical_to_cartesian

#: Fractional radial nudge keeping seeds strictly inside the domain shell: a tracer/interpolant
#: precondition; matches ``render.fieldlines._DOMAIN_MARGIN``.
_DOMAIN_MARGIN = 1.0e-9


def lonlat_grid(n_theta: int, n_phi: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the cell-centred ``(theta, phi)`` axes of a longitude/latitude sphere grid, radians.

    Colatitude θ is cell-centred in ``(0, π)`` (no seed on a pole); azimuth φ is cell-centred in
    ``[0, 2π)`` with no duplicate along the seam. Shared by :func:`lonlat_seeds` and the Q-map.
    """
    theta = (np.arange(n_theta) + 0.5) * (np.pi / n_theta)
    phi = (np.arange(n_phi) + 0.5) * (2.0 * np.pi / n_phi)
    return theta, phi


def lonlat_seeds(radius: float, n_theta: int = 100, n_phi: int = 100) -> np.ndarray:
    """Return seeds on a uniform longitude/latitude grid on the sphere of ``radius``.

    The grid is cell-centred in both colatitude θ and azimuth φ, so there is no seed at the poles
    and no duplicate along the φ seam. The radius is nudged a hair inside the domain shell, so
    passing the inner boundary radius (``field.domain.inner_radius``) directly gives seeds the
    tracer accepts.

    Parameters
    ----------
    radius
        Sphere radius in R☉ to seed on, typically the inner boundary.
    n_theta, n_phi
        Grid resolution in colatitude and azimuth.

    Returns
    -------
    numpy.ndarray
        ``(n_theta * n_phi, 3)`` Cartesian seed points in R☉, θ-major.
    """
    theta, phi = lonlat_grid(n_theta, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    r_grid = np.full_like(theta_grid, radius * (1.0 + _DOMAIN_MARGIN))
    return spherical_to_cartesian(
        np.stack([r_grid.ravel(), theta_grid.ravel(), phi_grid.ravel()], axis=-1)
    )


def lonlat_shell(radius: float, n_theta: int, n_phi: int) -> np.ndarray:
    """Return shell sample points on a longitude/latitude grid at exactly ``radius``, θ-major.

    The cell-centred ``(θ, φ)`` grid of :func:`lonlat_seeds` placed on the shell with no
    inward-domain nudge, for sampling the cached volume (the outer boundary included), not for
    seeding the tracer.
    """
    theta, phi = lonlat_grid(n_theta, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    r_grid = np.full_like(theta_grid, radius)
    return spherical_to_cartesian(
        np.stack([r_grid.ravel(), theta_grid.ravel(), phi_grid.ravel()], axis=-1)
    )


def fibonacci_seeds(n_seeds: int, radius: float) -> np.ndarray:
    """Return ``n_seeds`` seeds spread evenly over the sphere of ``radius`` by the golden-angle
    spiral.

    The spiral places points at cell-centred ``cos θ`` (no exact pole) and golden-angle azimuth,
    for even, non-aliased area density. The radius is nudged a hair inside the domain shell, as
    in :func:`lonlat_seeds`.

    Parameters
    ----------
    n_seeds
        Number of seeds.
    radius
        Sphere radius in R☉ to seed on, typically the inner boundary.

    Returns
    -------
    numpy.ndarray
        ``(n_seeds, 3)`` Cartesian seed points in R☉.
    """
    index = np.arange(n_seeds)
    z = 1.0 - (2.0 * index + 1.0) / n_seeds
    theta = np.arccos(z)
    phi = index * (np.pi * (3.0 - np.sqrt(5.0)))
    r = np.full(n_seeds, radius * (1.0 + _DOMAIN_MARGIN))
    return spherical_to_cartesian(np.stack([r, theta, phi], axis=-1))
