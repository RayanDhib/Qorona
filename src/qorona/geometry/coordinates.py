"""Cartesian ↔ spherical coordinate transforms for points and vector components.

Positions throughout Qorona are Cartesian ``(x, y, z)`` in solar radii; the spherical
coordinate is ``(r, θ, φ)`` with ``θ`` the colatitude measured from the north pole
(``+z``), ``θ ∈ [0, π]``, and ``φ`` the azimuth about ``+z``, ``φ ∈ [0, 2π)`` (physics /
ISO convention). The spherical form is used only for internal grid indexing and I/O; the
field and tracer operate in Cartesian, which is smooth through the poles.

These are the plain-array workhorses (positions in R☉, angles in radians); attaching or
stripping :mod:`astropy.units` is done by callers at the I/O edge. All functions accept any
leading batch shape ``(..., 3)``.
"""

from __future__ import annotations

import numpy as np


def cartesian_to_spherical(points: np.ndarray) -> np.ndarray:
    """Convert Cartesian points to spherical ``(r, θ, φ)``.

    Parameters
    ----------
    points
        ``(..., 3)`` Cartesian coordinates ``(x, y, z)`` in R☉.

    Returns
    -------
    numpy.ndarray
        ``(..., 3)`` spherical coordinates ``(r [R☉], θ [rad], φ [rad])`` with
        ``θ ∈ [0, π]`` the colatitude and ``φ ∈ [0, 2π)`` the azimuth.
    """
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    r = np.sqrt(x * x + y * y + z * z)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_theta = np.where(r > 0.0, z / r, 1.0)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    phi = np.arctan2(y, x) % (2.0 * np.pi)
    return np.stack([r, theta, phi], axis=-1)


def spherical_to_cartesian(coords: np.ndarray) -> np.ndarray:
    """Convert spherical ``(r, θ, φ)`` to Cartesian points.

    Parameters
    ----------
    coords
        ``(..., 3)`` spherical coordinates ``(r [R☉], θ [rad], φ [rad])`` with ``θ`` the
        colatitude from ``+z``.

    Returns
    -------
    numpy.ndarray
        ``(..., 3)`` Cartesian coordinates ``(x, y, z)`` in R☉.
    """
    r = coords[..., 0]
    theta = coords[..., 1]
    phi = coords[..., 2]
    sin_theta = np.sin(theta)
    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=-1)


def _spherical_basis_trig(points: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return ``(sinθ, cosθ, sinφ, cosφ)`` of the spherical basis at Cartesian ``points``.

    On the polar axis the azimuth is undefined; ``φ`` is taken as zero there
    (``cosφ = 1``, ``sinφ = 0``), a harmless choice for the axisymmetric validation
    fields and for any vector with no azimuthal component on the axis.
    """
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    cylindrical = np.sqrt(x * x + y * y)
    r = np.sqrt(cylindrical * cylindrical + z * z)
    on_axis = cylindrical == 0.0
    safe_r = np.where(r == 0.0, 1.0, r)
    safe_cylindrical = np.where(on_axis, 1.0, cylindrical)
    sin_theta = cylindrical / safe_r
    cos_theta = np.where(r == 0.0, 1.0, z / safe_r)
    cos_phi = np.where(on_axis, 1.0, x / safe_cylindrical)
    sin_phi = np.where(on_axis, 0.0, y / safe_cylindrical)
    return sin_theta, cos_theta, sin_phi, cos_phi


def spherical_coordinate_jacobian(points: np.ndarray) -> np.ndarray:
    """Return the Jacobian ``∂(r, θ, φ)/∂(x, y, z)`` of the spherical map at ``points``.

    This is the metric factor used to chain-rule a gradient taken in spherical (or grid-index)
    coordinates back to Cartesian: ``∂f/∂x_j = Σ_d (∂f/∂s_d)(∂s_d/∂x_j)`` with ``s = (r, θ, φ)``.
    The ``θ`` and ``φ`` rows carry the cylindrical ``1/d`` and ``1/d²`` factors
    (``d = sqrt(x²+y²)``) that diverge on the polar axis (the artificial coordinate
    singularity); this returns the exact factors (no regularization), as the field/gradient
    layer requires.

    Parameters
    ----------
    points
        ``(..., 3)`` Cartesian coordinates in R☉.

    Returns
    -------
    numpy.ndarray
        ``(..., 3, 3)`` Jacobian with ``[..., d, j] = ∂s_d/∂x_j`` for ``s = (r, θ, φ)``.
    """
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    cylindrical2 = x * x + y * y
    cylindrical = np.sqrt(cylindrical2)
    r2 = cylindrical2 + z * z
    r = np.sqrt(r2)

    d_r = np.stack([x / r, y / r, z / r], axis=-1)
    d_theta = np.stack(
        [x * z / (r2 * cylindrical), y * z / (r2 * cylindrical), -cylindrical / r2], axis=-1
    )
    d_phi = np.stack([-y / cylindrical2, x / cylindrical2, np.zeros_like(x)], axis=-1)
    return np.stack([d_r, d_theta, d_phi], axis=-2)


def cartesian_to_spherical_vectors(vectors: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Rotate Cartesian vector components into the spherical basis at ``points``.

    Parameters
    ----------
    vectors
        ``(..., 3)`` vector Cartesian components ``(V_x, V_y, V_z)``.
    points
        ``(..., 3)`` Cartesian locations at which the vectors are evaluated.

    Returns
    -------
    numpy.ndarray
        ``(..., 3)`` spherical components ``(V_r, V_θ, V_φ)`` at each point.
    """
    sin_theta, cos_theta, sin_phi, cos_phi = _spherical_basis_trig(points)
    vx = vectors[..., 0]
    vy = vectors[..., 1]
    vz = vectors[..., 2]
    v_r = vx * sin_theta * cos_phi + vy * sin_theta * sin_phi + vz * cos_theta
    v_theta = vx * cos_theta * cos_phi + vy * cos_theta * sin_phi - vz * sin_theta
    v_phi = -vx * sin_phi + vy * cos_phi
    return np.stack([v_r, v_theta, v_phi], axis=-1)


def spherical_to_cartesian_vectors(vectors: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Rotate spherical vector components into the Cartesian basis at ``points``.

    Parameters
    ----------
    vectors
        ``(..., 3)`` spherical components ``(V_r, V_θ, V_φ)``.
    points
        ``(..., 3)`` Cartesian locations at which the vectors are evaluated.

    Returns
    -------
    numpy.ndarray
        ``(..., 3)`` Cartesian components ``(V_x, V_y, V_z)`` at each point.
    """
    sin_theta, cos_theta, sin_phi, cos_phi = _spherical_basis_trig(points)
    v_r = vectors[..., 0]
    v_theta = vectors[..., 1]
    v_phi = vectors[..., 2]
    vx = v_r * sin_theta * cos_phi + v_theta * cos_theta * cos_phi - v_phi * sin_phi
    vy = v_r * sin_theta * sin_phi + v_theta * cos_theta * sin_phi + v_phi * cos_phi
    vz = v_r * cos_theta - v_theta * sin_theta
    return np.stack([vx, vy, vz], axis=-1)
