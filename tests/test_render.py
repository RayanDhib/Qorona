"""Render confirmation checks: camera orientation, scalar pole padding, and the LOS quadrature.

The primary render validation is visual (rendered dipole / COCONUT images inspected by eye), but
three cheap, targeted checks catch bugs the axisymmetric dipole image hides:

1. **Camera orientation / handedness**: a roll-sign flip or an up-vector handedness flip
   survives an axisymmetric image, so assert the projected solar-north direction directly
   (including a non-zero roll and the left-right handedness).
2. **Pole region**: the scalar Q⊥ volume reuses the field's θ reflect-through-pole padding,
   which is value-exact for a scalar; confirm a smooth field with across-pole azimuthal structure
   interpolates correctly through the pole.
3. **Quadrature**: the weight-normalised line-of-sight integrand must stay correct under partial
   masking, so check it against a closed-form weighted average including a partially-masked ray.
"""

from __future__ import annotations

import numpy as np
from astropy import units as u

from qorona.geometry import OrthographicCamera
from qorona.render.los import _weighted_average
from qorona.resample.grid import LogarithmicSpacing, SphericalGrid, pad_field
from qorona.squashing import QPerpVolume


def test_camera_orientation_and_handedness() -> None:
    # Observer on +x looking at the Sun, solution-north +z up.
    camera = OrthographicCamera(
        look=np.array([1.0, 0.0, 0.0]),
        up=np.array([0.0, 0.0, 1.0]),
        fov=4.0 * u.R_sun,
        pixels=(8, 8),
    )
    rays = camera.rays()
    north = np.array([0.0, 0.0, 1.0])

    # Solar north projects straight up in the image at roll 0.
    assert np.allclose([north @ rays.right, north @ rays.up], [0.0, 1.0], atol=1e-12)
    # Non-mirrored: for an observer on +x looking sunward, world +y is to their right, so it lands
    # on the right half of the image (positive image-right). This is the assertion that fails if the
    # right vector's cross-product order is flipped, mirroring the image left-right.
    assert np.array([0.0, 1.0, 0.0]) @ rays.right > 0.0
    # The image basis is right-handed as the observer sees it (right x up points out of the screen,
    # toward the observer, i.e. along +look), the un-mirrored screen convention.
    assert np.allclose(np.cross(rays.right, rays.up), rays.look, atol=1e-12)
    # The s = 0 origins lie on the plane of sky (perpendicular to the look axis).
    assert np.max(np.abs(rays.origins @ rays.look)) < 1e-12

    # A positive roll rotates solar north counter-clockwise: north -> (-sin a, cos a) in the image.
    angle = np.deg2rad(35.0)
    rolled = OrthographicCamera(
        look=np.array([1.0, 0.0, 0.0]),
        up=np.array([0.0, 0.0, 1.0]),
        roll=angle,
        fov=4.0 * u.R_sun,
        pixels=(8, 8),
    ).rays()
    assert np.allclose(
        [north @ rolled.right, north @ rolled.up], [-np.sin(angle), np.cos(angle)], atol=1e-12
    )


def test_scalar_pole_interpolation() -> None:
    # A smooth Cartesian field with azimuthal structure that survives only if the through-pole
    # padding shifts azimuth by pi correctly (a sign flip or wrong shift corrupts the pole region).
    grid = SphericalGrid(LogarithmicSpacing(1.0, 2.5), n_r=16, n_theta=48, n_phi=96)
    nodes = grid.node_points()
    field = nodes[..., 0]  # f = x = r sin(theta) cos(phi), smooth across the pole
    volume = QPerpVolume(grid=grid, log_q_perp=pad_field(field[..., None]))

    # Sample within ~2 cells of the north pole (where the tricubic stencil reaches the ghost rows).
    theta_step = np.pi / grid.n_theta
    theta = np.array([0.3, 0.8, 1.5, 2.2]) * theta_step
    phi = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    grid_theta, grid_phi = np.meshgrid(theta, phi, indexing="ij")
    radius = 1.8
    points = np.stack(
        [
            radius * np.sin(grid_theta) * np.cos(grid_phi),
            radius * np.sin(grid_theta) * np.sin(grid_phi),
            radius * np.cos(grid_theta),
        ],
        axis=-1,
    ).reshape(-1, 3)

    sampled = volume.sample(points)
    assert np.all(np.isfinite(sampled))
    # Interpolation-level agreement (~1e-3); a broken pole reflection gives an O(1) error here.
    assert np.max(np.abs(sampled - points[:, 0])) < 5e-3


def test_weighted_average_quadrature() -> None:
    n = 64
    s = np.linspace(-1.0, 1.0, n)
    weights = np.exp(-(s**2)) + 0.1  # arbitrary positive weights
    valid = np.ones(n, dtype=bool)
    valid[12:24] = False  # a partially-masked ray

    # A constant integrand returns the constant exactly, regardless of weights or which samples
    # are masked: the property weight-normalisation exists to guarantee.
    average, total = _weighted_average(np.full(n, 3.7), weights, valid)
    assert np.isclose(average, 3.7)
    assert np.isclose(total, weights[valid].sum())

    # A varying integrand matches the closed-form weighted mean over the valid samples only.
    values = 2.0 * s + 0.5
    average, _ = _weighted_average(values, weights, valid)
    expected = np.sum(weights[valid] * values[valid]) / np.sum(weights[valid])
    assert np.isclose(average, expected)

    # No valid sample -> NaN (a background pixel), not a division error.
    empty, _ = _weighted_average(values, weights, np.zeros(n, dtype=bool))
    assert np.isnan(empty)
