"""Validation gate (real-data path): the ``SampledField`` interpolation engine.

Complements ``test_dipole_field.py`` (which checks the analytic engine with no mesh). Here
a known field is placed directly on the internal spherical grid and queried through the full
interpolation path (ghost padding, the Keys tricubic, and the chain rule that rotates the
index-space gradient to the Cartesian Jacobian), so the load-bearing machinery the tracer and
Q⊥ transport depend on is exercised end to end. A field that is **linear in Cartesian space**
is smooth through the poles and has an exact, constant Jacobian, giving analytic ground truth;
it is placed on the nodes directly, bypassing the resampler, so the test isolates interpolation
accuracy alone.

Checks:

- the interpolated B reproduces the linear field within interpolation tolerance;
- the returned Jacobian equals a central finite difference of the interpolated field: the
  decisive check that the index→spherical→Cartesian chain rule (with no sinθ floor) is correct;
- that Jacobian also matches the analytic constant gradient within interpolation tolerance;
- a near-pole sample stays finite and accurate, confirming the reflect-through-pole ghost;
- the Cartesian↔spherical round trip is exact.
"""

from __future__ import annotations

import numpy as np

from qorona.field import SampledField
from qorona.geometry import cartesian_to_spherical, spherical_to_cartesian
from qorona.resample import LogarithmicSpacing, SphericalGrid
from qorona.resample.grid import pad_field

# A linear Cartesian field B(x) = M x + c: smooth through the poles, with exact Jacobian M.
_GRADIENT = np.array([[0.3, -0.7, 0.2], [0.5, 0.1, -0.4], [-0.2, 0.6, 0.8]])
_OFFSET = np.array([1.0, -2.0, 0.5])


def _linear_field(points: np.ndarray) -> np.ndarray:
    """Evaluate the analytic linear field ``B = M x + c`` at ``points``."""
    return points @ _GRADIENT.T + _OFFSET


def test_sampled_field_reproduces_linear_field_and_gradient() -> None:
    grid = SphericalGrid(LogarithmicSpacing(1.0, 2.5), n_r=64, n_theta=64, n_phi=128)
    field = SampledField(grid, pad_field(_linear_field(grid.node_points())), normalization="test")

    rng = np.random.default_rng(0)
    n = 2000
    radius = rng.uniform(1.2, 2.3, n)
    colat = rng.uniform(0.35, np.pi - 0.35, n)
    azim = rng.uniform(0.0, 2.0 * np.pi, n)
    points = spherical_to_cartesian(np.stack([radius, colat, azim], axis=-1))

    sample = field.sample(points)

    # Interpolated B reproduces the field (interpolation-limited).
    np.testing.assert_allclose(sample.b, _linear_field(points), atol=5e-5)

    # The Jacobian is the exact derivative of the interpolated field: compare against a central
    # finite difference of sample.b. This is the decisive chain-rule check (a transpose or a
    # missing dr/dξ / 1/(r sinθ) factor would show up here), independent of interpolation error.
    assert sample.grad_b is not None
    step = 1e-6
    finite_difference = np.empty((n, 3, 3))
    for axis in range(3):
        shift = np.zeros(3)
        shift[axis] = step
        forward = field.sample(points + shift, gradient=False).b
        backward = field.sample(points - shift, gradient=False).b
        finite_difference[:, :, axis] = (forward - backward) / (2.0 * step)
    np.testing.assert_allclose(sample.grad_b, finite_difference, atol=1e-6)

    # That Jacobian also matches the analytic constant gradient M (interpolation-limited).
    np.testing.assert_allclose(sample.grad_b, np.broadcast_to(_GRADIENT, (n, 3, 3)), atol=3e-3)

    # Near the pole the reflect-through-pole ghost keeps a smooth field finite and accurate.
    pole_colat = rng.uniform(0.01, 0.08, 200)
    pole_points = spherical_to_cartesian(
        np.stack([rng.uniform(1.3, 2.2, 200), pole_colat, rng.uniform(0.0, 2.0 * np.pi, 200)], -1)
    )
    pole_sample = field.sample(pole_points, gradient=False)
    assert np.isfinite(pole_sample.b).all()
    np.testing.assert_allclose(pole_sample.b, _linear_field(pole_points), atol=1e-4)

    # Round-trip Cartesian↔spherical is exact.
    round_trip = spherical_to_cartesian(cartesian_to_spherical(points))
    np.testing.assert_allclose(round_trip, points, atol=1e-12)
