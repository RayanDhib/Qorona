"""Validation gate: the analytic PFSS dipole `Field`.

Checks the `AnalyticField`/`PfssDipoleField` engine in isolation (no mesh, no interpolation)
against the closed-form PFSS dipole specification:

- the Cartesian B matches the independent spherical B_r/B_θ formula to machine precision (this
  also exercises the geometry vector-basis rotation);
- the returned Jacobian is the exact derivative of that B (complex-step, machine precision),
  which catches sign and coefficient errors in the hand-written gradient. The dipole is
  curl-free, so its Jacobian is symmetric and this cannot detect a transposed index convention;
  an asymmetric-Jacobian reference field catches that in ``test_dipole_squashing.py``.
- the separatrix colatitude is θ_SL = 50° at the seed radius;
- ``Domain.in_domain`` classifies points below / above / inside the shell.

The Q⊥ profile itself (Q⊥ = 2 in the closed band, divergence at 50°/130°) is exercised by
``test_dipole_squashing.py``.
"""

from __future__ import annotations

import numpy as np

from qorona.field import PfssDipoleField
from qorona.geometry import spherical_to_cartesian, spherical_to_cartesian_vectors


def _dipole_cartesian_b(points: np.ndarray, field: PfssDipoleField) -> np.ndarray:
    """Closed-form Cartesian B from the field's public parameters (for complex-step)."""
    normalization = field.r_sun**3 + 2.0 * field.r_source**3
    moment = field.strength * field.r_sun**3 * field.r_source**3 / normalization
    uniform = field.strength * field.r_sun**3 / normalization
    z_axis = np.array([0.0, 0.0, 1.0])
    z = points[..., 2]
    r2 = np.sum(points * points, axis=-1)
    return (
        moment * ((3.0 * z * r2**-2.5)[..., None] * points - (r2**-1.5)[..., None] * z_axis)
        + uniform * z_axis
    )


def test_dipole_field_matches_analytic() -> None:
    field = PfssDipoleField()
    r_sun, r_source = field.r_sun, field.r_source
    normalization = r_sun**3 + 2.0 * r_source**3

    rng = np.random.default_rng(1)
    n = 500
    radius = rng.uniform(r_sun + 0.01, r_source - 0.01, n)
    colat = rng.uniform(0.02, np.pi - 0.02, n)
    azim = rng.uniform(0.0, 2.0 * np.pi, n)
    points = spherical_to_cartesian(np.stack([radius, colat, azim], axis=-1))

    sample = field.sample(points)

    # B (Cartesian) vs the validation doc's spherical formula, rotated to Cartesian.
    b_r = (r_sun**3 / radius**3) * (2.0 * r_source**3 + radius**3) / normalization * np.cos(colat)
    b_theta = (r_sun**3 / radius**3) * (r_source**3 - radius**3) / normalization * np.sin(colat)
    b_spherical = np.stack([b_r, b_theta, np.zeros_like(b_r)], axis=-1)
    b_expected = spherical_to_cartesian_vectors(b_spherical, points)
    np.testing.assert_allclose(sample.b, b_expected, atol=1e-13)

    # Jacobian is the exact derivative of B (complex-step): catches sign/coefficient errors.
    assert sample.grad_b is not None
    jacobian_cs = np.empty((n, 3, 3))
    for j in range(3):
        shifted = points.astype(complex)
        shifted[:, j] += 1j * 1e-200
        jacobian_cs[:, :, j] = _dipole_cartesian_b(shifted, field).imag / 1e-200
    np.testing.assert_allclose(sample.grad_b, jacobian_cs, atol=1e-12)

    # Separatrix colatitude at the seed radius is 50° (doc value, rounded).
    assert abs(np.degrees(field.separatrix_colatitude(1.01)) - 50.0) < 0.05

    # Domain membership is the caller's boolean predicate (below / above / inside the shell).
    edge = np.array([[0.5, 0.0, 0.0], [3.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    np.testing.assert_array_equal(field.domain.in_domain(edge), [False, False, True])
