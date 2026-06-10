"""Validation gate: deviation-transport Q⊥ on the analytic PFSS dipole.

The project's make-or-break check against the closed-form PFSS dipole curve: run
:func:`~qorona.squashing.compute_squashing` end-to-end on the dipole and reproduce the analytic
Q⊥ (flat ``2`` across the closed band, the analytic curve over the polar caps, the divergence
localised at the separatrix, and the ``Q ≥ 2`` floor) at ``rtol`` 1e-4 *and* 1e-5, confirming the
value is converged rather than coincidental. Plus the ∇B̂-contraction guard: the symmetric dipole
cannot expose a transpose in the deviation derivative, so a hand-built **asymmetric** ``grad_b``
checks the contraction convention directly.

The squashing-factor regression gate; the dense theory-vs-code figure lives in the separate
study ``validation/dipole_q_perp.py``.
"""

from __future__ import annotations

import numpy as np

from qorona.field import PfssDipoleField
from qorona.geometry import spherical_to_cartesian
from qorona.squashing import compute_squashing
from qorona.squashing.transport import _deviation_derivative, _unit_field_gradient

#: Seed radius where θ_SL = 50.0°; seeds sit just above the inner boundary so both half-lines
#: are non-trivial.
R_SEED = 1.01


def _seeds(colatitudes_deg: np.ndarray, azimuth: float = 0.7) -> np.ndarray:
    """Return seeds at ``R_seed`` for the given colatitudes (degrees), at a fixed azimuth."""
    colatitude = np.deg2rad(np.asarray(colatitudes_deg, dtype=np.float64))
    spherical = np.stack(
        [np.full_like(colatitude, R_SEED), colatitude, np.full_like(colatitude, azimuth)], axis=-1
    )
    return spherical_to_cartesian(spherical)


def test_dipole_squashing_profile() -> None:
    field = PfssDipoleField()
    closed_band_deg = np.array([55.0, 70.0, 90.0, 110.0, 125.0])
    cap_deg = np.array([15.0, 30.0, 45.0, 49.0, 135.0, 150.0, 165.0])

    for rtol in (1e-4, 1e-5):
        # Closed band: Q⊥ = 2, asserted at 1e-6 (the engine typically delivers ~1e-9).
        band = compute_squashing(field, _seeds(closed_band_deg), rtol=rtol, show_progress=False)
        assert band.lines.is_closed.all()
        assert np.max(np.abs(band.q_perp - 2.0)) < 1e-6

        # Polar caps: Q⊥ follows the analytic boundary-to-boundary curve (both hemispheres).
        caps = compute_squashing(field, _seeds(cap_deg), rtol=rtol, show_progress=False)
        assert caps.lines.is_open.all()
        analytic = field.q_perp_analytic(np.deg2rad(cap_deg), R_SEED)
        assert np.max(np.abs(caps.q_perp / analytic - 1.0)) < 1e-4

        # Floor: Q⊥ ≥ 2 everywhere it is defined.
        for result in (band, caps):
            assert np.all(result.q_perp[result.valid] >= 2.0 - 1e-9)

        # Separatrix divergence localises at θ_SL = 50° / 130°: just inside the cap Q⊥ climbs well
        # above the floor, while the adjacent closed band stays at 2 (seeds bracket the flip with a
        # margin; those on the separatrix run toward the cusp null and are not classified).
        spike = compute_squashing(field, _seeds([49.5, 130.5]), rtol=rtol, show_progress=False)
        assert spike.lines.is_open.all()
        assert np.all(spike.q_perp > 5.0)

    # Launch↔target invariance: the master formula is symmetric in the two feet, so reversing the
    # seed azimuth (a mirror that swaps the two half-lines' roles) leaves Q⊥ unchanged.
    forward = compute_squashing(field, _seeds(cap_deg, azimuth=0.7), rtol=1e-5, show_progress=False)
    mirror = compute_squashing(
        field, _seeds(cap_deg, azimuth=0.7 + np.pi), rtol=1e-5, show_progress=False
    )
    assert np.allclose(forward.q_perp, mirror.q_perp, rtol=1e-6, atol=0.0)


def test_unit_field_gradient_contraction() -> None:
    # A hand-built ASYMMETRIC raw Jacobian: a transpose in the deviation derivative would change the
    # result, which the symmetric dipole Jacobian cannot expose.
    grad_b = np.array([[[2.0, 1.0, 0.0], [-3.0, 0.5, 4.0], [1.0, -2.0, 1.5]]])
    b = np.array([[0.3, -0.4, 1.2]])
    b_magnitude = np.linalg.norm(b, axis=1)
    deviation = np.array([[1.0, 2.0, -1.0]])

    # Closed-form unit-field Jacobian ∇B̂ = (I - B̂B̂ᵀ)·grad_b/|B| and directional derivative ∇B̂·U.
    b_hat = (b / b_magnitude[:, None])[0]
    grad_b_hat_expected = (np.eye(3) - np.outer(b_hat, b_hat)) @ grad_b[0] / b_magnitude[0]
    expected = grad_b_hat_expected @ deviation[0]

    got = _deviation_derivative(b, b_magnitude, grad_b, deviation[:, None, :])[0, 0]
    assert np.allclose(got, expected, rtol=0.0, atol=1e-12)
    assert np.allclose(
        _unit_field_gradient(b, b_magnitude, grad_b)[0], grad_b_hat_expected, atol=1e-12
    )

    # Asymmetry is real: the transpose contraction differs, so the guard actually bites.
    assert not np.allclose(got, grad_b_hat_expected.T @ deviation[0], atol=1e-6)
