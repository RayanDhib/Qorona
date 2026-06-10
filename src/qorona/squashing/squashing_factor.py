"""Assemble Q⊥ (and the classical-Q diagnostic) from transported endpoint deviations.

Given the deviation vectors ``U, V`` transported to a field line's two feet (the deviation
transport), the squashing factor follows from the master determinant formula. Two steps:

1. **Seed basis.** At each seed an orthonormal pair ``(U₀, V₀)`` spanning the plane ⊥ B̂₀ with
   unit area seeds the transport. Q⊥ is invariant to a rotation of this pair within ⊥ B̂₀,
   so the construction matters only for numerical conditioning.
2. **Reproject + assemble.** Transport does not keep ``U, V`` perpendicular to B, so at each foot
   they are reprojected before the master formula. With the perpendicular
   endpoint deviations ``a_F, b_F`` (forward foot) and ``a_B, b_B`` (backward foot),

       Q = (P / B₀²) · [ |a_F|²|b_B|² + |a_B|²|b_F|² - 2 (a_F·b_F)(a_B·b_B) ],

   constant along the line and ≥ 2. The **only** difference between
   Q⊥ and the classical Q is the reprojection normal and the prefactor P:

   - **Q⊥** (primary): reproject with ``n = B̂`` (the orthogonal projection onto ⊥ B̂);
     ``P = |B_F| |B_B|``, never vanishes.
   - **Q** (diagnostic): reproject with ``n = r̂`` (the boundary's outward radial); ``P =
     |B_{F,n} B_{B,n}|`` with ``B_n = B·r̂``, inflating where a line grazes a boundary (``B_n → 0``),
     the projection artifact Q⊥ exists to remove, so Q is secondary.

The prefactor is the flux-conservation (norm-determinant) substitution
``|cross(a_F, b_F)| = B₀/|B_F|`` that replaces the cross-product determinant of the perpendicular
deviations. The shared seed basis fixes ``B₀ⁿ = |B₀|`` for both quantities. This module is pure
array math: the caller (``squashing/__init__.py``) samples B at the seed and the two feet and
passes those values in.

References: the master squashing formula (Eq. 22; first given in this form by Tassev & Savcheva
2017, Eq. 11) and the perpendicular/classical reprojection (Eqs. 50-51) are implemented from
Scott, Pontin & Hornig (2017). Q is the squashing factor of Titov, Hornig & Démoulin (2002);
Q⊥ is its perpendicular variant from Titov (2007).
"""

from __future__ import annotations

import numpy as np


def _dot(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Row-wise dot product over the last axis, for any leading batch shape."""
    return np.einsum("...i,...i->...", u, v, optimize=True)


def _seed_basis(b_seed: np.ndarray, b_magnitude_seed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return an orthonormal deviation basis ``(U₀, V₀)`` ⊥ B̂₀ with unit area at each seed.

    For each seed pick the Cartesian axis ``ê`` least aligned with ``B̂₀`` (guaranteed not
    parallel to it), set ``U₀ = normalize(ê - (ê·B̂₀) B̂₀)`` and ``V₀ = cross(B̂₀, U₀)``. Then
    ``{U₀, V₀, B̂₀}`` is orthonormal and ``|B̂₀·cross(U₀, V₀)| = 1`` (unit area). Q⊥ is invariant
    to any rotation of ``(U₀, V₀)`` within the ⊥ B̂₀ plane, so the most-orthogonal-axis choice
    affects only numerical conditioning, never the value; the same basis serves Q⊥ and Q.

    Parameters
    ----------
    b_seed
        ``(n, 3)`` magnetic field at the seeds.
    b_magnitude_seed
        ``(n,)`` field strength ``|B₀|`` at the seeds.

    Returns
    -------
    tuple of numpy.ndarray
        ``(U₀, V₀)``, each ``(n, 3)``.
    """
    b_hat = b_seed / b_magnitude_seed[:, None]
    # The Cartesian axis least aligned with B̂₀ is the safest reference to project off it.
    reference = np.eye(3)[np.argmin(np.abs(b_hat), axis=1)]
    u0 = reference - _dot(reference, b_hat)[:, None] * b_hat
    u0 /= np.linalg.norm(u0, axis=1, keepdims=True)
    v0 = np.cross(b_hat, u0)
    return u0, v0


def _reproject(deviation: np.ndarray, b: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Reproject a transported deviation onto the plane ⊥ ``normal``.

    Returns ``Ũ = U - (U·n)/(B·n) · B``, the part of ``U`` lying in the plane perpendicular to
    ``n`` reached by sliding along B. For ``n = B̂`` this reduces to the orthogonal projection onto
    ⊥ B̂ (Q⊥); for ``n = r̂`` it is the oblique projection onto the boundary tangent plane (Q).
    Operates on any leading batch shape ``(..., 3)``.
    """
    return deviation - (_dot(deviation, normal) / _dot(b, normal))[..., None] * b


def _master_formula(
    a_f: np.ndarray,
    b_f: np.ndarray,
    a_b: np.ndarray,
    b_b: np.ndarray,
    prefactor: np.ndarray,
    b0_squared: np.ndarray,
) -> np.ndarray:
    """Evaluate ``Q = (P/B₀²)·[|a_F|²|b_B|² + |a_B|²|b_F|² - 2(a_F·b_F)(a_B·b_B)]``.

    ``a_F, b_F`` are the perpendicular deviations at the forward foot (the transported ``U₀, V₀``),
    ``a_B, b_B`` at the backward foot; ``prefactor`` is P and ``b0_squared`` is ``B₀²``. The bracket
    is symmetric under the forward↔backward foot swap, giving the launch↔target invariance of Q.
    """
    bracket = (
        _dot(a_f, a_f) * _dot(b_b, b_b)
        + _dot(a_b, a_b) * _dot(b_f, b_f)
        - 2.0 * _dot(a_f, b_f) * _dot(a_b, b_b)
    )
    return prefactor / b0_squared * bracket


def _assemble_squashing(
    deviations: np.ndarray,
    b_magnitude_seed: np.ndarray,
    feet: np.ndarray,
    b_foot: np.ndarray,
    b_magnitude_foot: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble Q⊥ and the classical-Q diagnostic for each line from the transported deviations.

    Reprojects the endpoint deviations twice, onto ⊥ B̂ for Q⊥ and onto the boundary tangent plane
    (⊥ r̂) for Q, and applies the master formula with the matching prefactor. Both quantities come
    from the one transport; the classical-Q channel adds only the ``r̂`` reprojection and the ``B_n``
    prefactor. Incomplete lines (``~valid``, a ``NaN`` foot) have no squashing factor and are
    written ``NaN``, so ``valid`` is the single source of truth for which rows are meaningful.

    Parameters
    ----------
    deviations
        ``(n, 2, 2, 3)`` deviations at the feet; axis 1 is the ``(backward, forward)`` foot,
        axis 2 is ``(U, V)``.
    b_magnitude_seed
        ``(n,)`` field strength ``|B₀|`` at the seeds.
    feet
        ``(n, 2, 3)`` foot positions, axis 1 ``(backward, forward)`` (for the ``r̂`` of the Q
        channel); ``NaN`` where incomplete.
    b_foot
        ``(n, 2, 3)`` magnetic field at the feet; ``NaN`` where incomplete.
    b_magnitude_foot
        ``(n, 2)`` field strength ``|B|`` at the feet; ``NaN`` where incomplete.
    valid
        ``(n,)`` bool: lines with two real feet (``FieldLines.is_complete``).

    Returns
    -------
    q_perp : numpy.ndarray
        ``(n,)`` perpendicular squashing factor Q⊥; ``NaN`` where ``~valid``.
    q : numpy.ndarray
        ``(n,)`` classical squashing factor Q (diagnostic); ``NaN`` where ``~valid``.
    """
    u_b, v_b = deviations[:, 0, 0], deviations[:, 0, 1]
    u_f, v_f = deviations[:, 1, 0], deviations[:, 1, 1]
    b_b, b_f = b_foot[:, 0], b_foot[:, 1]
    bmag_b, bmag_f = b_magnitude_foot[:, 0], b_magnitude_foot[:, 1]
    b0_squared = b_magnitude_seed**2

    with np.errstate(divide="ignore", invalid="ignore"):
        # Q⊥: reproject onto ⊥ B̂ (n = B̂); prefactor |B_F| |B_B|.
        normal_perp_b = b_b / bmag_b[:, None]
        normal_perp_f = b_f / bmag_f[:, None]
        q_perp = _master_formula(
            _reproject(u_f, b_f, normal_perp_f),
            _reproject(v_f, b_f, normal_perp_f),
            _reproject(u_b, b_b, normal_perp_b),
            _reproject(v_b, b_b, normal_perp_b),
            bmag_f * bmag_b,
            b0_squared,
        )

        # Classical Q (diagnostic): reproject onto the boundary tangent plane (n = r̂); prefactor
        # |B_{F,n} B_{B,n}| with B_n = B·r̂ (inflates as B_n → 0, the artifact Q⊥ removes).
        r_hat_b = feet[:, 0] / np.linalg.norm(feet[:, 0], axis=1, keepdims=True)
        r_hat_f = feet[:, 1] / np.linalg.norm(feet[:, 1], axis=1, keepdims=True)
        q = _master_formula(
            _reproject(u_f, b_f, r_hat_f),
            _reproject(v_f, b_f, r_hat_f),
            _reproject(u_b, b_b, r_hat_b),
            _reproject(v_b, b_b, r_hat_b),
            np.abs(_dot(b_f, r_hat_f) * _dot(b_b, r_hat_b)),
            b0_squared,
        )

    nan = np.full(len(b_magnitude_seed), np.nan)
    return np.where(valid, q_perp, nan), np.where(valid, q, nan)
