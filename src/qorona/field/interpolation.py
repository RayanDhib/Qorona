"""Tricubic interpolation with continuous analytic gradients (Keys 1981).

The deviation-transport RHS that drives the squashing-factor computation needs a smooth,
differentiable field: trilinear interpolation has a piecewise-constant gradient that is the
dominant accuracy limit, so Qorona interpolates with the cubic-convolution kernel of Keys
(1981), which is C¹ (continuous value *and* gradient) and third-order accurate. The kernel
parameter ``a = -1/2`` is Keys' choice giving that third-order accuracy (the Catmull-Rom
spline).

This module is the pure interpolation core: it operates on a uniform rectilinear array in
index coordinates and returns the value and its gradient *with respect to those index
coordinates*. It carries no knowledge of the spherical grid: periodicity, the radial
spacing law, and pole handling are applied by the caller as ghost-cell padding (so the
stencil here is always in range) and by chain-ruling the returned index-space gradient into
physical/Cartesian coordinates.

Reference: Keys, R. G. 1981, "Cubic Convolution Interpolation for Digital Image Processing,"
IEEE Trans. Acoust. Speech Signal Process., 29, 1153.
"""

from __future__ import annotations

import numpy as np

#: Stencil offsets of the four samples a Keys cubic uses around a position, in order.
_STENCIL = np.array([-1, 0, 1, 2])

#: Smallest absolute kept-weight a NaN-tolerant sample divides by. The Keys lobes are signed, so
#: dropping taps can leave a kept weight that is near zero or even negative (a surviving negative
#: lobe dominates); dividing by it would inflate the few finite taps into a meaningless value, so
#: the sample is returned as ``NaN`` instead.
_MIN_KEPT_WEIGHT = 1.0e-6


def _keys_weights(t: np.ndarray) -> np.ndarray:
    """Return the four Keys cubic-convolution weights at fractional offsets ``t``.

    Parameters
    ----------
    t
        ``(n,)`` fractional positions in ``[0, 1)`` past the lower-left stencil sample.

    Returns
    -------
    numpy.ndarray
        ``(n, 4)`` weights for the samples at offsets ``(-1, 0, 1, 2)`` (the ``a = -1/2``
        cubic-convolution kernel; the rows sum to one).
    """
    t2 = t * t
    t3 = t2 * t
    return np.stack(
        [
            -0.5 * t3 + t2 - 0.5 * t,
            1.5 * t3 - 2.5 * t2 + 1.0,
            -1.5 * t3 + 2.0 * t2 + 0.5 * t,
            0.5 * t3 - 0.5 * t2,
        ],
        axis=-1,
    )


def _keys_weight_derivatives(t: np.ndarray) -> np.ndarray:
    """Return the derivatives of the four Keys weights with respect to ``t``.

    Parameters
    ----------
    t
        ``(n,)`` fractional positions in ``[0, 1)``.

    Returns
    -------
    numpy.ndarray
        ``(n, 4)`` weight derivatives ``dW/dt`` for the samples at offsets ``(-1, 0, 1, 2)``
        (the rows sum to zero).
    """
    t2 = t * t
    return np.stack(
        [
            -1.5 * t2 + 2.0 * t - 0.5,
            4.5 * t2 - 5.0 * t,
            -4.5 * t2 + 4.0 * t + 0.5,
            1.5 * t2 - 1.0 * t,
        ],
        axis=-1,
    )


def tricubic(
    values: np.ndarray, coords: np.ndarray, *, gradient: bool = True, skip_nan: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    """Interpolate a uniform vector field and (optionally) its index-space gradient.

    The four-sample Keys stencil spans offsets ``-1 .. +2`` about ``floor(coord)`` on each
    axis, so every coordinate must satisfy ``1 ≤ floor(coord) ≤ N-3`` for that axis of size
    ``N``. Callers guarantee this by padding ``values`` with at least two ghost layers per
    side (periodic, extrapolated, or pole-reflected as appropriate) and offsetting ``coords``
    into the padded array.

    Parameters
    ----------
    values
        ``(N0, N1, N2, C)`` samples of a ``C``-component field on a uniform grid, in index
        coordinates (sample ``[i, j, k]`` sits at integer position ``(i, j, k)``).
    coords
        ``(n, 3)`` query positions in the same index coordinates.
    gradient
        Whether to also compute the gradient. When ``False`` the derivative weights and their
        three contractions are skipped and ``gradient`` is returned as ``None``.
    skip_nan
        NaN-tolerant value path, requiring ``gradient=False``. When ``False`` (the default) a
        single non-finite tap propagates ``NaN`` through the whole 4x4x4 stencil. When ``True`` a
        non-finite tap is instead dropped from both the weighted sum and the weight total, so the
        sample is ``Σ(w·v) / Σ(w)`` over the finite taps alone and is ``NaN`` only where every tap
        is non-finite or the kept weight cancels below :data:`_MIN_KEPT_WEIGHT` (signed Keys lobes).

    Returns
    -------
    value : numpy.ndarray
        ``(n, C)`` interpolated field.
    gradient : numpy.ndarray or None
        ``(n, 3, C)`` gradient ``∂value/∂coord`` with respect to the three index coordinates
        (``gradient[:, d, :]`` is the derivative along axis ``d``), or ``None`` if not requested.
    """
    if skip_nan and gradient:
        raise ValueError("skip_nan is a value-only path; call it with gradient=False")

    coords = np.asarray(coords, dtype=np.float64)
    base = np.floor(coords).astype(np.intp)
    frac = coords - base

    # Per-axis stencil indices (n, 4) and Keys weights.
    index = [base[:, d, None] + _STENCIL for d in range(3)]
    wx, wy, wz = (_keys_weights(frac[:, d]) for d in range(3))

    # Gather the 4x4x4 neighbourhood for every query point: (n, 4, 4, 4, C).
    neighbourhood = values[
        index[0][:, :, None, None],
        index[1][:, None, :, None],
        index[2][:, None, None, :],
    ]

    if skip_nan:
        # Drop any tap that is non-finite in any component from both the weighted sum and the
        # weight total, so the value renormalises over the finite taps alone (Σ w·v / Σ w). The
        # kept weight is one scalar per query; with no tap dropped it is the full stencil weight
        # (≡ 1) and this reduces to the plain tricubic below.
        finite = np.isfinite(neighbourhood).all(axis=-1)
        kept = np.where(finite[..., None], neighbourhood, 0.0)
        numerator = np.einsum("na,nb,nc,nabcC->nC", wx, wy, wz, kept, optimize=True)
        weight = np.einsum("na,nb,nc,nabc->n", wx, wy, wz, finite, optimize=True)[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(np.abs(weight) > _MIN_KEPT_WEIGHT, numerator / weight, np.nan), None

    value = np.einsum("na,nb,nc,nabcC->nC", wx, wy, wz, neighbourhood, optimize=True)
    if not gradient:
        return value, None

    dwx, dwy, dwz = (_keys_weight_derivatives(frac[:, d]) for d in range(3))
    grad = np.stack(
        [
            np.einsum("na,nb,nc,nabcC->nC", dwx, wy, wz, neighbourhood, optimize=True),
            np.einsum("na,nb,nc,nabcC->nC", wx, dwy, wz, neighbourhood, optimize=True),
            np.einsum("na,nb,nc,nabcC->nC", wx, wy, dwz, neighbourhood, optimize=True),
        ],
        axis=1,
    )
    return value, grad
