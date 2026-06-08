"""Exact foot-landing on the inner/outer spheres, and open/closed classification.

When an accepted step ends outside the domain it has crossed a boundary sphere. The crossing
is localised by root-finding ``|x(θ)|² - R² = 0`` on the step's DOPRI5 **dense-output
interpolant**, a polynomial already built from the step's stages, so the foot is found to
integration tolerance at **zero extra field evaluations** (no switch of the independent
variable to the radial coordinate). Whether each end landed on the inner or the outer sphere
is read straight off the crossing and stored as an :class:`~qorona.trace.fieldline.Endpoint`
code; open vs closed is then *derived* from the two codes by the container.

The root-finder is a vectorized **bisection**: robust for any (possibly non-monotonic)
interpolant, and over one CFL-limited sub-cell step the radius is effectively monotonic so it
converges on the single physical crossing. It is field- and integrator-agnostic: it consumes
only a callable that maps a fractional step position to a point, so it serves both field
implementations and is reused unchanged by the deviation integrator.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from qorona.trace.fieldline import Endpoint


def _classify_crossings(
    end_radius: np.ndarray, inner_radius: float, outer_radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify which accepted step endpoints left the domain, and through which sphere.

    Parameters
    ----------
    end_radius
        ``(m,)`` radius ``|x|`` of each accepted step's endpoint.
    inner_radius, outer_radius
        The domain's bounding sphere radii in R☉.

    Returns
    -------
    crossed : numpy.ndarray
        ``(m,)`` bool, ``True`` where the endpoint left the shell.
    target_radius : numpy.ndarray
        ``(m,)`` the radius of the sphere crossed (meaningful only where ``crossed``).
    code : numpy.ndarray
        ``(m,)`` ``int8`` :class:`Endpoint` code, ``OUTER`` past the outer sphere else ``INNER``
        (meaningful only where ``crossed``). A step cannot cross both spheres at once because the
        CFL ceiling keeps its radial advance below one cell.
    """
    # Inclusive bounds match Domain.in_domain: an endpoint landing exactly on a sphere counts as
    # a crossing (the foot root-find then resolves to the step end), never a point to sample past.
    crossed_outer = end_radius >= outer_radius
    crossed = crossed_outer | (end_radius <= inner_radius)
    target_radius = np.where(crossed_outer, outer_radius, inner_radius)
    code = np.where(crossed_outer, Endpoint.OUTER, Endpoint.INNER).astype(np.int8)
    return crossed, target_radius, code


def _localize_foot(
    interpolant: Callable[[np.ndarray], np.ndarray],
    target_radius: np.ndarray,
    *,
    max_iter: int = 80,
    tol: float = 1e-14,
) -> np.ndarray:
    """Return the fractional step position ``θ* ∈ [0, 1]`` where ``|x(θ)| = target_radius``.

    Vectorized bisection of ``g(θ) = |x(θ)|² - R²`` over each crossing lane. The bracket is the
    whole step ``[0, 1]``: the start is in the domain and the end is outside it, so ``g`` changes
    sign across it and the bracket is valid whichever sphere was crossed. The arc-length position
    of the foot is ``s + θ*·h``; the caller evaluates the interpolant at ``θ*`` for the full
    landing state.

    Parameters
    ----------
    interpolant
        Maps fractional step positions ``θ`` ``(m,)`` to interpolated points ``(m, 3)``: the
        step's dense-output evaluator, restricted to the crossing lanes.
    target_radius
        ``(m,)`` radius of the sphere each lane crosses.
    max_iter
        Maximum bisection iterations (the bracket halves each step; the default resolves ``θ``
        to machine precision on ``[0, 1]``).
    tol
        Early-stop bracket width; iteration ends once every lane's bracket is narrower.

    Returns
    -------
    numpy.ndarray
        ``(m,)`` fractional step position of the foot.
    """
    target_squared = target_radius * target_radius

    def gap(theta: np.ndarray) -> np.ndarray:
        point = interpolant(theta)
        return np.sum(point * point, axis=-1) - target_squared

    low = np.zeros_like(target_radius)
    high = np.ones_like(target_radius)
    sign_low = np.sign(gap(low))
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        # Keep the half-bracket that still straddles the root (where g(mid) shares g(low)'s sign,
        # the root lies in [mid, high]; otherwise in [low, mid]). Robust to non-monotonic g.
        same_side = np.sign(gap(mid)) == sign_low
        low = np.where(same_side, mid, low)
        high = np.where(same_side, high, mid)
        if np.all(high - low < tol):
            break
    return 0.5 * (low + high)
