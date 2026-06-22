"""Scalar-per-lane nopython mirror of the field-line transport hot loop (numba ``prange``).

This module is the deliberately **layer-flattened** JIT image of the reference math: it inlines the
Keys tricubic (``field/interpolation.py``), the spherical grid index map and its spacing laws
(``resample/grid.py``), the spherical-coordinate Jacobian (``geometry/coordinates.py``), the field
evaluation (``field/sampled.py`` and the analytic dipole of ``field/analytic.py``), the DOPRI5
stepper with PI control and dense-output foot-landing (``trace/integrator.py`` + ``trace/
boundaries.py``), and the deviation transport (``squashing/transport.py``) into one nopython unit,
because numba cannot cross the ``Field`` ABC / dataclasses / closures and inlines best within a
single module. The clean layered implementations are the source of truth; this runs whenever numba
is importable (numba ships in the default install; a lean install may omit it), with the NumPy path
as fallback and as the reference implementation for cross-checks.

Each field line is integrated independently in its own ``prange`` iteration, running its adaptive
loop to completion, so there is no per-step Python dispatch and no lockstep adaptive-straggler
waste. The numerics are byte-for-byte the reference ones: the DOPRI5 5(4)7M tableau, the PI-control
constants, and the numerical floors are **imported** from ``trace/integrator.py`` (single source);
only the floating-point accumulation order differs from the einsum path, so results agree to FP
noise (validated by the dipole gates running through this kernel).

The module also hosts the **line-of-sight render kernel** (:func:`render_batch_jit`, one ray per
``prange`` lane), the same scalar-per-lane port applied to ``render/los.py``'s NumPy quadrature: it
reuses the interpolation/grid primitives below (via :func:`_tricubic_point_scalar`) to integrate the
weighted log₁₀ Q⊥ volume, with the NumPy render kept as its fallback and reference.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from qorona.accel import JitField, JitGrid
from qorona.field.interpolation import _MIN_KEPT_WEIGHT
from qorona.resample.grid import GHOST
from qorona.trace.fieldline import Endpoint
from qorona.trace.integrator import (
    _A,
    _ALPHA,
    _B,
    _BETA,
    _DENSE_P,
    _E,
    _ERR_PREV_FLOOR,
    _H_MIN_FRACTION,
    _MAX_FACTOR,
    _MIN_FACTOR,
    _SAFETY,
)

# Endpoint codes as plain ints (the IntEnum does not survive nopython; kept in sync with it).
_INNER = int(Endpoint.INNER)
_OUTER = int(Endpoint.OUTER)
_NULL = int(Endpoint.NULL)
_MAX_STEPS = int(Endpoint.MAX_STEPS)
_STALLED = int(Endpoint.STALLED)
_DEFLECTED = int(Endpoint.DEFLECTED)

_TWO_PI = 2.0 * math.pi


@njit(cache=True)
def _keys_weights(t: float) -> tuple[float, float, float, float]:
    """The four Keys cubic-convolution weights (``a = -1/2``) at fractional offset ``t``."""
    t2 = t * t
    t3 = t2 * t
    return (
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    )


@njit(cache=True)
def _keys_weight_derivatives(t: float) -> tuple[float, float, float, float]:
    """The derivatives ``dW/dt`` of the four Keys weights at fractional offset ``t``."""
    t2 = t * t
    return (
        -1.5 * t2 + 2.0 * t - 0.5,
        4.5 * t2 - 5.0 * t,
        -4.5 * t2 + 4.0 * t + 0.5,
        1.5 * t2 - 1.0 * t,
    )


@njit(cache=True)
def _tricubic_point(
    b_padded: np.ndarray,
    c0: float,
    c1: float,
    c2: float,
    gradient: bool,
    value: np.ndarray,
    igrad: np.ndarray,
) -> None:
    """Interpolate the padded B array at one point, writing ``value[3]`` and ``igrad[3,3]``.

    ``igrad[d, i] = ∂value_i/∂coord_d`` is the index-space gradient (written only when
    ``gradient`` is ``True``; otherwise left untouched). The 4-point Keys stencil spans offsets
    ``-1..+2`` about ``floor(coord)`` on each axis; callers guarantee the stencil is in range via
    the ghost padding and the radial clip.
    """
    base0 = int(np.floor(c0))
    base1 = int(np.floor(c1))
    base2 = int(np.floor(c2))
    wx = _keys_weights(c0 - base0)
    wy = _keys_weights(c1 - base1)
    wz = _keys_weights(c2 - base2)

    value[0] = 0.0
    value[1] = 0.0
    value[2] = 0.0
    if gradient:
        for d in range(3):
            igrad[d, 0] = 0.0
            igrad[d, 1] = 0.0
            igrad[d, 2] = 0.0
        dwx = _keys_weight_derivatives(c0 - base0)
        dwy = _keys_weight_derivatives(c1 - base1)
        dwz = _keys_weight_derivatives(c2 - base2)

    for a in range(4):
        ia = base0 + a - 1
        for b in range(4):
            ib = base1 + b - 1
            for c in range(4):
                ic = base2 + c - 1
                n0 = b_padded[ia, ib, ic, 0]
                n1 = b_padded[ia, ib, ic, 1]
                n2 = b_padded[ia, ib, ic, 2]
                w = wx[a] * wy[b] * wz[c]
                value[0] += w * n0
                value[1] += w * n1
                value[2] += w * n2
                if gradient:
                    g0 = dwx[a] * wy[b] * wz[c]
                    g1 = wx[a] * dwy[b] * wz[c]
                    g2 = wx[a] * wy[b] * dwz[c]
                    igrad[0, 0] += g0 * n0
                    igrad[0, 1] += g0 * n1
                    igrad[0, 2] += g0 * n2
                    igrad[1, 0] += g1 * n0
                    igrad[1, 1] += g1 * n1
                    igrad[1, 2] += g1 * n2
                    igrad[2, 0] += g2 * n0
                    igrad[2, 1] += g2 * n1
                    igrad[2, 2] += g2 * n2


@njit(cache=True)
def _tricubic_point_scalar(vol: np.ndarray, c0: float, c1: float, c2: float) -> float:
    """NaN-tolerant scalar Keys tricubic of a width-1 padded volume at one point; returns the value.

    The scalar counterpart of :func:`_tricubic_point` (hardcoded to the 3-component B field) for the
    ``(..., 1)`` log₁₀ Q⊥ payload: gather the 4x4x4 Keys stencil, skip any non-finite tap from both
    the weighted sum and the kept weight, and return ``Σ w·v / Σ w`` over the finite taps; ``NaN``
    only where every tap is non-finite or the signed kept weight cancels below
    :data:`~qorona.field.interpolation._MIN_KEPT_WEIGHT`. Matches
    :func:`~qorona.field.interpolation.tricubic` (``skip_nan=True``) and
    :meth:`~qorona.squashing.volume.QPerpVolume.sample`.
    """
    base0 = int(np.floor(c0))
    base1 = int(np.floor(c1))
    base2 = int(np.floor(c2))
    wx = _keys_weights(c0 - base0)
    wy = _keys_weights(c1 - base1)
    wz = _keys_weights(c2 - base2)

    numerator = 0.0
    weight = 0.0
    for a in range(4):
        ia = base0 + a - 1
        for b in range(4):
            ib = base1 + b - 1
            for c in range(4):
                ic = base2 + c - 1
                tap = vol[ia, ib, ic, 0]
                if math.isfinite(tap):
                    w = wx[a] * wy[b] * wz[c]
                    numerator += w * tap
                    weight += w
    if abs(weight) > _MIN_KEPT_WEIGHT:
        return numerator / weight
    return math.nan


@njit(cache=True)
def _radial_parameter_and_derivative(
    radius: float, spacing_code: int, r_inner: float, r_outer: float, exponent: float
) -> tuple[float, float]:
    """Return ``(u, dr/du)`` at ``radius`` for the spacing law: the radial index map's core.

    ``u = parameter(radius)`` is the uniform coordinate the radial nodes are equally spaced in, and
    ``dr/du = radius_derivative(u)`` is the chain-rule factor (its reciprocal scales the radial
    gradient; it also sets the radial cell extent).
    """
    if spacing_code == 0:  # logarithmic
        log_ratio = math.log(r_outer / r_inner)
        return math.log(radius / r_inner) / log_ratio, radius * log_ratio
    if spacing_code == 1:  # power-law
        inv = 1.0 / exponent
        inner_root = r_inner**inv
        outer_root = r_outer**inv
        root = radius**inv
        parameter = (root - inner_root) / (outer_root - inner_root)
        return parameter, exponent * root ** (exponent - 1.0) * (outer_root - inner_root)
    # uniform
    return (radius - r_inner) / (r_outer - r_inner), r_outer - r_inner


@njit(cache=True)
def _spherical(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Cartesian → spherical ``(r, θ, φ)`` with ``θ`` colatitude and ``φ ∈ [0, 2π)``."""
    r = math.sqrt(x * x + y * y + z * z)
    cos_theta = z / r if r > 0.0 else 1.0
    if cos_theta > 1.0:
        cos_theta = 1.0
    elif cos_theta < -1.0:
        cos_theta = -1.0
    theta = math.acos(cos_theta)
    phi = math.atan2(y, x)
    if phi < 0.0:
        phi += _TWO_PI
    return r, theta, phi


@njit(cache=True)
def _char_len(fld: JitField, x: float, y: float, z: float) -> float:
    """The local CFL cell metric at a point (grid: smallest cell extent; dipole: constant)."""
    if fld.kind == 1:
        return fld.char_const
    r, theta, _ = _spherical(x, y, z)
    _, dr_du = _radial_parameter_and_derivative(
        r, fld.spacing_code, fld.r_inner, fld.r_outer, fld.exponent
    )
    radial = dr_du / (fld.n_r - 1)
    meridional = r * (math.pi / fld.n_theta)
    # Floor the azimuthal arc at the meridional arc: at the pole r·sinθ·Δφ collapses to zero from
    # meridian convergence, a coordinate artifact the C¹-through-pole field has no structure at, so
    # it must not tighten the step below the genuinely-resolved scale (mirrors cell_extent).
    azimuthal = r * math.sin(theta) * (_TWO_PI / fld.n_phi)
    if azimuthal < meridional:
        azimuthal = meridional
    extent = radial
    if meridional < extent:
        extent = meridional
    if azimuthal < extent:
        extent = azimuthal
    return extent


@njit(cache=True)
def _coord_jacobian(x: float, y: float, z: float, jac: np.ndarray) -> None:
    """Write ``jac[d, j] = ∂(r, θ, φ)_d/∂x_j``: exact, with no polar-axis regularization."""
    cyl2 = x * x + y * y
    cyl = math.sqrt(cyl2)
    r2 = cyl2 + z * z
    r = math.sqrt(r2)
    jac[0, 0] = x / r
    jac[0, 1] = y / r
    jac[0, 2] = z / r
    jac[1, 0] = x * z / (r2 * cyl)
    jac[1, 1] = y * z / (r2 * cyl)
    jac[1, 2] = -cyl / r2
    jac[2, 0] = -y / cyl2
    jac[2, 1] = x / cyl2
    jac[2, 2] = 0.0


@njit(cache=True)
def _sample_point(
    fld: JitField, x: float, y: float, z: float, gradient: bool, b: np.ndarray, grad_b: np.ndarray
) -> float:
    """Evaluate B (and ∇B if ``gradient``) at a point; write ``b[3]``/``grad_b[3,3]``, return |B|.

    Mirrors :meth:`SampledField.sample` (``kind == 0``) and :meth:`PfssDipoleField._evaluate`
    (``kind == 1``), including the gridded radial clip into the Keys in-range band and the
    chain-rule from the index-space gradient to the Cartesian Jacobian ``∂B_i/∂x_j`` (with the exact
    ``∂φ/∂x`` factor, no polar-axis floor, so an on-axis point yields a non-finite Jacobian by
    design, exactly as the NumPy path).
    """
    if fld.kind == 0:
        r, theta, phi = _spherical(x, y, z)
        parameter, dr_du = _radial_parameter_and_derivative(
            r, fld.spacing_code, fld.r_inner, fld.r_outer, fld.exponent
        )
        theta_step = math.pi / fld.n_theta
        phi_step = _TWO_PI / fld.n_phi
        c0 = parameter * (fld.n_r - 1) + GHOST
        c1 = theta / theta_step - 0.5 + GHOST
        c2 = phi / phi_step + GHOST
        # Clamp the radial coord into the Keys in-range band so an RK stage overrunning the shell
        # reads edge-extrapolation rather than off the padded array (mirrors SampledField.sample).
        hi = fld.b_padded.shape[0] - 3
        if c0 < 1.0:
            c0 = 1.0
        elif c0 > hi:
            c0 = hi

        igrad = np.empty((3, 3))
        _tricubic_point(fld.b_padded, c0, c1, c2, gradient, b, igrad)

        if gradient:
            d_index_r = (fld.n_r - 1) / dr_du
            d_index_theta = 1.0 / theta_step
            d_index_phi = 1.0 / phi_step
            jac = np.empty((3, 3))
            _coord_jacobian(x, y, z, jac)
            # grad_b[i, j] = Σ_d (igrad[d, i] · d_index_d) · jac[d, j]
            for i in range(3):
                for j in range(3):
                    grad_b[i, j] = (
                        igrad[0, i] * d_index_r * jac[0, j]
                        + igrad[1, i] * d_index_theta * jac[1, j]
                        + igrad[2, i] * d_index_phi * jac[2, j]
                    )
        return math.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2])

    # Dipole: B = m (3 z x / r⁵ - ẑ / r³) + B₀ ẑ, regular on the axis.
    moment = fld.moment
    r2 = x * x + y * y + z * z
    r3_inv = r2**-1.5
    r5_inv = r2**-2.5
    coef = 3.0 * z * r5_inv
    b[0] = moment * coef * x
    b[1] = moment * coef * y
    b[2] = moment * (coef * z - r3_inv) + fld.background
    if gradient:
        r7_inv = r2**-3.5
        c1d = 3.0 * moment * r5_inv
        c2d = 15.0 * moment * z * r7_inv
        p = (x, y, z)
        for i in range(3):
            zi = 1.0 if i == 2 else 0.0
            for j in range(3):
                zj = 1.0 if j == 2 else 0.0
                eye = 1.0 if i == j else 0.0
                grad_b[i, j] = c1d * (p[i] * zj + z * eye + zi * p[j]) - c2d * p[i] * p[j]
    return math.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2])


@njit(cache=True)
def _eval_rhs(
    out: np.ndarray, state: np.ndarray, transport: bool, direction: float, fld: JitField
) -> None:
    """Write the unit-field RHS into ``out`` (position, +deviation block if ``transport``).

    The position block is ``direction · B̂`` and the deviation block is
    ``direction · (∇B̂ · U, ∇B̂ · V)`` with ``∇B̂ = (I - B̂B̂ᵀ)·∇B/|B|`` contracted on its second index.
    At a magnetic null (``|B| = 0``) the ``1/|B|`` makes ``out`` non-finite; the integrator's
    top-of-step null guard detects this on the carried derivative, exactly as the NumPy path.
    """
    b = np.empty(3)
    grad_b = np.empty((3, 3))
    bmag = _sample_point(fld, state[0], state[1], state[2], transport, b, grad_b)
    inv = 1.0 / bmag
    bhx = b[0] * inv
    bhy = b[1] * inv
    bhz = b[2] * inv
    out[0] = direction * bhx
    out[1] = direction * bhy
    out[2] = direction * bhz

    if transport:
        # along[j] = Σ_k B̂_k ∂B_k/∂x_j  (contract B̂ with grad_b's first index).
        along0 = bhx * grad_b[0, 0] + bhy * grad_b[1, 0] + bhz * grad_b[2, 0]
        along1 = bhx * grad_b[0, 1] + bhy * grad_b[1, 1] + bhz * grad_b[2, 1]
        along2 = bhx * grad_b[0, 2] + bhy * grad_b[1, 2] + bhz * grad_b[2, 2]
        bh = (bhx, bhy, bhz)
        along = (along0, along1, along2)
        ghat = np.empty((3, 3))
        for i in range(3):
            for j in range(3):
                ghat[i, j] = (grad_b[i, j] - bh[i] * along[j]) * inv
        for i in range(3):
            du = ghat[i, 0] * state[3] + ghat[i, 1] * state[4] + ghat[i, 2] * state[5]
            dv = ghat[i, 0] * state[6] + ghat[i, 1] * state[7] + ghat[i, 2] * state[8]
            out[3 + i] = direction * du
            out[6 + i] = direction * dv


@njit(cache=True)
def _dopri5_step(
    stages: np.ndarray,
    new_state: np.ndarray,
    error: np.ndarray,
    state: np.ndarray,
    derivative: np.ndarray,
    step: float,
    transport: bool,
    direction: float,
    fld: JitField,
) -> None:
    """One DOPRI5 5(4) step (FSAL): fill ``stages[7,sd]``, ``new_state[sd]``, ``error[sd]``.

    ``stages[0]`` is the carried ``k1``; this evaluates the RHS six more times (stages 2-6 and the
    FSAL stage 7 at the new point). The autonomous unit-field RHS ignores the ``c`` nodes.
    """
    sd = state.shape[0]
    tmp = np.empty(sd)
    for k in range(sd):
        stages[0, k] = derivative[k]
    for stage in range(1, 6):
        for k in range(sd):
            inc = 0.0
            for j in range(stage):
                inc += _A[stage, j] * stages[j, k]
            tmp[k] = state[k] + step * inc
        _eval_rhs(stages[stage], tmp, transport, direction, fld)
    for k in range(sd):
        acc = 0.0
        for j in range(6):
            acc += _B[j] * stages[j, k]
        new_state[k] = state[k] + step * acc
    _eval_rhs(stages[6], new_state, transport, direction, fld)
    for k in range(sd):
        acc = 0.0
        for j in range(7):
            acc += _E[j] * stages[j, k]
        error[k] = step * acc


@njit(cache=True)
def _foot_gap(state: np.ndarray, coeff: np.ndarray, step: float, theta: float, tsq: float) -> float:
    """``|x(θ)|² - R²`` on the dense-output position interpolant (the foot root function)."""
    th2 = theta * theta
    th3 = th2 * theta
    th4 = th3 * theta
    p0 = state[0] + step * (
        theta * coeff[0, 0] + th2 * coeff[1, 0] + th3 * coeff[2, 0] + th4 * coeff[3, 0]
    )
    p1 = state[1] + step * (
        theta * coeff[0, 1] + th2 * coeff[1, 1] + th3 * coeff[2, 1] + th4 * coeff[3, 1]
    )
    p2 = state[2] + step * (
        theta * coeff[0, 2] + th2 * coeff[1, 2] + th3 * coeff[2, 2] + th4 * coeff[3, 2]
    )
    return p0 * p0 + p1 * p1 + p2 * p2 - tsq


@njit(cache=True)
def _sign(value: float) -> float:
    """Three-way sign, matching ``numpy.sign`` (``-1``/``0``/``+1``)."""
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


@njit(cache=True)
def _localize_foot(state: np.ndarray, coeff: np.ndarray, step: float, r_target: float) -> float:
    """Vectorless bisection of ``|x(θ)| = R`` over ``[0, 1]`` (the dense-output foot position)."""
    tsq = r_target * r_target
    low = 0.0
    high = 1.0
    sign_low = _sign(_foot_gap(state, coeff, step, 0.0, tsq))
    for _ in range(80):
        mid = 0.5 * (low + high)
        if _sign(_foot_gap(state, coeff, step, mid, tsq)) == sign_low:
            low = mid
        else:
            high = mid
        if high - low < 1e-14:
            break
    return 0.5 * (low + high)


@njit(cache=True)
def _integrate_line(
    state0: np.ndarray,
    direction: float,
    transport: bool,
    fld: JitField,
    atol: np.ndarray,
    rtol: float,
    cfl: float,
    max_steps: int,
    max_reversals: int,
    turn_cos: float,
    turn_radius: float,
    weak_threshold: float,
    turn_min: int,
    out_state: np.ndarray,
) -> tuple[int, float]:
    """Integrate one field line to a boundary; return ``(Endpoint code, arc length)``.

    Writes the landed terminal state into ``out_state`` on a clean foot; leaves it untouched (the
    caller pre-fills ``NaN``) for an aborted line. Byte-for-byte the reference adaptive loop: PI
    step control with the no-growth-after-rejection cap, the CFL ceiling, the parameter-free null
    guard, the stall guard (``max_reversals``), the sharp-turn guard (``turn_cos`` / ``turn_radius``
    / ``weak_threshold``), inclusive boundary crossing, and dense-output foot-landing.
    """
    sd = state0.shape[0]
    state = state0.copy()
    derivative = np.empty(sd)
    _eval_rhs(derivative, state, transport, direction, fld)

    step = cfl * _char_len(fld, state[0], state[1], state[2])
    err_prev = _ERR_PREV_FLOOR
    rejected_last = False
    arc_length = 0.0
    n_reversals = 0
    n_turns = 0
    r_inner = fld.r_inner
    r_outer = fld.r_outer

    stages = np.empty((7, sd))
    new_state = np.empty(sd)
    error = np.empty(sd)
    coeff = np.empty((4, sd))
    b_turn = np.empty(3)
    grad_turn = np.empty((3, 3))

    iteration = 0
    while iteration < max_steps:
        finite = True
        for k in range(sd):
            if not math.isfinite(derivative[k]):
                finite = False
                break
        if not finite:
            return _NULL, arc_length

        ceiling = cfl * _char_len(fld, state[0], state[1], state[2])
        step_now = step if step < ceiling else ceiling

        _dopri5_step(
            stages, new_state, error, state, derivative, step_now, transport, direction, fld
        )

        sq_sum = 0.0
        for k in range(sd):
            ref = abs(state[k])
            other = abs(new_state[k])
            scale = atol[k] + rtol * (ref if ref > other else other)
            scaled = error[k] / scale
            sq_sum += scaled * scaled
        err = math.sqrt(sq_sum / sd)
        if not math.isfinite(err):
            err = math.inf
        accepted = err <= 1.0

        end_radius = math.sqrt(
            new_state[0] * new_state[0] + new_state[1] * new_state[1] + new_state[2] * new_state[2]
        )
        crossed_outer = end_radius >= r_outer
        crossed = crossed_outer or (end_radius <= r_inner)

        if not math.isfinite(err):
            base_factor = 0.0
        elif err == 0.0:
            base_factor = math.inf
        else:
            base_factor = _SAFETY * err ** (-_ALPHA)

        if accepted and not crossed:
            # B̂_prev · B̂_new across the step (the position block of the FSAL derivative, still the
            # previous step's value here, against the new FSAL stage): one dot for both weak-field
            # guards, taken before the FSAL stage is rolled forward.
            if max_reversals > 0 or weak_threshold > 0.0:
                dot = (
                    derivative[0] * stages[6, 0]
                    + derivative[1] * stages[6, 1]
                    + derivative[2] * stages[6, 2]
                )
                # Sharp-turn guard: count sharp turns above turn_radius into field weaker than
                # weak_threshold, terminating once turn_min of them accumulate: a sustained
                # staircase, not a single legitimate null graze. |B| is sampled only at the rare
                # corner passing the cheap turn and radius tests; takes precedence over the stall
                # count when a step trips both.
                if weak_threshold > 0.0 and dot < turn_cos and end_radius > turn_radius:
                    bmag = _sample_point(
                        fld, new_state[0], new_state[1], new_state[2], False, b_turn, grad_turn
                    )
                    if bmag < weak_threshold:
                        n_turns += 1
                        if n_turns >= turn_min:
                            return _DEFLECTED, arc_length
                # Stall guard: a >90° turn is the thrashing-trap signature; count it.
                if max_reversals > 0 and dot < 0.0:
                    n_reversals += 1
                    if n_reversals >= max_reversals:
                        return _STALLED, arc_length
            growth = base_factor * err_prev**_BETA
            max_factor = 1.0 if rejected_last else _MAX_FACTOR
            for k in range(sd):
                state[k] = new_state[k]
                derivative[k] = stages[6, k]
            arc_length += step_now
            err_prev = err if err > _ERR_PREV_FLOOR else _ERR_PREV_FLOOR
            if growth < _MIN_FACTOR:
                growth = _MIN_FACTOR
            elif growth > max_factor:
                growth = max_factor
            step = step_now * growth
            rejected_last = False
        elif accepted and crossed:
            r_target = r_outer if crossed_outer else r_inner
            for p in range(4):
                for d in range(sd):
                    acc = 0.0
                    for s in range(7):
                        acc += stages[s, d] * _DENSE_P[s, p]
                    coeff[p, d] = acc
            theta_star = _localize_foot(state, coeff, step_now, r_target)
            th2 = theta_star * theta_star
            th3 = th2 * theta_star
            th4 = th3 * theta_star
            for d in range(sd):
                out_state[d] = state[d] + step_now * (
                    theta_star * coeff[0, d]
                    + th2 * coeff[1, d]
                    + th3 * coeff[2, d]
                    + th4 * coeff[3, d]
                )
            return (_OUTER if crossed_outer else _INNER), arc_length + theta_star * step_now
        else:
            if base_factor < _MIN_FACTOR:
                base_factor = _MIN_FACTOR
            elif base_factor > 1.0:
                base_factor = 1.0
            shrunk = step_now * base_factor
            step = shrunk
            rejected_last = True
            if shrunk < _H_MIN_FRACTION * ceiling:
                return _MAX_STEPS, arc_length

        iteration += 1

    return _MAX_STEPS, arc_length


@njit(parallel=True, cache=True)
def integrate_batch_jit(
    state0: np.ndarray,
    active: np.ndarray,
    directions: np.ndarray,
    transport: bool,
    fld: JitField,
    atol: np.ndarray,
    rtol: float,
    cfl: float,
    max_steps: int,
    max_reversals: int,
    turn_cos: float,
    turn_radius: float,
    weak_threshold: float,
    turn_min: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate a batch of lines in parallel, one line per ``prange`` iteration.

    Drop-in numba replacement for the NumPy ``_integrate_batch`` over the position-only (``sd = 3``)
    and deviation-transport (``sd = 9``) states. ``store_path`` is not supported here (routed to the
    NumPy path by the dispatcher). Aborted lines keep a ``NaN`` terminal state.

    Returns ``(terminal_state (n, sd), ends (n,) int8, lengths (n,))``.
    """
    n = state0.shape[0]
    sd = state0.shape[1]
    terminal_state = np.full((n, sd), np.nan)
    ends = np.zeros(n, dtype=np.int8)
    lengths = np.zeros(n)
    for i in prange(n):
        if not active[i]:
            continue
        code, length = _integrate_line(
            state0[i],
            directions[i],
            transport,
            fld,
            atol,
            rtol,
            cfl,
            max_steps,
            max_reversals,
            turn_cos,
            turn_radius,
            weak_threshold,
            turn_min,
            terminal_state[i],
        )
        ends[i] = code
        lengths[i] = length
    return terminal_state, ends, lengths


# --- Painting kernel ---------------------------------------------------------------------------
# The painting Q⊥-volume builder traces a fixed set of seeded lines and rasterizes each line's
# constant Q⊥ into every voxel its swept path crosses. Tracing reuses the position-only stepper
# above byte-for-byte; the painting-specific work is forward-binning sub-sampled path points into
# the *volume* grid (a JitGrid distinct from the traced field's JitField) and recording the deduped
# voxel run.
# Each prange lane writes only its own output row (no shared array, hence no atomics: the
# trace-parallel, paint-serial scheme); the caller applies the per-line value and scatters into the
# shared grid serially.


@njit(cache=True)
def _vol_flat(vg: JitGrid, x: float, y: float, z: float) -> int:
    """Return the flat C-order node index of the volume voxel a point falls in (φ wrap, θ/r clip).

    The forward of :meth:`SphericalGrid.index_coordinates` (floor the fractional indices to a cell,
    wrap φ periodically, clamp θ and r into the node range), flattened ``(i_r·n_θ + i_θ)·n_φ + i_φ``
    to match :func:`~qorona.squashing.volume._pack_volume`.
    """
    r, theta, phi = _spherical(x, y, z)
    parameter, _ = _radial_parameter_and_derivative(
        r, vg.spacing_code, vg.r_inner, vg.r_outer, vg.exponent
    )
    r_index = parameter * (vg.n_r - 1)
    theta_index = theta / (math.pi / vg.n_theta) - 0.5
    phi_index = phi / (_TWO_PI / vg.n_phi)

    i_r = int(np.floor(r_index))
    if i_r < 0:
        i_r = 0
    elif i_r > vg.n_r - 1:
        i_r = vg.n_r - 1
    i_theta = int(np.floor(theta_index))
    if i_theta < 0:
        i_theta = 0
    elif i_theta > vg.n_theta - 1:
        i_theta = vg.n_theta - 1
    i_phi = int(np.floor(phi_index)) % vg.n_phi
    return (i_r * vg.n_theta + i_theta) * vg.n_phi + i_phi


@njit(cache=True)
def _vol_cell_extent(vg: JitGrid, x: float, y: float, z: float) -> float:
    """Return the smallest local cell extent of the volume grid at a point (the paint-pitch metric).

    Mirrors :meth:`SphericalGrid.cell_extent` (the minimum of the radial, meridional, and
    azimuthal node spacings), so the sub-sample pitch ``paint_step · extent`` keeps consecutive
    deposits inside one cell of one another.
    """
    r, theta, _ = _spherical(x, y, z)
    _, dr_du = _radial_parameter_and_derivative(
        r, vg.spacing_code, vg.r_inner, vg.r_outer, vg.exponent
    )
    radial = dr_du / (vg.n_r - 1)
    meridional = r * (math.pi / vg.n_theta)
    # Azimuthal arc floored at the meridional arc: the pole de-singularization (see cell_extent).
    azimuthal = r * math.sin(theta) * (_TWO_PI / vg.n_phi)
    if azimuthal < meridional:
        azimuthal = meridional
    extent = radial
    if meridional < extent:
        extent = meridional
    if azimuthal < extent:
        extent = azimuthal
    return extent


@njit(cache=True)
def _dense_position(
    state: np.ndarray, coeff: np.ndarray, step: float, theta: float, out: np.ndarray
) -> None:
    """Write the dense-output position at fractional step ``theta`` into ``out[3]``."""
    th2 = theta * theta
    th3 = th2 * theta
    th4 = th3 * theta
    for d in range(3):
        out[d] = state[d] + step * (
            theta * coeff[0, d] + th2 * coeff[1, d] + th3 * coeff[2, d] + th4 * coeff[3, d]
        )


@njit(cache=True)
def _paint_half_line(
    seed: np.ndarray,
    direction: float,
    fld: JitField,
    vg: JitGrid,
    atol: np.ndarray,
    rtol: float,
    cfl: float,
    max_steps: int,
    paint_step: float,
    out_voxels: np.ndarray,
    count: int,
    last_flat: int,
    max_deposits: int,
) -> tuple[int, int, bool]:
    """Trace one half-line position-only, appending the voxels it sweeps to ``out_voxels``.

    Mirrors :func:`_integrate_line` (``sd = 3``) step for step (same DOPRI5 tableau, PI control,
    CFL ceiling, null guard, inclusive crossing, dense-output foot), but on each accepted step it
    sub-samples the dense output at a pitch of ``paint_step`` times the local volume cell and
    appends each newly-entered voxel (deduped against ``last_flat``) to ``out_voxels[count:]``.
    Returns the updated ``(count, last_flat, overflow)``; ``overflow`` is ``True`` if the run hit
    ``max_deposits`` (the line's tail is then dropped).
    """
    state = seed.copy()
    derivative = np.empty(3)
    _eval_rhs(derivative, state, False, direction, fld)

    step = cfl * _char_len(fld, state[0], state[1], state[2])
    err_prev = _ERR_PREV_FLOOR
    rejected_last = False
    r_inner = fld.r_inner
    r_outer = fld.r_outer

    stages = np.empty((7, 3))
    new_state = np.empty(3)
    error = np.empty(3)
    coeff = np.empty((4, 3))
    point = np.empty(3)

    iteration = 0
    while iteration < max_steps:
        finite = True
        for k in range(3):
            if not math.isfinite(derivative[k]):
                finite = False
                break
        if not finite:
            return count, last_flat, False

        ceiling = cfl * _char_len(fld, state[0], state[1], state[2])
        step_now = step if step < ceiling else ceiling
        _dopri5_step(stages, new_state, error, state, derivative, step_now, False, direction, fld)

        sq_sum = 0.0
        for k in range(3):
            ref = abs(state[k])
            other = abs(new_state[k])
            scale = atol[k] + rtol * (ref if ref > other else other)
            scaled = error[k] / scale
            sq_sum += scaled * scaled
        err = math.sqrt(sq_sum / 3.0)
        if not math.isfinite(err):
            err = math.inf
        accepted = err <= 1.0

        end_radius = math.sqrt(
            new_state[0] * new_state[0] + new_state[1] * new_state[1] + new_state[2] * new_state[2]
        )
        crossed_outer = end_radius >= r_outer
        crossed = crossed_outer or (end_radius <= r_inner)

        if not math.isfinite(err):
            base_factor = 0.0
        elif err == 0.0:
            base_factor = math.inf
        else:
            base_factor = _SAFETY * err ** (-_ALPHA)

        if accepted:
            # Dense-output coefficients of this step, used for the sub-sample (and any foot).
            for p in range(4):
                for d in range(3):
                    acc = 0.0
                    for s in range(7):
                        acc += stages[s, d] * _DENSE_P[s, p]
                    coeff[p, d] = acc

            theta_end = 1.0
            if crossed:
                theta_end = _localize_foot(
                    state, coeff, step_now, r_outer if crossed_outer else r_inner
                )

            # Pitch from the smaller of the step's start and end cell extents, so a step over a
            # region where the volume cell shrinks (toward the inner boundary or a pole) still
            # sub-samples finely enough not to skip a voxel.
            _dense_position(state, coeff, step_now, theta_end, point)
            extent = _vol_cell_extent(vg, state[0], state[1], state[2])
            extent_end = _vol_cell_extent(vg, point[0], point[1], point[2])
            if extent_end < extent:
                extent = extent_end
            pitch = paint_step * extent
            arc = theta_end * step_now
            n_sub = int(arc / pitch) + 1 if pitch > 0.0 else 1
            for j in range(1, n_sub + 1):
                _dense_position(state, coeff, step_now, (j / n_sub) * theta_end, point)
                flat = _vol_flat(vg, point[0], point[1], point[2])
                if flat != last_flat:
                    if count >= max_deposits:
                        return count, last_flat, True
                    out_voxels[count] = flat
                    count += 1
                    last_flat = flat

            if crossed:
                return count, last_flat, False

            growth = base_factor * err_prev**_BETA
            max_factor = 1.0 if rejected_last else _MAX_FACTOR
            for k in range(3):
                state[k] = new_state[k]
                derivative[k] = stages[6, k]
            err_prev = err if err > _ERR_PREV_FLOOR else _ERR_PREV_FLOOR
            if growth < _MIN_FACTOR:
                growth = _MIN_FACTOR
            elif growth > max_factor:
                growth = max_factor
            step = step_now * growth
            rejected_last = False
        else:
            if base_factor < _MIN_FACTOR:
                base_factor = _MIN_FACTOR
            elif base_factor > 1.0:
                base_factor = 1.0
            shrunk = step_now * base_factor
            step = shrunk
            rejected_last = True
            if shrunk < _H_MIN_FRACTION * ceiling:
                return count, last_flat, False

        iteration += 1

    return count, last_flat, False


@njit(parallel=True, cache=True)
def paint_batch_jit(
    seeds: np.ndarray,
    valid: np.ndarray,
    fld: JitField,
    vg: JitGrid,
    atol: np.ndarray,
    rtol: float,
    cfl: float,
    max_steps: int,
    paint_step: float,
    max_deposits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trace each valid seed both ways and record its painted voxels; one seed per ``prange`` lane.

    Returns ``(voxels, counts, overflow)``: ``voxels[i, :counts[i]]`` are the deduped flat node
    indices the line through seed ``i`` sweeps (the seed voxel first, then each half-line), and
    ``overflow[i]`` flags a line whose run exceeded ``max_deposits``. The per-line Q⊥ value is *not*
    needed here; the caller applies it in a serial scatter (paint-serial), so each lane writes only
    its own ``voxels`` row and no atomics are required. ``valid`` (the complete-line mask from the
    feet trace) skips seeds with no defined Q⊥.
    """
    n = seeds.shape[0]
    voxels = np.full((n, max_deposits), -1, dtype=np.int64)
    counts = np.zeros(n, dtype=np.int64)
    overflow = np.zeros(n, dtype=np.bool_)
    for i in prange(n):
        if not valid[i]:
            continue
        seed = seeds[i]
        seed_flat = _vol_flat(vg, seed[0], seed[1], seed[2])
        voxels[i, 0] = seed_flat
        count = 1
        # Both half-lines start at the seed voxel, so each dedups its first samples against it; the
        # seed is therefore deposited exactly once. The two halves diverge to opposite feet, so no
        # cross-half dedup is needed.
        count, _, overflow_back = _paint_half_line(
            seed,
            -1.0,
            fld,
            vg,
            atol,
            rtol,
            cfl,
            max_steps,
            paint_step,
            voxels[i],
            count,
            seed_flat,
            max_deposits,
        )
        overflow_forward = False
        if not overflow_back:
            count, _, overflow_forward = _paint_half_line(
                seed,
                1.0,
                fld,
                vg,
                atol,
                rtol,
                cfl,
                max_steps,
                paint_step,
                voxels[i],
                count,
                seed_flat,
                max_deposits,
            )
        counts[i] = count
        overflow[i] = overflow_back or overflow_forward
    return voxels, counts, overflow


# --- Render kernel -----------------------------------------------------------------------------
# The line-of-sight render integrates the weighted log₁₀ Q⊥ volume into an eclipse-like image. Each
# ray is independent: one ray per prange lane marches the shared s-grid, masks in-shell + (when
# occult_body) body-occulted samples, samples the scalar volume (the JitGrid index map +
# _tricubic_point_scalar, mirroring
# QPerpVolume.sample), clamps, and accumulates the weight-normalised per-channel average and the
# coverage. Each lane writes only its own output rows (no atomics); the caller reduces the per-lane
# clamp counts into the global provenance totals. Faithful port of render/los.py::_render_numpy;
# only the FP accumulation order differs.
#
# An optional Thomson scalar (electron density x single-electron intensity) biases the rendered Q⊥
# toward bright dense low-corona plasma: with use_thomson on, each valid sample reads Nₑ from
# a density JitGrid and the radius-only intensity coefficients from a precomputed table, and folds
# the scalar into the weighted-average accumulators num/den ONLY, never the geometric on-path /
# coverage budgets, so the depth-colour reconstruction and coverage stay byte-identical to the
# unweighted render (use_thomson off ⇒ the scalar is 1.0, an exact no-op).


@njit(cache=True)
def _coeff_lookup(
    r: float,
    coeff_log_inner: float,
    coeff_inv_dlog: float,
    c_tan_table: np.ndarray,
    c_pol_table: np.ndarray,
) -> tuple[float, float]:
    """Return ``(c_tan, c_pol)`` at radius ``r`` by linear interpolation in ``ln r`` of the table.

    The search-free, log-uniform node index, mirroring
    :meth:`~qorona.radiation.thomson.RadialCoefficients.evaluate` so the kernel and the NumPy path
    read identical coefficients.
    """
    size_minus_1 = c_tan_table.shape[0] - 1
    position = (math.log(r) - coeff_log_inner) * coeff_inv_dlog
    if position < 0.0:
        position = 0.0
    elif position > size_minus_1:
        position = size_minus_1
    lower = int(position)
    upper = lower + 1 if lower < size_minus_1 else lower
    frac = position - lower
    c_tan = c_tan_table[lower] * (1.0 - frac) + c_tan_table[upper] * frac
    c_pol = c_pol_table[lower] * (1.0 - frac) + c_pol_table[upper] * frac
    return c_tan, c_pol


@njit(cache=True)
def _thomson_intensity(
    r: float,
    rho: float,
    coeff_log_inner: float,
    coeff_inv_dlog: float,
    c_tan_table: np.ndarray,
    c_pol_table: np.ndarray,
    pb: bool,
) -> float:
    """Single-electron intensity ``I_tot`` (``pb`` off) or ``I_pol`` (``pb`` on) at one LOS sample.

    Reads ``c_tan(r)``, ``c_pol(r)`` from the table (:func:`_coeff_lookup`) and combines them with
    ``sin²χ̄ = rho²/r²``: ``I_pol = c_pol·sin²χ̄`` and ``I_tot = 2 c_tan - c_pol·sin²χ̄``.
    """
    c_tan, c_pol = _coeff_lookup(r, coeff_log_inner, coeff_inv_dlog, c_tan_table, c_pol_table)
    sin_sq_chi = rho * rho / (r * r)
    if pb:
        return c_pol * sin_sq_chi
    return 2.0 * c_tan - c_pol * sin_sq_chi


@njit(parallel=True, cache=True)
def render_batch_jit(
    origins: np.ndarray,
    look: np.ndarray,
    impact: np.ndarray,
    s_grid: np.ndarray,
    vol: np.ndarray,
    vg: JitGrid,
    sigma: float,
    powers: np.ndarray,
    use_powers: bool,
    scales: np.ndarray,
    use_scales: bool,
    floor: float,
    log_max: float,
    clamp_lower: bool,
    r_occult: float,
    occult_body: bool,
    use_thomson: bool,
    density_vol: np.ndarray,
    dg: JitGrid,
    thomson_pb: bool,
    coeff_log_inner: float,
    coeff_inv_dlog: float,
    c_tan_table: np.ndarray,
    c_pol_table: np.ndarray,
    use_polarity: bool,
    polarity_vol: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Render a batch of rays: the weight-normalised per-channel LOS average of clamped log₁₀ Q⊥.

    One ray per ``prange`` lane. Per step on the shared ``s_grid``: form ``r = sqrt(rho² + s²)`` and
    skip out-of-shell; when ``occult_body`` (the ``"opaque"`` body mode) skip occulted
    samples (``rho < r_occult`` and ``s`` behind the body); build the
    per-channel spatial weight (the preset factors: angular Gaussian ``sigma`` with a NaN sentinel,
    per-channel ``powers``/``scales`` under their use-flags, reproducing
    :meth:`WeightingPreset.channel_weights`); add it to the per-channel **on-path** weight budget
    (the depth-colour geometry, independent of the field and the NaN mask); then sample the scalar
    volume through the :class:`JitGrid` index map (mirroring :meth:`QPerpVolume.sample`, including
    its point-radius shell mask), count the lower/upper clamp breaches on the raw value, clamp to
    ``[floor, log_max]`` (upper-only when ``clamp_lower`` is ``False``, the ``raw`` mode), and
    accumulate the **valid** ``Σ w·t·v`` and ``Σ w·t`` per channel.

    The optional Thomson scalar ``t = Nₑ · I(r, χ̄)`` (``use_thomson``): at each valid sample it
    reads ``Nₑ`` from ``density_vol`` through the density grid ``dg``'s index map and the
    intensity from the coefficient table (:func:`_thomson_intensity`, ``thomson_pb`` selecting
    ``I_pol`` over ``I_tot``), and folds it into ``num``/``den`` only. ``use_thomson`` off makes
    the scalar ``1.0``, an exact no-op, so the result is byte-identical to the plain render and
    ``density_vol`` and the coefficient tables are unread placeholders (``dg`` is read for its
    shape only).

    The optional **net polarity** (``use_polarity``): at each valid sample the per-voxel footpoint
    sign is read NEAREST-CELL from ``polarity_vol`` (a sign is never interpolated) and summed as
    ``Σ w̄·sign`` with ``w̄ = (w0+w1+w2)/3`` the channel-mean geometric weight (the same budget the
    coverage uses), so the returned ``polarity = Σ w̄·sign / Σ w̄`` is the weight-averaged mean
    column polarity in ``[-1, +1]`` (``NaN`` where no valid sample, or with ``use_polarity`` off and
    ``polarity_vol`` an unread placeholder).

    Returns ``(signal (n, 3), coverage (n,), counts (n, 3), den (n, 3), onpath (n, 3),
    polarity (n,))``: the per-channel ``Σ w·t·v / Σ w·t`` (``NaN`` where no valid sample), the
    coverage ``Σw_valid/Σw_onpath`` (0 off-path, geometry only), per-ray ``(lower, upper, valid)``
    breach counts the caller reduces to the clamp provenance, the per-channel ``den = Σ_valid w·t``
    (so ``signal·den`` is the Thomson-weighted integral), the per-channel **on-path** geometric
    budget ``onpath = Σ_onpath w`` (so ``signal·onpath`` is the NaN-robust depth-colour integral),
    and the net column ``polarity``.
    """
    n_rays = origins.shape[0]
    n_steps = s_grid.shape[0]
    signal = np.full((n_rays, 3), np.nan)
    coverage = np.zeros(n_rays)
    counts = np.zeros((n_rays, 3), dtype=np.int64)
    den = np.zeros((n_rays, 3))
    onpath = np.zeros((n_rays, 3))
    polarity = np.full(n_rays, np.nan)

    use_sigma = not math.isnan(sigma)
    r_inner = vg.r_inner
    r_outer = vg.r_outer
    theta_step = math.pi / vg.n_theta
    phi_step = _TWO_PI / vg.n_phi
    density_theta_step = math.pi / dg.n_theta
    density_phi_step = _TWO_PI / dg.n_phi
    for i in prange(n_rays):
        ox = origins[i, 0]
        oy = origins[i, 1]
        oz = origins[i, 2]
        rho = impact[i]
        occulting = occult_body and rho < r_occult
        s_body = math.sqrt(r_occult * r_occult - rho * rho) if occulting else 0.0

        num0 = 0.0
        num1 = 0.0
        num2 = 0.0
        den0 = 0.0
        den1 = 0.0
        den2 = 0.0
        pden0 = 0.0
        pden1 = 0.0
        pden2 = 0.0
        valid_weight = 0.0
        path_weight = 0.0
        pol_num = 0.0
        n_lower = 0
        n_upper = 0
        n_valid = 0
        for k in range(n_steps):
            s = s_grid[k]
            r = math.sqrt(rho * rho + s * s)
            if r < r_inner or r > r_outer:
                continue
            if occulting and s < -s_body:
                continue

            w0 = 1.0
            w1 = 1.0
            w2 = 1.0
            if use_sigma:
                gaussian = math.exp(-0.5 * (s / (r * sigma)) ** 2)
                w0 *= gaussian
                w1 *= gaussian
                w2 *= gaussian
            if use_powers:
                w0 *= r ** (-powers[0])
                w1 *= r ** (-powers[1])
                w2 *= r ** (-powers[2])
            if use_scales:
                w0 *= math.exp(-r / scales[0])
                w1 *= math.exp(-r / scales[1])
                w2 *= math.exp(-r / scales[2])
            path_weight += (w0 + w1 + w2) / 3.0
            pden0 += w0
            pden1 += w1
            pden2 += w2

            px = ox + s * look[0]
            py = oy + s * look[1]
            pz = oz + s * look[2]
            r_point, theta, phi = _spherical(px, py, pz)
            # Mirror QPerpVolume.sample: the point-radius shell mask (NaN outside) and the same
            # index map.
            if r_point < r_inner or r_point > r_outer:
                continue
            parameter, _ = _radial_parameter_and_derivative(
                r_point, vg.spacing_code, r_inner, r_outer, vg.exponent
            )
            c0 = parameter * (vg.n_r - 1) + GHOST
            c1 = theta / theta_step - 0.5 + GHOST
            c2 = phi / phi_step + GHOST
            value = _tricubic_point_scalar(vol, c0, c1, c2)
            if not math.isfinite(value):
                continue

            n_valid += 1
            if value < floor:
                n_lower += 1
                if clamp_lower:
                    value = floor
            elif value > log_max:
                n_upper += 1
                value = log_max

            # The optional Thomson scalar Nₑ·I(r, χ̄): folded into the average only (num/den), the
            # geometric path/onpath/valid budgets above stay scalar-free. Off ⇒ an exact 1.0 no-op.
            thomson = 1.0
            if use_thomson:
                d_parameter, _ = _radial_parameter_and_derivative(
                    r_point, dg.spacing_code, dg.r_inner, dg.r_outer, dg.exponent
                )
                d0 = d_parameter * (dg.n_r - 1) + GHOST
                d1 = theta / density_theta_step - 0.5 + GHOST
                d2 = phi / density_phi_step + GHOST
                density = _tricubic_point_scalar(density_vol, d0, d1, d2)
                intensity = _thomson_intensity(
                    r, rho, coeff_log_inner, coeff_inv_dlog, c_tan_table, c_pol_table, thomson_pb
                )
                thomson = density * intensity

            num0 += w0 * thomson * value
            num1 += w1 * thomson * value
            num2 += w2 * thomson * value
            den0 += w0 * thomson
            den1 += w1 * thomson
            den2 += w2 * thomson
            wbar = (w0 + w1 + w2) / 3.0
            valid_weight += wbar
            # Net polarity: read the per-voxel footpoint sign NEAREST-CELL (a sign is never
            # interpolated) at this same sample and weight it by w̄, the coverage weight budget.
            if use_polarity:
                pic0 = int(c0 + 0.5)
                pic1 = int(c1 + 0.5)
                pic2 = int(c2 + 0.5)
                if pic0 < 0:
                    pic0 = 0
                elif pic0 >= polarity_vol.shape[0]:
                    pic0 = polarity_vol.shape[0] - 1
                if pic1 < 0:
                    pic1 = 0
                elif pic1 >= polarity_vol.shape[1]:
                    pic1 = polarity_vol.shape[1] - 1
                if pic2 < 0:
                    pic2 = 0
                elif pic2 >= polarity_vol.shape[2]:
                    pic2 = polarity_vol.shape[2] - 1
                pol_num += wbar * polarity_vol[pic0, pic1, pic2, 0]

        if den0 > 0.0:
            signal[i, 0] = num0 / den0
        if den1 > 0.0:
            signal[i, 1] = num1 / den1
        if den2 > 0.0:
            signal[i, 2] = num2 / den2
        if path_weight > 0.0:
            coverage[i] = valid_weight / path_weight
        if use_polarity and valid_weight > 0.0:
            polarity[i] = pol_num / valid_weight
        counts[i, 0] = n_lower
        counts[i, 1] = n_upper
        counts[i, 2] = n_valid
        den[i, 0] = den0
        den[i, 1] = den1
        den[i, 2] = den2
        onpath[i, 0] = pden0
        onpath[i, 1] = pden1
        onpath[i, 2] = pden2
    return signal, coverage, counts, den, onpath, polarity


# --- Brightness kernel -------------------------------------------------------------------------
# The standalone white-light / polarized-brightness product is a line-of-sight integral of the
# Thomson-scattered intensity over the electron density, the same camera and s-march as the Q⊥
# render, but the integrand is Nₑ·coefficient rather than weighted log₁₀ Q⊥, and there is no volume,
# clamp, NaN handling, or per-channel weighting. Each ray accumulates the tangential and polarized
# line integrals K_tan = ∫ Nₑ c_tan ds and K_pol = ∫ Nₑ c_pol sin²χ̄ ds; the caller forms the total
# brightness K_tot = 2 K_tan - K_pol and polarized brightness pB = K_pol. NaN-free (Nₑ dense).


@njit(parallel=True, cache=True)
def brightness_batch_jit(
    origins: np.ndarray,
    look: np.ndarray,
    impact: np.ndarray,
    s_grid: np.ndarray,
    density_vol: np.ndarray,
    dg: JitGrid,
    r_occult: float,
    occult_body: bool,
    coeff_log_inner: float,
    coeff_inv_dlog: float,
    c_tan_table: np.ndarray,
    c_pol_table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the tangential and polarized brightness over Nₑ along a batch of rays.

    One ray per ``prange`` lane, marching the shared ``s_grid``. Per in-shell step (and, with
    ``occult_body``, not behind the body): sample ``Nₑ`` via the density grid ``dg``'s index map,
    read ``c_tan``/``c_pol`` from the radial table, and accumulate ``K_tan += Nₑ c_tan`` and
    ``K_pol += Nₑ c_pol sin²χ̄`` with ``sin²χ̄ = rho²/r²``. The step length is applied by the caller.

    Returns ``(K_tan (n,), K_pol (n,))``: the polarized brightness ``pB = K_pol`` and the total
    ``K_tot = 2 K_tan - K_pol`` are formed on the host.
    """
    n_rays = origins.shape[0]
    n_steps = s_grid.shape[0]
    k_tan = np.zeros(n_rays)
    k_pol = np.zeros(n_rays)

    r_inner = dg.r_inner
    r_outer = dg.r_outer
    theta_step = math.pi / dg.n_theta
    phi_step = _TWO_PI / dg.n_phi
    for i in prange(n_rays):
        ox = origins[i, 0]
        oy = origins[i, 1]
        oz = origins[i, 2]
        rho = impact[i]
        occulting = occult_body and rho < r_occult
        s_body = math.sqrt(r_occult * r_occult - rho * rho) if occulting else 0.0

        tan_sum = 0.0
        pol_sum = 0.0
        for k in range(n_steps):
            s = s_grid[k]
            r = math.sqrt(rho * rho + s * s)
            if r < r_inner or r > r_outer:
                continue
            if occulting and s < -s_body:
                continue

            px = ox + s * look[0]
            py = oy + s * look[1]
            pz = oz + s * look[2]
            r_point, theta, phi = _spherical(px, py, pz)
            if r_point < r_inner or r_point > r_outer:
                continue
            parameter, _ = _radial_parameter_and_derivative(
                r_point, dg.spacing_code, r_inner, r_outer, dg.exponent
            )
            d0 = parameter * (dg.n_r - 1) + GHOST
            d1 = theta / theta_step - 0.5 + GHOST
            d2 = phi / phi_step + GHOST
            density = _tricubic_point_scalar(density_vol, d0, d1, d2)
            c_tan, c_pol = _coeff_lookup(
                r, coeff_log_inner, coeff_inv_dlog, c_tan_table, c_pol_table
            )
            sin_sq_chi = rho * rho / (r * r)
            tan_sum += density * c_tan
            pol_sum += density * c_pol * sin_sq_chi

        k_tan[i] = tan_sum
        k_pol[i] = pol_sum
    return k_tan, k_pol
