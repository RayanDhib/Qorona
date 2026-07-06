"""float64 CUDA mirror of the field-line transport hot loop (one line per GPU thread).

The GPU sibling of :mod:`qorona.accel.kernels`: the same scalar-per-lane DOPRI5 stepper, Keys
tricubic, spherical index map, coordinate Jacobian, field evaluation, and deviation transport,
ported to ``@cuda.jit`` device functions inlined into a per-thread loop. The port reproduces the
*algorithm*, not the bit pattern. The clean layered implementations are the source of truth; this
runs only when a CUDA GPU is present, with the numba-CPU kernel and the NumPy integrator as
fallbacks and as the references it is validated against. The module also hosts the **line-of-sight
render kernel** (:func:`render_batch_cuda`, one ray per thread), the CUDA twin of
:func:`qorona.accel.kernels.render_batch_jit`, under the same tiering and validation contract.

Three mechanical differences from the ``@njit`` twin, required by the device memory model:
``sd`` (state dim, 3 or 9) is a compile-time literal (``cuda.local.array`` needs a constant shape),
so the device functions come from a literal-templated factory, per ``sd`` and precision tier; every
per-thread scratch allocation is a fixed-shape ``cuda.local.array``; and FMA contraction is
*accepted*. NVVM contracts ``a*b+c`` into a fused multiply-add regardless of ``fastmath``, so the
device rounding of every multiply-add differs from the CPU path by at most one ULP. The bulk of
lines agree to float64 noise; the small tail of grazing, decision-divergent lines is reported by
``validation/cuda_parity.py`` rather than pinned. The DOPRI5 5(4)7M tableau, the PI constants, and
the numerical floors are imported from the single source (:mod:`qorona.trace.integrator`) and baked
as device globals; the deviation ``atol`` arrives per launch from :mod:`qorona.squashing.transport`.

A **mixed-precision variant** sits alongside the float64 kernels, selected per call by the
explicit ``precision`` argument (default ``mixed``; ``float64`` keeps the all-double reference
selectable; ``float32`` additionally swaps in the fully-float32 paint kernel, see
:data:`_PRECISION_MODES`). In ``mixed`` the only float32 is the tricubic stencil gather (the
dominant FLOP count and, with the field uploaded float32, the dominant device-memory object and
gather bandwidth), via
:func:`_sample_point_f32`; everything downstream (the DOPRI5 stepper, the deviation transport, the
error norm, the dense-output foot landing) stays float64, the f32 result cast up at the
:func:`_sample_point` boundary. The coordinate transcendentals and the dipole closed form stay
float64 too (no gather lever there). The float32 field upload and interpolation gather follow
FastQSL (Zhang et al. 2022), whose kernels otherwise run essentially all-float32; here the
f32/f64 boundary is drawn at the gather instead, keeping the transport arithmetic double. The
mixed-vs-float64 per-line Q⊥ agreement (the float32 tricubic noise floor) is reported by
``validation/cuda_parity.py``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numba import cuda, float32, float64

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

# Endpoint codes as plain ints (the IntEnum does not survive device code).
_INNER = int(Endpoint.INNER)
_OUTER = int(Endpoint.OUTER)
_NULL = int(Endpoint.NULL)
_MAX_STEPS = int(Endpoint.MAX_STEPS)
_STALLED = int(Endpoint.STALLED)
_DEFLECTED = int(Endpoint.DEFLECTED)

_TWO_PI = 2.0 * math.pi
_TWO_PI_F32 = np.float32(2.0 * math.pi)
_MIN_KEPT_WEIGHT_F32 = np.float32(_MIN_KEPT_WEIGHT)

# Tableau / PI / floor constants baked as device globals (read-only from device code, frozen at
# first compile). float64-contiguous copies so the device reads them without a host round-trip.
_A_D = np.ascontiguousarray(_A, dtype=np.float64)
_B_D = np.ascontiguousarray(_B, dtype=np.float64)
_E_D = np.ascontiguousarray(_E, dtype=np.float64)
_DENSE_P_D = np.ascontiguousarray(_DENSE_P, dtype=np.float64)
# float32 tableau copies for the fully-float32 paint kernel (precision="float32").
_A_F32 = np.ascontiguousarray(_A, dtype=np.float32)
_B_F32 = np.ascontiguousarray(_B, dtype=np.float32)
_E_F32 = np.ascontiguousarray(_E, dtype=np.float32)
_DENSE_P_F32 = np.ascontiguousarray(_DENSE_P, dtype=np.float32)

# Per-launch threads-per-block; each launching wrapper derives its block count from its own
# per-launch line/seed count.
_THREADS_PER_BLOCK = 256

# fastmath is off; NVVM still contracts a*b+c into an FMA regardless.
_FASTMATH = False

#: Kernel precision modes, received as an explicit ``precision`` argument threaded down from
#: :class:`~qorona.config.VolumeConfig` (the CPU tiers ignore it): ``"float64"`` = the all-double
#: reference; ``"mixed"`` = float32 tricubic gather + float32 field upload, float64 stepper /
#: state / accumulators (the production GPU mode); ``"float32"`` = additionally a fully-float32
#: paint kernel (f32 scratch + f32 voxel rasterization, the experimental fast variant). The
#: integrate kernels have no fully-float32 variant, so ``"float32"`` selects their mixed siblings.
_PRECISION_MODES = ("float64", "mixed", "float32")


def _check_precision(precision: str) -> None:
    """Validate a kernel ``precision`` argument (the schema validates the user-facing knob)."""
    if precision not in _PRECISION_MODES:
        raise ValueError(f"precision must be one of {_PRECISION_MODES}, got {precision!r}")


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
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


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
def _keys_weight_derivatives(t: float) -> tuple[float, float, float, float]:
    """The derivatives ``dW/dt`` of the four Keys weights at fractional offset ``t``."""
    t2 = t * t
    return (
        -1.5 * t2 + 2.0 * t - 0.5,
        4.5 * t2 - 5.0 * t,
        -4.5 * t2 + 4.0 * t + 0.5,
        1.5 * t2 - 1.0 * t,
    )


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
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

    ``igrad[d, i] = d value_i / d coord_d`` is the index-space gradient (zeroed/skipped when
    ``gradient`` is ``False``). The 4-point Keys stencil spans offsets ``-1..+2`` about
    ``floor(coord)`` on each axis; callers guarantee the stencil is in range via the ghost padding
    and the radial clip.
    """
    # int() cast is load-bearing on the device: numba CUDA's math.floor returns a float64, which
    # cannot index the padded array (the CPU kernel's int(np.floor(...)) yields an int directly).
    base0 = int(math.floor(c0))  # noqa: RUF046
    base1 = int(math.floor(c1))  # noqa: RUF046
    base2 = int(math.floor(c2))  # noqa: RUF046
    wx = _keys_weights(c0 - base0)
    wy = _keys_weights(c1 - base1)
    wz = _keys_weights(c2 - base2)

    value[0] = 0.0
    value[1] = 0.0
    value[2] = 0.0
    dwx = _keys_weight_derivatives(c0 - base0)
    dwy = _keys_weight_derivatives(c1 - base1)
    dwz = _keys_weight_derivatives(c2 - base2)
    for d in range(3):
        igrad[d, 0] = 0.0
        igrad[d, 1] = 0.0
        igrad[d, 2] = 0.0

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


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
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


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
def _spherical(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Cartesian -> spherical ``(r, theta, phi)``: ``theta`` colatitude, ``phi in [0, 2pi)``."""
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


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
def _coord_jacobian(x: float, y: float, z: float, jac: np.ndarray) -> None:
    """Write ``jac[d, j] = d(r, theta, phi)_d / d x_j``: exact, no polar-axis regularization."""
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


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
def _char_len(
    kind: int,
    char_const: float,
    spacing_code: int,
    r_inner: float,
    r_outer: float,
    exponent: float,
    n_r: int,
    n_theta: int,
    n_phi: int,
    x: float,
    y: float,
    z: float,
) -> float:
    """The local CFL cell metric at a point (grid: smallest cell extent; dipole: constant)."""
    if kind == 1:
        return char_const
    r, theta, _ = _spherical(x, y, z)
    _, dr_du = _radial_parameter_and_derivative(r, spacing_code, r_inner, r_outer, exponent)
    radial = dr_du / (n_r - 1)
    meridional = r * (math.pi / n_theta)
    # Floor the azimuthal arc at the meridional arc: at the pole r*sin(theta)*dphi collapses to zero
    # from meridian convergence, a coordinate artifact the C1-through-pole field has no structure
    # at, so it must not tighten the step below the genuinely-resolved scale (mirrors cell_extent).
    azimuthal = r * math.sin(theta) * (_TWO_PI / n_phi)
    if azimuthal < meridional:
        azimuthal = meridional
    extent = radial
    if meridional < extent:
        extent = meridional
    if azimuthal < extent:
        extent = azimuthal
    return extent


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
def _sample_point(
    kind: int,
    b_padded: np.ndarray,
    n_r: int,
    n_theta: int,
    n_phi: int,
    spacing_code: int,
    r_inner: float,
    r_outer: float,
    exponent: float,
    moment: float,
    background: float,
    x: float,
    y: float,
    z: float,
    gradient: bool,
    b: np.ndarray,
    grad_b: np.ndarray,
) -> float:
    """Evaluate B (and grad B if ``gradient``) at a point; write ``b``/``grad_b``, return |B|.

    Mirrors :meth:`SampledField.sample` (``kind == 0``) and :meth:`PfssDipoleField._evaluate`
    (``kind == 1``), including the gridded radial clip into the Keys in-range band and the
    chain-rule from the index-space gradient to the Cartesian Jacobian ``dB_i/dx_j`` (with the exact
    ``dphi/dx`` factor, no polar-axis floor, so an on-axis point yields a non-finite Jacobian by
    design, exactly as the NumPy path).
    """
    if kind == 0:
        r, theta, phi = _spherical(x, y, z)
        parameter, dr_du = _radial_parameter_and_derivative(
            r, spacing_code, r_inner, r_outer, exponent
        )
        theta_step = math.pi / n_theta
        phi_step = _TWO_PI / n_phi
        c0 = parameter * (n_r - 1) + GHOST
        c1 = theta / theta_step - 0.5 + GHOST
        c2 = phi / phi_step + GHOST
        # Clamp the radial coord into the Keys in-range band so an RK stage overrunning the shell
        # reads edge-extrapolation rather than off the padded array (mirrors SampledField.sample).
        hi = b_padded.shape[0] - 3
        if c0 < 1.0:
            c0 = 1.0
        elif c0 > hi:
            c0 = hi

        igrad = cuda.local.array((3, 3), float64)
        _tricubic_point(b_padded, c0, c1, c2, gradient, b, igrad)

        if gradient:
            d_index_r = (n_r - 1) / dr_du
            d_index_theta = 1.0 / theta_step
            d_index_phi = 1.0 / phi_step
            jac = cuda.local.array((3, 3), float64)
            _coord_jacobian(x, y, z, jac)
            # grad_b[i, j] = sum_d (igrad[d, i] * d_index_d) * jac[d, j]
            for i in range(3):
                for j in range(3):
                    grad_b[i, j] = (
                        igrad[0, i] * d_index_r * jac[0, j]
                        + igrad[1, i] * d_index_theta * jac[1, j]
                        + igrad[2, i] * d_index_phi * jac[2, j]
                    )
        return math.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2])

    # Dipole: B = m (3 z x / r^5 - zhat / r^3) + B0 zhat, regular on the axis.
    r2 = x * x + y * y + z * z
    r3_inv = r2**-1.5
    r5_inv = r2**-2.5
    coef = 3.0 * z * r5_inv
    b[0] = moment * coef * x
    b[1] = moment * coef * y
    b[2] = moment * (coef * z - r3_inv) + background
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


# --- Mixed-precision field evaluation ------------------------------------------------------------
# The single float32 island of the mixed kernel: the tricubic stencil gather (a gradient eval is
# ~64 stencil points x 12 mul-adds). _sample_point_f32 reads a float32 b_padded, accumulates the
# stencil in float32, and casts the result up at the _sample_point boundary, so it returns float64
# b / grad_b / |B| exactly like its f64 twin and everything downstream runs in float64 unchanged
# (see the module docstring for the precision rationale).


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
def _keys_weights_f32(t: float) -> tuple[float, float, float, float]:
    """The four Keys cubic-convolution weights at float32 offset ``t`` (float32 throughout)."""
    t2 = t * t
    t3 = t2 * t
    return (
        float32(-0.5) * t3 + t2 - float32(0.5) * t,
        float32(1.5) * t3 - float32(2.5) * t2 + float32(1.0),
        float32(-1.5) * t3 + float32(2.0) * t2 + float32(0.5) * t,
        float32(0.5) * t3 - float32(0.5) * t2,
    )


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
def _keys_weight_derivatives_f32(t: float) -> tuple[float, float, float, float]:
    """The derivatives ``dW/dt`` of the four Keys weights at float32 offset ``t`` (float32)."""
    t2 = t * t
    return (
        float32(-1.5) * t2 + float32(2.0) * t - float32(0.5),
        float32(4.5) * t2 - float32(5.0) * t,
        float32(-4.5) * t2 + float32(4.0) * t + float32(0.5),
        float32(1.5) * t2 - float32(1.0) * t,
    )


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
def _tricubic_point_f32(
    b_padded: np.ndarray,
    c0: float,
    c1: float,
    c2: float,
    gradient: bool,
    value: np.ndarray,
    igrad: np.ndarray,
) -> None:
    """float32 twin of :func:`_tricubic_point`: f32 ``b_padded`` gather into f32 scratch.

    The base index and the fractional offsets are taken from the float64 coordinates (the cheap
    transcendentals stay f64); only the Keys weights and the 64-point stencil accumulation run in
    float32, against the float32 padded field. ``value[3]`` and ``igrad[3,3]`` are float32 scratch;
    the caller casts them up to float64.
    """
    base0 = int(math.floor(c0))  # noqa: RUF046
    base1 = int(math.floor(c1))  # noqa: RUF046
    base2 = int(math.floor(c2))  # noqa: RUF046
    wx = _keys_weights_f32(float32(c0 - base0))
    wy = _keys_weights_f32(float32(c1 - base1))
    wz = _keys_weights_f32(float32(c2 - base2))

    value[0] = float32(0.0)
    value[1] = float32(0.0)
    value[2] = float32(0.0)
    dwx = _keys_weight_derivatives_f32(float32(c0 - base0))
    dwy = _keys_weight_derivatives_f32(float32(c1 - base1))
    dwz = _keys_weight_derivatives_f32(float32(c2 - base2))
    for d in range(3):
        igrad[d, 0] = float32(0.0)
        igrad[d, 1] = float32(0.0)
        igrad[d, 2] = float32(0.0)

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


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
def _sample_point_f32(
    kind: int,
    b_padded: np.ndarray,
    n_r: int,
    n_theta: int,
    n_phi: int,
    spacing_code: int,
    r_inner: float,
    r_outer: float,
    exponent: float,
    moment: float,
    background: float,
    x: float,
    y: float,
    z: float,
    gradient: bool,
    b: np.ndarray,
    grad_b: np.ndarray,
) -> float:
    """Mixed-precision twin of :func:`_sample_point`: f32 tricubic gather, f64 everything else.

    Same signature and float64 outputs (``b``/``grad_b``/return |B|) as :func:`_sample_point`, so it
    drops into the float64 stepper unchanged. The gridded (``kind == 0``) branch evaluates the
    coordinate map and the Jacobian chain rule in float64 but reads the float32 ``b_padded`` and
    accumulates the stencil in float32 (:func:`_tricubic_point_f32`), casting the f32 value/index
    gradient up to float64 before the chain rule. The dipole (``kind == 1``) branch is the float64
    closed form verbatim (no gather).
    """
    if kind == 0:
        r, theta, phi = _spherical(x, y, z)
        parameter, dr_du = _radial_parameter_and_derivative(
            r, spacing_code, r_inner, r_outer, exponent
        )
        theta_step = math.pi / n_theta
        phi_step = _TWO_PI / n_phi
        c0 = parameter * (n_r - 1) + GHOST
        c1 = theta / theta_step - 0.5 + GHOST
        c2 = phi / phi_step + GHOST
        hi = b_padded.shape[0] - 3
        if c0 < 1.0:
            c0 = 1.0
        elif c0 > hi:
            c0 = hi

        value = cuda.local.array(3, float32)
        igrad = cuda.local.array((3, 3), float32)
        _tricubic_point_f32(b_padded, c0, c1, c2, gradient, value, igrad)
        b[0] = float64(value[0])
        b[1] = float64(value[1])
        b[2] = float64(value[2])

        if gradient:
            d_index_r = (n_r - 1) / dr_du
            d_index_theta = 1.0 / theta_step
            d_index_phi = 1.0 / phi_step
            jac = cuda.local.array((3, 3), float64)
            _coord_jacobian(x, y, z, jac)
            for i in range(3):
                for j in range(3):
                    grad_b[i, j] = (
                        float64(igrad[0, i]) * d_index_r * jac[0, j]
                        + float64(igrad[1, i]) * d_index_theta * jac[1, j]
                        + float64(igrad[2, i]) * d_index_phi * jac[2, j]
                    )
        return math.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2])

    # Dipole: the float64 closed form, unchanged (regular on the axis); no gather to halve.
    r2 = x * x + y * y + z * z
    r3_inv = r2**-1.5
    r5_inv = r2**-2.5
    coef = 3.0 * z * r5_inv
    b[0] = moment * coef * x
    b[1] = moment * coef * y
    b[2] = moment * (coef * z - r3_inv) + background
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


# --- Fully-float32 leaf ops for the float32 paint kernel ---------------------------------------
# precision="float32" pushes the dominant paint sd=3 kernel entirely to float32: the per-step
# scratch AND the voxel rasterization, whose ``_vol_flat`` / ``_vol_cell_extent`` evaluate spherical
# transcendentals (sqrt / acos / atan2 / log) per sub-sample; float32 selects the fast SFU
# intrinsics over consumer GPUs' slow f64 path. These f32 twins mirror their f64 originals; the
# error norm still accumulates in float64 (the accept/reject predicate stays well-conditioned).


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
def _spherical_f32(x: float, y: float, z: float) -> tuple[float, float, float]:
    """float32 :func:`_spherical`: Cartesian -> ``(r, theta, phi)`` via the f32 SFU intrinsics."""
    r = math.sqrt(x * x + y * y + z * z)
    cos_theta = float32(z / r) if r > float32(0.0) else float32(1.0)
    if cos_theta > float32(1.0):
        cos_theta = float32(1.0)
    elif cos_theta < float32(-1.0):
        cos_theta = float32(-1.0)
    theta = math.acos(cos_theta)
    phi = float32(math.atan2(y, x))
    if phi < float32(0.0):
        phi = float32(phi + _TWO_PI_F32)
    return r, theta, phi


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
def _radial_parameter_and_derivative_f32(
    radius: float, spacing_code: int, r_inner: float, r_outer: float, exponent: float
) -> tuple[float, float]:
    """float32 :func:`_radial_parameter_and_derivative` (the radial index map's core)."""
    if spacing_code == 0:  # logarithmic
        log_ratio = math.log(r_outer / r_inner)
        return math.log(radius / r_inner) / log_ratio, radius * log_ratio
    if spacing_code == 1:  # power-law
        inv = float32(1.0) / exponent
        inner_root = r_inner**inv
        outer_root = r_outer**inv
        root = radius**inv
        parameter = (root - inner_root) / (outer_root - inner_root)
        return parameter, exponent * root ** (exponent - float32(1.0)) * (outer_root - inner_root)
    return (radius - r_inner) / (r_outer - r_inner), r_outer - r_inner


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
def _char_len_f32(
    kind: int,
    char_const: float,
    spacing_code: int,
    r_inner: float,
    r_outer: float,
    exponent: float,
    n_r: int,
    n_theta: int,
    n_phi: int,
    x: float,
    y: float,
    z: float,
) -> float:
    """float32 :func:`_char_len`: the local CFL cell metric (grid: smallest cell extent)."""
    if kind == 1:
        return char_const
    r, theta, _ = _spherical_f32(x, y, z)
    _, dr_du = _radial_parameter_and_derivative_f32(r, spacing_code, r_inner, r_outer, exponent)
    radial = dr_du / (n_r - 1)
    meridional = r * (float32(math.pi) / n_theta)
    azimuthal = r * math.sin(theta) * (_TWO_PI_F32 / n_phi)
    if azimuthal < meridional:
        azimuthal = meridional
    extent = radial
    if meridional < extent:
        extent = meridional
    if azimuthal < extent:
        extent = azimuthal
    return extent


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
def _foot_gap_f32(
    state: np.ndarray, coeff: np.ndarray, step: float, theta: float, tsq: float
) -> float:
    """float32 :func:`_foot_gap`: ``|x(theta)|^2 - R^2`` on the dense-output position spline."""
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


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
def _localize_foot_f32(state: np.ndarray, coeff: np.ndarray, step: float, r_target: float) -> float:
    """float32 :func:`_localize_foot`: bisection of ``|x(theta)| = R`` over ``[0, 1]``."""
    tsq = r_target * r_target
    low = float32(0.0)
    high = float32(1.0)
    sign_low = _sign(_foot_gap_f32(state, coeff, step, float32(0.0), tsq))
    for _ in range(80):
        mid = float32(0.5) * (low + high)
        if _sign(_foot_gap_f32(state, coeff, step, mid, tsq)) == sign_low:
            low = mid
        else:
            high = mid
        if high - low < float32(1e-6):
            break
    return float32(0.5) * (low + high)


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
def _eval_rhs_3_full(
    out: np.ndarray,
    state: np.ndarray,
    direction: float,
    kind: int,
    b_padded: np.ndarray,
    n_r: int,
    n_theta: int,
    n_phi: int,
    spacing_code: int,
    r_inner: float,
    r_outer: float,
    exponent: float,
    moment: float,
    background: float,
) -> None:
    """float32 position-only RHS: f32 unit field into ``out[3]`` (value-only f32 tricubic)."""
    b = cuda.local.array(3, float64)
    grad_b = cuda.local.array((3, 3), float64)
    bmag = _sample_point_f32(
        kind,
        b_padded,
        n_r,
        n_theta,
        n_phi,
        spacing_code,
        r_inner,
        r_outer,
        exponent,
        moment,
        background,
        float64(state[0]),
        float64(state[1]),
        float64(state[2]),
        False,
        b,
        grad_b,
    )
    d = float32(direction)
    inv = float32(1.0) / float32(bmag)
    out[0] = d * float32(b[0]) * inv
    out[1] = d * float32(b[1]) * inv
    out[2] = d * float32(b[2]) * inv


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
def _dopri5_step_3_full(
    stages: np.ndarray,
    new_state: np.ndarray,
    error: np.ndarray,
    state: np.ndarray,
    derivative: np.ndarray,
    step: float,
    direction: float,
    kind: int,
    b_padded: np.ndarray,
    n_r: int,
    n_theta: int,
    n_phi: int,
    spacing_code: int,
    r_inner: float,
    r_outer: float,
    exponent: float,
    moment: float,
    background: float,
) -> None:
    """float32 sd=3 DOPRI5 5(4) FSAL step: fill ``stages[7,3]``, ``new_state[3]``, ``error[3]``."""
    tmp = cuda.local.array(3, float32)
    for k in range(3):
        stages[0, k] = derivative[k]
    for stage in range(1, 6):
        for k in range(3):
            inc = float32(0.0)
            for j in range(stage):
                inc += _A_F32[stage, j] * stages[j, k]
            tmp[k] = state[k] + step * inc
        _eval_rhs_3_full(
            stages[stage],
            tmp,
            direction,
            kind,
            b_padded,
            n_r,
            n_theta,
            n_phi,
            spacing_code,
            r_inner,
            r_outer,
            exponent,
            moment,
            background,
        )
    for k in range(3):
        acc = float32(0.0)
        for j in range(6):
            acc += _B_F32[j] * stages[j, k]
        new_state[k] = state[k] + step * acc
    _eval_rhs_3_full(
        stages[6],
        new_state,
        direction,
        kind,
        b_padded,
        n_r,
        n_theta,
        n_phi,
        spacing_code,
        r_inner,
        r_outer,
        exponent,
        moment,
        background,
    )
    for k in range(3):
        acc = float32(0.0)
        for j in range(7):
            acc += _E_F32[j] * stages[j, k]
        error[k] = step * acc


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
def _foot_gap(state: np.ndarray, coeff: np.ndarray, step: float, theta: float, tsq: float) -> float:
    """``|x(theta)|^2 - R^2`` on the dense-output position interpolant (the foot root function)."""
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


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
def _sign(value: float) -> float:
    """Three-way sign, matching ``numpy.sign`` (``-1``/``0``/``+1``)."""
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
def _localize_foot(state: np.ndarray, coeff: np.ndarray, step: float, r_target: float) -> float:
    """Vectorless bisection of ``|x(theta)| = R`` over ``[0, 1]`` (the dense-output foot)."""
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


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
def _tricubic_scalar(vol: np.ndarray, c0: float, c1: float, c2: float) -> float:
    """NaN-tolerant scalar Keys tricubic of a padded volume at one point; returns the value.

    The device twin of :func:`qorona.accel.kernels._tricubic_point_scalar`: gather the 4x4x4
    stencil, skip non-finite taps from both the weighted sum and the kept weight, and return the
    renormalized average; NaN where every tap is non-finite or the kept weight cancels.
    """
    base0 = int(math.floor(c0))  # noqa: RUF046
    base1 = int(math.floor(c1))  # noqa: RUF046
    base2 = int(math.floor(c2))  # noqa: RUF046
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


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
def _tricubic_scalar_f32(vol: np.ndarray, c0: float, c1: float, c2: float) -> float:
    """float32 :func:`_tricubic_scalar`: f32 weights, gather, and accumulation."""
    base0 = int(math.floor(c0))  # noqa: RUF046
    base1 = int(math.floor(c1))  # noqa: RUF046
    base2 = int(math.floor(c2))  # noqa: RUF046
    wx = _keys_weights_f32(float32(c0 - base0))
    wy = _keys_weights_f32(float32(c1 - base1))
    wz = _keys_weights_f32(float32(c2 - base2))
    numerator = float32(0.0)
    weight = float32(0.0)
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
    if abs(weight) > _MIN_KEPT_WEIGHT_F32:
        return numerator / weight
    return math.nan


def _make_integrate_line(sd: int, mixed: bool = False) -> tuple[Any, Any, Any]:
    """Build the ``sd``-specialized ``integrate_line`` device function (``sd`` in {3, 9}).

    ``sd`` is captured as a Python constant so every ``cuda.local.array`` below has a compile-time
    shape. The body is a line-for-line port of :func:`qorona.accel.kernels._integrate_line` (and the
    ``_eval_rhs`` / ``_dopri5_step`` it calls), with ``transport = sd == 9``. The factory also
    returns ``eval_rhs`` and ``dopri5_step`` so the sd=3 paint kernel can reuse them.

    ``mixed`` swaps the float64 :func:`_sample_point` for the float32-tricubic
    :func:`_sample_point_f32`: the field eval reads a float32 ``b_padded`` and
    accumulates the stencil in float32, but returns float64, so the stepper, error norm, and foot
    landing below are byte-identical between the two precisions; only the field eval differs.

    Parameters
    ----------
    sd
        State dimension: ``3`` (position-only) or ``9`` (position + deviation transport).
    mixed
        ``True`` for the mixed-precision variant (float32 tricubic gather); ``False`` (default) for
        the all-float64 variant.

    Returns
    -------
    tuple
        ``(integrate_line, eval_rhs, dopri5_step)`` device functions for the ``(sd, mixed)`` combo.
    """
    transport = sd == 9
    sample_point = _sample_point_f32 if mixed else _sample_point

    @cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
    def eval_rhs(
        out: np.ndarray,
        state: np.ndarray,
        direction: float,
        kind: int,
        b_padded: np.ndarray,
        n_r: int,
        n_theta: int,
        n_phi: int,
        spacing_code: int,
        r_inner: float,
        r_outer: float,
        exponent: float,
        moment: float,
        background: float,
    ) -> None:
        """Write the unit-field RHS into ``out`` (position, +deviation block if ``transport``)."""
        b = cuda.local.array(3, float64)
        grad_b = cuda.local.array((3, 3), float64)
        bmag = sample_point(
            kind,
            b_padded,
            n_r,
            n_theta,
            n_phi,
            spacing_code,
            r_inner,
            r_outer,
            exponent,
            moment,
            background,
            state[0],
            state[1],
            state[2],
            transport,
            b,
            grad_b,
        )
        inv = 1.0 / bmag
        bhx = b[0] * inv
        bhy = b[1] * inv
        bhz = b[2] * inv
        out[0] = direction * bhx
        out[1] = direction * bhy
        out[2] = direction * bhz
        if transport:
            # along[j] = sum_k Bhat_k dB_k/dx_j  (contract Bhat with grad_b's first index).
            along0 = bhx * grad_b[0, 0] + bhy * grad_b[1, 0] + bhz * grad_b[2, 0]
            along1 = bhx * grad_b[0, 1] + bhy * grad_b[1, 1] + bhz * grad_b[2, 1]
            along2 = bhx * grad_b[0, 2] + bhy * grad_b[1, 2] + bhz * grad_b[2, 2]
            ghat = cuda.local.array((3, 3), float64)
            for i in range(3):
                bhi = bhx if i == 0 else (bhy if i == 1 else bhz)
                ghat[i, 0] = (grad_b[i, 0] - bhi * along0) * inv
                ghat[i, 1] = (grad_b[i, 1] - bhi * along1) * inv
                ghat[i, 2] = (grad_b[i, 2] - bhi * along2) * inv
            for i in range(3):
                du = ghat[i, 0] * state[3] + ghat[i, 1] * state[4] + ghat[i, 2] * state[5]
                dv = ghat[i, 0] * state[6] + ghat[i, 1] * state[7] + ghat[i, 2] * state[8]
                out[3 + i] = direction * du
                out[6 + i] = direction * dv

    @cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
    def dopri5_step(
        stages: np.ndarray,
        new_state: np.ndarray,
        error: np.ndarray,
        state: np.ndarray,
        derivative: np.ndarray,
        step: float,
        direction: float,
        kind: int,
        b_padded: np.ndarray,
        n_r: int,
        n_theta: int,
        n_phi: int,
        spacing_code: int,
        r_inner: float,
        r_outer: float,
        exponent: float,
        moment: float,
        background: float,
    ) -> None:
        """One DOPRI5 5(4) step (FSAL): fill ``stages[7,sd]``, ``new_state[sd]``, ``error[sd]``."""
        tmp = cuda.local.array(sd, float64)
        for k in range(sd):
            stages[0, k] = derivative[k]
        for stage in range(1, 6):
            for k in range(sd):
                inc = 0.0
                for j in range(stage):
                    inc += _A_D[stage, j] * stages[j, k]
                tmp[k] = state[k] + step * inc
            eval_rhs(
                stages[stage],
                tmp,
                direction,
                kind,
                b_padded,
                n_r,
                n_theta,
                n_phi,
                spacing_code,
                r_inner,
                r_outer,
                exponent,
                moment,
                background,
            )
        for k in range(sd):
            acc = 0.0
            for j in range(6):
                acc += _B_D[j] * stages[j, k]
            new_state[k] = state[k] + step * acc
        eval_rhs(
            stages[6],
            new_state,
            direction,
            kind,
            b_padded,
            n_r,
            n_theta,
            n_phi,
            spacing_code,
            r_inner,
            r_outer,
            exponent,
            moment,
            background,
        )
        for k in range(sd):
            acc = 0.0
            for j in range(7):
                acc += _E_D[j] * stages[j, k]
            error[k] = step * acc

    @cuda.jit(device=True, fastmath=_FASTMATH)
    def integrate_line(
        state0: np.ndarray,
        direction: float,
        kind: int,
        b_padded: np.ndarray,
        n_r: int,
        n_theta: int,
        n_phi: int,
        spacing_code: int,
        r_inner: float,
        r_outer: float,
        exponent: float,
        moment: float,
        background: float,
        char_const: float,
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

        Writes the landed terminal state into ``out_state`` on a clean foot; leaves it untouched
        (the caller pre-fills ``NaN``) for an aborted line. A line-for-line port of the reference
        adaptive loop: PI step control with the no-growth-after-rejection cap, the CFL ceiling, the
        parameter-free null guard, the stall guard (``max_reversals``), the sharp-turn guard
        (``turn_cos`` / ``turn_radius`` / ``weak_threshold``), inclusive boundary crossing, and
        dense-output foot-landing.
        """
        state = cuda.local.array(sd, float64)
        for k in range(sd):
            state[k] = state0[k]
        derivative = cuda.local.array(sd, float64)
        eval_rhs(
            derivative,
            state,
            direction,
            kind,
            b_padded,
            n_r,
            n_theta,
            n_phi,
            spacing_code,
            r_inner,
            r_outer,
            exponent,
            moment,
            background,
        )

        step = cfl * _char_len(
            kind,
            char_const,
            spacing_code,
            r_inner,
            r_outer,
            exponent,
            n_r,
            n_theta,
            n_phi,
            state[0],
            state[1],
            state[2],
        )
        err_prev = _ERR_PREV_FLOOR
        rejected_last = False
        arc_length = 0.0
        n_reversals = 0
        n_turns = 0

        stages = cuda.local.array((7, sd), float64)
        new_state = cuda.local.array(sd, float64)
        error = cuda.local.array(sd, float64)
        coeff = cuda.local.array((4, sd), float64)
        b_turn = cuda.local.array(3, float64)
        grad_turn = cuda.local.array((3, 3), float64)

        iteration = 0
        while iteration < max_steps:
            finite = True
            for k in range(sd):
                if not math.isfinite(derivative[k]):
                    finite = False
                    break
            if not finite:
                return _NULL, arc_length

            ceiling = cfl * _char_len(
                kind,
                char_const,
                spacing_code,
                r_inner,
                r_outer,
                exponent,
                n_r,
                n_theta,
                n_phi,
                state[0],
                state[1],
                state[2],
            )
            step_now = step if step < ceiling else ceiling

            dopri5_step(
                stages,
                new_state,
                error,
                state,
                derivative,
                step_now,
                direction,
                kind,
                b_padded,
                n_r,
                n_theta,
                n_phi,
                spacing_code,
                r_inner,
                r_outer,
                exponent,
                moment,
                background,
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
                new_state[0] * new_state[0]
                + new_state[1] * new_state[1]
                + new_state[2] * new_state[2]
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
                # Bhat_prev . Bhat_new across the step (the position block of the FSAL derivative,
                # still the previous step's value here, against the new FSAL stage): one dot for
                # both weak-field guards, taken before the FSAL stage is rolled forward.
                if max_reversals > 0 or weak_threshold > 0.0:
                    dot = (
                        derivative[0] * stages[6, 0]
                        + derivative[1] * stages[6, 1]
                        + derivative[2] * stages[6, 2]
                    )
                    # Sharp-turn guard: count sharp turns above turn_radius into field weaker than
                    # weak_threshold, terminating once turn_min of them accumulate: a sustained
                    # staircase, not a single legitimate null graze. |B| is sampled only at the
                    # rare corner passing the cheap turn and radius tests; takes precedence over the
                    # stall count when a step trips both.
                    if weak_threshold > 0.0 and dot < turn_cos and end_radius > turn_radius:
                        bmag = sample_point(
                            kind,
                            b_padded,
                            n_r,
                            n_theta,
                            n_phi,
                            spacing_code,
                            r_inner,
                            r_outer,
                            exponent,
                            moment,
                            background,
                            new_state[0],
                            new_state[1],
                            new_state[2],
                            False,
                            b_turn,
                            grad_turn,
                        )
                        if bmag < weak_threshold:
                            n_turns += 1
                            if n_turns >= turn_min:
                                return _DEFLECTED, arc_length
                    # Stall guard: a >90 deg turn is the thrashing-trap signature; count it.
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
                            acc += stages[s, d] * _DENSE_P_D[s, p]
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

    return integrate_line, eval_rhs, dopri5_step


_integrate_line_3, _eval_rhs_3, _dopri5_step_3 = _make_integrate_line(3)
_integrate_line_9, _eval_rhs_9, _dopri5_step_9 = _make_integrate_line(9)
# Mixed-precision siblings (float32 tricubic gather); selected per call by ``precision``.
_integrate_line_3m, _eval_rhs_3m, _dopri5_step_3m = _make_integrate_line(3, mixed=True)
_integrate_line_9m, _eval_rhs_9m, _dopri5_step_9m = _make_integrate_line(9, mixed=True)


def _make_batch_kernel(integrate_line: Any, sd: int) -> Any:
    """Build the entry ``@cuda.jit`` kernel wrapping an ``sd``-specialized ``integrate_line``.

    One line per thread (``cuda.grid(1)``, guarded ``if i < n`` and the active mask); aborted lines
    keep the pre-filled ``NaN`` terminal state. The same wrapper serves both precisions; the
    float32/float64 split lives entirely in the captured ``integrate_line``.
    """

    @cuda.jit(fastmath=_FASTMATH)
    def kernel(
        state0: np.ndarray,
        active: np.ndarray,
        directions: np.ndarray,
        b_padded: np.ndarray,
        kind: int,
        n_r: int,
        n_theta: int,
        n_phi: int,
        spacing_code: int,
        r_inner: float,
        r_outer: float,
        exponent: float,
        moment: float,
        background: float,
        char_const: float,
        atol: np.ndarray,
        rtol: float,
        cfl: float,
        max_steps: int,
        max_reversals: int,
        turn_cos: float,
        turn_radius: float,
        weak_threshold: float,
        turn_min: int,
        terminal_state: np.ndarray,
        ends: np.ndarray,
        lengths: np.ndarray,
    ) -> None:
        """One line per thread; writes terminal_state/ends/lengths in place."""
        i = cuda.grid(1)
        if i >= state0.shape[0] or not active[i]:
            return
        out = cuda.local.array(sd, float64)
        for k in range(sd):
            out[k] = math.nan
        code, length = integrate_line(
            state0[i],
            directions[i],
            kind,
            b_padded,
            n_r,
            n_theta,
            n_phi,
            spacing_code,
            r_inner,
            r_outer,
            exponent,
            moment,
            background,
            char_const,
            atol,
            rtol,
            cfl,
            max_steps,
            max_reversals,
            turn_cos,
            turn_radius,
            weak_threshold,
            turn_min,
            out,
        )
        for k in range(sd):
            terminal_state[i, k] = out[k]
        ends[i] = code
        lengths[i] = length

    return kernel


_integrate_batch_kernel_3 = _make_batch_kernel(_integrate_line_3, 3)
_integrate_batch_kernel_9 = _make_batch_kernel(_integrate_line_9, 9)
_integrate_batch_kernel_3m = _make_batch_kernel(_integrate_line_3m, 3)
_integrate_batch_kernel_9m = _make_batch_kernel(_integrate_line_9m, 9)


def integrate_batch_cuda(
    state0: np.ndarray,
    active: np.ndarray,
    directions: np.ndarray,
    transport: bool,
    jit_field: JitField,
    atol: np.ndarray,
    rtol: float,
    cfl: float,
    max_steps: int,
    max_reversals: int,
    turn_cos: float,
    turn_radius: float,
    weak_threshold: float,
    turn_min: int,
    precision: str = "mixed",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate a batch of lines on the GPU, one line per thread; the CUDA twin of
    :func:`qorona.accel.kernels.integrate_batch_jit`.

    Unpacks the :class:`~qorona.accel.JitField` (a NamedTuple is not a single device argument),
    uploads the read-only ``b_padded`` once via ``cuda.to_device``, launches the sd=3 or sd=9
    specialization, and copies the terminal state / end codes / arc lengths back. ``store_path`` is
    unsupported (the dispatcher routes it to NumPy). Aborted lines keep a ``NaN`` terminal state.

    Parameters
    ----------
    state0
        ``(n, sd)`` initial states (positions, plus the deviation block when ``transport``).
    active
        ``(n,)`` boolean mask of lines to integrate; inactive rows are not written.
    directions
        ``(n,)`` per-line integration sign (``+1``/``-1``).
    transport
        ``True`` for the sd=9 deviation-transport state, ``False`` for the sd=3 position-only state.
    jit_field
        The :class:`~qorona.accel.JitField` field descriptor (unpacked to device scalars).
    atol
        ``(sd,)`` per-component absolute-error floors.
    rtol, cfl
        Relative-error tolerance and the CFL step-ceiling fraction.
    max_steps, max_reversals, turn_min
        The step budget and the stall / sharp-turn guard counts.
    turn_cos, turn_radius, weak_threshold
        The sharp-turn guard thresholds (cosine, radius, weak-field |B|).
    precision
        Kernel precision: ``"float64"`` (the all-double reference), ``"mixed"`` (default,
        float32 tricubic gather), or ``"float32"`` (selects the mixed integrate kernels; the
        fully-float32 variant exists only for the paint kernel).

    Returns
    -------
    tuple
        ``(terminal_state (n, sd), ends (n,) int8, lengths (n,))``.
    """
    n, sd = state0.shape
    _check_precision(precision)
    # Only the field upload dtype and the kernel choice vary with precision (see the param doc);
    # "float32" shares the mixed integrate kernels (the fully-float32 variant is paint-only).
    mixed = precision != "float64"
    bpad_dtype = np.float32 if mixed else np.float64
    d_state0 = cuda.to_device(np.ascontiguousarray(state0, dtype=np.float64))
    d_active = cuda.to_device(np.ascontiguousarray(active, dtype=np.bool_))
    d_dirs = cuda.to_device(np.ascontiguousarray(directions, dtype=np.float64))
    d_bpad = cuda.to_device(np.ascontiguousarray(jit_field.b_padded, dtype=bpad_dtype))
    d_atol = cuda.to_device(np.ascontiguousarray(atol, dtype=np.float64))
    d_terminal = cuda.device_array((n, sd), dtype=np.float64)
    # Zero-init ends/lengths: skipped lanes are never written by the kernel, matching
    # integrate_batch_jit's np.zeros allocations.
    d_ends = cuda.to_device(np.zeros(n, dtype=np.int8))
    d_lengths = cuda.to_device(np.zeros(n, dtype=np.float64))
    blocks = (n + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
    if transport:
        kernel = _integrate_batch_kernel_9m if mixed else _integrate_batch_kernel_9
    else:
        kernel = _integrate_batch_kernel_3m if mixed else _integrate_batch_kernel_3
    kernel[blocks, _THREADS_PER_BLOCK](
        d_state0,
        d_active,
        d_dirs,
        d_bpad,
        int(jit_field.kind),
        int(jit_field.n_r),
        int(jit_field.n_theta),
        int(jit_field.n_phi),
        int(jit_field.spacing_code),
        float(jit_field.r_inner),
        float(jit_field.r_outer),
        float(jit_field.exponent),
        float(jit_field.moment),
        float(jit_field.background),
        float(jit_field.char_const),
        d_atol,
        float(rtol),
        float(cfl),
        int(max_steps),
        int(max_reversals),
        float(turn_cos),
        float(turn_radius),
        float(weak_threshold),
        int(turn_min),
        d_terminal,
        d_ends,
        d_lengths,
    )
    return (d_terminal.copy_to_host(), d_ends.copy_to_host(), d_lengths.copy_to_host())


# --- Paint kernel ------------------------------------------------------------------------------
# The CUDA twin of the painting Q⊥-volume builder (:func:`qorona.accel.kernels.paint_batch_jit`):
# each GPU thread traces one seed both ways position-only, rasterizing each line's swept path into
# the deduped voxel run of the *volume* grid (a JitGrid distinct from the traced field's JitField).
# Each thread writes only its own output row (no shared array, no atomics in the trace). The runs
# are then either copied to the host (:func:`paint_batch_cuda`, the CPU-scatter contract) or
# scattered on-device by the per-line atomic kernels further below (the production accumulation
# path). Voxel indices are int32, halving the (batch x max_deposits) device memory vs the CPU
# int64. Tracing reuses the sd=3 device closures (_eval_rhs_3 / _dopri5_step_3).


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
def _vol_flat(
    vn_r: int,
    vn_theta: int,
    vn_phi: int,
    vspacing_code: int,
    vr_inner: float,
    vr_outer: float,
    vexponent: float,
    x: float,
    y: float,
    z: float,
) -> int:
    """Return the flat C-order node index of the volume voxel a point falls in (φ wrap, θ/r clip).

    The forward of :meth:`SphericalGrid.index_coordinates` (floor the fractional indices to a cell,
    wrap φ periodically, clamp θ and r into the node range), flattened ``(i_r·n_θ + i_θ)·n_φ + i_φ``
    to match :func:`~qorona.squashing.volume._pack_volume`.
    """
    r, theta, phi = _spherical(x, y, z)
    parameter, _ = _radial_parameter_and_derivative(r, vspacing_code, vr_inner, vr_outer, vexponent)
    r_index = parameter * (vn_r - 1)
    theta_index = theta / (math.pi / vn_theta) - 0.5
    phi_index = phi / (_TWO_PI / vn_phi)

    # int() casts are load-bearing on the device: numba CUDA's math.floor returns a float64, which
    # cannot index/arithmetic as a node index (the CPU kernel's int(np.floor(...)) yields an int).
    i_r = int(math.floor(r_index))  # noqa: RUF046
    if i_r < 0:
        i_r = 0
    elif i_r > vn_r - 1:
        i_r = vn_r - 1
    i_theta = int(math.floor(theta_index))  # noqa: RUF046
    if i_theta < 0:
        i_theta = 0
    elif i_theta > vn_theta - 1:
        i_theta = vn_theta - 1
    i_phi = int(math.floor(phi_index)) % vn_phi  # noqa: RUF046
    return (i_r * vn_theta + i_theta) * vn_phi + i_phi


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
def _vol_cell_extent(
    vn_r: int,
    vn_theta: int,
    vn_phi: int,
    vspacing_code: int,
    vr_inner: float,
    vr_outer: float,
    vexponent: float,
    x: float,
    y: float,
    z: float,
) -> float:
    """Return the smallest local cell extent of the volume grid at a point (the paint-pitch metric).

    Mirrors :meth:`SphericalGrid.cell_extent` (the minimum of the radial, meridional, and
    azimuthal node spacings), so the sub-sample pitch ``paint_step · extent`` keeps consecutive
    deposits inside one cell of one another.
    """
    r, theta, _ = _spherical(x, y, z)
    _, dr_du = _radial_parameter_and_derivative(r, vspacing_code, vr_inner, vr_outer, vexponent)
    radial = dr_du / (vn_r - 1)
    meridional = r * (math.pi / vn_theta)
    # Azimuthal arc floored at the meridional arc: the pole de-singularization (see cell_extent).
    azimuthal = r * math.sin(theta) * (_TWO_PI / vn_phi)
    if azimuthal < meridional:
        azimuthal = meridional
    extent = radial
    if meridional < extent:
        extent = meridional
    if azimuthal < extent:
        extent = azimuthal
    return extent


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
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


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
def _vol_flat_f32(
    vn_r: int,
    vn_theta: int,
    vn_phi: int,
    vspacing_code: int,
    vr_inner: float,
    vr_outer: float,
    vexponent: float,
    x: float,
    y: float,
    z: float,
) -> int:
    """float32 :func:`_vol_flat`: the volume voxel flat index via the f32 spherical SFU path."""
    r, theta, phi = _spherical_f32(x, y, z)
    parameter, _ = _radial_parameter_and_derivative_f32(
        r, vspacing_code, vr_inner, vr_outer, vexponent
    )
    r_index = parameter * (vn_r - 1)
    theta_index = theta / (float32(math.pi) / vn_theta) - float32(0.5)
    phi_index = phi / (_TWO_PI_F32 / vn_phi)

    i_r = int(math.floor(r_index))  # noqa: RUF046
    if i_r < 0:
        i_r = 0
    elif i_r > vn_r - 1:
        i_r = vn_r - 1
    i_theta = int(math.floor(theta_index))  # noqa: RUF046
    if i_theta < 0:
        i_theta = 0
    elif i_theta > vn_theta - 1:
        i_theta = vn_theta - 1
    i_phi = int(math.floor(phi_index)) % vn_phi  # noqa: RUF046
    return (i_r * vn_theta + i_theta) * vn_phi + i_phi


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
def _vol_cell_extent_f32(
    vn_r: int,
    vn_theta: int,
    vn_phi: int,
    vspacing_code: int,
    vr_inner: float,
    vr_outer: float,
    vexponent: float,
    x: float,
    y: float,
    z: float,
) -> float:
    """float32 :func:`_vol_cell_extent`: smallest local cell extent (the paint-pitch metric)."""
    r, theta, _ = _spherical_f32(x, y, z)
    _, dr_du = _radial_parameter_and_derivative_f32(r, vspacing_code, vr_inner, vr_outer, vexponent)
    radial = dr_du / (vn_r - 1)
    meridional = r * (float32(math.pi) / vn_theta)
    azimuthal = r * math.sin(theta) * (_TWO_PI_F32 / vn_phi)
    if azimuthal < meridional:
        azimuthal = meridional
    extent = radial
    if meridional < extent:
        extent = meridional
    if azimuthal < extent:
        extent = azimuthal
    return extent


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
def _dense_position_f32(
    state: np.ndarray, coeff: np.ndarray, step: float, theta: float, out: np.ndarray
) -> None:
    """float32 :func:`_dense_position`: dense-output position at fractional step ``theta``."""
    th2 = theta * theta
    th3 = th2 * theta
    th4 = th3 * theta
    for d in range(3):
        out[d] = state[d] + step * (
            theta * coeff[0, d] + th2 * coeff[1, d] + th3 * coeff[2, d] + th4 * coeff[3, d]
        )


def _make_paint_half_line(
    eval_rhs: Any,
    dopri5_step: Any,
    *,
    real: Any = float64,
    char_len: Any = _char_len,
    localize_foot: Any = _localize_foot,
    vol_cell_extent: Any = _vol_cell_extent,
    vol_flat: Any = _vol_flat,
    dense_position: Any = _dense_position,
    dense_p: np.ndarray = _DENSE_P_D,
) -> Any:
    """Build the paint half-line tracer for one precision's device-function set.

    The float64 and mixed painters share the float64 body (``real=float64``, the f64 leaves), so
    only their captured ``(eval_rhs, dopri5_step)`` differ. The ``float32`` painter
    (``_paint_half_line_full``) instead passes ``real=float32`` and the float32 leaves
    (``_char_len_f32`` / ``_vol_flat_f32`` / ``_vol_cell_extent_f32`` / ``_dense_position_f32`` /
    ``_localize_foot_f32`` / ``_DENSE_P_F32``),
    so the per-step scratch and the voxel rasterization run in float32 (the rasterization's
    spherical transcendentals then hit the fast SFU path). The error norm still accumulates in
    float64 (the per-component error is divided by the f64 ``scale``).
    """

    @cuda.jit(device=True, fastmath=_FASTMATH)
    def paint_half_line(
        seed: np.ndarray,
        direction: float,
        kind: int,
        b_padded: np.ndarray,
        n_r: int,
        n_theta: int,
        n_phi: int,
        spacing_code: int,
        r_inner: float,
        r_outer: float,
        exponent: float,
        moment: float,
        background: float,
        char_const: float,
        vn_r: int,
        vn_theta: int,
        vn_phi: int,
        vspacing_code: int,
        vr_inner: float,
        vr_outer: float,
        vexponent: float,
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

        Mirrors :func:`~qorona.accel.kernels._integrate_line` (``sd = 3``) step for step (same
        DOPRI5 tableau, PI control, CFL ceiling, null guard, inclusive crossing, dense-output
        foot), but on each accepted step
        it sub-samples the dense output at a pitch of ``paint_step`` times the local volume cell and
        appends each newly-entered voxel (deduped against ``last_flat``) to ``out_voxels[count:]``.
        Returns the updated ``(count, last_flat, overflow)``; ``overflow`` is ``True`` if the run
        hit ``max_deposits`` (the line's tail is then dropped). The device port of
        :func:`qorona.accel.kernels._paint_half_line`.
        """
        state = cuda.local.array(3, real)
        for k in range(3):
            state[k] = seed[k]
        derivative = cuda.local.array(3, real)
        eval_rhs(
            derivative,
            state,
            direction,
            kind,
            b_padded,
            n_r,
            n_theta,
            n_phi,
            spacing_code,
            r_inner,
            r_outer,
            exponent,
            moment,
            background,
        )

        step = cfl * char_len(
            kind,
            char_const,
            spacing_code,
            r_inner,
            r_outer,
            exponent,
            n_r,
            n_theta,
            n_phi,
            state[0],
            state[1],
            state[2],
        )
        err_prev = _ERR_PREV_FLOOR
        rejected_last = False

        stages = cuda.local.array((7, 3), real)
        new_state = cuda.local.array(3, real)
        error = cuda.local.array(3, real)
        coeff = cuda.local.array((4, 3), real)
        point = cuda.local.array(3, real)

        iteration = 0
        while iteration < max_steps:
            finite = True
            for k in range(3):
                if not math.isfinite(derivative[k]):
                    finite = False
                    break
            if not finite:
                return count, last_flat, False

            ceiling = cfl * char_len(
                kind,
                char_const,
                spacing_code,
                r_inner,
                r_outer,
                exponent,
                n_r,
                n_theta,
                n_phi,
                state[0],
                state[1],
                state[2],
            )
            step_now = step if step < ceiling else ceiling
            dopri5_step(
                stages,
                new_state,
                error,
                state,
                derivative,
                step_now,
                direction,
                kind,
                b_padded,
                n_r,
                n_theta,
                n_phi,
                spacing_code,
                r_inner,
                r_outer,
                exponent,
                moment,
                background,
            )

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
                new_state[0] * new_state[0]
                + new_state[1] * new_state[1]
                + new_state[2] * new_state[2]
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
                        acc = real(0.0)
                        for s in range(7):
                            acc += stages[s, d] * dense_p[s, p]
                        coeff[p, d] = acc

                theta_end = real(1.0)
                if crossed:
                    theta_end = localize_foot(
                        state, coeff, step_now, r_outer if crossed_outer else r_inner
                    )

                # Pitch from the smaller of the step's start and end cell extents, so a step over a
                # region where the volume cell shrinks (toward the inner boundary or a pole) still
                # sub-samples finely enough not to skip a voxel.
                dense_position(state, coeff, step_now, theta_end, point)
                extent = vol_cell_extent(
                    vn_r,
                    vn_theta,
                    vn_phi,
                    vspacing_code,
                    vr_inner,
                    vr_outer,
                    vexponent,
                    state[0],
                    state[1],
                    state[2],
                )
                extent_end = vol_cell_extent(
                    vn_r,
                    vn_theta,
                    vn_phi,
                    vspacing_code,
                    vr_inner,
                    vr_outer,
                    vexponent,
                    point[0],
                    point[1],
                    point[2],
                )
                if extent_end < extent:
                    extent = extent_end
                pitch = paint_step * extent
                arc = theta_end * step_now
                n_sub = int(arc / pitch) + 1 if pitch > 0.0 else 1
                for j in range(1, n_sub + 1):
                    dense_position(state, coeff, step_now, (j / n_sub) * theta_end, point)
                    flat = vol_flat(
                        vn_r,
                        vn_theta,
                        vn_phi,
                        vspacing_code,
                        vr_inner,
                        vr_outer,
                        vexponent,
                        point[0],
                        point[1],
                        point[2],
                    )
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

    return paint_half_line


_paint_half_line = _make_paint_half_line(_eval_rhs_3, _dopri5_step_3)
_paint_half_line_m = _make_paint_half_line(_eval_rhs_3m, _dopri5_step_3m)
# Fully-float32 painter (precision="float32"): f32 scratch + f32 voxel rasterization.
_paint_half_line_full = _make_paint_half_line(
    _eval_rhs_3_full,
    _dopri5_step_3_full,
    real=float32,
    char_len=_char_len_f32,
    localize_foot=_localize_foot_f32,
    vol_cell_extent=_vol_cell_extent_f32,
    vol_flat=_vol_flat_f32,
    dense_position=_dense_position_f32,
    dense_p=_DENSE_P_F32,
)


def _make_paint_batch_kernel(paint_half_line: Any) -> Any:
    """Build the paint entry ``@cuda.jit`` kernel wrapping one precision's ``paint_half_line``."""

    @cuda.jit(fastmath=_FASTMATH)
    def kernel(
        seeds: np.ndarray,
        valid: np.ndarray,
        b_padded: np.ndarray,
        kind: int,
        n_r: int,
        n_theta: int,
        n_phi: int,
        spacing_code: int,
        r_inner: float,
        r_outer: float,
        exponent: float,
        moment: float,
        background: float,
        char_const: float,
        vn_r: int,
        vn_theta: int,
        vn_phi: int,
        vspacing_code: int,
        vr_inner: float,
        vr_outer: float,
        vexponent: float,
        atol: np.ndarray,
        rtol: float,
        cfl: float,
        max_steps: int,
        paint_step: float,
        max_deposits: int,
        voxels: np.ndarray,
        counts: np.ndarray,
        overflow: np.ndarray,
    ) -> None:
        """Trace each valid seed both ways and record its painted voxel run; the CUDA port of
        :func:`~qorona.accel.kernels.paint_batch_jit`.

        One seed per thread, writing only its own ``voxels[i]`` row (no atomics): the seed voxel
        first, then the two half-lines via the captured ``paint_half_line``.
        """
        i = cuda.grid(1)
        if i >= seeds.shape[0] or not valid[i]:
            return
        seed = seeds[i]
        seed_flat = _vol_flat(
            vn_r,
            vn_theta,
            vn_phi,
            vspacing_code,
            vr_inner,
            vr_outer,
            vexponent,
            seed[0],
            seed[1],
            seed[2],
        )
        voxels[i, 0] = seed_flat
        count = 1
        # Both half-lines start at the seed voxel, so each dedups its first samples against it; the
        # seed is therefore deposited exactly once. The two halves diverge to opposite feet, so no
        # cross-half dedup is needed.
        count, _, overflow_back = paint_half_line(
            seed,
            -1.0,
            kind,
            b_padded,
            n_r,
            n_theta,
            n_phi,
            spacing_code,
            r_inner,
            r_outer,
            exponent,
            moment,
            background,
            char_const,
            vn_r,
            vn_theta,
            vn_phi,
            vspacing_code,
            vr_inner,
            vr_outer,
            vexponent,
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
            count, _, overflow_forward = paint_half_line(
                seed,
                1.0,
                kind,
                b_padded,
                n_r,
                n_theta,
                n_phi,
                spacing_code,
                r_inner,
                r_outer,
                exponent,
                moment,
                background,
                char_const,
                vn_r,
                vn_theta,
                vn_phi,
                vspacing_code,
                vr_inner,
                vr_outer,
                vexponent,
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

    return kernel


_paint_batch_kernel = _make_paint_batch_kernel(_paint_half_line)
_paint_batch_kernel_m = _make_paint_batch_kernel(_paint_half_line_m)
_paint_batch_kernel_full = _make_paint_batch_kernel(_paint_half_line_full)


def _select_paint_kernel(precision: str) -> Any:
    """Return the paint kernel variant for a validated ``precision``: ``float64`` = all-double,
    ``mixed`` = f32 tricubic with f64 scratch/rasterization, ``float32`` = the fully-float32
    painter."""
    _check_precision(precision)
    if precision == "float32":
        return _paint_batch_kernel_full
    if precision == "mixed":
        return _paint_batch_kernel_m
    return _paint_batch_kernel


def _launch_paint(
    seeds: np.ndarray,
    valid: np.ndarray,
    jit_field: JitField,
    jit_grid: JitGrid,
    atol: np.ndarray,
    rtol: float,
    cfl: float,
    max_steps: int,
    paint_step: float,
    max_deposits: int,
    precision: str,
    d_bpad: Any,
) -> tuple[Any, Any, Any, int]:
    """Stage a chunk's inputs and launch the paint kernel: the shared trace half of the painters.

    Uploads the seeds, validity mask, and ``atol``; stages the field at the precision's dtype when
    ``d_bpad`` was not pre-staged by the caller (it normally is, once, across all chunks/tiles);
    and launches the ``precision``-selected paint kernel, one thread per seed. ``counts`` and
    ``overflow`` are zero-initialized because the kernel early-returns for invalid lines without
    writing their rows. Returns the device ``(voxels, counts, overflow)`` and the launch block
    count for follow-on per-line kernels over the same batch.
    """
    n = seeds.shape[0]
    d_seeds = cuda.to_device(np.ascontiguousarray(seeds, dtype=np.float64))
    d_valid = cuda.to_device(np.ascontiguousarray(valid, dtype=np.bool_))
    if d_bpad is None:
        bpad_dtype = np.float32 if precision != "float64" else np.float64
        d_bpad = cuda.to_device(np.ascontiguousarray(jit_field.b_padded, dtype=bpad_dtype))
    d_atol = cuda.to_device(np.ascontiguousarray(atol, dtype=np.float64))
    d_voxels = cuda.device_array((n, max_deposits), dtype=np.int32)
    d_counts = cuda.to_device(np.zeros(n, dtype=np.int64))
    d_overflow = cuda.to_device(np.zeros(n, dtype=np.bool_))
    blocks = (n + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
    paint_kernel = _select_paint_kernel(precision)
    paint_kernel[blocks, _THREADS_PER_BLOCK](
        d_seeds,
        d_valid,
        d_bpad,
        int(jit_field.kind),
        int(jit_field.n_r),
        int(jit_field.n_theta),
        int(jit_field.n_phi),
        int(jit_field.spacing_code),
        float(jit_field.r_inner),
        float(jit_field.r_outer),
        float(jit_field.exponent),
        float(jit_field.moment),
        float(jit_field.background),
        float(jit_field.char_const),
        int(jit_grid.n_r),
        int(jit_grid.n_theta),
        int(jit_grid.n_phi),
        int(jit_grid.spacing_code),
        float(jit_grid.r_inner),
        float(jit_grid.r_outer),
        float(jit_grid.exponent),
        d_atol,
        float(rtol),
        float(cfl),
        int(max_steps),
        float(paint_step),
        int(max_deposits),
        d_voxels,
        d_counts,
        d_overflow,
    )
    return d_voxels, d_counts, d_overflow, blocks


def paint_batch_cuda(
    seeds: np.ndarray,
    valid: np.ndarray,
    jit_field: JitField,
    jit_grid: JitGrid,
    atol: np.ndarray,
    rtol: float,
    cfl: float,
    max_steps: int,
    paint_step: float,
    max_deposits: int,
    precision: str = "mixed",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trace and paint a batch of seeded lines on the GPU; the CUDA twin of
    :func:`qorona.accel.kernels.paint_batch_jit`.

    Unpacks the :class:`~qorona.accel.JitField` and the volume :class:`~qorona.accel.JitGrid` (a
    NamedTuple is not a single device argument), uploads the read-only ``b_padded`` and seeds via
    ``cuda.to_device``, launches one thread per seed, and copies the per-line voxel runs back. The
    voxel indices are ``int32`` (halving the ``(n, max_deposits)`` device buffer vs the CPU
    ``int64``); the host scatter (``np.add.at``) is dtype-agnostic, so the contract is unchanged.

    Parameters
    ----------
    seeds
        ``(n, 3)`` Cartesian seed positions (one field line per seed).
    valid
        ``(n,)`` boolean mask of seeds with a defined Q⊥; invalid rows are skipped (count ``0``).
    jit_field
        The traced field's :class:`~qorona.accel.JitField` descriptor (unpacked to device scalars).
    jit_grid
        The volume :class:`~qorona.accel.JitGrid` the swept path is binned into.
    atol
        ``(3,)`` per-component absolute-error floors.
    rtol, cfl
        Relative-error tolerance and the CFL step-ceiling fraction.
    max_steps
        The per-half-line step budget.
    paint_step
        Sub-sample pitch as a fraction of the local volume cell extent.
    max_deposits
        The per-line voxel-run cap; a line exceeding it sets its ``overflow`` flag.
    precision
        Kernel precision: ``"float64"`` (all-double), ``"mixed"`` (default, float32 tricubic gather,
        float64 scratch/rasterization), or ``"float32"`` (the fully-float32 painter).

    Returns
    -------
    tuple
        ``(voxels (n, max_deposits) int32, counts (n,) int64, overflow (n,) bool)``: with the same
        shapes/semantics as :func:`paint_batch_jit`, so the host reads ``voxels[i, :counts[i]]``.
    """
    d_voxels, d_counts, d_overflow, _ = _launch_paint(
        seeds,
        valid,
        jit_field,
        jit_grid,
        atol,
        rtol,
        cfl,
        max_steps,
        paint_step,
        max_deposits,
        precision,
        None,
    )
    return (d_voxels.copy_to_host(), d_counts.copy_to_host(), d_overflow.copy_to_host())


@cuda.jit
def _scatter_batch_kernel(
    voxels: np.ndarray,
    counts: np.ndarray,
    line_q: np.ndarray,
    line_pol: np.ndarray,
    sum_q: np.ndarray,
    count: np.ndarray,
    sum_pol: np.ndarray,
    tile_lo: int,
    tile_hi: int,
) -> None:
    """One thread per line: atomic-add the line's value into each swept voxel of the current tile.

    The device twin of the host ``np.add.at`` scatter (``squashing.volume._paint_lines_jit``). For
    each of a line's ``counts[i]`` deduped swept voxels (flat indices in ``voxels[i, :counts[i]]``)
    it adds the line's linear-Q⊥ ``line_q[i]`` to ``sum_q``, ``1.0`` to ``count``, and the signed
    ``line_pol[i]`` to ``sum_pol``, but only for voxels in the ``[tile_lo, tile_hi)`` flat-index
    range, written at the tile-local offset ``vox - tile_lo`` (so the accumulators size to one tile,
    not the whole volume). Invalid lines carry ``counts[i] == 0`` and contribute nothing.
    """
    i = cuda.grid(1)
    if i >= voxels.shape[0]:
        return
    c = counts[i]
    q = line_q[i]
    pq = line_pol[i]
    for j in range(c):
        vox = voxels[i, j]
        if tile_lo <= vox < tile_hi:
            idx = vox - tile_lo
            cuda.atomic.add(sum_q, idx, q)
            cuda.atomic.add(count, idx, 1.0)
            cuda.atomic.add(sum_pol, idx, pq)


def paint_and_scatter_batch_cuda(
    seeds: np.ndarray,
    valid: np.ndarray,
    line_q: np.ndarray,
    line_pol: np.ndarray,
    jit_field: JitField,
    jit_grid: JitGrid,
    atol: np.ndarray,
    rtol: float,
    cfl: float,
    max_steps: int,
    paint_step: float,
    max_deposits: int,
    d_sum_q: Any,
    d_count: Any,
    d_sum_pol: Any,
    tile_lo: int,
    tile_hi: int,
    d_bpad: Any = None,
    precision: str = "mixed",
) -> int:
    """Trace+paint a batch of seeds and scatter their values into device accumulators, on-device.

    The high-res accumulation path: identical trace to :func:`paint_batch_cuda`, but instead of
    copying the ``(n, max_deposits)`` voxel runs back to
    the host for a serial ``np.add.at``, the swept voxels stay on the device and a per-line atomic
    scatter (:func:`_scatter_batch_kernel`) accumulates each line's precomputed ``line_q`` / ``1.0``
    / ``line_pol`` into the caller's persistent device ``sum_q`` / ``count`` / ``sum_pol`` for the
    ``[tile_lo, tile_hi)`` voxel tile. No host round-trip, no serial scatter. Returns the overflow
    line count (for the per-line voxel-cap warning).
    """
    d_voxels, d_counts, d_overflow, blocks = _launch_paint(
        seeds,
        valid,
        jit_field,
        jit_grid,
        atol,
        rtol,
        cfl,
        max_steps,
        paint_step,
        max_deposits,
        precision,
        d_bpad,
    )
    d_line_q = cuda.to_device(np.ascontiguousarray(line_q, dtype=np.float64))
    d_line_pol = cuda.to_device(np.ascontiguousarray(line_pol, dtype=np.float64))
    _scatter_batch_kernel[blocks, _THREADS_PER_BLOCK](
        d_voxels,
        d_counts,
        d_line_q,
        d_line_pol,
        d_sum_q,
        d_count,
        d_sum_pol,
        np.int64(tile_lo),
        np.int64(tile_hi),
    )
    return int(d_overflow.copy_to_host().sum())


@cuda.jit
def _compact_runs_kernel(
    voxels: np.ndarray,
    counts: np.ndarray,
    offsets: np.ndarray,
    flat: np.ndarray,
) -> None:
    """One thread per line: copy its deduped voxel run into the flat deposit stream at its offset.

    Compaction squeezes the sparse ``(n, max_deposits)`` voxel matrix into ``flat`` (total-deposit
    int32), so a multi-tile paint can keep every chunk's runs on the host and re-scatter later
    tiles without re-tracing.
    """
    i = cuda.grid(1)
    if i >= voxels.shape[0]:
        return
    base = offsets[i]
    for j in range(counts[i]):
        flat[base + j] = voxels[i, j]


@cuda.jit
def _scatter_runs_kernel(
    flat: np.ndarray,
    offsets: np.ndarray,
    line_q: np.ndarray,
    line_pol: np.ndarray,
    sum_q: np.ndarray,
    count: np.ndarray,
    sum_pol: np.ndarray,
    tile_lo: int,
    tile_hi: int,
) -> None:
    """One thread per line: atomic-add its value over its compacted run for the current tile.

    The replay twin of :func:`_scatter_batch_kernel`, reading the flat deposit stream (line ``i``'s
    run is ``flat[offsets[i]:offsets[i + 1]]``) instead of the per-line voxel matrix; semantics are
    otherwise identical, including the ``[tile_lo, tile_hi)`` in-range filter and tile-local offset.
    """
    i = cuda.grid(1)
    if i >= offsets.shape[0] - 1:
        return
    q = line_q[i]
    pq = line_pol[i]
    for j in range(offsets[i], offsets[i + 1]):
        vox = flat[j]
        if tile_lo <= vox < tile_hi:
            idx = vox - tile_lo
            cuda.atomic.add(sum_q, idx, q)
            cuda.atomic.add(count, idx, 1.0)
            cuda.atomic.add(sum_pol, idx, pq)


def paint_scatter_collect_batch_cuda(
    seeds: np.ndarray,
    valid: np.ndarray,
    line_q: np.ndarray,
    line_pol: np.ndarray,
    jit_field: JitField,
    jit_grid: JitGrid,
    atol: np.ndarray,
    rtol: float,
    cfl: float,
    max_steps: int,
    paint_step: float,
    max_deposits: int,
    d_sum_q: Any,
    d_count: Any,
    d_sum_pol: Any,
    tile_lo: int,
    tile_hi: int,
    d_bpad: Any = None,
    precision: str = "mixed",
) -> tuple[int, np.ndarray, np.ndarray]:
    """Trace+scatter one chunk like :func:`paint_and_scatter_batch_cuda`, also returning its runs.

    The first-tile pass of a multi-tile paint: identical trace and tile scatter, plus an on-device
    compaction (:func:`_compact_runs_kernel`) of the chunk's voxel runs into a flat int32 stream
    copied to the host, so later tiles replay the scatter (:func:`scatter_runs_cuda`) without
    re-tracing.

    Returns
    -------
    tuple
        ``(overflow_lines, flat (total,) int32, offsets (n + 1,) int64)``: line ``i``'s run is
        ``flat[offsets[i]:offsets[i + 1]]``.
    """
    d_voxels, d_counts, d_overflow, blocks = _launch_paint(
        seeds,
        valid,
        jit_field,
        jit_grid,
        atol,
        rtol,
        cfl,
        max_steps,
        paint_step,
        max_deposits,
        precision,
        d_bpad,
    )
    d_line_q = cuda.to_device(np.ascontiguousarray(line_q, dtype=np.float64))
    d_line_pol = cuda.to_device(np.ascontiguousarray(line_pol, dtype=np.float64))
    _scatter_batch_kernel[blocks, _THREADS_PER_BLOCK](
        d_voxels,
        d_counts,
        d_line_q,
        d_line_pol,
        d_sum_q,
        d_count,
        d_sum_pol,
        np.int64(tile_lo),
        np.int64(tile_hi),
    )
    counts = d_counts.copy_to_host()
    offsets = np.zeros(counts.shape[0] + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    d_offsets = cuda.to_device(offsets)
    d_flat = cuda.device_array(max(1, int(offsets[-1])), dtype=np.int32)
    _compact_runs_kernel[blocks, _THREADS_PER_BLOCK](d_voxels, d_counts, d_offsets, d_flat)
    flat = d_flat.copy_to_host()[: int(offsets[-1])]
    return int(d_overflow.copy_to_host().sum()), flat, offsets


def scatter_runs_cuda(
    flat: np.ndarray,
    offsets: np.ndarray,
    line_q: np.ndarray,
    line_pol: np.ndarray,
    d_sum_q: Any,
    d_count: Any,
    d_sum_pol: Any,
    tile_lo: int,
    tile_hi: int,
) -> None:
    """Replay a chunk's compacted voxel runs into the current tile's device accumulators.

    Uploads the host-stored ``flat``/``offsets`` (from :func:`paint_scatter_collect_batch_cuda`)
    and launches :func:`_scatter_runs_kernel`; the chunk is not re-traced.
    """
    n = offsets.shape[0] - 1
    d_flat = cuda.to_device(np.ascontiguousarray(flat, dtype=np.int32))
    d_offsets = cuda.to_device(np.ascontiguousarray(offsets, dtype=np.int64))
    d_line_q = cuda.to_device(np.ascontiguousarray(line_q, dtype=np.float64))
    d_line_pol = cuda.to_device(np.ascontiguousarray(line_pol, dtype=np.float64))
    blocks = (n + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
    _scatter_runs_kernel[blocks, _THREADS_PER_BLOCK](
        d_flat,
        d_offsets,
        d_line_q,
        d_line_pol,
        d_sum_q,
        d_count,
        d_sum_pol,
        np.int64(tile_lo),
        np.int64(tile_hi),
    )


# --- Render kernel -------------------------------------------------------------------------------
# The CUDA twin of qorona.accel.kernels.render_batch_jit: one ray per GPU thread marching the
# shared s-grid through the padded log10 Q_perp volume. Two precision tiers from one factory:
# float64 is the all-double reference; mixed (the default, "float32" aliases it) runs every
# per-sample quantity (spherical map, radial index map, tricubic gather, channel weights) in
# float32 through the SFU intrinsics and keeps the per-ray accumulators and reductions float64.
# Each thread writes only its own output rows (no atomics); the host reduces the per-ray clamp
# counts, exactly as the CPU kernel's caller does.


@cuda.jit(device=True, inline=True, fastmath=_FASTMATH)
def _render_coeff_lookup(
    r: float,
    coeff_log_inner: float,
    coeff_inv_dlog: float,
    c_tan_table: np.ndarray,
    c_pol_table: np.ndarray,
) -> tuple[float, float]:
    """Return ``(c_tan, c_pol)`` at radius ``r``: the device twin of the CPU ``_coeff_lookup``."""
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


def _make_render_kernel(mixed: bool) -> Any:
    """Build one precision tier's render entry kernel (``mixed``: f32 sampling, f64 accumulation).

    The body is a line-for-line port of :func:`qorona.accel.kernels.render_batch_jit`'s per-ray
    loop. ``mixed`` swaps in the f32 spherical / index-map / tricubic leaf ops and evaluates the
    channel weights in f32 (every scalar and literal the f32 expressions touch is ``to_real``-cast,
    else numba's promotion rules would silently lift them back to f64); the accumulators, the
    weighted averages, and the Thomson coefficient lookup stay float64 in both tiers, so the two
    kernels differ only in per-sample rounding.
    """
    if mixed:
        spherical = _spherical_f32
        radial_map = _radial_parameter_and_derivative_f32
        tricubic_scalar = _tricubic_scalar_f32
        to_real = float32
    else:
        spherical = _spherical
        radial_map = _radial_parameter_and_derivative
        tricubic_scalar = _tricubic_scalar
        to_real = float64

    @cuda.jit(fastmath=_FASTMATH)
    def kernel(
        origins: np.ndarray,
        look: np.ndarray,
        impact: np.ndarray,
        s_grid: np.ndarray,
        vol: np.ndarray,
        vn_r: int,
        vn_theta: int,
        vn_phi: int,
        vspacing_code: int,
        vr_inner: float,
        vr_outer: float,
        vexponent: float,
        sigma: float,
        use_sigma: bool,
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
        dn_r: int,
        dn_theta: int,
        dn_phi: int,
        dspacing_code: int,
        dr_inner: float,
        dr_outer: float,
        dexponent: float,
        thomson_pb: bool,
        coeff_log_inner: float,
        coeff_inv_dlog: float,
        c_tan_table: np.ndarray,
        c_pol_table: np.ndarray,
        use_polarity: bool,
        polarity_vol: np.ndarray,
        ray_start: int,
        ray_stop: int,
        signal: np.ndarray,
        coverage: np.ndarray,
        counts: np.ndarray,
        den: np.ndarray,
        onpath: np.ndarray,
        polarity: np.ndarray,
    ) -> None:
        """One ray per thread over ``[ray_start, ray_stop)``; writes the six output rows."""
        i = ray_start + cuda.grid(1)
        if i >= ray_stop:
            return

        # Per-ray scalars in the tier's working precision; the identity cast in the f64 tier.
        ox = to_real(origins[i, 0])
        oy = to_real(origins[i, 1])
        oz = to_real(origins[i, 2])
        lx = to_real(look[0])
        ly = to_real(look[1])
        lz = to_real(look[2])
        rho = to_real(impact[i])
        r_in = to_real(vr_inner)
        r_out = to_real(vr_outer)
        vexp = to_real(vexponent)
        occ_r = to_real(r_occult)
        occulting = occult_body and rho < occ_r
        s_body = math.sqrt(occ_r * occ_r - rho * rho) if occulting else to_real(0.0)

        sig = to_real(sigma)
        p0 = to_real(powers[0])
        p1 = to_real(powers[1])
        p2 = to_real(powers[2])
        sc0 = to_real(scales[0])
        sc1 = to_real(scales[1])
        sc2 = to_real(scales[2])
        one = to_real(1.0)
        half = to_real(0.5)
        neg_half = to_real(-0.5)
        ghost = to_real(GHOST)
        vnr1 = to_real(vn_r - 1)
        theta_step = to_real(math.pi / vn_theta)
        phi_step = to_real(_TWO_PI / vn_phi)
        d_in = to_real(dr_inner)
        d_out = to_real(dr_outer)
        dexp = to_real(dexponent)
        dnr1 = to_real(dn_r - 1)
        density_theta_step = to_real(math.pi / dn_theta)
        density_phi_step = to_real(_TWO_PI / dn_phi)

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
        n_steps = s_grid.shape[0]
        for k in range(n_steps):
            s = to_real(s_grid[k])
            r = math.sqrt(rho * rho + s * s)
            if r < r_in or r > r_out:
                continue
            if occulting and s < -s_body:
                continue

            w0 = one
            w1 = one
            w2 = one
            if use_sigma:
                gaussian = math.exp(neg_half * (s / (r * sig)) ** 2)
                w0 *= gaussian
                w1 *= gaussian
                w2 *= gaussian
            if use_powers:
                w0 *= r ** (-p0)
                w1 *= r ** (-p1)
                w2 *= r ** (-p2)
            if use_scales:
                w0 *= math.exp(-r / sc0)
                w1 *= math.exp(-r / sc1)
                w2 *= math.exp(-r / sc2)
            fw0 = float64(w0)
            fw1 = float64(w1)
            fw2 = float64(w2)
            path_weight += (fw0 + fw1 + fw2) / 3.0
            pden0 += fw0
            pden1 += fw1
            pden2 += fw2

            px = ox + s * lx
            py = oy + s * ly
            pz = oz + s * lz
            r_point, theta, phi = spherical(px, py, pz)
            # Mirror QPerpVolume.sample: the point-radius shell mask (NaN outside) and the same
            # index map.
            if r_point < r_in or r_point > r_out:
                continue
            parameter, _ = radial_map(r_point, vspacing_code, r_in, r_out, vexp)
            c0 = parameter * vnr1 + ghost
            c1 = theta / theta_step - half + ghost
            c2 = phi / phi_step + ghost
            value = float64(tricubic_scalar(vol, c0, c1, c2))
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

            # The optional Thomson scalar Ne * I(r, chi): folded into the average only (num/den),
            # the geometric path/onpath/valid budgets above stay scalar-free. Off => an exact 1.0
            # no-op. The coefficient lookup and intensity combination stay float64 in both tiers.
            thomson = 1.0
            if use_thomson:
                d_parameter, _ = radial_map(r_point, dspacing_code, d_in, d_out, dexp)
                d0 = d_parameter * dnr1 + ghost
                d1 = theta / density_theta_step - half + ghost
                d2 = phi / density_phi_step + ghost
                density = float64(tricubic_scalar(density_vol, d0, d1, d2))
                c_tan, c_pol = _render_coeff_lookup(
                    float64(r), coeff_log_inner, coeff_inv_dlog, c_tan_table, c_pol_table
                )
                sin_sq_chi = float64(rho) * float64(rho) / (float64(r) * float64(r))
                if thomson_pb:
                    intensity = c_pol * sin_sq_chi
                else:
                    intensity = 2.0 * c_tan - c_pol * sin_sq_chi
                thomson = density * intensity

            num0 += fw0 * thomson * value
            num1 += fw1 * thomson * value
            num2 += fw2 * thomson * value
            den0 += fw0 * thomson
            den1 += fw1 * thomson
            den2 += fw2 * thomson
            wbar = (fw0 + fw1 + fw2) / 3.0
            valid_weight += wbar
            # Net polarity: the per-voxel footpoint sign read nearest-cell at this same sample,
            # weighted by wbar, the coverage weight budget.
            if use_polarity:
                pic0 = int(c0 + half)
                pic1 = int(c1 + half)
                pic2 = int(c2 + half)
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
                pol_num += wbar * float64(polarity_vol[pic0, pic1, pic2, 0])

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

    return kernel


_render_kernel_f64 = _make_render_kernel(mixed=False)
_render_kernel_mixed = _make_render_kernel(mixed=True)


def render_batch_cuda(
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
    precision: str = "mixed",
    chunks: int = 1,
    progress: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Render a batch of rays on the GPU: the CUDA twin of
    :func:`qorona.accel.kernels.render_batch_jit`.

    Uploads the read-only arrays once (the volume in the precision tier's dtype: float32 for
    ``mixed``/``"float32"``, float64 for the reference), then launches the kernel over ``chunks``
    consecutive ray ranges, synchronizing after each and calling ``progress(rays_done)`` so the
    caller's progress bar stays live; the outputs are copied back once at the end. With
    ``use_thomson`` off, ``density_vol`` is the caller's placeholder (the volume itself) and is
    aliased to the already-uploaded volume rather than uploaded twice. No tiling: the volume must
    fit free VRAM in one piece (a 336M-voxel float32 volume is ~1.4 GB).

    Same argument set and return tuple as the CPU kernel (``signal (n, 3)``, ``coverage (n,)``,
    ``counts (n, 3)``, ``den (n, 3)``, ``onpath (n, 3)``, ``polarity (n,)``), plus
    ``precision``/``chunks``/``progress``.
    """
    _check_precision(precision)
    mixed = precision != "float64"
    vol_dtype = np.float32 if mixed else np.float64
    n = origins.shape[0]

    d_origins = cuda.to_device(np.ascontiguousarray(origins, dtype=np.float64))
    d_look = cuda.to_device(np.ascontiguousarray(look, dtype=np.float64))
    d_impact = cuda.to_device(np.ascontiguousarray(impact, dtype=np.float64))
    d_s = cuda.to_device(np.ascontiguousarray(s_grid, dtype=np.float64))
    d_vol = cuda.to_device(np.ascontiguousarray(vol, dtype=vol_dtype))
    d_density = (
        cuda.to_device(np.ascontiguousarray(density_vol, dtype=vol_dtype)) if use_thomson else d_vol
    )
    d_powers = cuda.to_device(np.ascontiguousarray(powers, dtype=np.float64))
    d_scales = cuda.to_device(np.ascontiguousarray(scales, dtype=np.float64))
    d_c_tan = cuda.to_device(np.ascontiguousarray(c_tan_table, dtype=np.float64))
    d_c_pol = cuda.to_device(np.ascontiguousarray(c_pol_table, dtype=np.float64))
    d_polarity_vol = cuda.to_device(np.ascontiguousarray(polarity_vol, dtype=np.float32))

    d_signal = cuda.to_device(np.full((n, 3), np.nan))
    d_coverage = cuda.to_device(np.zeros(n))
    d_counts = cuda.to_device(np.zeros((n, 3), dtype=np.int64))
    d_den = cuda.to_device(np.zeros((n, 3)))
    d_onpath = cuda.to_device(np.zeros((n, 3)))
    d_pol = cuda.to_device(np.full(n, np.nan))

    kernel = _render_kernel_mixed if mixed else _render_kernel_f64
    use_sigma = not math.isnan(sigma)
    # Floor the per-launch ray count at ~64K so a small image is not split into starved launches
    # (the chunking exists for the progress bar, not the hardware; 64K rays keep the GPU occupied).
    chunks = max(1, min(chunks, n // 65_536)) if n >= 65_536 else 1
    rays_per_chunk = max(1, -(-n // chunks))
    for start in range(0, n, rays_per_chunk):
        stop = min(start + rays_per_chunk, n)
        blocks = (stop - start + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
        kernel[blocks, _THREADS_PER_BLOCK](
            d_origins,
            d_look,
            d_impact,
            d_s,
            d_vol,
            vg.n_r,
            vg.n_theta,
            vg.n_phi,
            vg.spacing_code,
            vg.r_inner,
            vg.r_outer,
            vg.exponent,
            float(sigma),
            use_sigma,
            d_powers,
            use_powers,
            d_scales,
            use_scales,
            float(floor),
            float(log_max),
            clamp_lower,
            float(r_occult),
            occult_body,
            use_thomson,
            d_density,
            dg.n_r,
            dg.n_theta,
            dg.n_phi,
            dg.spacing_code,
            dg.r_inner,
            dg.r_outer,
            dg.exponent,
            thomson_pb,
            float(coeff_log_inner),
            float(coeff_inv_dlog),
            d_c_tan,
            d_c_pol,
            use_polarity,
            d_polarity_vol,
            start,
            stop,
            d_signal,
            d_coverage,
            d_counts,
            d_den,
            d_onpath,
            d_pol,
        )
        cuda.synchronize()
        if progress is not None:
            progress(stop)

    return (
        d_signal.copy_to_host(),
        d_coverage.copy_to_host(),
        d_counts.copy_to_host(),
        d_den.copy_to_host(),
        d_onpath.copy_to_host(),
        d_pol.copy_to_host(),
    )
