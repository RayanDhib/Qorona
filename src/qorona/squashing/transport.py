"""Co-integrate the deviation vectors with each field line.

The perpendicular squashing factor Q⊥ is assembled from how two transverse deviation vectors
``U, V`` are stretched between a field line's two footpoints. They are transported along the line
by the variational system co-integrated with the unit-field tracer:

    d/ds (r, U, V) = direction · (B̂, (U·∇)B̂, (V·∇)B̂),
    ∇B̂ = (I - B̂B̂ᵀ)·∇B / |B|,    (U·∇)B̂ = ∇B̂ · U.

This rides the field-line integrator unchanged: the position-only ``(n, 3)`` state becomes the
``(n, 9)`` state ``(r, U, V)`` and the position RHS becomes this 9-vector RHS, while the DOPRI5
stepper, the embedded PI control (over all nine components), the CFL ceiling, the dense-output
foot-landing, and the parameter-free null guard are reused verbatim; only the right-hand side
and the deviation error floor differ. The unit-field gradient ∇B̂ is formed *here* (not in the
field, which returns the raw Jacobian ∇B, nor in the position-only RHS) from a single
``field.sample(gradient=True)`` per stage, reusing the bundle's one ``b_magnitude`` so no two
paths disagree on |B|. A single per-lane ``direction ∈ {+1, -1}`` multiplies the whole 9-vector:
tracing backward reverses the arc-length parameter, so ``dU/ds`` flips to ``(U·∇)(-B̂)`` in
lockstep with the position and the deviation block needs no separate sign bookkeeping.

Reference: the deviation/variational system co-integrated with the field line (Eqs. 32-33) is
implemented from Scott, Pontin & Hornig (2017). The raw Jacobian uses the index convention
``grad_b[i, j] = ∂B_i/∂x_j``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from qorona.field.base import Field
from qorona.trace.fieldline import FieldLines, TurnGuard
from qorona.trace.integrator import _ATOL_POS, _fold_half_lines, _integrate

#: Per-component absolute floor for the deviation block of the error norm, so the relative weight
#: stays well-defined where a transported component passes through zero. Off the public signature
#: (the accuracy knob is ``rtol``); pinned alongside the position floor ``_ATOL_POS``.
_ATOL_DEV = 1.0e-7


def _unit_field_gradient(b: np.ndarray, b_magnitude: np.ndarray, grad_b: np.ndarray) -> np.ndarray:
    """Return the unit-field Jacobian ∇B̂, with ``[m, i, j] = ∂B̂_i/∂x_j``.

    Formed from the raw field Jacobian by the projector-quotient rule
    ``∇B̂ = (I - B̂B̂ᵀ)·∇B / |B|``, reusing the supplied ``b_magnitude`` so it agrees with the |B|
    used everywhere else. The directional derivative ``(U·∇)B̂`` that drives the transport is the
    contraction of this on its **second** index with U (see :func:`_deviation_derivative`).

    Parameters
    ----------
    b
        ``(m, 3)`` magnetic field.
    b_magnitude
        ``(m,)`` field strength ``|B|``.
    grad_b
        ``(m, 3, 3)`` raw Jacobian ``∂B_i/∂x_j``.

    Returns
    -------
    numpy.ndarray
        ``(m, 3, 3)`` unit-field Jacobian ``∂B̂_i/∂x_j``.
    """
    b_hat = b / b_magnitude[:, None]
    # along_j = B̂_k ∂B_k/∂x_j contracts B̂ with grad_b's FIRST index; the projector then removes
    # the component of each gradient column along B̂, and the 1/|B| completes the quotient rule.
    along = np.einsum("mk,mkj->mj", b_hat, grad_b, optimize=True)
    return (grad_b - b_hat[:, :, None] * along[:, None, :]) / b_magnitude[:, None, None]


def _deviation_derivative(
    b: np.ndarray, b_magnitude: np.ndarray, grad_b: np.ndarray, deviations: np.ndarray
) -> np.ndarray:
    """Return the deviation-transport block ``(d·∇)B̂`` for a stack of deviation vectors.

    The variational right-hand side ``∇B̂ · d``: the contraction of the unit-field Jacobian
    (:func:`_unit_field_gradient`) on its **second** index with each deviation vector ``d``. The
    second-index contraction is the pinned convention ``(d·∇)B̂ = ∇B̂ · d`` with
    ``∇B̂[i, j] = ∂B̂_i/∂x_j``; a transpose here is the silent bug the contraction guard catches.

    Parameters
    ----------
    b, b_magnitude, grad_b
        The field sample at the evaluation points (``(m, 3)``, ``(m,)``, ``(m, 3, 3)``).
    deviations
        ``(m, k, 3)`` the ``k`` deviation vectors per lane (here ``k = 2``: ``U`` and ``V``).

    Returns
    -------
    numpy.ndarray
        ``(m, k, 3)`` the directional derivative ``(d·∇)B̂`` of each deviation vector.
    """
    grad_b_hat = _unit_field_gradient(b, b_magnitude, grad_b)
    return np.einsum("mij,mkj->mki", grad_b_hat, deviations, optimize=True)


def _trace_and_transport(
    field: Field,
    seeds: np.ndarray,
    seed_basis: tuple[np.ndarray, np.ndarray],
    *,
    rtol: float,
    cfl: float,
    max_steps: int,
    max_reversals: int,
    turn_guard: TurnGuard,
    store_path: bool,
    device: str = "auto",
    precision: str = "mixed",
    progress: Callable[[int], None] | None,
) -> tuple[FieldLines, np.ndarray]:
    """Trace each seed both ways while co-transporting its deviation vectors to the feet.

    Builds the ``(2n, 9)`` initial state ``(seed, U₀, V₀)``: backward lanes ``[0, n)``, forward
    lanes ``[n, 2n)``, both halves seeded with the **same** ``(U₀, V₀)`` (the neighbouring line a
    deviation implies is shared by the two halves), and the 9-vector RHS, then hands them to the
    shared :func:`~qorona.trace.integrator._integrate` dispatcher. Each half-line's terminal state
    carries its foot position (folded into the geometric :class:`FieldLines`) and the deviation
    vectors transported to that foot (returned for assembly), both landed on the boundary by the
    same dense-output interpolant.

    Parameters
    ----------
    field
        The field to trace and transport through.
    seeds
        ``(n, 3)`` in-domain seed points.
    seed_basis
        ``(U₀, V₀)``, each ``(n, 3)``: the per-seed orthonormal deviation basis ⊥ B̂₀.
    rtol
        Embedded relative error tolerance (the accuracy knob), over all nine components.
    cfl
        CFL number in the step ceiling ``h_max = cfl · characteristic_length``; ``0 < cfl < 1``.
    max_steps
        Resource guard: maximum step attempts per half-line.
    max_reversals
        Stall guard: terminate a half-line after this many >90° direction reversals (a line trapped
        at a weak-field null); ``0`` disables it.
    turn_guard
        Sharp-turn guard: terminate a half-line that makes a single sharp turn in the outer corona
        where ``|B|`` is weak (a staircase deflection at a null); see :class:`TurnGuard`.
    store_path
        Whether to record each line's full ordered geometric path.
    device
        Compute backend: ``"auto"`` selects the GPU when present, else the numba/NumPy CPU tiers;
        ``"gpu"`` forces the CUDA kernel and raises if no usable GPU is available; ``"cpu"`` forces
        the CPU tiers regardless of GPU presence.
    precision
        CUDA kernel precision (GPU tier only; the CPU tiers always run float64): ``"mixed"``
        (default), ``"float64"``, or ``"float32"``. See :class:`~qorona.config.VolumeConfig`.
    progress
        Optional callback receiving the cumulative count of finished half-lines.

    Returns
    -------
    lines : FieldLines
        The geometric traced lines (feet, ends, lengths, optional paths), aligned to ``seeds``.
    deviations : numpy.ndarray
        ``(n, 2, 2, 3)`` deviation vectors transported to the feet; axis 1 is the
        ``(backward, forward)`` foot and axis 2 is ``(U, V)``. ``NaN`` for an aborted end.
    """
    n = len(seeds)
    u0, v0 = seed_basis

    state0 = np.empty((2 * n, 9))
    state0[:, :3] = np.tile(seeds, (2, 1))
    state0[:, 3:6] = np.tile(u0, (2, 1))
    state0[:, 6:9] = np.tile(v0, (2, 1))
    directions = np.concatenate([np.full(n, -1.0), np.full(n, 1.0)])

    def rhs(state: np.ndarray, lanes: np.ndarray) -> np.ndarray:
        sample = field.sample(state[:, :3], gradient=True)
        assert sample.grad_b is not None  # gradient=True always populates grad_b
        deviations = state[:, 3:].reshape(-1, 2, 3)
        with np.errstate(divide="ignore", invalid="ignore"):
            b_hat = sample.b / sample.b_magnitude[:, None]
            d_dev = _deviation_derivative(sample.b, sample.b_magnitude, sample.grad_b, deviations)
        derivative = np.concatenate([b_hat, d_dev.reshape(-1, 6)], axis=1)
        return directions[lanes, None] * derivative

    atol = np.concatenate([np.full(3, _ATOL_POS), np.full(6, _ATOL_DEV)])
    active = np.ones(2 * n, dtype=bool)

    terminal_state, ends_half, lengths_half, half_paths = _integrate(
        field,
        rhs,
        state0,
        active,
        directions,
        transport=True,
        atol=atol,
        rtol=rtol,
        cfl=cfl,
        max_steps=max_steps,
        max_reversals=max_reversals,
        turn_guard=turn_guard,
        store_path=store_path,
        device=device,
        precision=precision,
        progress=progress,
    )

    lines = _fold_half_lines(terminal_state[:, :3], ends_half, lengths_half, seeds, half_paths)
    foot_deviations = terminal_state[:, 3:].reshape(2 * n, 2, 3)
    deviations = np.stack([foot_deviations[:n], foot_deviations[n:]], axis=1)
    return lines, deviations
