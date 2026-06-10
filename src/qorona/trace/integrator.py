"""Adaptive field-line integration on the unit field, batched over many lines.

Field lines are integrated on the **unit field** so the parameter is arc length and accuracy
is uniform on a radially stretched mesh:

    dr/ds = B̂ = B / |B|.

The integrator is **DOPRI5**, a 7-stage embedded 5(4) pair that is FSAL
(First Same As Last: the last stage of an accepted step is the first of the next, so an
accepted step costs ~6 field evaluations) and ships a C¹ dense-output interpolant used, at zero
extra field evaluations, to land feet on the boundary spheres (``boundaries.py``). The stepper
is written **generic over the state shape ``(n, state_dim)`` with a pluggable right-hand
side**, so the deviation transport extends position-only ``(n, 3)`` to
position+deviation ``(n, 9)`` by swapping the RHS; the step control, active mask, foot-landing,
and null guard are unchanged.

Step size is set by an embedded **PI controller** on a per-component scaled-RMS error norm
``wt_i = atol_i + rtol·max(|y_i|, |y1_i|)``, under a geometric **CFL ceiling**
``h_max = cfl · field.characteristic_length`` that prevents any step from skipping sub-cell
structure on the stretched grid. Magnetic nulls are handled **parameter-free**: only the exact
``0/0`` (a non-finite ``B̂``) is detected and flagged; genuine near-nulls are physical high-Q
features that arc-length tracing passes through and the controller resolves. Lines are traced
both ways from each seed (``-B̂`` and ``+B̂``) to the inner/outer spheres; open vs closed falls
out of the per-end landing codes.

References: the DOPRI5 ``RK5(4)7M`` Butcher tableau is implemented from Dormand & Prince (1980),
*J. Comput. Appl. Math.* **6**, 19; its quartic dense-output interpolant from Shampine (1986),
*Math. Comp.* **46**, 135.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np

from qorona.accel import HAVE_CUDA, HAVE_NUMBA, resolve_device
from qorona.console import print_success, progress_bar
from qorona.field.base import Field
from qorona.trace.boundaries import _classify_crossings, _localize_foot
from qorona.trace.fieldline import DEFAULT_TURN_GUARD, Endpoint, FieldLines, TurnGuard

# --- DOPRI5 (RK5(4)7M) Butcher tableau -----------------------------------------
# Lower-triangular RK matrix a[i, j], i = stage (0..5), j < i.
_A = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0],
        [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0],
        [19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0, 0.0, 0.0],
        [9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0, 0.0],
    ]
)
# 5th-order solution weights (also the 7th-stage row a[6, :], which is why stage 7 = f(y_new)).
_B = np.array([35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0])
# Error weights b5 - b4 over all 7 stages (the embedded estimate y5 - y4).
_E = np.array(
    [
        71.0 / 57600.0,
        0.0,
        -71.0 / 16695.0,
        71.0 / 1920.0,
        -17253.0 / 339200.0,
        22.0 / 525.0,
        -1.0 / 40.0,
    ]
)
# Dense-output coefficients: the quartic continuous extension
# y(θ) = y_old + h·Σ_s k_s·Σ_p P[s,p]·θ^(p+1), θ ∈ [0, 1] across the step.
_DENSE_P = np.array(
    [
        [
            1.0,
            -8048581381.0 / 2820520608.0,
            8663915743.0 / 2820520608.0,
            -12715105075.0 / 11282082432.0,
        ],
        [0.0, 0.0, 0.0, 0.0],
        [
            0.0,
            131558114200.0 / 32700410799.0,
            -68118460800.0 / 10900136933.0,
            87487479700.0 / 32700410799.0,
        ],
        [
            0.0,
            -1754552775.0 / 470086768.0,
            14199869525.0 / 1410260304.0,
            -10690763975.0 / 1880347072.0,
        ],
        [
            0.0,
            127303824393.0 / 49829197408.0,
            -318862633887.0 / 49829197408.0,
            701980252875.0 / 199316789632.0,
        ],
        [
            0.0,
            -282668133.0 / 205662961.0,
            2019193451.0 / 616988883.0,
            -1453857185.0 / 822651844.0,
        ],
        [
            0.0,
            40617522.0 / 29380423.0,
            -110615467.0 / 29380423.0,
            69997945.0 / 29380423.0,
        ],
    ]
)

# --- PI step-size control (pinned constants) --------------------------------------------------
_SAFETY = 0.9
_BETA = 0.04
_ALPHA = 0.2 - 0.75 * _BETA  # = 0.17 = 1/(q+1) - 0.75·β with q = min(5, 4) = 4
_MIN_FACTOR = 0.2
_MAX_FACTOR = 10.0
_ERR_PREV_FLOOR = 1.0e-4  # floor / initial value of the previous-step error (PI memory term)

# --- Pinned numerical floors (off the public signature; accuracy is the rtol knob) -------------
#: Per-component position floor so the relative error weight stays well-defined where a Cartesian
#: component passes through zero (loop apexes near the equatorial plane; a meridian crossing sends
#: x or y → 0). Not an accuracy dial.
_ATOL_POS = 1.0e-7
#: Step-size underflow guard, as a fraction of the local cell metric: a line whose step shrinks
#: below this is surfaced as a flagged resource-guard termination, never silently accepted.
_H_MIN_FRACTION = 1.0e-11


def _dense_state(
    state_old: np.ndarray, coefficients: np.ndarray, step: np.ndarray, theta: np.ndarray
) -> np.ndarray:
    """Evaluate the DOPRI5 dense-output interpolant at fractional step positions ``theta``.

    Parameters
    ----------
    state_old
        ``(m, k)`` state at the start of the step (``k = 3`` for position, ``state_dim`` for the
        full state).
    coefficients
        ``(m, 4, k)`` dense-output coefficients ``Σ_s stages·P`` for the step.
    step
        ``(m,)`` step lengths ``h``.
    theta
        ``(m,)`` fractional positions in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        ``(m, k)`` interpolated state.
    """
    powers = np.stack([theta, theta**2, theta**3, theta**4], axis=-1)
    return state_old + step[:, None] * np.einsum("mp,mpd->md", powers, coefficients, optimize=True)


def _dopri5_step(
    rhs: Callable[[np.ndarray, np.ndarray], np.ndarray],
    state: np.ndarray,
    first_derivative: np.ndarray,
    step: np.ndarray,
    lanes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Take one DOPRI5 step for a batch of lanes, returning the new state, stages, and error.

    ``first_derivative`` is the FSAL stage carried from the previous accepted step (``k1``), so
    this evaluates the RHS exactly six times (stages 2-6 and the FSAL stage 7 at the new point).

    Parameters
    ----------
    rhs
        Right-hand side ``(state (m, sd), lanes (m,)) -> derivative (m, sd)``; ``lanes`` are the
        global lane indices so the RHS can subset per-lane data (e.g. trace direction).
    state
        ``(m, sd)`` state at the start of the step.
    first_derivative
        ``(m, sd)`` the carried ``k1 = rhs(state)``.
    step
        ``(m,)`` step lengths.
    lanes
        ``(m,)`` global lane indices passed through to ``rhs``.

    Returns
    -------
    new_state : numpy.ndarray
        ``(m, sd)`` 5th-order solution at the step end.
    stages : numpy.ndarray
        ``(m, 7, sd)`` the seven stage derivatives (``stages[:, 6]`` is the FSAL stage at the new
        point, i.e. the next step's ``k1``).
    error : numpy.ndarray
        ``(m, sd)`` embedded error estimate ``y5 - y4``.
    """
    m, state_dim = state.shape
    stages = np.empty((m, 7, state_dim))
    stages[:, 0] = first_derivative
    for stage in range(1, 6):
        increment = np.einsum("j,mjd->md", _A[stage, :stage], stages[:, :stage], optimize=True)
        stages[:, stage] = rhs(state + step[:, None] * increment, lanes)
    new_state = state + step[:, None] * np.einsum("j,mjd->md", _B, stages[:, :6], optimize=True)
    stages[:, 6] = rhs(new_state, lanes)
    error = step[:, None] * np.einsum("j,mjd->md", _E, stages, optimize=True)
    return new_state, stages, error


def _integrate_batch(
    field: Field,
    rhs: Callable[[np.ndarray, np.ndarray], np.ndarray],
    state0: np.ndarray,
    active: np.ndarray,
    *,
    atol: np.ndarray,
    rtol: float,
    cfl: float,
    max_steps: int,
    max_reversals: int,
    turn_cos: float,
    turn_radius: float,
    weak_threshold: float,
    turn_min: int,
    store_path: bool,
    progress: Callable[[int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray] | None]:
    """Integrate a flat batch of lines to the domain boundaries (the reusable DOPRI5 core).

    Generic over ``state_dim``: the first three components of the state are the Cartesian
    position (used for the CFL metric, boundary crossing, and foot-landing), and any further
    components ride the same stepper, error norm, and dense output. The deviation transport reuses
    this unchanged, passing a 9-component state with a 9-component RHS and a matching ``atol``.

    Each line carries its own ``(arc_length, step)`` and accept/reject decision; one batched RHS
    evaluation per stage gathers only the active lanes. A line terminates by landing a foot on a
    boundary sphere (clean), by reaching an exact null (non-finite derivative), or by a resource
    guard (step underflow / ``max_steps``); the terminal state of an aborted line is ``NaN``.

    Parameters
    ----------
    field
        The field being traced (supplies the domain radii and the CFL cell metric).
    rhs
        Right-hand side ``(state, lanes) -> derivative`` on the unit field (signed per direction).
    state0
        ``(n_lines, state_dim)`` initial state; ``state0[:, :3]`` are the seed positions.
    active
        ``(n_lines,)`` bool mask of lines to integrate.
    atol
        ``(state_dim,)`` per-component absolute floor for the error norm.
    rtol
        Relative error tolerance (the accuracy knob).
    cfl
        CFL number in ``h_max = cfl · characteristic_length`` (``0 < cfl < 1``).
    max_steps
        Maximum step attempts per line before the resource guard fires.
    max_reversals
        Stall guard: terminate a line once its direction reverses (>90° between consecutive
        committed steps) this many times, the signature of a line trapped at a weak-field null.
        ``0`` disables the guard.
    turn_cos, turn_radius, weak_threshold, turn_min
        Sharp-turn guard (resolved scalars): a committed step *qualifies* when its direction turns
        past ``turn_cos`` (``B̂·B̂ < turn_cos``) while the new point is above ``turn_radius`` and in
        field weaker than ``weak_threshold`` (absolute ``|B|``). A line is terminated once it has
        made ``turn_min`` qualifying turns: a sustained staircase at a weak-field outer null, as
        opposed to a single legitimate null graze. ``weak_threshold = 0`` disables the guard.
    store_path
        Whether to record each line's ordered path points.
    progress
        Optional callback receiving the cumulative count of finished lines.

    Returns
    -------
    terminal_state : numpy.ndarray
        ``(n_lines, state_dim)`` state at each line's end (``NaN`` for an aborted line).
    ends : numpy.ndarray
        ``(n_lines,)`` ``int8`` :class:`Endpoint` code per line.
    lengths : numpy.ndarray
        ``(n_lines,)`` arc length traced.
    paths : list[numpy.ndarray] or None
        Per-line ``(m_i, 3)`` ordered path points (seed → end), or ``None``.
    """
    state0 = np.ascontiguousarray(state0, dtype=np.float64)
    n_lines = state0.shape[0]
    inner_radius = field.domain.inner_radius
    outer_radius = field.domain.outer_radius

    state = state0.copy()
    arc_length = np.zeros(n_lines)
    reversals = np.zeros(n_lines, dtype=np.int64)
    turns = np.zeros(n_lines, dtype=np.int64)  # the sharp-turn guard's qualifying-turn count
    active = np.array(active, dtype=bool)
    step = cfl * field.characteristic_length(state0[:, :3])
    err_prev = np.full(n_lines, _ERR_PREV_FLOOR)
    # Per-lane "the previous attempt was rejected" flag: the step after a rejection may shrink or
    # hold but not grow, which damps oscillation on QSL-grazing lines.
    rejected_last = np.zeros(n_lines, dtype=bool)

    # FSAL k1, carried across steps; recomputed only when a lane commits to a new point.
    derivative = np.full_like(state0, np.nan)
    seeds_idx = np.flatnonzero(active)
    if seeds_idx.size:
        derivative[seeds_idx] = rhs(state0[seeds_idx], seeds_idx)

    terminal_state = np.full_like(state0, np.nan)
    ends = np.zeros(n_lines, dtype=np.int8)
    lengths = np.zeros(n_lines)
    paths: list[list[np.ndarray]] | None = (
        [[state0[i, :3].copy()] for i in range(n_lines)] if store_path else None
    )

    iteration = 0
    while active.any() and iteration < max_steps:
        idx = np.flatnonzero(active)

        # Null guard: a non-finite start derivative is the exact 0/0 at a magnetic null.
        finite = np.isfinite(derivative[idx]).all(axis=1)
        if not finite.all():
            null_idx = idx[~finite]
            ends[null_idx] = Endpoint.NULL
            lengths[null_idx] = arc_length[null_idx]
            active[null_idx] = False
            idx = idx[finite]
        if idx.size == 0:
            iteration += 1
            continue

        # CFL ceiling at the current position caps the step about to be taken.
        ceiling = cfl * field.characteristic_length(state[idx, :3])
        step_now = np.minimum(step[idx], ceiling)

        new_state, stages, error = _dopri5_step(rhs, state[idx], derivative[idx], step_now, idx)

        scale = atol + rtol * np.maximum(np.abs(state[idx]), np.abs(new_state))
        err = np.sqrt(np.mean((error / scale) ** 2, axis=1))
        err = np.where(np.isfinite(err), err, np.inf)
        accepted = err <= 1.0

        end_radius = np.sqrt(np.sum(new_state[:, :3] ** 2, axis=1))
        crossed, target_radius, code = _classify_crossings(end_radius, inner_radius, outer_radius)

        # The PI base factor fac·err**(-_ALPHA), shared by the accept and reject step updates.
        # The errstate covers the err = 0 divide (factor to infinity, clipped to the growth
        # ceiling); a non-finite err was mapped to inf above and decays to the floor warning-free.
        with np.errstate(divide="ignore", invalid="ignore"):
            base_factor = _SAFETY * err ** (-_ALPHA)

        # Accepted and still inside: commit, advance arc length, and hand the FSAL stage forward.
        commit = accepted & ~crossed
        if commit.any():
            committed = idx[commit]
            # B̂_prev · B̂_new across the step (the position block of the FSAL derivative, still the
            # previous step's value here, against the new FSAL stage): one dot feeds both
            # weak-field guards. Computed before the FSAL stage is rolled forward.
            guard_active = max_reversals > 0 or weak_threshold > 0.0
            turn_dot = (
                np.sum(derivative[committed, :3] * stages[commit, 6, :3], axis=1)
                if guard_active
                else np.empty(0)
            )
            # Stall guard: a >90° turn (turn_dot < 0) is the thrashing-trap signature; count it.
            if max_reversals > 0:
                reversals[committed] += turn_dot < 0.0
            growth = base_factor[commit] * err_prev[committed] ** _BETA
            # After a rejection the step may shrink or hold but not grow (facmax = 1).
            max_factor = np.where(rejected_last[committed], 1.0, _MAX_FACTOR)
            state[committed] = new_state[commit]
            arc_length[committed] += step_now[commit]
            derivative[committed] = stages[commit, 6]
            err_prev[committed] = np.maximum(err[commit], _ERR_PREV_FLOOR)
            step[committed] = step_now[commit] * np.clip(growth, _MIN_FACTOR, max_factor)
            rejected_last[committed] = False
            if paths is not None:
                points = new_state[commit, :3]
                for local, line in enumerate(committed):
                    paths[line].append(points[local].copy())
            # Sharp-turn guard: count steps turning past turn_cos, above turn_radius, into field
            # weaker than weak_threshold, and terminate once a line reaches turn_min of them: a
            # sustained staircase, not a single legitimate null graze. |B| is sampled only at the
            # (rare) corners passing the cheap turn and radius tests. No clean foot (terminal_state
            # stays NaN), so a deflected line is excluded like the null ends.
            deflected = np.zeros(committed.shape[0], dtype=bool)
            if weak_threshold > 0.0:
                sharp = (turn_dot < turn_cos) & (end_radius[commit] > turn_radius)
                if sharp.any():
                    sharp_idx = np.flatnonzero(sharp)
                    corners = new_state[commit][sharp_idx, :3]
                    weak = field.sample(corners, gradient=False).b_magnitude < weak_threshold
                    turns[committed[sharp_idx[weak]]] += 1
                deflected = turns[committed] >= turn_min
                if deflected.any():
                    deflected_lanes = committed[deflected]
                    ends[deflected_lanes] = Endpoint.DEFLECTED
                    lengths[deflected_lanes] = arc_length[deflected_lanes]
                    active[deflected_lanes] = False
            # Terminate lines over the reversal budget the sharp-turn guard did not already stop.
            if max_reversals > 0:
                stalled = (reversals[committed] >= max_reversals) & ~deflected
                if stalled.any():
                    stalled_lanes = committed[stalled]
                    ends[stalled_lanes] = Endpoint.STALLED
                    lengths[stalled_lanes] = arc_length[stalled_lanes]
                    active[stalled_lanes] = False

        # Accepted and left the domain: land the foot on the dense-output interpolant.
        landed = accepted & crossed
        if landed.any():
            land_idx = idx[landed]
            state_old = state[land_idx]
            step_land = step_now[landed]
            coefficients = np.einsum("msd,sp->mpd", stages[landed], _DENSE_P, optimize=True)

            # The foot root-find needs only the position components of the full-state dense output.
            position_at = partial(_dense_state, state_old[:, :3], coefficients[:, :, :3], step_land)
            theta_star = _localize_foot(position_at, target_radius[landed])
            terminal_state[land_idx] = _dense_state(state_old, coefficients, step_land, theta_star)
            lengths[land_idx] = arc_length[land_idx] + theta_star * step_land
            ends[land_idx] = code[landed]
            active[land_idx] = False
            if paths is not None:
                feet = terminal_state[land_idx, :3]
                for local, line in enumerate(land_idx):
                    paths[line].append(feet[local].copy())

        # Rejected: shrink the step (drop the PI memory term, cap at no-growth) and guard underflow.
        rejected = ~accepted
        if rejected.any():
            reject_idx = idx[rejected]
            shrunk = step_now[rejected] * np.clip(base_factor[rejected], _MIN_FACTOR, 1.0)
            step[reject_idx] = shrunk
            rejected_last[reject_idx] = True
            underflow = shrunk < _H_MIN_FRACTION * ceiling[rejected]
            if underflow.any():
                stalled = reject_idx[underflow]
                ends[stalled] = Endpoint.MAX_STEPS
                lengths[stalled] = arc_length[stalled]
                active[stalled] = False

        if progress is not None:
            progress(n_lines - int(active.sum()))
        iteration += 1

    # Lines still active at max_steps: a resource guard, surfaced (no clean foot).
    if active.any():
        remaining = np.flatnonzero(active)
        ends[remaining] = Endpoint.MAX_STEPS
        lengths[remaining] = arc_length[remaining]
        if progress is not None:
            progress(n_lines)

    paths_out = [np.array(points) for points in paths] if paths is not None else None
    return terminal_state, ends, lengths, paths_out


def _integrate(
    field: Field,
    rhs: Callable[[np.ndarray, np.ndarray], np.ndarray],
    state0: np.ndarray,
    active: np.ndarray,
    directions: np.ndarray,
    *,
    transport: bool,
    atol: np.ndarray,
    rtol: float,
    cfl: float,
    max_steps: int,
    max_reversals: int,
    turn_guard: TurnGuard,
    store_path: bool,
    device: str = "auto",
    precision: str = "mixed",
    progress: Callable[[int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray] | None]:
    """Integrate a flat batch of lines, via the numba kernel when usable, else the NumPy core.

    The accelerated path (``qorona.accel``) runs each line scalar-per-lane under ``prange`` and is
    taken when numba is installed, the field opts in via ``_jit_field()``, and no per-line path is
    requested (paths are awkward in nopython and unused by the volume build). Otherwise this calls
    the NumPy :func:`_integrate_batch` with the caller's ``rhs`` closure. That NumPy core is the
    reference implementation: the fallback when numba is unavailable, and what the kernel is
    validated against. Both engines produce the same results, so the choice is invisible to callers.

    ``rhs`` (with its captured ``directions``) drives only the NumPy core; the kernel reads
    ``directions`` and ``transport`` directly. The signatures and outputs match
    :func:`_integrate_batch` (``paths`` is ``None`` on the kernel path).

    Parameters
    ----------
    field
        The field being traced; supplies the JIT descriptor (``_jit_field``) and, for the NumPy
        path, the domain radii and CFL metric.
    rhs
        The NumPy right-hand side ``(state, lanes) -> derivative`` (position-only or transport),
        used only by the fallback.
    state0, active
        ``(n_lines, state_dim)`` initial state and the ``(n_lines,)`` active mask.
    directions
        ``(n_lines,)`` per-lane trace sign ``±1`` (used by the kernel).
    transport
        Whether the state carries the deviation block (``state_dim == 9``); selects the kernel RHS.
    atol, rtol, cfl, max_steps, max_reversals, store_path, progress
        Forwarded to either engine as in :func:`_integrate_batch`.
    turn_guard
        Sharp-turn guard parameters; its relative ``weak_fraction`` is resolved here against
        ``field.reference_strength()`` into the absolute ``weak_threshold`` both engines use, so the
        threshold is the same whichever engine runs.
    device
        Backend selection: ``"auto"`` runs on the GPU when one is present, else falls through to the
        numba/NumPy CPU tiers; ``"gpu"`` forces the CUDA kernel and raises if no GPU is present, or
        if ``store_path`` / a non-JIT-able field makes the kernel unusable; ``"cpu"`` skips the GPU
        tier entirely. The GPU path is the CUDA twin of the numba kernel, validated against it.
    precision
        CUDA kernel precision (GPU tier only; the CPU tiers always run float64): ``"mixed"``
        (default), ``"float64"``, or ``"float32"``. See :class:`~qorona.config.VolumeConfig`.

    Returns
    -------
    tuple
        ``(terminal_state, ends, lengths, paths)``; ``paths`` is ``None`` on the kernel path.
    """
    weak_threshold = (
        turn_guard.weak_fraction * field.reference_strength() if turn_guard.enabled else 0.0
    )
    turn_cos = turn_guard.cos_threshold
    turn_radius = turn_guard.radius
    turn_min = turn_guard.min_turns

    # JIT-ability tier (shared by the CUDA and numba tiers): the field must expose a descriptor and
    # paths are unsupported in either kernel (routed to the NumPy core).
    jit_field_method = getattr(field, "_jit_field", None) if not store_path else None
    jit_field = jit_field_method() if jit_field_method is not None else None

    # Explicit device='gpu' with an unusable JIT tier is a loud error (the hardware-tier loud error
    # for a missing GPU is raised inside resolve_device).
    if device == "gpu" and jit_field is None:
        reason = (
            "store_path is set" if store_path else "the field is not GPU-JIT-able (no _jit_field)"
        )
        raise ValueError(f"device='gpu' cannot be used because {reason}; use device='auto'")

    resolved = resolve_device(device)
    if resolved == "gpu" and HAVE_CUDA and jit_field is not None:
        from qorona.accel.cuda_kernels import integrate_batch_cuda

        state0 = np.ascontiguousarray(state0, dtype=np.float64)
        terminal_state, ends, lengths = integrate_batch_cuda(
            state0,
            np.ascontiguousarray(active, dtype=np.bool_),
            np.ascontiguousarray(directions, dtype=np.float64),
            transport,
            jit_field,
            np.ascontiguousarray(atol, dtype=np.float64),
            float(rtol),
            float(cfl),
            int(max_steps),
            int(max_reversals),
            float(turn_cos),
            float(turn_radius),
            float(weak_threshold),
            int(turn_min),
            precision=precision,
        )
        if progress is not None:
            progress(state0.shape[0])
        return terminal_state, ends, lengths, None

    if HAVE_NUMBA and jit_field is not None:
        from qorona.accel.kernels import integrate_batch_jit

        state0 = np.ascontiguousarray(state0, dtype=np.float64)
        terminal_state, ends, lengths = integrate_batch_jit(
            state0,
            np.ascontiguousarray(active, dtype=np.bool_),
            np.ascontiguousarray(directions, dtype=np.float64),
            transport,
            jit_field,
            np.ascontiguousarray(atol, dtype=np.float64),
            float(rtol),
            float(cfl),
            int(max_steps),
            int(max_reversals),
            float(turn_cos),
            float(turn_radius),
            float(weak_threshold),
            int(turn_min),
        )
        if progress is not None:
            progress(state0.shape[0])
        return terminal_state, ends, lengths, None

    return _integrate_batch(
        field,
        rhs,
        state0,
        active,
        atol=atol,
        rtol=rtol,
        cfl=cfl,
        max_steps=max_steps,
        max_reversals=max_reversals,
        turn_cos=turn_cos,
        turn_radius=turn_radius,
        weak_threshold=weak_threshold,
        turn_min=turn_min,
        store_path=store_path,
        progress=progress,
    )


def _assemble_paths(half_paths: list[np.ndarray], n: int) -> list[np.ndarray]:
    """Join the backward and forward half-line paths into one path per seed.

    Each half path runs seed → foot in trace order; the full path is the backward half reversed
    (foot → seed) followed by the forward half from just past the shared seed.
    """
    return [np.concatenate([half_paths[i][::-1], half_paths[n + i][1:]], axis=0) for i in range(n)]


def _fold_half_lines(
    positions: np.ndarray,
    ends: np.ndarray,
    lengths: np.ndarray,
    seeds: np.ndarray,
    half_paths: list[np.ndarray] | None,
) -> FieldLines:
    """Fold the ``2n`` half-line integrator outputs into the ``n``-row :class:`FieldLines`.

    Lanes ``[0, n)`` are the backward halves and ``[n, 2n)`` the forward halves (the order both
    :func:`trace_field_lines` and the squashing-factor transport build their batched state in),
    so the two ends stack along axis 1 as ``(backward, forward)``: feet from the terminal
    positions, per-end :class:`Endpoint` codes, and per-end arc lengths, plus the optionally
    joined paths. Shared by the tracer and the transport so the geometric struct is assembled
    one way.

    Parameters
    ----------
    positions
        ``(2n, 3)`` terminal position of each half-line (``NaN`` for an aborted end).
    ends
        ``(2n,)`` ``int8`` :class:`Endpoint` code per half-line.
    lengths
        ``(2n,)`` arc length traced by each half-line.
    seeds
        ``(n, 3)`` seed points, kept on the struct.
    half_paths
        ``2n`` per-half ordered paths (seed → foot), or ``None`` when paths were not stored.

    Returns
    -------
    FieldLines
        The ``n``-row struct-of-arrays.
    """
    n = len(seeds)
    return FieldLines(
        seeds=seeds,
        feet=np.stack([positions[:n], positions[n:]], axis=1),
        ends=np.stack([ends[:n], ends[n:]], axis=1),
        lengths=np.stack([lengths[:n], lengths[n:]], axis=1),
        paths=_assemble_paths(half_paths, n) if half_paths is not None else None,
    )


def trace_field_lines(
    field: Field,
    seeds: np.ndarray,
    *,
    rtol: float = 1e-4,
    cfl: float = 0.5,
    max_steps: int = 10_000,
    max_reversals: int = 8,
    turn_guard: TurnGuard = DEFAULT_TURN_GUARD,
    store_path: bool = False,
    show_progress: bool = True,
    device: str = "auto",
    precision: str = "mixed",
) -> FieldLines:
    """Trace field lines through ``seeds``, both ways, to the inner and outer boundary spheres.

    Each seed spawns two half-lines (backward along ``-B̂`` and forward along ``+B̂``) traced
    on the unit field with adaptive DOPRI5 until each lands a foot on a boundary sphere. The two
    feet, their :class:`Endpoint` codes, and the per-end arc lengths are returned as a
    :class:`FieldLines`; open vs closed follows from the codes.

    Parameters
    ----------
    field
        The field to trace (real :class:`~qorona.field.sampled.SampledField` or an analytic one).
    seeds
        ``(n, 3)`` Cartesian seed points in R☉, all inside ``field.domain``.
    rtol
        Embedded relative error tolerance: the accuracy knob.
    cfl
        CFL number in the step ceiling ``h_max = cfl · characteristic_length``; must satisfy
        ``0 < cfl < 1`` so a step's overshoot past a boundary stays within the field's ghost
        padding.
    max_steps
        Maximum step attempts per line before the resource guard flags it.
    max_reversals
        Stall guard: terminate a line once it reverses direction (>90° between consecutive committed
        steps) this many times (a line trapped at a weak-field null; see :attr:`Endpoint.STALLED`).
        ``0`` disables it.
    turn_guard
        Sharp-turn guard: terminate a line that makes a single sharp turn in the outer corona where
        ``|B|`` is weak, a grid-locked staircase deflection at a null the stall guard is blind to.
        :class:`TurnGuard` holds the thresholds; ``max_turn_angle = 0`` disables it.
    store_path
        When ``True`` also return each line's full ordered path (memory-heavy); otherwise only
        feet, lengths, and codes are kept.
    show_progress
        Whether to display progress (tracing is the pipeline's slow stage).
    device
        Backend selection: ``"auto"`` runs on the GPU when present, else the numba/NumPy CPU tiers;
        ``"gpu"`` forces the CUDA kernel and raises if no GPU is present or if ``store_path`` makes
        it unusable; ``"cpu"`` skips the GPU tier.
    precision
        CUDA kernel precision (GPU tier only; the CPU tiers always run float64): ``"mixed"``
        (default), ``"float64"``, or ``"float32"``. See :class:`~qorona.config.VolumeConfig`.

    Returns
    -------
    FieldLines
        The traced lines as a struct-of-arrays over the ``n`` seeds.
    """
    if not 0.0 < cfl < 1.0:
        raise ValueError(
            f"cfl must satisfy 0 < cfl < 1 so boundary overshoot stays within the ghost "
            f"padding, got {cfl}"
        )
    seeds = np.ascontiguousarray(seeds, dtype=np.float64)
    if seeds.ndim != 2 or seeds.shape[1] != 3:
        raise ValueError(f"seeds must have shape (n, 3), got {seeds.shape}")

    inside = field.domain.in_domain(seeds)
    if not inside.all():
        n_outside = int((~inside).sum())
        raise ValueError(
            f"{n_outside} of {len(seeds)} seeds lie outside the field domain "
            f"[{field.domain.inner_radius}, {field.domain.outer_radius}] R_sun"
        )
    n = len(seeds)

    # Two half-lines per seed: lanes [0, n) trace backward (-B̂), lanes [n, 2n) forward (+B̂).
    state0 = np.tile(seeds, (2, 1))
    directions = np.concatenate([np.full(n, -1.0), np.full(n, 1.0)])

    def rhs(state: np.ndarray, lanes: np.ndarray) -> np.ndarray:
        sample = field.sample(state[:, :3], gradient=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            unit = sample.b / sample.b_magnitude[:, None]
        return directions[lanes, None] * unit

    atol = np.full(3, _ATOL_POS)
    active = np.ones(2 * n, dtype=bool)

    with progress_bar("Tracing field lines", 2 * n, enabled=show_progress) as progress:
        terminal_state, ends_half, lengths_half, half_paths = _integrate(
            field,
            rhs,
            state0,
            active,
            directions,
            transport=False,
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
    if show_progress:
        print_success(f"Traced {n} field lines: {lines.summary()}")
    return lines
