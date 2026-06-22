"""The viewpoint-independent Q⊥ volume the render integrates.

A :class:`QPerpVolume` is a scalar **log₁₀ Q⊥** field on the internal spherical grid (the same
:class:`~qorona.resample.grid.SphericalGrid` machinery the field uses, reused as-is): a width-1
payload goes through :func:`~qorona.resample.grid.pad_field` and the generic
:func:`~qorona.field.interpolation.tricubic` with no code change, and the θ reflect-through-pole
padding is value-exact for a scalar (the ghost row is the true Q⊥ at the antipodal azimuth, with no
vector component to flip). Storing log₁₀ Q⊥ keeps the interpolated quantity tame across its many
decades and is exactly what the render integrates.

Three build paths, cross-validated against each other (the seed-invariant per-line engine makes
the reference a valid ground truth for the other two):

- :func:`build_volume_reference`, the **reference**. Seed the squashing engine at every voxel
  centre and assign each voxel its line's Q⊥: the literal definition of Q⊥ at every point (one full
  trace-and-transport per voxel), so it is slow and used only on small/coarse grids.

- :func:`build_volume_per_voxel`, **production, complete coverage**. Because Q⊥ is constant along
  a field line, it need only be computed once on the inner and outer boundary spheres
  (supersampled); an interior voxel then inherits its value by tracing position-only to both feet
  and combining the two precomputed boundary values. Far cheaper, since the boundary maps are built
  once and reused across every voxel and every camera; the cost still scales with the voxel count.

- :func:`build_volume_paint`, **production, default**. Trace a *seed set* of lines and paint each
  line's one Q⊥ along its full swept path, so the cost scales with the seed count instead of the
  voxel count; the trade is inter-line coverage gaps (``NaN``), which the render's
  weight-normalised integral tolerates.

The volume is **truthful, not display-clamped**: it stores the genuine log₁₀ Q⊥ wherever Q⊥ is
defined, including the real-data ``(0, 2)`` sub-floor tail (a resampling ∇·B artifact, since the
theoretical floor is Q⊥ ≥ 2). ``NaN`` marks only voxels with no representable value: Q⊥ ≤ 0
(undefined log), an incomplete line, or a foot whose boundary map cell is itself ``NaN``
(:class:`QPerpVolume` carries the full contract).

The interior recovery (trace each point both ways to the boundaries, interpolate Q⊥ at the two feet
in cubic and average) is a direct consequence of Q⊥ being constant along a line; the per-line Q⊥
values it interpolates between come from :func:`~qorona.squashing.compute_squashing`.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from qorona.accel import (
    HAVE_CUDA,
    HAVE_NUMBA,
    JitField,
    JitGrid,
    apply_workers,
    gpu_memory_budget,
    resolve_device,
)
from qorona.console import print_success, print_warning, progress_bar, status
from qorona.field.base import Field
from qorona.field.interpolation import tricubic
from qorona.field.sampled import SampledField
from qorona.geometry import cartesian_to_spherical, spherical_to_cartesian
from qorona.resample.grid import GHOST, SphericalGrid, pad_field, pad_phi, pad_theta
from qorona.trace import DEFAULT_TURN_GUARD, Endpoint, TurnGuard, trace_field_lines

# ``compute_squashing`` lives in this package's ``__init__`` (the public surface), so it is
# imported lazily inside the builders below to avoid a package-initialisation import cycle.

__all__ = [
    "QPerpVolume",
    "build_volume_paint",
    "build_volume_per_voxel",
    "build_volume_reference",
    "radial_sign_on_grid",
]

#: Upper bound on the points a single build call may hold, capping the peak memory of the per-call
#: ``(chunk, ...)`` arrays (the tricubic neighbourhood and the integrator state). The public memory
#: knob (``chunk_size`` on the builders), the same knob the resampler exposes. Lower it on
#: memory-tight machines. The *working* batch is the smaller :data:`_STREAM_BATCH` (below), so
#: this is only the ceiling, reached only if a caller sets ``chunk_size`` below it.
_CHUNK_SIZE = 500_000

#: Seeds processed per kernel launch inside a streamed stage: the progress *granularity*, kept well
#: below the :data:`_CHUNK_SIZE` memory ceiling so the bar advances (and its time estimate appears)
#: every few seconds instead of once per multi-minute chunk. The numba kernel reports progress only
#: once per launch (it cannot report from inside the ``prange``), so the launch size *is* the
#: granularity; each launch still saturates every core, so the extra launches add negligible
#: overhead while the per-call memory only shrinks. The effective batch is ``min(chunk_size,
#: _STREAM_BATCH)``, so a caller can still drop below it for tighter memory.
_STREAM_BATCH = 25_000

#: Seeds per GPU kernel launch, sized for the GPU's per-launch + PCIe transfer cost (the CPU
#: :data:`_STREAM_BATCH` granularity does not apply here). On the display-shared card the batch
#: also bounds per-launch wall-time (so a long float64 launch cannot stutter the desktop); 100k
#: balances the two against per-launch overhead. The effective GPU batch is
#: ``min(chunk_size, _GPU_STREAM_BATCH)``.
_GPU_STREAM_BATCH = 100_000

#: Relative radial margin by which boundary-touching seeds are pulled inside the shell before
#: seeding. A node at exactly R_inner/R_outer leaves the strict ``in_domain`` shell by a float
#: round-trip error (``spherical_to_cartesian`` makes ``|x|`` differ from ``r`` by ~1e-16·r); this
#: margin (≫ that error, ≪ the rtol=1e-4 engine tolerance) restores in-domain membership without
#: moving Q⊥; seed-invariance makes a hair-inside seed return the on-boundary line's value.
_BOUNDARY_MARGIN = 1.0e-9

#: Ceiling on the along-line samples the painter places per traced line. The painter sub-samples a
#: swept path at ``paint_step`` times the local cell extent (fine enough that a roughly-straight
#: segment skips no voxel); this cap bounds the rare long line whose smallest cell would otherwise
#: demand a runaway sample count, at the cost of a few possibly-skipped voxels on that line.
_MAX_PAINT_SAMPLES = 100_000

#: Ceiling on the kernel paint pass's per-chunk voxel-index buffer, in entries. The buffer is
#: ``(paint_chunk, max_deposits)`` ``int64``, so the paint chunk is sized to keep it under this
#: (≈ 128 MB): the single resolution-independent allocation of the trace-parallel + paint-serial
#: scheme (the shared grid is the only resolution-bound array, and there is just one of it).
_PAINT_BUFFER = 16_000_000


def _clip_to_domain(points: np.ndarray, inner_radius: float, outer_radius: float) -> np.ndarray:
    """Rescale each point radially into ``[R_inner(1+m), R_outer(1-m)]`` (``m = _BOUNDARY_MARGIN``).

    Points already strictly inside are returned unchanged; only boundary-touching ones are nudged a
    negligible amount along their own radius (so the field line through them, hence Q⊥, is unchanged
    to engine tolerance) to satisfy the seeding precondition.
    """
    radius = np.sqrt(np.sum(points * points, axis=-1))
    lower = inner_radius * (1.0 + _BOUNDARY_MARGIN)
    upper = outer_radius * (1.0 - _BOUNDARY_MARGIN)
    scale = np.where(radius > 0.0, np.clip(radius, lower, upper) / radius, 1.0)
    return points * scale[:, None]


@dataclass(frozen=True, slots=True)
class QPerpVolume:
    """Scalar log₁₀ Q⊥ on the internal spherical grid, raw and truthful.

    The real-data ``(0, 2)`` sub-floor tail is retained as finite values; ``NaN`` marks only voxels
    with no representable value: Q⊥ ≤ 0 (undefined log), an incomplete line, or a foot off a
    boundary map. The display floor and upper clamp are the render's job, not the volume's. The
    payload is ghost-padded once at construction so :meth:`sample` is a plain scalar tricubic.

    Attributes
    ----------
    grid
        The spherical grid the volume is stored on (its pitch sets the thinnest renderable QSL
        sheet, roughly one interior voxel).
    log_q_perp
        ``(n_r + 2·GHOST, n_theta + 2·GHOST, n_phi + 2·GHOST, 1)`` ghost-padded log₁₀ Q⊥, ready for
        the edge-agnostic tricubic with grid indices offset by :data:`~qorona.resample.grid.GHOST`.
    polarity
        Optional ghost-padded footpoint magnetic polarity, same shape as
        :attr:`log_q_perp`: the inner-footpoint ``sign(B·r̂)`` in
        ``{-1, 0, +1}`` (outward / neutral-or-closed / inward). A
        viewpoint-independent boundary-to-boundary channel for colouring the
        render by polarity; ``None`` when built without it.
    radial_sign
        Optional ghost-padded local radial sign ``sign(B·r̂)`` in ``{-1, 0, +1}`` on
        :attr:`radial_sign_grid`, signing the Q-map; ``None`` when absent. Distinct from
        :attr:`polarity`, the footpoint sign used by the render.
    radial_sign_grid
        The grid :attr:`radial_sign` is stored on (the field grid, coarser than :attr:`grid`);
        ``None`` when no radial-sign channel is present.
    """

    grid: SphericalGrid
    log_q_perp: np.ndarray
    polarity: np.ndarray | None = None
    radial_sign: np.ndarray | None = None
    radial_sign_grid: SphericalGrid | None = None

    def sample(self, points: np.ndarray) -> np.ndarray:
        """Return ``(n,)`` interpolated log₁₀ Q⊥ at ``points``; ``NaN`` outside the shell.

        Scalar tricubic on the padded payload, on the field's interpolation kernel but in its
        NaN-tolerant mode (``skip_nan=True``): a non-finite voxel is dropped from the 4x4x4 stencil
        and the kept weights renormalise, so a sample stays finite as long as one neighbour is, and
        the paint builder's inter-line gaps stay local instead of blacking out the render. Only
        where every neighbour is ``NaN`` (or the kept weight cancels to ~0) does the sample return
        ``NaN``. Points outside the radial shell ``[R_inner, R_outer]`` are not interpolated
        (``index_coordinates`` extrapolates off the ghost padding past the grid, returning garbage
        rather than ``NaN``), so they are masked out explicitly and returned ``NaN``.

        Parameters
        ----------
        points
            ``(n, 3)`` Cartesian coordinates in R☉.

        Returns
        -------
        numpy.ndarray
            ``(n,)`` log₁₀ Q⊥; ``NaN`` outside the shell or where the local stencil is all ``NaN``.
        """
        points = np.asarray(points, dtype=np.float64)
        radius = np.sqrt(np.sum(points * points, axis=-1))
        inside = (radius >= self.grid.radii[0]) & (radius <= self.grid.radii[-1])

        values = np.full(points.shape[0], np.nan)
        if inside.any():
            index, _ = self.grid.index_coordinates(points[inside])
            interpolated, _ = tricubic(
                self.log_q_perp, index + GHOST, gradient=False, skip_nan=True
            )
            values[inside] = interpolated[:, 0]
        return values

    def sample_polarity(self, points: np.ndarray) -> np.ndarray:
        """Return ``(n,)`` nearest-cell footpoint polarity at ``points``; ``NaN`` outside the shell.

        The polarity channel's companion to :meth:`sample`. A magnetic-polarity sign is discrete and
        cannot be interpolated (a tricubic would invent fractional values straddling the neutral
        line), so it is read NEAREST-CELL: the fractional grid index is rounded with
        ``floor(index + 0.5)``, matching the render kernel exactly, and the padded array indexed
        directly. ``NaN`` where the volume carries no polarity or the point is outside the radial
        shell; this is the NumPy reference for the kernel's in-loop polarity sampling.
        """
        points = np.asarray(points, dtype=np.float64)
        values = np.full(points.shape[0], np.nan)
        if self.polarity is None:
            return values
        radius = np.sqrt(np.sum(points * points, axis=-1))
        inside = (radius >= self.grid.radii[0]) & (radius <= self.grid.radii[-1])
        if inside.any():
            index, _ = self.grid.index_coordinates(points[inside])
            cell = np.floor(index + 0.5).astype(np.intp) + GHOST
            shape = self.polarity.shape
            i0 = np.clip(cell[:, 0], 0, shape[0] - 1)
            i1 = np.clip(cell[:, 1], 0, shape[1] - 1)
            i2 = np.clip(cell[:, 2], 0, shape[2] - 1)
            values[inside] = self.polarity[i0, i1, i2, 0]
        return values

    def sample_radial_sign(self, points: np.ndarray) -> np.ndarray:
        """Return ``(n,)`` local radial-field sign at ``points``; ``NaN`` if absent.

        The Q-map's signing companion to :meth:`sample`: the stored ``sign(B·r̂)`` is interpolated
        on :attr:`radial_sign_grid` and re-signed, so the warm↔cool boundary resolves to a smooth
        curve rather than in grid cells. ``NaN`` where no radial-sign channel is present or the
        point is outside that grid's radial shell.
        """
        points = np.asarray(points, dtype=np.float64)
        values = np.full(points.shape[0], np.nan)
        if self.radial_sign is None or self.radial_sign_grid is None:
            return values
        grid = self.radial_sign_grid
        radius = np.sqrt(np.sum(points * points, axis=-1))
        inside = (radius >= grid.radii[0]) & (radius <= grid.radii[-1])
        if inside.any():
            index, _ = grid.index_coordinates(points[inside])
            coords = index + GHOST
            coords[:, 0] = np.clip(coords[:, 0], 1.0, self.radial_sign.shape[0] - 3)
            interpolated, _ = tricubic(self.radial_sign, coords, gradient=False)
            values[inside] = np.sign(interpolated[:, 0])
        return values


def _pack_volume(
    grid: SphericalGrid, log_q_flat: np.ndarray, polarity_flat: np.ndarray | None = None
) -> QPerpVolume:
    """Reshape flat per-node log₁₀ Q⊥ (and optional polarity) into a ghost-padded volume.

    ``polarity_flat`` (per-node ``sign(B·r̂)`` in ``{-1, 0, +1}``) is
    ghost-padded like the payload: the reflect-through-pole / φ-wrap ghosts
    carry the antipodal sign, the true neighbouring polarity for a scalar. A
    builder that computes no polarity passes ``None``, leaving it absent.
    """
    grid_values = log_q_flat.reshape(grid.n_r, grid.n_theta, grid.n_phi, 1)
    polarity = None
    if polarity_flat is not None:
        polarity = pad_field(polarity_flat.reshape(grid.n_r, grid.n_theta, grid.n_phi, 1)).astype(
            np.float32, copy=False
        )
    return QPerpVolume(grid=grid, log_q_perp=pad_field(grid_values), polarity=polarity)


def radial_sign_on_grid(field: SampledField) -> np.ndarray:
    """Return per-node ``sign(B·r̂)`` on the field grid as int8 ``{-1, 0, +1}``.

    B at each grid node dotted with the node's outward radial direction, read from the resampled
    field. The local radial polarity that signs the Q-map, distinct from the footpoint
    :attr:`QPerpVolume.polarity`.
    """
    nodes = field.grid.node_points()
    radial = np.sum(field.b_at_nodes() * nodes, axis=-1)
    return np.sign(radial).ravel().astype(np.int8)


def _safe_log10(q_perp: np.ndarray) -> np.ndarray:
    """Return log₁₀ Q⊥ where Q⊥ is finite and positive, else ``NaN`` (Q⊥ ≤ 0 has no log)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(q_perp) & (q_perp > 0.0), np.log10(q_perp), np.nan)


# --- Parallel streaming over seed chunks -------------------------------------------------------
# The field-line work (boundary squashing, interior tracing) is embarrassingly parallel over seeds,
# and the parallelism lives *inside* the numba transport kernel (``qorona.accel``): each line runs
# on its own thread under ``prange``, so the volume build simply streams seeds through the engine in
# fixed-size chunks. Chunking bounds the per-call seed/output arrays and drives the progress bar;
# the kernel saturates the cores within a chunk. Without numba the engine falls back to the
# single-core NumPy integrator, so a no-numba build is serial; install numba (in the default
# install) for production-resolution volumes.

_ResultT = TypeVar("_ResultT")


@contextmanager
def _cuda_alloc_guard(what: str) -> Iterator[None]:
    """Convert a CUDA out-of-memory failure into an actionable error instead of a raw API dump."""
    try:
        yield
    except Exception as exc:
        if "OUT_OF_MEMORY" in str(exc).replace(" ", "_").upper():
            raise MemoryError(
                f"the GPU ran out of memory while allocating {what}; lower --resolution-factor "
                f"or --supersample, free GPU memory, or rerun with --device cpu"
            ) from exc
        raise


def _available_host_bytes() -> int:
    """Best-effort available host memory (0 when the platform exposes no probe)."""
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):  # non-POSIX or unsupported key
        return 0


def _stream_batch(device: str) -> int:
    """Return the per-launch streaming batch for the resolved backend (GPU vs CPU granularity)."""
    return _GPU_STREAM_BATCH if resolve_device(device) == "gpu" else _STREAM_BATCH


def _stream_chunks(
    seeds: np.ndarray,
    worker: Callable[[np.ndarray], _ResultT],
    *,
    chunk_size: int,
    batch: int,
    label: str,
    show_progress: bool,
) -> Iterator[tuple[int, int, _ResultT]]:
    """Apply ``worker`` to ``seeds`` in fixed-size batches, yielding ``(start, stop, result)``.

    A generator: each batch's result is yielded (and consumed/freed) before the next is computed, so
    peak memory stays bounded by one batch regardless of the seed count: the chunking contract. The
    effective batch is ``min(chunk_size, batch)``: ``batch`` is the backend's launch granularity
    (the smaller :data:`_STREAM_BATCH` for the CPU ``prange`` kernel, the larger
    :data:`_GPU_STREAM_BATCH` for the GPU; see :func:`_stream_batch`), driving the progress
    granularity (the bar advances and its time estimate appears within seconds), while
    ``chunk_size`` stays the caller's memory ceiling.
    """
    n = len(seeds)
    batch = min(chunk_size, batch)
    with progress_bar(label, n, enabled=show_progress) as progress:
        finished = 0
        for start in range(0, n, batch):
            stop = min(start + batch, n)
            result = worker(seeds[start:stop])
            finished += stop - start
            progress(finished)
            yield start, stop, result


def _warm_kernels(
    field: Field,
    grid: SphericalGrid,
    *,
    paint: bool,
    device: str = "auto",
    precision: str = "mixed",
    show_progress: bool,
) -> None:
    """Trigger the one-time numba kernel compile up front, announced with a spinner.

    The first kernel launch of a build pays a ~30-60 s JIT compile that is otherwise invisible: the
    progress bar sits at 0% with no time estimate and looks hung. Compiling here on a single seed,
    under a ``status`` spinner, moves that pause out of the bar (same total work, since the compile
    would happen on the first real batch regardless) and labels it, so every subsequent bar advances
    from its first batch. A no-op without numba or for a field/grid the kernel cannot JIT (the NumPy
    fallback has nothing to compile). One ``compute_squashing`` call compiles the shared tracer
    kernel for both the transport (boundary) and position-only (interior / paint feet) uses; the
    paint builder additionally compiles its rasterising kernel.
    """
    jit_field = getattr(field, "_jit_field", lambda: None)()
    jit_grid = grid._jit_grid()
    if not HAVE_NUMBA or jit_field is None or jit_grid is None:
        return
    from qorona.squashing import compute_squashing

    mid = 0.5 * (field.domain.inner_radius + field.domain.outer_radius)
    seed = np.array([[mid, 0.0, 0.0]])

    resolved = resolve_device(device)
    if resolved == "gpu" and HAVE_CUDA:
        from qorona.accel.cuda_kernels import paint_batch_cuda
        from qorona.trace.integrator import _ATOL_POS

        with status("Compiling CUDA kernels (one-time)...", enabled=show_progress):
            compute_squashing(field, seed, device="gpu", precision=precision, show_progress=False)
            if paint:
                paint_batch_cuda(
                    np.ascontiguousarray(seed),
                    np.array([True]),
                    jit_field,
                    jit_grid,
                    np.full(3, _ATOL_POS),
                    1e-4,
                    0.5,
                    10,
                    0.5,
                    4 * (grid.n_r + grid.n_theta + grid.n_phi),
                    precision=precision,
                )
        return

    with status("Compiling kernels (one-time)...", enabled=show_progress):
        compute_squashing(field, seed, show_progress=False)
        if paint:
            from qorona.accel.kernels import paint_batch_jit
            from qorona.trace.integrator import _ATOL_POS

            paint_batch_jit(
                np.ascontiguousarray(seed),
                np.array([True]),
                jit_field,
                jit_grid,
                np.full(3, _ATOL_POS),
                1e-4,
                0.5,
                10,
                0.5,
                4 * (grid.n_r + grid.n_theta + grid.n_phi),
            )


def build_volume_reference(
    field: Field,
    grid: SphericalGrid,
    *,
    rtol: float = 1e-4,
    cfl: float = 0.5,
    max_steps: int = 10_000,
    max_reversals: int = 8,
    turn_guard: TurnGuard = DEFAULT_TURN_GUARD,
    chunk_size: int = _CHUNK_SIZE,
    workers: int | None = None,
    device: str = "auto",
    precision: str = "mixed",
    show_progress: bool = True,
) -> QPerpVolume:
    """Build the reference Q⊥ volume by seeding the squashing engine at every voxel centre.

    The literal definition of Q⊥ at every grid node, one full trace-and-transport per node via
    :func:`~qorona.squashing.compute_squashing`, so it is expensive and meant only as the
    correctness ground truth on small/coarse grids (it validates :func:`build_volume_per_voxel`).
    The seeds are streamed in fixed-size batches to bound peak memory.

    Parameters
    ----------
    field
        The field to compute on (analytic dipole for validation, or a real
        :class:`~qorona.field.sampled.SampledField`).
    grid
        The spherical grid to build the volume on; its node centres are the seeds and its pitch sets
        the thinnest renderable sheet.
    rtol, cfl, max_steps, max_reversals, turn_guard
        Forwarded to :func:`~qorona.squashing.compute_squashing` (the accuracy knob, the CFL step
        ceiling, the per-half-line resource guard, the stall guard, and the sharp-turn guard).
    chunk_size
        Voxels processed per batch.
    workers
        numba thread count for the kernel (``None`` = all cores; ``1`` = serial). Ignored without
        numba; the NumPy fallback is single-core.
    device
        Compute backend: ``"auto"`` (GPU if available, else CPU), ``"gpu"``, or ``"cpu"``. Forwarded
        to the squashing engine, which routes to the CUDA tier on the GPU.
    precision
        CUDA kernel precision (GPU tier only; the CPU tiers always run float64): ``"mixed"``
        (default), ``"float64"``, or ``"float32"``. See :class:`~qorona.config.VolumeConfig`.
    show_progress
        Whether to display progress.

    Returns
    -------
    QPerpVolume
        The truthful log₁₀ Q⊥ volume; ``NaN`` where the seed's line is incomplete or Q⊥ ≤ 0.
    """
    nodes = _clip_to_domain(
        grid.node_points().reshape(-1, 3), float(grid.radii[0]), float(grid.radii[-1])
    )
    apply_workers(workers)
    _warm_kernels(
        field, grid, paint=False, device=device, precision=precision, show_progress=show_progress
    )

    def squash(chunk: np.ndarray) -> np.ndarray:
        from qorona.squashing import compute_squashing

        return compute_squashing(
            field,
            chunk,
            rtol=rtol,
            cfl=cfl,
            max_steps=max_steps,
            max_reversals=max_reversals,
            turn_guard=turn_guard,
            device=device,
            precision=precision,
            show_progress=False,
        ).q_perp

    q_perp = np.empty(nodes.shape[0])
    for start, stop, chunk_q in _stream_chunks(
        nodes,
        squash,
        chunk_size=chunk_size,
        batch=_stream_batch(device),
        label="Building Q⊥ volume (reference)",
        show_progress=show_progress,
    ):
        q_perp[start:stop] = chunk_q

    volume = _pack_volume(grid, _safe_log10(q_perp))
    if show_progress:
        print_success(f"Built reference Q⊥ volume on {grid.n_r}x{grid.n_theta}x{grid.n_phi} grid")
    return volume


class _BoundaryMap:
    """Cubic interpolant of a precomputed boundary log₁₀ Q⊥ map on the sphere.

    The map is a scalar on a cell-centred ``(θ, φ)`` grid at a fixed radius, padded with the grid's
    own ghost conventions, θ reflect-through-pole and φ periodic wrap (reusing
    :func:`~qorona.resample.grid.pad_theta` / :func:`~qorona.resample.grid.pad_phi` on a width-1
    payload), and interpolated by the **same Keys cubic the volume uses**, so the boundary map and
    the volume share one interpolation and one pole/periodic convention. A local Keys stencil also
    avoids the global-spline ringing a separatrix singularity would otherwise spread across the map,
    and ``NaN`` map cells propagate through the stencil to nearby queries, so a foot landing on an
    undefined boundary cell yields a ``NaN`` voxel, exactly as for the volume itself.
    """

    def __init__(self, log_map: np.ndarray) -> None:
        n_theta, n_phi = log_map.shape
        # θ reflect must read the real φ count, so it precedes the φ wrap (the pad_field order). The
        # 2-D map rides the 3-D Keys kernel through a degenerate radial axis: four identical layers
        # so the radial stencil is in range at index 1 and resolves to the map exactly.
        padded = pad_phi(pad_theta(log_map[None, :, :, None]))
        self._values = np.repeat(padded, 4, axis=0)
        self._theta_step = np.pi / n_theta
        self._phi_step = 2.0 * np.pi / n_phi

    def __call__(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Return ``(m,)`` interpolated log₁₀ Q⊥ at colatitudes ``theta`` and azimuths ``phi``."""
        theta_index = theta / self._theta_step - 0.5 + GHOST
        phi_index = (phi % (2.0 * np.pi)) / self._phi_step + GHOST
        coords = np.stack([np.ones_like(theta_index), theta_index, phi_index], axis=-1)
        values, _ = tricubic(self._values, coords, gradient=False)
        return values[:, 0]


def _sphere_seed_grid(
    radius: float, n_theta: int, n_phi: int, inner_radius: float, outer_radius: float
) -> np.ndarray:
    """Return the cell-centred ``(θ, φ)`` seed grid on the ``radius`` sphere, as Cartesian points.

    A node sits at the centre of each ``(θ, φ)`` cell, so none lands on a pole, and the points are
    nudged a hair inside ``[inner_radius, outer_radius]`` by :func:`_clip_to_domain` to satisfy the
    seeding precondition without moving the field line (hence Q⊥) through them. Shared by the
    boundary-map precompute and the painting builder's surface seeding so both sample one grid.
    """
    theta = (np.arange(n_theta) + 0.5) * (np.pi / n_theta)
    phi = np.arange(n_phi) * (2.0 * np.pi / n_phi)
    grid_theta, grid_phi = np.meshgrid(theta, phi, indexing="ij")
    spherical = np.stack(
        [np.full(grid_theta.shape, radius), grid_theta, grid_phi], axis=-1
    ).reshape(-1, 3)
    return _clip_to_domain(spherical_to_cartesian(spherical), inner_radius, outer_radius)


def _reference_angular_resolution(field: Field, grid: SphericalGrid) -> tuple[int, int]:
    """Return the ``(n_theta, n_phi)`` the boundary supersampling multiplies.

    The field's native angular grid when it is grid-backed (a
    :class:`~qorona.field.sampled.SampledField`, so the default 4x places 16 boundary samples per
    native mesh point), else the interior volume grid for a grid-free analytic field.
    """
    native = getattr(field, "grid", None)
    if isinstance(native, SphericalGrid):
        return native.n_theta, native.n_phi
    return grid.n_theta, grid.n_phi


def _build_boundary_map(
    field: Field,
    radius: float,
    n_theta: int,
    n_phi: int,
    *,
    rtol: float,
    cfl: float,
    max_steps: int,
    max_reversals: int,
    turn_guard: TurnGuard,
    chunk_size: int,
    device: str = "auto",
    precision: str = "mixed",
    show_progress: bool,
    label: str,
) -> _BoundaryMap:
    """Precompute log₁₀ Q⊥ on one boundary sphere and return its cubic interpolant.

    Seeds a cell-centred ``(θ, φ)`` grid at ``radius`` (so no seed sits on a pole) and runs the
    squashing engine on it in batches. On-boundary seeding is the degenerate near-boundary limit:
    one half-line vanishes and the formula collapses to the single-mapping form.
    """
    seeds = _sphere_seed_grid(
        radius, n_theta, n_phi, field.domain.inner_radius, field.domain.outer_radius
    )

    def squash(chunk: np.ndarray) -> np.ndarray:
        from qorona.squashing import compute_squashing

        return compute_squashing(
            field,
            chunk,
            rtol=rtol,
            cfl=cfl,
            max_steps=max_steps,
            max_reversals=max_reversals,
            turn_guard=turn_guard,
            device=device,
            precision=precision,
            show_progress=False,
        ).q_perp

    q_perp = np.empty(seeds.shape[0])
    for start, stop, chunk_q in _stream_chunks(
        seeds,
        squash,
        chunk_size=chunk_size,
        batch=_stream_batch(device),
        label=label,
        show_progress=show_progress,
    ):
        q_perp[start:stop] = chunk_q

    return _BoundaryMap(_safe_log10(q_perp).reshape(n_theta, n_phi))


def _build_boundary_maps(
    field: Field,
    grid: SphericalGrid,
    supersample: int,
    *,
    rtol: float,
    cfl: float,
    max_steps: int,
    max_reversals: int,
    turn_guard: TurnGuard,
    chunk_size: int,
    device: str = "auto",
    precision: str = "mixed",
    show_progress: bool,
) -> tuple[_BoundaryMap, _BoundaryMap, int, int]:
    """Precompute the inner and outer boundary Q⊥ maps shared by the per-voxel and paint builders.

    Returns the two cubic interpolants and the ``(n_theta_b, n_phi_b)`` boundary resolution
    (``supersample`` times the field's native angular grid). The interior fill (the per-voxel
    builder) and the swept-path paint (the paint builder) both consume exactly these maps.
    """
    n_theta_b, n_phi_b = (supersample * n for n in _reference_angular_resolution(field, grid))
    inner_map = _build_boundary_map(
        field,
        float(grid.radii[0]),
        n_theta_b,
        n_phi_b,
        rtol=rtol,
        cfl=cfl,
        max_steps=max_steps,
        max_reversals=max_reversals,
        turn_guard=turn_guard,
        chunk_size=chunk_size,
        device=device,
        precision=precision,
        show_progress=show_progress,
        label=f"Precomputing inner-boundary Q⊥ ({n_theta_b}x{n_phi_b})",
    )
    outer_map = _build_boundary_map(
        field,
        float(grid.radii[-1]),
        n_theta_b,
        n_phi_b,
        rtol=rtol,
        cfl=cfl,
        max_steps=max_steps,
        max_reversals=max_reversals,
        turn_guard=turn_guard,
        chunk_size=chunk_size,
        device=device,
        precision=precision,
        show_progress=show_progress,
        label=f"Precomputing outer-boundary Q⊥ ({n_theta_b}x{n_phi_b})",
    )
    return inner_map, outer_map, n_theta_b, n_phi_b


def _lookup_feet(
    feet: np.ndarray, ends: np.ndarray, inner_map: _BoundaryMap, outer_map: _BoundaryMap
) -> np.ndarray:
    """Look up the precomputed boundary log₁₀ Q⊥ at each foot, on the sphere its end names.

    Parameters
    ----------
    feet
        ``(m, 2, 3)`` the two foot positions per line, axis 1 ``(backward, forward)``.
    ends
        ``(m, 2)`` :class:`~qorona.trace.Endpoint` code per foot.
    inner_map, outer_map
        The two boundary interpolants.

    Returns
    -------
    numpy.ndarray
        ``(m, 2)`` log₁₀ Q⊥ read at each foot; ``NaN`` for a foot that landed on neither sphere
        (an incomplete end) or whose boundary map cell is ``NaN``.
    """
    foot_log_q = np.full(feet.shape[:2], np.nan)
    spherical = cartesian_to_spherical(feet)
    theta, phi = spherical[..., 1], spherical[..., 2]
    for boundary_map, code in ((inner_map, Endpoint.INNER), (outer_map, Endpoint.OUTER)):
        on_sphere = ends == code
        if on_sphere.any():
            foot_log_q[on_sphere] = boundary_map(theta[on_sphere], phi[on_sphere])
    return foot_log_q


def _combine_feet(foot_log_q: np.ndarray) -> np.ndarray:
    """Average the two feet in linear Q⊥ and return log₁₀ of the mean (peak-preserving).

    Q⊥ is constant along the line, so the two feet should agree; the average damps boundary-map
    interpolation mismatch. Averaging in **linear** Q⊥ (not log) keeps a thin high-Q sheet where the
    two feet straddle a connectivity discontinuity: a linear mean ≈ the max, a log mean would wash
    it out. ``NaN`` at either foot makes the combined value ``NaN``.

    Parameters
    ----------
    foot_log_q
        ``(m, 2)`` log₁₀ Q⊥ at the two feet.

    Returns
    -------
    numpy.ndarray
        ``(m,)`` combined log₁₀ Q⊥.
    """
    linear = 0.5 * (10.0 ** foot_log_q[:, 0] + 10.0 ** foot_log_q[:, 1])
    return _safe_log10(linear)


def _combine_polarity(
    field: Field,
    feet: np.ndarray,
    ends: np.ndarray,
    inner_radius: float,
    outer_radius: float,
    *,
    closed: str = "neutral",
) -> np.ndarray:
    """Return ``(m,)`` inner-footpoint polarity in ``{-1, 0, +1}`` per line.

    The polarity of a line is the sign of ``B·r̂`` at its **inner** footpoint, the photospheric
    rooting the structures of interest carry, matching the field-line view's convention
    (``render/fieldlines.py``). B is sampled at the inner feet and *then* signed, so the smooth
    field is interpolated rather than a discrete sign (which cannot be interpolated).

    An open line has one inner foot → its sign. A closed loop has two inner feet of opposite
    polarity; ``closed="neutral"`` (the default) averages their signs to ``0`` (no single rooting
    polarity), while ``closed="dominant"`` takes the sign of the stronger-``|B·r̂|`` foot. A foot on
    the outer sphere or an incomplete end contributes nothing; a line with no inner foot is ``0``.

    Parameters
    ----------
    feet
        ``(m, 2, 3)`` the two foot positions per line.
    ends
        ``(m, 2)`` :class:`~qorona.trace.Endpoint` code per foot.
    inner_radius, outer_radius
        Shell bounds; feet are nudged strictly inside before sampling B.
    closed
        ``"neutral"`` (closed loops → ``0``) or ``"dominant"`` (closed loops →
        the stronger foot's sign).
    """
    if closed not in ("neutral", "dominant"):
        raise ValueError(f"closed must be 'neutral' or 'dominant', not {closed!r}")
    is_inner = ends == Endpoint.INNER
    # Only an inner foot carries a polarity, and only a clean INNER foot is finite (an
    # incomplete line's foot is NaN), so B is sampled at the inner feet alone. The other
    # feet keep a placeholder zero that the is_inner mask below never lets contribute.
    feet_in = _clip_to_domain(feet.reshape(-1, 3), inner_radius, outer_radius)
    inner_flat = is_inner.reshape(-1)
    b_radial_flat = np.zeros(feet_in.shape[0])
    if inner_flat.any():
        inner_feet = feet_in[inner_flat]
        b = field.sample(inner_feet, gradient=False).b
        b_radial_flat[inner_flat] = np.sum(b * inner_feet, axis=1)  # sign matches B·r̂
    b_radial = b_radial_flat.reshape(feet.shape[0], 2)
    if closed == "dominant":
        strength = np.where(is_inner, np.abs(b_radial), -1.0)
        pick = np.argmax(strength, axis=1)
        chosen = b_radial[np.arange(b_radial.shape[0]), pick]
        return np.where(is_inner.any(axis=1), np.sign(chosen), 0.0)
    # Sign of the summed inner-foot signs: a single inner foot keeps its sign, a closed loop's two
    # opposite feet cancel to 0 (neutral), and a line with no inner foot is 0.
    sign_sum = np.where(is_inner, np.sign(b_radial), 0.0).sum(axis=1)
    return np.where(is_inner.any(axis=1), np.sign(sign_sum), 0.0)


def build_volume_per_voxel(
    field: Field,
    grid: SphericalGrid,
    *,
    supersample: int = 4,
    closed: str = "neutral",
    rtol: float = 1e-4,
    cfl: float = 0.5,
    max_steps: int = 10_000,
    max_reversals: int = 8,
    turn_guard: TurnGuard = DEFAULT_TURN_GUARD,
    chunk_size: int = _CHUNK_SIZE,
    workers: int | None = None,
    device: str = "auto",
    precision: str = "mixed",
    show_progress: bool = True,
) -> QPerpVolume:
    """Build the production Q⊥ volume by precomputing on the boundaries and recovering the interior.

    Three steps. First, run :func:`~qorona.squashing.compute_squashing` to map log₁₀ Q⊥ on the
    supersampled inner and outer boundary spheres. Then trace every interior voxel position-only to
    its two feet. Finally, read the precomputed boundary value at each foot (cubic interpolation in
    log₁₀ Q⊥, keyed off the foot's :class:`~qorona.trace.Endpoint`) and combine the two by a
    peak-preserving linear-Q⊥ average. The fill is branch-free in open/closed (closed ⇒ two inner
    feet, open ⇒ one inner + one outer); a voxel is ``NaN`` if its line is incomplete or either
    boundary value is ``NaN``. The interior trace is streamed in fixed-size batches so peak memory
    is independent of the voxel count.

    Parameters
    ----------
    field
        The field to build on.
    grid
        The interior volume grid (its pitch sets the thinnest renderable sheet).
    supersample
        Boundary-sphere angular resolution, as a multiple of the field's native angular grid (the
        interior grid for a grid-free analytic field; see :func:`_reference_angular_resolution`).
        The default 4x places 16 boundary samples per native mesh point, orthogonal to the interior
        ``grid`` pitch (the boundary maps cap map fidelity; the interior pitch caps sheet width).
    closed
        Closed-loop polarity convention for the polarity channel: ``"neutral"`` (default, the two
        opposite inner feet average to 0) or ``"dominant"`` (the stronger-``|B·r̂|`` foot's sign).
        See :func:`_combine_polarity`.
    rtol, cfl, max_steps, max_reversals, turn_guard
        Forwarded to the boundary squashing seeds and the interior tracer (the stall guard and the
        sharp-turn guard apply to both).
    chunk_size
        Voxels (and boundary seeds) processed per batch.
    workers
        numba thread count for the kernel (``None`` = all cores; ``1`` = serial). Ignored without
        numba; the NumPy fallback is single-core.
    device
        Compute backend: ``"auto"`` (GPU if available, else CPU), ``"gpu"``, or ``"cpu"``. Forwarded
        to the boundary squashing seeds and the interior tracer, which route to the CUDA tier on the
        GPU.
    precision
        CUDA kernel precision (GPU tier only; the CPU tiers always run float64): ``"mixed"``
        (default), ``"float64"``, or ``"float32"``. See :class:`~qorona.config.VolumeConfig`.
    show_progress
        Whether to display progress.

    Returns
    -------
    QPerpVolume
        The truthful log₁₀ Q⊥ volume.
    """
    nodes = _clip_to_domain(
        grid.node_points().reshape(-1, 3), float(grid.radii[0]), float(grid.radii[-1])
    )
    apply_workers(workers)
    _warm_kernels(
        field, grid, paint=False, device=device, precision=precision, show_progress=show_progress
    )
    inner_map, outer_map, n_theta_b, n_phi_b = _build_boundary_maps(
        field,
        grid,
        supersample,
        rtol=rtol,
        cfl=cfl,
        max_steps=max_steps,
        max_reversals=max_reversals,
        turn_guard=turn_guard,
        chunk_size=chunk_size,
        device=device,
        precision=precision,
        show_progress=show_progress,
    )

    # The interior trace is the cost; the per-foot boundary lookup + linear-Q combine are cheap and
    # stay here, per chunk, so the boundary maps are never shipped into the kernel.
    def trace(chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lines = trace_field_lines(
            field,
            chunk,
            rtol=rtol,
            cfl=cfl,
            max_steps=max_steps,
            max_reversals=max_reversals,
            turn_guard=turn_guard,
            device=device,
            precision=precision,
            show_progress=False,
        )
        return lines.feet, lines.ends

    inner_radius, outer_radius = float(grid.radii[0]), float(grid.radii[-1])
    log_q = np.empty(nodes.shape[0])
    polarity = np.empty(nodes.shape[0])
    for start, stop, feet_ends in _stream_chunks(
        nodes,
        trace,
        chunk_size=chunk_size,
        batch=_stream_batch(device),
        label="Filling Q⊥ volume interior (trace to boundaries)",
        show_progress=show_progress,
    ):
        feet, ends = feet_ends
        foot_log_q = _lookup_feet(feet, ends, inner_map, outer_map)
        log_q[start:stop] = _combine_feet(foot_log_q)
        polarity[start:stop] = _combine_polarity(
            field, feet, ends, inner_radius, outer_radius, closed=closed
        )

    volume = _pack_volume(grid, log_q, polarity)
    if show_progress:
        print_success(
            f"Built per-voxel Q⊥ volume on {grid.n_r}x{grid.n_theta}x{grid.n_phi} grid "
            f"(boundaries {n_theta_b}x{n_phi_b})"
        )
    return volume


def _line_voxels(path: np.ndarray, grid: SphericalGrid, paint_step: float) -> np.ndarray:
    """Return the flat node indices of the voxels a swept path crosses, deduped consecutively.

    The path is resampled by arc length at a pitch of ``paint_step`` times its smallest local cell
    extent (so a roughly-straight segment skips no voxel), each sample forward-binned to its
    ``(i_r, i_θ, i_φ)`` cell (φ wrapped, θ/r clipped to the node range), then flattened C-order
    to match :func:`_pack_volume`. Consecutive duplicates are dropped so a line, which visits a
    voxel in one contiguous run, deposits its value once per voxel (the painter's per-line dedup).

    Parameters
    ----------
    path
        ``(m, 3)`` ordered Cartesian polyline of one field line (foot → foot, through the seed).
    grid
        The volume grid the path is binned into.
    paint_step
        Along-line sample pitch as a fraction of the local cell extent.

    Returns
    -------
    numpy.ndarray
        ``(k,)`` flat node indices, with no two consecutive entries equal.
    """
    if path.shape[0] >= 2:
        cumulative = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        )
        total = float(cumulative[-1])
    else:
        total = 0.0

    if total <= 0.0:
        points = path[:1]
    else:
        # Pitch from the smallest cell on the path; the cap floors it so a tiny near-pole cell can
        # never demand a runaway sample count.
        pitch = max(paint_step * float(grid.cell_extent(path).min()), total / _MAX_PAINT_SAMPLES)
        s_samples = np.linspace(0.0, total, int(total / pitch) + 1)
        points = np.stack([np.interp(s_samples, cumulative, path[:, d]) for d in range(3)], axis=-1)

    index, _ = grid.index_coordinates(points)
    cell = np.floor(index).astype(np.intp)
    i_r = np.clip(cell[:, 0], 0, grid.n_r - 1)
    i_theta = np.clip(cell[:, 1], 0, grid.n_theta - 1)
    i_phi = cell[:, 2] % grid.n_phi
    flat = (i_r * grid.n_theta + i_theta) * grid.n_phi + i_phi

    keep = np.ones(flat.shape, dtype=bool)
    keep[1:] = flat[1:] != flat[:-1]
    return flat[keep]


def _paint_lines_numpy(
    field: Field,
    seeds: np.ndarray,
    grid: SphericalGrid,
    inner_map: _BoundaryMap,
    outer_map: _BoundaryMap,
    *,
    paint_step: float,
    closed: str,
    sum_q: np.ndarray,
    count: np.ndarray,
    sum_pol: np.ndarray,
    rtol: float,
    cfl: float,
    max_steps: int,
    max_reversals: int,
    turn_guard: TurnGuard,
    chunk_size: int,
    device: str = "auto",
    show_progress: bool,
) -> int:
    """Paint each line's Q⊥ and footpoint polarity along its swept path, per voxel.

    The single-threaded reference painter the kernel is validated against. Each seed is traced both
    ways with ``store_path=True`` (which routes to the NumPy integrator, so this path is serial by
    construction), its line value read from the boundary maps and combined exactly as
    :func:`build_volume_per_voxel` does, and that value scattered into every voxel the line's path
    crosses (:func:`_line_voxels`). The line's inner-footpoint polarity (:func:`_combine_polarity`)
    is scattered the same way, **weighted by its linear Q⊥**, so once ``sum_pol`` is signed the
    dominant (highest-Q) line through a voxel sets its sign. Incomplete lines (a ``NaN`` combined
    value) are skipped. Seeds stream in fixed-size chunks so the stored paths never exceed one chunk
    in memory; ``sum_q``, ``count`` and ``sum_pol`` are mutated in place. Returns the lines painted.
    """
    inner_radius, outer_radius = float(grid.radii[0]), float(grid.radii[-1])
    n = len(seeds)
    batch = min(chunk_size, _stream_batch(device))
    painted_lines = 0
    with progress_bar("Painting Q⊥ volume (NumPy)", n, enabled=show_progress) as progress:
        for start in range(0, n, batch):
            stop = min(start + batch, n)
            lines = trace_field_lines(
                field,
                seeds[start:stop],
                rtol=rtol,
                cfl=cfl,
                max_steps=max_steps,
                max_reversals=max_reversals,
                turn_guard=turn_guard,
                device=device,
                store_path=True,
                show_progress=False,
            )
            values = _combine_feet(_lookup_feet(lines.feet, lines.ends, inner_map, outer_map))
            polarities = _combine_polarity(
                field, lines.feet, lines.ends, inner_radius, outer_radius, closed=closed
            )
            assert lines.paths is not None  # store_path=True guarantees paths
            for path, value, sign in zip(lines.paths, values, polarities, strict=True):
                if not np.isfinite(value):
                    continue
                voxels = _line_voxels(path, grid, paint_step)
                line_q = 10.0**value
                np.add.at(sum_q, voxels, line_q)
                np.add.at(count, voxels, 1.0)
                np.add.at(sum_pol, voxels, sign * line_q)
                painted_lines += 1
            progress(stop)
    return painted_lines


def _paint_lines_jit(
    field: Field,
    seeds: np.ndarray,
    grid: SphericalGrid,
    values: np.ndarray,
    polarities: np.ndarray,
    *,
    paint_step: float,
    sum_q: np.ndarray,
    count: np.ndarray,
    sum_pol: np.ndarray,
    rtol: float,
    cfl: float,
    max_steps: int,
    chunk_size: int,
    device: str = "auto",
    precision: str = "mixed",
    show_progress: bool,
) -> int:
    """Trace and paint the seeded lines via the numba or CUDA kernel, scattering each value in.

    The production painter (the trace-parallel + paint-serial scheme): the kernel traces each seed
    and emits its deduped swept voxel indices, each lane writing only its own row, so no atomics,
    and this serial loop applies the precomputed per-line value (the boundary-map combine in
    ``values``, ``NaN`` for incomplete lines, which are skipped) once per voxel into ``sum_q`` and
    ``count``, and the per-line polarity (``polarities``) weighted by linear Q⊥ into ``sum_pol``.
    Seeds stream in chunks sized so the per-chunk index buffer stays bounded (:data:`_PAINT_BUFFER`)
    regardless of grid resolution. Returns the number of lines painted.
    """
    from qorona.trace.integrator import _ATOL_POS

    use_gpu = resolve_device(device) == "gpu" and HAVE_CUDA

    jit_field = field._jit_field()  # type: ignore[attr-defined]
    jit_grid = grid._jit_grid()
    assert jit_grid is not None  # build_volume_paint's kernel-path guard ensures this
    atol = np.full(3, _ATOL_POS)
    max_deposits = 4 * (grid.n_r + grid.n_theta + grid.n_phi)
    n = len(seeds)

    if use_gpu:
        # GPU: trace + on-device atomic scatter into tiled f64 accumulators.
        return _paint_lines_cuda(
            seeds,
            values,
            polarities,
            jit_field,
            jit_grid,
            atol,
            max_deposits,
            sum_q=sum_q,
            count=count,
            sum_pol=sum_pol,
            paint_step=paint_step,
            rtol=rtol,
            cfl=cfl,
            max_steps=max_steps,
            chunk_size=chunk_size,
            device=device,
            precision=precision,
            show_progress=show_progress,
        )

    # CPU (numba) path: trace-parallel + serial host scatter into the host accumulators.
    from qorona.accel.kernels import paint_batch_jit as paint_batch

    paint_chunk = max(1, min(chunk_size, _stream_batch(device), _PAINT_BUFFER // max_deposits))
    deposit_slot = np.arange(max_deposits)
    painted_lines = 0
    overflow_lines = 0
    with progress_bar("Painting Q⊥ volume (numba)", n, enabled=show_progress) as progress:
        for start in range(0, n, paint_chunk):
            stop = min(start + paint_chunk, n)
            chunk_values = values[start:stop]
            chunk_valid = np.isfinite(chunk_values)
            voxels, counts, overflow = paint_batch(
                np.ascontiguousarray(seeds[start:stop]),
                np.ascontiguousarray(chunk_valid),
                jit_field,
                jit_grid,
                atol,
                float(rtol),
                float(cfl),
                int(max_steps),
                float(paint_step),
                int(max_deposits),
            )
            # Flatten the deduped per-line deposits and add each line's value once per swept voxel.
            selected = deposit_slot[None, :] < counts[:, None]
            flat_voxels = voxels[selected]
            line_q = np.where(chunk_valid, 10.0**chunk_values, 0.0)
            np.add.at(sum_q, flat_voxels, np.repeat(line_q, counts))
            np.add.at(count, flat_voxels, 1.0)
            line_pol_q = np.where(chunk_valid, polarities[start:stop] * 10.0**chunk_values, 0.0)
            np.add.at(sum_pol, flat_voxels, np.repeat(line_pol_q, counts))
            painted_lines += int(chunk_valid.sum())
            overflow_lines += int(overflow.sum())
            progress(stop)

    if overflow_lines and show_progress:
        print_warning(
            f"{overflow_lines} lines exceeded the per-line voxel cap ({max_deposits}); their tails "
            f"were dropped; raise paint_step if this is pervasive."
        )
    return painted_lines


def _paint_lines_cuda(
    seeds: np.ndarray,
    values: np.ndarray,
    polarities: np.ndarray,
    jit_field: JitField,
    jit_grid: JitGrid,
    atol: np.ndarray,
    max_deposits: int,
    *,
    sum_q: np.ndarray,
    count: np.ndarray,
    sum_pol: np.ndarray,
    paint_step: float,
    rtol: float,
    cfl: float,
    max_steps: int,
    chunk_size: int,
    device: str,
    precision: str,
    show_progress: bool,
    n_tiles: int | None = None,
) -> int:
    """GPU paint + accumulate: on-device atomic scatter into f64 accumulators, tiled to fit VRAM.

    The volume's f64 ``sum_q`` / ``count`` / ``sum_pol`` are partitioned into ``n_tiles`` contiguous
    flat-index tiles sized to free VRAM (a caller-supplied ``n_tiles`` overrides the probe, the
    validation hook); each tile scatters only its in-range deposits on-device, then copies the
    finished tile into the host accumulators, so the build runs in bounded device memory at any
    resolution. The seeds are traced **once**: the first tile traces and scatters
    (:func:`~qorona.accel.cuda_kernels.paint_scatter_collect_batch_cuda`), keeping each chunk's
    compacted voxel runs on the host, and the later tiles replay those runs
    (:func:`~qorona.accel.cuda_kernels.scatter_runs_cuda`) without re-tracing. A run store that
    would exceed half the available host memory (extrapolated from the first chunk) falls back to
    re-tracing the later tiles (:func:`~qorona.accel.cuda_kernels.paint_and_scatter_batch_cuda`),
    which is correct, just slower. ``n_tiles == 1`` when the whole volume fits one tile.
    """
    from numba import cuda

    from qorona.accel.cuda_kernels import (
        paint_and_scatter_batch_cuda,
        paint_scatter_collect_batch_cuda,
        scatter_runs_cuda,
    )

    n = len(seeds)
    n_nodes = sum_q.shape[0]
    # Upload the read-only field once and reuse it across every chunk and tile: at fine field
    # ``b_padded`` is ~1 GB, and the small per-line voxel cap forces many small paint chunks, so a
    # per-chunk re-upload would swamp the build in PCIe traffic. The kernel reads it unchanged.
    bpad_dtype = np.float32 if precision != "float64" else np.float64
    with _cuda_alloc_guard(f"the traced field ({jit_field.b_padded.nbytes / 1e9:.1f} GB)"):
        d_bpad = cuda.to_device(np.ascontiguousarray(jit_field.b_padded, dtype=bpad_dtype))

    # The probed VRAM grants (taken after the field upload): the per-chunk swept-voxel buffer
    # ``(chunk, max_deposits)`` int32 lives on the *device*, so the paint chunk is sized to its
    # grant rather than the host ``_PAINT_BUFFER`` (keeping fine-field chunks large: good
    # occupancy, few launches), and the 3 x n_nodes x 8 B f64 accumulators are tiled to theirs;
    # n_tiles == 1 when the whole volume fits one tile.
    budget = gpu_memory_budget()
    chunk_grant = budget.voxel_buffer_bytes // (max_deposits * 4)
    paint_chunk = max(1, min(int(chunk_size), _stream_batch(device), chunk_grant))
    if n_tiles is None:
        n_tiles = max(1, -(-3 * n_nodes * 8 // budget.accumulator_bytes))
    tile_size = -(-n_nodes // n_tiles)

    valid_all = np.isfinite(values)
    line_q_all = np.where(valid_all, 10.0**values, 0.0)
    line_pol_all = np.where(valid_all, polarities * 10.0**values, 0.0)
    painted_lines = int(valid_all.sum())
    overflow_lines = 0

    # Multi-tile: replay each chunk's compacted runs for tiles > 0 unless the extrapolated store
    # would not fit the host budget, in which case the later tiles re-trace instead.
    replay = n_tiles > 1
    replay_budget = _available_host_bytes() // 2
    runs: list[tuple[int, int, np.ndarray, np.ndarray]] = []

    with progress_bar("Painting Q⊥ volume (CUDA)", n_tiles * n, enabled=show_progress) as progress:
        for t in range(n_tiles):
            lo = t * tile_size
            hi = min(lo + tile_size, n_nodes)
            with _cuda_alloc_guard(f"an accumulator tile ({3 * (hi - lo) * 8 / 1e9:.1f} GB)"):
                d_sum_q = cuda.to_device(np.zeros(hi - lo, dtype=np.float64))
                d_count = cuda.to_device(np.zeros(hi - lo, dtype=np.float64))
                d_sum_pol = cuda.to_device(np.zeros(hi - lo, dtype=np.float64))
            if t > 0 and replay:
                for start, stop, flat, offsets in runs:
                    scatter_runs_cuda(
                        flat,
                        offsets,
                        line_q_all[start:stop],
                        line_pol_all[start:stop],
                        d_sum_q,
                        d_count,
                        d_sum_pol,
                        int(lo),
                        int(hi),
                    )
                    progress(t * n + stop)
            else:
                for start in range(0, n, paint_chunk):
                    stop = min(start + paint_chunk, n)
                    # Host-side chunk inputs only; the device accumulators and the staged field are
                    # passed separately so no reference to them outlives the chunk call.
                    chunk_args = (
                        np.ascontiguousarray(seeds[start:stop]),
                        np.ascontiguousarray(valid_all[start:stop]),
                        np.ascontiguousarray(line_q_all[start:stop]),
                        np.ascontiguousarray(line_pol_all[start:stop]),
                        jit_field,
                        jit_grid,
                        atol,
                        float(rtol),
                        float(cfl),
                        int(max_steps),
                        float(paint_step),
                        int(max_deposits),
                    )
                    if t == 0 and replay:
                        overflow, flat, offsets = paint_scatter_collect_batch_cuda(
                            *chunk_args,
                            d_sum_q,
                            d_count,
                            d_sum_pol,
                            int(lo),
                            int(hi),
                            d_bpad,
                            precision=precision,
                        )
                        runs.append((start, stop, flat, offsets))
                        if start == 0:
                            n_chunks = -(-n // paint_chunk)
                            projected = (flat.nbytes + offsets.nbytes) * n_chunks
                            if projected > replay_budget:
                                replay = False
                                runs.clear()
                    else:
                        overflow = paint_and_scatter_batch_cuda(
                            *chunk_args,
                            d_sum_q,
                            d_count,
                            d_sum_pol,
                            int(lo),
                            int(hi),
                            d_bpad,
                            precision=precision,
                        )
                    if t == 0:
                        overflow_lines += overflow
                    progress(t * n + stop)
            sum_q[lo:hi] = d_sum_q.copy_to_host()
            count[lo:hi] = d_count.copy_to_host()
            sum_pol[lo:hi] = d_sum_pol.copy_to_host()
            del d_sum_q, d_count, d_sum_pol
    runs.clear()

    if overflow_lines and show_progress:
        print_warning(
            f"{overflow_lines} lines exceeded the per-line voxel cap ({max_deposits}); their tails "
            f"were dropped; raise paint_step if this is pervasive."
        )
    return painted_lines


def build_volume_paint(
    field: Field,
    grid: SphericalGrid,
    *,
    supersample: int = 4,
    paint_step: float = 0.5,
    closed: str = "neutral",
    rtol: float = 1e-4,
    cfl: float = 0.5,
    max_steps: int = 10_000,
    max_reversals: int = 8,
    turn_guard: TurnGuard = DEFAULT_TURN_GUARD,
    chunk_size: int = _CHUNK_SIZE,
    workers: int | None = None,
    device: str = "auto",
    precision: str = "mixed",
    show_progress: bool = True,
    timings: dict[str, float] | None = None,
) -> QPerpVolume:
    """Build the Q⊥ volume by painting seeded lines along their swept paths into the grid.

    The cheap builder: because Q⊥ is constant along a field line, one trace fills *every* voxel the
    line crosses, so the trace count is set by the number of **seeds** (a coverage knob) rather than
    the voxel count, decoupling resolution from cost. It reuses :func:`build_volume_per_voxel`'s
    machinery for the per-line value (the same boundary maps and linear-Q foot-combine), so the only
    difference from that builder is the spreading. Four steps:

    1. precompute log₁₀ Q⊥ on the supersampled inner and outer boundary spheres (shared with
       :func:`build_volume_per_voxel`);
    2. seed both boundary surfaces at the boundary-map resolution;
    3. trace each seed to its two feet, then read and linear-Q-combine the boundary values into the
       line's one Q⊥;
    4. paint that value along the line's full swept path, accumulating ``Σ Q⊥`` and a line ``count``
       per voxel, then storing ``log₁₀(Σ Q⊥ / count)`` (the linear-Q overlap average) where any line
       painted, ``NaN`` elsewhere.

    Alongside Q⊥, each line's inner-footpoint polarity (:func:`_combine_polarity`) is painted along
    the same swept path and combined per voxel **weighted by linear Q⊥**, the sign of the dominant
    (highest-Q) line through a voxel, neutral (0) where opposite polarities cancel, populating the
    volume's polarity channel for a polarity-coloured render.

    Seeding both surfaces fills the high-corona radial fan that inner-only seeding leaves gappy;
    residual inter-line gaps stay ``NaN`` (the render's weight-normalised integral tolerates them),
    and the painted-voxel fraction is reported so coverage is auditable.

    Parameters
    ----------
    field
        The field to build on.
    grid
        The interior volume grid (its pitch sets the thinnest renderable sheet).
    supersample
        Boundary-sphere and per-surface seed resolution, as a multiple of the field's native angular
        grid (the interior grid for a grid-free analytic field; see
        :func:`_reference_angular_resolution`). Sets both the boundary-map fidelity and the seed
        density. **Coverage scales with the *seed* angular density, not with the build cost:** the
        build time is set by the seed count and is nearly independent of ``grid`` resolution, but a
        volume finer than the seeds is painted too sparsely, so its mostly-``NaN`` interior breaks
        the render through the tricubic stencil. Choose ``supersample`` so the seed angular grid is
        at least the volume ``grid``'s, roughly ``grid_angular / field_angular`` when the volume
        out-resolves the field, and check the reported covered fraction.
    paint_step
        Along-line sample pitch as a fraction of the local cell extent (smaller paints denser).
    closed
        Polarity convention for closed loops (two opposite-polarity inner feet): ``"neutral"``
        (default, their signs average to 0) or ``"dominant"`` (the stronger-``|B·r̂|`` foot's sign).
        See :func:`_combine_polarity`.
    rtol, cfl, max_steps, max_reversals, turn_guard
        Forwarded to the boundary squashing seeds and the line traces (the stall guard and the
        sharp-turn guard apply to both).
    chunk_size
        Seeds (and boundary seeds) processed per batch.
    workers
        numba thread count for the kernel (``None`` = all cores; ``1`` = serial). Ignored without
        numba; the NumPy painter is single-core.
    device
        Compute backend: ``"auto"`` (GPU if available, else CPU), ``"gpu"``, or ``"cpu"``. Forwarded
        to the boundary squashing seeds, the line traces, and the paint kernel, which route to the
        CUDA tier on the GPU.
    precision
        CUDA kernel precision (GPU tier only; the CPU tiers always run float64): ``"mixed"``
        (default), ``"float64"``, or ``"float32"``. See :class:`~qorona.config.VolumeConfig`.
    show_progress
        Whether to display progress.
    timings
        Optional dict receiving the per-stage durations in seconds (``"boundary"`` boundary maps,
        ``"trace"`` per-line Q⊥ values, ``"paint"`` trace+scatter), for the end-of-run summary.
        The NumPy fallback path records only ``"boundary"``.

    Returns
    -------
    QPerpVolume
        The truthful log₁₀ Q⊥ volume; ``NaN`` in voxels no line painted.
    """
    if closed not in ("neutral", "dominant"):
        raise ValueError(f"closed must be 'neutral' or 'dominant', not {closed!r}")
    inner_radius = float(grid.radii[0])
    outer_radius = float(grid.radii[-1])
    apply_workers(workers)
    _warm_kernels(
        field, grid, paint=True, device=device, precision=precision, show_progress=show_progress
    )
    stage_start = time.perf_counter()
    inner_map, outer_map, n_theta_b, n_phi_b = _build_boundary_maps(
        field,
        grid,
        supersample,
        rtol=rtol,
        cfl=cfl,
        max_steps=max_steps,
        max_reversals=max_reversals,
        turn_guard=turn_guard,
        chunk_size=chunk_size,
        device=device,
        precision=precision,
        show_progress=show_progress,
    )
    if timings is not None:
        timings["boundary"] = time.perf_counter() - stage_start

    seeds = np.concatenate(
        [
            _sphere_seed_grid(inner_radius, n_theta_b, n_phi_b, inner_radius, outer_radius),
            _sphere_seed_grid(outer_radius, n_theta_b, n_phi_b, inner_radius, outer_radius),
        ]
    )

    n_nodes = grid.n_r * grid.n_theta * grid.n_phi
    sum_q = np.zeros(n_nodes)
    count = np.zeros(n_nodes)
    sum_pol = np.zeros(n_nodes)  # Σ sign·Q⊥ per voxel; its sign is the dominant-line polarity

    jit_field = field._jit_field() if HAVE_NUMBA else None  # type: ignore[attr-defined]
    if jit_field is not None and grid._jit_grid() is not None:
        # Kernel path: a fast feet-only trace gives each line's value (the boundary-map combine),
        # then the paint kernel re-traces and rasterizes each value along its swept path. The feet
        # trace is streamed (not one launch over all seeds at once) so the bar shows a live ETA and
        # each launch re-balances the prange idle tail: the long ``max-steps`` lines otherwise
        # cluster onto a few threads in one static-scheduled launch, leaving a slow tail.
        values = np.empty(len(seeds))
        polarities = np.empty(len(seeds))
        # closed, open, null, stalled, deflected, max-steps (trace diagnostics).
        tally = np.zeros(6, dtype=np.int64)

        def trace_values(chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            lines = trace_field_lines(
                field,
                chunk,
                rtol=rtol,
                cfl=cfl,
                max_steps=max_steps,
                max_reversals=max_reversals,
                turn_guard=turn_guard,
                device=device,
                precision=precision,
                show_progress=False,
            )
            null = lines.is_null
            stalled = lines.is_stalled
            deflected = lines.is_deflected
            tally[:] += (
                int(lines.is_closed.sum()),
                int(lines.is_open.sum()),
                int(null.sum()),
                int(stalled.sum()),
                int(deflected.sum()),
                int((lines.is_incomplete & ~null & ~stalled & ~deflected).sum()),
            )
            line_values = _combine_feet(_lookup_feet(lines.feet, lines.ends, inner_map, outer_map))
            line_polarity = _combine_polarity(
                field, lines.feet, lines.ends, inner_radius, outer_radius, closed=closed
            )
            return line_values, line_polarity

        stage_start = time.perf_counter()
        for start, stop, (chunk_values, chunk_polarity) in _stream_chunks(
            seeds,
            trace_values,
            chunk_size=chunk_size,
            batch=_stream_batch(device),
            label="Tracing field lines for Q⊥ values",
            show_progress=show_progress,
        ):
            values[start:stop] = chunk_values
            polarities[start:stop] = chunk_polarity
        if timings is not None:
            timings["trace"] = time.perf_counter() - stage_start
        if show_progress:
            print_success(
                f"Traced {len(seeds)} field lines: {tally[0]} closed · {tally[1]} open · "
                f"{tally[2]} null · {tally[3]} stalled · {tally[4]} deflected · "
                f"{tally[5]} max-steps"
            )
        stage_start = time.perf_counter()
        painted_lines = _paint_lines_jit(
            field,
            seeds,
            grid,
            values,
            polarities,
            paint_step=paint_step,
            sum_q=sum_q,
            count=count,
            sum_pol=sum_pol,
            rtol=rtol,
            cfl=cfl,
            max_steps=max_steps,
            chunk_size=chunk_size,
            device=device,
            precision=precision,
            show_progress=show_progress,
        )
        if timings is not None:
            timings["paint"] = time.perf_counter() - stage_start
    else:
        painted_lines = _paint_lines_numpy(
            field,
            seeds,
            grid,
            inner_map,
            outer_map,
            paint_step=paint_step,
            closed=closed,
            sum_q=sum_q,
            count=count,
            sum_pol=sum_pol,
            rtol=rtol,
            cfl=cfl,
            max_steps=max_steps,
            max_reversals=max_reversals,
            turn_guard=turn_guard,
            chunk_size=chunk_size,
            device=device,
            show_progress=show_progress,
        )

    # In-place finalize to bound host memory at very high resolution: with hundreds of millions of
    # voxels, host memory cannot hold the three f64 sum arrays, a separate mean, a separate
    # polarity, and the packed volume at once, so reuse ``sum_q`` for the mean and ``sum_pol`` for
    # the polarity, freeing each array the moment it is consumed.
    covered = float((count > 0.0).mean())
    with np.errstate(divide="ignore", invalid="ignore"):
        sum_q /= count  # sum_q -> mean linear Q⊥ (count==0 -> NaN/Inf, fixed next)
    sum_q[count == 0.0] = np.nan  # unpainted voxels carry no Q⊥
    del count
    # Per-voxel polarity = sign of the Q⊥-weighted polarity sum (the dominant high-Q line's sign),
    # in {-1, 0, +1}; 0 (neutral) where unpainted or where opposite polarities cancel.
    np.sign(sum_pol, out=sum_pol)
    log_q = _safe_log10(sum_q)  # mean Q⊥ -> log10 (Inf/NaN -> NaN)
    del sum_q
    volume = _pack_volume(grid, log_q, sum_pol)
    if show_progress:
        print_success(
            f"Painted Q⊥ volume on {grid.n_r}x{grid.n_theta}x{grid.n_phi} grid from "
            f"{painted_lines}/{len(seeds)} lines (boundaries {n_theta_b}x{n_phi_b}); "
            f"{covered:.1%} of voxels covered"
        )
    return volume
