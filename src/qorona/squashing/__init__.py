"""Q⊥ (and the classical-Q diagnostic) by deviation-vector transport: the squashing-factor engine.

The public surface is :func:`compute_squashing`, returning a :class:`SquashingResult` for the
field line through each seed. The engine is **seeding-agnostic**: a seed is only a handle that
selects a line, and the ``B₀²`` normalization makes Q⊥ seed-position-invariant, so the returned
value is the line's boundary-to-boundary, constant-along-the-line squashing factor regardless of
where on the line the seed sits. Where seeds come from is the caller's concern: a validation
colatitude sweep, dense boundary-sphere grids (the Q⊥ volume), per-pixel line-of-sight samples.

Internally: sample B at the seeds (giving ``B₀`` and the seed deviation basis), co-transport the
two deviation vectors ``U, V`` with each line to both feet (:mod:`~qorona.squashing.transport`),
sample B at the feet, and assemble Q⊥ and Q from the master formula
(:mod:`~qorona.squashing.squashing_factor`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qorona.console import print_success, progress_bar
from qorona.field.base import Field
from qorona.squashing.squashing_factor import _assemble_squashing, _seed_basis
from qorona.squashing.transport import _trace_and_transport
from qorona.squashing.volume import (
    QPerpVolume,
    build_volume_boundary,
    build_volume_paint,
    build_volume_per_voxel,
)
from qorona.trace.fieldline import DEFAULT_TURN_GUARD, FieldLines, TurnGuard

__all__ = [
    "QPerpVolume",
    "SquashingResult",
    "build_volume_boundary",
    "build_volume_paint",
    "build_volume_per_voxel",
    "compute_squashing",
]


@dataclass(frozen=True, slots=True)
class SquashingResult:
    """Q⊥ (primary) and Q (secondary diagnostic) for the line through each seed, with geometry.

    Composes the geometric :class:`~qorona.trace.fieldline.FieldLines` (which stays purely
    geometric) and adds the squashing payload. Rows align to the input seeds.

    Attributes
    ----------
    lines
        The traced lines (seeds, feet, ends, lengths, optional paths), reused from the tracer.
    q_perp
        ``(n,)`` perpendicular squashing factor Q⊥ (the primary product); ``NaN`` where the line
        is incomplete (``~lines.is_complete``).
    q
        ``(n,)`` classical squashing factor Q (a diagnostic that inflates at grazing incidence);
        ``NaN`` where the line is incomplete.
    """

    lines: FieldLines
    q_perp: np.ndarray
    q: np.ndarray

    @property
    def valid(self) -> np.ndarray:
        """``(n,)`` bool: lines with a valid squashing factor (exactly ``lines.is_complete``).

        The single source of truth for which rows carry a finite Q⊥/Q: a complete line has two
        real feet; an incomplete one (aborted at a null or a resource guard) carries ``NaN`` Q.
        """
        return self.lines.is_complete

    def summary(self) -> str:
        """Return a one-line end-of-run summary: valid fraction, median Q⊥, and its log range."""
        valid = self.valid
        n_total = valid.size
        n_valid = int(valid.sum())
        if n_valid == 0:
            return f"0/{n_total} lines with a valid Q⊥"
        q_perp = self.q_perp[valid]
        median = float(np.nanmedian(q_perp))
        finite = q_perp[np.isfinite(q_perp) & (q_perp > 0.0)]
        if finite.size:
            log_q = np.log10(finite)
            spread = f"log₁₀ Q⊥ ∈ [{log_q.min():.2f}, {log_q.max():.2f}]"
        else:
            spread = "Q⊥ all non-finite"
        return f"{n_valid}/{n_total} valid · median Q⊥ {median:.3g} · {spread}"


def compute_squashing(
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
) -> SquashingResult:
    """Compute Q⊥ (and the classical-Q diagnostic) for the field line through each seed.

    Each seed's line is traced both ways to the boundary spheres while two transverse deviation
    vectors are co-transported with it; Q⊥ is assembled from the endpoint deviations by the master
    formula, and the classical-Q diagnostic falls out of the same transport. Q⊥ is per line and
    **constant along it**, so two seeds on the same line return the same value (the engine does not
    deduplicate; that is a consumer optimization).

    Parameters
    ----------
    field
        The field to compute on (real :class:`~qorona.field.sampled.SampledField` or analytic).
    seeds
        ``(n, 3)`` Cartesian in-domain seeds in R☉ (any seeding strategy).
    rtol
        Embedded relative error tolerance: the accuracy knob, applied over all nine transported
        components (position and the two deviation vectors).
    cfl
        CFL number in the step ceiling ``h_max = cfl · characteristic_length``; ``0 < cfl < 1``.
    max_steps
        Resource guard: maximum step attempts per half-line.
    max_reversals
        Stall guard: terminate a half-line after this many >90° direction reversals, the signature
        of a line trapped and thrashing at a weak-field null (the current sheet). ``0`` disables it.
    turn_guard
        Sharp-turn guard: terminate a half-line that makes a single sharp turn in the outer corona
        where ``|B|`` is weak (a staircase deflection at a null); see :class:`TurnGuard`.
    store_path
        Whether to also keep each line's full ordered geometric path (memory-heavy).
    show_progress
        Whether to display progress (this is the pipeline's slow stage).

    Returns
    -------
    SquashingResult
        Q⊥, Q, and the geometric :class:`~qorona.trace.fieldline.FieldLines`, aligned to ``seeds``.
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

    # Seed B-sample (out of loop): B̂₀ for the deviation basis and B₀ for the master-formula
    # prefactor, taken once and shared so no two paths disagree on |B₀|.
    seed_sample = field.sample(seeds, gradient=False)
    basis = _seed_basis(seed_sample.b, seed_sample.b_magnitude)

    with progress_bar("Computing squashing factor Q⊥", 2 * n, enabled=show_progress) as progress:
        lines, deviations = _trace_and_transport(
            field,
            seeds,
            basis,
            rtol=rtol,
            cfl=cfl,
            max_steps=max_steps,
            max_reversals=max_reversals,
            turn_guard=turn_guard,
            store_path=store_path,
            progress=progress,
        )

    # Foot B-sample (out of loop): only complete lines have real feet, which sit on a boundary
    # sphere (in-domain, inclusive); incomplete lines keep NaN feet and yield NaN Q.
    valid = lines.is_complete
    b_foot = np.full((n, 2, 3), np.nan)
    b_magnitude_foot = np.full((n, 2), np.nan)
    if valid.any():
        foot_sample = field.sample(lines.feet[valid].reshape(-1, 3), gradient=False)
        b_foot[valid] = foot_sample.b.reshape(-1, 2, 3)
        b_magnitude_foot[valid] = foot_sample.b_magnitude.reshape(-1, 2)

    q_perp, q = _assemble_squashing(
        deviations, seed_sample.b_magnitude, lines.feet, b_foot, b_magnitude_foot, valid
    )

    result = SquashingResult(lines=lines, q_perp=q_perp, q=q)
    if show_progress:
        print_success(f"Computed Q⊥ for {n} field lines: {result.summary()}")
    return result
