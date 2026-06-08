"""Optional numba acceleration surface: the probe and the Field-to-kernel descriptor.

The hot path (the per-line DOPRI5 transport that builds the Q⊥ volume) is accelerated by a
scalar-per-lane nopython kernel in :mod:`qorona.accel.kernels`, run one field line per thread under
``prange``. numba is an **optional** dependency (the ``accel`` extra): when it is absent the tracer
falls back to the validated NumPy integrator, so importing this package never fails. The dispatcher
in :mod:`qorona.trace.integrator` asks :data:`HAVE_NUMBA` once and only imports the kernel module
(which does ``import numba``) when it is available.

A JIT-capable :class:`~qorona.field.base.Field` opts in by implementing ``_jit_field()`` returning a
:class:`JitField`, the raw payload the transport kernel needs, since the ``Field`` ABC, the
grid/spacing dataclasses, and the RHS closures do not survive nopython. The painting kernel
additionally bins points into a *volume* grid described by :class:`JitGrid`, returned by
:meth:`~qorona.resample.grid.SphericalGrid._jit_grid`.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

try:
    import numba as _numba  # noqa: F401

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised only in no-numba installs
    HAVE_NUMBA = False


def apply_workers(workers: int | None) -> None:
    """Set the numba thread count for the accelerated kernels: the shared thread-control helper.

    ``None`` leaves numba at its current count (all cores by default); ``1`` is serial; any value is
    clamped to numba's configured ceiling. A no-op without numba (every NumPy fallback is
    single-threaded). ``set_num_threads`` is process-global, so a stage that leaves ``workers`` at
    ``None`` inherits the count an earlier stage set.
    """
    if not HAVE_NUMBA or workers is None:
        return
    import numba

    numba.set_num_threads(max(1, min(workers, numba.config.NUMBA_NUM_THREADS)))


class JitField(NamedTuple):
    """The raw, nopython-friendly payload a field hands to the kernel.

    One flat record covers both supported field kinds; the fields irrelevant to a kind carry
    harmless placeholders (a ``(1, 1, 1, 3)`` array for the dipole's ``b_padded``; zeros for the
    grid's dipole scalars), so the kernel sees a single, consistently-typed argument.

    Attributes
    ----------
    kind
        ``0`` = gridded :class:`~qorona.field.sampled.SampledField`; ``1`` = analytic dipole.
    b_padded
        Gridded: the ghost-padded ``(n_r+2G, n_theta+2G, n_phi+2G, 3)`` Cartesian B array the
        tricubic reads. Dipole: an unused placeholder.
    n_r, n_theta, n_phi
        Unpadded grid node counts (gridded only).
    spacing_code
        Radial spacing law: ``0`` logarithmic, ``1`` power-law, ``2`` uniform (gridded only).
    r_inner, r_outer
        Domain bounding radii (the spacing endpoints for the grid; ``R_⊙``/``R_S`` for the dipole),
        used for both the radial index map and the boundary-crossing classification.
    exponent
        Power-law spacing exponent (gridded power-law only; ``1.0`` otherwise).
    moment, background
        Dipole moment ``m`` and background field ``B₀`` (dipole only).
    char_const
        The dipole's constant CFL cell metric ``_CHARACTERISTIC_FRACTION·(R_S - R_⊙)`` (dipole
        only; the grid computes its metric per point).
    """

    kind: int
    b_padded: np.ndarray
    n_r: int
    n_theta: int
    n_phi: int
    spacing_code: int
    r_inner: float
    r_outer: float
    exponent: float
    moment: float
    background: float
    char_const: float


class JitGrid(NamedTuple):
    """The raw spherical-grid geometry the paint kernel forward-bins swept-path points into.

    Just the index map a :class:`~qorona.resample.grid.SphericalGrid` realizes (node counts, the
    radial spacing law and its endpoints, the power-law exponent), handed to the kernel since the
    grid/spacing dataclasses do not survive nopython. The volume grid the painter fills is distinct
    from (and usually finer than) the field grid the lines are traced on, so it is passed alongside
    the field's :class:`JitField` rather than read from it.

    Attributes
    ----------
    n_r, n_theta, n_phi
        Unpadded grid node counts.
    spacing_code
        Radial spacing law: ``0`` logarithmic, ``1`` power-law, ``2`` uniform.
    r_inner, r_outer
        Radial spacing endpoints (the inner/outer node radii), for the radial index map.
    exponent
        Power-law spacing exponent (``1.0`` for the other laws).
    """

    n_r: int
    n_theta: int
    n_phi: int
    spacing_code: int
    r_inner: float
    r_outer: float
    exponent: float


__all__ = ["HAVE_NUMBA", "JitField", "JitGrid", "apply_workers"]
