"""Validation gate: field-line tracing on the analytic PFSS dipole.

Geometry-only checks; the Q⊥ *value* gate lives in ``test_dipole_squashing.py``:

- **flux conservation**: the poloidal flux function ``Ψ`` is constant along each traced line
  (field lines are ``Ψ``-contours), to integrator tolerance;
- **open/closed**: seeds in the closed band ``(θ_SL, 180° - θ_SL)`` return closed (both feet
  on the inner sphere), polar-cap seeds return open (a foot on the source surface), and the
  open→closed flip matches the analytic separatrix ``θ_SL = 50° / 130°`` at ``R_seed = 1.01``;
- **foot-landing**: every clean endpoint lies on its boundary sphere;
- **stall guard**: the direction-reversal guard is a no-op on the smooth dipole (no false stalls)
  and the engines return identical end codes with it active. (The analytic dipole never thrashes,
  so the guard's *firing* is not exercised here.)

Run at ``rtol = 1e-4`` *and* ``1e-5`` to confirm the geometry is converged, not coincidental.
One tracer test; validation studies stay separate from the unit suite.
"""

from __future__ import annotations

import numpy as np

from qorona.field import PfssDipoleField
from qorona.geometry import spherical_to_cartesian
from qorona.trace import Endpoint, trace_field_lines

#: Seed radius where θ_SL = 50.0°. Seeds sit just above the
#: inner boundary so both the near foot and the far end of each line are exercised.
R_SEED = 1.01


def _seeds(colatitudes_deg: np.ndarray, azimuth: float = 0.7) -> np.ndarray:
    """Return seeds at ``R_seed`` for the given colatitudes (degrees), at a fixed azimuth."""
    colatitude = np.deg2rad(np.asarray(colatitudes_deg, dtype=np.float64))
    spherical = np.stack(
        [np.full_like(colatitude, R_SEED), colatitude, np.full_like(colatitude, azimuth)], axis=-1
    )
    return spherical_to_cartesian(spherical)


def test_dipole_tracer_geometry() -> None:
    field = PfssDipoleField()
    inner, outer = field.domain.inner_radius, field.domain.outer_radius
    theta_sl = np.degrees(field.separatrix_colatitude(R_SEED))

    for rtol in (1e-4, 1e-5):
        closed_lines = trace_field_lines(
            field, _seeds([60, 75, 90, 105, 120]), rtol=rtol, store_path=True, show_progress=False
        )
        open_lines = trace_field_lines(
            field, _seeds([20, 35, 145, 160]), rtol=rtol, show_progress=False
        )

        # Closed band: both feet land on the inner sphere; polar caps: a foot on the outer sphere.
        assert closed_lines.is_closed.all()
        assert open_lines.is_open.all()

        # Flux conservation: Ψ is constant along each traced line (field lines are Ψ-contours).
        for path in closed_lines.paths:
            psi = field.flux_function(path)
            assert (psi.max() - psi.min()) / np.abs(psi).mean() < 1e-6

        # Foot-landing: every clean endpoint lies on the sphere its Endpoint code names.
        for lines in (closed_lines, open_lines):
            feet = lines.feet.reshape(-1, 3)
            ends = lines.ends.reshape(-1)
            clean = (ends == Endpoint.INNER) | (ends == Endpoint.OUTER)
            radius = np.linalg.norm(feet[clean], axis=1)
            target = np.where(ends[clean] == Endpoint.OUTER, outer, inner)
            assert np.max(np.abs(radius - target)) < 1e-9

        # Open→closed flip matches the analytic separatrix to < 0.5° (seeds bracket it with a
        # margin; those on the separatrix would run toward the cusp null and are not classified).
        sweep_deg = np.linspace(theta_sl - 2.0, theta_sl + 2.0, 81)
        sweep = trace_field_lines(field, _seeds(sweep_deg), rtol=rtol, show_progress=False)
        closed = sweep.is_closed
        first_closed = sweep_deg[closed][0]
        opens_below = sweep_deg[~closed][sweep_deg[~closed] < first_closed]
        flip = 0.5 * (first_closed + opens_below[-1])
        assert abs(flip - theta_sl) < 0.5

        # Stall guard: a no-op on the clean dipole (no line thrashes, so none is falsely stalled),
        # and the engines agree with it active. `sweep` ran on the accelerated engine (numba or
        # CUDA, device="auto") with the guard on (default); the same seeds with the guard off must
        # give identical geometry, and the NumPy core (forced by store_path) must return identical
        # end codes.
        assert not sweep.is_stalled.any()
        guard_off = trace_field_lines(
            field, _seeds(sweep_deg), rtol=rtol, max_reversals=0, show_progress=False
        )
        numpy_engine = trace_field_lines(
            field, _seeds(sweep_deg), rtol=rtol, store_path=True, show_progress=False
        )
        assert np.array_equal(sweep.ends, guard_off.ends)
        assert np.array_equal(sweep.ends, numpy_engine.ends)
