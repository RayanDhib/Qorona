"""Field-line tracing: adaptive integration of the unit field to the boundary spheres.

The public surface is :func:`trace_field_lines`, which traces seeds both ways to the inner and
outer spheres and returns a :class:`FieldLines` struct-of-arrays; :class:`Endpoint` names how
each end terminated. The DOPRI5 stepper underneath is generic over the state shape, so the
squashing-factor stage reuses it by co-integrating deviation vectors.
"""

from __future__ import annotations

from qorona.trace.fieldline import DEFAULT_TURN_GUARD, Endpoint, FieldLines, TurnGuard
from qorona.trace.integrator import trace_field_lines

__all__ = ["DEFAULT_TURN_GUARD", "Endpoint", "FieldLines", "TurnGuard", "trace_field_lines"]
