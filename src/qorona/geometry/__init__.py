"""Geometry: coordinate transforms and frames.

Cartesian ↔ spherical transforms for points and vector components. The field and tracer
operate in Cartesian (smooth through the poles); the spherical form is used for internal
grid indexing and I/O.
"""

from __future__ import annotations

from qorona.geometry.camera import OrthographicCamera, Rays
from qorona.geometry.coordinates import (
    cartesian_to_spherical,
    cartesian_to_spherical_vectors,
    spherical_coordinate_jacobian,
    spherical_to_cartesian,
    spherical_to_cartesian_vectors,
)

__all__ = [
    "OrthographicCamera",
    "Rays",
    "cartesian_to_spherical",
    "cartesian_to_spherical_vectors",
    "spherical_coordinate_jacobian",
    "spherical_to_cartesian",
    "spherical_to_cartesian_vectors",
]
