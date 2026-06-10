"""Geometry: coordinate transforms and the rendering camera.

Cartesian ↔ spherical transforms for points and vector components (the field and tracer
operate in Cartesian, smooth through the poles; the spherical form is used for internal
grid indexing and I/O), plus the orthographic plane-of-sky camera and its ray bundle.
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
