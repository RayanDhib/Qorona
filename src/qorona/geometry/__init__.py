"""Geometry: coordinate transforms and the rendering cameras.

Cartesian ↔ spherical transforms for points and vector components (the field and tracer
operate in Cartesian, smooth through the poles; the spherical form is used for internal
grid indexing and I/O), plus the rendering cameras and their ray bundle. The cameras are the
orthographic plane-of-sky camera and the perspective camera from a finite observer.
"""

from __future__ import annotations

from qorona.geometry.camera import Camera, OrthographicCamera, PerspectiveCamera, Rays
from qorona.geometry.coordinates import (
    cartesian_to_spherical,
    cartesian_to_spherical_vectors,
    spherical_coordinate_jacobian,
    spherical_to_cartesian,
    spherical_to_cartesian_vectors,
)

__all__ = [
    "Camera",
    "OrthographicCamera",
    "PerspectiveCamera",
    "Rays",
    "cartesian_to_spherical",
    "cartesian_to_spherical_vectors",
    "spherical_coordinate_jacobian",
    "spherical_to_cartesian",
    "spherical_to_cartesian_vectors",
]
