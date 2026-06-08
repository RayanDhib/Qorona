"""Render: turn the corona into eclipse-like imagery on an orthographic plane-of-sky camera.

The primary product (``los.py``) is the weight-normalised line-of-sight integral of log₁₀ Q⊥ in
three depth-faking colour channels. The secondary product (``fieldlines.py``) is a plane-of-sky
magnetic field-line render: traced lines drawn in projection, coloured by polarity / open/closed.
Both share the camera in :mod:`qorona.geometry.camera`.
"""

from __future__ import annotations

from qorona.render.fieldlines import FieldLineImage, render_field_lines
from qorona.render.los import (
    LARGE_FOV,
    SMALL_FOV,
    RenderResult,
    WeightingPreset,
    render,
)

__all__ = [
    "LARGE_FOV",
    "SMALL_FOV",
    "FieldLineImage",
    "RenderResult",
    "WeightingPreset",
    "render",
    "render_field_lines",
]
