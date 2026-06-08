"""Radiation: Thomson scattering, K-corona, and polarized-brightness (pB) imagery.

The secondary product family: white-light / polarized brightness from the electron density, and the
optional Thomson weighting of the primary Q⊥ render. Two coupled deliverables share one core (the
Minnaert/Billings coefficients and the single-electron intensities in :mod:`.thomson`):

- :class:`ThomsonWeight`: the optional, off-by-default radiometric LOS weight for the Q⊥ render
  (``render(..., thomson=ThomsonWeight(density, mode))``), biasing it toward bright dense plasma.
- :func:`render_brightness`: the standalone pB / total-brightness product.
- :func:`newkirk_vignette` and :func:`mgn_enhance`: the two display treatments that finish the pB
  image (radial vignetting and multi-scale fine-structure enhancement).
"""

from __future__ import annotations

from qorona.radiation.brightness import BrightnessResult, render_brightness
from qorona.radiation.display import mgn_enhance, newkirk_vignette, save_pb_png
from qorona.radiation.thomson import (
    RadialCoefficients,
    ThomsonWeight,
    build_coefficient_table,
    intensity_coefficients,
    minnaert_coefficients,
)

__all__ = [
    "BrightnessResult",
    "RadialCoefficients",
    "ThomsonWeight",
    "build_coefficient_table",
    "intensity_coefficients",
    "mgn_enhance",
    "minnaert_coefficients",
    "newkirk_vignette",
    "render_brightness",
    "save_pb_png",
]
