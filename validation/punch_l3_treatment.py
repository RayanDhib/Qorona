"""Validation study: the PUNCH Level 3 treatment of a spherically symmetric corona.

The treatment subtracts, for each pixel and polarizer channel, the 2nd percentile of the frames
over ±16 days. A spherically symmetric corona looks the same from every longitude, so the only
change over the window is the Earth-Sun distance. The treated frame must therefore stay a small
fraction of the untreated one. The corona is a standard electron-density profile, rendered in the
PUNCH mosaic frame (reduced to 256 pixels) from Earth. The profile is implemented from Leblanc et
al. (1998).

Report-only: prints the ratio of the treated to the untreated ring medians and asserts nothing.
Run on demand:

    python validation/punch_l3_treatment.py
"""

from __future__ import annotations

import numpy as np

from qorona.config import CAMERA_INSTRUMENTS, BrightnessConfig, CameraConfig
from qorona.ephemeris import sub_observer_point
from qorona.field.density import DensityVolume
from qorona.pipeline import render_punch_l3
from qorona.resample.grid import LogarithmicSpacing, SphericalGrid

#: The solution time of the synthetic corona.
_TIMESTAMP = "2025-09-23T03:04:00"
#: The image size, in pixels (the mosaic frame scaled down).
_PIXELS = 256


def leblanc(radius: np.ndarray) -> np.ndarray:
    """The electron density, in cm⁻³, at ``radius`` (R☉)."""
    return 3.3e5 * radius**-2 + 4.1e6 * radius**-4 + 8.0e7 * radius**-6


def main() -> None:
    grid = SphericalGrid(LogarithmicSpacing(2.0, 250.0), n_r=400, n_theta=6, n_phi=12)
    nodes = grid.node_points().reshape(-1, 3)
    radius = np.linalg.norm(nodes, axis=1).reshape(grid.n_r, grid.n_theta, grid.n_phi)
    density = DensityVolume.from_grid_values(grid, leblanc(radius))

    mosaic = CAMERA_INSTRUMENTS["punch-mosaic"]
    scale = _PIXELS / mosaic.pixels[1]
    longitude, latitude, distance = sub_observer_point("earth", _TIMESTAMP, "carrington")
    camera = CameraConfig(
        projection="perspective",
        longitude=longitude,
        latitude=latitude,
        observer_distance=distance,
        fov_deg=mosaic.fov_deg,
        pixels=(_PIXELS, _PIXELS),
        projection_code=mosaic.projection_code,
        reference_pixel=(
            (mosaic.reference_pixel[0] + 0.5) * scale - 0.5,
            (mosaic.reference_pixel[1] + 0.5) * scale - 0.5,
        ),
    )
    result = render_punch_l3(
        density,
        BrightnessConfig(vignette="none", punch_l3=True),
        camera,
        timestamp=_TIMESTAMP,
        frame="carrington",
        category="heliosphere",
        show_progress=True,
    )
    full, treated = result.full, result.treated
    elongation = np.hypot(full.x[None, :], full.y[:, None])
    print("spherically symmetric corona: treated / untreated ring medians (pB and total)")
    for low in range(6, 42, 4):
        ring = (elongation >= low) & (elongation < low + 4) & (full.total > 0.0)
        ratios = [
            float(np.median(treated_image[ring]) / np.median(full_image[ring]))
            for treated_image, full_image in (
                (treated.polarized, full.polarized),
                (treated.total, full.total),
            )
        ]
        print(f"    {low:2d}-{low + 4:2d} deg: pB {ratios[0]:+.4f}   total {ratios[1]:+.4f}")


if __name__ == "__main__":
    main()
