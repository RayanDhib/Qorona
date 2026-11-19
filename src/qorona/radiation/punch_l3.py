"""The PUNCH Level 3 background treatment, applied to model frames.

PUNCH Level 3 removes a background from each image. For each pixel and each polarizer channel, the
background is the 2nd percentile of that channel over about 32 days of frames. Level 3 subtracts
it per channel and then converts the three channels to total brightness, pB and pB'. The result
shows "now minus the usual": the changes, not the steady corona. Implemented from the PUNCH
pipeline punchbowl (``level3/f_corona_model.py`` and ``level3/polarization.py``).

A model solution is one snapshot. Its background frames are views of that snapshot from the
observer positions over a window around the snapshot time. The background therefore changes only
with the rotation of the Sun. The functions here are pure NumPy;
:func:`qorona.pipeline.render_punch_l3` renders the frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qorona.radiation.brightness import BrightnessResult

__all__ = [
    "BACKGROUND_PERCENTILE",
    "BACKGROUND_SIZE",
    "POLARIZER_ANGLES",
    "STEP_HOURS",
    "WINDOW_DAYS",
    "PunchL3Result",
    "background",
    "from_channels",
    "radial_angle",
    "to_channels",
    "upsample",
    "window_offsets",
]

#: The angles of the three polarizer channels (M, Z, P), in radians from the image's up axis.
POLARIZER_ANGLES = np.deg2rad(np.array([-60.0, 0.0, 60.0]))
#: The percentile of each channel over the window that makes the background.
BACKGROUND_PERCENTILE = 2.0
#: The half-width of the background window, in days. The full window (32 days) covers more than
#: one solar rotation, so every longitude is seen.
WINDOW_DAYS = 16.0
#: The time between two background frames, in hours.
STEP_HOURS = 12.0
#: The larger side, in pixels, of the background frames. Larger frames are scaled down to it.
BACKGROUND_SIZE = 512


def window_offsets() -> np.ndarray:
    """Return the times of the background frames, in hours from the snapshot time."""
    half = round(WINDOW_DAYS * 24.0 / STEP_HOURS)
    return STEP_HOURS * np.arange(-half, half + 1, dtype=float)


def radial_angle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the ``(H, W)`` angle of the radial direction at each pixel, in radians.

    ``x`` (``(W,)``, to image right) and ``y`` (``(H,)``, to image up) are the pixel offsets from
    Sun centre. The angle is measured from the image's up axis, as the polarizer angles are.
    """
    return np.arctan2(-x[None, :], y[:, None])


def to_channels(total: np.ndarray, polarized: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Return the ``(3, H, W)`` brightness behind the three polarizers.

    Thomson-scattered light is polarized tangentially. A polarizer at angle ``θ`` therefore passes
    ``½ (tB - pB cos 2(θ - alpha))``, with ``alpha`` the radial angle of the pixel.
    """
    phase = 2.0 * (POLARIZER_ANGLES[:, None, None] - alpha[None])
    return 0.5 * (total[None] - polarized[None] * np.cos(phase))


def from_channels(channels: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return total brightness and pB from the ``(3, H, W)`` polarizer channels.

    ``tB = (2/3) Σ Bᵢ`` and ``pB = -(4/3) Σ Bᵢ cos 2(θᵢ - alpha)``. pB is signed (tangential
    minus radial).
    """
    phase = 2.0 * (POLARIZER_ANGLES[:, None, None] - alpha[None])
    total = (2.0 / 3.0) * channels.sum(axis=0)
    polarized = -(4.0 / 3.0) * (channels * np.cos(phase)).sum(axis=0)
    return total, polarized


def background(channel_frames: np.ndarray) -> np.ndarray:
    """Return the ``(3, h, w)`` background of ``(n, 3, h, w)`` channel frames.

    For each channel and each pixel, it is the :data:`BACKGROUND_PERCENTILE` over the frames.
    NaN samples (no corona in that frame) do not count, as in PUNCH. A pixel with no valid frame
    gets NaN.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(channel_frames, BACKGROUND_PERCENTILE, axis=0)


def upsample(
    channels: np.ndarray, valid: np.ndarray, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(3, h, w)`` ``channels`` interpolated (bilinear) to ``shape`` ``(H, W)``.

    The two grids cover the same field of view. Pixel ``i`` of the large grid therefore sits at
    ``(i + ½) h / H - ½`` on the small one. Only the ``valid`` ``(h, w)`` pixels contribute: the
    interpolation weights are renormalized over them. The second return is ``(H, W)``: true where
    all the contributing small pixels and their neighbours are valid. Within two small pixels of an
    occulter edge it is false: the soft edge moves with the observer distance, and the interpolated
    background there is not reliable.
    """
    from scipy.ndimage import binary_erosion, map_coordinates

    height, width = shape
    rows = (np.arange(height) + 0.5) * channels.shape[1] / height - 0.5
    cols = (np.arange(width) + 0.5) * channels.shape[2] / width - 0.5
    grid = np.meshgrid(rows, cols, indexing="ij")
    weight = valid.astype(np.float64)
    channels = np.where(valid[None], channels, 0.0)

    def interpolate(image: np.ndarray) -> np.ndarray:
        return np.asarray(map_coordinates(image, grid, order=1, mode="nearest"))

    coverage = interpolate(weight)
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.stack(
            [
                np.where(coverage > 0.0, interpolate(channel * weight) / coverage, 0.0)
                for channel in channels
            ]
        )
    inner = binary_erosion(valid, border_value=True).astype(np.float64)
    return values, interpolate(inner) > 1.0 - 1e-9


@dataclass(frozen=True)
class PunchL3Result:
    """A white-light frame with the PUNCH Level 3 background treatment.

    Attributes
    ----------
    treated
        The frame after the treatment: its ``total`` and ``polarized`` are the treated tB and pB,
        and the rest is the geometry of the untreated frame.
    full
        The untreated frame, with the full K corona.
    valid
        ``(H, W)`` true where the treated frame holds: the pixels with corona, away from the
        occulter edges. The treated frames are zero elsewhere.
    frames
        The number of background frames.
    background_pixels
        ``(height, width)`` of the background frames.
    """

    treated: BrightnessResult
    full: BrightnessResult
    valid: np.ndarray
    frames: int
    background_pixels: tuple[int, int]

    def to_provenance(self) -> dict[str, object]:
        return {
            "window_days": WINDOW_DAYS,
            "step_hours": STEP_HOURS,
            "percentile": BACKGROUND_PERCENTILE,
            "frames": self.frames,
            "background_pixels": list(self.background_pixels),
        }
