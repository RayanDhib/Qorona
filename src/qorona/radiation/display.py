"""Display treatments for a brightness frame: the radial / fine-structure detrends for a frame.

A finished brightness image (pB or total) spans many decades radially, so it is displayed through a
detrend that reveals the faint structure at all heights. Three treatments, all 2-D post-processes on
the integrated frame (no field or line-of-sight access):

- **Power-law radial filter** (:func:`radial_filter`): multiply by the impact parameter ``rho``
  raised to a power, a generic graded radial detrend that lifts the outer corona while staying
  bright near the limb; the lightest, model-free option.
- **Radial vignetting** (:func:`newkirk_vignette`): divide out a radial detrend following the
  Newkirk coronal-density model ``Nₑ ∝ 10^(4.32 R☉/r)``, lifting the faint outer corona; self-
  contained, no extra dependency.
- **Fine-structure enhancement** (:func:`mgn_enhance`): Multi-scale Gaussian Normalization, the
  multi-scale generalisation of a single-scale log unsharp mask, via ``sunkit_image.enhance.mgn``.
  ``sunkit-image`` is not installed by default; without it only this treatment is unavailable and it
  fails with a friendly note, while the other treatments and the raw frame still render. MGN is
  calibrated for the steep-gradient pB frame; on total it still runs but is less meaningful.

All leave the integrated frame untouched and return a new display frame; :func:`save_pb_png` writes
any of them to an 8-bit grayscale PNG with a percentile stretch.

Newkirk radial model: Newkirk (1961), ApJ 133, 983. Multi-scale Gaussian Normalization:
Morgan & Druckmüller (2014), Solar Physics 289, 2945.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from qorona.render.los import _scale_intensity, _write_png

__all__ = [
    "MGN_MISSING_HINT",
    "NEWKIRK_REFERENCE_RADIUS",
    "NEWKIRK_SCALE",
    "RADIAL_FILTER_POWER",
    "mgn_enhance",
    "newkirk_vignette",
    "radial_filter",
    "save_pb_png",
]

#: Default exponent of the power-law radial filter; ``BrightnessConfig.radial_power`` mirrors this.
#: Chosen just below the typical brightness radial-falloff power (≈3.8) so the filter lifts the
#: outer corona while keeping the image bright near the limb, with no gradient inversion.
RADIAL_FILTER_POWER = 3.0

#: Newkirk coronal-density exponent: ``Nₑ ∝ 10^(scale · R☉/r)``. A tunable parameter; the
#: default is Newkirk's value.
NEWKIRK_SCALE = 4.32

#: Impact parameter (R☉) at which the radial vignette has unit transmission; anchored
#: here and lifts radii beyond it. A tunable parameter; the default is the limb.
NEWKIRK_REFERENCE_RADIUS = 1.0

#: Friendly guidance shown when the MGN treatment is requested without ``sunkit-image`` installed.
MGN_MISSING_HINT = (
    "the MGN pB fine-structure enhancement needs sunkit-image; install it with "
    "`pip install sunkit-image`, or use the Newkirk vignetting / raw pB treatments instead"
)


def radial_filter(
    frame: np.ndarray, impact: np.ndarray, *, power: float = RADIAL_FILTER_POWER
) -> np.ndarray:
    """Return ``frame`` through the power-law radial filter ``frame · rho**power``.

    A generic power-law detrend, not a named coronal-density model: scaling by the impact parameter
    ``rho`` raised to ``power`` flattens the steep radial brightness falloff enough to reveal the
    outer streamers, while a ``power`` below the falloff exponent keeps the corona brightest near
    the limb (no gradient inversion). Sign-preserving and parameter-light, the lightest alternative
    to the fixed :func:`newkirk_vignette` model. The occulted disk (``rho = 0``) maps to ``0``.

    Parameters
    ----------
    frame
        ``(H, W)`` brightness frame (pB or total).
    impact
        ``(H, W)`` per-pixel impact parameter rho (R☉).
    power
        Radial exponent (default :data:`RADIAL_FILTER_POWER`).

    Returns
    -------
    numpy.ndarray
        ``(H, W)`` radially filtered frame.
    """
    return frame * impact**power


def newkirk_vignette(
    polarized: np.ndarray,
    impact: np.ndarray,
    *,
    reference_radius: float = NEWKIRK_REFERENCE_RADIUS,
    scale: float = NEWKIRK_SCALE,
) -> np.ndarray:
    """Return the pB frame detrended by the Newkirk radial vignette (lifts the faint outer corona).

    Multiplies pB by the inverse Newkirk coronal-density profile
    ``T(rho) = 10^(scale·(1/reference_radius - 1/rho))`` = ``1`` at ``reference_radius``, rising
    outward, so the steep radial falloff of pB is flattened and structure stands out at all heights
    (a dependency-free radial divide). The occulted disk (``rho → 0``) maps to ``0``.

    Parameters
    ----------
    polarized
        ``(H, W)`` polarized-brightness frame.
    impact
        ``(H, W)`` per-pixel impact parameter rho (R☉).
    reference_radius
        Radius (R☉) of unit transmission (default :data:`NEWKIRK_REFERENCE_RADIUS`).
    scale
        Newkirk density exponent (default :data:`NEWKIRK_SCALE`).

    Returns
    -------
    numpy.ndarray
        ``(H, W)`` detrended pB frame.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        transmission = 10.0 ** (scale * (1.0 / reference_radius - 1.0 / impact))
    return polarized * np.nan_to_num(transmission, nan=0.0, posinf=0.0)


def mgn_enhance(
    polarized: np.ndarray,
    *,
    sigma: list[float] | None = None,
    k: float = 0.7,
    gamma: float = 3.2,
    h: float = 0.7,
    weights: list[float] | None = None,
) -> np.ndarray:
    """Return the pB frame enhanced by Multi-scale Gaussian Normalization (fine-structure detail).

    A thin wrapper over ``sunkit_image.enhance.mgn`` (array-in / array-out, so no sunpy ``Map`` is
    needed); the defaults are MGN's own. Non-finite samples are zeroed first (MGN does not accept
    ``NaN``). Raises :class:`ImportError` with :data:`MGN_MISSING_HINT` if ``sunkit-image`` is not
    installed; the only treatment that needs it.

    Parameters
    ----------
    polarized
        ``(H, W)`` polarized-brightness frame.
    sigma
        Gaussian widths (px) of the normalization scales (``None`` ⇒ MGN's default scale set).
    k, gamma, h, weights
        MGN's std-deviation weight, gamma, global/local mix, and per-scale weights (defaults are
        MGN's own).

    Returns
    -------
    numpy.ndarray
        ``(H, W)`` MGN-enhanced frame.
    """
    try:
        from sunkit_image.enhance import mgn  # type: ignore
    except ImportError as error:  # pragma: no cover - exercised only without sunkit-image installed
        raise ImportError(MGN_MISSING_HINT) from error
    data = np.nan_to_num(np.asarray(polarized, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return mgn(data, sigma=sigma, k=k, gamma=gamma, h=h, weights=weights)


def save_pb_png(
    frame: np.ndarray,
    path: str | Path,
    *,
    scaling: Literal["linear", "log"] = "log",
    percentiles: tuple[float, float] = (1.0, 99.5),
) -> None:
    """Write a 2-D pB display frame to ``path`` as an 8-bit grayscale PNG with a percentile stretch.

    The shared writer for the raw, Newkirk, and MGN frames: a per-image percentile
    stretch (logarithmic by default, matching how pB is conventionally shown; ``"linear"`` suits the
    already-normalised MGN output) to ``[0, 1]``, then grayscale 8-bit.

    The stretch percentiles are anchored on the pixels carrying positive brightness: the occulted
    disk and the off-shell background are zero and are not corona measurements, so they are excluded
    from the percentile (a log stretch would otherwise read those zeros as ``log10(0)`` and collapse
    the corona's dynamic range to the top of the scale, washing it to white).
    """
    arr = np.asarray(frame, dtype=float)
    flat = arr.reshape(-1)
    stretched = _scale_intensity(flat[:, None], scaling, percentiles, anchor=flat > 0.0)[:, 0]
    gray = (np.clip(np.nan_to_num(stretched), 0.0, 1.0) * 255.0).round().astype(np.uint8)
    image = np.repeat(gray.reshape(*arr.shape, 1), 3, axis=2)
    _write_png(Path(path), np.ascontiguousarray(image))
