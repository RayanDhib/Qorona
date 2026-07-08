"""Display treatments for a brightness frame: radial detrends, whitening, and enhancement.

A finished brightness image (pB or total) spans many decades radially, so it is displayed through
a detrend that reveals the faint structure at all heights. Three kinds of stage, all 2-D
post-processes on the integrated frame (no MHD data access):

- **Radial detrend** (the vignette): divide the frame by a reference radial brightness curve.
  The ``newkirk`` channel (:func:`newkirk_vignette`) uses the brightness the smooth Newkirk
  background corona would produce along the same lines of sight (:func:`newkirk_profile`). The
  ``adaptive`` channel (:func:`adaptive_vignette`) self-calibrates the same hydrostatic curve
  family to the image's own falloff and amplifies the remaining azimuthal structure, for coronae
  whose stratification departs from the Newkirk profile.
- **Wavelet whitening** (:func:`wow_enhance`): Wavelets Optimized Whitening, which equalizes the
  power of the frame's wavelet spectrum across locations and scales, via
  ``sunkit_image.enhance.wow``. Whitening flattens the radial falloff along the way, so this is
  the ``wow`` vignette channel: an alternative treatment applied to the raw frame, not a stage
  on top of a vignette. Its output is signed (whitened structure is zero-centred), unlike the
  always-positive vignette outputs.
- **Fine-structure enhancement** (:func:`mgn_enhance`): Multi-scale Gaussian Normalization, a
  local (neighbourhood-based) contrast equalization, the multi-scale generalisation of an
  unsharp mask, via ``sunkit_image.enhance.mgn``. MGN is calibrated for the steep-gradient pB
  frame; on total it still runs but is less meaningful.

``sunkit-image`` (plus ``watroo`` for WOW) is not installed by default; without it only the two
enhancement stages are unavailable and they fail with a friendly note, while the detrends and
the raw frame still render.

All leave the integrated frame untouched and return a new display frame; :func:`save_pb_png`
writes any of them to an 8-bit grayscale PNG with a percentile stretch.

Newkirk background corona: Newkirk (1961), ApJ 133, 983. Multi-scale Gaussian Normalization:
Morgan & Druckmüller (2014), Solar Physics 289, 2945. Wavelets Optimized Whitening: Auchère,
Soubrié, Pelouze & Buchlin (2023), A&A 670, A66.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from qorona.radiation.thomson import (
    ASYMPTOTIC_CROSSOVER,
    LIMB_DARKENING,
    build_coefficient_table,
)
from qorona.render.image import scale_intensity, write_png

__all__ = [
    "ADAPTIVE_GAIN",
    "ADAPTIVE_MARGIN",
    "MGN_MISSING_HINT",
    "NEWKIRK_EXPONENT",
    "WOW_MISSING_HINT",
    "adaptive_vignette",
    "mgn_enhance",
    "newkirk_profile",
    "newkirk_vignette",
    "save_pb_png",
    "wow_enhance",
]

#: Newkirk (1961) coronal-density exponent: the background model is ``Nₑ ∝ 10^(4.32 R☉/r)``.
NEWKIRK_EXPONENT = 4.32

#: Adaptive-channel envelope under-compensation, subtracted from the fitted hydrostatic exponent:
#: the envelope is left slightly shallower than the image's own falloff, so the displayed corona
#: keeps the gentle outward fade of a radially graded filter instead of coming out flat.
ADAPTIVE_MARGIN = 0.9

#: Adaptive-channel contrast gain: the exponent applied to the normalized azimuthal structure,
#: lifting a weak-contrast corona to the structure contrast the ``newkirk`` channel yields on a
#: strongly stratified one.
ADAPTIVE_GAIN = 3.0

#: Radial nodes of a precomputed vignette profile, interpolated per pixel; dense enough that the
#: interpolation error is invisible next to the display stretch.
_PROFILE_NODES = 1024

#: Line-of-sight quadrature samples per profile node (half line; the integrand is even in ``s``).
_PROFILE_SAMPLES = 2048

#: Ring bins of the adaptive channel's radial statistics, the minimum positive pixels a ring
#: needs before its median is trusted, and the moving-average window (bins, odd) that smooths the
#: measured profile so its bin-to-bin noise does not print as concentric ripples.
_RING_BINS = 256
_RING_MIN_PIXELS = 8
_RING_SMOOTHING = 5

#: Hydrostatic exponents scanned when fitting the adaptive envelope to the image's own profile.
_FIT_EXPONENT_RANGE = (1.5, 6.0)
_FIT_EXPONENT_STEP = 0.05

#: Friendly guidance shown when the MGN stage is requested without ``sunkit-image`` installed.
MGN_MISSING_HINT = (
    "the MGN pB fine-structure enhancement needs sunkit-image; install it with "
    "`pip install sunkit-image`, or use the newkirk / adaptive vignettes or the raw frame instead"
)

#: Friendly guidance shown when the ``wow`` channel is requested without its packages installed.
WOW_MISSING_HINT = (
    "the wow display channel needs sunkit-image and watroo; install them with "
    "`pip install sunkit-image watroo`, or use the newkirk / adaptive vignettes or the raw "
    "frame instead"
)


def _hydrostatic_profile(
    rho: np.ndarray,
    exponent: float,
    *,
    frame: Literal["polarized", "total"],
    r_inner: float,
    r_outer: float,
    u: float = LIMB_DARKENING,
    crossover: float = ASYMPTOTIC_CROSSOVER,
    samples: int = _PROFILE_SAMPLES,
) -> np.ndarray:
    """Line-of-sight brightness of a hydrostatic ``Nₑ ∝ 10^(exponent/r)`` corona at ``rho``.

    The reference-curve engine shared by both vignette channels: the polarized (or total)
    brightness a smooth, spherically symmetric hydrostatic corona produces along a ray of impact
    parameter ``rho``, integrated with the same single-electron coefficients as the data frame and
    over the same radial shell ``[r_inner, r_outer]`` the data occupied. The shared-shell
    truncation mirrors the data frame's finite integration domain, so the ratio carries no
    spurious radial trend from mismatched limits (the untruncated integral would converge
    regardless: the single-electron coefficients fall off with the Sun's solid angle). Relative
    units, like the frames themselves.
    """
    rho = np.atleast_1d(np.asarray(rho, dtype=np.float64))
    table = build_coefficient_table(r_inner, r_outer, u=u, crossover=crossover)
    s_max = np.sqrt(np.clip(r_outer**2 - rho**2, 0.0, None))
    s = s_max[:, None] * np.linspace(0.0, 1.0, samples)[None, :]
    radius = np.sqrt(rho[:, None] ** 2 + s**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        c_tan, c_pol = table.evaluate(radius)
    sin_sq_chi = np.divide(
        rho[:, None] ** 2, radius**2, out=np.zeros_like(radius), where=radius > 0.0
    )
    # The r_inner floor only guards the 10**x overflow at tiny radii; those samples sit inside the
    # shell's inner boundary and are zeroed by the mask below.
    density = 10.0 ** (exponent / np.maximum(radius, r_inner))
    if frame == "polarized":
        integrand = density * c_pol * sin_sq_chi
    else:
        integrand = density * (2.0 * c_tan - c_pol * sin_sq_chi)
    integrand = np.where(radius >= r_inner, integrand, 0.0)
    # The integrand is even in s, so the half line doubled is the full line-of-sight integral.
    return 2.0 * np.trapezoid(integrand, s, axis=1)


def newkirk_profile(
    rho: np.ndarray,
    *,
    frame: Literal["polarized", "total"],
    r_inner: float,
    r_outer: float,
    u: float = LIMB_DARKENING,
    crossover: float = ASYMPTOTIC_CROSSOVER,
    samples: int = _PROFILE_SAMPLES,
) -> np.ndarray:
    """Line-of-sight brightness of the Newkirk background corona at impact parameters ``rho``.

    The reference curve of the ``newkirk`` vignette: :func:`_hydrostatic_profile` at the Newkirk
    (1961) exponent.

    Parameters
    ----------
    rho
        ``(n,)`` impact parameters (R☉).
    frame
        Which brightness the curve carries: ``"polarized"`` (pB) or ``"total"``.
    r_inner, r_outer
        The radial shell (R☉) the data frame integrated over (``BrightnessResult.r_inner`` /
        ``r_outer``).
    u, crossover
        The limb darkening and coefficient crossover the data frame was built with.
    samples
        Quadrature samples along the half line of sight.

    Returns
    -------
    numpy.ndarray
        ``(n,)`` model brightness; ``0`` where the ray misses the shell (``rho >= r_outer``).
    """
    return _hydrostatic_profile(
        rho,
        NEWKIRK_EXPONENT,
        frame=frame,
        r_inner=r_inner,
        r_outer=r_outer,
        u=u,
        crossover=crossover,
        samples=samples,
    )


def newkirk_vignette(
    image: np.ndarray,
    impact: np.ndarray,
    *,
    frame: Literal["polarized", "total"],
    r_inner: float,
    r_outer: float,
    reference_radius: float,
    u: float = LIMB_DARKENING,
    crossover: float = ASYMPTOTIC_CROSSOVER,
) -> np.ndarray:
    """Divide ``image`` by the Newkirk background corona's brightness at each pixel's radius.

    The classic radial detrend of the white-light products: brightness over model brightness, so
    what remains is the deviation from a smooth featureless corona and structure stands out at
    every height. After the division the radial intensity scale is display-arbitrary; only the
    latitudinal structure carries meaning. Normalized to unit transmission at
    ``reference_radius`` (the occulter edge), a pure display scale. Pixels below
    ``reference_radius`` map to ``0`` (the occulter edge and its feather carry darkened data,
    not corona measurements), as do pixels where the model brightness is zero (beyond the
    shell).

    Parameters
    ----------
    image
        ``(H, W)`` brightness frame (pB or total).
    impact
        ``(H, W)`` per-pixel impact parameter rho (R☉).
    frame
        Which brightness ``image`` carries; the model curve matches it.
    r_inner, r_outer
        The radial shell (R☉) the frame integrated over.
    reference_radius
        Radius (R☉) of unit transmission.
    u, crossover
        The limb darkening and coefficient crossover the frame was built with.

    Returns
    -------
    numpy.ndarray
        ``(H, W)`` detrended frame.
    """
    nodes = np.linspace(0.0, float(np.max(impact)), _PROFILE_NODES)
    profile = newkirk_profile(
        nodes, frame=frame, r_inner=r_inner, r_outer=r_outer, u=u, crossover=crossover
    )
    reference = float(np.interp(min(reference_radius, nodes[-1]), nodes, profile))
    if reference <= 0.0:
        reference = float(profile.max()) if float(profile.max()) > 0.0 else 1.0
    model = np.interp(impact, nodes, profile)
    with np.errstate(divide="ignore", invalid="ignore"):
        detrended = np.where(
            (model > 0.0) & (impact >= reference_radius), image * (reference / model), 0.0
        )
    return np.nan_to_num(detrended, nan=0.0, posinf=0.0, neginf=0.0)


def _ring_medians(
    image: np.ndarray, impact: np.ndarray, floor: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(centers, medians)`` of the positive pixels binned by impact parameter.

    The image's own radial brightness profile: ``_RING_BINS`` equal-width rings from ``floor``
    (pixels below it, the occulter edge and its feather, carry darkened data and are excluded) to
    the radial span of the positive pixels. A ring's median is ``NaN`` when it holds fewer than
    ``_RING_MIN_PIXELS`` positive pixels; the measured profile is smoothed in log space over
    ``_RING_SMOOTHING`` bins.
    """
    positive = (image > 0.0) & (impact >= floor)
    rho = impact[positive]
    values = image[positive]
    order = np.argsort(rho)
    rho = rho[order]
    values = values[order]
    edges = np.linspace(rho[0], rho[-1], _RING_BINS + 1)
    splits = np.searchsorted(rho, edges)
    medians = np.full(_RING_BINS, np.nan)
    for i in range(_RING_BINS):
        ring = values[splits[i] : splits[i + 1]]
        if ring.size >= _RING_MIN_PIXELS:
            medians[i] = np.median(ring)
    centers = 0.5 * (edges[:-1] + edges[1:])

    good = np.isfinite(medians) & (medians > 0.0)
    if int(good.sum()) > _RING_SMOOTHING:
        kernel = np.ones(_RING_SMOOTHING) / _RING_SMOOTHING
        log_med = np.log10(medians[good])
        padded = np.pad(log_med, _RING_SMOOTHING // 2, mode="edge")
        medians[good] = 10.0 ** np.convolve(padded, kernel, mode="valid")
    return centers, medians


def adaptive_vignette(
    image: np.ndarray,
    impact: np.ndarray,
    *,
    frame: Literal["polarized", "total"],
    r_inner: float,
    r_outer: float,
    reference_radius: float,
    u: float = LIMB_DARKENING,
    crossover: float = ASYMPTOTIC_CROSSOVER,
    margin: float = ADAPTIVE_MARGIN,
    gain: float = ADAPTIVE_GAIN,
) -> np.ndarray:
    """Detrend by the image's own radial envelope and amplify the normalized structure.

    The self-calibrating vignette, for coronae whose stratification departs from the Newkirk
    profile (where the fixed ``newkirk`` curve leaves a residual radial trend). Three steps, all
    derived from the frame itself:

    1. Measure the image's radial profile (per-ring medians of the positive pixels) and fit it,
       in brightness space through the same line-of-sight integral, with the hydrostatic curve
       family ``Nₑ ∝ 10^(a/r)``. The ``newkirk`` channel is the fixed member ``a = 4.32`` of the
       same family, so on a Newkirk-like corona this channel converges to it.
    2. Divide by the fitted curve under-compensated by ``margin`` (the envelope), keeping the
       gentle outward fade of a radially graded filter.
    3. Raise the remaining normalized structure ``image / median(rho)`` to ``gain``, lifting weak
       azimuthal contrast to the level the ``newkirk`` channel yields on a strongly stratified
       corona.

    As for every vignette, the resulting radial intensity scale is display-arbitrary; only the
    latitudinal structure carries meaning. Pixels without a measurable ring median map to ``0``.

    Parameters
    ----------
    image
        ``(H, W)`` brightness frame (pB or total).
    impact
        ``(H, W)`` per-pixel impact parameter rho (R☉).
    frame
        Which brightness ``image`` carries; the fitted curve matches it.
    r_inner, r_outer
        The radial shell (R☉) the frame integrated over.
    reference_radius
        The occulter radius (R☉): the radial statistics start there, and pixels below it map to
        ``0`` (the occulter edge and its feather carry darkened data, not corona measurements).
    u, crossover
        The limb darkening and coefficient crossover the frame was built with.
    margin
        Envelope under-compensation (default :data:`ADAPTIVE_MARGIN`).
    gain
        Structure contrast gain (default :data:`ADAPTIVE_GAIN`).

    Returns
    -------
    numpy.ndarray
        ``(H, W)`` detrended, contrast-amplified frame.
    """
    if not bool(np.any((image > 0.0) & (impact >= reference_radius))):
        return np.zeros_like(image)
    centers, medians = _ring_medians(image, impact, reference_radius)
    good = np.isfinite(medians) & (medians > 0.0)
    if int(good.sum()) < _RING_MIN_PIXELS:
        # Too little corona to fit; fall back to the fixed Newkirk member of the curve family.
        exponent = NEWKIRK_EXPONENT
    else:
        lo, hi = _FIT_EXPONENT_RANGE
        candidates = np.arange(lo, hi + _FIT_EXPONENT_STEP / 2, _FIT_EXPONENT_STEP)
        log_mu = np.log10(medians[good])
        scores = np.empty(candidates.size)
        for k, a in enumerate(candidates):
            curve = _hydrostatic_profile(
                centers[good],
                float(a),
                frame=frame,
                r_inner=r_inner,
                r_outer=r_outer,
                u=u,
                crossover=crossover,
                samples=512,
            )
            with np.errstate(divide="ignore"):
                residual = log_mu - np.log10(curve)
            residual = residual[np.isfinite(residual)]
            scores[k] = np.var(residual) if residual.size else np.inf
        exponent = float(candidates[int(np.argmin(scores))])

    envelope_curve = _hydrostatic_profile(
        centers,
        exponent - margin,
        frame=frame,
        r_inner=r_inner,
        r_outer=r_outer,
        u=u,
        crossover=crossover,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        envelope_nodes = np.where(envelope_curve > 0.0, medians / envelope_curve, np.nan)
    usable = good & np.isfinite(envelope_nodes)
    if not bool(usable.any()):
        return np.zeros_like(image)
    envelope = np.interp(impact, centers[usable], envelope_nodes[usable])
    mu = np.interp(impact, centers[good], medians[good])
    with np.errstate(divide="ignore", invalid="ignore"):
        structure = np.where(mu > 0.0, image / mu, 0.0)
    detrended = envelope * np.power(np.clip(structure, 0.0, None), gain)
    detrended = np.where((image > 0.0) & (impact >= reference_radius), detrended, 0.0)
    return np.nan_to_num(detrended, nan=0.0, posinf=0.0, neginf=0.0)


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
    installed; the only stage that needs it.

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
        from sunkit_image.enhance import mgn
    except ImportError as error:  # pragma: no cover - exercised only without sunkit-image installed
        raise ImportError(MGN_MISSING_HINT) from error
    data = np.nan_to_num(np.asarray(polarized, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return mgn(data, sigma=sigma, k=k, gamma=gamma, h=h, weights=weights)


def wow_enhance(
    image: np.ndarray,
    *,
    h: float = 0.9,
    gamma: float = 3.2,
    n_scales: int | None = None,
    denoise_coefficients: list[float] | None = None,
) -> np.ndarray:
    """Return the frame treated by Wavelets Optimized Whitening (the ``wow`` display channel).

    A thin wrapper over ``sunkit_image.enhance.wow`` (array-in / array-out, so no sunpy ``Map``
    is needed). The channel default ``h = 0.9`` merges the gamma-stretched original at weight ``h``
    with the whitened detail at weight ``1 - h``; WOW's own default is the pure whitening
    (``h = 0``), a much stronger look. The output is signed: whitened structure is zero-centred,
    so the display stretch must span the full range (:func:`save_pb_png` with ``valid``).
    Non-finite samples are zeroed first (WOW does not accept ``NaN``). Raises
    :class:`ImportError` with :data:`WOW_MISSING_HINT` if ``sunkit-image`` or ``watroo`` is not
    installed (the latter is imported lazily inside ``sunkit_image.enhance.wow``).

    Parameters
    ----------
    image
        ``(H, W)`` brightness frame (pB or total).
    h
        Merge weight of the gamma-stretched original (channel default ``0.9``).
    gamma
        Stretch exponent of the merged original (WOW's own default).
    n_scales
        Wavelet scales (``None`` ⇒ the maximum the frame size allows).
    denoise_coefficients
        Per-scale noise thresholds, in noise standard deviations (``None`` ⇒ no denoising).

    Returns
    -------
    numpy.ndarray
        ``(H, W)`` whitened frame, signed.
    """
    try:
        from sunkit_image.enhance import wow
    except ImportError as error:  # pragma: no cover - exercised only without sunkit-image installed
        raise ImportError(WOW_MISSING_HINT) from error
    data = np.nan_to_num(np.asarray(image, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        return wow(
            data, h=h, gamma=gamma, n_scales=n_scales, denoise_coefficients=denoise_coefficients
        )
    except ImportError as error:  # pragma: no cover - exercised only without watroo installed
        raise ImportError(WOW_MISSING_HINT) from error


def save_pb_png(
    frame: np.ndarray,
    path: str | Path,
    *,
    scaling: Literal["linear", "log"] = "log",
    percentiles: tuple[float, float] = (1.0, 99.5),
    valid: np.ndarray | None = None,
) -> None:
    """Write a 2-D brightness display frame to ``path`` as an 8-bit grayscale PNG.

    The shared writer for every treatment stack: a per-image percentile stretch (linear for the
    already-flattened vignetted or MGN frames, logarithmic for the raw falloff) to ``[0, 1]``,
    then grayscale 8-bit.

    By default the stretch percentiles are anchored on the pixels carrying positive brightness:
    the occulted disk and the off-shell background are zero and are not corona measurements, so
    they are excluded from the percentile (a log stretch would otherwise read those zeros as
    ``log10(0)`` and collapse the corona's dynamic range to the top of the scale, washing it to
    white). A signed frame (the ``wow`` channel) carries corona structure on both sides of zero,
    so the positive-pixel anchor does not apply; the caller passes ``valid``, an ``(H, W)``
    boolean mask of the corona pixels, which anchors the stretch instead and blanks the
    non-valid pixels to black after it.
    """
    arr = np.asarray(frame, dtype=float)
    flat = arr.reshape(-1)
    anchor = (flat > 0.0) if valid is None else np.asarray(valid, dtype=bool).reshape(-1)
    stretched = scale_intensity(flat[:, None], scaling, percentiles, anchor=anchor)[:, 0]
    if valid is not None:
        stretched = np.where(anchor, stretched, 0.0)
    gray = (np.clip(np.nan_to_num(stretched), 0.0, 1.0) * 255.0).round().astype(np.uint8)
    image = np.repeat(gray.reshape(*arr.shape, 1), 3, axis=2)
    write_png(Path(path), np.ascontiguousarray(image))
