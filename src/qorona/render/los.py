"""Line-of-sight render of the weighted log₁₀ Q⊥ volume into eclipse-like imagery.

Each pixel integrates spatially-weighted log₁₀ Q⊥ along its line of sight through the
:class:`~qorona.squashing.QPerpVolume`, on the orthographic plane-of-sky
:class:`~qorona.geometry.camera.OrthographicCamera`. Three colour channels carry the *same*
log₁₀ Q⊥ under three different *spatial* weightings, faking depth.

These details are load-bearing:

- **The quantitative signal is a weight-normalised average, not a raw sum.** Naïvely dropping
  ``NaN`` / out-of-domain samples would shorten the effective path and break the weight
  normalisation, so pixels with different missing fractions would stop being comparable. Instead the
  retained quantitative ``signal`` is, per channel, ``Σ wᵢ·log₁₀Q⊥ᵢ / Σ wᵢ`` over the valid samples
  (the weight stays in the denominator), with a per-pixel **coverage** (valid weight / on-path
  weight) alongside so low-coverage pixels are auditable.

- **The depth colour comes from the LOS *integral*, reconstructed from that signal.** The
  weight-normalised average is grayscale by construction: three *averages* of the same log₁₀ Q⊥
  differ only marginally, so the cross-channel *magnitude* gradient that fakes depth is exactly the
  part ``Σ w`` divides out. Recovering depth needs the raw integral ``Σ w·log₁₀Q⊥`` instead (and the
  per-preset linear/log scaling only makes sense on an integral), so the *image* is built from a
  per-channel display magnitude selected by the ``display`` mode: ``"balanced"`` (default)
  ``signal · Σ_onpath w`` (the integral rebuilt as the NaN-comparable average · a gap-independent
  geometric weight budget), ``"raw"`` ``signal · Σ_valid w`` (gappy rays dim), or ``"coverage"``
  ``signal · Σ_valid w / coverage`` (an approximate completion), then a **per-channel** percentile
  stretch (a pooled stretch crushes the steep channel to ≈0). A separate grayscale measurement image
  (the channel-mean of ``signal``, linearly stretched) is emitted alongside.

- **The display clamp is the render's, not the volume's.** The volume stores raw, truthful log₁₀ Q⊥
  (including the real-data sub-floor tail); the render clamps to ``[log₁₀2, log_max]`` for an 8-bit
  range (the lower bound lifts the retained sub-floor tail to the floor, the upper tames the
  separatrix singularities) and reports the fraction clamped at each end, so nothing is silently
  discarded. ``floor=False`` skips the lower clamp for diagnostics.

- **Occultation is two orthogonal mechanisms.** An *in-integral* far-side mask (what a line of sight
  sees past an opaque body) and an *image-level* dark disk (the eclipse occulter). The ``occult``
  mode selects them: ``"eclipse"`` (default, the primary synthetic-eclipse image) integrates the
  full off-limb corona and darkens the disk; ``"opaque"`` keeps the body opaque (a 3-D view);
  ``"composite"`` renders the eclipse view with the disk filled by the near-limb view,
  toned down (two passes of the standard machinery combined by :func:`_composite_image`);
  ``"none"`` disables both. The in-integral mask is the only part in the hot loop (kept
  kernel/NumPy parity-exact); the dark disk is a post-stretch radial vignette in
  :func:`_finalize`, identical across both paths.

The two default weighting profiles reproduce the published total-eclipse squashing-factor render
recipe: the explicit large- and small-field-of-view weightings of Mikić et al. (2018),
Supplementary Note 2, implemented verbatim, except that Qorona integrates log₁₀ **Q⊥** in place
of log₁₀ Q. Every constant is overridable: they are display choices, not physics.

The quadrature runs on a CUDA one-ray-per-thread kernel when a GPU is present and the grids are
JIT-able, else on a numba ``prange``-over-rays kernel (each ray on its own lane); the NumPy
implementation is the no-numba fallback and both kernels' reference, and all paths agree to
floating-point noise (``validation/render_parity.py`` characterizes the GPU tiers).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from astropy import units as u

from qorona.accel import HAVE_NUMBA, JitGrid, apply_workers
from qorona.console import print_success, progress_bar
from qorona.geometry.camera import OrthographicCamera
from qorona.render.image import eclipse_alpha, occultation_mask, scale_intensity, write_png
from qorona.squashing.volume import QPerpVolume

if TYPE_CHECKING:
    # Type-only: the Thomson weight is duck-typed at runtime and imported here for annotations only.
    from qorona.radiation.thomson import RadialCoefficients, ThomsonWeight

__all__ = [
    "LARGE_FOV",
    "SMALL_FOV",
    "RenderResult",
    "WeightingPreset",
    "render",
]

#: The theoretical Q⊥ floor in log₁₀: the default display lower bound, lifting the retained
#: real-data sub-floor tail to the floor for a clean image.
LOG_FLOOR = float(np.log10(2.0))

#: One solar radius in megametres, for stating scale heights in physical Mm while the kernels work
#: in R☉.
_MM_PER_R_SUN = (1.0 * u.R_sun).to_value(u.Mm)

#: Number of ray-chunks the kernel render splits the image into, for progress only. Unlike the NumPy
#: path (whose chunk bounds the materialised neighbourhood memory), the kernel touches one stencil
#: at a time and needs no memory bound, so it chunks only for the progress bar. A fixed, small count
#: keeps the bar live while avoiding the per-launch ``prange`` overhead of many tiny chunks.
_RENDER_PROGRESS_CHUNKS = 64

#: Diverging warm/cool colours for ``--polarity-mode``: inward (-1), neutral (0), outward (+1); the
#: standard slog-Q magnetic palette applied to the line-of-sight net polarity. A display choice.
_POLARITY_STOPS = ((0.13, 0.40, 0.92), (0.97, 0.97, 0.97), (0.82, 0.12, 0.12))

#: Outer edge of the composite mode's disk rim glow, in units of ``r_occult``: the disk layer keeps
#: full weight inside the limb and decays smoothly to nothing here.
_COMPOSITE_GLOW_OUTER = 1.15

#: Line-of-sight step ceiling (R☉) for the composite mode's disk pass: the small-fov weighting's
#: sharpest channel has a 21 Mm (~0.03 R☉) scale height, which the default 0.02 R☉ step aliases
#: into concentric rings on the disk. The base pass keeps the caller's ``step``.
_COMPOSITE_DISK_STEP = 0.005


@dataclass(frozen=True)
class WeightingPreset:
    """A depth-weighting profile: per-channel spatial weights and an intensity scaling.

    The weight of a sample at signed line-of-sight distance ``s`` and heliocentric radius ``r`` is a
    product of the factors that are set:

    - ``sigma``: a line-of-sight Gaussian ``exp(-s²/(2 r² sigma²))``; ``sigma`` is **angular**
      (``s/r`` is a ratio), a cone of fixed angular half-width at every height, shared by all
      channels.
    - ``height_powers``: a per-channel radial power ``r^(-n)`` (``n = 1`` is equal weighting).
    - ``scale_heights``: a per-channel radial exponential ``exp(-r/λ)`` with ``λ`` in R☉.

    The two named presets are deliberately asymmetric (one carries the Gaussian and the radial
    power, the other only the radial exponential); do not merge them. ``scaling`` maps the
    per-channel integrated signal to 8-bit intensity, linearly or logarithmically.

    Attributes
    ----------
    name
        Short identifier (used in the run summary and provenance).
    sigma
        Angular Gaussian width in radians, or ``None`` for no line-of-sight Gaussian.
    height_powers
        Per-channel ``(R, G, B)`` powers ``n`` in ``r^(-n)``, or ``None``.
    scale_heights
        Per-channel ``(R, G, B)`` scale heights ``λ`` in R☉ in ``exp(-r/λ)``, or ``None``.
    scaling
        ``"linear"`` or ``"log"`` mapping of channel signal to display intensity.
    stretch_radius
        If set, the per-channel intensity percentiles are anchored on pixels within this impact
        parameter (R☉), the disk and near limb, then applied to the whole frame. This keeps the
        faint wide off-limb periphery (whose ``exp(-r/λ)`` weight collapses toward zero for a
        low-corona preset) from dragging the stretch and washing the disk out at a large field of
        view. ``None`` anchors on the full frame (the whole-corona default).
    """

    name: str
    sigma: float | None
    height_powers: tuple[float, float, float] | None
    scale_heights: tuple[float, float, float] | None
    scaling: Literal["linear", "log"]
    stretch_radius: float | None = None

    def channel_weights(self, s: np.ndarray, r: np.ndarray) -> np.ndarray:
        """Return the per-channel spatial weight, shape ``(3, *r.shape)``.

        Parameters
        ----------
        s
            Signed line-of-sight distance from the plane of sky, broadcastable to ``r``.
        r
            Heliocentric radius of each sample.
        """
        weight = np.ones((3, *r.shape))
        channel_shape = (3,) + (1,) * r.ndim
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.sigma is not None:
                weight = weight * np.exp(-0.5 * (s / (r * self.sigma)) ** 2)[None]
            if self.height_powers is not None:
                powers = np.asarray(self.height_powers).reshape(channel_shape)
                weight = weight * r[None] ** (-powers)
            if self.scale_heights is not None:
                scale = np.asarray(self.scale_heights).reshape(channel_shape)
                weight = weight * np.exp(-r[None] / scale)
        return weight


#: Large field-of-view profile (the reference whole-corona render): a line-of-sight Gaussian of
#: angular FWHM 40° (sigma stored numerically) times a radial power ``r^(-n)`` with
#: ``n = 3, 2, 1.5`` for R, G, B, linear intensity scaling.
LARGE_FOV = WeightingPreset(
    name="large-fov",
    sigma=20.0 * np.pi / 180.0 / np.sqrt(2.0 * np.log(2.0)),  # FWHM 40° → sigma ≈ 0.2965 rad ≈ 17°
    height_powers=(3.0, 2.0, 1.5),
    scale_heights=None,
    scaling="linear",
    stretch_radius=None,
)

#: Small field-of-view profile (emphasising low-corona structure): a radial exponential
#: ``exp(-r/λ)`` with scale heights ``λ = 21, 84, 140 Mm`` for R, G, B (no line-of-sight Gaussian),
#: logarithmic intensity scaling. The intensity stretch is anchored on the disk and near limb
#: (``stretch_radius = 1.1 R☉``) so the faint wide-field periphery cannot wash the disk out.
SMALL_FOV = WeightingPreset(
    name="small-fov",
    sigma=None,
    height_powers=None,
    scale_heights=(21.0 / _MM_PER_R_SUN, 84.0 / _MM_PER_R_SUN, 140.0 / _MM_PER_R_SUN),
    scaling="log",
    stretch_radius=1.1,
)


@dataclass(frozen=True)
class RenderResult:
    """The rendered image and the provenance needed to read it quantitatively.

    Attributes
    ----------
    image
        ``(H, W, 3)`` depth-coloured display intensity in ``[0, 1]`` (``0`` where a pixel has no
        valid sample): the per-channel LOS-integral reconstruction set by ``display_mode``, after
        the per-channel intensity stretch.
    grayscale
        ``(H, W)`` quantitative measurement image in ``[0, 1]``: the channel-mean of ``signal``
        (line-of-sight-average log₁₀ Q⊥), linearly stretched, independent of ``display_mode``.
    coverage
        ``(H, W)`` fraction of each pixel's on-path weight that came from valid (finite) samples;
        low where the line of sight crossed many ``NaN`` voxels.
    signal
        ``(H, W, 3)`` the raw weight-normalised log₁₀ Q⊥ per channel, before intensity scaling;
        ``NaN`` where a pixel has no valid sample. The quantitative product: comparable across
        pixels regardless of missing fraction (the weight-normalised average). In
        ``"composite"`` mode the occulter hole (``NaN`` in the eclipse base) carries the disk
        pass's near-side surface Q instead, with a hard hand-off at the occulter radius;
        ``grayscale`` and ``coverage`` follow the same merge.
    polarity
        ``(H, W)`` per-pixel net magnetic polarity in ``[-1, +1]``: the weight-averaged mean
        footpoint sign along the line of sight (warm ``> 0`` outward / cool ``< 0`` inward), or
        ``None`` unless a polarity-colouring mode was requested; ``NaN`` where no valid sample.
    lower_clamped_fraction, upper_clamped_fraction
        Fraction of all valid samples clamped at the display floor (the retained sub-floor tail) and
        at ``log_max`` (the separatrix singularities), so the breach numbers stay recoverable.
    preset_name
        The weighting preset used.
    display_mode
        The display reconstruction used for ``image`` (``"balanced"``/``"raw"``/``"coverage"``).
    """

    image: np.ndarray
    grayscale: np.ndarray
    coverage: np.ndarray
    signal: np.ndarray
    polarity: np.ndarray | None
    lower_clamped_fraction: float
    upper_clamped_fraction: float
    preset_name: str
    display_mode: str

    def summary(self) -> str:
        """Return a one-line end-of-run summary: preset, coverage, and clamp fractions."""
        covered = self.coverage[self.coverage > 0.0]
        mean_coverage = float(np.mean(covered)) if covered.size else 0.0
        polarity_note = " · polarity-coloured" if self.polarity is not None else ""
        return (
            f"{self.preset_name}{polarity_note} · mean coverage {mean_coverage:.2f} · "
            f"clamped {self.lower_clamped_fraction:.1%} at floor, "
            f"{self.upper_clamped_fraction:.1%} at log_max"
        )

    def save_png(self, path: str | Path) -> None:
        """Write the depth-coloured RGB image to ``path`` as an 8-bit PNG."""
        rgb = np.ascontiguousarray(
            (np.clip(np.nan_to_num(self.image), 0.0, 1.0) * 255.0).round().astype(np.uint8)
        )
        write_png(Path(path), rgb)

    def save_grayscale_png(self, path: str | Path) -> None:
        """Write the grayscale measurement image to ``path`` as an 8-bit PNG."""
        gray = (np.clip(np.nan_to_num(self.grayscale), 0.0, 1.0) * 255.0).round().astype(np.uint8)
        write_png(Path(path), np.ascontiguousarray(np.repeat(gray[:, :, None], 3, axis=2)))


def _weighted_average(
    values: np.ndarray, weights: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Weight-normalised average of ``values`` over the last axis, counting only ``valid`` samples.

    Returns ``Σ wᵢ·valuesᵢ / Σ wᵢ`` over the valid samples (missing samples contribute to neither
    sum) and the total valid weight alongside; the average is ``NaN`` where no sample is valid. This
    is the render's quadrature; it stays correct under partial masking because the weight remains
    in the denominator, so pixels with different missing fractions stay comparable.
    """
    safe_weight = np.where(valid, weights, 0.0)
    total = np.sum(safe_weight, axis=-1)
    weighted = np.sum(safe_weight * np.where(valid, values, 0.0), axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0.0, weighted / total, np.nan), total


def _polarity_colour(polarity: np.ndarray) -> np.ndarray:
    """Map per-ray net polarity in ``[-1, +1]`` to warm/cool RGB (:data:`_POLARITY_STOPS`).

    ``-1`` (inward) → cool, ``0`` (neutral / closed / cancelled) → near-white, ``+1`` (outward) →
    warm, linearly interpolated. ``NaN`` maps to the neutral midpoint, harmless, as it is then
    multiplied by the zero structure brightness of an unlit pixel. Returns ``(n, 3)``.
    """
    t = (np.clip(np.nan_to_num(polarity), -1.0, 1.0) + 1.0) / 2.0
    stops = np.asarray(_POLARITY_STOPS)
    nodes = np.array([0.0, 0.5, 1.0])
    return np.stack([np.interp(t, nodes, stops[:, channel]) for channel in range(3)], axis=-1)


def _display_magnitude(
    signal: np.ndarray,
    den: np.ndarray,
    onpath: np.ndarray,
    coverage: np.ndarray,
    display: Literal["balanced", "raw", "coverage"],
) -> np.ndarray:
    """Per-channel display magnitude: the depth-colour-bearing reconstruction of the LOS integral.

    ``signal`` is the weight-normalised average ``Σw·v/Σw``, the quantitative,
    NaN-comparable product, but grayscale on its own (three averages of the same log₁₀ Q⊥ barely
    differ). The cross-channel magnitude gradient that fakes depth is the raw integral ``Σw·v``,
    recovered here via the per-channel weight budgets without sacrificing comparability:

    - ``"balanced"``: ``signal · onpath`` with ``onpath = Σ_onpath w`` the per-channel **on-path**
      geometric weight budget (a function of impact parameter only, independent of the field and the
      NaN mask), so the integral is the NaN-comparable average · a gap-independent geometry.
    - ``"raw"``: ``signal · den = Σ_valid w·v``, the literal integral; gappy rays read dimmer.
    - ``"coverage"``: ``(signal · den) / coverage``, the valid integral rescaled by the shared
      coverage scalar (an approximate NaN completion between the other two).

    All three coincide on fully-covered rays; the result is ``NaN`` wherever ``signal`` is.
    """
    if display == "balanced":
        return signal * onpath
    if display == "raw":
        return signal * den
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(coverage > 0.0, coverage, np.nan)
        return signal * den / scale[:, None]


def _preset_factors(
    preset: WeightingPreset,
) -> tuple[float, np.ndarray, bool, np.ndarray, bool]:
    """Pack a preset's optional weight factors into nopython-safe render-kernel arguments.

    ``sigma`` carries a ``NaN`` sentinel for "no line-of-sight Gaussian"; the per-channel ``powers``
    and ``scales`` ride a fixed ``(3,)`` array with a use-flag, so the kernel sees one
    consistently-typed argument set for either preset (a disabled factor's array is never read).
    """
    sigma = float(preset.sigma) if preset.sigma is not None else np.nan
    use_powers = preset.height_powers is not None
    powers = np.asarray(preset.height_powers if use_powers else (0.0, 0.0, 0.0), dtype=np.float64)
    use_scales = preset.scale_heights is not None
    scales = np.asarray(preset.scale_heights if use_scales else (1.0, 1.0, 1.0), dtype=np.float64)
    return sigma, powers, use_powers, scales, use_scales


def _thomson_average_weight(
    thomson: ThomsonWeight,
    table: RadialCoefficients,
    points: np.ndarray,
    impact: np.ndarray,
    radius: np.ndarray,
) -> np.ndarray:
    """Return the per-sample Thomson scalar ``Nₑ·I(r, χ̄)`` for the NumPy render's average weights.

    The NumPy companion of the kernel's per-step Thomson fold: samples ``Nₑ`` from the density
    volume and the radius-only coefficients from the *same* table the kernel uses, combining them
    with ``sin²χ̄ = rho²/r²``. Out-of-shell samples carry a zero density (and are masked from the
    average regardless), so the ``1/r²`` is harmless there.

    Parameters
    ----------
    thomson
        The Thomson weight (density volume + mode + limb darkening).
    table
        The radial coefficient table (built once, shared with the kernel for parity).
    points
        ``(m, n_steps, 3)`` line-of-sight sample coordinates.
    impact
        ``(m,)`` per-ray impact parameter rho.
    radius
        ``(m, n_steps)`` heliocentric radius of each sample.

    Returns
    -------
    numpy.ndarray
        ``(m, n_steps)`` scalar weight ``Nₑ·I``.
    """
    density = thomson.density.sample(points.reshape(-1, 3)).reshape(radius.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        c_tan, c_pol = table.evaluate(radius)
    sin_sq_chi = np.divide(
        impact[:, None] ** 2, radius**2, out=np.zeros_like(radius), where=radius > 0.0
    )
    if thomson.mode == "pB":
        intensity = c_pol * sin_sq_chi
    else:
        intensity = 2.0 * c_tan - c_pol * sin_sq_chi
    return density * intensity


def _thomson_kernel_args(
    thomson: ThomsonWeight | None, vol_placeholder: np.ndarray, grid_placeholder: object
) -> tuple[bool, np.ndarray, object, bool, float, float, np.ndarray, np.ndarray]:
    """Pack a Thomson weight into nopython-safe kernel arguments (placeholders when ``None``).

    Returns ``(use_thomson, density_volume, density_jit_grid, mode_is_pB, coeff_log_inner,
    coeff_inv_dlog, c_tan_table, c_pol_table)``. With ``thomson=None`` the density/grid/table slots
    carry harmless placeholders (the volume's own array and JIT grid, empty tables) the kernel
    never reads, so the argument set stays one consistently-typed shape for either case.
    """
    if thomson is None:
        empty = np.zeros(2)
        return (False, vol_placeholder, grid_placeholder, False, 0.0, 0.0, empty, empty)
    table = thomson.coefficient_table()
    return (
        True,
        np.ascontiguousarray(thomson.density.density),
        thomson.density.grid._jit_grid(),
        thomson.mode == "pB",
        table.log_inner,
        table.inv_dlog,
        np.ascontiguousarray(table.c_tan),
        np.ascontiguousarray(table.c_pol),
    )


def _composite_image(
    base: np.ndarray,
    disk: np.ndarray,
    impact: np.ndarray,
    r_occult: float,
    disk_tone: float,
    disk_desat: float,
) -> np.ndarray:
    """Screen the composite mode's toned disk layer over the eclipse ``base`` image.

    ``base`` is the finished eclipse render and ``disk`` the finished small-fov opaque render of
    the same volume and camera (``(H, W, 3)`` in ``[0, 1]``, with ``impact`` the matching per-pixel
    impact parameter). The disk layer is desaturated by ``disk_desat``, scaled to sit under the
    corona (its limb annulus matched to the base's inner corona and damped by ``disk_tone``), and
    screen-blended: full weight inside the limb, decaying to zero at :data:`_COMPOSITE_GLOW_OUTER`
    (the disk's rim glow). Screen keeps the streamer roots visible through the glow and reduces to
    the pure disk layer over the black occulter core; a hand-off that instead fades the layer into
    the occulter's feather produces a dark band at the limb.
    """
    layer = disk
    if disk_desat > 0.0:
        luminance = layer.mean(axis=2, keepdims=True)
        layer = luminance + (1.0 - disk_desat) * (layer - luminance)

    # Ring match: the disk layer's limb annulus lands at disk_tone times the base's inner corona,
    # keeping the balance stable across solutions; disk_tone is the taste knob.
    base_ring = (impact > 1.02 * r_occult) & (impact < 1.10 * r_occult)
    layer_ring = (impact > 0.90 * r_occult) & (impact < 0.98 * r_occult)
    tone = disk_tone
    if base_ring.any() and layer_ring.any():
        ring_level = layer[layer_ring].mean()
        if ring_level > 0.0:
            tone = disk_tone * base[base_ring].mean() / ring_level

    on_disk = impact < r_occult
    ramp = np.clip(
        (_COMPOSITE_GLOW_OUTER * r_occult - impact) / ((_COMPOSITE_GLOW_OUTER - 1.0) * r_occult),
        0.0,
        1.0,
    )
    weight = np.where(on_disk, 1.0, ramp * ramp * (3.0 - 2.0 * ramp))
    overlay = np.clip(layer * tone * weight[..., None], 0.0, 1.0)
    return 1.0 - (1.0 - base) * (1.0 - overlay)


def _finalize(
    signal: np.ndarray,
    coverage: np.ndarray,
    den: np.ndarray,
    onpath: np.ndarray,
    lower_clamped: int,
    upper_clamped: int,
    valid_total: int,
    *,
    preset: WeightingPreset,
    percentiles: tuple[float, float],
    impact: np.ndarray,
    display: Literal["balanced", "raw", "coverage"],
    polarity: np.ndarray | None,
    polarity_mode: Literal["none", "hue"],
    occult: Literal["eclipse", "opaque", "none"],
    r_occult: float,
    occult_softness: float,
    height: int,
    width: int,
    show_progress: bool,
) -> RenderResult:
    """Assemble the :class:`RenderResult` from the accumulated signal, weight budgets, and totals.

    The shared tail of both render paths: the per-channel display reconstruction
    (:func:`_display_magnitude`), the per-channel percentile stretch, the grayscale measurement
    image, the image-level eclipse occulter, the ``(H, W, ...)`` reshape, and the clamp-fraction
    provenance, all NumPy, cheap, run once, not in the hot path. Keeping it in one place makes the
    result construction identical across the kernel and NumPy paths, so neither the colour
    reconstruction nor the occulter can break their parity.

    The depth-coloured ``image`` comes from ``signal · weight-budget`` per the ``display`` mode (the
    ``signal`` carries no depth colour on its own; see :func:`_display_magnitude`); the
    ``grayscale`` image is the channel-mean of ``signal`` itself (line-of-sight-average log₁₀ Q⊥),
    linearly stretched and mode-independent. When the preset sets ``stretch_radius`` the colour
    stretch is anchored on the disk/near-limb pixels (``impact < stretch_radius``); the grayscale
    measurement image is left full-frame.

    When ``polarity_mode == "hue"`` the ``image`` is instead the per-ray net polarity coloured
    warm/cool (:func:`_polarity_colour`) times the grayscale structure brightness; the depth-colour
    reconstruction is skipped, so polarity owns the colour axis while structure owns the luminance.

    In ``"eclipse"`` mode the disk is darkened image-side (the orthogonal companion to the
    in-integral body mask, :func:`~qorona.render.image.occultation_mask`): the opaque core is
    ``NaN``-ed in ``signal`` and zeroed in ``coverage`` *before* the stretch (so its hidden
    through-disk column neither shows nor biases the off-limb contrast, and the metrics match the
    black image) while the partially-visible feather annulus stays as computed and both images are
    faded at the limb by the :func:`~qorona.render.image.eclipse_alpha` darkening. The scalar clamp
    fractions are left full-column: a volume
    dynamic-range diagnostic, not a per-pixel visible metric. ``"opaque"`` and ``"none"`` leave the
    images untouched here. The ``"composite"`` mode never reaches here: :func:`render` assembles it
    from an eclipse pass and a small-fov opaque pass (:func:`_composite_image`).
    """
    if occult == "eclipse":
        alpha = eclipse_alpha(impact, r_occult, occult_softness)
        core = alpha <= 0.0
        signal = np.where(core[:, None], np.nan, signal)
        coverage = np.where(core, 0.0, coverage)
    else:
        alpha = None

    grayscale = scale_intensity(signal.mean(axis=1)[:, None], "linear", percentiles)[:, 0]
    if alpha is not None:
        grayscale = grayscale * alpha

    if polarity_mode == "hue" and polarity is not None:
        # Polarity colouring: hue from the net column polarity, brightness from the structure (the
        # same geometric-weighted grayscale measurement, already eclipse-vignetted above).
        image = _polarity_colour(polarity) * grayscale[:, None]
        polarity_image: np.ndarray | None = polarity.reshape(height, width)
    else:
        magnitude = _display_magnitude(signal, den, onpath, coverage, display)
        anchor = None if preset.stretch_radius is None else impact < preset.stretch_radius
        image = scale_intensity(magnitude, preset.scaling, percentiles, anchor=anchor)
        if alpha is not None:
            image = image * alpha[:, None]
        polarity_image = None

    result = RenderResult(
        image=image.reshape(height, width, 3),
        grayscale=grayscale.reshape(height, width),
        coverage=coverage.reshape(height, width),
        signal=signal.reshape(height, width, 3),
        polarity=polarity_image,
        lower_clamped_fraction=lower_clamped / valid_total if valid_total else 0.0,
        upper_clamped_fraction=upper_clamped / valid_total if valid_total else 0.0,
        preset_name=preset.name,
        display_mode=display,
    )
    if show_progress:
        print_success(f"Rendered {width}x{height} image: {result.summary()}")
    return result


def _render_numpy(
    volume: QPerpVolume,
    camera: OrthographicCamera,
    *,
    preset: WeightingPreset,
    thomson: ThomsonWeight | None,
    clamp: tuple[float, float],
    floor: bool,
    step: float,
    occult: Literal["eclipse", "opaque", "none"],
    r_occult: float,
    occult_softness: float,
    percentiles: tuple[float, float],
    display: Literal["balanced", "raw", "coverage"],
    polarity_mode: Literal["none", "hue"],
    chunk_size: int,
    show_progress: bool,
) -> RenderResult:
    """Single-threaded NumPy render: the no-numba fallback and the kernel's reference.

    The reference quadrature: per ray-chunk it samples the volume on the shared ``s``-grid, masks
    in-shell + occulted + non-finite samples, and forms each channel's weight-normalised average,
    its coverage, and the per-channel valid/on-path weight budgets the display reconstruction needs.
    The in-integral body mask
    (the opaque photosphere) is on only for the ``"opaque"`` mode; the display reconstruction and
    the image-level eclipse occulter are applied later in :func:`_finalize`.

    An optional ``thomson`` weight multiplies its scalar ``Nₑ·I(r, χ̄)`` into the weighted-average
    accumulators only (the geometric on-path / coverage budgets stay scalar-free), via the same
    radial coefficient table the kernel reads, so the two paths stay parity-exact.
    """
    log_floor, log_max = clamp
    occult_body = occult == "opaque"
    thomson_table = thomson.coefficient_table() if thomson is not None else None
    rays = camera.rays()
    height, width = camera.pixels
    origins = rays.origins.reshape(-1, 3)
    impact = rays.impact.reshape(-1)
    n_rays = origins.shape[0]

    outer_radius = float(volume.grid.radii[-1])
    inner_radius = float(volume.grid.radii[0])
    n_steps = int(np.ceil(2.0 * outer_radius / step)) + 1
    s = np.linspace(-outer_radius, outer_radius, n_steps)

    signal = np.full((n_rays, 3), np.nan)
    coverage = np.zeros(n_rays)
    den = np.zeros((n_rays, 3))
    onpath = np.zeros((n_rays, 3))
    polarity = np.full(n_rays, np.nan) if polarity_mode != "none" else None
    valid_total = 0
    lower_clamped = 0
    upper_clamped = 0

    rays_per_chunk = max(1, chunk_size // n_steps)
    with progress_bar("Rendering line-of-sight Q⊥", n_rays, enabled=show_progress) as progress:
        for start in range(0, n_rays, rays_per_chunk):
            stop = min(start + rays_per_chunk, n_rays)
            batch_origins = origins[start:stop]
            batch_impact = impact[start:stop]

            points = batch_origins[:, None, :] + s[None, :, None] * rays.look
            radius = np.sqrt(batch_impact[:, None] ** 2 + s[None, :] ** 2)
            log_q = volume.sample(points.reshape(-1, 3)).reshape(batch_origins.shape[0], n_steps)

            in_shell = (radius >= inner_radius) & (radius <= outer_radius)
            on_path = in_shell
            if occult_body:
                on_path = on_path & ~occultation_mask(batch_impact, s, r_occult)
            valid = on_path & np.isfinite(log_q)

            lower_clamped += int(np.count_nonzero(valid & (log_q < log_floor)))
            upper_clamped += int(np.count_nonzero(valid & (log_q > log_max)))
            valid_total += int(np.count_nonzero(valid))

            clamped = np.clip(log_q, log_floor if floor else None, log_max)
            weights = preset.channel_weights(s, radius)  # (3, m, n_steps)
            # Thomson enters the average weight only; on-path/coverage budgets below stay geometric.
            average_weights = weights
            if thomson is not None and thomson_table is not None:
                scalar = _thomson_average_weight(
                    thomson, thomson_table, points, batch_impact, radius
                )
                average_weights = weights * scalar[None]
            channel_signal, channel_den = _weighted_average(
                clamped[None], average_weights, valid[None]
            )
            signal[start:stop] = channel_signal.T
            den[start:stop] = channel_den.T
            onpath[start:stop] = np.sum(np.where(on_path[None], weights, 0.0), axis=-1).T

            # Coverage: the valid fraction of the on-path weight, under a channel-averaged weight.
            reference_weight = np.mean(weights, axis=0)
            valid_weight = np.sum(np.where(valid, reference_weight, 0.0), axis=1)
            path_weight = np.sum(np.where(on_path, reference_weight, 0.0), axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                coverage[start:stop] = np.where(path_weight > 0.0, valid_weight / path_weight, 0.0)

            # Net polarity: the magnitude-weighted mean nearest-cell footpoint sign over valid
            # samples, under the same channel-mean ``reference_weight`` budget as coverage.
            if polarity is not None:
                sign = volume.sample_polarity(points.reshape(-1, 3)).reshape(
                    batch_origins.shape[0], n_steps
                )
                pol_num = np.sum(reference_weight * np.where(valid, sign, 0.0), axis=1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    polarity[start:stop] = np.where(
                        valid_weight > 0.0, pol_num / valid_weight, np.nan
                    )
            progress(stop)

    return _finalize(
        signal,
        coverage,
        den,
        onpath,
        lower_clamped,
        upper_clamped,
        valid_total,
        preset=preset,
        percentiles=percentiles,
        impact=impact,
        display=display,
        polarity=polarity,
        polarity_mode=polarity_mode,
        occult=occult,
        r_occult=r_occult,
        occult_softness=occult_softness,
        height=height,
        width=width,
        show_progress=show_progress,
    )


def _render_numba(
    volume: QPerpVolume,
    camera: OrthographicCamera,
    *,
    preset: WeightingPreset,
    thomson: ThomsonWeight | None,
    clamp: tuple[float, float],
    floor: bool,
    step: float,
    occult: Literal["eclipse", "opaque", "none"],
    r_occult: float,
    occult_softness: float,
    percentiles: tuple[float, float],
    display: Literal["balanced", "raw", "coverage"],
    polarity_mode: Literal["none", "hue"],
    workers: int | None,
    show_progress: bool,
) -> RenderResult:
    """numba ``prange``-over-rays render: output-identical to :func:`_render_numpy` to FP noise.

    Drives :func:`~qorona.accel.kernels.render_batch_jit` on ``workers`` threads, accumulating the
    per-chunk signal/coverage/weight-budgets and reducing the per-lane clamp counts into the global
    provenance. The image is split into a fixed small number of ray-chunks
    (:data:`_RENDER_PROGRESS_CHUNKS`) only to advance the progress bar; the kernel needs no memory
    bound. The kernel carries only the in-integral body mask (toggled by ``occult_body``); the
    display reconstruction and the image-level eclipse occulter are applied later in
    :func:`_finalize`, so the kernel needs no display or eclipse logic.

    An optional ``thomson`` weight passes its density JIT grid and radial coefficient table into the
    kernel, which folds ``Nₑ·I(r, χ̄)`` into the average accumulators only.
    """
    from qorona.accel.kernels import render_batch_jit

    apply_workers(workers)
    log_floor, log_max = clamp
    clamp_lower = floor
    occult_body = occult == "opaque"
    rays = camera.rays()
    height, width = camera.pixels
    origins = np.ascontiguousarray(rays.origins.reshape(-1, 3))
    impact = np.ascontiguousarray(rays.impact.reshape(-1))
    look = np.ascontiguousarray(rays.look, dtype=np.float64)
    n_rays = origins.shape[0]

    outer_radius = float(volume.grid.radii[-1])
    n_steps = int(np.ceil(2.0 * outer_radius / step)) + 1
    s = np.linspace(-outer_radius, outer_radius, n_steps)

    jit_grid = volume.grid._jit_grid()
    log_q_perp = np.ascontiguousarray(volume.log_q_perp)
    sigma, powers, use_powers, scales, use_scales = _preset_factors(preset)
    (
        use_thomson,
        density_vol,
        density_grid,
        thomson_pb,
        coeff_log_inner,
        coeff_inv_dlog,
        c_tan_table,
        c_pol_table,
    ) = _thomson_kernel_args(thomson, log_q_perp, jit_grid)
    compute_polarity = polarity_mode != "none"
    polarity_vol = (
        np.ascontiguousarray(volume.polarity, dtype=np.float32)
        if compute_polarity and volume.polarity is not None
        else np.zeros((1, 1, 1, 1), dtype=np.float32)
    )

    signal = np.full((n_rays, 3), np.nan)
    coverage = np.zeros(n_rays)
    den = np.zeros((n_rays, 3))
    onpath = np.zeros((n_rays, 3))
    polarity = np.full(n_rays, np.nan) if compute_polarity else None
    valid_total = 0
    lower_clamped = 0
    upper_clamped = 0

    rays_per_chunk = max(1, -(-n_rays // _RENDER_PROGRESS_CHUNKS))
    with progress_bar(
        "Rendering line-of-sight Q⊥ (numba)", n_rays, enabled=show_progress
    ) as progress:
        for start in range(0, n_rays, rays_per_chunk):
            stop = min(start + rays_per_chunk, n_rays)
            (
                chunk_signal,
                chunk_coverage,
                chunk_counts,
                chunk_den,
                chunk_onpath,
                chunk_polarity,
            ) = render_batch_jit(
                origins[start:stop],
                look,
                impact[start:stop],
                s,
                log_q_perp,
                jit_grid,
                sigma,
                powers,
                use_powers,
                scales,
                use_scales,
                float(log_floor),
                float(log_max),
                clamp_lower,
                float(r_occult),
                occult_body,
                use_thomson,
                density_vol,
                density_grid,
                thomson_pb,
                coeff_log_inner,
                coeff_inv_dlog,
                c_tan_table,
                c_pol_table,
                compute_polarity,
                polarity_vol,
            )
            signal[start:stop] = chunk_signal
            coverage[start:stop] = chunk_coverage
            den[start:stop] = chunk_den
            onpath[start:stop] = chunk_onpath
            if polarity is not None:
                polarity[start:stop] = chunk_polarity
            lower_clamped += int(chunk_counts[:, 0].sum())
            upper_clamped += int(chunk_counts[:, 1].sum())
            valid_total += int(chunk_counts[:, 2].sum())
            progress(stop)

    return _finalize(
        signal,
        coverage,
        den,
        onpath,
        lower_clamped,
        upper_clamped,
        valid_total,
        preset=preset,
        percentiles=percentiles,
        impact=impact,
        display=display,
        polarity=polarity,
        polarity_mode=polarity_mode,
        occult=occult,
        r_occult=r_occult,
        occult_softness=occult_softness,
        height=height,
        width=width,
        show_progress=show_progress,
    )


def _render_cuda(
    volume: QPerpVolume,
    camera: OrthographicCamera,
    *,
    preset: WeightingPreset,
    thomson: ThomsonWeight | None,
    clamp: tuple[float, float],
    floor: bool,
    step: float,
    occult: Literal["eclipse", "opaque", "none"],
    r_occult: float,
    occult_softness: float,
    percentiles: tuple[float, float],
    display: Literal["balanced", "raw", "coverage"],
    polarity_mode: Literal["none", "hue"],
    precision: str,
    show_progress: bool,
) -> RenderResult:
    """CUDA one-ray-per-thread render: output-identical to :func:`_render_numpy` to FP noise.

    Drives :func:`~qorona.accel.cuda_kernels.render_batch_cuda`, which uploads the volume once and
    launches the kernel in ray chunks purely to advance the progress bar. ``precision`` selects the
    kernel tier: ``"mixed"`` (default; ``"float32"`` aliases it) samples in float32 and accumulates
    in float64, ``"float64"`` is the all-double reference. The per-ray accumulation semantics and
    the :func:`_finalize` tail are shared with the CPU paths, so the display reconstruction and the
    eclipse occulter cannot diverge across backends.
    """
    from qorona.accel.cuda_kernels import render_batch_cuda

    log_floor, log_max = clamp
    clamp_lower = floor
    occult_body = occult == "opaque"
    rays = camera.rays()
    height, width = camera.pixels
    origins = np.ascontiguousarray(rays.origins.reshape(-1, 3))
    impact = np.ascontiguousarray(rays.impact.reshape(-1))
    look = np.ascontiguousarray(rays.look, dtype=np.float64)
    n_rays = origins.shape[0]

    outer_radius = float(volume.grid.radii[-1])
    n_steps = int(np.ceil(2.0 * outer_radius / step)) + 1
    s = np.linspace(-outer_radius, outer_radius, n_steps)

    # The dispatcher admits only JIT-able grids here, so the casts are narrowing, not conversions.
    jit_grid = cast(JitGrid, volume.grid._jit_grid())
    log_q_perp = np.ascontiguousarray(volume.log_q_perp)
    sigma, powers, use_powers, scales, use_scales = _preset_factors(preset)
    (
        use_thomson,
        density_vol,
        density_grid,
        thomson_pb,
        coeff_log_inner,
        coeff_inv_dlog,
        c_tan_table,
        c_pol_table,
    ) = _thomson_kernel_args(thomson, log_q_perp, jit_grid)
    compute_polarity = polarity_mode != "none"
    polarity_vol = (
        np.ascontiguousarray(volume.polarity, dtype=np.float32)
        if compute_polarity and volume.polarity is not None
        else np.zeros((1, 1, 1, 1), dtype=np.float32)
    )

    with progress_bar(
        "Rendering line-of-sight Q⊥ (cuda)", n_rays, enabled=show_progress
    ) as progress:
        signal, coverage, counts, den, onpath, polarity_out = render_batch_cuda(
            origins,
            look,
            impact,
            s,
            log_q_perp,
            jit_grid,
            sigma,
            powers,
            use_powers,
            scales,
            use_scales,
            float(log_floor),
            float(log_max),
            clamp_lower,
            float(r_occult),
            occult_body,
            use_thomson,
            density_vol,
            cast(JitGrid, density_grid),
            thomson_pb,
            coeff_log_inner,
            coeff_inv_dlog,
            c_tan_table,
            c_pol_table,
            compute_polarity,
            polarity_vol,
            precision=precision,
            chunks=_RENDER_PROGRESS_CHUNKS,
            progress=progress,
        )

    polarity = polarity_out if compute_polarity else None
    return _finalize(
        signal,
        coverage,
        den,
        onpath,
        int(counts[:, 0].sum()),
        int(counts[:, 1].sum()),
        int(counts[:, 2].sum()),
        preset=preset,
        percentiles=percentiles,
        impact=impact,
        display=display,
        polarity=polarity,
        polarity_mode=polarity_mode,
        occult=occult,
        r_occult=r_occult,
        occult_softness=occult_softness,
        height=height,
        width=width,
        show_progress=show_progress,
    )


def render(
    volume: QPerpVolume,
    camera: OrthographicCamera,
    *,
    preset: WeightingPreset = LARGE_FOV,
    thomson: ThomsonWeight | None = None,
    clamp: tuple[float, float] = (LOG_FLOOR, 7.0),
    floor: bool = True,
    step: float = 0.02,
    occult: Literal["eclipse", "opaque", "composite", "none"] = "eclipse",
    r_occult: float = 1.0,
    occult_softness: float = 0.03,
    disk_tone: float = 0.8,
    disk_desat: float = 0.4,
    percentiles: tuple[float, float] = (1.0, 99.5),
    display: Literal["balanced", "raw", "coverage"] = "balanced",
    polarity_mode: Literal["none", "hue"] = "none",
    chunk_size: int = 500_000,
    workers: int | None = None,
    device: str = "auto",
    precision: str = "mixed",
    show_progress: bool = True,
) -> RenderResult:
    """Render the Q⊥ volume to an eclipse-like image from a camera viewpoint.

    Dispatches to a CUDA one-ray-per-thread kernel when a GPU is usable (``device``) and the grids
    are JIT-able, else to a numba ``prange``-over-rays kernel when numba is installed, else to the
    single-threaded NumPy path (also both kernels' reference); all paths are output-identical to
    floating-point noise. ``device="gpu"`` demands a GPU and raises without one; a non-JIT-able
    grid falls back silently, exactly as the volume build does.

    The optional Thomson weight (``thomson``) is an off-by-default radiometric factor on an
    axis orthogonal to the geometric ``preset``: it biases the rendered Q⊥ toward bright dense
    low-corona plasma by multiplying ``Nₑ·I(r, χ̄)`` into the weighted-average ``signal`` only; the
    depth-colour reconstruction and the coverage stay identical to ``thomson=None``.

    The body of radius ``r_occult`` is occulted by one of two orthogonal mechanisms, selected by
    ``occult``: an *in-integral* far-side mask (what a line of sight actually sees past an opaque
    body) and an *image-level* dark disk (the eclipse occulter). ``"eclipse"`` (the default, the
    primary synthetic-eclipse image) integrates the full off-limb corona on both sides and darkens
    the disk image-side; ``"opaque"`` keeps the body opaque so near-side structure shows
    (a 3-D view); ``"composite"`` renders the eclipse view with the disk filled by the
    near-limb view, separately stretched, toned by ``disk_tone`` / ``disk_desat``, and
    finished with a rim glow (a disk-corona composite; see :func:`_composite_image`); ``"none"``
    disables both for a fully translucent corona.

    Parameters
    ----------
    volume
        The log₁₀ Q⊥ volume to integrate.
    camera
        The orthographic plane-of-sky camera.
    preset
        The depth-weighting profile (default :data:`LARGE_FOV`). A preset may also anchor the
        intensity stretch on the disk (``stretch_radius``); :data:`SMALL_FOV` does, so its
        low-corona render keeps its contrast instead of washing out at a wide field of view.
    thomson
        Optional Thomson/pB radiometric weight (a :class:`~qorona.radiation.thomson.ThomsonWeight`),
        off by default. When set it biases the weighted-average ``signal`` toward bright dense
        plasma (``"K"`` total-brightness or ``"pB"`` polarized emphasis); the depth colour and
        coverage are unaffected. Composes with ``preset`` on an orthogonal axis.
    clamp
        Display ``(log_floor, log_max)`` applied to log₁₀ Q⊥ before integrating: the floor lifts
        the retained sub-floor tail, ``log_max`` tames the separatrix singularities.
    floor
        When ``False`` skip the lower clamp (keep the sub-floor tail) for diagnostics; the upper
        clamp still applies.
    step
        Line-of-sight sample spacing in R☉.
    occult
        Occultation mode. ``"eclipse"``: dark solar disk, off-limb corona only (the total-eclipse
        look); ``"opaque"``: opaque body, near-side corona over the disk shows (a 3-D view);
        ``"composite"``: the eclipse view with the disk filled by the near-limb view,
        toned down (a disk-corona composite); ``"none"``: no occultation, full corona on both
        sides including behind the disk.
    r_occult
        Body / occulter radius in R☉, the photosphere. Shared by both mechanisms: the far side of
        this body is dropped in ``"opaque"`` / ``"composite"``, and the eclipse disk has this
        radius.
    occult_softness
        Radial feather width in R☉ of the eclipse disk edge: ``0`` is a hard black circle; ``> 0``
        ramps the opacity ``0 → 1`` (smoothstep) across ``[r_occult - occult_softness, r_occult]``
        for a soft, slightly-transparent limb. Used by ``"eclipse"`` and by ``"composite"``'s
        base layer.
    disk_tone
        Composite mode: the disk layer's brightness relative to the base's inner corona
        (its limb annulus lands at this fraction of the corona's). Ignored by the other modes.
    disk_desat
        Composite mode: desaturation of the disk layer, ``0`` (untouched) to ``1`` (grayscale).
        Ignored by the other modes.
    percentiles
        Low/high percentiles for the per-channel intensity stretch.
    display
        Depth-colour reconstruction for the ``image`` (the quantitative ``signal`` is unaffected).
        ``"balanced"`` (default), ``signal · on-path weight budget``: the LOS integral rebuilt as
        the NaN-comparable average times a gap-independent geometric weight, depth colour without
        dimming gappy rays. ``"raw"``, ``signal · valid weight budget``: the literal
        integral, gappy rays read dimmer (a coverage diagnostic). ``"coverage"``: the raw integral
        rescaled by the shared coverage scalar (an approximate completion). All three coincide on
        fully-covered rays; see :func:`_display_magnitude`.
    polarity_mode
        Magnetic-polarity colouring. ``"none"`` (default): the depth-coloured image, unchanged.
        ``"hue"``: colour by the line-of-sight **net polarity** (warm outward / cool inward,
        neutral white), brightness from the structure; needs a volume carrying the polarity channel.
    chunk_size
        Rays · line-of-sight steps processed per batch on the NumPy path, bounding its peak memory.
        Unused by the kernel path, which touches one stencil at a time and chunks only for progress.
    workers
        numba thread count for the CPU kernel (``None`` = all cores; ``1`` = serial). Ignored by
        the CUDA path and without numba; the NumPy fallback is single-threaded.
    device
        Compute backend: ``"auto"`` (the default) selects the GPU when one is usable and the CPU
        otherwise; ``"gpu"`` demands a GPU (raises without one); ``"cpu"`` forces the CPU tiers.
    precision
        CUDA kernel tier, GPU only (the CPU tiers always compute in float64): ``"mixed"`` (the
        default) samples in float32 and accumulates in float64; ``"float64"`` is the all-double
        reference; ``"float32"`` is accepted as an alias of ``"mixed"``.
    show_progress
        Whether to display progress.

    Returns
    -------
    RenderResult
        The depth-coloured image, the grayscale measurement image, coverage, the raw per-channel
        signal, and clamp provenance.
    """
    if occult not in ("eclipse", "opaque", "composite", "none"):
        raise ValueError(
            f"occult must be 'eclipse', 'opaque', 'composite', or 'none', not {occult!r}"
        )
    if occult == "composite":
        # The composite is two passes of the standard machinery: the eclipse base, and the disk
        # layer as a small-fov opaque render (its scale-height weighting and disk-anchored log
        # stretch are what keep the on-disk structure legible and coloured), screened together
        # image-side. The quantitative arrays carry the base's off-limb corona with the disk
        # pass's near-side surface Q filling the occulter hole: a hard hand-off at ``r_occult``,
        # never a blend, because the two passes weight the line of sight differently and mixed
        # values would not be a measurement of either. The grayscale is rederived from the merged
        # signal the way :func:`_finalize` derives it (channel-mean, linear stretch), minus the
        # eclipse vignette, which would black out the filled hole.
        base = render(
            volume,
            camera,
            preset=preset,
            thomson=thomson,
            clamp=clamp,
            floor=floor,
            step=step,
            occult="eclipse",
            r_occult=r_occult,
            occult_softness=occult_softness,
            percentiles=percentiles,
            display=display,
            polarity_mode=polarity_mode,
            chunk_size=chunk_size,
            workers=workers,
            device=device,
            precision=precision,
            show_progress=show_progress,
        )
        disk = render(
            volume,
            camera,
            preset=SMALL_FOV,
            thomson=thomson,
            clamp=clamp,
            floor=floor,
            step=min(step, _COMPOSITE_DISK_STEP),
            occult="opaque",
            r_occult=r_occult,
            occult_softness=occult_softness,
            percentiles=percentiles,
            display=display,
            polarity_mode=polarity_mode,
            chunk_size=chunk_size,
            workers=workers,
            device=device,
            precision=precision,
            show_progress=show_progress,
        )
        impact = camera.rays().impact
        image = _composite_image(base.image, disk.image, impact, r_occult, disk_tone, disk_desat)
        hole = np.isnan(base.signal) & (impact < r_occult)[..., None]
        signal = np.where(hole, disk.signal, base.signal)
        hole_pixel = hole.all(axis=2)
        coverage = np.where(hole_pixel, disk.coverage, base.coverage)
        polarity = base.polarity
        if polarity is not None and disk.polarity is not None:
            polarity = np.where(hole_pixel, disk.polarity, polarity)
        grayscale = scale_intensity(signal.mean(axis=2)[..., None], "linear", percentiles)[..., 0]
        return replace(
            base,
            image=image,
            grayscale=grayscale,
            coverage=coverage,
            signal=signal,
            polarity=polarity,
        )
    if display not in ("balanced", "raw", "coverage"):
        raise ValueError(f"display must be 'balanced', 'raw', or 'coverage', not {display!r}")
    if polarity_mode not in ("none", "hue"):
        raise ValueError(f"polarity_mode must be 'none' or 'hue', not {polarity_mode!r}")
    if polarity_mode != "none" and volume.polarity is None:
        raise ValueError(
            "polarity colouring needs a volume with the polarity channel, but this one has none; "
            "rebuild it with the paint or per-voxel builder (the reference builder omits polarity)"
        )
    from qorona.accel import resolve_device

    # The kernels need both the volume grid and (when weighting) the density grid to be JIT-able.
    thomson_jit = thomson is None or thomson.density.grid._jit_grid() is not None
    kernel_grids = volume.grid._jit_grid() is not None and thomson_jit
    if resolve_device(device) == "gpu" and kernel_grids:
        return _render_cuda(
            volume,
            camera,
            preset=preset,
            thomson=thomson,
            clamp=clamp,
            floor=floor,
            step=step,
            occult=occult,
            r_occult=r_occult,
            occult_softness=occult_softness,
            percentiles=percentiles,
            display=display,
            polarity_mode=polarity_mode,
            precision=precision,
            show_progress=show_progress,
        )
    if HAVE_NUMBA and kernel_grids:
        return _render_numba(
            volume,
            camera,
            preset=preset,
            thomson=thomson,
            clamp=clamp,
            floor=floor,
            step=step,
            occult=occult,
            r_occult=r_occult,
            occult_softness=occult_softness,
            percentiles=percentiles,
            display=display,
            polarity_mode=polarity_mode,
            workers=workers,
            show_progress=show_progress,
        )
    return _render_numpy(
        volume,
        camera,
        preset=preset,
        thomson=thomson,
        clamp=clamp,
        floor=floor,
        step=step,
        occult=occult,
        r_occult=r_occult,
        occult_softness=occult_softness,
        percentiles=percentiles,
        display=display,
        polarity_mode=polarity_mode,
        chunk_size=chunk_size,
        show_progress=show_progress,
    )
