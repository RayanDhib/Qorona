"""Image output and the optional on-image provenance stamp.

Writes the rendered :class:`~qorona.render.los.RenderResult` to PNG, reusing the dependency-free
``save_png`` / ``save_grayscale_png`` in :mod:`qorona.render.los` (a hand-rolled PNG writer over
stdlib zlib, no Pillow), and owns *which* products are written per run: the images, the post-write
annotation stamp, and the requested data sidecars (the ``.npz`` exports here, the WCS-registered
FITS via :mod:`qorona.io.fits`), keeping :mod:`qorona.render` self-contained.

The stamp is a corner text overlay (CR · UTC · sub-observer φ/θ · roll · FOV, or just the date/time
with ``annotate_content="date"``) drawn on the *saved* PNG so it sits at final resolution, following
the frame-labelling convention of eclipse-prediction renders; ``annotate=False`` is a one-flag
bypass. It needs a font renderer (**Pillow**), in the
default install: when Pillow is absent the overlay is skipped with a note and the run continues.
Non-ASCII glyphs degrade to ASCII surrogates only on the bitmap-font fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from qorona.config import BrightnessConfig, EuvConfig, OutputConfig, QMapConfig
from qorona.console import print_warning
from qorona.render.los import RenderResult

if TYPE_CHECKING:
    import numpy as np

    from qorona.radiation.brightness import BrightnessResult
    from qorona.radiation.euv import EuvResult
    from qorona.radiation.punch_l3 import PunchL3Result
    from qorona.radiation.response import EuvChannel
    from qorona.render.fieldlines import FieldLineImage
    from qorona.render.shell import QMap

#: Margin between the stamp text and the image edge, in pixels.
_MARGIN_PX = 12
#: Outline radius for the text's dark halo, in pixels (see ``_draw_outlined``).
_OUTLINE_PX = 1


def write_outputs(
    result: RenderResult, output_cfg: OutputConfig, provenance: dict[str, Any]
) -> list[Path]:
    """Write the render's PNG(s), stamp them, and export any FITS sidecar, returning the paths.

    The depth-coloured image is always written; the grayscale measurement image is written when
    ``output_cfg.save_grayscale``. When ``output_cfg.annotate`` and Pillow is installed, each
    written PNG is stamped with the provenance corner overlay; without Pillow the stamp is skipped
    with a friendly note and the images are still written. When ``"fits"`` is in
    ``output_cfg.export_formats``, the WCS-registered quantitative FITS
    (:func:`qorona.io.fits.write_render_fits`) is written beside the image.

    Returns
    -------
    list of Path
        The files written, in write order (colour PNG first).
    """
    written: list[Path] = []
    result.save_png(output_cfg.path)
    written.append(output_cfg.path)
    if output_cfg.save_grayscale:
        grayscale_path = output_cfg.grayscale_path()
        result.save_grayscale_png(grayscale_path)
        written.append(grayscale_path)
    _apply_stamp(written, output_cfg, provenance)
    if "fits" in output_cfg.export_formats:
        from qorona.io.fits import write_render_fits

        fits_path = output_cfg.export_path("fits")
        write_render_fits(result, fits_path, provenance)
        written.append(fits_path)
    return written


def write_fieldlines(
    result: FieldLineImage, output_cfg: OutputConfig, provenance: dict[str, Any]
) -> list[Path]:
    """Write the field-line render's PNG and stamp it, returning the path(s) written.

    The field-line view is a single colour image (no grayscale companion); it shares the colour PNG
    writer and the provenance stamp with the Q⊥ render.

    Returns
    -------
    list of Path
        The image file written.
    """
    written: list[Path] = [output_cfg.path]
    result.save_png(output_cfg.path)
    _apply_stamp(written, output_cfg, provenance)
    return written


def write_brightness(
    result: BrightnessResult,
    brightness_cfg: BrightnessConfig,
    output_cfg: OutputConfig,
    provenance: dict[str, Any],
    *,
    punch: PunchL3Result | None = None,
) -> list[Path]:
    """Write the white-light / pB render's PNG and stamp it, returning the paths written.

    Selects the requested component (the polarized ``pB`` or the total white-light brightness) and
    finishes it through the display stages, in order: the vignette treatment (the ``newkirk`` /
    ``adaptive`` radial detrends, the ``wow`` wavelet whitening of the raw frame, skipped for
    ``none``) and the optional MGN local fine-structure enhancement. Writes a
    percentile-stretched 8-bit grayscale PNG and applies the shared provenance stamp. The
    ``wow`` output is signed, so its stretch is anchored on a corona-pixel mask instead of the
    positive pixels. MGN is calibrated for the pB frame; on the total frame it still renders but
    is less physically meaningful, so a note is printed. When ``"fits"`` is in
    ``output_cfg.export_formats``, the WCS-registered white-light FITS
    (:func:`qorona.io.fits.write_brightness_fits`) is written beside the image, carrying the
    stretched display frame plus the raw ``PB``/``TOTAL`` extensions.

    With ``punch`` (the PUNCH Level 3 treatment), the image is the treated component under the
    PUNCH quicklook stretch, and the FITS carries the treated and the untreated frames.

    Raises
    ------
    ImportError
        If ``mgn`` or the ``wow`` channel is requested without its optional packages installed
        (``sunkit-image``, plus ``watroo`` for ``wow``); the message names the missing packages
        and the alternatives.

    Returns
    -------
    list of Path
        The files written, image first.
    """
    from qorona.radiation.display import (
        adaptive_vignette,
        mgn_enhance,
        newkirk_vignette,
        save_pb_png,
        wow_enhance,
    )

    # The occulter edge the render integrated behind (the resolved r_occult, recorded on the
    # result) is the reference for the display detrends, so the vignettes and the wow mask read it
    # from the result rather than the config, whose sentinel may still be unresolved.
    if punch is not None:
        result = punch.treated
    reference_radius = result.r_occult
    image = result.total if brightness_cfg.component == "total" else result.polarized
    valid = None
    if brightness_cfg.vignette == "newkirk":
        image = newkirk_vignette(
            image,
            result.impact,
            component=cast(Any, brightness_cfg.component),
            r_inner=result.r_inner,
            r_outer=result.r_outer,
            reference_radius=reference_radius,
            u=brightness_cfg.u,
            crossover=brightness_cfg.crossover,
            observer_distance=result.observer_distance,
            observer_limit=result.observer_limit,
        )
    elif brightness_cfg.vignette == "adaptive":
        image = adaptive_vignette(
            image,
            result.impact,
            component=cast(Any, brightness_cfg.component),
            r_inner=result.r_inner,
            r_outer=result.r_outer,
            reference_radius=reference_radius,
            u=brightness_cfg.u,
            crossover=brightness_cfg.crossover,
        )
    elif brightness_cfg.vignette == "wow":
        # The same corona definition the vignettes encode by zeroing: whitened structure is
        # signed, so the non-corona pixels are masked at PNG time instead.
        valid = (image > 0.0) & (result.impact >= reference_radius)
        image = wow_enhance(image)
    if brightness_cfg.mgn:
        if brightness_cfg.component == "total":
            print_warning(
                "MGN is calibrated for the polarized (pB) frame; on the total frame it still "
                "renders but is less physically meaningful"
            )
        image = mgn_enhance(image)
    save_pb_png(
        image,
        output_cfg.path,
        scaling=cast(Any, brightness_cfg.scaling),
        percentiles=brightness_cfg.percentiles,
        valid=valid,
    )
    written = [output_cfg.path]
    _apply_stamp(written, output_cfg, provenance)
    if "fits" in output_cfg.export_formats:
        from qorona.io.fits import write_brightness_fits
        from qorona.radiation.display import stretch_frame

        display = stretch_frame(
            image,
            scaling=cast(Any, brightness_cfg.scaling),
            percentiles=brightness_cfg.percentiles,
            valid=valid,
        )
        fits_path = output_cfg.export_path("fits")
        write_brightness_fits(result, display, fits_path, provenance, punch=punch)
        written.append(fits_path)
    return written


def write_qmap(
    result: QMap, qmap_cfg: QMapConfig, output_cfg: OutputConfig, provenance: dict[str, Any]
) -> list[Path]:
    """Write the Q-map figure (and optional ``.npz``) and return the paths written, image first.

    The headline product is the publication figure (lon/lat axes, diverging colour bar, title),
    drawn with matplotlib. Without matplotlib the bare colour raster is written instead, with the
    provenance corner stamp (the figure carries its provenance in the title). When
    ``"npz"`` is in ``qmap_cfg.export_formats`` the raw shell arrays ride alongside as a
    dependency-free ``.npz``.
    """
    import json

    written: list[Path] = [output_cfg.path]
    try:
        result.save_figure(
            output_cfg.path,
            slog_max=qmap_cfg.slog_max,
            title=_qmap_title(result.radius, provenance),
        )
    except ImportError:
        print_warning(
            "matplotlib not found; writing the bare raster map instead of the axed figure "
            "(install matplotlib, the figure backend, to enable it)"
        )
        result.save_png(output_cfg.path, slog_max=qmap_cfg.slog_max)
        _apply_stamp(written, output_cfg, provenance)
    if "npz" in qmap_cfg.export_formats:
        npz_path = output_cfg.export_path("npz")
        result.save_npz(npz_path, meta=json.dumps(provenance, default=str))
        written.append(npz_path)
    return written


def _qmap_title(radius: float, provenance: dict[str, Any]) -> str:
    """Compose the figure title from radius and (if recorded) the source volume's Carrington
    rotation."""
    source = provenance.get("source_volume", {})
    inp = source.get("input", {}) if isinstance(source, dict) else {}
    title = r"signed $\log_{10} Q_\perp$ at r = " + f"{radius:g} " + r"R$_\odot$"
    if isinstance(inp, dict) and inp.get("cr") is not None:
        title += f"  ·  CR {inp['cr']}"
    return title


def export_brightness(
    result: BrightnessResult,
    output_cfg: OutputConfig,
    provenance: dict[str, Any],
    *,
    punch: PunchL3Result | None = None,
) -> list[Path]:
    """Write the requested raw data sidecars for a brightness render, returning the paths.

    The export carries the *raw* frames (both the polarized ``pB`` and the total white-light
    brightness) with the pixel coordinate axes, the impact-parameter grid, and the integrated radial
    shell, independent of the displayed frame/stages, so a downstream tool (NRGF, WOW, or a custom
    detrend) processes the unstyled data. The frames are in the result's unit, mean solar
    brightness or relative, which the provenance in ``meta`` records. For the orthographic camera,
    the axes are ``x_rsun`` / ``y_rsun`` in R☉. For the perspective camera, they are ``x_deg`` /
    ``y_deg``, projection-plane offsets from Sun centre in degrees. The dependency-free ``.npz`` is
    this function's format (the FITS sidecar belongs to the product writers); the run provenance
    travels as a JSON string in ``meta``. With ``punch`` (the PUNCH Level 3 treatment), ``total``
    and ``pb`` are the treated frames, with the untreated ``total_full`` and ``pb_full``.

    Returns
    -------
    list of Path
        The sidecar files this function wrote; formats owned by other writers are not listed.
    """
    import json

    import numpy as np

    written: list[Path] = []
    suffix = "rsun" if result.axis_unit == "R_sun" else "deg"
    arrays: dict[str, Any] = {
        "total": result.total.astype(np.float32),
        "pb": result.polarized.astype(np.float32),
        f"x_{suffix}": result.x.astype(np.float32),
        f"y_{suffix}": result.y.astype(np.float32),
        "impact": result.impact.astype(np.float32),
        "shell": np.array([result.r_inner, result.r_outer], dtype=np.float32),
        "meta": np.array(json.dumps(provenance, default=str)),
    }
    if punch is not None:
        arrays["total"] = punch.treated.total.astype(np.float32)
        arrays["pb"] = punch.treated.polarized.astype(np.float32)
        arrays["total_full"] = punch.full.total.astype(np.float32)
        arrays["pb_full"] = punch.full.polarized.astype(np.float32)
    for fmt in output_cfg.export_formats:
        if fmt == "npz":
            path = output_cfg.export_path(fmt)
            np.savez_compressed(path, **arrays)
            written.append(path)
    return written


#: Percentiles of the positive pixels that set the EUV display range with ``auto_range``.
_EUV_AUTO_PERCENTILES = (0.5, 99.9)
#: The fraction of the disk pixels below the display range above which ``write_euv`` tells the
#: user that the image is dark.
_EUV_DARK_FRACTION = 0.99


def euv_paths(
    output_cfg: OutputConfig, channels: tuple[str, ...], suffix: str = ""
) -> dict[str, Path]:
    """The file of each EUV channel: ``OUT`` itself for one channel, ``OUT_<channel>`` for several.

    ``suffix`` (``".fits"``, ``".npz"``) replaces the image suffix for a data sidecar.
    """
    path = output_cfg.path
    ext = suffix or path.suffix
    if len(channels) == 1:
        return {channels[0]: path.with_suffix(ext)}
    return {name: path.with_name(f"{path.stem}_{name}{ext}") for name in channels}


def euv_display_range(
    frame: np.ndarray, channel: EuvChannel, euv_cfg: EuvConfig
) -> tuple[float, float]:
    """The ``(low, high)`` of an EUV frame's log display, in the channel unit.

    The config's range wins; with ``auto_range`` the range is the 0.5 and 99.9 percentiles of
    the positive pixels; otherwise it is the channel's default.
    """
    import numpy as np

    if euv_cfg.display_range is not None:
        return euv_cfg.display_range
    if euv_cfg.auto_range:
        positive = frame[np.isfinite(frame) & (frame > 0.0)]
        if positive.size:
            low, high = np.percentile(positive, _EUV_AUTO_PERCENTILES)
            if high > low > 0.0:
                return float(low), float(high)
    return channel.display_range


def write_euv(
    result: EuvResult, euv_cfg: EuvConfig, output_cfg: OutputConfig, provenance: dict[str, Any]
) -> list[Path]:
    """Write one PNG per EUV channel, stamp it, and add its colour bar; return the paths written.

    Each frame is shown as ``log₁₀`` of the channel unit on its display range
    (:func:`euv_display_range`) in the channel's sunpy colour map, with a colour bar beside it
    unless ``euv_cfg.colorbar`` is off. When ``"fits"`` is in ``output_cfg.export_formats``, a
    WCS-registered FITS per channel holds the physical frame
    (:func:`qorona.io.fits.write_euv_fits`).

    Returns
    -------
    list of Path
        The files written, images first.
    """
    from qorona.render.image import write_png

    channels = tuple(result.frames)
    images = euv_paths(output_cfg, channels)
    ranges: dict[str, tuple[float, float]] = {}
    for name, frame in result.frames.items():
        channel = result.tables[name].channel
        ranges[name] = euv_display_range(frame, channel, euv_cfg)
        write_png(images[name], _euv_rgb(frame, channel.colormap, *ranges[name]))
        low, high = ranges[name]
        if _below_display_range(frame, result.impact, low):
            print_warning(
                f"{name}: the image is below its display range ({low:g} to {high:g} "
                f"{result.unit(name)}); try --range auto"
            )
    written = [images[name] for name in channels]
    _apply_stamp(written, output_cfg, provenance)
    if euv_cfg.colorbar:
        try:
            for name in channels:
                channel = result.tables[name].channel
                _append_colorbar(images[name], channel.colormap, *ranges[name], result.unit(name))
        except ImportError:
            print_warning(
                "pillow not found, skipping the colour bar (install pillow, part of the default "
                "install, to enable it)"
            )
    if "fits" in output_cfg.export_formats:
        from qorona.io.fits import write_euv_fits

        for name, path in euv_paths(output_cfg, channels, ".fits").items():
            write_euv_fits(result, name, path, provenance, display_range=ranges[name])
            written.append(path)
    return written


def export_euv(
    result: EuvResult, output_cfg: OutputConfig, provenance: dict[str, Any]
) -> list[Path]:
    """Write the requested ``.npz`` sidecars of an EUV render, one per channel.

    Each holds the physical ``frame`` (channel unit per second and native pixel), the pixel axes
    in degrees (``x_deg``, ``y_deg``), the impact parameter (R☉) and the run provenance as JSON in
    ``meta``.
    """
    import json

    import numpy as np

    if "npz" not in output_cfg.export_formats:
        return []
    written = []
    for name, path in euv_paths(output_cfg, tuple(result.frames), ".npz").items():
        np.savez_compressed(
            path,
            frame=result.frames[name].astype(np.float32),
            x_deg=result.x.astype(np.float32),
            y_deg=result.y.astype(np.float32),
            impact=result.impact.astype(np.float32),
            unit=np.array(result.unit(name)),
            meta=np.array(json.dumps(provenance, default=str)),
        )
        written.append(path)
    return written


def _euv_colormap(name: str) -> Any:
    """The sunpy colour map ``name``, or matplotlib's gray when it is unavailable."""
    import matplotlib

    try:
        import sunpy.visualization.colormaps  # noqa: F401  (registers the instrument maps)

        return matplotlib.colormaps[name]
    except (ImportError, KeyError):
        print_warning(f"colour map {name!r} not found; using gray")
        return matplotlib.colormaps["gray"]


def _euv_rgb(frame: np.ndarray, colormap: str, low: float, high: float) -> np.ndarray:
    """``(H, W, 3)`` ``uint8`` colours of ``log₁₀ frame`` on ``[log₁₀ low, log₁₀ high]``."""
    import numpy as np

    with np.errstate(divide="ignore", invalid="ignore"):
        level = (np.log10(frame) - np.log10(low)) / (np.log10(high) - np.log10(low))
    level = np.clip(np.nan_to_num(level, nan=0.0, neginf=0.0), 0.0, 1.0)
    rgba = _euv_colormap(colormap)(level)
    return np.ascontiguousarray((rgba[..., :3] * 255.0 + 0.5).astype(np.uint8))


def _decade_labels(low: float, high: float) -> list[float]:
    """The powers of ten in ``[low, high]``, or the two ends when the range holds fewer than two."""
    import math

    first, last = math.ceil(math.log10(low) - 1e-9), math.floor(math.log10(high) + 1e-9)
    decades = [10.0**k for k in range(first, last + 1)]
    return decades if len(decades) >= 2 else [low, high]


def _below_display_range(frame: np.ndarray, impact: np.ndarray, low: float) -> bool:
    """Whether nearly all of the frame is below ``low``: the disk pixels, or all pixels off it."""
    import numpy as np

    disk = impact < 1.0
    region = frame[disk] if np.any(disk) else frame.ravel()
    return bool(np.mean(region < low) > _EUV_DARK_FRACTION)


def _tick_text(value: float) -> str:
    """A compact tick label: ``0.1``, ``10``, ``1000``, ``1e+04``."""
    return f"{value:g}"


def _append_colorbar(path: Path, colormap: str, low: float, high: float, unit: str) -> None:
    """Append a colour-bar strip on the right of the PNG at ``path``, labelled in ``unit``.

    The bar runs from ``low`` (bottom) to ``high`` (top) on the same ``log₁₀`` scale as the
    image, with a tick at every power of ten. Raises :class:`ImportError` without Pillow.
    """
    import math

    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    with Image.open(path) as raw:
        image = raw.convert("RGB")
    width, height = image.size
    font_size = max(12, height // 50)
    font = _load_font(font_size)
    unicode_font = isinstance(font, ImageFont.FreeTypeFont)
    label = unit.replace("s-1", "s⁻¹").replace("pix-1", "pix⁻¹") if unicode_font else unit
    strip_width = max(6 * font_size, width // 7)
    bar_width = max(10, font_size)
    top, bottom = int(0.1 * height), int(0.92 * height)
    canvas = Image.new("RGB", (width + strip_width, height), (0, 0, 0))
    canvas.paste(image, (0, 0))
    levels = np.linspace(1.0, 0.0, bottom - top)[:, None].repeat(bar_width, axis=1)
    bar = (_euv_colormap(colormap)(levels)[..., :3] * 255.0 + 0.5).astype(np.uint8)
    x0 = width + font_size
    canvas.paste(Image.fromarray(bar), (x0, top))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((x0 - 1, top - 1, x0 + bar_width, bottom), outline=(200, 200, 200))
    span = math.log10(high) - math.log10(low)
    for value in _decade_labels(low, high):
        y = bottom - (math.log10(value) - math.log10(low)) / span * (bottom - top)
        y = round(min(max(y, top), bottom))
        draw.line((x0 + bar_width, y, x0 + bar_width + font_size // 2, y), fill=(220, 220, 220))
        text = _tick_text(value)
        box = draw.textbbox((0, 0), text, font=font)
        draw.text(
            (x0 + bar_width + font_size * 3 // 4, y - (box[3] - box[1]) // 2 - box[1]),
            text,
            font=font,
            fill=(230, 230, 230),
        )
    box = draw.textbbox((0, 0), label, font=font)
    draw.text(
        (width + max(0, (strip_width - (box[2] - box[0])) // 2), max(2, top - 2 * font_size)),
        label,
        font=font,
        fill=(230, 230, 230),
    )
    canvas.save(path)


def _apply_stamp(written: list[Path], output_cfg: OutputConfig, provenance: dict[str, Any]) -> None:
    """Burn the provenance corner stamp onto each written PNG, when annotation is on and Pillow is
    present.

    A no-op when ``output_cfg.annotate`` is off; when Pillow is missing the stamp is skipped with a
    friendly note and the already-written images are left as-is. Shared by all this module's
    writers so every product carries the identical stamp.
    """
    if not output_cfg.annotate:
        return
    lines = _stamp_lines(provenance, output_cfg.annotate_content)
    if not lines and output_cfg.annotate_content == "date":
        print_warning(
            "--annotate-content date needs a --timestamp to know the date; images written unstamped"
        )
        return
    try:
        for path in written:
            _annotate_png(path, lines, position=output_cfg.annotate_position)
    except ImportError:
        print_warning(
            "pillow not found, skipping the on-image provenance stamp; the run summary still "
            "records the provenance (install pillow, part of the default install, to enable it)"
        )


def _stamp_lines(provenance: dict[str, Any], content: str = "full") -> list[str]:
    """Assemble the stamp's text lines from the run provenance.

    With ``content="full"``, the CR and date lines appear only when a ``--timestamp`` was supplied
    (the mesh has no date); the camera angle / roll / FOV lines always stamp. The FOV is in R☉ for
    the orthographic camera, and in degrees with the projection code for the perspective camera
    (``FOV 92.16° ARC``). ``R_sun`` is spelled out so it renders without a special glyph. With
    ``content="date"``, the stamp is the single date/time line, or nothing without a
    ``--timestamp``.
    """
    inp = provenance.get("input", {}) if isinstance(provenance.get("input"), dict) else {}
    if content == "date":
        return [_date_line(inp["timestamp"])] if inp.get("timestamp") else []
    lines: list[str] = []
    camera = provenance.get("camera", {}) if isinstance(provenance.get("camera"), dict) else {}
    if inp.get("cr") is not None:
        lines.append(f"CR {inp['cr']}")
    if inp.get("timestamp"):
        lines.append(f"{inp['timestamp']} UTC")
    if camera:
        lines.append(
            f"φ={float(camera['longitude']):+.2f}° θ={float(camera['latitude']):+.2f}° "
            f"roll={float(camera['roll']):+.2f}°"
        )
        if camera.get("projection") == "perspective":
            code = str(camera["projection_code"]).upper()
            lines.append(f"FOV {float(camera['fov_deg']):g}° {code}")
        else:
            lines.append(f"FOV {float(camera['fov']):g} R_sun")
    qmap = provenance.get("qmap", {}) if isinstance(provenance.get("qmap"), dict) else {}
    if qmap:
        lines.append(f"Q-map  r = {float(qmap['radius']):g} R_sun")
    return lines


def _date_line(timestamp: str) -> str:
    """Format the ``--timestamp`` as the date-only stamp line (``2012-06-14 13:30 UTC``).

    Seconds appear only when non-zero; a date-only timestamp stamps without a time of day. An
    unparseable string (the CLI validates, but provenance may come from elsewhere) stamps verbatim.
    """
    from datetime import datetime

    try:
        moment = datetime.fromisoformat(timestamp)
    except ValueError:
        return f"{timestamp} UTC"
    if "T" not in timestamp and ":" not in timestamp:
        return moment.strftime("%Y-%m-%d")
    clock = "%H:%M:%S" if moment.second or moment.microsecond else "%H:%M"
    return moment.strftime(f"%Y-%m-%d {clock} UTC")


def _ascii_safe(text: str) -> str:
    """Replace the non-ASCII stamp glyphs with ASCII surrogates (for the bitmap-font fallback)."""
    return text.replace("φ", "phi=").replace("θ", "theta=").replace("°", "deg")


def _annotate_png(path: Path, lines: list[str], *, position: str = "bottom-left") -> None:
    """Overlay ``lines`` onto the PNG at ``path``, in one corner, and save back in place.

    Drawn in white DejaVuSans (preferred for its φ/θ/° coverage; system fonts then the PIL bitmap
    default as fallbacks) with a thin dark outline. Raises :class:`ImportError` if Pillow is
    unavailable, which :func:`_apply_stamp` turns into a friendly skip.
    """
    from PIL import Image, ImageDraw, ImageFont

    if not lines:
        return
    # Read fully and close the source handle before writing back to the same path.
    with Image.open(path) as raw:
        image = raw.convert("RGB")
    width, height = image.size
    font_size = max(12, height // 50)
    font = _load_font(font_size)
    if not isinstance(font, ImageFont.FreeTypeFont):
        lines = [_ascii_safe(line) for line in lines]
    draw = ImageDraw.Draw(image)

    bboxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    line_heights = [int(bbox[3]) for bbox in bboxes]
    line_widths = [int(bbox[2]) for bbox in bboxes]
    spacing = max(2, font_size // 6)
    block_height = sum(line_heights) + spacing * (len(lines) - 1)
    block_width = max(line_widths)
    x0, y0 = _corner(position, width, height, block_width, block_height)

    y = y0
    for line, line_height in zip(lines, line_heights, strict=True):
        _draw_outlined(draw, (x0, y), line, font)
        y += line_height + spacing
    image.save(path)


def _corner(
    position: str, width: int, height: int, block_width: int, block_height: int
) -> tuple[int, int]:
    """Return the top-left pixel of the text block for the requested corner ``position``."""
    left = _MARGIN_PX
    right = width - _MARGIN_PX - block_width
    top = _MARGIN_PX
    bottom = height - _MARGIN_PX - block_height
    corners = {
        "bottom-left": (left, bottom),
        "bottom-right": (right, bottom),
        "top-left": (left, top),
        "top-right": (right, top),
    }
    return corners[position]


def _draw_outlined(draw: Any, xy: tuple[int, int], text: str, font: Any) -> None:
    """Draw ``text`` in white with a thin black outline for legibility on any background."""
    x, y = xy
    for dx in range(-_OUTLINE_PX, _OUTLINE_PX + 1):
        for dy in range(-_OUTLINE_PX, _OUTLINE_PX + 1):
            if dx or dy:
                draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(255, 255, 255))


def _load_font(size: int) -> Any:
    """Return the best available font at ``size``: DejaVuSans (for its φ/θ/° glyphs), then a system
    TrueType, then the PIL bitmap default."""
    from PIL import ImageFont

    candidates: list[str] = []
    try:
        import matplotlib  # matplotlib bundles DejaVuSans.

        candidates.append(
            str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")
        )
    except Exception:
        pass
    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except Exception:
                continue
    return ImageFont.load_default()
