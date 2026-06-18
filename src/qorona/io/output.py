"""Image output and the optional on-image provenance stamp.

Writes the rendered :class:`~qorona.render.los.RenderResult` to PNG, reusing the dependency-free
``save_png`` / ``save_grayscale_png`` in :mod:`qorona.render.los` (a hand-rolled PNG writer over
stdlib zlib, no Pillow), and owns *which* images are written and the post-write annotation stamp,
keeping :mod:`qorona.render` self-contained.

The stamp is a corner text overlay (CR · UTC · sub-observer φ/θ · roll · FOV) drawn on the *saved*
PNG so it sits at final resolution, following the frame-labelling convention of eclipse-prediction
renders; ``annotate=False`` is a one-flag bypass. It needs a font renderer (**Pillow**), in the
default install: when Pillow is absent (a deliberately leaned-out install) the overlay is skipped
with a friendly note and the run still completes (the metrics still print). Non-ASCII glyphs
degrade to ASCII surrogates only on the bitmap-font fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from qorona.config import BrightnessConfig, OutputConfig
from qorona.console import print_warning
from qorona.render.los import RenderResult

if TYPE_CHECKING:
    from qorona.radiation.brightness import BrightnessResult
    from qorona.render.fieldlines import FieldLineImage

#: Margin between the stamp text and the image edge, in pixels.
_MARGIN_PX = 12
#: Outline radius for the text's dark halo, in pixels (see ``_draw_outlined``).
_OUTLINE_PX = 1


def write_outputs(
    result: RenderResult, output_cfg: OutputConfig, provenance: dict[str, Any]
) -> list[Path]:
    """Write the render's PNG(s) and stamp them, returning the paths written.

    The depth-coloured image is always written; the grayscale measurement image is written when
    ``output_cfg.save_grayscale``. When ``output_cfg.annotate`` and Pillow is installed, each
    written PNG is stamped with the provenance corner overlay; without Pillow the stamp is skipped
    with a friendly note and the images are still written.

    Returns
    -------
    list of Path
        The image files written, in write order (colour first).
    """
    written: list[Path] = []
    result.save_png(output_cfg.path)
    written.append(output_cfg.path)
    if output_cfg.save_grayscale:
        grayscale_path = output_cfg.grayscale_path()
        result.save_grayscale_png(grayscale_path)
        written.append(grayscale_path)
    _apply_stamp(written, output_cfg, provenance)
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
) -> list[Path]:
    """Write the white-light / pB render's PNG and stamp it, returning the path written.

    Selects the requested frame (the polarized ``pB`` or the total white-light brightness) and, for
    the polarized frame, the display treatment (raw, the Newkirk radial vignette, or the MGN
    fine-structure enhancement), writes a percentile-stretched 8-bit grayscale PNG, and applies the
    shared provenance stamp.

    Raises
    ------
    ImportError
        If the ``mgn`` treatment is requested without ``sunkit-image`` installed (the only treatment
        that needs it); the message names the missing package and the alternatives.

    Returns
    -------
    list of Path
        The image file written.
    """
    from qorona.radiation.display import mgn_enhance, newkirk_vignette, save_pb_png

    if brightness_cfg.frame == "total":
        frame = result.total
    elif brightness_cfg.treatment == "newkirk":
        frame = newkirk_vignette(result.polarized, result.impact)
    elif brightness_cfg.treatment == "mgn":
        frame = mgn_enhance(result.polarized)
    else:
        frame = result.polarized
    save_pb_png(
        frame,
        output_cfg.path,
        scaling=cast(Any, brightness_cfg.scaling),
        percentiles=brightness_cfg.percentiles,
    )
    written = [output_cfg.path]
    _apply_stamp(written, output_cfg, provenance)
    return written


def _apply_stamp(
    written: list[Path], output_cfg: OutputConfig, provenance: dict[str, Any]
) -> None:
    """Burn the provenance corner stamp onto each written PNG, when annotation is on and Pillow is
    present.

    A no-op when ``output_cfg.annotate`` is off; when Pillow is missing the stamp is skipped with a
    friendly note and the already-written images are left as-is. Shared by all this module's
    writers so every product carries the identical stamp.
    """
    if not output_cfg.annotate:
        return
    lines = _stamp_lines(provenance)
    try:
        for path in written:
            _annotate_png(path, lines, position=output_cfg.annotate_position)
    except ImportError:
        print_warning(
            "pillow not found, skipping the on-image provenance stamp; the run summary still "
            "records the provenance (install pillow, part of the default install, to enable it)"
        )


def _stamp_lines(provenance: dict[str, Any]) -> list[str]:
    """Assemble the stamp's text lines from the run provenance.

    The CR and date lines appear only when a ``--timestamp`` was supplied (the mesh has no date);
    the camera angle / roll / FOV lines always stamp. ``R_sun`` is spelled out so it renders without
    a special glyph.
    """
    lines: list[str] = []
    inp = provenance.get("input", {}) if isinstance(provenance.get("input"), dict) else {}
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
        lines.append(f"FOV {float(camera['fov']):g} R_sun")
    return lines


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
        import matplotlib  # matplotlib bundles DejaVuSans; an optional, soft preference only.

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
