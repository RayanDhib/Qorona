"""Plane-of-sky magnetic field-line render with limb-ring seeding and a B_r magnetogram disk.

The secondary, morphological product of the pipeline, not a line-of-sight integral. Seeds are
traced both ways to the boundaries (the field-line tracer) and the resulting 3-D polylines are
projected through the same orthographic plane-of-sky camera as the Q⊥ render and drawn over the
photosphere disk.

Two seeding strategies:

- ``limb`` (the default eclipse-like look): the open fan is seeded on a ring around the **limb**,
  the plane-of-sky great circle where the line of sight grazes the photosphere, so it radiates
  cleanly from the disk edge; the front (observer-facing) hemisphere contributes only short closed
  loops close to the Sun; the back interior is left unseeded. Produces a clean field-line fan.
- ``uniform``: a golden-angle Fibonacci spiral over the whole inner sphere, every line, the honest
  analytical view.

Two colour modes: ``rainbow`` (a distinct matte hue per line, golden-angle walk, for legibility)
and ``polarity`` (open lines tinted by inner-foot ``B·r̂`` sign, closed loops
neutral, physical). The disk is either a ``B_r`` magnetogram built from the data (sampled on the
near hemisphere) or a flat occulter.

The rasteriser is dependency-free: an anti-aliased soft-disk splat onto an ``(H, W, 3)`` float
canvas, composited far-layer → disk → near-layer so the disk occludes the field behind it without a
z-buffer. Within a layer the coverage-weighted contributions are accumulated (order-independent),
which for a line bundle reads better than a strict painter's overwrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import units as u

from qorona.accel import apply_workers
from qorona.console import status
from qorona.field.base import Field
from qorona.geometry.camera import OrthographicCamera
from qorona.trace import (
    DEFAULT_TURN_GUARD,
    Endpoint,
    FieldLines,
    TurnGuard,
    fibonacci_seeds,
    trace_field_lines,
)

#: Line / disk colours as linear RGB in [0, 1].
_BACKGROUND = (0.0, 0.0, 0.0)
_FLAT_DISK_COLOUR = (25 / 255, 25 / 255, 25 / 255)  # the plain occulter (8-bit value 25)
_OPEN_POSITIVE = (0.902, 0.314, 0.235)  # polarity mode: outward (B·r̂ ≥ 0), warm
_OPEN_NEGATIVE = (0.275, 0.471, 0.902)  # polarity mode: inward  (B·r̂ < 0), cool
_CLOSED = (0.5, 0.5, 0.5)  # polarity mode: closed loop, neutral grey
#: Magnetogram disk: quiet Sun and the diverging outward/inward field colours.
_PHOTOSPHERE = (0.92, 0.92, 0.92)
_FIELD_OUTWARD = (0.75, 0.12, 0.10)
_FIELD_INWARD = (0.12, 0.28, 0.75)

#: Rainbow colour mode: a golden-angle hue walk at matte saturation/value (distinct, soft hues).
_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949
_RAINBOW_SATURATION = 0.45
_RAINBOW_VALUE = 0.80

#: Percentile of |B_r| that saturates the magnetogram colour scale (robust to active-region spikes).
_MAGNETOGRAM_PERCENTILE = 97.0
#: Fractional radial nudge keeping seeds and sampled footpoints strictly inside the domain shell:
#: a tracer/interpolant precondition; matches ``squashing.volume._BOUNDARY_MARGIN``.
_DOMAIN_MARGIN = 1.0e-9
#: Segments are rasterised in batches of this many to bound peak memory of the splat expansion.
_SEGMENT_CHUNK = 200_000


@dataclass(frozen=True)
class FieldLineImage:
    """The rendered field-line image and the line-classification tallies for the run summary.

    Attributes
    ----------
    image
        ``(H, W, 3)`` ``uint8`` RGB: the field lines over the disk, ready for the PNG writer and
        the shared provenance stamp.
    n_open, n_closed, n_incomplete
        Counts of open, closed, and incomplete (dropped) traced lines.
    lines
        The full traced bundle (with polylines).
    keep
        ``(n,)`` bool: the drawn lines; the SunJSON export writes exactly this sub-bundle.
    colours
        ``(keep.sum(), 3)`` per-drawn-line colour, linear RGB in [0, 1], in draw order.
    """

    image: np.ndarray
    n_open: int
    n_closed: int
    n_incomplete: int
    lines: FieldLines
    keep: np.ndarray
    colours: np.ndarray

    def summary(self) -> str:
        """Return a one-line classification breakdown for end-of-run reporting."""
        return (
            f"{self.n_open} open · {self.n_closed} closed · "
            f"{self.n_incomplete} incomplete (dropped)"
        )

    def save_png(self, path: str | Path) -> None:
        """Write the image to ``path`` as an 8-bit PNG (the dependency-free writer shared with the
        Q⊥ render)."""
        from qorona.render.image import write_png

        write_png(Path(path), np.ascontiguousarray(self.image))


def render_field_lines(
    field: Field,
    camera: OrthographicCamera,
    *,
    seeding: str = "limb",
    n_seeds: int = 1500,
    limb_seeds: int = 375,
    front_loop_length: float = 1.2,
    colour: str = "polarity",
    magnetogram: bool = True,
    show: str = "all",
    line_width: float = 1.5,
    depth_fade: float = 0.4,
    rtol: float = 1e-4,
    cfl: float = 0.5,
    max_steps: int = 10_000,
    turn_guard: TurnGuard = DEFAULT_TURN_GUARD,
    workers: int | None = None,
    show_progress: bool = True,
) -> FieldLineImage:
    """Render the magnetic field lines of ``field`` from ``camera``'s viewpoint.

    Seeds (``limb`` or ``uniform``, see the module docstring), traces both ways to the boundaries,
    and draws the surviving lines in projection over the photosphere disk. The shell radii are read
    from ``field.domain``.

    Parameters
    ----------
    field
        The magnetic field to trace (a real :class:`~qorona.field.sampled.SampledField` or an
        analytic field).
    camera
        The orthographic plane-of-sky camera (shared with the Q⊥ render).
    seeding
        ``"limb"`` (front short loops + a limb ring; the eclipse-like look) or ``"uniform"``
        (a full-sphere Fibonacci spiral; every line).
    n_seeds
        Fibonacci seed budget over the sphere: the whole set in ``uniform``; the near-hemisphere
        source of the short front loops in ``limb``.
    limb_seeds
        Seeds around the limb great circle, the open fan (``limb`` seeding only).
    front_loop_length
        Total arc length (R☉) below which a near-side closed loop counts as a short front loop
        (``limb`` seeding only).
    colour
        ``"rainbow"`` (a distinct matte hue per line) or ``"polarity"`` (open by inner-foot ``B·r̂``
        sign, closed neutral).
    magnetogram
        Render the disk as a ``B_r`` magnetogram from the data; otherwise a flat occulter.
    show
        Which lines to draw: ``"all"``, ``"open"``, or ``"closed"``.
    line_width
        Drawn line width in pixels.
    depth_fade
        Dim far-side lines by up to this fraction (``0`` disables the depth cue).
    rtol, cfl, max_steps
        Tracer knobs (the DOPRI5 engine): ``rtol`` accuracy, ``cfl`` step ceiling, ``max_steps``
        step-count guard.
    turn_guard
        Sharp-turn guard: terminate (and drop) a line that makes a single sharp turn in the outer
        corona where ``|B|`` is weak: a staircase deflection at a null; see
        :class:`~qorona.trace.TurnGuard`. ``max_turn_angle = 0`` disables it.
    workers
        Numba thread count for the tracer (``None`` = all cores), applied via
        :func:`~qorona.accel.apply_workers`.
    show_progress
        Whether to show the tracer progress bar and the rasterisation spinner.

    Returns
    -------
    FieldLineImage
        The ``uint8`` image and the open / closed / incomplete tallies.
    """
    apply_workers(workers)
    inner_radius = field.domain.inner_radius
    outer_radius = field.domain.outer_radius

    seeds, n_front = _build_seeds(seeding, n_seeds, limb_seeds, inner_radius, camera)
    lines = trace_field_lines(
        field,
        seeds,
        rtol=rtol,
        cfl=cfl,
        max_steps=max_steps,
        turn_guard=turn_guard,
        store_path=True,
        show_progress=show_progress,
    )
    assert lines.paths is not None  # store_path=True always populates paths

    with status("Rasterising field lines", enabled=show_progress):
        keep = _keep_lines(lines, seeding, n_front, front_loop_length, show)
        kept_paths = [lines.paths[i] for i in np.nonzero(keep)[0]]
        kept_colours = _line_colours(colour, field, lines, keep, inner_radius, outer_radius)
        shape = camera.pixels
        disk = (
            _magnetogram_disk(shape, camera, field, inner_radius)
            if magnetogram
            else _flat_disk(shape, camera, inner_radius)
        )
        image = _draw(
            kept_paths,
            kept_colours,
            camera,
            disk_layer=disk,
            outer_radius=outer_radius,
            line_width=line_width,
            depth_fade=depth_fade,
        )

    return FieldLineImage(
        image=image,
        n_open=int(lines.is_open.sum()),
        n_closed=int(lines.is_closed.sum()),
        n_incomplete=int(lines.is_incomplete.sum()),
        lines=lines,
        keep=keep,
        colours=kept_colours,
    )


# --- Seeding -----------------------------------------------------------------------------------


def _build_seeds(
    seeding: str, n_seeds: int, limb_seeds: int, inner_radius: float, camera: OrthographicCamera
) -> tuple[np.ndarray, int]:
    """Return ``(seeds, n_front)`` for the chosen strategy.

    ``n_front`` is the count of leading near-hemisphere seeds (the short-loop source) in ``limb``
    seeding; for ``uniform`` it spans every seed and is unused downstream.
    """
    fibonacci = fibonacci_seeds(n_seeds, inner_radius)
    if seeding == "uniform":
        return fibonacci, fibonacci.shape[0]
    look = camera._basis()[0]
    front = fibonacci[fibonacci @ look > 0.0]  # observer-facing hemisphere
    ring = _limb_ring(limb_seeds, inner_radius, camera)
    return np.vstack([front, ring]), front.shape[0]


def _limb_ring(n_seeds: int, inner_radius: float, camera: OrthographicCamera) -> np.ndarray:
    """Return ``(n_seeds, 3)`` seeds evenly spaced on the limb, the plane-of-sky great circle.

    The limb is the circle on the inner sphere where the line of sight grazes the photosphere; it is
    spanned by the camera's image-right and image-up axes. Open lines rooted here radiate outward
    around the disk edge, giving the clean fan.
    """
    _, right, up = camera._basis()
    phi = np.linspace(0.0, 2.0 * np.pi, n_seeds, endpoint=False)
    ring = np.cos(phi)[:, None] * right + np.sin(phi)[:, None] * up
    return ring * (inner_radius * (1.0 + _DOMAIN_MARGIN))


def _keep_lines(
    lines: FieldLines, seeding: str, n_front: int, front_loop_length: float, show: str
) -> np.ndarray:
    """Return the ``(n,)`` bool mask of lines to draw.

    ``uniform`` keeps every complete line (filtered by ``show``). ``limb`` keeps only short closed
    loops from the near-hemisphere seeds and every complete limb-ring line, then applies ``show``.
    """
    base = _show_mask(lines, show)
    if seeding == "uniform":
        return base
    length = lines.lengths.sum(axis=1)
    keep = np.zeros(lines.seeds.shape[0], dtype=bool)
    keep[:n_front] = lines.is_closed[:n_front] & (length[:n_front] < front_loop_length)
    keep[n_front:] = ~lines.is_incomplete[n_front:]
    return keep & base


def _show_mask(lines: FieldLines, show: str) -> np.ndarray:
    """Return the ``(n,)`` topology filter: never incomplete, restricted by ``show``."""
    keep = np.zeros(lines.seeds.shape[0], dtype=bool)
    if show in ("all", "open"):
        keep |= lines.is_open
    if show in ("all", "closed"):
        keep |= lines.is_closed
    return keep


# --- Colour ------------------------------------------------------------------------------------


def _line_colours(
    colour: str,
    field: Field,
    lines: FieldLines,
    keep: np.ndarray,
    inner_radius: float,
    outer_radius: float,
) -> np.ndarray:
    """Return the ``(n_kept, 3)`` draw colour for the kept lines, in kept order."""
    if colour == "rainbow":
        return _rainbow_colours(int(keep.sum()))
    return polarity_colours(field, lines, inner_radius, outer_radius)[keep]


def _rainbow_colours(n_lines: int) -> np.ndarray:
    """A distinct matte hue per line (golden-angle walk); for legibility, encodes no physics."""
    hue = (np.arange(n_lines) * _GOLDEN_RATIO_CONJUGATE) % 1.0
    return _hsv_to_rgb(hue, _RAINBOW_SATURATION, _RAINBOW_VALUE)


def _hsv_to_rgb(hue: np.ndarray, saturation: float, value: float) -> np.ndarray:
    """Vectorised HSV→RGB for hue array ``hue`` (in [0,1)) and scalar ``saturation``/``value``."""
    h6 = (np.asarray(hue, dtype=np.float64) % 1.0) * 6.0
    sextant = np.floor(h6).astype(int) % 6
    frac = h6 - np.floor(h6)
    p = value * (1.0 - saturation)
    q = value * (1.0 - frac * saturation)
    t = value * (1.0 - (1.0 - frac) * saturation)
    v = np.full_like(frac, value)
    red = np.choose(sextant, [v, q, p, p, t, v])
    green = np.choose(sextant, [t, v, v, q, p, p])
    blue = np.choose(sextant, [p, p, t, v, v, q])
    return np.stack([red, green, blue], axis=-1)


def polarity_colours(
    field: Field, lines: FieldLines, inner_radius: float, outer_radius: float
) -> np.ndarray:
    """Return the ``(n, 3)`` per-line polarity colour: open by inner-foot ``B·r̂``, closed neutral.

    Open lines are coloured by the sign of ``B·r̂`` at their inner foot, the end whose
    :class:`~qorona.trace.Endpoint` is ``INNER`` (sampled in one batched call, the foot nudged
    inside the shell); closed loops take the neutral tone. Incomplete lines get an arbitrary
    colour; the render filters them out before drawing. Linear RGB in [0, 1]; public so an
    exporter can colour lines with the same palette.
    """
    colours = np.tile(np.array(_CLOSED), (lines.seeds.shape[0], 1))
    open_idx = np.nonzero(lines.is_open)[0]
    if open_idx.size:
        is_inner = lines.ends[open_idx] == Endpoint.INNER  # (k, 2)
        # Pick the inner foot where present (an open line has at most one). The rare both-outer open
        # line has no inner foot; it falls back to foot 0 and is coloured by that outer foot's B·r̂.
        inner_end = np.where(is_inner[:, 0], 0, np.where(is_inner[:, 1], 1, 0))
        feet = _clip_radius(lines.feet[open_idx, inner_end], inner_radius, outer_radius)
        b = field.sample(feet, gradient=False).b  # (k, 3)
        b_radial = np.sum(b * feet, axis=1)  # sign matches B·r̂ (feet points radially out)
        colours[open_idx] = np.where(
            b_radial[:, None] >= 0.0, np.array(_OPEN_POSITIVE), np.array(_OPEN_NEGATIVE)
        )
    return colours


def _clip_radius(points: np.ndarray, inner_radius: float, outer_radius: float) -> np.ndarray:
    """Clamp each point's radius into ``[inner(1+m), outer(1-m)]`` so it lies strictly inside the
    shell. Points already interior are returned unchanged; only boundary-touching feet are nudged a
    negligible amount along their own radius (the field through them is unmoved)."""
    radius = np.linalg.norm(points, axis=-1)
    lower = inner_radius * (1.0 + _DOMAIN_MARGIN)
    upper = outer_radius * (1.0 - _DOMAIN_MARGIN)
    scale = np.where(radius > 0.0, np.clip(radius, lower, upper) / radius, 1.0)
    return points * scale[:, None]


# --- Projection and rasterisation --------------------------------------------------------------


def _draw(
    paths: list[np.ndarray],
    colours: np.ndarray,
    camera: OrthographicCamera,
    *,
    disk_layer: tuple[np.ndarray, np.ndarray],
    outer_radius: float,
    line_width: float,
    depth_fade: float,
) -> np.ndarray:
    """Project the field-line ``paths`` and draw them over the ``disk_layer`` into a uint8 image.

    Each path's vertices are projected once; consecutive vertices form segments carrying the line's
    colour and a depth-faded alpha. Segments split into far (behind the plane of sky) and near
    layers; the canvas is built far-layer → disk → near-layer so the disk occludes the field behind
    it.
    """
    height, width = camera.pixels
    canvas = np.empty((height, width, 3), dtype=np.float64)
    canvas[:] = _BACKGROUND

    segments = _build_segments(paths, colours, camera, outer_radius, depth_fade)
    far = segments.depth < 0.0

    canvas = _composite(canvas, *_rasterise(_select(segments, far), (height, width), line_width))
    canvas = _composite(canvas, *disk_layer)
    canvas = _composite(canvas, *_rasterise(_select(segments, ~far), (height, width), line_width))

    return (np.clip(canvas, 0.0, 1.0) * 255.0).round().astype(np.uint8)


@dataclass(frozen=True)
class _Segments:
    """Projected line segments to rasterise: pixel endpoints, colour, alpha, and mean depth."""

    col0: np.ndarray
    row0: np.ndarray
    col1: np.ndarray
    row1: np.ndarray
    colour: np.ndarray  # (s, 3)
    alpha: np.ndarray  # (s,)
    depth: np.ndarray  # (s,) mean signed depth


def _select(segments: _Segments, mask: np.ndarray | slice) -> _Segments:
    """Return the subset of ``segments`` selected by a boolean mask or a slice."""
    return _Segments(
        col0=segments.col0[mask],
        row0=segments.row0[mask],
        col1=segments.col1[mask],
        row1=segments.row1[mask],
        colour=segments.colour[mask],
        alpha=segments.alpha[mask],
        depth=segments.depth[mask],
    )


def _build_segments(
    paths: list[np.ndarray],
    colours: np.ndarray,
    camera: OrthographicCamera,
    outer_radius: float,
    depth_fade: float,
) -> _Segments:
    """Project every path vertex once and assemble the per-segment endpoint / colour / alpha arrays.

    Vertices of all kept paths are concatenated and projected in one call; a segment joins each
    vertex to the next within the same path (the last vertex of each path starts no segment). Paths
    with fewer than two vertices contribute no segment and are dropped first. The depth-faded alpha
    dims far-side segments by up to ``depth_fade``.
    """
    kept = [(path, colour) for path, colour in zip(paths, colours, strict=True) if len(path) >= 2]
    if not kept:
        empty = np.empty(0)
        return _Segments(empty, empty, empty, empty, np.empty((0, 3)), empty, empty)

    counts = np.array([len(path) for path, _ in kept])
    vertices = np.concatenate([path for path, _ in kept])
    vertex_colour = np.repeat(np.array([colour for _, colour in kept]), counts, axis=0)
    cols, rows, depth = camera.project(vertices)

    is_last = np.zeros(vertices.shape[0], dtype=bool)
    is_last[np.cumsum(counts) - 1] = True
    start = np.nonzero(~is_last)[0]
    end = start + 1

    mean_depth = 0.5 * (depth[start] + depth[end])
    near = np.clip(0.5 * (mean_depth / outer_radius + 1.0), 0.0, 1.0)  # 0 far → 1 near
    alpha = (1.0 - depth_fade) + depth_fade * near
    return _Segments(
        col0=cols[start],
        row0=rows[start],
        col1=cols[end],
        row1=rows[end],
        colour=vertex_colour[start],
        alpha=alpha,
        depth=mean_depth,
    )


def _rasterise(
    segments: _Segments, shape: tuple[int, int], line_width: float
) -> tuple[np.ndarray, np.ndarray]:
    """Anti-alias-rasterise ``segments`` into a premultiplied colour layer ``(C, A)``.

    Each segment is sampled at ~1 px and every sample splats a soft disk of width ``line_width`` (a
    1 px-feathered coverage falloff) into the canvas; overlapping coverage accumulates and the alpha
    is clamped to 1. Returns the premultiplied colour ``C`` ``(H, W, 3)`` and coverage ``A`` ``(H,
    W)`` ready for :func:`_composite`.
    """
    height, width = shape
    colour_acc = np.zeros((height * width, 3), dtype=np.float64)
    alpha_acc = np.zeros(height * width, dtype=np.float64)
    n = segments.col0.size
    if n:
        half = 0.5 * line_width
        radius = max(1, int(np.ceil(half + 0.5)))
        for lo in range(0, n, _SEGMENT_CHUNK):
            _splat_chunk(
                colour_acc,
                alpha_acc,
                _select(segments, slice(lo, lo + _SEGMENT_CHUNK)),
                width,
                height,
                half,
                radius,
            )

    # Renormalise where overlapping coverage pushed alpha past 1, keeping the premultiplied
    # invariant (colour ≤ alpha).
    divisor = np.where(alpha_acc > 1.0, alpha_acc, 1.0)
    colour = (colour_acc / divisor[:, None]).reshape(height, width, 3)
    alpha = np.clip(alpha_acc, 0.0, 1.0).reshape(height, width)
    return colour, alpha


def _splat_chunk(
    colour_acc: np.ndarray,
    alpha_acc: np.ndarray,
    segments: _Segments,
    width: int,
    height: int,
    half: float,
    radius: int,
) -> None:
    """Sample ``segments`` along their length and splat each sample's soft disk into the flat
    accumulators (in place)."""
    delta_col = segments.col1 - segments.col0
    delta_row = segments.row1 - segments.row0
    n_samples = np.ceil(np.hypot(delta_col, delta_row)).astype(int) + 1
    total = int(n_samples.sum())
    if total == 0:
        return

    seg = np.repeat(np.arange(segments.col0.size), n_samples)
    local = np.arange(total) - np.repeat(np.cumsum(n_samples) - n_samples, n_samples)
    t = local / np.where(n_samples > 1, n_samples - 1, 1)[seg]
    sample_col = segments.col0[seg] + t * delta_col[seg]
    sample_row = segments.row0[seg] + t * delta_row[seg]
    sample_colour = segments.colour[seg]
    sample_alpha = segments.alpha[seg]

    base_col = np.floor(sample_col).astype(int)
    base_row = np.floor(sample_row).astype(int)
    for offset_row in range(-radius, radius + 1):
        for offset_col in range(-radius, radius + 1):
            px = base_col + offset_col
            py = base_row + offset_row
            dist = np.hypot(px - sample_col, py - sample_row)
            weight = np.clip(half + 0.5 - dist, 0.0, 1.0) * sample_alpha
            inside = (px >= 0) & (px < width) & (py >= 0) & (py < height) & (weight > 0.0)
            if not inside.any():
                continue
            flat = py[inside] * width + px[inside]
            contribution = weight[inside]
            alpha_acc += np.bincount(flat, weights=contribution, minlength=alpha_acc.size)
            premultiplied = contribution[:, None] * sample_colour[inside]
            for channel in range(3):
                colour_acc[:, channel] += np.bincount(
                    flat, weights=premultiplied[:, channel], minlength=alpha_acc.size
                )


# --- Disk layers -------------------------------------------------------------------------------


def _flat_disk(
    shape: tuple[int, int], camera: OrthographicCamera, inner_radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the premultiplied flat-occulter disk layer ``(C, A)``: a filled, 1 px-feathered
    circle of radius ``inner_radius`` centred on the projected Sun centre."""
    alpha = _disk_alpha(shape, camera, inner_radius)
    return alpha[:, :, None] * np.array(_FLAT_DISK_COLOUR), alpha


def _magnetogram_disk(
    shape: tuple[int, int], camera: OrthographicCamera, field: Field, inner_radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the premultiplied magnetogram disk layer ``(C, A)``: the near-hemisphere ``B_r``.

    For each disk pixel the near-side intersection of its line of sight with the inner sphere is
    found, ``B·r̂`` sampled there from the data, and mapped through a diverging red (outward) / white
    (quiet) / blue (inward) scale saturated at the |B_r| percentile :data:`_MAGNETOGRAM_PERCENTILE`.
    """
    height, width = shape
    look, right, up = camera._basis()
    pixel_scale = camera.fov.to_value(u.R_sun) / width
    alpha = _disk_alpha(shape, camera, inner_radius)
    inside = alpha > 0.0
    if not inside.any():  # disk smaller than a pixel (very wide FOV / tiny inner radius)
        return np.zeros((height, width, 3)), alpha

    rows, cols = np.mgrid[0:height, 0:width].astype(np.float64)
    x = (cols[inside] - 0.5 * (width - 1)) * pixel_scale
    y = (0.5 * (height - 1) - rows[inside]) * pixel_scale
    rho = np.minimum(np.hypot(x, y), inner_radius * (1.0 - _DOMAIN_MARGIN))
    depth = np.sqrt(inner_radius**2 - rho**2)  # near-side line-of-sight ∩ sphere
    surface = x[:, None] * right + y[:, None] * up + depth[:, None] * look
    norm = np.linalg.norm(surface, axis=1, keepdims=True)
    surface *= (inner_radius * (1.0 + _DOMAIN_MARGIN)) / norm

    b = field.sample(surface, gradient=False).b
    b_radial = np.sum(b * surface, axis=1) / norm[:, 0]
    scale = float(np.percentile(np.abs(b_radial), _MAGNETOGRAM_PERCENTILE))
    if not (scale > 0.0 and np.isfinite(scale)):  # degenerate |B_r| scale
        scale = 1.0
    signed = np.clip(b_radial / scale, -1.0, 1.0)
    outward = np.clip(signed, 0.0, 1.0)[:, None]
    inward = np.clip(-signed, 0.0, 1.0)[:, None]
    rgb = (
        np.array(_PHOTOSPHERE) * (1.0 - outward - inward)
        + np.array(_FIELD_OUTWARD) * outward
        + np.array(_FIELD_INWARD) * inward
    )

    colour = np.zeros((height, width, 3))
    colour[inside] = rgb * alpha[inside][:, None]
    return colour, alpha


def _disk_alpha(
    shape: tuple[int, int], camera: OrthographicCamera, inner_radius: float
) -> np.ndarray:
    """Return the ``(H, W)`` 1 px-feathered coverage of the photosphere disk on the plane of sky."""
    height, width = shape
    radius_px = inner_radius / (camera.fov.to_value(u.R_sun) / width)
    rows, cols = np.mgrid[0:height, 0:width]
    dist = np.hypot(cols - 0.5 * (width - 1), rows - 0.5 * (height - 1))
    return np.clip(radius_px + 0.5 - dist, 0.0, 1.0)


def _composite(canvas: np.ndarray, colour: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Composite a premultiplied layer ``(colour, alpha)`` over ``canvas`` (the "over" operator)."""
    return colour + canvas * (1.0 - alpha[:, :, None])
