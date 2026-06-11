"""End-to-end orchestration for a single snapshot: read → resample → Q⊥ volume → render.

Pure functions that take a config (slice) and return the stage product, plus the volume
(de)serialization and the provenance assembly. No CLI or presentation logic lives here, so this
layer is callable as a library. A time series is run per snapshot and combined later.

The two cost axes are kept separate:
:func:`build_field` + :func:`build_volume` bake the viewpoint-independent Q⊥ volume once (the
minutes-scale stage), :func:`render_volume` integrates it for any camera (seconds), and
:func:`save_volume` / :func:`load_volume` persist the volume between the two as a dependency-free
``.npz`` (the ``.qor`` suffix aliases it).

The CR/JD derivation (:func:`derive_cr`, :func:`derive_jd`) turns a user-supplied UTC
``--timestamp`` into a Carrington rotation (sunpy) and Julian date (astropy), both already Qorona
dependencies; the ``.CFmesh`` mesh carries no date, so these are *derived*, never inferred.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
from astropy import units as u

from qorona import __version__
from qorona.config import (
    BrightnessConfig,
    CameraConfig,
    ExportConfig,
    FieldLinesConfig,
    GridConfig,
    InputConfig,
    OutputConfig,
    RenderConfig,
    RunConfig,
    VolumeConfig,
    WeightingConfig,
)
from qorona.field.base import Field
from qorona.field.density import DensityVolume
from qorona.field.sampled import SampledField
from qorona.geometry.camera import OrthographicCamera
from qorona.io import read_solution
from qorona.radiation.brightness import BrightnessResult
from qorona.radiation.brightness import render_brightness as _brightness_los
from qorona.render.fieldlines import FieldLineImage, render_field_lines
from qorona.render.los import LARGE_FOV, LOG_FLOOR, SMALL_FOV, RenderResult, WeightingPreset, render
from qorona.resample.grid import (
    GHOST,
    LogarithmicSpacing,
    PowerLawSpacing,
    RadialSpacing,
    SphericalGrid,
    UniformSpacing,
)
from qorona.resample.resampler import KnnMlsResampler, NearestCellResampler, Resampler
from qorona.squashing.volume import (
    QPerpVolume,
    build_volume_paint,
    build_volume_per_voxel,
    build_volume_reference,
)
from qorona.trace import FieldLines, TurnGuard, lonlat_seeds, trace_field_lines

#: Schema selector strings → the concrete strategy classes / preset instances they name. One place
#: each, so the config names and the pipeline dispatch never drift.
_SPACING_CLASSES: dict[str, type[RadialSpacing]] = {
    "logarithmic": LogarithmicSpacing,
    "power": PowerLawSpacing,
    "uniform": UniformSpacing,
}
_SPACING_NAMES: dict[type, str] = {cls: name for name, cls in _SPACING_CLASSES.items()}
_RESAMPLERS: dict[str, type[Resampler]] = {
    "knn-mls": KnnMlsResampler,
    "nearest-cell": NearestCellResampler,
}
_PRESETS: dict[str, WeightingPreset] = {"large-fov": LARGE_FOV, "small-fov": SMALL_FOV}

#: Magic string tagging the volume-artifact metadata so a format change is detectable.
_ARTIFACT_FORMAT = "qorona-volume-v1"


def _resolved_backend(device: str) -> str:
    """Resolve a requested device to the concrete backend string stamped in provenance.

    ``"gpu:NVIDIA GeForce RTX 4080"`` when the volume baked on the GPU, else ``"cpu"``. Resolved
    at stamp time (the ``.qor`` is cached and reused across cameras, so the *resolved* backend,
    not the requested ``auto``, is the reproducible fact). The hardware tier only; a non-JIT-able
    field that forced a CPU fallback at the dispatcher is not reflected here (it cannot occur for
    the gridded production field, which is always JIT-able).
    """
    from qorona.accel import gpu_name, resolve_device

    if resolve_device(device) == "gpu":
        name = gpu_name()
        return f"gpu:{name}" if name else "gpu"
    return "cpu"


# --- Grid / strategy construction --------------------------------------------------------------


def _spacing(name: str, inner: float, outer: float, exponent: float = 2.0) -> RadialSpacing:
    """Build the named radial spacing law over ``[inner, outer]`` (``exponent`` is power-only)."""
    if name == "logarithmic":
        return LogarithmicSpacing(inner=inner, outer=outer)
    if name == "power":
        return PowerLawSpacing(inner=inner, outer=outer, exponent=exponent)
    if name == "uniform":
        return UniformSpacing(inner=inner, outer=outer)
    raise ValueError(f"unknown spacing law {name!r}")


def _grid_config_spacing(grid_cfg: GridConfig) -> RadialSpacing:
    """Build the spacing law from a :class:`GridConfig` (its inner/outer radii)."""
    return _spacing(grid_cfg.spacing, grid_cfg.inner_radius, grid_cfg.outer_radius)


def _field_grid(grid_cfg: GridConfig) -> SphericalGrid:
    """Build the internal **field** grid (the tracer/interpolant grid) from ``grid_cfg``."""
    return SphericalGrid(
        spacing=_grid_config_spacing(grid_cfg),
        n_r=grid_cfg.n_r,
        n_theta=grid_cfg.n_theta,
        n_phi=grid_cfg.n_phi,
    )


def _volume_grid(grid_cfg: GridConfig, resolution_factor: int) -> SphericalGrid:
    """Build the **volume** grid: the field grid's node counts scaled by ``resolution_factor``,
    sharing the spacing law and radii so it sits on the same shell at a finer pitch."""
    return SphericalGrid(
        spacing=_grid_config_spacing(grid_cfg),
        n_r=grid_cfg.n_r * resolution_factor,
        n_theta=grid_cfg.n_theta * resolution_factor,
        n_phi=grid_cfg.n_phi * resolution_factor,
    )


# --- Stage functions ---------------------------------------------------------------------------


def build_field(
    input_cfg: InputConfig,
    grid_cfg: GridConfig,
    *,
    show_progress: bool = True,
    timings: dict[str, float] | None = None,
) -> SampledField:
    """Read a solution and resample it onto the internal field grid.

    Reads the solution (model inferred from the extension unless ``input_cfg.model`` is set), builds
    the field grid from ``grid_cfg``, and resamples B onto it with the named resampler. When a
    ``timings`` dict is supplied, the ``"read"`` and ``"resample"`` stage durations (seconds) are
    recorded into it for the end-of-run summary.

    Returns
    -------
    SampledField
        The interpolatable field on the internal spherical grid.
    """
    reader_kwargs: dict[str, object] = {}
    if input_cfg.variables is not None:
        reader_kwargs["variables"] = input_cfg.variables
    start = time.perf_counter()
    solution = read_solution(
        input_cfg.path, model=input_cfg.model, show_progress=show_progress, **reader_kwargs
    )
    if timings is not None:
        timings["read"] = time.perf_counter() - start
    grid = _field_grid(grid_cfg)
    resampler = _RESAMPLERS[grid_cfg.resampler]()
    start = time.perf_counter()
    field = SampledField.from_solution(
        solution, grid, resampler=resampler, show_progress=show_progress
    )
    if timings is not None:
        timings["resample"] = time.perf_counter() - start
    return field


def build_volume(
    field: Field,
    volume_cfg: VolumeConfig,
    grid_cfg: GridConfig,
    *,
    show_progress: bool = True,
    timings: dict[str, float] | None = None,
) -> QPerpVolume:
    """Bake the viewpoint-independent Q⊥ volume on the refined volume grid.

    The volume grid is ``grid_cfg`` scaled by ``volume_cfg.resolution_factor``; the builder is
    dispatched on ``volume_cfg.builder`` (``paint`` default, ``per-voxel``/``reference``
    selectable), passing only the knobs that builder reads. A supplied ``timings`` dict receives
    the paint builder's per-stage durations (``"boundary"`` / ``"trace"`` / ``"paint"``).

    Returns
    -------
    QPerpVolume
        The truthful log₁₀ Q⊥ volume.
    """
    grid = _volume_grid(grid_cfg, volume_cfg.resolution_factor)
    common: dict[str, Any] = {
        "rtol": volume_cfg.rtol,
        "cfl": volume_cfg.cfl,
        "max_steps": volume_cfg.max_steps,
        "max_reversals": volume_cfg.max_reversals,
        "turn_guard": TurnGuard(
            max_turn_angle=volume_cfg.max_turn_angle,
            radius=volume_cfg.turn_guard_radius,
            weak_fraction=volume_cfg.turn_guard_weak_fraction,
            min_turns=volume_cfg.min_turns,
        ),
        "workers": volume_cfg.workers,
        "device": volume_cfg.device,
        "precision": volume_cfg.precision,
        "show_progress": show_progress,
    }
    if volume_cfg.builder == "paint":
        return build_volume_paint(
            field,
            grid,
            supersample=volume_cfg.supersample,
            paint_step=volume_cfg.paint_step,
            closed=volume_cfg.closed,
            timings=timings,
            **common,
        )
    if volume_cfg.builder == "per-voxel":
        return build_volume_per_voxel(
            field, grid, supersample=volume_cfg.supersample, closed=volume_cfg.closed, **common
        )
    return build_volume_reference(field, grid, **common)


def render_volume(
    volume: QPerpVolume,
    camera_cfg: CameraConfig,
    weighting_cfg: WeightingConfig,
    render_cfg: RenderConfig,
    *,
    density: DensityVolume | None = None,
    show_progress: bool = True,
) -> RenderResult:
    """Integrate a baked volume for one viewpoint into an eclipse-like image.

    Builds the orthographic camera from the sub-observer angles, converting the config's degrees and
    solar radii into the radians the camera expects, selects the geometric weighting preset, and
    renders. When ``weighting_cfg.thomson`` is set, the baked-in ``density`` is wrapped in a
    :class:`~qorona.radiation.thomson.ThomsonWeight` and composed into the render as the optional
    radiometric factor (an error if the volume carries no density).

    Returns
    -------
    RenderResult
        The depth-coloured image, the grayscale measurement image, coverage, and clamp provenance.
    """
    camera = OrthographicCamera.from_sub_observer(
        longitude=camera_cfg.longitude,
        latitude=camera_cfg.latitude,
        roll=float(np.deg2rad(camera_cfg.roll)),
        fov=camera_cfg.fov * u.R_sun,
        pixels=camera_cfg.pixels,
    )
    thomson = None
    if weighting_cfg.thomson is not None:
        if density is None:
            raise ValueError(
                "Thomson weighting was requested but the volume carries no electron density; "
                "rebuild it from a solution that provides density (e.g. COCONUT 'rho')"
            )
        from qorona.radiation.thomson import ThomsonWeight

        thomson_cfg = weighting_cfg.thomson
        thomson = ThomsonWeight(
            density,
            mode=cast(Any, thomson_cfg.mode),
            u=thomson_cfg.u,
            crossover=thomson_cfg.crossover,
        )
    return render(
        volume,
        camera,
        preset=_PRESETS[weighting_cfg.preset],
        thomson=thomson,
        clamp=render_cfg.clamp,
        raw=render_cfg.raw,
        step=render_cfg.step,
        occult=cast(Any, render_cfg.occult),
        r_occult=render_cfg.r_occult,
        occult_softness=render_cfg.occult_softness,
        percentiles=render_cfg.percentiles,
        display=cast(Any, render_cfg.display),
        polarity_mode=cast(Any, render_cfg.polarity_mode),
        workers=render_cfg.workers,
        show_progress=show_progress,
    )


def render_fieldlines(
    field: Field,
    fieldlines_cfg: FieldLinesConfig,
    camera_cfg: CameraConfig,
    *,
    show_progress: bool = True,
) -> FieldLineImage:
    """Render the field-line view of a field from one viewpoint.

    Builds the orthographic camera from the sub-observer angles (the same degrees-and-solar-radii to
    radians conversion as :func:`render_volume`), then traces and draws a Fibonacci-seeded bundle of
    field lines; the shell radii come from ``field.domain``.

    Returns
    -------
    FieldLineImage
        The drawn image and the open / closed / incomplete line tallies.
    """
    camera = OrthographicCamera.from_sub_observer(
        longitude=camera_cfg.longitude,
        latitude=camera_cfg.latitude,
        roll=float(np.deg2rad(camera_cfg.roll)),
        fov=camera_cfg.fov * u.R_sun,
        pixels=camera_cfg.pixels,
    )
    return render_field_lines(
        field,
        camera,
        seeding=fieldlines_cfg.seeding,
        n_seeds=fieldlines_cfg.n_seeds,
        limb_seeds=fieldlines_cfg.limb_seeds,
        front_loop_length=fieldlines_cfg.front_loop_length,
        colour=fieldlines_cfg.colour,
        magnetogram=fieldlines_cfg.magnetogram,
        show=fieldlines_cfg.show,
        line_width=fieldlines_cfg.line_width,
        depth_fade=fieldlines_cfg.depth_fade,
        rtol=fieldlines_cfg.rtol,
        cfl=fieldlines_cfg.cfl,
        max_steps=fieldlines_cfg.max_steps,
        turn_guard=TurnGuard(
            max_turn_angle=fieldlines_cfg.max_turn_angle,
            radius=fieldlines_cfg.turn_guard_radius,
            weak_fraction=fieldlines_cfg.turn_guard_weak_fraction,
            min_turns=fieldlines_cfg.min_turns,
        ),
        workers=fieldlines_cfg.workers,
        show_progress=show_progress,
    )


def export_lines(
    field: Field,
    export_cfg: ExportConfig,
    *,
    show_progress: bool = True,
) -> FieldLines:
    """Trace the field-line bundle for export.

    Seeds a uniform longitude/latitude grid on the seed sphere (the field's inner boundary unless
    ``export_cfg.seed_radius`` overrides it) and traces every seed both ways to the boundaries,
    keeping the full polylines.

    Returns
    -------
    FieldLines
        The traced bundle, with ``paths`` populated.
    """
    from qorona.accel import apply_workers

    apply_workers(export_cfg.workers)
    radius = (
        export_cfg.seed_radius
        if export_cfg.seed_radius is not None
        else field.domain.inner_radius
    )
    seeds = lonlat_seeds(radius, n_theta=export_cfg.n_theta, n_phi=export_cfg.n_phi)
    return trace_field_lines(
        field,
        seeds,
        rtol=export_cfg.rtol,
        cfl=export_cfg.cfl,
        max_steps=export_cfg.max_steps,
        turn_guard=TurnGuard(
            max_turn_angle=export_cfg.max_turn_angle,
            radius=export_cfg.turn_guard_radius,
            weak_fraction=export_cfg.turn_guard_weak_fraction,
            min_turns=export_cfg.min_turns,
        ),
        store_path=True,
        show_progress=show_progress,
    )


def render_brightness(
    field: SampledField,
    brightness_cfg: BrightnessConfig,
    camera_cfg: CameraConfig,
    *,
    show_progress: bool = True,
) -> BrightnessResult:
    """Render the white-light / polarized-brightness corona of a field from one viewpoint.

    Builds the orthographic camera from the sub-observer angles (the same degrees-and-solar-radii to
    radians conversion as :func:`render_volume`), then integrates the Thomson-scattering brightness
    over the field's electron density along each line of sight. Independent of the Q⊥ volume (the
    density is the only field input), so this runs straight off a read-in solution.

    Returns
    -------
    BrightnessResult
        The polarized (``pB``) and total brightness frames and the geometry the display treatments
        consume.

    Raises
    ------
    ValueError
        If the field carries no electron density (the solution did not provide it).
    """
    if field.density is None:
        raise ValueError(
            "the white-light / pB product needs an electron density, but this solution carries "
            "none; read a solution that provides density (e.g. COCONUT 'rho')"
        )
    camera = OrthographicCamera.from_sub_observer(
        longitude=camera_cfg.longitude,
        latitude=camera_cfg.latitude,
        roll=float(np.deg2rad(camera_cfg.roll)),
        fov=camera_cfg.fov * u.R_sun,
        pixels=camera_cfg.pixels,
    )
    return _brightness_los(
        field.density,
        camera,
        u=brightness_cfg.u,
        crossover=brightness_cfg.crossover,
        step=brightness_cfg.step,
        occult=cast(Any, brightness_cfg.occult),
        r_occult=brightness_cfg.r_occult,
        occult_softness=brightness_cfg.occult_softness,
        workers=brightness_cfg.workers,
        show_progress=show_progress,
    )


def run(cfg: RunConfig, *, show_progress: bool = True) -> RenderResult:
    """The one-shot chain: read → resample → Q⊥ volume → render, all in memory.

    The library convenience path. The CLI's ``run`` command orchestrates the same three stage
    functions directly so it can time each, assemble provenance, and optionally persist the
    intermediate volume; this function is the minimal end-to-end call for programmatic use. The
    top-level
    ``cfg.workers``, when set, fans out to both the volume build and the render (overriding the
    sub-configs' own ``workers``).
    """
    volume_cfg = cfg.volume if cfg.workers is None else replace(cfg.volume, workers=cfg.workers)
    if cfg.device != "auto":
        volume_cfg = replace(volume_cfg, device=cfg.device)
    render_cfg = cfg.render if cfg.workers is None else replace(cfg.render, workers=cfg.workers)
    if cfg.device != "auto":
        render_cfg = replace(render_cfg, device=cfg.device)
    field = build_field(cfg.input, cfg.grid, show_progress=show_progress)
    volume = build_volume(field, volume_cfg, cfg.grid, show_progress=show_progress)
    return render_volume(
        volume, cfg.camera, cfg.weighting, render_cfg,
        density=field.density, show_progress=show_progress,
    )


# --- Carrington rotation / Julian date derivation -----------------------------------------


def _utc_time(timestamp: str) -> Any:
    """Parse a UTC ISO-8601 ``timestamp`` into an :class:`astropy.time.Time` (friendly on error)."""
    from astropy.time import Time

    try:
        return Time(timestamp, scale="utc")
    except Exception as exc:  # astropy raises a variety of types for malformed input
        raise ValueError(
            f"could not parse timestamp {timestamp!r} as a UTC ISO-8601 datetime "
            f"(e.g. '2024-04-08T18:17:00'): {exc}"
        ) from exc


def derive_cr(timestamp: str) -> int:
    """Return the active Carrington-rotation number at a UTC ISO-8601 ``timestamp``.

    The active rotation is ``floor(carrington_rotation_number(t))``; sunpy's fractional part is the
    phase within the rotation.
    """
    from sunpy.coordinates.sun import carrington_rotation_number

    return int(carrington_rotation_number(_utc_time(timestamp)))


def derive_jd(timestamp: str) -> float:
    """Return the Julian date for a UTC ISO-8601 ``timestamp``."""
    return float(_utc_time(timestamp).jd)


def _ephemeris(timestamp: str | None) -> dict[str, object]:
    """Return ``{"cr", "jd"}`` derived from ``timestamp``, or an empty mapping if it is ``None``."""
    if timestamp is None:
        return {}
    return {"cr": derive_cr(timestamp), "jd": derive_jd(timestamp)}


# --- Provenance assembly -----------------------------------------------------------------------


def content_hash(path: str | Path) -> str:
    """Return a short SHA-256 fingerprint of a file's contents (so a render can confirm its volume
    matches the input it was baked from)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def _interior(volume: QPerpVolume) -> np.ndarray:
    """Return the unpadded interior log₁₀ Q⊥ voxels (the real grid nodes, ghost layers dropped)."""
    return volume.log_q_perp[GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST, 0]


def covered_fraction(volume: QPerpVolume) -> float:
    """Return the fraction of interior voxels carrying a finite log₁₀ Q⊥: the volume's coverage
    (only the paint builder leaves gaps), recomputed so it lands in the provenance and summary."""
    return float(np.isfinite(_interior(volume)).mean())


def sub_floor_voxels(volume: QPerpVolume) -> int:
    """Count interior finite voxels below the theoretical Q⊥ ≥ 2 floor (the real-data sub-floor
    tail, a resampling ∇·B artifact the volume keeps truthfully; reported as a volume
    diagnostic)."""
    interior = _interior(volume)
    return int(np.count_nonzero(np.isfinite(interior) & (interior < LOG_FLOOR)))


def build_provenance(
    input_cfg: InputConfig,
    grid_cfg: GridConfig,
    volume_cfg: VolumeConfig,
    *,
    field: SampledField,
    volume: QPerpVolume,
) -> dict[str, Any]:
    """Assemble the build-time provenance: the single mapping stored in the volume artifact and
    read back by ``render`` for the stamp and summary.

    Carries the input path + content hash + derived CR/JD, the field grid + normalization, and the
    resolved volume parameters + grid + covered fraction. Format-independent: flat strings and
    numbers that map one-to-one onto image-header keywords.
    """
    return {
        "format": _ARTIFACT_FORMAT,
        "version": __version__,
        "input": {
            **input_cfg.to_provenance(),
            "content_hash": content_hash(input_cfg.path),
            **_ephemeris(input_cfg.timestamp),
        },
        "field": {**grid_cfg.to_provenance(), "normalization": field.normalization},
        "volume": {
            **volume_cfg.to_provenance(),
            "backend": _resolved_backend(volume_cfg.device),
            "grid": f"{volume.grid.n_r}x{volume.grid.n_theta}x{volume.grid.n_phi}",
            "covered_fraction": covered_fraction(volume),
            "sub_floor_voxels": sub_floor_voxels(volume),
            "has_density": field.density is not None,
        },
    }


def render_provenance(
    build_prov: dict[str, Any],
    camera_cfg: CameraConfig,
    weighting_cfg: WeightingConfig,
    render_cfg: RenderConfig,
    output_cfg: OutputConfig,
    result: RenderResult,
    *,
    timestamp_override: str | None = None,
) -> dict[str, Any]:
    """Extend a build provenance with the render's camera / weighting / render / output and metrics.

    A render off a baked volume recovers the date for the stamp from ``build_prov`` without
    re-supplying it; an explicit ``timestamp_override`` re-derives CR/JD for the stamp.
    """
    prov = json.loads(json.dumps(build_prov))  # deep copy; build_prov is JSON-safe by construction
    if timestamp_override is not None:
        prov["input"] = {
            **prov.get("input", {}),
            "timestamp": timestamp_override,
            **_ephemeris(timestamp_override),
        }
    covered = result.coverage[result.coverage > 0.0]
    prov["camera"] = camera_cfg.to_provenance()
    prov["weighting"] = weighting_cfg.to_provenance()
    prov["render"] = {
        **render_cfg.to_provenance(),
        "backend": "cpu",  # the render runs on the CPU (no GPU render backend)
        "preset": result.preset_name,
        "display_mode": result.display_mode,
        "mean_coverage": float(np.mean(covered)) if covered.size else 0.0,
        "lower_clamped_fraction": result.lower_clamped_fraction,
        "upper_clamped_fraction": result.upper_clamped_fraction,
    }
    prov["output"] = output_cfg.to_provenance()
    return prov


def fieldlines_provenance(
    input_cfg: InputConfig,
    grid_cfg: GridConfig,
    fieldlines_cfg: FieldLinesConfig,
    camera_cfg: CameraConfig,
    output_cfg: OutputConfig,
    result: FieldLineImage,
) -> dict[str, Any]:
    """Assemble the field-line render's provenance: the mapping the stamp and summary read.

    Mirrors :func:`build_provenance` / :func:`render_provenance`: the input (path + content hash +
    derived CR/JD), the field grid, the field-line parameters + line tallies, the camera, and the
    output. JSON-safe by construction (the field-line render has no volume artifact, so its own
    format tag distinguishes it).
    """
    return {
        "format": "qorona-fieldlines-v1",
        "version": __version__,
        "input": {
            **input_cfg.to_provenance(),
            "content_hash": content_hash(input_cfg.path),
            **_ephemeris(input_cfg.timestamp),
        },
        "field": grid_cfg.to_provenance(),
        "fieldlines": {
            **fieldlines_cfg.to_provenance(),
            "n_open": result.n_open,
            "n_closed": result.n_closed,
            "n_incomplete": result.n_incomplete,
        },
        "camera": camera_cfg.to_provenance(),
        "output": output_cfg.to_provenance(),
    }


def export_provenance(
    input_cfg: InputConfig,
    grid_cfg: GridConfig,
    export_cfg: ExportConfig,
    field: Field,
    lines: FieldLines,
) -> dict[str, Any]:
    """Assemble the export's provenance: the mapping the summary and the file's metadata read.

    Mirrors :func:`fieldlines_provenance` (minus camera and image output): the input (path +
    content hash + derived CR/JD), the field grid, and the export parameters with the resolved
    seed radius and the line tallies. JSON-safe, so the writer stores it as the file's
    ``metadata`` block.
    """
    seed_radius = (
        export_cfg.seed_radius
        if export_cfg.seed_radius is not None
        else field.domain.inner_radius
    )
    return {
        "version": __version__,
        "input": {
            **input_cfg.to_provenance(),
            "content_hash": content_hash(input_cfg.path),
            **_ephemeris(input_cfg.timestamp),
        },
        "field": grid_cfg.to_provenance(),
        "export": {
            **export_cfg.to_provenance(),
            "seed_radius": seed_radius,
            "n_open": int(lines.is_open.sum()),
            "n_closed": int(lines.is_closed.sum()),
            "n_incomplete": int(lines.is_incomplete.sum()),
        },
    }


def brightness_provenance(
    input_cfg: InputConfig,
    grid_cfg: GridConfig,
    brightness_cfg: BrightnessConfig,
    camera_cfg: CameraConfig,
    output_cfg: OutputConfig,
    result: BrightnessResult,
) -> dict[str, Any]:
    """Assemble the white-light render's provenance: the mapping the stamp and summary read.

    Mirrors :func:`fieldlines_provenance`: the input (path + content hash + derived CR/JD), the
    field grid, the brightness parameters with the image's polarization and dynamic-range metrics,
    the camera, and the output. JSON-safe by construction (no volume artifact, so its own format tag
    distinguishes it).
    """
    positive_pb = result.polarized[result.polarized > 0.0]
    decades = float(np.log10(positive_pb.max() / positive_pb.min())) if positive_pb.size else 0.0
    positive_total = result.total > 0.0
    median_polarization = (
        float(np.median(result.polarization()[positive_total]))
        if bool(np.any(positive_total))
        else 0.0
    )
    return {
        "format": "qorona-brightness-v1",
        "version": __version__,
        "input": {
            **input_cfg.to_provenance(),
            "content_hash": content_hash(input_cfg.path),
            **_ephemeris(input_cfg.timestamp),
        },
        "field": grid_cfg.to_provenance(),
        "brightness": {
            **brightness_cfg.to_provenance(),
            "median_polarization": median_polarization,
            "pb_decades": decades,
        },
        "camera": camera_cfg.to_provenance(),
        "output": output_cfg.to_provenance(),
    }


# --- Volume disk artifact -----------------------------------------------------------------


def _grid_spec(grid: SphericalGrid) -> dict[str, Any]:
    """Serialize a grid's spacing law + node counts to a JSON-safe spec (round-tripped on load).

    The inner/outer radii are taken from the first and last node radii rather than read off the
    spacing object, so every spacing law serializes the same way regardless of its own fields. The
    exponent applies only to the power law; ``getattr`` defaults it to 2.0 for laws that lack one.
    """
    spacing = grid.spacing
    return {
        "spacing": _SPACING_NAMES[type(spacing)],
        "inner": float(grid.radii[0]),
        "outer": float(grid.radii[-1]),
        "exponent": float(getattr(spacing, "exponent", 2.0)),
        "n_r": grid.n_r,
        "n_theta": grid.n_theta,
        "n_phi": grid.n_phi,
    }


def _grid_from_spec(spec: dict[str, Any]) -> SphericalGrid:
    """Reconstruct a :class:`SphericalGrid` from a :func:`_grid_spec` mapping."""
    spacing = _spacing(
        str(spec["spacing"]),
        float(spec["inner"]),
        float(spec["outer"]),
        float(spec.get("exponent", 2.0)),
    )
    return SphericalGrid(
        spacing=spacing,
        n_r=int(spec["n_r"]),
        n_theta=int(spec["n_theta"]),
        n_phi=int(spec["n_phi"]),
    )


def save_volume(
    volume: QPerpVolume,
    path: str | Path,
    provenance: dict[str, Any],
    *,
    density: DensityVolume | None = None,
    dtype: str = "float32",
    compress: bool = True,
) -> None:
    """Persist a Q⊥ volume (+ optional polarity, density) + build provenance to a ``.npz``.

    The padded ``log_q_perp`` payload is stored ``float32`` by default, lossless relative to the
    engine's ``rtol ≈ 1e-4`` (storing log Q⊥ keeps the precision uniform across the dynamic range,
    NaN gaps bit-exact), and DEFLATE-compressed by default. ``dtype="float64"`` gives a
    bit-exact checkpoint; ``compress=False`` skips the one-time compression. The file is written
    through an open handle so the user's exact suffix (``.qor``) is preserved rather than numpy's
    ``.npz`` default.

    When a ``density`` is supplied (the Thomson / brightness branch) its padded payload and grid
    ride the same archive, so a separate ``render`` has ``Nₑ`` for the optional pB weighting; an
    artifact baked without it simply loads back with no density (backward-compatible).

    Parameters
    ----------
    volume
        The baked volume to persist.
    path
        Destination path (any suffix; ``.qor`` and ``.npz`` are both plain npz archives).
    provenance
        The build provenance (:func:`build_provenance`), stored alongside the grid spec.
    density
        The electron-density volume to persist alongside, or ``None`` to omit it.
    dtype
        ``"float32"`` (default) or ``"float64"`` for the stored arrays.
    compress
        Whether to DEFLATE-compress the archive.
    """
    if dtype not in ("float32", "float64"):
        raise ValueError(f"dtype must be 'float32' or 'float64', got {dtype!r}")
    cast = np.float32 if dtype == "float32" else np.float64
    arrays: dict[str, np.ndarray] = {"log_q_perp": volume.log_q_perp.astype(cast)}
    meta: dict[str, Any] = {
        "format": _ARTIFACT_FORMAT,
        "grid": _grid_spec(volume.grid),
        "provenance": provenance,
    }
    # The polarity channel is the discrete sign {-1, 0, +1}, so it rides as int8 regardless of the
    # float ``dtype`` (lossless, small); it is absent when the volume carries none. The padded
    # ghosts go unused by the render's nearest-cell polarity lookup, so their values do not matter.
    if volume.polarity is not None:
        arrays["polarity"] = volume.polarity.astype(np.int8)
    if density is not None:
        arrays["density"] = density.density.astype(cast)
        meta["density_grid"] = _grid_spec(density.grid)
    # ``np.savez``'s keyword-only ``allow_pickle`` makes the ``**arrays`` unpack ambiguous to the
    # type checker (a dynamic key could shadow it), so the saver is typed loosely here.
    saver: Any = np.savez_compressed if compress else np.savez
    with open(path, "wb") as handle:
        saver(handle, meta=np.array(json.dumps(meta)), **arrays)


def load_volume(
    path: str | Path,
) -> tuple[QPerpVolume, DensityVolume | None, dict[str, Any]]:
    """Load a Q⊥ volume, its electron density (if any), and its build provenance from an artifact.

    The stored arrays (``float32`` or ``float64``) are restored to the ``float64`` the render kernel
    consumes; the grids are reconstructed from their specs, so the loaded volume renders identically
    to the one that was baked. ``density`` is ``None`` for an artifact baked without it; the
    polarity channel (int8 on disk) is restored to ``float32``, or ``None`` when it is absent.

    Returns
    -------
    volume : QPerpVolume
        The reconstructed volume.
    density : DensityVolume or None
        The reconstructed electron-density volume, or ``None`` if the artifact carries none.
    provenance : dict
        The build provenance recorded at bake time.
    """
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["meta"].item()))
        log_q_perp = np.ascontiguousarray(archive["log_q_perp"], dtype=np.float64)
        polarity = None
        if "polarity" in archive.files:
            polarity = np.ascontiguousarray(archive["polarity"], dtype=np.float32)
        density = None
        if "density" in archive.files and "density_grid" in meta:
            density = DensityVolume(
                grid=_grid_from_spec(meta["density_grid"]),
                density=np.ascontiguousarray(archive["density"], dtype=np.float64),
            )
    grid = _grid_from_spec(meta["grid"])
    return (
        QPerpVolume(grid=grid, log_q_perp=log_q_perp, polarity=polarity),
        density,
        meta["provenance"],
    )
