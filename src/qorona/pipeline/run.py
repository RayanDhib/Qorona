"""End-to-end orchestration for a single snapshot: read → resample → Q⊥ volume → render.

Pure functions that take a config (slice) and return the stage product, plus the volume
(de)serialization and the provenance assembly. No CLI or presentation logic lives here, so this
layer is callable as a library. A time series is run per snapshot and combined later.

The two cost axes are kept separate:
:func:`build_field` + :func:`build_volume` bake the viewpoint-independent Q⊥ volume once (the
minutes-scale stage), :func:`render_volume` integrates it for any camera (seconds), and
:func:`save_volume` / :func:`load_volume` persist the volume between the two as a dependency-free
``.npz`` (the ``.qor`` suffix aliases it).

The CR/JD derivation lives in :mod:`qorona.ephemeris`; this layer only records the Carrington
rotation and Julian date it returns into a run's provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
from astropy import units as u

from qorona import __version__
from qorona.config import (
    BrightnessConfig,
    CameraConfig,
    EuvConfig,
    ExportConfig,
    FieldLinesConfig,
    GridConfig,
    InputConfig,
    OutputConfig,
    QMapConfig,
    RenderConfig,
    RunConfig,
    SolutionFrame,
    VolumeConfig,
    WeightingConfig,
)
from qorona.config.schema import (
    CAMERA_INSTRUMENTS,
    CORONAL_FOV,
    CORONAL_LATITUDE,
    CORONAL_QMAP_RADIUS,
    CORONAL_R_OCCULT,
    CORONAL_SHELL,
    CORONAL_STEP,
    CORONAL_WL_R_OCCULT,
    EUV_OUTER_RADIUS,
    HELIO_FOV_FACTOR,
    HELIO_LATITUDE,
    HELIO_PERSPECTIVE_LATITUDE,
    HELIO_QMAP_FACTOR,
    STEP_PER_FOV,
    WL_OCCULT_FACTOR,
)
from qorona.console import progress_bar
from qorona.ephemeris import derive_cr, derive_jd, sub_observer_point, utc_time
from qorona.field.base import Field
from qorona.field.density import DensityVolume
from qorona.field.plasma import TRANSITION_REGION_TEMPERATURE, PlasmaVolume, decode_plasma
from qorona.field.sampled import SampledField
from qorona.geometry.camera import Camera, OrthographicCamera, PerspectiveCamera, check_arc_reach
from qorona.io import NativeSolution, read_solution
from qorona.io.readers import category_of, resolve_model
from qorona.radiation import punch_l3
from qorona.radiation.brightness import BrightnessResult
from qorona.radiation.brightness import render_brightness as _brightness_los
from qorona.radiation.euv import EuvResult
from qorona.radiation.euv import render_euv as _euv_los
from qorona.radiation.response import CHANNELS
from qorona.render.fieldlines import FieldLineImage, render_field_lines
from qorona.render.los import LARGE_FOV, LOG_FLOOR, SMALL_FOV, RenderResult, WeightingPreset, render
from qorona.render.shell import QMap
from qorona.resample.grid import (
    GHOST,
    LogarithmicSpacing,
    PowerLawSpacing,
    RadialSpacing,
    SphericalGrid,
    UniformSpacing,
    pad_field,
)
from qorona.resample.resampler import (
    KnnMlsResampler,
    NearestCellResampler,
    Resampler,
    StructuredGridResampler,
)
from qorona.squashing.volume import (
    QPerpVolume,
    build_volume_paint,
    build_volume_per_voxel,
    build_volume_reference,
    radial_sign_on_grid,
)
from qorona.trace import FieldLines, TurnGuard, lonlat_seeds, trace_field_lines
from qorona.trace.seeding import lonlat_grid, lonlat_shell

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
    "structured": StructuredGridResampler,
}
_PRESETS: dict[str, WeightingPreset] = {"large-fov": LARGE_FOV, "small-fov": SMALL_FOV}

#: Magic string tagging the volume-artifact metadata so a format change is detectable.
_ARTIFACT_FORMAT = "qorona-volume-v1"


def _resolved_backend(device: str) -> str:
    """Resolve a requested device to the concrete backend string stamped in provenance.

    ``"gpu:NVIDIA GeForce RTX 4080"`` when the stage ran on the GPU, else ``"cpu"``. Resolved at
    stamp time (the ``.qor`` is cached and reused across cameras, so the *resolved* backend, not
    the requested ``auto``, is the reproducible fact). The hardware tier only (gpu/cpu); shared by
    the volume-build and render stamps.
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
    """Build the spacing law from a :class:`GridConfig`'s **resolved** inner/outer radii.

    Both grid builders run only after the shell sentinels are resolved (in :func:`build_field`,
    and off the field's shell in :func:`build_volume`), so the radii are concrete here.
    """
    inner, outer = grid_cfg.inner_radius, grid_cfg.outer_radius
    assert inner is not None and outer is not None
    return _spacing(grid_cfg.spacing, inner, outer)


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


def _resolve_resampler(name: str, solution: NativeSolution) -> str:
    """Resolve the ``auto`` resampler to the solution's natural one: ``structured`` when the
    solution carries structured axes, else the ``knn-mls`` default."""
    if name != "auto":
        return name
    return "structured" if solution.structured is not None else "knn-mls"


# --- Category-keyed default resolution ---------------------------------------------------------
#
# The shell-dependent geometry fields default to ``None`` sentinels in :mod:`qorona.config.schema`;
# these resolvers turn each ``None`` into a concrete number by solution category, once at each
# pipeline entry. Coronal solutions resolve to the locked constants (shell-independent);
# heliospheric solutions derive from the resolved shell. Everything downstream (engine calls,
# provenance, stamps) then reads the resolved config, so provenance always spells a number.


def _resolve_grid(grid_cfg: GridConfig, solution: NativeSolution) -> GridConfig:
    """Resolve ``None`` shell radii to the solution category's natural shell."""
    inner, outer = grid_cfg.inner_radius, grid_cfg.outer_radius
    if inner is None or outer is None:
        if solution.metadata.category == "heliosphere":
            native = (
                solution.boundaries["inner"].mean_radius.to_value(u.R_sun),
                solution.boundaries["outer"].mean_radius.to_value(u.R_sun),
            )
        else:
            native = CORONAL_SHELL
        inner = native[0] if inner is None else inner
        outer = native[1] if outer is None else outer
    if not 0.0 < inner < outer:
        raise ValueError(f"resolved shell needs 0 < inner < outer, got ({inner}, {outer}) R_sun")
    return replace(grid_cfg, inner_radius=float(inner), outer_radius=float(outer))


def _resolve_projection(camera_cfg: CameraConfig, category: str) -> CameraConfig:
    """Resolve a ``None`` projection and check that the camera settings agree with it.

    The projection resolves by category (corona: orthographic; heliosphere: perspective), and a
    named instrument frame is always perspective. This function is independent of the solution's
    shell, so the CLI calls it before any expensive stage to fail fast.

    Raises :class:`ValueError` when a setting belongs to the other camera: ``fov`` in R☉ under the
    perspective camera, or ``fov_deg`` or an instrument under the orthographic one. It also raises
    for a coronal perspective view without ``fov_deg``.
    """
    helio = category == "heliosphere"
    projection = camera_cfg.projection
    if projection is None:
        perspective = helio or camera_cfg.instrument is not None
        projection = "perspective" if perspective else "orthographic"
    if projection == "orthographic":
        if camera_cfg.instrument is not None:
            raise ValueError(
                f"--instrument {camera_cfg.instrument} is a perspective frame; drop "
                "--projection orthographic, or drop --instrument"
            )
        if camera_cfg.fov_deg is not None:
            raise ValueError(
                "--fov-deg is an angular width for the perspective camera; the orthographic "
                "camera takes its width in R_sun with --fov"
            )
    else:
        if camera_cfg.fov is not None:
            reason = (
                "heliospheric inputs default to the perspective camera"
                if helio and camera_cfg.projection is None and camera_cfg.instrument is None
                else "this view uses the perspective camera"
            )
            raise ValueError(
                f"--fov is a width in R_sun for the orthographic camera; {reason}. Use --fov-deg "
                "for an angular field of view, --instrument punch-mosaic for the PUNCH framing, "
                "or --projection orthographic to keep --fov in R_sun"
            )
        if camera_cfg.fov_deg is None and not helio:
            raise ValueError(
                "a perspective view of a coronal solution needs its angular width: pass "
                "--fov-deg, or an --instrument frame"
            )
        if camera_cfg.fov_deg is not None and camera_cfg.projection_code in (None, "arc"):
            check_arc_reach(camera_cfg.fov_deg, camera_cfg.pixels, camera_cfg.reference_pixel)
    return replace(camera_cfg, projection=projection)


def _resolve_camera(
    camera_cfg: CameraConfig,
    category: str,
    inner: float,
    outer: float,
    *,
    coronal_fov: float = CORONAL_FOV,
) -> CameraConfig:
    """Resolve the camera's ``None`` fields by category, off the solution's shell.

    The projection resolves first (:func:`_resolve_projection`, which also checks the settings),
    and the rest follow the camera it picks:

    - Orthographic: a ``None`` ``fov`` resolves to ``coronal_fov`` (the coronal constant by
      default) or to the full heliospheric domain. A ``None`` latitude resolves to 0 (corona,
      edge-on) or 90 (the heliospheric equatorial-plane overview).
    - Perspective: a ``None`` ``fov_deg`` resolves to the PUNCH mosaic width (heliosphere only).
      A ``None`` latitude resolves to 0, and a ``None`` ``projection_code`` to ``arc``. Raises
      :class:`ValueError` when the observer sits inside the inner boundary ``inner`` (R☉).
    """
    camera_cfg = _resolve_projection(camera_cfg, category)
    helio = category == "heliosphere"
    if camera_cfg.projection == "orthographic":
        fov = camera_cfg.fov
        if fov is None:
            fov = float(HELIO_FOV_FACTOR * outer if helio else coronal_fov)
        latitude = camera_cfg.latitude
        if latitude is None:
            latitude = HELIO_LATITUDE if helio else CORONAL_LATITUDE
        return replace(camera_cfg, fov=fov, latitude=latitude)

    distance = float((camera_cfg.observer_distance * u.AU).to_value(u.R_sun))
    if distance <= inner:
        raise ValueError(
            f"the observer at {camera_cfg.observer_distance:.3g} AU ({distance:.1f} R_sun) is "
            f"inside the solution's inner boundary at {inner:g} R_sun; choose a more distant "
            "observer"
        )
    fov_deg = camera_cfg.fov_deg
    if fov_deg is None:
        fov_deg = CAMERA_INSTRUMENTS["punch-mosaic"].fov_deg
    latitude = camera_cfg.latitude
    if latitude is None:
        latitude = HELIO_PERSPECTIVE_LATITUDE if helio else CORONAL_LATITUDE
    if camera_cfg.projection_code in (None, "arc"):
        check_arc_reach(fov_deg, camera_cfg.pixels, camera_cfg.reference_pixel)
    return replace(
        camera_cfg,
        fov_deg=fov_deg,
        latitude=latitude,
        projection_code=camera_cfg.projection_code or "arc",
    )


def _camera(camera_cfg: CameraConfig) -> Camera:
    """Build the rendering camera of a resolved ``camera_cfg``.

    Converts the config's degrees, solar radii and astronomical units into the radians and
    quantities the cameras take.
    """
    latitude = camera_cfg.latitude
    assert latitude is not None  # resolved by _resolve_camera
    roll = float(np.deg2rad(camera_cfg.roll))
    if camera_cfg.projection == "perspective":
        fov_deg = camera_cfg.fov_deg
        assert fov_deg is not None  # resolved by _resolve_camera
        return PerspectiveCamera.from_sub_observer(
            longitude=camera_cfg.longitude,
            latitude=latitude,
            observer_distance=camera_cfg.observer_distance * u.AU,
            fov=fov_deg * u.deg,
            pixels=camera_cfg.pixels,
            roll=roll,
            projection_code=cast(Any, camera_cfg.projection_code or "arc"),
            reference_pixel=camera_cfg.reference_pixel,
        )
    fov = camera_cfg.fov
    assert fov is not None  # resolved by _resolve_camera
    return OrthographicCamera.from_sub_observer(
        longitude=camera_cfg.longitude,
        latitude=latitude,
        roll=roll,
        fov=fov * u.R_sun,
        pixels=camera_cfg.pixels,
    )


def _default_step(category: str, camera_cfg: CameraConfig, outer: float) -> float:
    """The category-keyed line-of-sight step (R☉) for a resolved ``camera_cfg``.

    Corona: the locked constant. Heliosphere: the orthographic width over
    :data:`STEP_PER_FOV`. The perspective camera has an angular width, so it uses the default
    orthographic width, the shell diameter, instead.
    """
    if category != "heliosphere":
        return CORONAL_STEP
    if camera_cfg.projection == "perspective":
        return float(HELIO_FOV_FACTOR * outer / STEP_PER_FOV)
    fov = camera_cfg.fov
    assert fov is not None  # resolved by _resolve_camera
    return float(fov / STEP_PER_FOV)


def _resolve_render(
    render_cfg: RenderConfig,
    category: str,
    camera_cfg: CameraConfig,
    inner: float,
    outer: float,
) -> RenderConfig:
    """Resolve ``None`` step/r_occult by category (constants, or shell-derived) for a resolved
    ``camera_cfg``."""
    helio = category == "heliosphere"
    if render_cfg.step is None:
        render_cfg = replace(render_cfg, step=_default_step(category, camera_cfg, outer))
    if render_cfg.r_occult is None:
        render_cfg = replace(render_cfg, r_occult=float(inner) if helio else CORONAL_R_OCCULT)
    return render_cfg


def _resolve_brightness(
    brightness_cfg: BrightnessConfig,
    category: str,
    camera_cfg: CameraConfig,
    inner: float,
    outer: float,
) -> BrightnessConfig:
    """Resolve ``None`` step/r_occult for the brightness render, by category, for a resolved
    ``camera_cfg``."""
    helio = category == "heliosphere"
    if brightness_cfg.step is None:
        brightness_cfg = replace(brightness_cfg, step=_default_step(category, camera_cfg, outer))
    if brightness_cfg.r_occult is None:
        brightness_cfg = replace(
            brightness_cfg,
            r_occult=float(WL_OCCULT_FACTOR * inner) if helio else CORONAL_WL_R_OCCULT,
        )
    return brightness_cfg


def _resolve_qmap(qmap_cfg: QMapConfig, category: str, inner: float) -> QMapConfig:
    """Resolve a ``None`` Q-map shell radius by category."""
    if qmap_cfg.radius is not None:
        return qmap_cfg
    radius = HELIO_QMAP_FACTOR * inner if category == "heliosphere" else CORONAL_QMAP_RADIUS
    return replace(qmap_cfg, radius=float(radius))


def build_field(
    input_cfg: InputConfig,
    grid_cfg: GridConfig,
    *,
    show_progress: bool = True,
    timings: dict[str, float] | None = None,
) -> SampledField:
    """Read a solution and resample it onto the internal field grid.

    Reads the solution (model inferred from the extension unless ``input_cfg.model`` is set), builds
    the field grid from ``grid_cfg``, and resamples B onto it with the named resampler. A set
    ``input_cfg.mean_molecular_weight`` replaces the μ that the reader declares. When a
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
    if input_cfg.mean_molecular_weight is not None:
        solution.metadata.mean_molecular_weight = input_cfg.mean_molecular_weight
    grid_cfg = _resolve_grid(grid_cfg, solution)
    grid = _field_grid(grid_cfg)
    resampler_name = _resolve_resampler(grid_cfg.resampler, solution)
    resampler: Resampler
    if resampler_name == "knn-mls":
        resampler = KnnMlsResampler(n_neighbors=grid_cfg.n_neighbors)
    else:
        resampler = _RESAMPLERS[resampler_name]()
    start = time.perf_counter()
    field = SampledField.from_solution(
        solution, grid, resampler=resampler, show_progress=show_progress
    )
    field.resampler_name = resampler_name
    field.category = solution.metadata.category
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

    The volume grid must sit on the same shell as the field grid. For a
    :class:`~qorona.field.sampled.SampledField`, any ``None`` shell sentinels in ``grid_cfg`` are
    resolved off the field's grid (whose radii the field-build settled by solution category), so the
    caller need not pre-resolve them; an analytic field carries no such grid, so its ``grid_cfg``
    radii must be set explicitly.

    Returns
    -------
    QPerpVolume
        The truthful log₁₀ Q⊥ volume.
    """
    if isinstance(field, SampledField):
        grid_cfg = replace(
            grid_cfg,
            inner_radius=float(field.grid.radii[0]),
            outer_radius=float(field.grid.radii[-1]),
        )
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
        volume = build_volume_paint(
            field,
            grid,
            supersample=volume_cfg.supersample,
            paint_step=volume_cfg.paint_step,
            closed=volume_cfg.closed,
            timings=timings,
            **common,
        )
    elif volume_cfg.builder == "per-voxel":
        volume = build_volume_per_voxel(
            field, grid, supersample=volume_cfg.supersample, closed=volume_cfg.closed, **common
        )
    else:
        volume = build_volume_reference(field, grid, **common)

    # The Q-map's local radial sign, baked on the field grid (the production SampledField).
    if isinstance(field, SampledField):
        field_grid = field.grid
        sign_flat = radial_sign_on_grid(field)
        sign_padded = pad_field(
            sign_flat.reshape(field_grid.n_r, field_grid.n_theta, field_grid.n_phi, 1)
        ).astype(np.float32, copy=False)
        volume = replace(volume, radial_sign=sign_padded, radial_sign_grid=field_grid)
    return volume


def render_volume(
    volume: QPerpVolume,
    camera_cfg: CameraConfig,
    weighting_cfg: WeightingConfig,
    render_cfg: RenderConfig,
    *,
    category: str = "corona",
    density: DensityVolume | None = None,
    show_progress: bool = True,
) -> RenderResult:
    """Integrate a baked volume for one viewpoint into an eclipse-like image.

    Builds the camera the resolved config selects (:func:`_camera`), selects the geometric
    weighting preset, and renders. When ``weighting_cfg.thomson`` is set, the baked-in ``density``
    is wrapped in a :class:`~qorona.radiation.thomson.ThomsonWeight` and composed into the render
    as the optional radiometric factor (an error if the volume carries no density). The ``None``
    camera fields, LOS step, and occulter radius resolve by ``category`` off the volume's shell
    (from the input's solution category).

    Returns
    -------
    RenderResult
        The depth-coloured image, the grayscale measurement image, coverage, and clamp provenance.
    """
    inner, outer = float(volume.grid.radii[0]), float(volume.grid.radii[-1])
    camera_cfg = _resolve_camera(camera_cfg, category, inner, outer)
    render_cfg = _resolve_render(render_cfg, category, camera_cfg, inner, outer)
    step, r_occult = render_cfg.step, render_cfg.r_occult
    assert step is not None and r_occult is not None
    camera = _camera(camera_cfg)
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
        floor=render_cfg.floor,
        step=step,
        occult=cast(Any, render_cfg.occult),
        r_occult=r_occult,
        occult_softness=render_cfg.occult_softness,
        disk_tone=render_cfg.disk_tone,
        disk_desat=render_cfg.disk_desat,
        percentiles=render_cfg.percentiles,
        display=cast(Any, render_cfg.display),
        polarity_mode=cast(Any, render_cfg.polarity_mode),
        workers=render_cfg.workers,
        device=render_cfg.device,
        precision=render_cfg.precision,
        show_progress=show_progress,
    )


def qmap_from_volume(
    volume: QPerpVolume, qmap_cfg: QMapConfig, *, category: str = "corona"
) -> QMap:
    """Slice a Q-map from the cached Q⊥ volume at ``qmap_cfg.radius``.

    Samples log₁₀ Q⊥ and the local radial-field sign (both interpolated) on an ``n_theta x n_phi``
    longitude/latitude shell, a pure read of the cached volume, the viewpoint-independent sibling
    of :func:`render_volume`. A ``None`` shell radius resolves by ``category`` off the volume's
    inner shell. The shell radius is clamped to the outer boundary, so mapping at the bake's outer
    radius (the ``[1, r]`` mapping domain) is supported.
    """
    qmap_cfg = _resolve_qmap(qmap_cfg, category, float(volume.grid.radii[0]))
    shell_radius = qmap_cfg.radius
    assert shell_radius is not None
    n_theta, n_phi = qmap_cfg.n_theta, qmap_cfg.n_phi
    radius = min(shell_radius, float(volume.grid.radii[-1]) * (1.0 - 1e-9))
    points = lonlat_shell(radius, n_theta, n_phi)
    theta, phi = lonlat_grid(n_theta, n_phi)
    log_q_perp = volume.sample(points).reshape(n_theta, n_phi)
    radial_sign = volume.sample_radial_sign(points).reshape(n_theta, n_phi)
    return QMap(
        radius=shell_radius,
        theta=theta,
        phi=phi,
        log_q_perp=log_q_perp,
        radial_sign=radial_sign,
    )


def render_fieldlines(
    field: Field,
    fieldlines_cfg: FieldLinesConfig,
    camera_cfg: CameraConfig,
    *,
    category: str = "corona",
    show_progress: bool = True,
) -> FieldLineImage:
    """Render the field-line view of a field from one viewpoint.

    Builds the camera the resolved config selects (:func:`_camera`, as in :func:`render_volume`),
    then traces and draws a Fibonacci-seeded bundle of field lines. The shell radii come from
    ``field.domain``. The ``None`` camera fields resolve by ``category`` off the field's shell.

    Returns
    -------
    FieldLineImage
        The drawn image and the open / closed / incomplete line tallies.
    """
    camera_cfg = _resolve_camera(
        camera_cfg, category, field.domain.inner_radius, field.domain.outer_radius
    )
    return render_field_lines(
        field,
        _camera(camera_cfg),
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
        export_cfg.seed_radius if export_cfg.seed_radius is not None else field.domain.inner_radius
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
    density: DensityVolume,
    brightness_cfg: BrightnessConfig,
    camera_cfg: CameraConfig,
    *,
    category: str = "corona",
    show_progress: bool = True,
) -> BrightnessResult:
    """Render the white-light / polarized-brightness corona of a density from one viewpoint.

    Builds the camera the resolved config selects (:func:`_camera`, as in :func:`render_volume`),
    then integrates the Thomson-scattering brightness over the electron density along each line of
    sight. Independent of the Q⊥ volume (the density
    is the only field input), so the caller resolves the density from either a read-in solution
    (``field.density``) or a baked volume artifact (:func:`load_volume`). The ``None`` camera FOV,
    LOS step, and occulter radius resolve by ``category`` off the density's shell.

    Returns
    -------
    BrightnessResult
        The polarized (``pB``) and total brightness frames and the geometry the display stages
        consume.
    """
    inner, outer = float(density.grid.radii[0]), float(density.grid.radii[-1])
    camera_cfg = _resolve_camera(camera_cfg, category, inner, outer)
    brightness_cfg = _resolve_brightness(brightness_cfg, category, camera_cfg, inner, outer)
    step, r_occult = brightness_cfg.step, brightness_cfg.r_occult
    assert step is not None and r_occult is not None
    return _brightness_los(
        density,
        _camera(camera_cfg),
        u=brightness_cfg.u,
        crossover=brightness_cfg.crossover,
        step=step,
        occult=cast(Any, brightness_cfg.occult),
        r_occult=r_occult,
        occult_softness=brightness_cfg.occult_softness,
        scale=cast(Any, brightness_cfg.brightness_scale),
        workers=brightness_cfg.workers,
        show_progress=show_progress,
    )


def render_punch_l3(
    density: DensityVolume,
    brightness_cfg: BrightnessConfig,
    camera_cfg: CameraConfig,
    *,
    timestamp: str,
    frame: SolutionFrame,
    category: str = "corona",
    show_progress: bool = True,
) -> punch_l3.PunchL3Result:
    """Render a frame from Earth and apply the PUNCH Level 3 background treatment.

    ``camera_cfg`` must place the observer at Earth at ``timestamp``. The background frames view
    the same density from Earth at the times of :func:`punch_l3.window_offsets`, with the sub-Earth
    point in the solution ``frame`` (a Stonyhurst solution holds at ``timestamp`` and turns with
    the Sun). They are rendered at :data:`punch_l3.BACKGROUND_SIZE` and interpolated to the frame.
    A background frame counts at a pixel only where it shows corona there. Pixels with no corona in
    the untreated frame stay zero, and so do pixels within two background pixels of an occulter
    edge, where the interpolated background is not reliable.
    """
    inner, outer = float(density.grid.radii[0]), float(density.grid.radii[-1])
    camera_cfg = _resolve_camera(camera_cfg, category, inner, outer)
    brightness_cfg = _resolve_brightness(brightness_cfg, category, camera_cfg, inner, outer)
    full = render_brightness(
        density, brightness_cfg, camera_cfg, category=category, show_progress=show_progress
    )

    height, width = camera_cfg.pixels
    scale = min(1.0, punch_l3.BACKGROUND_SIZE / max(height, width))
    small = (max(1, round(height * scale)), max(1, round(width * scale)))
    reference = camera_cfg.reference_pixel
    if reference is not None:
        reference = (
            (reference[0] + 0.5) * small[1] / width - 0.5,
            (reference[1] + 0.5) * small[0] / height - 0.5,
        )
    small_cfg = replace(camera_cfg, pixels=small, reference_pixel=reference)

    offsets = punch_l3.window_offsets()
    start = utc_time(timestamp)
    stack = np.empty((offsets.size, 3, *small), dtype=np.float32)
    alpha_small: np.ndarray | None = None
    with progress_bar("Background frames", offsets.size, enabled=show_progress) as tick:
        for index, hours in enumerate(offsets):
            when = start + hours * u.hour
            when.precision = 0
            longitude, latitude, distance = sub_observer_point(
                "earth", str(when.isot), frame, epoch=timestamp
            )
            view = replace(
                small_cfg, longitude=longitude, latitude=latitude, observer_distance=distance
            )
            result = render_brightness(
                density, brightness_cfg, view, category=category, show_progress=False
            )
            if alpha_small is None:
                alpha_small = punch_l3.radial_angle(result.x, result.y)
            channels = punch_l3.to_channels(result.total, result.polarized, alpha_small)
            stack[index] = np.where(result.total > 0.0, channels, np.nan)
            tick(index + 1)

    small_usual = punch_l3.background(stack)
    usual, complete = punch_l3.upsample(small_usual, np.isfinite(small_usual[0]), (height, width))
    alpha = punch_l3.radial_angle(full.x, full.y)
    channels = punch_l3.to_channels(full.total, full.polarized, alpha) - usual
    total, polarized = punch_l3.from_channels(channels, alpha)
    corona = (full.total > 0.0) & complete
    treated = replace(
        full,
        total=np.where(corona, total, 0.0),
        polarized=np.where(corona, polarized, 0.0),
    )
    return punch_l3.PunchL3Result(
        treated=treated,
        full=full,
        valid=corona,
        frames=int(offsets.size),
        background_pixels=small,
    )


# --- Synthetic EUV ------------------------------------------------------------------------------


def build_plasma(
    input_cfg: InputConfig,
    grid_cfg: GridConfig,
    *,
    euv_cfg: EuvConfig | None = None,
    field_radius: float | None = None,
    show_progress: bool = True,
    timings: dict[str, float] | None = None,
) -> tuple[PlasmaVolume, GridConfig]:
    """Read a coronal solution and resample its electron density and temperature for EUV.

    The plasma grid is ``grid_cfg``, with a ``None`` inner radius resolved to the coronal shell
    and a ``None`` outer radius to :data:`~qorona.config.schema.EUV_OUTER_RADIUS`, or further when
    the camera's ``field_radius`` (:func:`euv_field_radius`) reaches past it (:func:`_euv_shell`).
    The k-NN MLS resampler keeps its stencil at ``grid_cfg.n_neighbors`` on every mesh
    (no scaling with the cell count), which keeps the transition region of a refined mesh sharp.
    A set ``input_cfg.mean_molecular_weight`` replaces the μ that the reader declares. With an
    ``euv_cfg``, the transition-region check of :func:`resolve_tr_temperature` runs on the decode,
    before the resample, so a missing broadening temperature fails fast. When a ``timings`` dict
    is supplied, the ``"read"`` and ``"resample"`` durations (seconds) are recorded into it.

    Returns
    -------
    plasma
        The plasma volume.
    grid_cfg
        The plasma grid config with its shell and resampler resolved.

    Raises
    ------
    ValueError
        If the solution is heliospheric (its inner boundary is far above the EUV corona), or if
        the transition-region check fails.
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
    if solution.metadata.category == "heliosphere":
        inner = solution.boundaries["inner"].mean_radius.to_value(u.R_sun)
        raise ValueError(
            f"the EUV render needs a coronal solution; this {solution.metadata.model.upper()} "
            f"solution starts at {inner:.3g} R_sun, far above the EUV corona"
        )
    if input_cfg.mean_molecular_weight is not None:
        solution.metadata.mean_molecular_weight = input_cfg.mean_molecular_weight
    decoded = decode_plasma(solution)
    if euv_cfg is not None:
        _tr_temperature(decoded.transition_region, solution.metadata.tr_temperature, euv_cfg)
    if grid_cfg.outer_radius is None:
        grid_cfg = _euv_shell(grid_cfg, solution, field_radius)
    grid_cfg = _resolve_grid(grid_cfg, solution)
    resampler_name = _resolve_resampler(grid_cfg.resampler, solution)
    resampler: Resampler
    if resampler_name == "knn-mls":
        resampler = KnnMlsResampler(n_neighbors=grid_cfg.n_neighbors, reference_cell_count=None)
    else:
        resampler = _RESAMPLERS[resampler_name]()
    start = time.perf_counter()
    plasma = PlasmaVolume.from_solution(
        solution, _field_grid(grid_cfg), resampler, decoded=decoded, show_progress=show_progress
    )
    if timings is not None:
        timings["resample"] = time.perf_counter() - start
    return plasma, replace(grid_cfg, resampler=resampler_name)


def _euv_shell(
    grid_cfg: GridConfig, solution: NativeSolution, field_radius: float | None
) -> GridConfig:
    """The default outer radius of the EUV plasma grid: the EUV shell, or the camera's field.

    Without a ``field_radius`` beyond :data:`~qorona.config.schema.EUV_OUTER_RADIUS`, the outer
    radius is that constant. Otherwise it is ``field_radius`` rounded up to the next 0.5 R☉, at most
    the solution's outer boundary, so the image has no edge where the grid stops. A logarithmic
    grid then scales ``n_r`` with the log of the shell, so its radial spacing (and its base cell)
    stays that of ``n_r`` nodes out to the constant.
    """
    if field_radius is None or field_radius <= EUV_OUTER_RADIUS:
        return replace(grid_cfg, outer_radius=EUV_OUTER_RADIUS)
    boundary = solution.boundaries["outer"].mean_radius.to_value(u.R_sun)
    outer = max(EUV_OUTER_RADIUS, min(math.ceil(2.0 * field_radius) / 2.0, float(boundary)))
    n_r = grid_cfg.n_r
    if grid_cfg.spacing == "logarithmic":
        inner = grid_cfg.inner_radius if grid_cfg.inner_radius is not None else CORONAL_SHELL[0]
        n_r = math.ceil(n_r * math.log(outer / inner) / math.log(EUV_OUTER_RADIUS / inner))
    return replace(grid_cfg, outer_radius=outer, n_r=n_r)


def euv_field_radius(camera_cfg: CameraConfig, euv_cfg: EuvConfig) -> float:
    """The largest distance (R☉) from Sun centre that the EUV camera's field reaches.

    This is the impact parameter of the ray through the farthest image corner, for the camera as
    :func:`resolve_euv_camera` resolves it. The plasma grid uses it to cover the whole field
    (:func:`build_plasma`).
    """
    camera = _camera(resolve_euv_camera(camera_cfg, euv_cfg, CORONAL_SHELL[0], EUV_OUTER_RADIUS))
    height, width = camera.pixels
    corners = (0, width - 1, (height - 1) * width, height * width - 1)
    return max(float(camera.ray_batch(i, i + 1).packed()[2][0]) for i in corners)


def resolve_tr_temperature(plasma: PlasmaVolume, euv_cfg: EuvConfig) -> float | None:
    """The transition-region broadening temperature (K) that the EUV response is corrected for.

    The config's value wins, then the solution's own record. ``None`` applies no correction:
    the correction is off, or the solution has no transition region and no known temperature.

    Raises
    ------
    ValueError
        If the correction is on and the solution includes the transition region, but no
        broadening temperature is known.
    """
    return _tr_temperature(plasma.transition_region, plasma.tr_temperature, euv_cfg)


def _tr_temperature(
    transition_region: bool, recorded: float | None, euv_cfg: EuvConfig
) -> float | None:
    """The rule of :func:`resolve_tr_temperature`, on the solution's own facts."""
    if not euv_cfg.tr_correction:
        return None
    tr_temperature = euv_cfg.tr_temperature if euv_cfg.tr_temperature is not None else recorded
    if tr_temperature is None and transition_region:
        raise ValueError(
            f"this solution includes the transition region (its base is below "
            f"{TRANSITION_REGION_TEMPERATURE:.0e} K). Models broaden that layer artificially below "
            "a temperature Tc set in the run's configuration, which makes its EUV emission far "
            "too bright unless it is corrected. Pass the run's Tc with --tr-temperature (for "
            "example --tr-temperature 2e5), or --tr-temperature none to render without the "
            "correction"
        )
    return tr_temperature


def resolve_euv_camera(
    camera_cfg: CameraConfig, euv_cfg: EuvConfig, inner: float, outer: float
) -> CameraConfig:
    """Resolve the EUV camera: perspective, framed by the channels' instrument by default.

    A ``None`` angular width and projection code take the instrument frame's
    (:data:`~qorona.config.schema.CAMERA_INSTRUMENTS`); the rest resolves as for any coronal
    perspective view (:func:`_resolve_camera`).

    Raises
    ------
    ValueError
        For the orthographic camera, or a width in R☉: the value of one native instrument pixel
        is a solid angle seen from a finite distance.
    """
    if camera_cfg.projection == "orthographic" or camera_cfg.fov is not None:
        raise ValueError(
            "the EUV render uses the perspective camera, because one instrument pixel is a solid "
            "angle seen from the observer; drop --projection orthographic and --fov, and set the "
            "width with --fov-deg or --instrument"
        )
    frame = CAMERA_INSTRUMENTS[CHANNELS[euv_cfg.channels[0]].frame]
    camera_cfg = replace(
        camera_cfg,
        projection="perspective",
        fov_deg=camera_cfg.fov_deg if camera_cfg.fov_deg is not None else frame.fov_deg,
        projection_code=camera_cfg.projection_code or frame.projection_code,
    )
    return _resolve_camera(camera_cfg, "corona", inner, outer)


def render_euv(
    plasma: PlasmaVolume,
    euv_cfg: EuvConfig,
    camera_cfg: CameraConfig,
    *,
    show_progress: bool = True,
) -> EuvResult:
    """Render synthetic EUV frames of a plasma volume from one viewpoint.

    Resolves the camera (:func:`resolve_euv_camera`) and the broadening temperature
    (:func:`resolve_tr_temperature`), then integrates the emission of every channel along each
    line of sight in one march.

    Returns
    -------
    EuvResult
        One frame per channel, in the channel unit per second and native pixel.
    """
    inner, outer = float(plasma.grid.radii[0]), float(plasma.grid.radii[-1])
    camera_cfg = resolve_euv_camera(camera_cfg, euv_cfg, inner, outer)
    return _euv_los(
        plasma,
        _camera(camera_cfg),
        euv_cfg.channels,
        abundances=euv_cfg.abundances,
        step=euv_cfg.step,
        absorption=euv_cfg.absorption,
        tr_temperature=resolve_tr_temperature(plasma, euv_cfg),
        workers=euv_cfg.workers,
        show_progress=show_progress,
    )


def euv_provenance(
    input_cfg: InputConfig,
    grid_cfg: GridConfig,
    euv_cfg: EuvConfig,
    camera_cfg: CameraConfig,
    output_cfg: OutputConfig,
    result: EuvResult,
) -> dict[str, Any]:
    """Assemble the EUV render's provenance: input, plasma grid, recipe, camera and output.

    The EUV block carries the configuration, the broadening temperature that was applied, the
    atomic recipe of the response tables (CHIANTI version, abundance file, ionization
    equilibrium, response files), the instrument effects left out, and the level of each frame.
    """
    first = next(iter(result.tables.values()))
    recipe = first.provenance
    frames = {}
    for name, frame in result.frames.items():
        positive = frame[np.isfinite(frame) & (frame > 0.0)]
        median, high = np.percentile(positive, [50.0, 99.0]) if positive.size else (0.0, 0.0)
        frames[name] = {"unit": result.unit(name), "median": float(median), "p99": float(high)}
    return {
        "format": "qorona-euv-v1",
        "version": __version__,
        "input": _input_provenance(input_cfg),
        "plasma": {**grid_cfg.to_provenance(), **result.plasma},
        "euv": {
            **euv_cfg.to_provenance(),
            "instrument": first.channel.instrument,
            "tr_temperature": result.tr_temperature,
            "recipe": {
                "chianti_version": recipe.get("chianti_version"),
                "abundance_file": recipe.get("abundance_file"),
                "ionization_equilibrium": recipe.get("ionization_equilibrium"),
                "n_h_over_n_e": recipe.get("n_h_over_n_e"),
                "continuum": recipe.get("continuum"),
                "responses": {
                    name: table.provenance.get("response", {}).get("file")
                    for name, table in result.tables.items()
                },
            },
            "instrument_effects": "none (no degradation, no point-spread function)",
            "frames": frames,
        },
        "camera": camera_cfg.to_provenance(),
        "output": output_cfg.to_provenance(),
    }


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
        volume,
        cfg.camera,
        cfg.weighting,
        render_cfg,
        category=field.category,
        density=field.density,
        show_progress=show_progress,
    )


# --- Provenance assembly -----------------------------------------------------------------------


def _ephemeris(timestamp: str | None) -> dict[str, object]:
    """Return ``{"cr", "jd"}`` derived from ``timestamp``, or an empty mapping if it is ``None``."""
    if timestamp is None:
        return {}
    return {"cr": derive_cr(timestamp), "jd": derive_jd(timestamp)}


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


def _input_provenance(input_cfg: InputConfig) -> dict[str, Any]:
    """The input block every provenance record carries: the config's fields with the model
    resolved through the reader registry, its solution category, the content hash, and the
    derived CR/JD."""
    model = resolve_model(input_cfg.path, model=input_cfg.model)
    return {
        **input_cfg.to_provenance(),
        "model": model,
        "category": category_of(model),
        "content_hash": content_hash(input_cfg.path),
        **_ephemeris(input_cfg.timestamp),
    }


def _field_provenance(grid_cfg: GridConfig, field: Field | None = None) -> dict[str, Any]:
    """The field-grid block of a provenance record, with the resampler recorded *resolved*
    (never ``auto``) and the shell radii recorded *resolved* (never the ``None`` sentinels) from the
    built field, whose grid carries the shell the category resolution settled on."""
    block = grid_cfg.to_provenance()
    resolved = getattr(field, "resampler_name", None)
    if resolved is not None:
        block["resampler"] = resolved
    if isinstance(field, SampledField):
        block["inner_radius"] = float(field.grid.radii[0])
        block["outer_radius"] = float(field.grid.radii[-1])
    return block


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
        "input": _input_provenance(input_cfg),
        "field": {**_field_provenance(grid_cfg, field), "normalization": field.normalization},
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
    frame_override: str | None = None,
) -> dict[str, Any]:
    """Extend a build provenance with the render's camera / weighting / render / output and metrics.

    A render off a baked volume recovers the date for the stamp from ``build_prov`` without
    re-supplying it; an explicit ``timestamp_override`` re-derives CR/JD for the stamp, and a
    ``frame_override`` declares the solution frame for a volume baked without one (or corrects it).
    """
    prov = json.loads(json.dumps(build_prov))  # deep copy
    if timestamp_override is not None:
        prov["input"] = {
            **prov.get("input", {}),
            "timestamp": timestamp_override,
            **_ephemeris(timestamp_override),
        }
    if frame_override is not None:
        prov["input"] = {**prov.get("input", {}), "frame": frame_override}
    covered = result.coverage[result.coverage > 0.0]
    prov["camera"] = camera_cfg.to_provenance()
    prov["weighting"] = weighting_cfg.to_provenance()
    prov["render"] = {
        **render_cfg.to_provenance(),
        "backend": _resolved_backend(render_cfg.device),
        "preset": result.preset_name,
        "display_mode": result.display_mode,
        "mean_coverage": float(np.mean(covered)) if covered.size else 0.0,
        "lower_clamped_fraction": result.lower_clamped_fraction,
        "upper_clamped_fraction": result.upper_clamped_fraction,
    }
    prov["output"] = output_cfg.to_provenance()
    return prov


def qmap_provenance(
    qmap_cfg: QMapConfig,
    output_cfg: OutputConfig,
    build_provenance: dict[str, Any],
    result: QMap,
) -> dict[str, Any]:
    """Assemble the Q-map provenance: the source volume's build record + the shell parameters.

    The Q-map is a slice of the cached volume (no re-ingest, no trace), so it carries the volume's
    own build provenance as ``source_volume`` and adds the shell parameters and coverage metrics.
    """
    return {
        "format": "qorona-qmap-v1",
        "version": __version__,
        "source_volume": build_provenance,
        "qmap": {
            **qmap_cfg.to_provenance(),
            "coverage": result.coverage,
            "sub_floor_fraction": result.sub_floor_fraction,
        },
        "output": output_cfg.to_provenance(),
    }


def fieldlines_provenance(
    input_cfg: InputConfig,
    grid_cfg: GridConfig,
    fieldlines_cfg: FieldLinesConfig,
    camera_cfg: CameraConfig,
    output_cfg: OutputConfig,
    result: FieldLineImage,
    *,
    field: Field | None = None,
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
        "input": _input_provenance(input_cfg),
        "field": _field_provenance(grid_cfg, field),
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
    """Assemble the export's provenance: the mapping the end-of-run summary reads.

    Mirrors :func:`fieldlines_provenance` (minus camera and image output): the input (path +
    content hash + derived CR/JD), the field grid, and the export parameters with the resolved
    seed radius and the line tallies. SunJSON has no metadata slot, so the file carries only the
    input timestamp (the layer's ``time``); the summary is the run's provenance record.
    """
    seed_radius = (
        export_cfg.seed_radius if export_cfg.seed_radius is not None else field.domain.inner_radius
    )
    return {
        "version": __version__,
        "input": _input_provenance(input_cfg),
        "field": _field_provenance(grid_cfg, field),
        "export": {
            **export_cfg.to_provenance(),
            "seed_radius": seed_radius,
            "n_open": int(lines.is_open.sum()),
            "n_closed": int(lines.is_closed.sum()),
            "n_incomplete": int(lines.is_incomplete.sum()),
        },
    }


def _brightness_metrics(result: BrightnessResult) -> dict[str, Any]:
    """Return the image metrics the brightness provenance and summary carry."""
    positive_pb = result.polarized[result.polarized > 0.0]
    decades = float(np.log10(positive_pb.max() / positive_pb.min())) if positive_pb.size else 0.0
    positive_total = result.total > 0.0
    median_polarization = (
        float(np.median(result.polarization()[positive_total]))
        if bool(np.any(positive_total))
        else 0.0
    )
    return {
        "median_polarization": median_polarization,
        "pb_decades": decades,
        "unit": result.unit,
    }


def brightness_provenance(
    input_cfg: InputConfig,
    grid_cfg: GridConfig,
    brightness_cfg: BrightnessConfig,
    camera_cfg: CameraConfig,
    output_cfg: OutputConfig,
    result: BrightnessResult,
    *,
    field: Field | None = None,
) -> dict[str, Any]:
    """Assemble the white-light render's provenance for a render straight off a solution.

    Mirrors :func:`fieldlines_provenance`: the input (path + content hash + derived CR/JD), the
    field grid, the brightness parameters with the image's polarization and dynamic-range metrics,
    the camera, and the output. JSON-safe by construction (no volume artifact, so its own format tag
    distinguishes it).
    """
    return {
        "format": "qorona-brightness-v1",
        "version": __version__,
        "input": _input_provenance(input_cfg),
        "field": _field_provenance(grid_cfg, field),
        "brightness": {**brightness_cfg.to_provenance(), **_brightness_metrics(result)},
        "camera": camera_cfg.to_provenance(),
        "output": output_cfg.to_provenance(),
    }


def volume_brightness_provenance(
    build_prov: dict[str, Any],
    brightness_cfg: BrightnessConfig,
    camera_cfg: CameraConfig,
    output_cfg: OutputConfig,
    result: BrightnessResult,
    *,
    timestamp_override: str | None = None,
    frame_override: str | None = None,
) -> dict[str, Any]:
    """Assemble the white-light render's provenance for a render off a baked volume artifact,
    with an optional ``timestamp_override`` re-deriving CR/JD for the stamp, and a
    ``frame_override`` declaring the solution frame for a volume baked without one (or correcting
    it).

    Mirrors :func:`render_provenance`: the volume's build provenance rides whole (so the stamp and
    summary read the original input and field blocks without re-supplying them), extended with the
    brightness parameters and image metrics, the camera, and the output.
    """
    prov = json.loads(json.dumps(build_prov))  # deep copy
    if timestamp_override is not None:
        prov["input"] = {
            **prov.get("input", {}),
            "timestamp": timestamp_override,
            **_ephemeris(timestamp_override),
        }
    if frame_override is not None:
        prov["input"] = {**prov.get("input", {}), "frame": frame_override}
    prov["brightness"] = {**brightness_cfg.to_provenance(), **_brightness_metrics(result)}
    prov["camera"] = camera_cfg.to_provenance()
    prov["output"] = output_cfg.to_provenance()
    return prov


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
    """Persist a Q⊥ volume (+ optional polarity, radial sign, density) and provenance to a ``.npz``.

    The padded ``log_q_perp`` payload is stored ``float32`` by default, lossless relative to the
    engine's ``rtol ≈ 1e-4`` (storing log Q⊥ keeps the precision uniform across the dynamic range,
    NaN gaps bit-exact), and DEFLATE-compressed by default. ``dtype="float64"`` gives a
    bit-exact checkpoint; ``compress=False`` skips the one-time compression. The file is written
    through an open handle so the user's exact suffix (``.qor``) is preserved rather than numpy's
    ``.npz`` default.

    When a ``density`` is supplied (the Thomson / brightness branch) its padded payload and grid
    ride the same archive, so a separate ``render`` has ``Nₑ`` for the optional pB weighting; an
    artifact baked without it loads back with no density. If the reader declared the density's
    physical units, the archive stores them too, so ``wl`` can report absolute brightness.

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
    # float ``dtype`` (lossless, small); it is absent when the volume carries none.
    if volume.polarity is not None:
        arrays["polarity"] = volume.polarity.astype(np.int8)
    # The local radial-field sign rides like the density channel: an int8 array on its own grid spec
    # (the field grid, coarser than the volume grid), absent when the volume carries none.
    if volume.radial_sign is not None and volume.radial_sign_grid is not None:
        arrays["radial_sign"] = volume.radial_sign.astype(np.int8)
        meta["radial_sign_grid"] = _grid_spec(volume.radial_sign_grid)
    if density is not None:
        arrays["density"] = density.density.astype(cast)
        meta["density_grid"] = _grid_spec(density.grid)
        # The physical units, as plain numbers: Nₑ of one stored value (m⁻³) and the mean molecular
        # weight it uses (both absent when unknown), and the model length unit (m).
        if density.unit is not None:
            meta["density_unit_m3"] = float(density.unit.to(u.m**-3))
            meta["mean_molecular_weight"] = density.mean_molecular_weight
        meta["length_unit_m"] = float(density.length_unit.to(u.m))
    # Select compressed vs plain savez.
    saver: Any = np.savez_compressed if compress else np.savez
    with open(path, "wb") as handle:
        saver(handle, meta=np.array(json.dumps(meta)), **arrays)


def load_volume(
    path: str | Path,
) -> tuple[QPerpVolume, DensityVolume | None, dict[str, Any]]:
    """Load a Q⊥ volume, its electron density (if any), and its build provenance from an artifact.

    The stored arrays (``float32`` or ``float64``) are restored to the ``float64`` the render kernel
    consumes; the grids are reconstructed from their specs, so the loaded volume renders identically
    to the one that was baked. ``density`` is ``None`` for an artifact baked without it; it carries
    the recorded density unit, length unit and μ, and an artifact without the unit record loads as
    a relative density. The polarity and radial-sign channels (int8 on disk) are restored to
    ``float32``, or ``None`` when absent.

    Returns
    -------
    volume : QPerpVolume
        The reconstructed volume.
    density : DensityVolume or None
        The reconstructed electron-density volume, or ``None`` if the artifact carries none.
    provenance : dict
        The build provenance recorded at bake time.

    Raises
    ------
    ValueError
        If ``path`` is not a readable Qorona volume artifact, or was written in a volume
        format this version does not read.
    """
    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError(
            f"{path} is not a readable Qorona volume artifact (.qor): {error}"
        ) from error
    with archive:
        if "meta" not in archive.files:
            raise ValueError(
                f"{path} is not a Qorona volume artifact: it carries no 'meta' record. "
                "Build one with `qorona build`."
            )
        meta = json.loads(str(archive["meta"].item()))
        stored_format = meta.get("format")
        if stored_format != _ARTIFACT_FORMAT:
            raise ValueError(
                f"{path} was written in volume format {stored_format!r}, but this Qorona reads "
                f"{_ARTIFACT_FORMAT!r}; rebuild it with `qorona build`."
            )
        log_q_perp = np.ascontiguousarray(archive["log_q_perp"], dtype=np.float64)
        polarity = None
        if "polarity" in archive.files:
            polarity = np.ascontiguousarray(archive["polarity"], dtype=np.float32)
        radial_sign = None
        if "radial_sign" in archive.files and "radial_sign_grid" in meta:
            radial_sign = np.ascontiguousarray(archive["radial_sign"], dtype=np.float32)
        density = None
        if "density" in archive.files and "density_grid" in meta:
            # Artifacts without the unit records load as relative densities in R☉ coordinates.
            per_value = meta.get("density_unit_m3")
            density = DensityVolume(
                grid=_grid_from_spec(meta["density_grid"]),
                density=np.ascontiguousarray(archive["density"], dtype=np.float64),
                unit=u.Unit(per_value * u.m**-3) if per_value is not None else None,
                length_unit=u.Unit(meta.get("length_unit_m", u.R_sun.to(u.m)) * u.m),
                mean_molecular_weight=meta.get("mean_molecular_weight"),
            )
    grid = _grid_from_spec(meta["grid"])
    radial_sign_grid = (
        _grid_from_spec(meta["radial_sign_grid"]) if radial_sign is not None else None
    )
    return (
        QPerpVolume(
            grid=grid,
            log_q_perp=log_q_perp,
            polarity=polarity,
            radial_sign=radial_sign,
            radial_sign_grid=radial_sign_grid,
        ),
        density,
        meta["provenance"],
    )
