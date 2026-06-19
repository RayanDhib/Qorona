"""The ``qorona`` CLI: ``build`` / ``render`` / ``run`` / ``fieldlines`` / ``export-lines`` /
``info``.

A :mod:`click` group wired to the shared :mod:`qorona.console` Rich surface, so progress and styling
stay uniform. It splits the two cost axes of the pipeline into separate commands: ``build`` bakes
the viewpoint-independent Q⊥ volume **once** (the minutes-scale stage, where resolution / seeding /
supersampling sweeps live), ``render`` integrates a baked volume for any camera / preset (seconds,
where viewpoint / weighting sweeps live), and ``run`` chains both; ``fieldlines`` draws the
field-line view, ``export-lines`` serialises traced field lines for external tools, and ``info``
inspects a solution.

Every flag populates the typed :mod:`qorona.config` schema (the single source of truth for defaults
and validation); a flag left unset defers to the dataclass, so the help text's stated defaults are
documentation, not a second behavioural source (the two documented exceptions: the single-axis
image-dimension fallback and the volume-cache write options). After any command that produces a
result, a polished end-of-run summary prints the run's parameters and quantitative metrics, the
printed counterpart of the on-image stamp.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click

from qorona import __version__, pipeline
from qorona.config import (
    ANNOTATE_POSITIONS,
    BRIGHTNESS_FRAMES,
    BRIGHTNESS_SCALINGS,
    BRIGHTNESS_TREATMENTS,
    CLOSED_TREATMENTS,
    DEVICE_MODES,
    DISPLAY_MODES,
    EXPORT_FORMATS,
    FIELDLINE_COLOUR,
    FIELDLINE_SEEDING,
    FIELDLINE_SHOW,
    OCCULT_MODES,
    POLARITY_MODES,
    PRECISION_MODES,
    RESAMPLERS,
    SPACING_LAWS,
    VOLUME_BUILDERS,
    WEIGHTING_PRESETS,
    BrightnessConfig,
    CameraConfig,
    ExportConfig,
    FieldLinesConfig,
    GridConfig,
    InputConfig,
    OutputConfig,
    RenderConfig,
    VolumeConfig,
    WeightingConfig,
)
from qorona.console import console, print_step, print_success
from qorona.io.fieldlines_export import write_fieldlines_json
from qorona.io.output import (
    export_brightness,
    write_brightness,
    write_fieldlines,
    write_outputs,
)

#: Default image dimension used only when exactly one of ``--width`` / ``--height`` is supplied; the
#: full default ``pixels`` otherwise lives once on :class:`~qorona.config.CameraConfig`.
_DEFAULT_DIMENSION = 1024


# --- Reusable option groups --------------------------------------------------------------------
# Defined once and shared across the commands so the flag surface never drifts between them.
# Every option default is ``None`` (``--quiet``, a display-only flag, excepted) so the typed
# schema remains the sole source of defaults.


def _compose(*options: Callable[[Callable], Callable]) -> Callable[[Callable], Callable]:
    """Apply a sequence of click option decorators in declaration order."""

    def wrap(func: Callable) -> Callable:
        for option in reversed(options):
            func = option(func)
        return func

    return wrap


_input_options = _compose(
    click.option(
        "--model", default=None, help="Solution model; inferred from the extension if unset."
    ),
    click.option(
        "--timestamp",
        default=None,
        help="UTC ISO-8601 observation time → Carrington rotation + Julian date.",
    ),
    click.option(
        "--variables",
        default=None,
        help="Comma-separated state-variable names (advanced; reader override).",
    ),
)
_grid_options = _compose(
    click.option("--n-r", type=int, default=None, help="Radial field-grid nodes (default 192)."),
    click.option(
        "--n-theta", type=int, default=None, help="Colatitude field-grid nodes (default 180)."
    ),
    click.option(
        "--n-phi", type=int, default=None, help="Azimuth field-grid nodes, even (default 360)."
    ),
    click.option(
        "--inner-radius",
        type=float,
        default=None,
        help="Inner shell radius in R_sun (default 1.0).",
    ),
    click.option(
        "--outer-radius",
        type=float,
        default=None,
        help="Outer shell radius in R_sun (default 12.5).",
    ),
    click.option(
        "--spacing",
        type=click.Choice(SPACING_LAWS),
        default=None,
        help="Radial spacing law (default logarithmic).",
    ),
    click.option(
        "--resampler",
        type=click.Choice(RESAMPLERS),
        default=None,
        help="Cell→grid resampler (default knn-mls).",
    ),
)
#: Sharp-turn guard knobs, shared (hidden/advanced) by `build`/`run` and `fieldlines`. The
#: defaults apply automatically; `--max-turn-angle 0` disables the guard.
_turn_guard_options = (
    click.option(
        "--max-turn-angle",
        type=float,
        default=None,
        hidden=True,
        help="Sharp-turn guard: terminate a line that turns more than this many degrees in one "
        "step in the weak-field outer corona, a deflection at a null (default 45; 0 disables).",
    ),
    click.option(
        "--turn-guard-radius",
        type=float,
        default=None,
        hidden=True,
        help="Sharp-turn guard: fire only above this radius in R_sun (default 2.0).",
    ),
    click.option(
        "--turn-guard-weak-fraction",
        type=float,
        default=None,
        hidden=True,
        help="Sharp-turn guard: fire only where |B| is below this fraction of the field's peak |B| "
        "(default 1e-5).",
    ),
    click.option(
        "--min-turns",
        type=int,
        default=None,
        hidden=True,
        help="Sharp-turn guard: number of qualifying sharp turns that triggers termination; "
        "occasional null grazes are kept (default 1 for volume builds, 3 for fieldlines).",
    ),
)
_volume_options = _compose(
    click.option(
        "--builder",
        type=click.Choice(VOLUME_BUILDERS),
        default=None,
        help="Q⊥ volume builder: paint (fast production fill), per-voxel (every voxel traced "
        "to its feet; complete coverage, cost grows with voxels), reference (validation "
        "ground truth). Default paint.",
    ),
    click.option(
        "--resolution-factor",
        type=int,
        default=None,
        help="Volume grid = field grid refined by this factor (default 2).",
    ),
    click.option(
        "--supersample",
        type=int,
        default=None,
        help="Boundary/seed angular supersampling (default 4).",
    ),
    click.option(
        "--paint-step",
        type=float,
        default=None,
        help="Paint along-line pitch as a fraction of the cell extent (default 0.5).",
    ),
    click.option(
        "--closed",
        type=click.Choice(CLOSED_TREATMENTS),
        default=None,
        help="Closed-loop polarity: neutral (feet cancel to 0) or dominant (default neutral).",
    ),
    click.option(
        "--rtol",
        type=float,
        default=None,
        help="Tracer/transport relative tolerance (default 1e-4).",
    ),
    click.option(
        "--cfl", type=float, default=None, help="CFL step ceiling, 0<cfl<1 (default 0.5)."
    ),
    click.option(
        "--max-steps", type=int, default=None, help="Per-half-line step guard (default 10000)."
    ),
    click.option(
        "--max-reversals",
        type=int,
        default=None,
        hidden=True,
        help="Stall guard: terminate a line after this many >90° direction reversals, a line "
        "trapped at a weak-field null (default 8; 0 disables).",
    ),
    click.option(
        "--precision",
        type=click.Choice(PRECISION_MODES),
        default=None,
        help="CUDA kernel precision: mixed (f32 field interpolation, f64 elsewhere; default), "
        "float64 (all-double reference), float32 (experimental fully-float32 painter). GPU only.",
    ),
    *_turn_guard_options,
)
_cache_options = _compose(
    click.option(
        "--cache-dtype",
        type=click.Choice(("float32", "float64")),
        default=None,
        help="Stored volume dtype (default float32).",
    ),
    click.option(
        "--compress/--no-compress",
        "compress",
        default=None,
        help="DEFLATE-compress the volume artifact (default on).",
    ),
)
_camera_options = _compose(
    click.option(
        "--longitude",
        type=float,
        default=None,
        help="Sub-observer heliographic longitude in degrees (default 0).",
    ),
    click.option(
        "--latitude",
        type=float,
        default=None,
        help="Sub-observer heliographic latitude in degrees (default 0).",
    ),
    click.option(
        "--roll",
        type=float,
        default=None,
        help="Camera roll about the line of sight in degrees (default 0).",
    ),
    click.option(
        "--fov", type=float, default=None, help="Field of view (full width) in R_sun (default 25)."
    ),
    click.option("--width", type=int, default=None, help="Image width in pixels (default 1024)."),
    click.option("--height", type=int, default=None, help="Image height in pixels (default 1024)."),
)
_weighting_options = _compose(
    click.option(
        "--preset",
        type=click.Choice(WEIGHTING_PRESETS),
        default=None,
        help="Geometric depth-weighting preset (default large-fov).",
    ),
)
_render_options = _compose(
    click.option(
        "--display",
        type=click.Choice(DISPLAY_MODES),
        default=None,
        help="Depth-colour reconstruction (default balanced).",
    ),
    click.option(
        "--polarity-mode",
        type=click.Choice(POLARITY_MODES),
        default=None,
        help="Colour by magnetic polarity: hue=warm outward/cool inward (default none).",
    ),
    click.option(
        "--occult",
        type=click.Choice(OCCULT_MODES),
        default=None,
        help="Occultation mode (default eclipse).",
    ),
    click.option(
        "--r-occult", type=float, default=None, help="Body/occulter radius in R_sun (default 1.0)."
    ),
    click.option(
        "--occult-softness",
        type=float,
        default=None,
        help="Eclipse-edge feather in R_sun (default 0.03).",
    ),
    click.option(
        "--clamp",
        type=float,
        nargs=2,
        default=None,
        help="Display log10 Q⊥ clamp 'LOW HIGH' (default log10(2) 7.0).",
    ),
    click.option(
        "--raw/--no-raw",
        "raw",
        default=None,
        help="Keep the sub-floor tail (skip the lower clamp).",
    ),
    click.option(
        "--step",
        type=float,
        default=None,
        help="Line-of-sight sample spacing in R_sun (default 0.02).",
    ),
    click.option(
        "--percentiles",
        type=float,
        nargs=2,
        default=None,
        help="Per-channel stretch percentiles 'LOW HIGH' (default 1.0 99.5).",
    ),
)
_annotate_options = _compose(
    click.option(
        "--annotate/--no-annotate",
        "annotate",
        default=None,
        help="Burn the provenance stamp onto the PNG (default on).",
    ),
    click.option(
        "--annotate-position",
        type=click.Choice(ANNOTATE_POSITIONS),
        default=None,
        help="Stamp corner (default bottom-left).",
    ),
)
_output_options = _compose(
    click.option(
        "--grayscale/--no-grayscale",
        "grayscale",
        default=None,
        help="Also write the grayscale measurement PNG (default off).",
    ),
    _annotate_options,
)
_fieldlines_options = _compose(
    click.option(
        "--seeding",
        type=click.Choice(FIELDLINE_SEEDING),
        default=None,
        help="Seeding: limb (front loops + limb fan) or uniform sphere (default limb).",
    ),
    click.option(
        "--seeds",
        "n_seeds",
        type=int,
        default=None,
        help="Fibonacci seed budget: full sphere (uniform) or front-loop source (limb) "
        "(default 1500).",
    ),
    click.option(
        "--limb-seeds",
        type=int,
        default=None,
        help="Seeds around the limb ring (the open fan; limb seeding) (default 375).",
    ),
    click.option(
        "--front-loop-length",
        type=float,
        default=None,
        help="Max arc length R_sun for a kept near-side closed loop (limb seeding) (default 1.2).",
    ),
    click.option(
        "--colour",
        type=click.Choice(FIELDLINE_COLOUR),
        default=None,
        help="Line colour: rainbow (per-line hue) or polarity (B_r sign) (default polarity).",
    ),
    click.option(
        "--magnetogram/--no-magnetogram",
        "magnetogram",
        default=None,
        help="Render the disk as a B_r magnetogram from the data (default on).",
    ),
    click.option(
        "--line-width", type=float, default=None, help="Drawn line width in pixels (default 1.5)."
    ),
    click.option(
        "--show",
        type=click.Choice(FIELDLINE_SHOW),
        default=None,
        help="Which lines to draw: all / open / closed (default all).",
    ),
    click.option(
        "--depth-fade",
        type=float,
        default=None,
        help="Dim far-side lines by up to this fraction, 0-1 (default 0.4).",
    ),
    click.option(
        "--rtol", type=float, default=None, help="Tracer relative tolerance (default 1e-4)."
    ),
    click.option(
        "--cfl", type=float, default=None, help="CFL step ceiling, 0<cfl<1 (default 0.5)."
    ),
    click.option(
        "--max-steps", type=int, default=None, help="Per-half-line step guard (default 10000)."
    ),
    *_turn_guard_options,
)
_export_options = _compose(
    click.option(
        "--seeds",
        "seed_grid",
        type=int,
        nargs=2,
        default=None,
        help="Seed grid resolution 'N_THETA N_PHI' on the seed sphere (default 100 100).",
    ),
    click.option(
        "--seed-radius",
        type=float,
        default=None,
        help="Seed sphere radius in R_sun (default: the inner boundary).",
    ),
    click.option(
        "--rtol", type=float, default=None, help="Tracer relative tolerance (default 1e-4)."
    ),
    click.option(
        "--cfl", type=float, default=None, help="CFL step ceiling, 0<cfl<1 (default 0.5)."
    ),
    click.option(
        "--max-steps", type=int, default=None, help="Per-half-line step guard (default 10000)."
    ),
    *_turn_guard_options,
)
_brightness_options = _compose(
    click.option(
        "--frame",
        type=click.Choice(BRIGHTNESS_FRAMES),
        default=None,
        help="Brightness frame: polarized (pB) or total (white-light) (default polarized).",
    ),
    click.option(
        "--treatment",
        type=click.Choice(BRIGHTNESS_TREATMENTS),
        default=None,
        help="Display treatment (applied to the selected --frame): raw / radial rho-power filter / "
        "newkirk radial vignette / mgn fine-structure enhancement (default raw).",
    ),
    click.option(
        "--radial-power",
        type=float,
        default=None,
        help="Exponent for --treatment radial: brightness is scaled by rho**power (rho = "
        "plane-of-sky radius in R_sun); below the radial-falloff power it lifts the outer corona "
        "while staying bright near the limb (default 3.0).",
    ),
    click.option(
        "--export",
        "export_formats",
        type=click.Choice(EXPORT_FORMATS),
        multiple=True,
        help="Also write the raw data (both pB and total B + plane-of-sky coordinates) to this "
        "format beside the PNG; repeatable. Currently only npz.",
    ),
    click.option(
        "--limb-darkening",
        "limb_darkening",
        type=float,
        default=None,
        help="Thomson limb-darkening coefficient u, 0-1 (default 0.6).",
    ),
    click.option(
        "--crossover",
        type=float,
        default=None,
        help="Closed-form→asymptotic coefficient crossover radius in R_sun (default 10).",
    ),
    click.option(
        "--step",
        type=float,
        default=None,
        help="Line-of-sight sample spacing in R_sun (default 0.02).",
    ),
    click.option(
        "--occult",
        type=click.Choice(OCCULT_MODES),
        default=None,
        help="Occultation mode (default eclipse).",
    ),
    click.option(
        "--r-occult", type=float, default=None, help="Body/occulter radius in R_sun (default 1.0)."
    ),
    click.option(
        "--occult-softness",
        type=float,
        default=None,
        help="Eclipse-edge feather in R_sun (default 0.03).",
    ),
    click.option(
        "--scaling",
        type=click.Choice(BRIGHTNESS_SCALINGS),
        default=None,
        help="Intensity stretch: log or linear (default log; linear for mgn).",
    ),
    click.option(
        "--percentiles",
        type=float,
        nargs=2,
        default=None,
        help="Per-image stretch percentiles 'LOW HIGH' (default 1.0 99.5).",
    ),
)
_workers_option = click.option(
    "--workers",
    type=int,
    default=None,
    help="Worker threads for the numba kernels (default all cores).",
)
_device_option = click.option(
    "--device",
    type=click.Choice(DEVICE_MODES),
    default=None,
    help="Compute backend for the volume build: auto (GPU when present, else CPU), gpu (force; "
    "errors if absent), cpu. Renders always run on the CPU. Default auto.",
)
_quiet_option = click.option(
    "--quiet", is_flag=True, default=False, help="Suppress progress bars and the spinner."
)


# --- Config construction from CLI kwargs -------------------------------------------------------


def _present(kw: dict[str, Any], *names: str) -> dict[str, Any]:
    """Return the ``kw`` entries that are not ``None`` (so the schema supplies the defaults)."""
    return {name: kw[name] for name in names if kw.get(name) is not None}


def _input_config(kw: dict[str, Any]) -> InputConfig:
    variables = tuple(kw["variables"].split(",")) if kw.get("variables") else None
    return InputConfig(
        path=kw["input_path"],
        model=kw.get("model"),
        timestamp=kw.get("timestamp"),
        variables=variables,
    )


def _grid_config(kw: dict[str, Any]) -> GridConfig:
    return GridConfig(
        **_present(
            kw, "n_r", "n_theta", "n_phi", "inner_radius", "outer_radius", "spacing", "resampler"
        )
    )


def _volume_config(kw: dict[str, Any], workers: int | None) -> VolumeConfig:
    fields = _present(
        kw,
        "builder",
        "resolution_factor",
        "supersample",
        "paint_step",
        "closed",
        "rtol",
        "cfl",
        "max_steps",
        "max_reversals",
        "max_turn_angle",
        "turn_guard_radius",
        "turn_guard_weak_fraction",
        "min_turns",
        "device",
        "precision",
    )
    if workers is not None:
        fields["workers"] = workers
    return VolumeConfig(**fields)


def _camera_config(kw: dict[str, Any]) -> CameraConfig:
    fields = _present(kw, "longitude", "latitude", "roll", "fov")
    height, width = kw.get("height"), kw.get("width")
    if height is not None or width is not None:
        fields["pixels"] = (
            height if height is not None else _DEFAULT_DIMENSION,
            width if width is not None else _DEFAULT_DIMENSION,
        )
    return CameraConfig(**fields)


def _weighting_config(kw: dict[str, Any]) -> WeightingConfig:
    return WeightingConfig(**_present(kw, "preset"))


def _render_config(kw: dict[str, Any], workers: int | None) -> RenderConfig:
    fields = _present(
        kw, "display", "occult", "r_occult", "occult_softness", "step", "polarity_mode", "device"
    )
    if kw.get("clamp") is not None:
        fields["clamp"] = tuple(kw["clamp"])
    if kw.get("percentiles") is not None:
        fields["percentiles"] = tuple(kw["percentiles"])
    if kw.get("raw") is not None:
        fields["raw"] = kw["raw"]
    if workers is not None:
        fields["workers"] = workers
    return RenderConfig(**fields)


def _save_options(kw: dict[str, Any]) -> dict[str, Any]:
    """Resolve the volume-cache write options (``dtype`` / ``compress``) from the CLI kwargs.

    These two go straight to :func:`~qorona.pipeline.save_volume` (not through the schema), so their
    defaults are resolved here: ``float32`` storage with compression on, lossless relative to the
    engine's tolerance, and roughly halving the artifact on disk.
    """
    return {
        "dtype": kw.get("cache_dtype") or "float32",
        "compress": kw["compress"] if kw.get("compress") is not None else True,
    }


def _output_config(kw: dict[str, Any], output_path: Path) -> OutputConfig:
    fields: dict[str, Any] = {"path": output_path}
    if kw.get("grayscale") is not None:
        fields["save_grayscale"] = kw["grayscale"]
    if kw.get("annotate") is not None:
        fields["annotate"] = kw["annotate"]
    if kw.get("annotate_position") is not None:
        fields["annotate_position"] = kw["annotate_position"]
    if kw.get("export_formats"):
        fields["export_formats"] = tuple(kw["export_formats"])
    return OutputConfig(**fields)


def _fieldlines_config(kw: dict[str, Any], workers: int | None) -> FieldLinesConfig:
    fields = _present(
        kw,
        "seeding",
        "n_seeds",
        "limb_seeds",
        "front_loop_length",
        "colour",
        "show",
        "line_width",
        "depth_fade",
        "rtol",
        "cfl",
        "max_steps",
        "max_turn_angle",
        "turn_guard_radius",
        "turn_guard_weak_fraction",
        "min_turns",
    )
    if kw.get("magnetogram") is not None:
        fields["magnetogram"] = kw["magnetogram"]
    if workers is not None:
        fields["workers"] = workers
    return FieldLinesConfig(**fields)


def _export_config(kw: dict[str, Any], workers: int | None) -> ExportConfig:
    fields = _present(
        kw,
        "seed_radius",
        "rtol",
        "cfl",
        "max_steps",
        "max_turn_angle",
        "turn_guard_radius",
        "turn_guard_weak_fraction",
        "min_turns",
    )
    if kw.get("seed_grid") is not None:
        fields["n_theta"], fields["n_phi"] = kw["seed_grid"]
    if workers is not None:
        fields["workers"] = workers
    return ExportConfig(**fields)


def _brightness_config(kw: dict[str, Any], workers: int | None) -> BrightnessConfig:
    fields = _present(
        kw,
        "frame",
        "treatment",
        "radial_power",
        "crossover",
        "step",
        "occult",
        "r_occult",
        "occult_softness",
        "scaling",
    )
    if kw.get("limb_darkening") is not None:
        fields["u"] = kw["limb_darkening"]
    if kw.get("percentiles") is not None:
        fields["percentiles"] = tuple(kw["percentiles"])
    if workers is not None:
        fields["workers"] = workers
    return BrightnessConfig(**fields)


# --- The command group -------------------------------------------------------------------------


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "-V", "--version", prog_name="qorona")
def main() -> None:
    """Qorona: synthetic coronal imagery from global MHD solutions.

    Render the line-of-sight magnetic squashing factor Q⊥ of a coronal MHD solution into
    eclipse-like imagery. Bake the viewpoint-independent volume once with `build`, then render any
    number of viewpoints cheaply with `render`; `run` does both in one shot, `fieldlines` draws
    the field-line view, `info` inspects a file.
    """
    # The numba CUDA dispatcher warns about low-occupancy launches on the one-seed kernel warm-up
    # and the final partial chunks; both are deliberate, so the warning is noise on the CLI.
    try:
        from numba.core.errors import NumbaPerformanceWarning

        warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
    except ImportError:
        pass


@main.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Destination volume artifact (.qor / .npz).",
)
@_input_options
@_grid_options
@_volume_options
@_cache_options
@_workers_option
@_device_option
@_quiet_option
def build(
    input_path: Path,
    output_path: Path,
    workers: int | None,
    device: str | None,
    quiet: bool,
    **kw: Any,
) -> None:
    """Bake the viewpoint-independent Q⊥ volume from a solution to a cache file.

    The minutes-scale stage: read → resample → Q⊥ volume, written to a dependency-free .qor/.npz
    with its build provenance (input hash, derived CR/JD, every resolved parameter), so any number
    of cheap `render`s can reuse it.
    """
    kw["input_path"] = input_path
    kw["device"] = device
    show_progress = not quiet
    input_cfg = _input_config(kw)
    grid_cfg = _grid_config(kw)
    volume_cfg = _volume_config(kw, workers)

    print_step(f"Baking Q⊥ volume from [bold]{input_path.name}[/bold]")
    stage_timings: dict[str, float] = {}
    start = time.perf_counter()
    field = pipeline.build_field(
        input_cfg, grid_cfg, show_progress=show_progress, timings=stage_timings
    )
    field_time = time.perf_counter() - start
    build_start = time.perf_counter()
    volume = pipeline.build_volume(
        field, volume_cfg, grid_cfg, show_progress=show_progress, timings=stage_timings
    )
    build_time = time.perf_counter() - build_start

    provenance = pipeline.build_provenance(
        input_cfg, grid_cfg, volume_cfg, field=field, volume=volume
    )
    pipeline.save_volume(
        volume, output_path, provenance, density=field.density, **_save_options(kw)
    )
    print_success(f"Saved volume → [bold]{output_path}[/bold]")
    _print_summary(
        provenance,
        {"field": field_time, "build": build_time, **stage_timings},
        volume_path=output_path,
        header="build",
    )


@main.command()
@click.argument("volume_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Destination image (.png).",
)
@click.option(
    "--timestamp",
    default=None,
    help="Override the volume's timestamp for the stamp (re-derives CR/JD).",
)
@_camera_options
@_weighting_options
@_render_options
@_output_options
@_workers_option
@_device_option
@_quiet_option
def render(
    volume_path: Path,
    output_path: Path,
    timestamp: str | None,
    workers: int | None,
    device: str | None,
    quiet: bool,
    **kw: Any,
) -> None:
    """Render a baked Q⊥ volume to an eclipse-like image from a viewpoint.

    The seconds-scale stage: load the volume, integrate it for one camera / preset / display, write
    the PNG(s) with the on-image stamp, and print the metrics. Repeat for new viewpoints off the
    same volume.
    """
    kw["device"] = device
    show_progress = not quiet
    camera_cfg = _camera_config(kw)
    weighting_cfg = _weighting_config(kw)
    render_cfg = _render_config(kw, workers)
    output_cfg = _output_config(kw, output_path)

    print_step(f"Loading volume [bold]{volume_path.name}[/bold]")
    volume, density, build_prov = pipeline.load_volume(volume_path)
    start = time.perf_counter()
    result = pipeline.render_volume(
        volume, camera_cfg, weighting_cfg, render_cfg, density=density, show_progress=show_progress
    )
    render_time = time.perf_counter() - start

    provenance = pipeline.render_provenance(
        build_prov,
        camera_cfg,
        weighting_cfg,
        render_cfg,
        output_cfg,
        result,
        timestamp_override=timestamp,
    )
    written = write_outputs(result, output_cfg, provenance)
    print_success(f"Wrote [bold]{output_path}[/bold]")
    _print_summary(provenance, {"render": render_time}, written=written, header="render")


@main.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Destination image (.png).",
)
@click.option(
    "--save-volume",
    "save_volume_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Also persist the baked volume to this .qor/.npz.",
)
@_input_options
@_grid_options
@_volume_options
@_cache_options
@_camera_options
@_weighting_options
@_render_options
@_output_options
@_workers_option
@_device_option
@_quiet_option
def run(
    input_path: Path,
    output_path: Path,
    save_volume_path: Path | None,
    workers: int | None,
    device: str | None,
    quiet: bool,
    **kw: Any,
) -> None:
    """Run the whole pipeline in one shot: read → volume → render → image.

    The one-shot path with every flag available. Use `--save-volume` to also persist the
    intermediate so subsequent `render`s are instant.
    """
    kw["input_path"] = input_path
    kw["device"] = device
    show_progress = not quiet
    input_cfg = _input_config(kw)
    grid_cfg = _grid_config(kw)
    volume_cfg = _volume_config(kw, workers)
    camera_cfg = _camera_config(kw)
    weighting_cfg = _weighting_config(kw)
    render_cfg = _render_config(kw, workers)
    output_cfg = _output_config(kw, output_path)

    print_step(f"Running the full pipeline on [bold]{input_path.name}[/bold]")
    stage_timings: dict[str, float] = {}
    start = time.perf_counter()
    field = pipeline.build_field(
        input_cfg, grid_cfg, show_progress=show_progress, timings=stage_timings
    )
    field_time = time.perf_counter() - start
    build_start = time.perf_counter()
    volume = pipeline.build_volume(
        field, volume_cfg, grid_cfg, show_progress=show_progress, timings=stage_timings
    )
    build_time = time.perf_counter() - build_start

    build_prov = pipeline.build_provenance(
        input_cfg, grid_cfg, volume_cfg, field=field, volume=volume
    )
    if save_volume_path is not None:
        pipeline.save_volume(
            volume, save_volume_path, build_prov, density=field.density, **_save_options(kw)
        )
        print_success(f"Saved volume → [bold]{save_volume_path}[/bold]")

    render_start = time.perf_counter()
    result = pipeline.render_volume(
        volume,
        camera_cfg,
        weighting_cfg,
        render_cfg,
        density=field.density,
        show_progress=show_progress,
    )
    render_time = time.perf_counter() - render_start

    provenance = pipeline.render_provenance(
        build_prov, camera_cfg, weighting_cfg, render_cfg, output_cfg, result
    )
    written = write_outputs(result, output_cfg, provenance)
    print_success(f"Wrote [bold]{output_path}[/bold]")
    _print_summary(
        provenance,
        {"field": field_time, "build": build_time, "render": render_time, **stage_timings},
        written=written,
        header="run",
    )


@main.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@_input_options
@_quiet_option
def info(input_path: Path, quiet: bool, **kw: Any) -> None:
    """Inspect a solution's metadata (model, mesh, variables, boundaries) without rendering."""
    from rich.panel import Panel
    from rich.table import Table

    from qorona.io import read_solution

    reader_kwargs: dict[str, Any] = {}
    if kw.get("variables"):
        reader_kwargs["variables"] = tuple(kw["variables"].split(","))
    solution = read_solution(
        input_path, model=kw.get("model"), show_progress=not quiet, **reader_kwargs
    )
    meta = solution.metadata

    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", justify="right")
    table.add_column()
    table.add_row("Model", f"{meta.model} · {meta.file_format} · {meta.element_type}")
    table.add_row("Mesh", f"{solution.n_cells:,} cells · {solution.n_nodes:,} nodes")
    table.add_row("Variables", ", ".join(solution.variable_names))
    for role in ("inner", "outer"):
        boundary = solution.boundaries.get(role)
        if boundary is not None:
            table.add_row(
                f"{role.capitalize()} boundary",
                f"{boundary.source_name} · r̄ = {boundary.mean_radius.value:.3f} R_sun · "
                f"{boundary.n_faces:,} faces",
            )
    table.add_row("Normalization", meta.normalization)
    if kw.get("timestamp"):
        table.add_row(
            "Timestamp",
            f"{kw['timestamp']} UTC → CR {pipeline.derive_cr(kw['timestamp'])} / "
            f"JD {pipeline.derive_jd(kw['timestamp']):.3f}",
        )
    console.print(
        Panel(table, title=f"[bold]{input_path.name}[/bold]", border_style="cyan", expand=False)
    )


@main.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Destination image (.png).",
)
@_input_options
@_grid_options
@_fieldlines_options
@_camera_options
@_annotate_options
@_workers_option
@_quiet_option
def fieldlines(
    input_path: Path, output_path: Path, workers: int | None, quiet: bool, **kw: Any
) -> None:
    """Render the magnetic field lines of a solution from a viewpoint.

    Reads the solution, traces a bundle of field lines, and draws them in projection over the
    photosphere disk. The default eclipse-photograph look seeds the open fan on the limb with short
    closed loops on the front face (``--seeding limb``), colours open lines by their inner-foot
    ``B·r̂`` polarity (``--colour polarity``), and renders a B_r magnetogram (``--magnetogram``).
    A self-contained command: it traces the field directly and does not use a baked volume.
    """
    kw["input_path"] = input_path
    show_progress = not quiet
    input_cfg = _input_config(kw)
    grid_cfg = _grid_config(kw)
    fieldlines_cfg = _fieldlines_config(kw, workers)
    camera_cfg = _camera_config(kw)
    output_cfg = _output_config(kw, output_path)

    print_step(f"Tracing field lines from [bold]{input_path.name}[/bold]")
    start = time.perf_counter()
    field = pipeline.build_field(input_cfg, grid_cfg, show_progress=show_progress)
    field_time = time.perf_counter() - start
    render_start = time.perf_counter()
    result = pipeline.render_fieldlines(
        field, fieldlines_cfg, camera_cfg, show_progress=show_progress
    )
    render_time = time.perf_counter() - render_start

    provenance = pipeline.fieldlines_provenance(
        input_cfg, grid_cfg, fieldlines_cfg, camera_cfg, output_cfg, result
    )
    written = write_fieldlines(result, output_cfg, provenance)
    print_success(f"Wrote [bold]{output_path}[/bold]")
    _print_summary(
        provenance,
        {"field": field_time, "fieldlines": render_time},
        written=written,
        header="fieldlines",
    )


@main.command("export-lines")
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Destination field-line file (.json).",
)
@_input_options
@_grid_options
@_export_options
@_workers_option
@_quiet_option
def export_lines(
    input_path: Path, output_path: Path, workers: int | None, quiet: bool, **kw: Any
) -> None:
    """Export traced field lines of a solution to JSON for external tools.

    Reads the solution, traces field lines seeded on a uniform longitude/latitude grid
    (``--seeds``, default 100x100, on the inner boundary unless ``--seed-radius`` overrides it),
    and writes the polylines with their open/closed topology. A self-contained command: it traces
    the field directly and does not use a baked volume. The file schema is documented in
    ``qorona/io/fieldlines_export.py``.
    """
    kw["input_path"] = input_path
    show_progress = not quiet
    input_cfg = _input_config(kw)
    grid_cfg = _grid_config(kw)
    export_cfg = _export_config(kw, workers)

    print_step(f"Exporting field lines from [bold]{input_path.name}[/bold]")
    start = time.perf_counter()
    field = pipeline.build_field(input_cfg, grid_cfg, show_progress=show_progress)
    field_time = time.perf_counter() - start
    trace_start = time.perf_counter()
    lines = pipeline.export_lines(field, export_cfg, show_progress=show_progress)
    trace_time = time.perf_counter() - trace_start

    provenance = pipeline.export_provenance(input_cfg, grid_cfg, export_cfg, field, lines)
    written = write_fieldlines_json(lines, output_path, provenance)
    print_success(f"Wrote [bold]{output_path}[/bold]")
    _print_summary(
        provenance,
        {"field": field_time, "export": trace_time},
        written=[written],
        header="export-lines",
    )


@main.command(hidden=True)
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Destination image (.png).",
)
@_input_options
@_grid_options
@_brightness_options
@_camera_options
@_annotate_options
@_workers_option
@_quiet_option
def wl(input_path: Path, output_path: Path, workers: int | None, quiet: bool, **kw: Any) -> None:
    """Render the white-light / polarized-brightness corona of a solution from a viewpoint.

    Reads and resamples the solution, then integrates the Thomson-scattering brightness over
    the electron density along each line of sight: the polarized brightness pB by default, or the
    total white-light brightness with ``--frame total``. Detrend the chosen frame with
    ``--treatment radial`` (a rho-power filter, ``--radial-power``), ``--treatment newkirk`` (radial
    vignette), or ``--treatment mgn`` (fine-structure enhancement). ``--export npz`` also writes the
    raw frames (both pB and total) with their plane-of-sky coordinates. A self-contained command: it
    needs only the density and neither builds nor uses a Q⊥ volume.
    """
    kw["input_path"] = input_path
    show_progress = not quiet
    input_cfg = _input_config(kw)
    grid_cfg = _grid_config(kw)
    brightness_cfg = _brightness_config(kw, workers)
    camera_cfg = _camera_config(kw)
    output_cfg = _output_config(kw, output_path)

    print_step(f"Rendering white-light corona from [bold]{input_path.name}[/bold]")
    start = time.perf_counter()
    field = pipeline.build_field(input_cfg, grid_cfg, show_progress=show_progress)
    field_time = time.perf_counter() - start
    render_start = time.perf_counter()
    result = pipeline.render_brightness(
        field, brightness_cfg, camera_cfg, show_progress=show_progress
    )
    render_time = time.perf_counter() - render_start

    provenance = pipeline.brightness_provenance(
        input_cfg, grid_cfg, brightness_cfg, camera_cfg, output_cfg, result
    )
    try:
        written = write_brightness(result, brightness_cfg, output_cfg, provenance)
    except ImportError as error:
        raise click.ClickException(str(error)) from error
    written += export_brightness(result, output_cfg, provenance)
    print_success(f"Wrote [bold]{output_path}[/bold]")
    _print_summary(
        provenance,
        {"field": field_time, "brightness": render_time},
        written=written,
        header="wl",
    )


# --- The end-of-run summary --------------------------------------------------------------------

#: Timing keys that are whole pipeline stages (and therefore sum to the footer total); every other
#: key is a sub-stage breakdown displayed inline on its stage's line.
_TOP_LEVEL_TIMINGS = ("field", "build", "render", "fieldlines", "export", "brightness")


def _input_line(prov: dict[str, Any]) -> str:
    inp = prov["input"]
    parts = [Path(str(inp["path"])).name]
    if inp.get("model"):
        parts.append(str(inp["model"]))
    if inp.get("timestamp"):
        stamp = f"{inp['timestamp']} UTC"
        if inp.get("cr") is not None:
            stamp += f" → CR {inp['cr']}"
        if inp.get("jd") is not None:
            stamp += f" / JD {float(inp['jd']):.3f}"
        parts.append(stamp)
    return " · ".join(parts)


def _field_line(prov: dict[str, Any], timings: dict[str, float]) -> str:
    fld = prov["field"]
    base = (
        f"{fld['n_r']}x{fld['n_theta']}x{fld['n_phi']} · "
        f"r ∈ [{fld['inner_radius']}, {fld['outer_radius']}] R☉ · "
        f"{fld['spacing']} · {fld['resampler']}"
    )
    extra = " · ".join(
        f"{name} {timings[name]:.0f} s" for name in ("read", "resample") if name in timings
    )
    return f"{base} · {extra}" if extra else base


def _volume_line(prov: dict[str, Any], timings: dict[str, float]) -> str:
    vol = prov["volume"]
    parts = [str(vol["builder"]), str(vol["grid"])]
    if vol["builder"] != "reference":
        parts.append(f"supersample {vol['supersample']}")
    parts.append(f"{float(vol['covered_fraction']):.1%} voxels covered")
    if vol.get("sub_floor_voxels") is not None:
        parts.append(f"{int(vol['sub_floor_voxels']):,} sub-floor voxels")
    if vol.get("backend"):
        backend = str(vol["backend"])
        if backend.startswith("gpu") and vol.get("precision"):
            backend += f" · {vol['precision']}"
        parts.append(backend)
    if "build" in timings:
        stages = " · ".join(
            f"{name} {timings[name]:.0f}"
            for name in ("boundary", "trace", "paint")
            if name in timings
        )
        parts.append(
            f"build {timings['build']:.0f} s ({stages})"
            if stages
            else f"build {timings['build']:.0f} s"
        )
    return " · ".join(parts)


def _camera_line(prov: dict[str, Any]) -> str:
    cam = prov["camera"]
    pixels = cam["pixels"]
    return (
        f"sub-observer ({float(cam['longitude']):+.0f}°, {float(cam['latitude']):+.0f}°) · "
        f"roll {float(cam['roll']):+.0f}° · FOV {float(cam['fov']):.0f} R☉ · "
        f"{int(pixels[1])}x{int(pixels[0])}"
    )


def _render_line(prov: dict[str, Any], timings: dict[str, float]) -> str:
    ren = prov["render"]
    parts = [
        str(ren["preset"]),
        str(ren["display_mode"]),
        str(ren["occult"]),
        f"mean coverage {float(ren['mean_coverage']):.2f}",
        f"clamped {float(ren['lower_clamped_fraction']):.1%} at floor, "
        f"{float(ren['upper_clamped_fraction']):.1%} at log_max",
    ]
    if ren.get("backend"):
        parts.append(str(ren["backend"]))
    if "render" in timings:
        parts.append(f"render {timings['render']:.1f} s")
    return " · ".join(parts)


def _fieldlines_line(prov: dict[str, Any], timings: dict[str, float]) -> str:
    fld = prov["fieldlines"]
    disk = "magnetogram" if fld.get("magnetogram") else "flat disk"
    parts = [
        f"{fld['seeding']} seeding · {fld['colour']} · {disk}",
        f"{int(fld['n_open'])} open · {int(fld['n_closed'])} closed · "
        f"{int(fld['n_incomplete'])} incomplete",
        f"width {float(fld['line_width']):.1f} px",
    ]
    if "fieldlines" in timings:
        parts.append(f"trace+draw {timings['fieldlines']:.1f} s")
    return " · ".join(parts)


def _export_line(prov: dict[str, Any], timings: dict[str, float]) -> str:
    exp = prov["export"]
    parts = [
        f"{exp['n_theta']}x{exp['n_phi']} lon/lat seeds at r = {float(exp['seed_radius']):.2f}",
        f"{int(exp['n_open'])} open · {int(exp['n_closed'])} closed · "
        f"{int(exp['n_incomplete'])} incomplete (dropped)",
    ]
    if "export" in timings:
        parts.append(f"trace {timings['export']:.1f} s")
    return " · ".join(parts)


def _brightness_line(prov: dict[str, Any], timings: dict[str, float]) -> str:
    bri = prov["brightness"]
    frame = "pB" if bri["frame"] == "polarized" else "white-light"
    treatment = (
        f"radial r^{float(bri['radial_power']):g}"
        if bri["treatment"] == "radial"
        else str(bri["treatment"])
    )
    parts = [
        f"{frame} · {treatment}",
        str(bri["occult"]),
        f"median polarization {float(bri['median_polarization']):.2f}",
        f"pB spans {float(bri['pb_decades']):.1f} decades",
    ]
    exported = prov.get("output", {}).get("export_formats")
    if exported:
        parts.append(f"export {'+'.join(exported)} (raw pB+B)")
    if "brightness" in timings:
        parts.append(f"render {timings['brightness']:.1f} s")
    return " · ".join(parts)


def _output_line(written: list[Path] | None, volume_path: Path | None) -> str:
    if volume_path is not None:
        return f"wrote {volume_path.name}"
    if not written:
        return "-"
    head = f"wrote {written[0].name}"
    if len(written) > 1:
        head += "  (+ " + ", ".join(path.name for path in written[1:]) + ")"
    return head


def _print_summary(
    provenance: dict[str, Any],
    timings: dict[str, float],
    *,
    written: list[Path] | None = None,
    volume_path: Path | None = None,
    header: str,
) -> None:
    """Print the grouped end-of-run summary: the printed counterpart of the on-image stamp.

    Reports, per section, the resolved parameters and quantitative metrics drawn from the single
    provenance mapping and the stage timings, so the on-image stamp and this panel never disagree.
    Sections absent from a given command's provenance (e.g. Camera/Render for `build`) are skipped.
    """
    from rich.panel import Panel
    from rich.table import Table

    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", justify="right")
    table.add_column()
    table.add_row("Input", _input_line(provenance))
    if "field" in provenance:
        table.add_row("Field", _field_line(provenance, timings))
    if "volume" in provenance:
        table.add_row("Volume", _volume_line(provenance, timings))
    if "fieldlines" in provenance:
        table.add_row("Field lines", _fieldlines_line(provenance, timings))
    if "export" in provenance:
        table.add_row("Export", _export_line(provenance, timings))
    if "camera" in provenance:
        table.add_row("Camera", _camera_line(provenance))
    if "render" in provenance:
        table.add_row("Render", _render_line(provenance, timings))
    if "brightness" in provenance:
        table.add_row("Brightness", _brightness_line(provenance, timings))
    table.add_row("Output", _output_line(written, volume_path))

    # The footer total sums the top-level stages only; the sub-stage entries (read/resample within
    # field, boundary/trace/paint within build) are breakdowns of those and would double-count.
    total = sum(seconds for name, seconds in timings.items() if name in _TOP_LEVEL_TIMINGS)
    console.print(
        Panel(
            table,
            title=f"[bold]qorona {header}[/bold]",
            subtitle=(
                f"[dim]total {total:.1f} s · qorona {provenance.get('version', __version__)}[/dim]"
            ),
            border_style="green",
            expand=False,
        )
    )


if __name__ == "__main__":
    main()
