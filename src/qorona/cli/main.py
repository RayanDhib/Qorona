"""The ``qorona`` CLI: ``build`` / ``render`` / ``qmap`` / ``run`` / ``fieldlines`` /
``export-lines`` / ``info``.

A :mod:`click` group wired to the shared :mod:`qorona.console` Rich surface, so progress and styling
stay uniform. It splits the two cost axes of the pipeline into separate commands: ``build`` produces
the viewpoint-independent Q⊥ volume **once** (the minutes-scale stage, where resolution / seeding /
supersampling sweeps live), ``render`` integrates a built volume for any camera / preset (seconds,
where viewpoint / weighting sweeps live), and ``run`` chains both; ``qmap`` slices a fixed-radius
signed-log-Q⊥ shell from a built volume, ``fieldlines`` draws the field-line view, ``export-lines``
serialises traced field lines for external tools, and ``info`` inspects a solution.

Every flag populates the typed :mod:`qorona.config` schema (the single source of truth for defaults
and validation); a flag left unset defers to the dataclass, so the help text's stated defaults are
documentation, not a second behavioural source (the documented exceptions: the single-axis
image-dimension fallback, the volume-cache write options, the ``--quality`` preset, a second
layer of resolution defaults resolved before the schema, and the model-aware ``wl`` vignette
default). Help has two levels
(:mod:`qorona.cli.help`): ``--help`` lists a command's common options, ``--help-all`` every option
grouped by pipeline stage. After any command that produces a result, a polished end-of-run summary
prints the run's parameters and quantitative metrics, the printed counterpart of the on-image
stamp.
"""

from __future__ import annotations

import os
import time
import warnings
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click

from qorona import __version__, pipeline
from qorona.cli.help import QoronaGroup, option
from qorona.config import (
    ANNOTATE_CONTENTS,
    ANNOTATE_POSITIONS,
    BRIGHTNESS_FRAMES,
    BRIGHTNESS_OCCULT_MODES,
    BRIGHTNESS_SCALINGS,
    BRIGHTNESS_VIGNETTES,
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
    QMAP_EXPORT_FORMATS,
    QUALITY_PRESETS,
    RENDER_EXPORT_FORMATS,
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
    QMapConfig,
    RenderConfig,
    VolumeConfig,
    WeightingConfig,
)
from qorona.console import console, print_step, print_success, print_warning
from qorona.io.fieldlines_export import write_fieldlines_json
from qorona.io.output import (
    export_brightness,
    write_brightness,
    write_fieldlines,
    write_outputs,
    write_qmap,
)
from qorona.io.readers import resolve_model

#: Default image dimension used only when exactly one of ``--width`` / ``--height`` is supplied; the
#: full default ``pixels`` otherwise lives once on :class:`~qorona.config.CameraConfig`.
_DEFAULT_DIMENSION = 1024

#: Default image dimension of the white-light product, matching the Q⊥ render; the camera flags
#: override it.
_WL_DEFAULT_DIMENSION = 1024

#: Default field of view (full width, R_sun) of the white-light product, wider than the render's
#: 8 so the streamers keep headroom instead of touching the frame. ``--fov`` overrides it.
_WL_DEFAULT_FOV = 10.0


# --- Reusable option groups --------------------------------------------------------------------
# Defined once and shared across the commands so the flag surface never drifts between them.
# Every option default is ``None`` (``--quiet``, a display-only flag, excepted) so the typed
# schema remains the sole source of defaults.


def _compose(*options: Callable[[Callable], Callable]) -> Callable[[Callable], Callable]:
    """Apply a sequence of click option decorators in declaration order."""

    def wrap(func: Callable) -> Callable:
        for decorator in reversed(options):
            func = decorator(func)
        return func

    return wrap


def _writable_output(ctx: click.Context, param: click.Parameter, value: Path | None) -> Path | None:
    """Validate an output path at parse time, before any expensive work runs.

    Creates a missing parent directory (so ``-o results/run/out.png`` just works) and errors
    clearly when the destination cannot be written, instead of crashing at the final save.
    """
    if value is None:
        return value
    parent = value.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise click.BadParameter(f"cannot create output directory {parent}: {error}") from error
    if not os.access(parent, os.W_OK):
        raise click.BadParameter(f"output directory is not writable: {parent}")
    if value.exists() and not os.access(value, os.W_OK):
        raise click.BadParameter(f"output file is not writable: {value}")
    return value


def _valid_timestamp(ctx: click.Context, param: click.Parameter, value: str | None) -> str | None:
    """Reject an unparseable ``--timestamp`` at parse time, not after the build."""
    if value is None:
        return value
    try:
        pipeline.derive_jd(value)
    except Exception as error:
        raise click.BadParameter(str(error)) from error
    return value


_input_options = _compose(
    option(
        "--model",
        default=None,
        section="Input",
        advanced=True,
        help="Solution model; inferred from the extension if unset.",
    ),
    option(
        "--timestamp",
        default=None,
        callback=_valid_timestamp,
        section="Input",
        help="UTC ISO-8601 observation time → Carrington rotation + Julian date.",
    ),
    option(
        "--variables",
        default=None,
        section="Input",
        advanced=True,
        help="Comma-separated state-variable names (reader override).",
    ),
)
_grid_options = _compose(
    option(
        "--n-r",
        type=int,
        default=None,
        section="Field grid",
        advanced=True,
        help="Radial field-grid nodes (default 192).",
    ),
    option(
        "--n-theta",
        type=int,
        default=None,
        section="Field grid",
        advanced=True,
        help="Colatitude field-grid nodes (default 180).",
    ),
    option(
        "--n-phi",
        type=int,
        default=None,
        section="Field grid",
        advanced=True,
        help="Azimuth field-grid nodes, even (default 360).",
    ),
    option(
        "--inner-radius",
        type=float,
        default=None,
        section="Field grid",
        advanced=True,
        help="Inner shell radius in R_sun (default 1.0).",
    ),
    option(
        "--outer-radius",
        type=float,
        default=None,
        section="Field grid",
        help="Outer shell radius in R_sun (default 12.5).",
    ),
    option(
        "--spacing",
        type=click.Choice(SPACING_LAWS),
        default=None,
        section="Field grid",
        advanced=True,
        help="Radial spacing law (default logarithmic).",
    ),
    option(
        "--resampler",
        type=click.Choice(RESAMPLERS),
        default=None,
        section="Field grid",
        advanced=True,
        help="Cell→grid resampler (default auto: structured for structured-grid inputs, "
        "knn-mls otherwise).",
    ),
    option(
        "--mls-k",
        "n_neighbors",
        type=int,
        default=None,
        section="Field grid",
        advanced=True,
        help=(
            "k-NN MLS neighbour count for resampling (default 30, scales up on finer meshes; "
            "try 48 if a solution shows resampling artefacts)."
        ),
    ),
)


def _turn_guard_options(section: str) -> tuple[Callable[[Callable], Callable], ...]:
    """The sharp-turn guard knobs, shared by the volume, field-line, and export tracers.

    The defaults apply automatically; ``--max-turn-angle 0`` disables the guard. In the
    ``--help-all`` view the flags appear under ``section``, the config they feed.
    """
    return (
        option(
            "--max-turn-angle",
            type=float,
            default=None,
            section=section,
            advanced=True,
            help="Sharp-turn guard: terminate a line that turns more than this many degrees in one "
            "step in the weak-field outer corona, a deflection at a null (default 45; 0 disables).",
        ),
        option(
            "--turn-guard-radius",
            type=float,
            default=None,
            section=section,
            advanced=True,
            help="Sharp-turn guard: fire only above this radius in R_sun (default 2.0).",
        ),
        option(
            "--turn-guard-weak-fraction",
            type=float,
            default=None,
            section=section,
            advanced=True,
            help="Sharp-turn guard: fire only where |B| is below this fraction of the field's "
            "peak |B| (default 1e-5).",
        ),
        option(
            "--min-turns",
            type=int,
            default=None,
            section=section,
            advanced=True,
            help="Sharp-turn guard: number of qualifying sharp turns that triggers termination; "
            "occasional null grazes are kept (default 1 for volume builds, 3 for fieldlines).",
        ),
    )


_volume_options = _compose(
    option(
        "--builder",
        type=click.Choice(VOLUME_BUILDERS),
        default=None,
        section="Volume",
        advanced=True,
        help="Q⊥ volume builder: paint (fast production fill), per-voxel (every voxel traced "
        "to its feet; complete coverage, cost grows with voxels), reference (validation "
        "ground truth). Default paint.",
    ),
    option(
        "--resolution-factor",
        type=int,
        default=None,
        section="Volume",
        advanced=True,
        help="Volume grid = field grid refined by this factor (default 2).",
    ),
    option(
        "--supersample",
        type=int,
        default=None,
        section="Volume",
        advanced=True,
        help="Boundary/seed angular supersampling (default 4).",
    ),
    option(
        "--paint-step",
        type=float,
        default=None,
        section="Volume",
        advanced=True,
        help="Paint along-line pitch as a fraction of the cell extent (default 0.5).",
    ),
    option(
        "--closed",
        type=click.Choice(CLOSED_TREATMENTS),
        default=None,
        section="Volume",
        advanced=True,
        help="Closed-loop polarity: neutral (feet cancel to 0) or dominant (default neutral).",
    ),
    option(
        "--rtol",
        type=float,
        default=None,
        section="Volume",
        advanced=True,
        help="Tracer/transport relative tolerance (default 1e-4).",
    ),
    option(
        "--cfl",
        type=float,
        default=None,
        section="Volume",
        advanced=True,
        help="CFL step ceiling, 0<cfl<1 (default 0.5).",
    ),
    option(
        "--max-steps",
        type=int,
        default=None,
        section="Volume",
        advanced=True,
        help="Per-half-line step guard (default 10000).",
    ),
    option(
        "--max-reversals",
        type=int,
        default=None,
        section="Volume",
        advanced=True,
        help="Stall guard: terminate a line after this many >90° direction reversals, a line "
        "trapped at a weak-field null (default 8; 0 disables).",
    ),
    option(
        "--precision",
        type=click.Choice(PRECISION_MODES),
        default=None,
        section="Volume",
        advanced=True,
        help="CUDA kernel precision: mixed (f32 field interpolation, f64 elsewhere; default), "
        "float64 (all-double reference), float32 (experimental fully-float32 painter). GPU only; "
        "on `run` it applies to both the build and the render.",
    ),
    *_turn_guard_options("Volume"),
)
_cache_options = _compose(
    option(
        "--cache-dtype",
        type=click.Choice(("float32", "float64")),
        default=None,
        section="Cache",
        advanced=True,
        help="Stored volume dtype (default float32).",
    ),
    option(
        "--compress/--no-compress",
        "compress",
        default=None,
        section="Cache",
        advanced=True,
        help="DEFLATE-compress the volume artifact (default on).",
    ),
)


def _camera_options(
    default_dimension: int = _DEFAULT_DIMENSION, default_fov: float = 25.0
) -> Callable[[Callable], Callable]:
    """The shared camera option group; the defaults state a command's image and FOV defaults."""
    return _compose(
        option(
            "--longitude",
            type=float,
            default=None,
            section="Camera",
            help="Sub-observer heliographic longitude in degrees (default 0).",
        ),
        option(
            "--latitude",
            type=float,
            default=None,
            section="Camera",
            help="Sub-observer heliographic latitude in degrees (default 0).",
        ),
        option(
            "--roll",
            type=float,
            default=None,
            section="Camera",
            help="Camera roll about the line of sight in degrees (default 0).",
        ),
        option(
            "--fov",
            type=float,
            default=None,
            section="Camera",
            help=f"Field of view (full width) in R_sun (default {default_fov:g}).",
        ),
        option(
            "--width",
            type=int,
            default=None,
            section="Camera",
            help=f"Image width in pixels (default {default_dimension}).",
        ),
        option(
            "--height",
            type=int,
            default=None,
            section="Camera",
            help=f"Image height in pixels (default {default_dimension}).",
        ),
        option(
            "--observer",
            type=click.Choice(("earth",)),
            default=None,
            section="Camera",
            help="Named observer: derive the sub-observer longitude/latitude (and the "
            "FITS observer distance) from its ephemeris at the timestamp. Currently: earth. "
            "Assumes the solution frame is Carrington-aligned.",
        ),
    )


_weighting_options = _compose(
    option(
        "--preset",
        type=click.Choice(WEIGHTING_PRESETS),
        default=None,
        section="Render",
        help="Geometric depth-weighting preset (default large-fov).",
    ),
)
_render_options = _compose(
    option(
        "--display",
        type=click.Choice(DISPLAY_MODES),
        default=None,
        section="Render",
        advanced=True,
        help="Depth-colour reconstruction (default balanced).",
    ),
    option(
        "--polarity-mode",
        type=click.Choice(POLARITY_MODES),
        default=None,
        section="Render",
        help="Colour by magnetic polarity: hue=warm outward/cool inward (default none).",
    ),
    option(
        "--occult",
        type=click.Choice(OCCULT_MODES),
        default=None,
        section="Render",
        help="Body treatment: eclipse (dark disk), opaque (solid 3-D body), composite (the "
        "eclipse view with the disk filled by the near-limb view, toned down), none "
        "(default eclipse).",
    ),
    option(
        "--r-occult",
        type=float,
        default=None,
        section="Render",
        advanced=True,
        help="Body/occulter radius in R_sun (default 1.0).",
    ),
    option(
        "--occult-softness",
        type=float,
        default=None,
        section="Render",
        advanced=True,
        help="Eclipse-edge feather in R_sun (default 0.03).",
    ),
    option(
        "--disk-tone",
        type=float,
        default=None,
        section="Render",
        advanced=True,
        help="Composite mode: disk brightness relative to the inner corona (default 0.8).",
    ),
    option(
        "--disk-desat",
        type=float,
        default=None,
        section="Render",
        advanced=True,
        help="Composite mode: disk desaturation, 0-1 (default 0.4).",
    ),
    option(
        "--clamp",
        type=float,
        nargs=2,
        default=None,
        section="Render",
        advanced=True,
        help="Display log10 Q⊥ clamp 'LOW HIGH' (default log10(2) 7.0).",
    ),
    option(
        "--floor/--no-floor",
        "floor",
        default=None,
        section="Render",
        advanced=True,
        help="Apply the lower display clamp; --no-floor keeps the sub-floor tail (default on).",
    ),
    option(
        "--step",
        type=float,
        default=None,
        section="Render",
        help="Line-of-sight sample spacing in R_sun (default 0.02).",
    ),
    option(
        "--percentiles",
        type=float,
        nargs=2,
        default=None,
        section="Render",
        advanced=True,
        help="Per-channel stretch percentiles 'LOW HIGH' (default 1.0 99.5).",
    ),
)
_annotate_options = _compose(
    option(
        "--annotate/--no-annotate",
        "annotate",
        default=None,
        section="Output",
        advanced=True,
        help="Burn the provenance stamp onto the PNG (default on).",
    ),
    option(
        "--annotate-position",
        type=click.Choice(ANNOTATE_POSITIONS),
        default=None,
        section="Output",
        advanced=True,
        help="Stamp corner (default bottom-left).",
    ),
    option(
        "--annotate-content",
        type=click.Choice(ANNOTATE_CONTENTS),
        default=None,
        section="Output",
        advanced=True,
        help="Stamp text: the full provenance block, or just the observation "
        "date and time (default full).",
    ),
)
_output_options = _compose(
    option(
        "--grayscale/--no-grayscale",
        "grayscale",
        default=None,
        section="Output",
        advanced=True,
        help="Also write the grayscale measurement PNG (default off).",
    ),
    option(
        "--export",
        "export_formats",
        type=click.Choice(RENDER_EXPORT_FORMATS),
        multiple=True,
        section="Output",
        advanced=True,
        help="Also write the quantitative render (LOS-averaged log10 Q_perp, "
        "WCS-registered) beside the PNG; repeatable. Currently only fits.",
    ),
    _annotate_options,
)
_fieldlines_options = _compose(
    option(
        "--seeding",
        type=click.Choice(FIELDLINE_SEEDING),
        default=None,
        section="Field lines",
        help="Seeding: limb (front loops + limb fan) or uniform sphere (default limb).",
    ),
    option(
        "--seeds",
        "n_seeds",
        type=int,
        default=None,
        section="Field lines",
        help="Fibonacci seed budget: full sphere (uniform) or front-loop source (limb) "
        "(default 1500).",
    ),
    option(
        "--limb-seeds",
        type=int,
        default=None,
        section="Field lines",
        advanced=True,
        help="Seeds around the limb ring (the open fan; limb seeding) (default 375).",
    ),
    option(
        "--front-loop-length",
        type=float,
        default=None,
        section="Field lines",
        advanced=True,
        help="Max arc length R_sun for a kept near-side closed loop (limb seeding) (default 1.2).",
    ),
    option(
        "--colour",
        type=click.Choice(FIELDLINE_COLOUR),
        default=None,
        section="Field lines",
        help="Line colour: rainbow (per-line hue) or polarity (B_r sign) (default polarity).",
    ),
    option(
        "--magnetogram/--no-magnetogram",
        "magnetogram",
        default=None,
        section="Field lines",
        advanced=True,
        help="Render the disk as a B_r magnetogram from the data (default on).",
    ),
    option(
        "--line-width",
        type=float,
        default=None,
        section="Field lines",
        advanced=True,
        help="Drawn line width in pixels (default 1.5).",
    ),
    option(
        "--show",
        type=click.Choice(FIELDLINE_SHOW),
        default=None,
        section="Field lines",
        help="Which lines to draw: all / open / closed (default all).",
    ),
    option(
        "--depth-fade",
        type=float,
        default=None,
        section="Field lines",
        advanced=True,
        help="Dim far-side lines by up to this fraction, 0-1 (default 0.4).",
    ),
    option(
        "--rtol",
        type=float,
        default=None,
        section="Field lines",
        advanced=True,
        help="Tracer relative tolerance (default 1e-4).",
    ),
    option(
        "--cfl",
        type=float,
        default=None,
        section="Field lines",
        advanced=True,
        help="CFL step ceiling, 0<cfl<1 (default 0.5).",
    ),
    option(
        "--max-steps",
        type=int,
        default=None,
        section="Field lines",
        advanced=True,
        help="Per-half-line step guard (default 10000).",
    ),
    *_turn_guard_options("Field lines"),
)
_export_options = _compose(
    option(
        "--seeds",
        "seed_grid",
        type=int,
        nargs=2,
        default=None,
        section="Export",
        help="Seed grid resolution 'N_THETA N_PHI' on the seed sphere (default 100 100).",
    ),
    option(
        "--seed-radius",
        type=float,
        default=None,
        section="Export",
        help="Seed sphere radius in R_sun (default: the inner boundary).",
    ),
    option(
        "--rtol",
        type=float,
        default=None,
        section="Export",
        advanced=True,
        help="Tracer relative tolerance (default 1e-4).",
    ),
    option(
        "--cfl",
        type=float,
        default=None,
        section="Export",
        advanced=True,
        help="CFL step ceiling, 0<cfl<1 (default 0.5).",
    ),
    option(
        "--max-steps",
        type=int,
        default=None,
        section="Export",
        advanced=True,
        help="Per-half-line step guard (default 10000).",
    ),
    *_turn_guard_options("Export"),
)
_brightness_options = _compose(
    option(
        "--frame",
        type=click.Choice(BRIGHTNESS_FRAMES),
        default=None,
        section="Brightness",
        help="Brightness frame: polarized (pB) or total (white-light) (default polarized).",
    ),
    option(
        "--vignette",
        type=click.Choice(BRIGHTNESS_VIGNETTES),
        default=None,
        section="Brightness",
        help="Display treatment of the selected --frame: newkirk divides by the brightness of "
        "the smooth Newkirk background corona; adaptive self-calibrates the same curve family "
        "to the image's own falloff and amplifies its structure; wow whitens the raw frame's "
        "wavelet spectrum (needs sunkit-image and watroo); none keeps the raw falloff "
        "(default newkirk; adaptive for inputs whose falloff departs from it).",
    ),
    option(
        "--mgn",
        is_flag=True,
        default=False,
        section="Brightness",
        advanced=True,
        help="Multi-scale Gaussian normalization fine-structure enhancement, applied last "
        "(needs sunkit-image).",
    ),
    option(
        "--export",
        "export_formats",
        type=click.Choice(EXPORT_FORMATS),
        multiple=True,
        section="Output",
        advanced=True,
        help="Also write raw data beside the PNG; repeatable. npz: both raw frames + "
        "plane-of-sky coordinates. fits: the display frame + raw frames, WCS-registered "
        "(needs a timestamp).",
    ),
    option(
        "--limb-darkening",
        "limb_darkening",
        type=float,
        default=None,
        section="Brightness",
        advanced=True,
        help="Thomson limb-darkening coefficient u, 0-1 (default 0.6).",
    ),
    option(
        "--crossover",
        type=float,
        default=None,
        section="Brightness",
        advanced=True,
        help="Closed-form→asymptotic coefficient crossover radius in R_sun (default 10).",
    ),
    option(
        "--step",
        type=float,
        default=None,
        section="Brightness",
        help="Line-of-sight sample spacing in R_sun (default 0.02).",
    ),
    option(
        "--occult",
        type=click.Choice(BRIGHTNESS_OCCULT_MODES),
        default=None,
        section="Brightness",
        help="Occultation mode (default eclipse).",
    ),
    option(
        "--r-occult",
        type=float,
        default=None,
        section="Brightness",
        advanced=True,
        help="Body/occulter radius in R_sun (default 1.02, just above the limb so the "
        "overwhelming near-limb ring does not eat the display stretch).",
    ),
    option(
        "--occult-softness",
        type=float,
        default=None,
        section="Brightness",
        advanced=True,
        help="Eclipse-edge feather in R_sun (default 0.03).",
    ),
    option(
        "--scaling",
        type=click.Choice(BRIGHTNESS_SCALINGS),
        default=None,
        section="Brightness",
        advanced=True,
        help="Intensity stretch: log or linear (default: linear with a vignette or --mgn, "
        "log for the raw falloff).",
    ),
    option(
        "--percentiles",
        type=float,
        nargs=2,
        default=None,
        section="Brightness",
        advanced=True,
        help="Per-image stretch percentiles 'LOW HIGH' (default 1 99.5; use 0 100 for the "
        "full untrimmed range).",
    ),
)
_workers_option = option(
    "--workers",
    type=int,
    default=None,
    section="Execution",
    help="Worker threads for the numba kernels (default all cores).",
)
_device_option = option(
    "--device",
    type=click.Choice(DEVICE_MODES),
    default=None,
    section="Execution",
    help="Compute backend: auto (GPU when present, else CPU), gpu (force; errors if absent), "
    "cpu. Default auto.",
)
_render_precision_option = option(
    "--precision",
    type=click.Choice(PRECISION_MODES),
    default=None,
    section="Render",
    advanced=True,
    help="CUDA render kernel precision: mixed (f32 sampling, f64 accumulation; default), "
    "float64 (all-double reference); float32 is accepted as mixed. GPU only.",
)
_quiet_option = option(
    "--quiet",
    is_flag=True,
    default=False,
    section="Execution",
    help="Suppress progress bars and the spinner.",
)
_quality_option = option(
    "--quality",
    type=click.Choice(tuple(QUALITY_PRESETS)),
    default=None,
    section="Volume",
    help="Volume quality preset: fast (= --resolution-factor 1 --supersample 2, ~12M voxels at "
    "the default grid), standard (= 2/4, ~100M, the default), high (= 3/6, ~336M). Explicit "
    "--resolution-factor / --supersample override the preset.",
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
            kw,
            "n_r",
            "n_theta",
            "n_phi",
            "inner_radius",
            "outer_radius",
            "spacing",
            "resampler",
            "n_neighbors",
        )
    )


def _volume_config(kw: dict[str, Any], workers: int | None) -> VolumeConfig:
    # The --quality preset is a second layer of resolution defaults: it fills resolution_factor /
    # supersample only where no explicit flag was given; the schema supplies everything else.
    if kw.get("quality") is not None:
        preset_factor, preset_supersample = QUALITY_PRESETS[kw["quality"]]
        if kw.get("resolution_factor") is None:
            kw["resolution_factor"] = preset_factor
        if kw.get("supersample") is None:
            kw["supersample"] = preset_supersample
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


def _observer_gate(kw: dict[str, Any], timestamp: str | None) -> None:
    """Reject ``--observer earth`` calls that conflict with manual pointing or lack an epoch.

    Called by ``_camera_config`` for every command; commands whose expensive compute precedes
    camera resolution (the ``wl`` raw-solution path) also call it up front to fail fast.
    """
    if kw.get("observer") != "earth":
        return
    if kw.get("longitude") is not None or kw.get("latitude") is not None:
        raise click.UsageError(
            "--observer earth derives the sub-observer point from the ephemeris; "
            "drop --longitude/--latitude, or drop --observer to point manually"
        )
    if timestamp is None:
        raise click.UsageError(
            "--observer earth needs an epoch; pass --timestamp (or bake the volume with one)"
        )


def _camera_config(
    kw: dict[str, Any],
    *,
    default_dimension: int = _DEFAULT_DIMENSION,
    default_fov: float | None = None,
    timestamp: str | None = None,
) -> CameraConfig:
    fields = _present(kw, "longitude", "latitude", "roll", "fov")
    _observer_gate(kw, timestamp)
    if kw.get("observer") == "earth":
        assert timestamp is not None  # _observer_gate rejects the missing-epoch case
        longitude, latitude, distance = pipeline.sub_earth_point(timestamp)
        fields["longitude"], fields["latitude"] = longitude, latitude
        fields["observer_distance"] = distance
    if default_fov is not None and "fov" not in fields:
        fields["fov"] = default_fov
    height, width = kw.get("height"), kw.get("width")
    if height is not None or width is not None or default_dimension != _DEFAULT_DIMENSION:
        fields["pixels"] = (
            height if height is not None else default_dimension,
            width if width is not None else default_dimension,
        )
    return CameraConfig(**fields)


def _weighting_config(kw: dict[str, Any]) -> WeightingConfig:
    return WeightingConfig(**_present(kw, "preset"))


def _render_config(kw: dict[str, Any], workers: int | None) -> RenderConfig:
    fields = _present(
        kw,
        "display",
        "occult",
        "r_occult",
        "occult_softness",
        "disk_tone",
        "disk_desat",
        "step",
        "polarity_mode",
        "device",
        "precision",
    )
    if kw.get("clamp") is not None:
        fields["clamp"] = tuple(kw["clamp"])
    if kw.get("percentiles") is not None:
        fields["percentiles"] = tuple(kw["percentiles"])
    if kw.get("floor") is not None:
        fields["floor"] = kw["floor"]
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
    if kw.get("annotate_content") is not None:
        fields["annotate_content"] = kw["annotate_content"]
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
        "vignette",
        "crossover",
        "step",
        "occult",
        "r_occult",
        "occult_softness",
        "scaling",
    )
    if kw.get("mgn"):
        fields["mgn"] = True
    if kw.get("limb_darkening") is not None:
        fields["u"] = kw["limb_darkening"]
    if kw.get("percentiles") is not None:
        fields["percentiles"] = tuple(kw["percentiles"])
    if workers is not None:
        fields["workers"] = workers
    return BrightnessConfig(**fields)


def _parse_resolution(text: str) -> tuple[int, int]:
    """Parse a ``'NTHETAxNPHI'`` resolution string into ``(n_theta, n_phi)``."""
    parts = text.lower().split("x")
    if len(parts) != 2:
        raise click.BadParameter("resolution must be 'NTHETAxNPHI', e.g. 720x1440")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        raise click.BadParameter("resolution must be 'NTHETAxNPHI', e.g. 720x1440") from None


def _qmap_config(kw: dict[str, Any]) -> QMapConfig:
    fields = _present(kw, "radius", "slog_max")
    if kw.get("export_formats"):
        fields["export_formats"] = tuple(kw["export_formats"])
    if kw.get("resolution"):
        fields["n_theta"], fields["n_phi"] = _parse_resolution(kw["resolution"])
    return QMapConfig(**fields)


#: Build-flag defaults; a provenance value equal to its default is left out of the rebuild command.
_BUILD_DEFAULTS = {
    "n_r": 192,
    "n_theta": 180,
    "n_phi": 360,
    "inner_radius": 1.0,
    "spacing": "logarithmic",
    "resampler": "auto",
    "builder": "paint",
    "resolution_factor": 2,
    "supersample": 4,
}


def _rebuild_command(build_prov: dict[str, Any], outer_radius: float) -> str:
    """Reconstruct the ``qorona build`` command from a volume's provenance, with the outer radius
    set to ``outer_radius``, the build of the canonical Q-map volume (mapping domain
    ``[1, outer_radius]``). Flags left at their default are omitted."""
    inp = build_prov["input"]
    field = build_prov.get("field", {})
    volume = build_prov.get("volume", {})
    flags = []
    if inp.get("timestamp"):
        flags.append(f"--timestamp {inp['timestamp']}")
    flags.append(f"--outer-radius {outer_radius:g}")
    sources = (
        (field, ("inner_radius", "n_r", "n_theta", "n_phi", "spacing", "resampler")),
        (volume, ("builder", "resolution_factor", "supersample")),
    )
    for source, keys in sources:
        for key in keys:
            value = source.get(key)
            if value is None or value == _BUILD_DEFAULTS[key]:
                continue
            rendered = f"{value:g}" if isinstance(value, float) else f"{value}"
            flags.append(f"--{key.replace('_', '-')} {rendered}")
    stem = Path(inp["path"]).name.split(".")[0]
    out_name = f"{stem}_or{outer_radius:g}.qor"
    return f"  qorona build {inp['path']} -o {out_name} \\\n      " + " ".join(flags)


def _warn_qmap_outer_radius(
    outer_radius: float, qmap_cfg: QMapConfig, build_prov: dict[str, Any]
) -> None:
    """Warn when the volume's outer radius does not sit at the Q-map radius.

    The canonical Q-map maps the domain ``[1, r]``: baking with the outer radius at the map radius
    puts the heliospheric current sheet on the Q⊥ ridges. A deeper volume slices the ``[1, outer]``
    mapping, where the current sheet can drift off the ridges; a shallower one cannot reach that
    radius, so the map is clamped to the boundary. Either way, print the build command that produces
    it right.
    """
    radius = qmap_cfg.radius
    if radius > outer_radius * (1.0 + 1e-6):
        print_warning(
            f"r = {radius:g} R_sun is outside this volume (outer = {outer_radius:g}); the map was "
            f"clamped to the boundary. To map at r = {radius:g}, rebuild at that radius:"
        )
    elif outer_radius > radius * 1.01:
        print_warning(
            f"Q-map at r = {radius:g} R_sun sliced from an outer = {outer_radius:g} volume: the "
            f"current sheet may sit off the Q⊥ ridges (a slice of the [1, {outer_radius:g}] "
            "mapping). For the canonical map, rebuild with the outer radius at the map radius:"
        )
    else:
        return
    console.print(f"[dim]{_rebuild_command(build_prov, radius)}[/dim]")


# --- The command group -------------------------------------------------------------------------


@click.group(
    cls=QoronaGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
    epilog="Each command's --help lists its common options; --help-all shows every option, "
    "grouped by pipeline stage.",
)
@click.version_option(__version__, "-V", "--version", prog_name="qorona")
def main() -> None:
    """Qorona: synthetic coronal imagery from global MHD solutions.

    Render the line-of-sight magnetic squashing factor Q⊥ of a coronal MHD solution into
    eclipse-like imagery. Build the viewpoint-independent volume once with `build`, then render any
    number of viewpoints cheaply with `render`; `run` does both in one shot, `qmap` slices a
    fixed-radius Q⊥ shell, `fieldlines` draws the field-line view, `export-lines` serialises traced
    lines, and `info` inspects a file.
    """
    # Silence numba CUDA low-occupancy warnings from the one-seed warm-up and partial final chunks.
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
    callback=_writable_output,
    help="Destination volume artifact (.qor / .npz).",
)
@_quality_option
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
    """Build the viewpoint-independent Q⊥ volume from a solution to a cache file.

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

    print_step(f"Building Q⊥ volume from [bold]{input_path.name}[/bold]")
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
    callback=_writable_output,
    help="Destination image (.png).",
)
@option(
    "--timestamp",
    default=None,
    callback=_valid_timestamp,
    section="Input",
    advanced=True,
    help="Override the volume's timestamp for the stamp (re-derives CR/JD).",
)
@_camera_options()
@_weighting_options
@_render_options
@_render_precision_option
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
    """Render a built Q⊥ volume to an eclipse-like image from a viewpoint.

    The seconds-scale stage: load the volume, integrate it for one camera / preset / display, write
    the PNG(s) with the on-image stamp, and print the metrics. Repeat for new viewpoints off the
    same volume. Runs on the GPU when one is present (`--device` selects; `--device cpu` forces
    the reference CPU path).
    """
    kw["device"] = device
    show_progress = not quiet
    weighting_cfg = _weighting_config(kw)
    render_cfg = _render_config(kw, workers)
    output_cfg = _output_config(kw, output_path)

    print_step(f"Loading volume [bold]{volume_path.name}[/bold]")
    try:
        volume, density, build_prov = pipeline.load_volume(volume_path)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    resolved_timestamp = timestamp or build_prov.get("input", {}).get("timestamp")
    if "fits" in output_cfg.export_formats and not resolved_timestamp:
        raise click.ClickException(
            "--export fits needs a timestamp for DATE-OBS and the observer ephemeris; "
            "pass --timestamp (or bake the volume with one)"
        )
    camera_cfg = _camera_config(kw, timestamp=resolved_timestamp)
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
@click.argument("volume_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    callback=_writable_output,
    help="Destination figure (.png).",
)
@option(
    "--radius", type=float, default=None, section="Q-map", help="Shell radius in R_sun (default 3)."
)
@option(
    "--resolution",
    default=None,
    section="Q-map",
    help="Display grid 'NTHETAxNPHI' (default 720x1440; interpolated, capped by the build pitch).",
)
@option(
    "--slog-max",
    "slog_max",
    type=float,
    default=None,
    section="Q-map",
    help="Colour ceiling for slog Q⊥ (default 5).",
)
@option(
    "--export",
    "export_formats",
    type=click.Choice(QMAP_EXPORT_FORMATS),
    multiple=True,
    section="Q-map",
    help="Also write the raw shell arrays to this format beside the figure; repeatable. "
    "Currently only npz.",
)
@_annotate_options
@_quiet_option
def qmap(volume_path: Path, output_path: Path, quiet: bool, **kw: Any) -> None:
    """Slice a signed-log-Q⊥ map from a cached Q⊥ volume at a fixed radius.

    Reads the `.qor` and samples log₁₀ Q⊥ and the local radial-field sign on a longitude/latitude
    shell at `--radius`, with no re-ingest or tracing. The displayed quantity is sign(B·r̂)·log₁₀ Q⊥:
    the heliospheric current sheet is the warm↔cool boundary, the S-web arcs the saturated ridges.
    The viewpoint-independent sibling of `render`.
    """
    qmap_cfg = _qmap_config(kw)
    output_cfg = _output_config(kw, output_path)

    print_step(f"Slicing Q-map from [bold]{volume_path.name}[/bold]")
    start = time.perf_counter()
    try:
        volume, _density, build_prov = pipeline.load_volume(volume_path)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    if volume.radial_sign is None:
        raise click.ClickException(
            f"{volume_path.name} has no radial-sign channel (built before the Q-map feature); "
            "re-run `qorona build` to add it."
        )
    _warn_qmap_outer_radius(float(volume.grid.radii[-1]), qmap_cfg, build_prov)
    result = pipeline.qmap_from_volume(volume, qmap_cfg)
    qmap_time = time.perf_counter() - start

    provenance = pipeline.qmap_provenance(qmap_cfg, output_cfg, build_prov, result)
    written = write_qmap(result, qmap_cfg, output_cfg, provenance)
    print_success(f"Wrote [bold]{output_path}[/bold]")
    _print_summary(provenance, {"qmap": qmap_time}, written=written, header="qmap")


@main.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    callback=_writable_output,
    help="Destination image (.png).",
)
@_quality_option
@click.option(
    "--save-volume",
    "save_volume_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    callback=_writable_output,
    help="Also persist the built volume to this .qor/.npz.",
)
@_input_options
@_grid_options
@_volume_options
@_cache_options
@_camera_options()
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
    camera_cfg = _camera_config(kw, timestamp=kw.get("timestamp"))
    weighting_cfg = _weighting_config(kw)
    render_cfg = _render_config(kw, workers)
    output_cfg = _output_config(kw, output_path)
    if "fits" in output_cfg.export_formats and kw.get("timestamp") is None:
        raise click.ClickException(
            "--export fits needs a timestamp for DATE-OBS and the observer ephemeris; "
            "pass --timestamp"
        )

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
    callback=_writable_output,
    help="Destination image (.png).",
)
@_input_options
@_grid_options
@_fieldlines_options
@_camera_options()
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
    A self-contained command: it traces the field directly and does not use a built volume.
    """
    kw["input_path"] = input_path
    show_progress = not quiet
    input_cfg = _input_config(kw)
    grid_cfg = _grid_config(kw)
    fieldlines_cfg = _fieldlines_config(kw, workers)
    camera_cfg = _camera_config(kw, timestamp=kw.get("timestamp"))
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
        input_cfg, grid_cfg, fieldlines_cfg, camera_cfg, output_cfg, result, field=field
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
    callback=_writable_output,
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
    the field directly and does not use a built volume. The file schema is documented in
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


def _input_model(
    from_volume: bool, input_path: Path, build_prov: dict[str, Any] | None, model_flag: str | None
) -> str | None:
    """Return the model identity of a ``wl`` input, or ``None`` when it cannot be resolved.

    A volume artifact is judged by its recorded build provenance (the resolved model, or the
    recorded solution path put back through the reader registry for artifacts that predate the
    model record); a raw solution goes through the registry directly, an explicit ``--model``
    short-circuiting it.
    """
    if from_volume:
        recorded = (build_prov or {}).get("input", {})
        model = recorded.get("model")
        if model:
            return str(model).lower()
        path = recorded.get("path")
        return resolve_model(str(path)) if path else None
    return resolve_model(input_path, model=model_flag)


#: Flags that configure the re-ingest of a raw solution; meaningless with a built volume artifact,
#: which carries its own field grid and metadata, so they are rejected there instead of ignored.
_INGEST_ONLY_KEYS = (
    "model",
    "variables",
    "n_r",
    "n_theta",
    "n_phi",
    "inner_radius",
    "outer_radius",
    "spacing",
    "resampler",
    "n_neighbors",
)


@main.command(hidden=True)
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    callback=_writable_output,
    help="Destination image (.png).",
)
@_input_options
@_grid_options
@_brightness_options
@_camera_options(_WL_DEFAULT_DIMENSION, _WL_DEFAULT_FOV)
@_annotate_options
@_workers_option
@_quiet_option
def wl(input_path: Path, output_path: Path, workers: int | None, quiet: bool, **kw: Any) -> None:
    """Render the white-light / polarized-brightness corona of a solution from a viewpoint.

    INPUT is either a raw solution (read and resampled, as ``build`` would) or a built ``.qor``
    volume artifact (whose stored electron density is reused, skipping the resample). The
    Thomson-scattering brightness is integrated over the density along each line of sight: the
    polarized brightness pB by default, or the total white-light brightness with ``--frame
    total``. The frame is finished by two display stages: the ``--vignette`` treatment
    (``newkirk`` by default, ``adaptive`` for inputs whose falloff departs from it, ``wow`` for
    wavelet whitening, ``none`` for the raw falloff) and optional ``--mgn`` fine-structure
    enhancement. ``--export npz`` also writes the raw frames (both pB and total) with their
    plane-of-sky coordinates. ``--export fits`` writes the display frame plus the raw frames as
    a WCS-registered FITS and needs a timestamp. Needs only the density; it neither builds nor
    uses the Q⊥ payload.
    """
    kw["input_path"] = input_path
    show_progress = not quiet
    from_volume = zipfile.is_zipfile(input_path)
    output_cfg = _output_config(kw, output_path)
    # For a raw solution the --observer and FITS timestamp gates fire here, before the expensive
    # resample; a volume artifact may carry the epoch itself, so its gates wait for the fast
    # load below.
    if not from_volume:
        _observer_gate(kw, kw.get("timestamp"))
        if "fits" in output_cfg.export_formats and kw.get("timestamp") is None:
            raise click.ClickException(
                "--export fits needs a timestamp for DATE-OBS and the observer ephemeris; "
                "pass --timestamp"
            )

    print_step(f"Rendering white-light corona from [bold]{input_path.name}[/bold]")
    start = time.perf_counter()
    build_prov: dict[str, Any] | None = None
    if from_volume:
        rejected = [key for key in _INGEST_ONLY_KEYS if kw.get(key) is not None]
        if rejected:
            flags = ", ".join("--" + key.replace("_", "-") for key in rejected)
            raise click.ClickException(
                f"{flags}: these flags apply only when INPUT is a raw solution; a volume "
                "artifact carries its own field grid and metadata"
            )
        try:
            _volume, density, build_prov = pipeline.load_volume(input_path)
        except ValueError as error:
            raise click.ClickException(str(error)) from error
        if density is None:
            raise click.ClickException(
                "this volume artifact carries no electron density; rebuild it with "
                "`qorona build` from a solution that provides density (e.g. COCONUT 'rho'), "
                "or pass the solution itself"
            )
    else:
        input_cfg = _input_config(kw)
        grid_cfg = _grid_config(kw)
        field = pipeline.build_field(input_cfg, grid_cfg, show_progress=show_progress)
        density = field.density
        if density is None:
            raise click.ClickException(
                "the white-light / pB product needs an electron density, but this solution "
                "carries none; read a solution that provides density (e.g. COCONUT 'rho')"
            )
    field_time = time.perf_counter() - start

    resolved_timestamp = kw.get("timestamp") or (
        (build_prov or {}).get("input", {}).get("timestamp")
    )
    if "fits" in output_cfg.export_formats and not resolved_timestamp:
        raise click.ClickException(
            "--export fits needs a timestamp for DATE-OBS and the observer ephemeris; "
            "pass --timestamp (or bake the volume with one)"
        )
    camera_cfg = _camera_config(
        kw,
        default_dimension=_WL_DEFAULT_DIMENSION,
        default_fov=_WL_DEFAULT_FOV,
        timestamp=resolved_timestamp,
    )

    if (
        kw.get("vignette") is None
        and _input_model(from_volume, input_path, build_prov, kw.get("model")) == "coconut"
    ):
        # Model-aware default: the self-calibrating adaptive channel for COCONUT input; the
        # schema's newkirk default stands for every other input.
        kw["vignette"] = "adaptive"
    brightness_cfg = _brightness_config(kw, workers)

    render_start = time.perf_counter()
    result = pipeline.render_brightness(
        density, brightness_cfg, camera_cfg, show_progress=show_progress
    )
    render_time = time.perf_counter() - render_start

    if from_volume:
        assert build_prov is not None  # set with the volume load above
        provenance = pipeline.volume_brightness_provenance(
            build_prov,
            brightness_cfg,
            camera_cfg,
            output_cfg,
            result,
            timestamp_override=kw.get("timestamp"),
        )
    else:
        provenance = pipeline.brightness_provenance(
            input_cfg, grid_cfg, brightness_cfg, camera_cfg, output_cfg, result, field=field
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
_TOP_LEVEL_TIMINGS = ("field", "build", "render", "fieldlines", "export", "brightness", "qmap")


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
        backend = str(ren["backend"])
        if backend.startswith("gpu") and ren.get("precision"):
            backend += f" · {ren['precision']}"
        parts.append(backend)
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
    vignette = bri.get("vignette", "none")
    finish = ["raw" if vignette == "none" else str(vignette)]
    if bri.get("mgn"):
        finish.append("mgn")
    parts = [
        f"{frame} · {'+'.join(finish)}",
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


def _qmap_line(prov: dict[str, Any], timings: dict[str, float]) -> str:
    qm = prov["qmap"]
    parts = [
        f"Q⊥ · r = {float(qm['radius']):g} R☉ · {qm.get('resolution', '')}",
        f"coverage {float(qm['coverage']):.1%}",
        f"sub-floor {float(qm['sub_floor_fraction']):.1%}",
    ]
    exported = qm.get("export_formats")
    if exported:
        parts.append(f"export {'+'.join(exported)} (raw shell)")
    if "qmap" in timings:
        parts.append(f"slice {timings['qmap']:.1f} s")
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
    if "input" in provenance:
        table.add_row("Input", _input_line(provenance))
    if "field" in provenance:
        table.add_row("Field", _field_line(provenance, timings))
    if "volume" in provenance:
        table.add_row("Volume", _volume_line(provenance, timings))
    if "qmap" in provenance:
        table.add_row("Q-map", _qmap_line(provenance, timings))
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
