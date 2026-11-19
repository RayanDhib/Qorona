"""The ``qorona`` CLI: ``build`` / ``render`` / ``qmap`` / ``run`` / ``fieldlines`` /
``export-lines`` / ``info``.

A :mod:`click` group wired to the shared :mod:`qorona.console` Rich surface, so progress and styling
stay uniform. It splits the two cost axes of the pipeline into separate commands: ``build`` produces
the viewpoint-independent Q⊥ volume **once** (the minutes-scale stage, where resolution / seeding /
supersampling sweeps live), ``render`` integrates a built volume for any camera / preset (seconds,
where viewpoint / weighting sweeps live), and ``run`` chains both; ``qmap`` slices a fixed-radius
signed-log-Q⊥ shell from a built volume, ``fieldlines`` draws the field-line view, ``export-lines``
serialises traced field lines to SunJSON for JHelioviewer, and ``info`` inspects a solution.

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

from qorona import __version__, ephemeris, pipeline
from qorona.cli.help import QoronaGroup, option
from qorona.config import (
    ANNOTATE_CONTENTS,
    ANNOTATE_POSITIONS,
    BRIGHTNESS_COMPONENTS,
    BRIGHTNESS_OCCULT_MODES,
    BRIGHTNESS_SCALES,
    BRIGHTNESS_SCALINGS,
    BRIGHTNESS_VIGNETTES,
    CAMERA_INSTRUMENTS,
    CAMERA_PROJECTIONS,
    CLOSED_TREATMENTS,
    DEVICE_MODES,
    DISPLAY_MODES,
    EXPORT_FORMATS,
    FIELDLINE_COLOUR,
    FIELDLINE_SEEDING,
    FIELDLINE_SHOW,
    FIELDLINES_EXPORT_FORMATS,
    FRAME_ALIASES,
    ICARUS_FRAME,
    OCCULT_MODES,
    POLARITY_MODES,
    PRECISION_MODES,
    QMAP_EXPORT_FORMATS,
    QUALITY_PRESETS,
    RENDER_EXPORT_FORMATS,
    RESAMPLERS,
    SOLUTION_FRAMES,
    SPACING_LAWS,
    VOLUME_BUILDERS,
    WEIGHTING_PRESETS,
    BrightnessConfig,
    CameraConfig,
    ExportConfig,
    FieldLinesConfig,
    GridConfig,
    IcarusRunConfig,
    InputConfig,
    OutputConfig,
    QMapConfig,
    RenderConfig,
    SolutionFrame,
    VolumeConfig,
    WeightingConfig,
    frame_from_provenance,
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
from qorona.io.readers import category_of, resolve_model
from qorona.io.readers.icarus import read_icarus_par, snapshot_time
from qorona.radiation.punch_l3 import PunchL3Result

#: Default image dimension used only when exactly one of ``--width`` / ``--height`` is supplied; the
#: full default ``pixels`` otherwise lives once on :class:`~qorona.config.CameraConfig`.
_DEFAULT_DIMENSION = 1024

#: Default image dimension of the white-light product, matching the Q⊥ render; the camera flags
#: override it.
_WL_DEFAULT_DIMENSION = 1024

#: Default field of view (full width, R_sun) of the white-light product, wider than the docs'
#: canonical ``--fov 8`` eclipse framing so the streamers keep headroom instead of touching the
#: frame. ``--fov`` overrides it.
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
        ephemeris.derive_jd(value)
    except Exception as error:
        raise click.BadParameter(str(error)) from error
    return value


def _canonical_frame(ctx: click.Context, param: click.Parameter, value: str | None) -> str | None:
    """Normalise a ``--frame`` spelling to its canonical name (``heeq`` is Stonyhurst)."""
    return FRAME_ALIASES.get(value, value) if value is not None else None


_FRAME_CHOICES = click.Choice(SOLUTION_FRAMES + tuple(FRAME_ALIASES))


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
        "--frame",
        type=_FRAME_CHOICES,
        default=None,
        callback=_canonical_frame,
        section="Input",
        help="Heliographic frame the solution's longitudes are in: carrington or stonyhurst "
        "(heeq accepted). Establish it for your input and declare it; --observer, --export fits "
        "and --export sunjson resolve against it and warn when it is undeclared (Carrington is "
        "then assumed).",
    ),
    option(
        "--variables",
        default=None,
        section="Input",
        advanced=True,
        help="Comma-separated state-variable names (reader override).",
    ),
)
_composition_option = option(
    "--mean-molecular-weight",
    type=click.FloatRange(min=0.0, min_open=True),
    default=None,
    section="Input",
    advanced=True,
    help="Mean molecular weight μ of the model, in p = 2 rho k T/(μ m_H); the electron density is "
    "rho/(μ m_p) (default: the model's own, COCONUT 1.27, MAS 1, Icarus 1).",
)
#: The run settings of an Icarus input. Icarus snapshots are in a frame that turns with the Sun;
#: these settings anchor it, so the observer and the FITS export can be placed.
_icarus_options = _compose(
    option(
        "--magnetogram-time",
        default=None,
        callback=_valid_timestamp,
        section="Icarus input",
        advanced=True,
        help="UTC ISO-8601 time of the run's magnetogram. With --relaxation, it sets the snapshot "
        "time and the Icarus frame. Without --par, the CME insertion delay and delta_phi are 0.",
    ),
    option(
        "--relaxation",
        type=float,
        default=None,
        section="Icarus input",
        advanced=True,
        help="Relaxation period before the magnetogram time, in days (0 if the run starts at the "
        "magnetogram time). No default.",
    ),
    option(
        "--par",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
        section="Icarus input",
        advanced=True,
        help="The run's .par file, read for the magnetogram time, relaxation, CME insertion and "
        "delta_phi. --magnetogram-time and --relaxation override it.",
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
        help="Inner shell radius in R_sun (default: the category shell; corona 1.0, "
        "heliosphere the native inner boundary).",
    ),
    option(
        "--outer-radius",
        type=float,
        default=None,
        section="Field grid",
        help="Outer shell radius in R_sun (default: the category shell; corona 12.5, "
        "heliosphere the native outer boundary).",
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
            "k-NN MLS neighbour count for resampling (default 48, scales up on finer meshes; "
            "lower is sharper but noisier)."
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
        help=(
            "Closed-loop polarity: neutral cancels the opposite feet to 0, dominant takes "
            "the stronger foot (default neutral)."
        ),
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
    default_dimension: int = _DEFAULT_DIMENSION,
    default_fov: float | None = None,
    *,
    instrument_in_short_help: bool = False,
) -> Callable[[Callable], Callable]:
    """The shared camera option group; the defaults state a command's image and FOV defaults.

    A ``None`` ``default_fov`` means the coronal orthographic FOV is the locked constant, stated as
    such in the ``--fov`` help. A concrete ``default_fov`` (the ``wl`` product's wider frame)
    states its own number. ``instrument_in_short_help`` lists ``--instrument`` in the short help
    (the ``wl`` command, where the PUNCH frames are used); elsewhere it is in ``--help-all``.
    """
    coronal_fov = f"{default_fov:g}" if default_fov is not None else "12.5"
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
            help="Sub-observer heliographic latitude in degrees (default: corona 0 (edge-on); "
            "heliosphere 0 in perspective, 90 for the orthographic equatorial-plane overview).",
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
            help="Orthographic field of view (full width) in R_sun (default: corona "
            f"{coronal_fov}, heliosphere spans the domain). Rejected under the perspective camera.",
        ),
        option(
            "--fov-deg",
            type=float,
            default=None,
            section="Camera",
            help="Perspective field of view (full width) in degrees, below 180: the angle for ARC, "
            "the plate scale times the width for TAN (default: heliosphere 92.16, the PUNCH "
            "mosaic width; a coronal perspective view needs it).",
        ),
        option(
            "--projection",
            type=click.Choice(CAMERA_PROJECTIONS),
            default=None,
            section="Camera",
            help="Camera: orthographic (parallel lines of sight, field of view in R_sun) or "
            "perspective (an observer at a finite distance, field of view in degrees) "
            "(default: corona orthographic, heliosphere perspective).",
        ),
        option(
            "--instrument",
            type=click.Choice(tuple(CAMERA_INSTRUMENTS)),
            default=None,
            section="Camera",
            advanced=not instrument_in_short_help,
            help="Instrument frame of the perspective camera, as in the PUNCH Level 3 headers: "
            'punch-mosaic (ARC, 0.0225°/px, 4096x4096) or punch-nfi (TAN, 30"/px, 2048x2048). '
            "Selects the perspective camera on any input; --fov-deg, --width and --height "
            "override the frame.",
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
            type=click.Choice(ephemeris.OBSERVERS),
            default=None,
            section="Camera",
            help="Named observer: derive the sub-observer longitude/latitude and the observer "
            "distance from its ephemeris at the timestamp, in the solution frame (--frame). "
            "earth resolves offline; stereo-a, solo and psp query JPL Horizons (network).",
        ),
        option(
            "--observer-distance",
            type=float,
            default=None,
            section="Camera",
            help="Observer distance from Sun centre in AU: the perspective camera's position and "
            "the FITS observer distance (default 1; --observer derives it).",
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
        help="Body/occulter radius in R_sun (default: corona 1.0, heliosphere the inner shell).",
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
        help="Line-of-sight sample spacing in R_sun (default: corona 0.02, heliosphere 2 x outer "
        "radius/1250, or fov/1250 with --projection orthographic).",
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
        "--component",
        type=click.Choice(BRIGHTNESS_COMPONENTS),
        default=None,
        section="Brightness",
        help="Brightness component: polarized (pB) or total (white-light) (default polarized).",
    ),
    option(
        "--brightness-scale",
        type=click.Choice(BRIGHTNESS_SCALES),
        default=None,
        section="Brightness",
        help="Units of the raw frames (npz and the FITS PB/TOTAL extensions): auto gives "
        "absolute mean solar brightness (MSB, as PUNCH) when the reader knows the density's "
        "physical unit (COCONUT, MAS, Icarus), else relative; relative always gives relative "
        "frames (default auto).",
    ),
    option(
        "--vignette",
        type=click.Choice(BRIGHTNESS_VIGNETTES),
        default=None,
        section="Brightness",
        help="Display treatment of the selected --component: newkirk divides by the brightness of "
        "the smooth Newkirk background corona; adaptive self-calibrates the same curve family "
        "to the image's own falloff and amplifies its structure; wow whitens the raw frame's "
        "wavelet spectrum (needs sunkit-image and watroo); none keeps the raw falloff "
        "(default newkirk; adaptive for COCONUT input).",
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
        help="Also write raw data beside the PNG; repeatable. npz: both raw frames + image "
        "coordinates (R_sun, or degrees for the perspective camera). fits: the display frame + "
        "raw frames, WCS-registered (needs a timestamp).",
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
        help="Line-of-sight sample spacing in R_sun (default: corona 0.02, heliosphere 2 x outer "
        "radius/1250, or fov/1250 with --projection orthographic).",
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
        help="Body/occulter radius in R_sun (default: corona 1.02, heliosphere 1.02 x inner "
        "shell, just above the boundary so the overwhelming near-limb ring does not eat the "
        "display stretch).",
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
    option(
        "--punch-l3",
        is_flag=True,
        default=False,
        section="Brightness",
        help="Apply the PUNCH Level 3 background treatment: for each pixel, subtract its usual "
        "brightness over 16 days before and after, as PUNCH does, so the pixel values compare "
        "with PUNCH Level 3 images. Needs --instrument and the solution time; the observer is "
        "Earth.",
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
        frame=kw.get("frame"),
        variables=variables,
        icarus_run=kw.get("icarus_run"),
        mean_molecular_weight=kw.get("mean_molecular_weight"),
    )


#: The flags that carry the run settings of an Icarus input.
_ICARUS_RUN_KEYS = ("magnetogram_time", "relaxation", "par")


def _apply_icarus_run(kw: dict[str, Any]) -> None:
    """Resolve the Icarus run settings of a raw input into its timestamp and its frame.

    An Icarus snapshot is in the Icarus frame, so ``--frame`` is refused for it. With the run
    settings, the snapshot time follows from ``TIME``, so ``--timestamp`` is refused too. Without
    them, the frame stays unresolved, and :func:`_frame_gate` stops every frame-sensitive feature.
    Other inputs must not carry the Icarus flags.
    """
    given = ["--" + key.replace("_", "-") for key in _ICARUS_RUN_KEYS if kw.get(key) is not None]
    if resolve_model(kw["input_path"], model=kw.get("model")) != "icarus":
        if given:
            raise click.ClickException(f"{', '.join(given)}: these flags apply to Icarus input")
        return
    if kw.get("frame") is not None:
        raise click.ClickException(
            "--frame: an Icarus snapshot is in the Icarus frame; drop --frame and pass "
            "--magnetogram-time and --relaxation (or --par)"
        )
    kw["frame"] = ICARUS_FRAME
    if not given:
        return
    try:
        base = read_icarus_par(kw["par"]) if kw.get("par") is not None else None
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    magnetogram_time = kw.get("magnetogram_time") or (base.magnetogram_time if base else None)
    relaxation = kw["relaxation"] if kw.get("relaxation") is not None else None
    if relaxation is None and base is not None:
        relaxation = base.relaxation
    if magnetogram_time is None:
        raise click.ClickException("Icarus input: pass --magnetogram-time (or --par)")
    if relaxation is None:
        raise click.ClickException(
            "Icarus input: pass --relaxation in days (0 if the run starts at the magnetogram "
            "time), or --par"
        )
    try:
        run = IcarusRunConfig(
            magnetogram_time=magnetogram_time,
            relaxation=relaxation,
            cme_insertion=base.cme_insertion if base else 0.0,
            delta_phi=base.delta_phi if base else 0.0,
        )
        ephemeris.icarus_epoch(run, 0.0)
    except ValueError as error:
        raise click.ClickException(f"Icarus run settings: {error}") from error
    time_hours = snapshot_time(kw["input_path"])
    if time_hours is None:
        raise click.ClickException(
            f"{Path(kw['input_path']).name} has no TIME field value; its snapshot time is unknown"
        )
    if kw.get("timestamp") is not None:
        raise click.ClickException(
            "--timestamp: the snapshot time comes from the Icarus run settings and TIME; drop "
            "--timestamp"
        )
    kw["timestamp"] = ephemeris.icarus_epoch(run, time_hours)
    kw["icarus_run"] = run


def _solution_frame(kw: dict[str, Any]) -> SolutionFrame:
    """The solution frame of a raw input, as the frame-sensitive consumers take it."""
    run: IcarusRunConfig | None = kw.get("icarus_run")
    return run if run is not None else kw.get("frame")


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


def _observer_label(kw: dict[str, Any]) -> str:
    """The flag that set the observer, as the messages name it: ``--punch-l3`` sets Earth itself."""
    return "--punch-l3" if kw.get("punch_l3") else f"--observer {kw['observer']}"


def _observer_gate(kw: dict[str, Any], timestamp: str | None) -> None:
    """Reject ``--observer`` calls that conflict with manual pointing or lack an epoch.

    Called by ``_camera_config`` for every command; commands whose expensive compute precedes
    camera resolution (the ``wl`` raw-solution path) also call it up front to fail fast.
    """
    observer = kw.get("observer")
    if observer is None:
        return
    if kw.get("longitude") is not None or kw.get("latitude") is not None:
        raise click.UsageError(
            f"--observer {observer} derives the sub-observer point from the ephemeris; "
            "drop --longitude/--latitude, or drop --observer to point manually"
        )
    if kw.get("observer_distance") is not None:
        raise click.UsageError(
            f"--observer {observer} derives the observer distance from the ephemeris; "
            "drop --observer-distance, or drop --observer to place the observer manually"
        )
    label = _observer_label(kw)
    if timestamp is None and kw.get("frame") == ICARUS_FRAME:
        raise click.UsageError(
            f"{label}: an Icarus snapshot is in the Icarus frame, which needs the run settings; "
            "pass --magnetogram-time and --relaxation (or --par)"
        )
    if timestamp is None:
        raise click.UsageError(
            f"{label} needs an epoch; pass --timestamp (or bake the volume with one)"
        )


def _declared_frame(
    flag: str | None, build_prov: dict[str, Any] | None = None, timestamp: str | None = None
) -> SolutionFrame:
    """The declared solution frame: the command's ``--frame`` ``flag``, else the one baked into
    the volume, else ``None`` (undeclared; every consumer takes it as Carrington).

    A volume built from Icarus input keeps its frame and its snapshot time, so a ``flag`` or a
    ``timestamp`` override is refused for it.
    """
    recorded = (build_prov or {}).get("input", {})
    baked = frame_from_provenance(recorded)
    if isinstance(baked, IcarusRunConfig) or baked == ICARUS_FRAME:
        if flag is not None:
            raise click.ClickException(
                "--frame: this volume was built from Icarus input, which is in the Icarus frame"
            )
        if timestamp is not None and isinstance(baked, IcarusRunConfig):
            raise click.ClickException(
                "--timestamp: this volume was built from Icarus input with its run settings, "
                "which set the snapshot time"
            )
        return baked
    return flag or baked


#: How the frame warning names the exports that convert between the solution frame and Carrington.
_FRAME_EXPORTS = {"fits": "the FITS export", "sunjson": "the SunJSON export"}


def _frame_consumers(kw: dict[str, Any], output_cfg: OutputConfig) -> list[str]:
    """Name the requested features that depend on the solution frame: ``--observer`` and the
    ``fits`` / ``sunjson`` exports."""
    consumers = [_observer_label(kw)] if kw.get("observer") else []
    consumers += [name for fmt, name in _FRAME_EXPORTS.items() if fmt in output_cfg.export_formats]
    return consumers


def _frame_gate(frame: SolutionFrame, consumers: list[str]) -> None:
    """Warn once when frame-sensitive ``consumers`` run on an undeclared frame; a declared frame,
    or a run that requested nothing frame-sensitive, is silent. An Icarus frame without its run
    settings stops the run, because no consumer can resolve it."""
    if frame == ICARUS_FRAME and consumers:
        raise click.ClickException(
            f"{' and '.join(consumers)}: an Icarus snapshot is in the Icarus frame, which needs "
            "the run settings; pass --magnetogram-time and --relaxation (or --par) with the "
            "snapshot, and rebuild a .qor volume with them"
        )
    if frame is not None or not consumers:
        return
    print_warning(
        f"{' and '.join(consumers)}: no --frame declared, so the solution is taken as "
        "Carrington-aligned; declare its frame with --frame carrington or --frame stonyhurst, "
        "or point manually with --longitude/--latitude if the frame is unknown"
    )


def _camera_fields(kw: dict[str, Any], default_dimension: int) -> dict[str, Any]:
    """Return the :class:`CameraConfig` fields the camera flags set, the instrument frame applied.

    A named ``--instrument`` fills the perspective frame (angular width, pixels, projection code,
    reference pixel). Explicit ``--fov-deg``, ``--width`` and ``--height`` win over it. The
    instrument's reference pixel is kept only while the image keeps the instrument's shape.
    """
    fields = _present(
        kw,
        "projection",
        "instrument",
        "longitude",
        "latitude",
        "roll",
        "fov",
        "fov_deg",
        "observer_distance",
    )
    height, width = kw.get("height"), kw.get("width")
    instrument = kw.get("instrument")
    if instrument is not None:
        frame_preset = CAMERA_INSTRUMENTS[instrument]
        fields.setdefault("fov_deg", frame_preset.fov_deg)
        fields["projection_code"] = frame_preset.projection_code
        fields["pixels"] = (
            height if height is not None else frame_preset.pixels[0],
            width if width is not None else frame_preset.pixels[1],
        )
        if fields["pixels"] == frame_preset.pixels:
            fields["reference_pixel"] = frame_preset.reference_pixel
    elif height is not None or width is not None or default_dimension != _DEFAULT_DIMENSION:
        fields["pixels"] = (
            height if height is not None else default_dimension,
            width if width is not None else default_dimension,
        )
    return fields


def _projection_gate(fields: dict[str, Any], category: str) -> CameraConfig:
    """Build the camera config from ``fields`` with its projection resolved by ``category``.
    Raise a usage error when the camera flags contradict each other."""
    try:
        return pipeline.resolve_projection(CameraConfig(**fields), category)
    except ValueError as error:
        raise click.UsageError(str(error)) from error


def _camera_config(
    kw: dict[str, Any],
    *,
    category: str,
    default_dimension: int = _DEFAULT_DIMENSION,
    timestamp: str | None = None,
    frame: SolutionFrame = None,
) -> CameraConfig:
    """Build the camera config from the camera flags, with its projection resolved.

    A named ``--observer`` sets the sub-observer point and the observer distance from its
    ephemeris. The projection resolves by ``category`` here, so contradicting flags fail before
    any expensive stage.
    """
    fields = _camera_fields(kw, default_dimension)
    _observer_gate(kw, timestamp)
    observer = kw.get("observer")
    if observer is not None:
        assert timestamp is not None  # _observer_gate rejects the missing-epoch case
        try:
            longitude, latitude, distance = ephemeris.sub_observer_point(observer, timestamp, frame)
        except ValueError as error:
            raise click.ClickException(str(error)) from error
        fields["longitude"], fields["latitude"] = longitude, latitude
        fields["observer_distance"] = distance
    return _projection_gate(fields, category)


def _resolved_camera(
    camera_cfg: CameraConfig,
    category: str,
    inner: float,
    outer: float,
    *,
    coronal_fov: float | None = None,
) -> CameraConfig:
    """Resolve the camera's shell-dependent defaults at the CLI edge, with a friendly error on
    failure. ``coronal_fov`` overrides the coronal orthographic width (the ``wl`` frame)."""
    extra = {} if coronal_fov is None else {"coronal_fov": coronal_fov}
    try:
        resolved = pipeline.resolve_camera(camera_cfg, category, inner, outer, **extra)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    _warn_frame_past_shell(resolved, outer)
    return resolved


def _warn_frame_past_shell(camera_cfg: CameraConfig, outer: float) -> None:
    """Warn when the image frame reaches past the outer radius of the solution shell.

    The frame reach is measured along the image axes from Sun centre. Lines of sight past the shell
    carry no corona, and lines of sight near the shell edge are cut short. A perspective observer
    inside the shell sees the shell on every line of sight, so that case is silent.
    """
    import math

    from astropy import units as u

    if camera_cfg.projection == "perspective":
        assert camera_cfg.fov_deg is not None  # resolved by pipeline.resolve_camera
        distance = float((camera_cfg.observer_distance * u.AU).to_value(u.R_sun))
        if distance <= outer:
            return
        half = 0.5 * camera_cfg.fov_deg
        if camera_cfg.projection_code == "tan":
            half = math.degrees(math.atan(math.radians(half)))
        edge = math.degrees(math.asin(outer / distance))
        if half <= edge:
            return
        reach = f"{half:.1f} degrees from Sun centre"
        shell = f"{outer:g} R_sun ({edge:.1f} degrees from the observer)"
    else:
        assert camera_cfg.fov is not None  # resolved by pipeline.resolve_camera
        half = 0.5 * camera_cfg.fov
        if half <= outer:
            return
        reach = f"{half:g} R_sun from Sun centre"
        shell = f"{outer:g} R_sun"
    print_warning(
        f"The frame reaches {reach}, but the solution shell ends at {shell}. Lines of sight past "
        "the shell are empty. To fill the frame, raise --outer-radius up to the solution's own "
        "outer boundary (for a .qor volume, rebuild it with that radius)."
    )


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
        "component",
        "vignette",
        "crossover",
        "step",
        "occult",
        "r_occult",
        "occult_softness",
        "scaling",
        "brightness_scale",
    )
    if kw.get("mgn"):
        fields["mgn"] = True
    if kw.get("limb_darkening") is not None:
        fields["u"] = kw["limb_darkening"]
    if kw.get("percentiles") is not None:
        fields["percentiles"] = tuple(kw["percentiles"])
    if workers is not None:
        fields["workers"] = workers
    if kw.get("punch_l3"):
        fields["punch_l3"] = True
    return BrightnessConfig(**fields)


def _punch_l3_gate(kw: dict[str, Any]) -> None:
    """Check the flags of the PUNCH Level 3 treatment and set its fixed choices.

    The treatment needs a PUNCH instrument frame, views from Earth, and uses the PUNCH quicklook
    display, so other observers, manual pointing and the display flags are refused.
    """
    if not kw.get("punch_l3"):
        return
    if kw.get("instrument") is None:
        raise click.UsageError(
            "--punch-l3 makes a PUNCH Level 3 frame; pass --instrument punch-mosaic or "
            "--instrument punch-nfi"
        )
    observer = kw.get("observer")
    if observer not in (None, "earth"):
        raise click.UsageError(f"--punch-l3: PUNCH observes from Earth; drop --observer {observer}")
    pointing = [
        "--" + key.replace("_", "-")
        for key in ("longitude", "latitude", "observer_distance")
        if kw.get(key) is not None
    ]
    if pointing:
        raise click.UsageError(
            f"{', '.join(pointing)}: --punch-l3 places the observer at Earth; drop these flags"
        )
    display = [
        "--" + key for key in ("vignette", "scaling", "percentiles") if kw.get(key) is not None
    ] + (["--mgn"] if kw.get("mgn") else [])
    if display:
        raise click.UsageError(
            f"{', '.join(display)}: --punch-l3 uses the PUNCH quicklook display; drop these flags"
        )
    kw["observer"] = "earth"
    kw["vignette"] = "none"


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
#: Provenance records the *resolved* shell, so a heliospheric rebuild spells ``--inner-radius``
#: explicitly (its resolved inner radius is the native boundary, not the coronal 1.0 default).
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
    # An Icarus volume takes its time and frame from the run settings, which --timestamp and
    # --frame cannot carry; the Q-map needs neither.
    if inp.get("frame") != ICARUS_FRAME:
        if inp.get("timestamp"):
            flags.append(f"--timestamp {inp['timestamp']}")
        if inp.get("frame"):
            flags.append(f"--frame {inp['frame']}")
    if inp.get("mean_molecular_weight") is not None:
        flags.append(f"--mean-molecular-weight {inp['mean_molecular_weight']:g}")
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
    inner_radius: float, outer_radius: float, qmap_cfg: QMapConfig, build_prov: dict[str, Any]
) -> None:
    """Warn when the volume's outer radius does not sit at the Q-map radius.

    The canonical Q-map maps the domain ``[1, r]``: baking with the outer radius at the map radius
    puts the heliospheric current sheet on the Q⊥ ridges. A deeper volume slices the ``[1, outer]``
    mapping, where the current sheet can drift off the ridges; a shallower one cannot reach that
    radius, so the map is clamped to the boundary. Either way, print the build command that produces
    it right. ``inner_radius``, ``outer_radius`` and ``qmap_cfg.radius`` all arrive resolved off the
    loaded volume, so all are concrete numbers here.
    """
    radius = qmap_cfg.radius
    assert radius is not None  # the Q-map config is resolved at the CLI edge before this call
    if radius > outer_radius * (1.0 + 1e-6):
        print_warning(
            f"r = {radius:g} R_sun is outside this volume (outer = {outer_radius:g}); the map was "
            f"clamped to the boundary. To map at r = {radius:g}, rebuild at that radius:"
        )
    elif outer_radius > radius * 1.01:
        print_warning(
            f"Q-map at r = {radius:g} R_sun sliced from an outer = {outer_radius:g} volume: the "
            f"current sheet may sit off the Q⊥ ridges (a slice of the "
            f"[{inner_radius:g}, {outer_radius:g}] "
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
    lines to SunJSON for JHelioviewer, `wl` renders the white-light and polarized brightness, and
    `info` inspects a file.
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
@_icarus_options
@_composition_option
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
    _apply_icarus_run(kw)
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
@option(
    "--frame",
    type=_FRAME_CHOICES,
    default=None,
    callback=_canonical_frame,
    section="Input",
    advanced=True,
    help="Declare or override the volume's solution frame: carrington or stonyhurst (heeq "
    "accepted).",
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
    frame: str | None,
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
    declared_frame = _declared_frame(frame, build_prov, timestamp)
    _frame_gate(declared_frame, _frame_consumers(kw, output_cfg))
    # The volume renders as built, so the category is the one recorded in its provenance (old
    # volumes default to corona).
    category = build_prov.get("input", {}).get("category", "corona")
    camera_cfg = _camera_config(
        kw, category=category, timestamp=resolved_timestamp, frame=declared_frame
    )
    # Pre-resolve the shell-dependent sentinels at the edge so provenance and the stamp spell the
    # resolved numbers; the pipeline's own resolution then no-ops.
    inner, outer = float(volume.grid.radii[0]), float(volume.grid.radii[-1])
    camera_cfg = _resolved_camera(camera_cfg, category, inner, outer)
    render_cfg = pipeline.resolve_render(render_cfg, category, camera_cfg, inner, outer)
    start = time.perf_counter()
    result = pipeline.render_volume(
        volume,
        camera_cfg,
        weighting_cfg,
        render_cfg,
        category=category,
        density=density,
        show_progress=show_progress,
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
        frame_override=frame,
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
    "--radius",
    type=float,
    default=None,
    section="Q-map",
    help="Shell radius in R_sun (default: corona 3.0, heliosphere 3 x inner shell).",
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
    # Resolve the shell radius off the loaded volume's category before the warning and the slice, so
    # the warning, the provenance, and the stamp all read the same concrete radius.
    category = build_prov.get("input", {}).get("category", "corona")
    qmap_cfg = pipeline.resolve_qmap(qmap_cfg, category, float(volume.grid.radii[0]))
    _warn_qmap_outer_radius(
        float(volume.grid.radii[0]), float(volume.grid.radii[-1]), qmap_cfg, build_prov
    )
    result = pipeline.qmap_from_volume(volume, qmap_cfg, category=category)
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
@_icarus_options
@_composition_option
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
    _apply_icarus_run(kw)
    kw["device"] = device
    show_progress = not quiet
    input_cfg = _input_config(kw)
    grid_cfg = _grid_config(kw)
    volume_cfg = _volume_config(kw, workers)
    output_cfg = _output_config(kw, output_path)
    if "fits" in output_cfg.export_formats and kw.get("timestamp") is None:
        raise click.ClickException(
            "--export fits needs a timestamp for DATE-OBS and the observer ephemeris; "
            "pass --timestamp"
        )
    _frame_gate(input_cfg.solution_frame, _frame_consumers(kw, output_cfg))
    camera_cfg = _camera_config(
        kw,
        category=_solution_category(input_path, kw),
        timestamp=kw.get("timestamp"),
        frame=input_cfg.solution_frame,
    )
    weighting_cfg = _weighting_config(kw)
    render_cfg = _render_config(kw, workers)

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

    # Resolve the shell-dependent camera / render sentinels off the built volume's shell and the
    # solution category, so the render and its provenance / stamp all read the resolved numbers.
    category = build_prov.get("input", {}).get("category", "corona")
    inner, outer = float(volume.grid.radii[0]), float(volume.grid.radii[-1])
    camera_cfg = _resolved_camera(camera_cfg, category, inner, outer)
    render_cfg = pipeline.resolve_render(render_cfg, category, camera_cfg, inner, outer)

    render_start = time.perf_counter()
    result = pipeline.render_volume(
        volume,
        camera_cfg,
        weighting_cfg,
        render_cfg,
        category=category,
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
    table.add_row("Category", meta.category)
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
            f"{kw['timestamp']} UTC → CR {ephemeris.derive_cr(kw['timestamp'])} / "
            f"JD {ephemeris.derive_jd(kw['timestamp']):.3f}",
        )
    if kw.get("frame"):
        table.add_row("Frame", kw["frame"])
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
@_icarus_options
@_grid_options
@_fieldlines_options
@_camera_options()
@option(
    "--export",
    "export_formats",
    type=click.Choice(FIELDLINES_EXPORT_FORMATS),
    multiple=True,
    section="Output",
    advanced=True,
    help="Also write the drawn line bundle beside the PNG; repeatable. sunjson: the kept "
    "lines with their draw colours, for JHelioviewer (needs a timestamp).",
)
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
    ``--export sunjson`` also writes exactly the drawn bundle as SunJSON for JHelioviewer beside
    the PNG. A self-contained command: it traces the field directly and does not use a built
    volume.
    """
    kw["input_path"] = input_path
    _apply_icarus_run(kw)
    show_progress = not quiet
    input_cfg = _input_config(kw)
    grid_cfg = _grid_config(kw)
    fieldlines_cfg = _fieldlines_config(kw, workers)
    output_cfg = _output_config(kw, output_path)
    if "sunjson" in output_cfg.export_formats and input_cfg.timestamp is None:
        raise click.ClickException(
            "--export sunjson needs a timestamp for the JHelioviewer timeline; pass --timestamp"
        )
    _frame_gate(input_cfg.solution_frame, _frame_consumers(kw, output_cfg))
    camera_cfg = _camera_config(
        kw,
        category=_solution_category(input_path, kw),
        timestamp=kw.get("timestamp"),
        frame=input_cfg.solution_frame,
    )

    print_step(f"Tracing field lines from [bold]{input_path.name}[/bold]")
    start = time.perf_counter()
    field = pipeline.build_field(input_cfg, grid_cfg, show_progress=show_progress)
    field_time = time.perf_counter() - start
    # Resolve the camera sentinels off the field's shell and category, so the render and its
    # provenance / stamp read the same resolved camera.
    category = field.category
    camera_cfg = _resolved_camera(
        camera_cfg, category, field.domain.inner_radius, field.domain.outer_radius
    )
    render_start = time.perf_counter()
    result = pipeline.render_fieldlines(
        field, fieldlines_cfg, camera_cfg, category=category, show_progress=show_progress
    )
    render_time = time.perf_counter() - render_start

    provenance = pipeline.fieldlines_provenance(
        input_cfg, grid_cfg, fieldlines_cfg, camera_cfg, output_cfg, result, field=field
    )
    written = write_fieldlines(result, output_cfg, provenance)
    if "sunjson" in output_cfg.export_formats:
        written.append(
            write_fieldlines_json(
                result.lines,
                output_cfg.export_path("sunjson"),
                provenance,
                keep=result.keep,
                colours=result.colours,
            )
        )
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
    help="Destination SunJSON file (.json).",
)
@_input_options
@_icarus_options
@_grid_options
@_export_options
@_workers_option
@_quiet_option
def export_lines(
    input_path: Path, output_path: Path, workers: int | None, quiet: bool, **kw: Any
) -> None:
    """Export traced field lines of a solution to SunJSON for JHelioviewer.

    Reads the solution, traces field lines seeded on a uniform longitude/latitude grid
    (``--seeds``, default 100x100, on the inner boundary unless ``--seed-radius`` overrides it),
    and writes the polylines with their open/closed topology, coloured by inner-foot ``B·r̂``
    polarity. Drop the file onto a running JHelioviewer to load it; ``--timestamp`` places it on
    the timeline. A self-contained command: it traces the field directly and does not use a built
    volume. The file schema is documented in ``qorona/io/fieldlines_export.py``.
    """
    kw["input_path"] = input_path
    _apply_icarus_run(kw)
    show_progress = not quiet
    input_cfg = _input_config(kw)
    grid_cfg = _grid_config(kw)
    export_cfg = _export_config(kw, workers)
    if input_cfg.timestamp is None:
        raise click.ClickException(
            "SunJSON needs an observation time for the JHelioviewer timeline; pass --timestamp"
        )
    _frame_gate(input_cfg.solution_frame, [_FRAME_EXPORTS["sunjson"]])

    print_step(f"Exporting field lines from [bold]{input_path.name}[/bold]")
    start = time.perf_counter()
    field = pipeline.build_field(input_cfg, grid_cfg, show_progress=show_progress)
    field_time = time.perf_counter() - start
    trace_start = time.perf_counter()
    lines = pipeline.export_lines(field, export_cfg, show_progress=show_progress)
    trace_time = time.perf_counter() - trace_start

    provenance = pipeline.export_provenance(input_cfg, grid_cfg, export_cfg, field, lines)
    written = write_fieldlines_json(lines, output_path, provenance, field=field)
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


def _solution_category(input_path: Path, kw: dict[str, Any]) -> str:
    """Return the solution category of a raw ``input_path`` from the reader registry, without
    reading the file (``--model`` short-circuits the lookup)."""
    return category_of(resolve_model(input_path, model=kw.get("model")))


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
    "mean_molecular_weight",
    *_ICARUS_RUN_KEYS,
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
@_icarus_options
@_composition_option
@_grid_options
@_brightness_options
@_camera_options(_WL_DEFAULT_DIMENSION, _WL_DEFAULT_FOV, instrument_in_short_help=True)
@_annotate_options
@_workers_option
@_quiet_option
def wl(input_path: Path, output_path: Path, workers: int | None, quiet: bool, **kw: Any) -> None:
    """Render the white-light / polarized-brightness corona of a solution from a viewpoint.

    INPUT is either a raw solution (read and resampled, as ``build`` would) or a built ``.qor``
    volume artifact (whose stored electron density is reused, skipping the resample). The
    Thomson-scattering brightness is integrated over the density along each line of sight: the
    polarized brightness pB by default, or the total white-light brightness with ``--component
    total``. The component is finished by two display stages: the ``--vignette`` treatment
    (``newkirk`` by default, ``adaptive`` by default for COCONUT input, ``wow`` for
    wavelet whitening, ``none`` for the raw falloff) and optional ``--mgn`` fine-structure
    enhancement. ``--export npz`` also writes the raw frames (both pB and total) with their
    image coordinates. ``--export fits`` writes the display frame plus the raw frames as
    a WCS-registered FITS and needs a timestamp. Needs only the density; it neither builds nor
    uses the Q⊥ payload.

    ``--punch-l3`` applies the PUNCH Level 3 background treatment to a PUNCH instrument frame
    seen from Earth, and shows it with the PUNCH quicklook display.
    """
    kw["input_path"] = input_path
    show_progress = not quiet
    from_volume = zipfile.is_zipfile(input_path)
    if not from_volume:
        _apply_icarus_run(kw)
    _punch_l3_gate(kw)
    output_cfg = _output_config(kw, output_path)
    # For a raw solution the --observer and FITS timestamp gates fire here, before the expensive
    # resample; a volume artifact may carry the epoch itself, so its gates wait for the fast
    # load below.
    if not from_volume:
        _observer_gate(kw, kw.get("timestamp"))
        _projection_gate(
            _camera_fields(kw, _WL_DEFAULT_DIMENSION), _solution_category(input_path, kw)
        )
        if "fits" in output_cfg.export_formats and kw.get("timestamp") is None:
            raise click.ClickException(
                "--export fits needs a timestamp for DATE-OBS and the observer ephemeris; "
                "pass --timestamp"
            )
        _frame_gate(_solution_frame(kw), _frame_consumers(kw, output_cfg))

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
    declared_frame = (
        _declared_frame(kw.get("frame"), build_prov, kw.get("timestamp"))
        if from_volume
        else _solution_frame(kw)
    )
    if from_volume:
        _frame_gate(declared_frame, _frame_consumers(kw, output_cfg))
    # The input's model and solution category drive the edge defaults: a volume renders as built
    # (its recorded category, old volumes defaulting to corona), a raw solution goes through the
    # registry. The coronal orthographic wl frame keeps its wider constant FOV (resolved below).
    model = _input_model(from_volume, input_path, build_prov, kw.get("model"))
    category = (
        (build_prov or {}).get("input", {}).get("category", "corona")
        if from_volume
        else category_of(model)
    )
    camera_cfg = _camera_config(
        kw,
        category=category,
        default_dimension=_WL_DEFAULT_DIMENSION,
        timestamp=resolved_timestamp,
        frame=declared_frame,
    )

    if kw.get("vignette") is None and model == "coconut":
        # Model-aware default: the self-calibrating adaptive channel for COCONUT input. The
        # schema's newkirk default stands for every other input, heliospheric ones included.
        kw["vignette"] = "adaptive"
    brightness_cfg = _brightness_config(kw, workers)

    # Pre-resolve the shell-dependent camera / brightness sentinels off the density's shell, so the
    # render and its provenance / stamp read the resolved numbers; the pipeline's own resolution
    # then no-ops.
    inner, outer = float(density.grid.radii[0]), float(density.grid.radii[-1])
    camera_cfg = _resolved_camera(camera_cfg, category, inner, outer, coronal_fov=_WL_DEFAULT_FOV)
    brightness_cfg = pipeline.resolve_brightness(brightness_cfg, category, camera_cfg, inner, outer)

    render_start = time.perf_counter()
    punch: PunchL3Result | None = None
    if brightness_cfg.punch_l3:
        assert resolved_timestamp is not None  # the Earth observer gate needs the epoch
        punch = pipeline.render_punch_l3(
            density,
            brightness_cfg,
            camera_cfg,
            timestamp=resolved_timestamp,
            frame=declared_frame,
            category=category,
            show_progress=show_progress,
        )
        result = punch.full
    else:
        result = pipeline.render_brightness(
            density, brightness_cfg, camera_cfg, category=category, show_progress=show_progress
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
            frame_override=kw.get("frame"),
        )
    else:
        provenance = pipeline.brightness_provenance(
            input_cfg, grid_cfg, brightness_cfg, camera_cfg, output_cfg, result, field=field
        )
    if result.unit == "MSB":
        provenance["brightness"]["mean_molecular_weight"] = density.mean_molecular_weight
    if punch is not None:
        provenance["punch_l3"] = punch.to_provenance()
    try:
        written = write_brightness(result, brightness_cfg, output_cfg, provenance, punch=punch)
    except ImportError as error:
        raise click.ClickException(str(error)) from error
    written += export_brightness(result, output_cfg, provenance, punch=punch)
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
    frame = frame_from_provenance(inp)
    if isinstance(frame, IcarusRunConfig):
        parts.append(_icarus_frame_text(frame))
    else:
        parts.append(f"{frame} frame" if frame else "carrington frame (assumed)")
    return " · ".join(parts)


def _icarus_frame_text(run: IcarusRunConfig) -> str:
    """One line on the Icarus frame and the run settings that anchor it."""
    text = f"icarus frame (magnetogram {run.magnetogram_time}, relaxation {run.relaxation:g} d"
    if run.cme_insertion:
        text += f", CME insertion {run.cme_insertion:g} d"
    if run.delta_phi:
        text += f", delta_phi {run.delta_phi:.2f}°"
    return text + ")"


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
    """The summary's camera row: the projection, the pointing, the field of view in its own unit,
    and the frame. A perspective row also names the observer distance and the instrument frame,
    noting when explicit flags changed it."""
    cam = prov["camera"]
    pixels = cam["pixels"]
    pointing = (
        f"sub-observer ({float(cam['longitude']):+.0f}°, {float(cam['latitude']):+.0f}°) · "
        f"roll {float(cam['roll']):+.0f}°"
    )
    frame = f"{int(pixels[1])}x{int(pixels[0])}"
    if cam.get("projection") != "perspective":
        return f"orthographic · {pointing} · FOV {float(cam['fov']):.0f} R☉ · {frame}"
    line = (
        f"perspective · {pointing} · {float(cam['observer_distance']):.3g} AU · "
        f"FOV {float(cam['fov_deg']):g}° {str(cam['projection_code']).upper()} · {frame}"
    )
    instrument = cam.get("instrument")
    if instrument:
        preset = CAMERA_INSTRUMENTS[instrument]
        matches = (
            float(cam["fov_deg"]) == preset.fov_deg
            and tuple(int(n) for n in pixels) == preset.pixels
        )
        line += f" · {instrument}" + ("" if matches else " (frame changed by flags)")
    return line


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
    exported = prov.get("output", {}).get("export_formats")
    if exported:
        parts.append(f"export {'+'.join(exported)} (drawn bundle)")
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
    component = "pB" if bri["component"] == "polarized" else "white-light"
    vignette = bri.get("vignette", "none")
    finish = ["raw" if vignette == "none" else str(vignette)]
    if bri.get("mgn"):
        finish.append("mgn")
    punch = prov.get("punch_l3")
    if punch:
        height, width = punch["background_pixels"]
        finish = [
            f"PUNCH L3 treatment ({punch['frames']} frames, ±{punch['window_days']:g} d, "
            f"{width}x{height} background)"
        ]
    unit = "relative"
    if bri.get("unit") == "MSB":
        mu = bri.get("mean_molecular_weight")
        unit = "MSB" if mu is None else f"MSB · μ {mu:g}"
    parts = [
        f"{component} · {'+'.join(finish)}",
        unit,
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
