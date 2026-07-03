"""The typed run-configuration schema: every parameter, its default, and its validation.

A small package of **frozen dataclasses** that is the single source of truth for Qorona's run
parameters. They mirror the pipeline call signatures one-to-one (so :mod:`qorona.pipeline` is a thin
adapter and no default or name drifts to a second place), validate eagerly in ``__post_init__`` with
a friendly message, and each exposes :meth:`to_provenance`, the JSON-safe mapping consumed by the
on-image stamp, the volume artifact, and the end-of-run summary.

The CLI populates these from ``--flags``; a YAML/TOML loader or light GUI would be just another
front-end that constructs the same dataclasses, leaving the pipeline and schema untouched.

Defaults mirror the engine functions' own (the volume builders and ``render`` define
``clamp`` / ``step`` / ``occult`` / ``supersample`` / ``paint_step`` / ``rtol``), so the schema only
*names* each number, it never re-invents one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from qorona import __version__

#: Allowed values for the string-keyed selectors, surfaced in the friendly validation messages and
#: dispatched on by :mod:`qorona.pipeline`.
SPACING_LAWS = ("logarithmic", "power", "uniform")
RESAMPLERS = ("knn-mls", "nearest-cell")
VOLUME_BUILDERS = ("paint", "per-voxel", "reference")
CLOSED_TREATMENTS = ("neutral", "dominant")
POLARITY_MODES = ("none", "hue")
WEIGHTING_PRESETS = ("large-fov", "small-fov")
THOMSON_MODES = ("K", "pB")
DISPLAY_MODES = ("balanced", "raw", "coverage")
OCCULT_MODES = ("eclipse", "opaque", "none")
ANNOTATE_POSITIONS = ("bottom-left", "bottom-right", "top-left", "top-right")
FIELDLINE_SHOW = ("all", "open", "closed")
FIELDLINE_SEEDING = ("limb", "uniform")
FIELDLINE_COLOUR = ("rainbow", "polarity")
BRIGHTNESS_FRAMES = ("polarized", "total")
BRIGHTNESS_TREATMENTS = ("raw", "radial", "newkirk", "mgn")
BRIGHTNESS_SCALINGS = ("linear", "log")
EXPORT_FORMATS = ("npz",)
DEVICE_MODES = ("auto", "gpu", "cpu")
PRECISION_MODES = ("float64", "mixed", "float32")

#: The theoretical Q⊥ floor in log₁₀; the render's default display lower clamp (mirrors
#: ``qorona.render.los.LOG_FLOOR``).
_LOG_FLOOR = math.log10(2.0)


def _require(condition: bool, message: str) -> None:
    """Raise :class:`ValueError` with ``message`` unless ``condition`` holds (fail fast)."""
    if not condition:
        raise ValueError(message)


def _one_of(value: str, allowed: tuple[str, ...], name: str) -> None:
    """Validate that ``value`` is one of ``allowed``, naming the offending field on failure."""
    _require(value in allowed, f"{name} must be one of {allowed}, got {value!r}")


def _validate_turn_guard(
    max_turn_angle: float, radius: float, weak_fraction: float, min_turns: int
) -> None:
    """Validate the sharp-turn-guard thresholds shared by the volume and field-line configs."""
    _require(
        0.0 <= max_turn_angle < 180.0,
        f"max_turn_angle must be in [0, 180) degrees (0 disables the sharp-turn guard), "
        f"got {max_turn_angle}",
    )
    _require(radius > 0.0, f"turn_guard_radius must be > 0, got {radius}")
    _require(
        weak_fraction >= 0.0,
        f"turn_guard_weak_fraction must be >= 0 (0 disables the guard), got {weak_fraction}",
    )
    _require(min_turns >= 1, f"min_turns must be >= 1, got {min_turns}")


@dataclass(frozen=True)
class InputConfig:
    """The solution to read and the optional UTC timestamp from which CR/JD are derived.

    The ``.CFmesh`` mesh carries no date, so a Carrington rotation and Julian date are *derived*
    from a user-supplied ``timestamp`` (UTC ISO-8601), never inferred; without it the CR/date
    provenance is simply absent.
    """

    path: Path
    model: str | None = None
    timestamp: str | None = None
    variables: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if self.variables is not None:
            object.__setattr__(self, "variables", tuple(self.variables))

    def to_provenance(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "model": self.model,
            "timestamp": self.timestamp,
            "variables": list(self.variables) if self.variables is not None else None,
        }


@dataclass(frozen=True)
class GridConfig:
    """The internal regular spherical (r, θ, φ) grid the resampler and interpolant operate on.

    Node counts in each direction, the inner/outer shell radii, the radial spacing law, and the
    cell→grid resampler with its k-NN neighbour count (``n_neighbors``).
    :class:`VolumeConfig`'s ``resolution_factor`` scales these node counts to
    the finer volume grid; the spacing law and radii are shared, so the field grid and the volume
    grid sit on the same shell at different pitch.
    """

    n_r: int = 192
    n_theta: int = 180
    n_phi: int = 360
    inner_radius: float = 1.0
    outer_radius: float = 12.5
    spacing: str = "logarithmic"
    resampler: str = "knn-mls"
    n_neighbors: int = 30

    def __post_init__(self) -> None:
        _require(self.n_r >= 4, f"n_r must be >= 4, got {self.n_r}")
        _require(self.n_theta >= 4, f"n_theta must be >= 4, got {self.n_theta}")
        _require(
            self.n_phi >= 4 and self.n_phi % 2 == 0,
            f"n_phi must be even and >= 4 (pole reflection), got {self.n_phi}",
        )
        _require(
            0.0 < self.inner_radius < self.outer_radius,
            f"need 0 < inner_radius < outer_radius, got "
            f"({self.inner_radius}, {self.outer_radius}) R_sun",
        )
        _one_of(self.spacing, SPACING_LAWS, "spacing")
        _one_of(self.resampler, RESAMPLERS, "resampler")
        _require(
            self.n_neighbors > 4,
            f"n_neighbors must exceed the four linear unknowns, got {self.n_neighbors}",
        )

    def to_provenance(self) -> dict[str, object]:
        return {
            "n_r": self.n_r,
            "n_theta": self.n_theta,
            "n_phi": self.n_phi,
            "inner_radius": self.inner_radius,
            "outer_radius": self.outer_radius,
            "spacing": self.spacing,
            "resampler": self.resampler,
            "n_neighbors": self.n_neighbors,
        }


@dataclass(frozen=True)
class VolumeConfig:
    """How the Q⊥ volume is baked: builder, the field→volume grid refinement, and the engine knobs.

    The default builder is ``paint``, the cheap high-resolution path that decouples interior
    resolution from cost, so it is the right default for experimentation; ``per-voxel`` (every
    voxel traced to its feet and filled from the boundary maps: complete coverage, cost
    proportional to the voxel count) and ``reference`` (the full-transport-per-voxel validation
    ground truth) are selectable. ``supersample`` and ``paint_step`` are read only by the builders
    that use them (``reference`` uses neither; ``per-voxel`` ignores ``paint_step``). ``closed`` is
    the closed-loop polarity convention baked into the volume's polarity channel (the ``paint`` and
    ``per-voxel`` builders carry it; ``reference`` omits polarity). ``device`` selects the compute
    backend (``auto`` uses the GPU when present, else the multi-core CPU kernel; ``gpu`` forces it
    and errors if absent; ``cpu`` forces the CPU). ``precision`` selects the CUDA kernel precision
    (GPU only; the CPU tiers are always float64): ``mixed`` (default) runs the tricubic field
    interpolation in float32 and everything else (stepper, error control, accumulators) in float64;
    ``float64`` is the all-double reference; ``float32`` is the experimental fully-float32 paint
    variant.
    """

    builder: str = "paint"
    resolution_factor: int = 2
    supersample: int = 4
    paint_step: float = 0.5
    closed: str = "neutral"
    rtol: float = 1e-4
    cfl: float = 0.5
    max_steps: int = 10_000
    max_reversals: int = 8
    max_turn_angle: float = 45.0
    turn_guard_radius: float = 2.0
    turn_guard_weak_fraction: float = 1.0e-5
    min_turns: int = 1
    workers: int | None = None
    device: str = "auto"
    precision: str = "mixed"

    def __post_init__(self) -> None:
        _one_of(self.builder, VOLUME_BUILDERS, "builder")
        _one_of(self.closed, CLOSED_TREATMENTS, "closed")
        _require(
            self.resolution_factor >= 1,
            f"resolution_factor must be >= 1, got {self.resolution_factor}",
        )
        _require(self.supersample >= 1, f"supersample must be >= 1, got {self.supersample}")
        _require(self.paint_step > 0.0, f"paint_step must be > 0, got {self.paint_step}")
        _require(self.rtol > 0.0, f"rtol must be > 0, got {self.rtol}")
        _require(0.0 < self.cfl < 1.0, f"cfl must satisfy 0 < cfl < 1, got {self.cfl}")
        _require(self.max_steps > 0, f"max_steps must be > 0, got {self.max_steps}")
        _require(
            self.max_reversals >= 0,
            f"max_reversals must be >= 0 (0 disables the stall guard), got {self.max_reversals}",
        )
        _validate_turn_guard(
            self.max_turn_angle,
            self.turn_guard_radius,
            self.turn_guard_weak_fraction,
            self.min_turns,
        )
        _require(
            self.workers is None or self.workers >= 1,
            f"workers must be None or >= 1, got {self.workers}",
        )
        _one_of(self.device, DEVICE_MODES, "device")
        _one_of(self.precision, PRECISION_MODES, "precision")

    def to_provenance(self) -> dict[str, object]:
        return {
            "builder": self.builder,
            "resolution_factor": self.resolution_factor,
            "supersample": self.supersample,
            "paint_step": self.paint_step,
            "closed": self.closed,
            "rtol": self.rtol,
            "cfl": self.cfl,
            "max_steps": self.max_steps,
            "max_reversals": self.max_reversals,
            "max_turn_angle": self.max_turn_angle,
            "turn_guard_radius": self.turn_guard_radius,
            "turn_guard_weak_fraction": self.turn_guard_weak_fraction,
            "min_turns": self.min_turns,
            "workers": self.workers,
            "device": self.device,
            "precision": self.precision,
        }


#: Named ``(resolution_factor, supersample)`` bundles behind the CLI's ``--quality`` preset.
#: ``standard`` derives from :class:`VolumeConfig`, so the preset and the schema defaults cannot
#: drift apart; the preset is a CLI shorthand and never appears in provenance (the resolved
#: ``resolution_factor`` / ``supersample`` do).
QUALITY_PRESETS: dict[str, tuple[int, int]] = {
    "fast": (1, 2),
    "standard": (VolumeConfig.resolution_factor, VolumeConfig.supersample),
    "high": (3, 6),
}


@dataclass(frozen=True)
class CameraConfig:
    """The orthographic plane-of-sky viewpoint: sub-observer angles (deg), roll, FOV, and pixels.

    Angles are in **degrees** here (friendly at the CLI) and converted to the camera's radians/R☉
    convention at the pipeline edge. ``pixels`` is ``(height, width)`` to match
    :class:`~qorona.geometry.camera.OrthographicCamera`.
    """

    longitude: float = 0.0
    latitude: float = 0.0
    roll: float = 0.0
    fov: float = 25.0
    pixels: tuple[int, int] = (1024, 1024)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pixels", (int(self.pixels[0]), int(self.pixels[1])))
        _require(self.fov > 0.0, f"fov must be > 0 R_sun, got {self.fov}")
        _require(
            self.pixels[0] > 0 and self.pixels[1] > 0,
            f"pixels must be positive, got {self.pixels}",
        )

    def to_provenance(self) -> dict[str, object]:
        return {
            "longitude": self.longitude,
            "latitude": self.latitude,
            "roll": self.roll,
            "fov": self.fov,
            "pixels": list(self.pixels),
        }


@dataclass(frozen=True)
class ThomsonConfig:
    """The optional Thomson/pB radiometric weighting, present only when it is wanted.

    A composable factor on an axis orthogonal to the geometric preset: its presence in a
    :class:`WeightingConfig` turns the weighting on. ``mode`` picks total-brightness (``K``) or
    polarized (``pB``) emphasis; ``u`` is the limb darkening and ``crossover`` the closed-form →
    asymptotic coefficient radius (R☉), both exposed as parameters with physical defaults.
    """

    mode: str = "K"
    u: float = 0.6
    crossover: float = 10.0

    def __post_init__(self) -> None:
        _one_of(self.mode, THOMSON_MODES, "thomson mode")
        _require(0.0 <= self.u <= 1.0, f"thomson u must be in [0, 1], got {self.u}")
        _require(self.crossover > 0.0, f"thomson crossover must be > 0 R_sun, got {self.crossover}")

    def to_provenance(self) -> dict[str, object]:
        return {"mode": self.mode, "u": self.u, "crossover": self.crossover}


@dataclass(frozen=True)
class WeightingConfig:
    """The LOS depth weighting: a geometric preset, with an optional Thomson/pB weighting.

    ``thomson`` is ``None`` by default (the geometric preset alone). A :class:`ThomsonConfig` turns
    on the radiometric weighting as a composable factor multiplied into the render's weighted
    average: an independent axis that leaves the geometric depth colour and coverage untouched.
    """

    preset: str = "large-fov"
    thomson: ThomsonConfig | None = None

    def __post_init__(self) -> None:
        _one_of(self.preset, WEIGHTING_PRESETS, "preset")

    def to_provenance(self) -> dict[str, object]:
        return {
            "preset": self.preset,
            "thomson": self.thomson.to_provenance() if self.thomson is not None else None,
        }


@dataclass(frozen=True)
class RenderConfig:
    """The LOS render knobs: display reconstruction, occultation, clamp, and sampling.

    Defaults mirror :func:`qorona.render.los.render` exactly. ``workers`` maps to that function's
    numba thread count (``None`` = all cores). ``device`` is accepted for a uniform surface; the
    render runs on the CPU regardless (it has no GPU backend).
    """

    display: str = "balanced"
    occult: str = "eclipse"
    r_occult: float = 1.0
    occult_softness: float = 0.03
    clamp: tuple[float, float] = (_LOG_FLOOR, 7.0)
    floor: bool = True
    step: float = 0.02
    percentiles: tuple[float, float] = (1.0, 99.5)
    polarity_mode: str = "none"
    workers: int | None = None
    device: str = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(self, "clamp", (float(self.clamp[0]), float(self.clamp[1])))
        object.__setattr__(
            self, "percentiles", (float(self.percentiles[0]), float(self.percentiles[1]))
        )
        _one_of(self.display, DISPLAY_MODES, "display")
        _one_of(self.occult, OCCULT_MODES, "occult")
        _one_of(self.polarity_mode, POLARITY_MODES, "polarity_mode")
        _require(self.r_occult > 0.0, f"r_occult must be > 0 R_sun, got {self.r_occult}")
        _require(
            self.occult_softness >= 0.0, f"occult_softness must be >= 0, got {self.occult_softness}"
        )
        _require(
            self.clamp[0] < self.clamp[1],
            f"clamp must be (low, high) with low < high, got {self.clamp}",
        )
        _require(self.step > 0.0, f"step must be > 0 R_sun, got {self.step}")
        low, high = self.percentiles
        _require(
            0.0 <= low < high <= 100.0,
            f"percentiles must satisfy 0 <= low < high <= 100, got {self.percentiles}",
        )
        _require(
            self.workers is None or self.workers >= 1,
            f"workers must be None or >= 1, got {self.workers}",
        )
        _one_of(self.device, DEVICE_MODES, "device")

    def to_provenance(self) -> dict[str, object]:
        return {
            "display": self.display,
            "occult": self.occult,
            "r_occult": self.r_occult,
            "occult_softness": self.occult_softness,
            "clamp": list(self.clamp),
            "floor": self.floor,
            "step": self.step,
            "percentiles": list(self.percentiles),
            "polarity_mode": self.polarity_mode,
            "workers": self.workers,
            "device": self.device,
        }


@dataclass(frozen=True)
class QMapConfig:
    """The Q-map product: a signed-log-Q⊥ shell sliced from the cached Q⊥ volume.

    ``radius`` (R☉) is the shell radius; ``n_theta``/``n_phi`` the longitude/latitude sample grid;
    ``slog_max`` the colour ceiling (``None`` → :data:`~qorona.render.shell.DEFAULT_SLOG_MAX`);
    ``export_npz`` also writes the raw shell arrays.
    """

    radius: float = 3.0
    n_theta: int = 720
    n_phi: int = 1440
    slog_max: float | None = None
    export_npz: bool = False

    def __post_init__(self) -> None:
        _require(self.radius > 0.0, f"radius must be positive, got {self.radius}")
        _require(
            self.n_theta > 0 and self.n_phi > 0,
            f"resolution must be positive, got {self.n_theta}x{self.n_phi}",
        )
        _require(
            self.slog_max is None or self.slog_max > 0.0,
            f"slog_max must be positive, got {self.slog_max}",
        )

    def to_provenance(self) -> dict[str, object]:
        return {
            "radius": self.radius,
            "resolution": f"{self.n_theta}x{self.n_phi}",
            "slog_max": self.slog_max,
            "export_npz": self.export_npz,
        }


@dataclass(frozen=True)
class FieldLinesConfig:
    """The field-line view: how lines are seeded, drawn, and coloured, plus the tracer knobs.

    The viewpoint-independent inputs to the field-line render (the camera is a separate config, as
    for the Q⊥ render). Defaults give the eclipse-photograph look: ``limb`` seeding (a limb ring for
    the open fan plus short front-side loops), ``polarity`` colouring (open lines by inner-foot
    ``B·r̂`` sign, closed loops neutral grey), and a ``B_r`` magnetogram disk. The three tracer knobs
    (``rtol`` / ``cfl`` / ``max_steps``) mirror
    :class:`VolumeConfig`, since both drive the same DOPRI5 integrator.
    """

    seeding: str = "limb"
    n_seeds: int = 1500
    limb_seeds: int = 375
    front_loop_length: float = 1.2
    colour: str = "polarity"
    magnetogram: bool = True
    show: str = "all"
    line_width: float = 1.5
    depth_fade: float = 0.4
    rtol: float = 1e-4
    cfl: float = 0.5
    max_steps: int = 10_000
    max_turn_angle: float = 45.0
    turn_guard_radius: float = 2.0
    turn_guard_weak_fraction: float = 1.0e-5
    min_turns: int = 3
    workers: int | None = None

    def __post_init__(self) -> None:
        _one_of(self.seeding, FIELDLINE_SEEDING, "seeding")
        _one_of(self.colour, FIELDLINE_COLOUR, "colour")
        _one_of(self.show, FIELDLINE_SHOW, "show")
        _require(self.n_seeds >= 1, f"n_seeds must be >= 1, got {self.n_seeds}")
        _require(self.limb_seeds >= 0, f"limb_seeds must be >= 0, got {self.limb_seeds}")
        _require(
            self.front_loop_length > 0.0,
            f"front_loop_length must be > 0, got {self.front_loop_length}",
        )
        _require(self.line_width > 0.0, f"line_width must be > 0, got {self.line_width}")
        _require(
            0.0 <= self.depth_fade <= 1.0, f"depth_fade must be in [0, 1], got {self.depth_fade}"
        )
        _require(self.rtol > 0.0, f"rtol must be > 0, got {self.rtol}")
        _require(0.0 < self.cfl < 1.0, f"cfl must satisfy 0 < cfl < 1, got {self.cfl}")
        _require(self.max_steps > 0, f"max_steps must be > 0, got {self.max_steps}")
        _validate_turn_guard(
            self.max_turn_angle,
            self.turn_guard_radius,
            self.turn_guard_weak_fraction,
            self.min_turns,
        )
        _require(
            self.workers is None or self.workers >= 1,
            f"workers must be None or >= 1, got {self.workers}",
        )

    def to_provenance(self) -> dict[str, object]:
        return {
            "seeding": self.seeding,
            "n_seeds": self.n_seeds,
            "limb_seeds": self.limb_seeds,
            "front_loop_length": self.front_loop_length,
            "colour": self.colour,
            "magnetogram": self.magnetogram,
            "show": self.show,
            "line_width": self.line_width,
            "depth_fade": self.depth_fade,
            "rtol": self.rtol,
            "cfl": self.cfl,
            "max_steps": self.max_steps,
            "max_turn_angle": self.max_turn_angle,
            "turn_guard_radius": self.turn_guard_radius,
            "turn_guard_weak_fraction": self.turn_guard_weak_fraction,
            "min_turns": self.min_turns,
            "workers": self.workers,
        }


@dataclass(frozen=True)
class ExportConfig:
    """The field-line export: the seed grid and the tracer knobs.

    Drives ``export-lines``, which traces a bundle of field lines and writes the polylines to a
    file. Seeds sit on a uniform longitude/latitude grid
    (:func:`~qorona.trace.lonlat_seeds`) on the sphere of ``seed_radius`` (``None`` means the
    field's inner boundary). The tracer knobs are the same as :class:`FieldLinesConfig`.
    """

    n_theta: int = 100
    n_phi: int = 100
    seed_radius: float | None = None
    rtol: float = 1e-4
    cfl: float = 0.5
    max_steps: int = 10_000
    max_turn_angle: float = 45.0
    turn_guard_radius: float = 2.0
    turn_guard_weak_fraction: float = 1.0e-5
    min_turns: int = 3
    workers: int | None = None

    def __post_init__(self) -> None:
        _require(self.n_theta >= 1, f"n_theta must be >= 1, got {self.n_theta}")
        _require(self.n_phi >= 1, f"n_phi must be >= 1, got {self.n_phi}")
        _require(
            self.seed_radius is None or self.seed_radius > 0.0,
            f"seed_radius must be None or > 0, got {self.seed_radius}",
        )
        _require(self.rtol > 0.0, f"rtol must be > 0, got {self.rtol}")
        _require(0.0 < self.cfl < 1.0, f"cfl must satisfy 0 < cfl < 1, got {self.cfl}")
        _require(self.max_steps > 0, f"max_steps must be > 0, got {self.max_steps}")
        _validate_turn_guard(
            self.max_turn_angle,
            self.turn_guard_radius,
            self.turn_guard_weak_fraction,
            self.min_turns,
        )
        _require(
            self.workers is None or self.workers >= 1,
            f"workers must be None or >= 1, got {self.workers}",
        )

    def to_provenance(self) -> dict[str, object]:
        return {
            "n_theta": self.n_theta,
            "n_phi": self.n_phi,
            "seed_radius": self.seed_radius,
            "rtol": self.rtol,
            "cfl": self.cfl,
            "max_steps": self.max_steps,
            "max_turn_angle": self.max_turn_angle,
            "turn_guard_radius": self.turn_guard_radius,
            "turn_guard_weak_fraction": self.turn_guard_weak_fraction,
            "min_turns": self.min_turns,
            "workers": self.workers,
        }


@dataclass(frozen=True)
class BrightnessConfig:
    """The white-light / polarized-brightness (pB) render: frame, display treatment, and knobs.

    The viewpoint-independent inputs to the standalone brightness product (the camera is a separate
    config, as for the Q⊥ render). ``frame`` selects the polarized brightness ``pB`` (the default,
    the reference target) or the total white-light brightness; ``treatment`` finishes the selected
    frame raw, with the ``radial`` power-law filter (brightness times ``rho**radial_power``, lifting
    the outer corona while staying bright near the limb), with the Newkirk radial vignette, or with
    multi-scale Gaussian-normalization enhancement. MGN is calibrated for the steep-gradient pB
    frame; on the total frame it still runs but is less physically meaningful. ``u`` and
    ``crossover`` are the Thomson coefficient knobs, ``scaling`` / ``percentiles`` the display
    stretch; the line-of-sight ``step`` and the occultation triple mirror :class:`RenderConfig`.
    ``scaling`` defaults to ``None`` and resolves to ``"linear"`` for the already-normalised MGN
    treatment, else ``"log"``.
    """

    frame: str = "polarized"
    treatment: str = "raw"
    #: Exponent of the ``radial`` power-law filter; mirrors ``display.RADIAL_FILTER_POWER``.
    radial_power: float = 3.0
    u: float = 0.6
    crossover: float = 10.0
    step: float = 0.02
    occult: str = "eclipse"
    r_occult: float = 1.0
    occult_softness: float = 0.03
    scaling: str | None = None
    percentiles: tuple[float, float] = (1.0, 99.5)
    workers: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "percentiles", (float(self.percentiles[0]), float(self.percentiles[1]))
        )
        _one_of(self.frame, BRIGHTNESS_FRAMES, "frame")
        _one_of(self.treatment, BRIGHTNESS_TREATMENTS, "treatment")
        _one_of(self.occult, OCCULT_MODES, "occult")
        scaling = self.scaling
        if scaling is None:
            scaling = "linear" if self.treatment == "mgn" else "log"
            object.__setattr__(self, "scaling", scaling)
        _one_of(scaling, BRIGHTNESS_SCALINGS, "scaling")
        _require(0.0 <= self.u <= 1.0, f"limb darkening u must be in [0, 1], got {self.u}")
        _require(self.crossover > 0.0, f"crossover must be > 0 R_sun, got {self.crossover}")
        _require(self.radial_power > 0.0, f"radial_power must be > 0, got {self.radial_power}")
        _require(self.step > 0.0, f"step must be > 0 R_sun, got {self.step}")
        _require(self.r_occult > 0.0, f"r_occult must be > 0 R_sun, got {self.r_occult}")
        _require(
            self.occult_softness >= 0.0, f"occult_softness must be >= 0, got {self.occult_softness}"
        )
        low, high = self.percentiles
        _require(
            0.0 <= low < high <= 100.0,
            f"percentiles must satisfy 0 <= low < high <= 100, got {self.percentiles}",
        )
        _require(
            self.workers is None or self.workers >= 1,
            f"workers must be None or >= 1, got {self.workers}",
        )

    def to_provenance(self) -> dict[str, object]:
        return {
            "frame": self.frame,
            "treatment": self.treatment,
            "radial_power": self.radial_power,
            "u": self.u,
            "crossover": self.crossover,
            "step": self.step,
            "occult": self.occult,
            "r_occult": self.r_occult,
            "occult_softness": self.occult_softness,
            "scaling": self.scaling,
            "percentiles": list(self.percentiles),
            "workers": self.workers,
        }


@dataclass(frozen=True)
class OutputConfig:
    """Where the image is written, how it is annotated, and which data sidecars to export.

    The colour PNG is the headline product; the grayscale measurement image is opt-in. The on-image
    provenance stamp is on by default and bypassed with ``annotate=False``. ``export_formats`` lists
    the dependency-free data sidecars to write beside the image (currently ``"npz"``; the brightness
    products write the raw frames + plane-of-sky coordinates there).
    """

    path: Path
    save_grayscale: bool = False
    annotate: bool = True
    annotate_position: str = "bottom-left"
    export_formats: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "export_formats", tuple(self.export_formats))
        _one_of(self.annotate_position, ANNOTATE_POSITIONS, "annotate_position")
        for fmt in self.export_formats:
            _one_of(fmt, EXPORT_FORMATS, "export format")

    def grayscale_path(self) -> Path:
        """Return the companion grayscale PNG path (``<stem>_grayscale<suffix>``)."""
        return self.path.with_name(f"{self.path.stem}_grayscale{self.path.suffix}")

    def export_path(self, fmt: str) -> Path:
        """Return the data-sidecar path for ``fmt`` (``<stem>.<fmt>`` beside the image)."""
        return self.path.with_suffix(f".{fmt}")

    def to_provenance(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "save_grayscale": self.save_grayscale,
            "annotate": self.annotate,
            "annotate_position": self.annotate_position,
            "export_formats": list(self.export_formats),
        }


@dataclass(frozen=True)
class RunConfig:
    """The whole-pipeline configuration: the one-shot ``run`` path composes every sub-config.

    ``workers`` is the single top-level thread-count knob :func:`qorona.pipeline.run` fans out to
    both the volume build and the render; ``device`` fans out the same way (the render accepts it
    for a uniform surface and runs on the CPU regardless); ``version`` records the Qorona version
    that produced the run.
    """

    input: InputConfig
    grid: GridConfig
    volume: VolumeConfig
    camera: CameraConfig
    weighting: WeightingConfig
    render: RenderConfig
    output: OutputConfig
    version: str = __version__
    workers: int | None = None
    device: str = "auto"

    def to_provenance(self) -> dict[str, object]:
        """Return the nested, JSON-safe provenance mapping (the pipeline augments it with the
        derived CR/JD, the input content hash, the field normalization, and the run metrics)."""
        return {
            "input": self.input.to_provenance(),
            "grid": self.grid.to_provenance(),
            "volume": self.volume.to_provenance(),
            "camera": self.camera.to_provenance(),
            "weighting": self.weighting.to_provenance(),
            "render": self.render.to_provenance(),
            "output": self.output.to_provenance(),
            "version": self.version,
            "workers": self.workers,
            "device": self.device,
        }
