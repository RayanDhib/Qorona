"""The typed run-configuration schema.

Frozen dataclasses that name every run parameter once (its default and its validation) and feed
the provenance that the on-image stamp, the volume artifact, and the end-of-run summary all read.
The CLI populates them from ``--flags``; a YAML or GUI loader would be just another front-end
constructing the same dataclasses.
"""

from __future__ import annotations

from qorona.config.schema import (
    ANNOTATE_POSITIONS,
    BRIGHTNESS_FRAMES,
    BRIGHTNESS_SCALINGS,
    BRIGHTNESS_TREATMENTS,
    CLOSED_TREATMENTS,
    DEVICE_MODES,
    DISPLAY_MODES,
    FIELDLINE_COLOUR,
    FIELDLINE_SEEDING,
    FIELDLINE_SHOW,
    OCCULT_MODES,
    POLARITY_MODES,
    PRECISION_MODES,
    RESAMPLERS,
    SPACING_LAWS,
    THOMSON_MODES,
    VOLUME_BUILDERS,
    WEIGHTING_PRESETS,
    BrightnessConfig,
    CameraConfig,
    FieldLinesConfig,
    GridConfig,
    InputConfig,
    OutputConfig,
    RenderConfig,
    RunConfig,
    ThomsonConfig,
    VolumeConfig,
    WeightingConfig,
)

__all__ = [
    "ANNOTATE_POSITIONS",
    "BRIGHTNESS_FRAMES",
    "BRIGHTNESS_SCALINGS",
    "BRIGHTNESS_TREATMENTS",
    "CLOSED_TREATMENTS",
    "DEVICE_MODES",
    "DISPLAY_MODES",
    "FIELDLINE_COLOUR",
    "FIELDLINE_SEEDING",
    "FIELDLINE_SHOW",
    "OCCULT_MODES",
    "POLARITY_MODES",
    "PRECISION_MODES",
    "RESAMPLERS",
    "SPACING_LAWS",
    "THOMSON_MODES",
    "VOLUME_BUILDERS",
    "WEIGHTING_PRESETS",
    "BrightnessConfig",
    "CameraConfig",
    "FieldLinesConfig",
    "GridConfig",
    "InputConfig",
    "OutputConfig",
    "RenderConfig",
    "RunConfig",
    "ThomsonConfig",
    "VolumeConfig",
    "WeightingConfig",
]
