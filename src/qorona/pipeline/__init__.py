"""End-to-end pipeline orchestration for a single snapshot.

Wires read → resample → Q⊥ volume → render behind config-driven stage functions
(:func:`build_field`, :func:`build_volume`, :func:`render_volume`, :func:`run`), with the
viewpoint-independent volume cached to a dependency-free ``.npz`` (:func:`save_volume` /
:func:`load_volume`) so the expensive bake and the cheap render are separable. The CR/JD derivation
and provenance assembly that feed the on-image stamp and the end-of-run summary live here too. No
CLI or presentation logic; this layer is callable as a library.
"""

from __future__ import annotations

from qorona.pipeline.run import (
    brightness_provenance,
    build_field,
    build_provenance,
    build_volume,
    content_hash,
    covered_fraction,
    derive_cr,
    derive_jd,
    fieldlines_provenance,
    load_volume,
    render_brightness,
    render_fieldlines,
    render_provenance,
    render_volume,
    run,
    save_volume,
    sub_floor_voxels,
)

__all__ = [
    "brightness_provenance",
    "build_field",
    "build_provenance",
    "build_volume",
    "content_hash",
    "covered_fraction",
    "derive_cr",
    "derive_jd",
    "fieldlines_provenance",
    "load_volume",
    "render_brightness",
    "render_fieldlines",
    "render_provenance",
    "render_volume",
    "run",
    "save_volume",
    "sub_floor_voxels",
]
