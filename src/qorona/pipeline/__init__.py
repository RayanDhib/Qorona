"""End-to-end pipeline orchestration for a single snapshot.

Wires read → resample → Q⊥ volume → render behind config-driven stage functions
(:func:`build_field`, :func:`build_volume`, :func:`render_volume`, :func:`run`), with the
viewpoint-independent volume cached to a dependency-free ``.npz`` (:func:`save_volume` /
:func:`load_volume`) so the expensive bake and the cheap render are separable. The provenance
assembly that feeds the on-image stamp and the end-of-run summary lives here too. No CLI or
presentation logic; this layer is callable as a library.

The category-keyed default resolvers (:func:`resolve_projection`, :func:`resolve_camera`,
:func:`resolve_render`, :func:`resolve_brightness`, :func:`resolve_qmap`) are re-exported so the CLI
can pre-resolve the shell-dependent geometry sentinels at its edge (feeding the *resolved* configs
to both the pipeline call and the provenance / stamp assembly). They are idempotent, so the stage
functions' own internal resolution then stands as a safety net; the resolution laws live once, here.
The CLI also calls the resolvers to check the camera flags before any expensive stage.
"""

from __future__ import annotations

from qorona.pipeline.run import (
    _resolve_brightness as resolve_brightness,
)
from qorona.pipeline.run import (
    _resolve_camera as resolve_camera,
)
from qorona.pipeline.run import (
    _resolve_projection as resolve_projection,
)
from qorona.pipeline.run import (
    _resolve_qmap as resolve_qmap,
)
from qorona.pipeline.run import (
    _resolve_render as resolve_render,
)
from qorona.pipeline.run import (
    brightness_provenance,
    build_field,
    build_plasma,
    build_provenance,
    build_volume,
    content_hash,
    covered_fraction,
    euv_field_radius,
    euv_provenance,
    export_lines,
    export_provenance,
    fieldlines_provenance,
    load_volume,
    qmap_from_volume,
    qmap_provenance,
    render_brightness,
    render_euv,
    render_fieldlines,
    render_provenance,
    render_punch_l3,
    render_volume,
    resolve_euv_camera,
    resolve_tr_temperature,
    run,
    save_volume,
    sub_floor_voxels,
    volume_brightness_provenance,
)

__all__ = [
    "brightness_provenance",
    "build_field",
    "build_plasma",
    "build_provenance",
    "build_volume",
    "content_hash",
    "covered_fraction",
    "euv_field_radius",
    "euv_provenance",
    "export_lines",
    "export_provenance",
    "fieldlines_provenance",
    "load_volume",
    "qmap_from_volume",
    "qmap_provenance",
    "render_brightness",
    "render_euv",
    "render_fieldlines",
    "render_provenance",
    "render_punch_l3",
    "render_volume",
    "resolve_brightness",
    "resolve_camera",
    "resolve_euv_camera",
    "resolve_projection",
    "resolve_qmap",
    "resolve_render",
    "resolve_tr_temperature",
    "run",
    "save_volume",
    "sub_floor_voxels",
    "volume_brightness_provenance",
]
