"""Write traced field lines in JHelioviewer's compact SunJSON format.

Qorona's tracer already returns one Cartesian ``(N, 3)`` polyline per seed.  The export only
converts those points to the spherical coordinates expected by JHelioviewer::

    {
      "type": "SunJSON",
      "time": "2024-01-01T00:00:00",
      "geometry": [
        {
          "type": "line",
          "coordinates": [[radius_Rsun, carrington_lon_deg, lat_deg], ...],
          "colors": [[red, green, blue, 255]],
          "thickness": 0.004,
          "topology": "open" | "closed"
        },
        ...
      ]
    }

Incomplete lines (an end stopped at a null or a guard) are omitted.  Cartesian coordinates use the
ingest convention: x-y is the solar equatorial plane, z is solar north, and ``atan2(y, x)`` is the
solution frame's longitude, taken to Carrington (what JHelioviewer expects) by the declared frame's
offset, read from the same metadata as the time.  Colours match the PNG renderer's polarity
palette: outward/inward open lines are warm/cool and closed lines are neutral grey.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from qorona.field.base import Field
from qorona.render.fieldlines import polarity_colours
from qorona.trace import FieldLines

# Keep files manageable for the default 100 x 100 seed grid while retaining substantially more
# precision than the tracer tolerance.
_COORDINATE_DECIMALS = 6
_DEFAULT_COLOR = [255, 255, 255, 255]
_DEFAULT_THICKNESS = 0.004


def _xyz_to_sunjson_coordinates(xyz: np.ndarray, longitude_offset: float = 0.0) -> np.ndarray:
    """Convert one Cartesian ``(N, 3)`` R_sun polyline to ``[r, lon, lat]`` SunJSON points, with
    ``longitude_offset`` (deg) taking the solution frame's longitude to Carrington."""
    points = np.asarray(xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"field-line path must have shape (N, 3), got {points.shape}")

    x, y, z = points.T
    radius = np.linalg.norm(points, axis=1)
    longitude = (np.degrees(np.arctan2(y, x)) + longitude_offset) % 360.0
    with np.errstate(divide="ignore", invalid="ignore"):
        latitude = np.degrees(np.arcsin(np.clip(z / radius, -1.0, 1.0)))
    latitude = np.nan_to_num(latitude)
    return np.column_stack((radius, longitude, latitude))


def _longitude_offset(metadata: dict[str, Any]) -> float:
    """Return the offset taking the declared frame's longitudes to Carrington, zero for a
    Carrington-aligned (or undeclared) frame; a Stonyhurst frame needs the timestamp."""
    from qorona.config.schema import DEFAULT_FRAME
    from qorona.ephemeris import carrington_offset

    input_metadata = metadata.get("input")
    if not isinstance(input_metadata, dict):
        return 0.0
    frame = input_metadata.get("frame")
    if frame in (None, DEFAULT_FRAME):
        return 0.0
    timestamp = input_metadata.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        raise ValueError(
            f"a {frame} solution needs a timestamp to place its longitudes in Carrington; "
            "pass --timestamp"
        )
    return carrington_offset(frame, timestamp)


def _sunjson_time(metadata: dict[str, Any]) -> str:
    """Return the provenance timestamp in SunJSON form, or the current UTC time if absent."""
    input_metadata = metadata.get("input")
    timestamp = input_metadata.get("timestamp") if isinstance(input_metadata, dict) else None
    if not isinstance(timestamp, str) or not timestamp:
        return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")

    # InputConfig has already validated this timestamp.  Normalising here removes a timezone
    # suffix because that is the representation used by the reference JHelioviewer exporter.
    moment = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if moment.tzinfo is not None:
        moment = moment.astimezone(UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%S")


def _sunjson_colours(lines: FieldLines, field: Field | None) -> np.ndarray:
    """Return one RGBA SunJSON colour per line, matching the PNG polarity palette."""
    if field is None:
        return np.tile(np.array(_DEFAULT_COLOR, dtype=np.uint8), (lines.seeds.shape[0], 1))

    rgb = polarity_colours(
        field,
        lines,
        field.domain.inner_radius,
        field.domain.outer_radius,
    )
    rgb_8bit = np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    alpha = np.full((rgb_8bit.shape[0], 1), 255, dtype=np.uint8)
    return np.column_stack((rgb_8bit, alpha))


def write_fieldlines_json(
    lines: FieldLines,
    path: str | Path,
    metadata: dict[str, Any],
    *,
    field: Field | None = None,
    keep: np.ndarray | None = None,
    colours: np.ndarray | None = None,
) -> Path:
    """Write complete traced lines as JHelioviewer SunJSON and return the destination path.

    ``metadata["input"]`` supplies the observation timestamp and the solution frame declaration
    already recorded by Qorona's pipeline; a Stonyhurst frame without a timestamp raises
    :class:`ValueError`, since its longitudes cannot be placed in Carrington.  When
    ``field`` is supplied, open lines use the same inner-foot polarity colours as the PNG
    field-line render and closed lines use its neutral grey; the library-level fallback is white.
    ``keep`` restricts the export to a sub-bundle (a boolean mask over the traced lines, complete
    lines only; default all complete lines) and ``colours`` overrides the palette with explicit
    per-kept-line linear RGB in [0, 1]; together they let the field-line view export exactly its
    drawn bundle in its drawn colours.
    """
    if lines.paths is None:
        raise ValueError("lines carry no polylines; trace with store_path=True")
    keep = lines.is_complete if keep is None else np.asarray(keep, dtype=bool)
    if bool((keep & lines.is_incomplete).any()):
        raise ValueError("keep selects incomplete lines; export complete lines only")
    selected = np.nonzero(keep)[0]
    if colours is None:
        rgba = _sunjson_colours(lines, field)[selected]
    else:
        if colours.shape[0] != selected.size:
            raise ValueError(
                f"colours must have one row per kept line, got {colours.shape[0]} "
                f"for {selected.size} kept"
            )
        rgb_8bit = np.round(np.clip(colours, 0.0, 1.0) * 255.0).astype(np.uint8)
        rgba = np.column_stack((rgb_8bit, np.full((rgb_8bit.shape[0], 1), 255, dtype=np.uint8)))

    closed = lines.is_closed
    longitude_offset = _longitude_offset(metadata)
    geometry = []
    for position, index in enumerate(selected):
        coordinates = _xyz_to_sunjson_coordinates(lines.paths[index], longitude_offset)
        geometry.append(
            {
                "type": "line",
                "coordinates": np.round(coordinates, _COORDINATE_DECIMALS).tolist(),
                "colors": [[int(channel) for channel in rgba[position]]],
                "thickness": _DEFAULT_THICKNESS,
                "topology": "closed" if closed[index] else "open",
            }
        )

    payload = {
        "type": "SunJSON",
        "time": _sunjson_time(metadata),
        "geometry": geometry,
    }
    destination = Path(path)
    destination.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return destination
