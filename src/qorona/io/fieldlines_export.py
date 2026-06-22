"""Write traced field lines to a JSON file.

:func:`write_fieldlines_json` is the single place the output format is built. A viewer-specific
format (e.g. SunJSON for JHelioviewer) is added by changing the payload here; the seeding,
tracing, and CLI around it stay as they are. Current schema::

    {
      "format": "qorona-fieldlines-json-v1",
      "units": "R_sun",
      "frame": "solution Cartesian",
      "metadata": { ... },                  # the run's settings (input, grid, export)
      "lines": [
        {"topology": "open" | "closed",
         "points": [[x, y, z], ...]},       # ordered backward-foot → seed → forward-foot
        ...
      ]
    }

Coordinates are Cartesian in the solution's own frame, in R☉. Incomplete lines (an end stopped
at a null or a guard) are dropped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from qorona.trace import FieldLines

#: Decimals kept on exported coordinates (~10⁻⁶ R☉, well below the tracer tolerance).
_COORDINATE_DECIMALS = 6


def write_fieldlines_json(lines: FieldLines, path: str | Path, metadata: dict[str, Any]) -> Path:
    """Write the complete lines of ``lines`` to ``path`` as JSON and return the path.

    Parameters
    ----------
    lines
        The traced bundle; must carry its polylines (traced with ``store_path=True``).
    path
        Destination file.
    metadata
        The run's settings, stored as the ``metadata`` block.
    """
    if lines.paths is None:
        raise ValueError("lines carry no polylines; trace with store_path=True")
    keep = np.nonzero(lines.is_complete)[0]
    closed = lines.is_closed
    payload = {
        "format": "qorona-fieldlines-json-v1",
        "units": "R_sun",
        "frame": "solution Cartesian",
        "metadata": metadata,
        "lines": [
            {
                "topology": "closed" if closed[index] else "open",
                "points": np.round(lines.paths[index], _COORDINATE_DECIMALS).tolist(),
            }
            for index in keep
        ],
    }
    path = Path(path)
    path.write_text(json.dumps(payload, separators=(",", ":")))
    return path
