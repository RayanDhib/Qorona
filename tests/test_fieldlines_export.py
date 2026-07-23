"""SunJSON field-line export tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from qorona.io.fieldlines_export import write_fieldlines_json
from qorona.trace import Endpoint, FieldLines


class _OutwardField:
    """Minimal sampled field with positive radial polarity for exporter testing."""

    domain = SimpleNamespace(inner_radius=1.0, outer_radius=3.0)

    def sample(self, points: np.ndarray, *, gradient: bool = False) -> Any:
        del gradient
        return SimpleNamespace(b=np.asarray(points))


def test_write_fieldlines_sunjson(tmp_path: Path) -> None:
    paths = [
        np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
        np.array([[0.0, -1.0, 0.0], [1.0, 1.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0], [1.1, 0.0, 0.0]]),
    ]
    lines = FieldLines(
        seeds=np.zeros((3, 3)),
        feet=np.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [np.nan, np.nan, np.nan]],
            ]
        ),
        ends=np.array(
            [
                [Endpoint.INNER, Endpoint.INNER],
                [Endpoint.INNER, Endpoint.OUTER],
                [Endpoint.INNER, Endpoint.MAX_STEPS],
            ],
            dtype=np.int8,
        ),
        lengths=np.zeros((3, 2)),
        paths=paths,
    )
    output = tmp_path / "lines.json"

    returned = write_fieldlines_json(
        lines,
        output,
        {"input": {"timestamp": "2024-01-02T03:04:05+00:00"}},
        field=_OutwardField(),  # type: ignore[arg-type]
    )

    assert returned == output
    payload = json.loads(output.read_text())
    assert payload["type"] == "SunJSON"
    assert payload["time"] == "2024-01-02T03:04:05"
    assert len(payload["geometry"]) == 2  # Incomplete paths are deliberately omitted.

    closed, open_ = payload["geometry"]
    assert closed == {
        "type": "line",
        "coordinates": [[1.0, 0.0, 0.0], [2.0, 90.0, 0.0], [3.0, 0.0, 90.0]],
        "colors": [[128, 128, 128, 255]],
        "thickness": 0.004,
        "topology": "closed",
    }
    assert open_["coordinates"] == [[1.0, 270.0, 0.0], [1.414214, 45.0, 0.0]]
    assert open_["colors"] == [[230, 80, 60, 255]]
    assert open_["topology"] == "open"


def test_write_fieldlines_requires_stored_paths(tmp_path: Path) -> None:
    lines = FieldLines(
        seeds=np.zeros((1, 3)),
        feet=np.zeros((1, 2, 3)),
        ends=np.array([[Endpoint.INNER, Endpoint.INNER]], dtype=np.int8),
        lengths=np.zeros((1, 2)),
        paths=None,
    )

    try:
        write_fieldlines_json(lines, tmp_path / "lines.json", {})
    except ValueError as error:
        assert "store_path=True" in str(error)
    else:
        raise AssertionError("expected missing paths to be rejected")
