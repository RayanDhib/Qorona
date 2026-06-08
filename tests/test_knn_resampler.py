"""Resampler check: the k-NN MLS resampler reproduces a linear field exactly.

The defining property of the degree-1 moving-least-squares reconstruction (the default
resampler): a field that is exactly linear in space is recovered at every grid node to numerical
precision, regardless of the scattered cell-centre layout. This is what makes it first-order and
divergence-faithful (it does not manufacture ∇·B), and it catches centring/weighting bugs that
symmetric layouts would hide.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy import units as u

from qorona.io.native import NativeSolution, SolutionMetadata
from qorona.resample import KnnMlsResampler, LogarithmicSpacing, SphericalGrid

#: An exactly-linear vector field B_i = OFFSET_i + (GRADIENT · x)_i, shared by the fixture and the
#: expectation so the two cannot drift.
OFFSET = np.array([0.3, -0.7, 1.1])
GRADIENT = np.array([[0.5, -0.2, 0.1], [0.0, 0.4, -0.3], [0.2, 0.1, -0.6]])
COMPONENTS = ("Bx", "By", "Bz")


def _linear_field(points: np.ndarray) -> np.ndarray:
    """Evaluate the linear field at ``(..., 3)`` points, returning ``(..., 3)``."""
    return OFFSET + points @ GRADIENT.T


def _linear_solution() -> NativeSolution:
    """A scattered cell cloud filling the grid's shell (with margin), carrying the linear field."""
    rng = np.random.default_rng(0)
    radius = rng.uniform(0.9, 2.6, 4000)
    direction = rng.normal(size=(4000, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    centers = radius[:, None] * direction
    field = _linear_field(centers)
    metadata = SolutionMetadata(
        model="test",
        file_format="test",
        source_path=Path("test"),
        normalization="test",
        dimension=3,
        n_equations=3,
        element_type="none",
    )
    return NativeSolution(
        nodes=centers * u.R_sun,
        connectivity=np.zeros((0, 6), dtype=np.int64),
        cell_centers=centers * u.R_sun,
        variables={name: field[:, i] for i, name in enumerate(COMPONENTS)},
        boundaries={},
        metadata=metadata,
    )


def test_knn_mls_reproduces_linear_field() -> None:
    grid = SphericalGrid(LogarithmicSpacing(1.0, 2.5), n_r=16, n_theta=24, n_phi=48)
    resampler = KnnMlsResampler()
    resampled = resampler.resample(_linear_solution(), grid, COMPONENTS, show_progress=False)

    expected = _linear_field(grid.node_points())
    for i, name in enumerate(COMPONENTS):
        assert np.allclose(resampled[name], expected[..., i], rtol=0.0, atol=1e-6)
