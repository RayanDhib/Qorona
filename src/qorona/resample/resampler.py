"""Resample a native cell-centred solution onto the internal spherical grid.

A :class:`Resampler` maps the model-agnostic :class:`~qorona.io.native.NativeSolution`
(cell-centre coordinates and cell-centred field values, plus connectivity and boundaries)
onto the regular spherical grid that the field/tracer/render stages consume. It uses only the
generic native interface (no assumption about the native discretisation), so any model that
produces a ``NativeSolution`` resamples through the same code.

The default :class:`KnnMlsResampler` fits a local moving-least-squares linear model at each grid
node: first-order, smooth, and, unlike a piecewise-constant copy, it does not *manufacture*
spurious ``∇·B``. That is load-bearing for the squashing factor, whose prefactor rests on flux
conservation: a nearest-cell staircase injects ``|∇·B|/|B| ~ O(1)`` per R_sun that drives Q⊥ below
its theoretical floor of 2 on real data, while the source solutions are themselves well-cleaned, so
faithfully reconstructing the existing field restores the floor. :class:`NearestCellResampler` is
the fast, finite-volume-faithful baseline. The smooth, differentiable field the tracer needs is
supplied downstream by the tricubic interpolant.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from astropy import units as u
from scipy.spatial import cKDTree

from qorona.accel import HAVE_NUMBA
from qorona.console import progress_bar, status
from qorona.io.native import NativeSolution
from qorona.resample.grid import SphericalGrid


class Resampler(ABC):
    """Maps a :class:`NativeSolution` onto a :class:`SphericalGrid`."""

    @abstractmethod
    def resample(
        self,
        solution: NativeSolution,
        grid: SphericalGrid,
        variables: tuple[str, ...],
        *,
        show_progress: bool = True,
    ) -> dict[str, np.ndarray]:
        """Resample named cell-centred variables onto the grid nodes.

        Parameters
        ----------
        solution
            The native solution to resample from.
        grid
            The target internal spherical grid.
        variables
            Names of the cell-centred variables to resample (e.g. ``("Bx", "By", "Bz")``).
        show_progress
            Whether to display progress for the resampling.

        Returns
        -------
        dict[str, numpy.ndarray]
            Each requested variable as an ``(n_r, n_theta, n_phi)`` array on the grid nodes.
        """


class NearestCellResampler(Resampler):
    """Assign each grid node the value of the nearest native cell centre (k-d tree).

    Model-agnostic and faithful to piecewise-constant finite-volume data; the downstream
    tricubic interpolant provides the smooth field for tracing.
    """

    def resample(
        self,
        solution: NativeSolution,
        grid: SphericalGrid,
        variables: tuple[str, ...],
        *,
        show_progress: bool = True,
    ) -> dict[str, np.ndarray]:
        cell_centers = solution.cell_centers.to_value(u.R_sun)
        node_points = grid.node_points()
        shape = node_points.shape[:3]

        with status("Building cell-centre index", enabled=show_progress):
            tree = cKDTree(cell_centers)
        with status(f"Resampling {len(variables)} variables onto the grid", enabled=show_progress):
            _, nearest = tree.query(node_points.reshape(-1, 3), workers=-1)
            resampled = {
                name: solution.variables[name][nearest].reshape(shape) for name in variables
            }
        return resampled


class KnnMlsResampler(Resampler):
    """Resample by a local moving-least-squares (degree-1) fit at each grid node.

    For each grid node, fit a linear model ``f(y) = c0 + g·(y - x_node)`` to the ``k`` nearest
    native cell centres by Gaussian-distance-weighted least squares; the resampled value is the fit
    evaluated at the node (``c0``). Being first-order, the reconstruction reproduces a linear field
    exactly and stays smooth, so, unlike :class:`NearestCellResampler`'s piecewise-constant copy,
    it does not manufacture spurious ``∇·B`` (see the module header). Fully model-agnostic: it
    uses only cell centres and values.

    The work is batched over grid nodes (one k-d tree query and one batched 4-by-4 solve per chunk),
    so all requested variables, which share the node geometry, hence the same design matrix and
    weights, are fit together. Moving least squares: Lancaster & Salkauskas (1981).

    The stencil adapts to the source mesh. A fixed neighbour count spans a physical region that
    shrinks as the mesh refines, so a finer input is fit too locally and the reconstruction ripples;
    the effective ``k`` therefore scales with the cell count (``n_neighbors`` and
    ``reference_cell_count``) to hold the fit support at a consistent physical size. Coarser meshes
    floor to ``n_neighbors`` and resample unchanged.

    Parameters
    ----------
    n_neighbors
        Minimum stencil size ``k``: nearest cell centres per fit, and the size used at or below
        ``reference_cell_count``. The effective size scales up in proportion to the cell count on
        finer meshes. Must exceed the four linear unknowns; larger smooths more (and conditions
        better on the anisotropic near-r=1 cells), smaller is sharper but noisier.
    reference_cell_count
        Source cell count at which the stencil equals ``n_neighbors``. Calibrated to the standard
        COCONUT corona mesh; a mesh with ``N`` times more cells uses about ``N * n_neighbors``
        neighbours, holding the physical support fixed. Coarser meshes floor to ``n_neighbors``.
    ridge
        Tikhonov regularization added to each node's normal-equations diagonal (relative to its
        largest entry), keeping one-sided boundary and near-degenerate stencils solvable and
        degrading gracefully toward the weighted mean.
    chunk_size
        Grid nodes processed per batch, bounding the peak memory of the ``(chunk, k, ...)`` arrays.
    """

    def __init__(
        self,
        *,
        n_neighbors: int = 30,
        reference_cell_count: int = 2_000_000,
        ridge: float = 1.0e-8,
        chunk_size: int = 500_000,
    ) -> None:
        if n_neighbors <= 4:
            raise ValueError(
                f"n_neighbors must exceed the four linear unknowns (1 + 3 gradient), "
                f"got {n_neighbors}"
            )
        if reference_cell_count <= 0:
            raise ValueError(f"reference_cell_count must be positive, got {reference_cell_count}")
        self.n_neighbors = n_neighbors
        self.reference_cell_count = reference_cell_count
        self.ridge = ridge
        self.chunk_size = chunk_size

    def _effective_neighbors(self, n_cells: int) -> int:
        """Stencil size for a mesh of ``n_cells`` cells: ``n_neighbors`` scaled by the cell count
        past ``reference_cell_count``, floored at ``n_neighbors`` and capped at the cell count, so
        the fit support stays a fixed physical size as the mesh refines."""
        scaled = round(self.n_neighbors * n_cells / self.reference_cell_count)
        return max(self.n_neighbors, min(scaled, n_cells))

    def resample(
        self,
        solution: NativeSolution,
        grid: SphericalGrid,
        variables: tuple[str, ...],
        *,
        show_progress: bool = True,
    ) -> dict[str, np.ndarray]:
        cell_centers = solution.cell_centers.to_value(u.R_sun)
        values = np.stack([solution.variables[name] for name in variables], axis=1)
        node_field = grid.node_points()
        shape = node_field.shape[:3]
        nodes = node_field.reshape(-1, 3)
        n_nodes = nodes.shape[0]
        k = self._effective_neighbors(cell_centers.shape[0])

        with status("Building cell-centre index", enabled=show_progress):
            tree = cKDTree(cell_centers)

        fitted = np.empty((n_nodes, len(variables)))
        label = f"Resampling {len(variables)} variables (k-NN MLS, k={k})"
        with progress_bar(label, n_nodes, enabled=show_progress) as progress:
            for start in range(0, n_nodes, self.chunk_size):
                stop = min(start + self.chunk_size, n_nodes)
                fitted[start:stop] = self._fit_chunk(
                    nodes[start:stop], tree, cell_centers, values, k
                )
                progress(stop)
        return {name: fitted[:, v].reshape(shape) for v, name in enumerate(variables)}

    def _fit_chunk(
        self,
        nodes: np.ndarray,
        tree: cKDTree,
        cell_centers: np.ndarray,
        values: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """Fit the degree-1 MLS model at a chunk of nodes; return node values ``(m, n_vars)``.

        The exact-k neighbours come from the CPU ``cKDTree`` (a small fraction of the cost); the
        dominant per-node assemble + 4x4 solve runs in a numba ``prange`` kernel when numba is
        present,
        reproducing the NumPy reference below to float64 round-off. Without numba the NumPy
        einsum path is used directly.
        """
        distance, neighbor = tree.query(nodes, k=k, workers=-1)
        if HAVE_NUMBA:
            from qorona.resample._mls_jit import fit_mls_chunk

            return fit_mls_chunk(
                np.ascontiguousarray(nodes, dtype=np.float64),
                np.ascontiguousarray(neighbor),
                np.ascontiguousarray(distance, dtype=np.float64),
                np.ascontiguousarray(cell_centers, dtype=np.float64),
                np.ascontiguousarray(values, dtype=np.float64),
                float(self.ridge),
            )
        offset = cell_centers[neighbor] - nodes[:, None, :]  # (m, k, 3), local coordinates
        # Gaussian weights with a per-node bandwidth = mean neighbour distance (floored above zero).
        bandwidth = np.maximum(distance.mean(axis=1, keepdims=True), 1.0e-30)
        weight = np.exp(-((distance / bandwidth) ** 2))  # (m, k)
        design = np.concatenate([np.ones((*distance.shape, 1)), offset], axis=2)  # (m, k, 4)
        weighted = weight[..., None] * design  # (m, k, 4)
        normal = np.einsum("mki,mkj->mij", weighted, design, optimize=True)  # (m, 4, 4)
        rhs = np.einsum("mki,mkv->miv", weighted, values[neighbor], optimize=True)  # (m, 4, n_vars)
        # Relative ridge: bump every diagonal by a fraction of the node's largest entry, so a
        # near-degenerate (e.g. one-sided boundary) stencil stays positive-definite and solvable.
        diagonal = np.einsum("mii->mi", normal)
        index = np.arange(4)
        normal[:, index, index] += self.ridge * diagonal.max(axis=1, keepdims=True)
        parameters = np.linalg.solve(normal, rhs)  # (m, 4, n_vars)
        return parameters[:, 0, :]  # the constant term is the fit at the node
