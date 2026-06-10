"""numba ``prange`` kernel for the k-NN MLS normal-equations assemble + 4x4 solve.

The hot path of :class:`~qorona.resample.resampler.KnnMlsResampler`: a per-node degree-1 moving
least-squares fit. The cost is the *assemble* of the weighted normal equations and the 4x4 solve,
not the k-NN search, so the neighbour query stays on the CPU ``cKDTree`` (exact-k) and only the
embarrassingly-parallel per-node assemble + solve run here, over the same neighbours. This
reproduces ``KnnMlsResampler._fit_chunk`` to float64 round-off: a pure parallelization with no new
approximation.

Imported only when numba is present (``HAVE_NUMBA``); the NumPy einsum path in ``resampler.py`` is
the fallback and the reference for cross-checks.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def fit_mls_chunk(
    nodes: np.ndarray,
    neighbor: np.ndarray,
    distance: np.ndarray,
    cell_centers: np.ndarray,
    values: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """Fit the degree-1 MLS model at each node; return the constant terms ``(m, n_vars)``.

    Per node (one ``prange`` lane): Gaussian weights with bandwidth = mean neighbour distance, the
    weighted 4x4 normal matrix and ``(4, n_vars)`` rhs accumulated over the ``k`` neighbours in
    index order (matching the reference einsum reduction), a relative ridge on the diagonal, then a
    4x4 solve whose constant term is the fit. ``nodes (m,3)``, ``neighbor``/``distance (m,k)``,
    ``cell_centers (N,3)``, ``values (N,n_vars)``.
    """
    m = nodes.shape[0]
    k = neighbor.shape[1]
    n_vars = values.shape[1]
    out = np.empty((m, n_vars))
    for p in prange(m):
        # Bandwidth = mean neighbour distance (floored), summed in index order like distance.mean.
        s = 0.0
        for kk in range(k):
            s += distance[p, kk]
        bw = s / k
        if bw < 1.0e-30:
            bw = 1.0e-30
        normal = np.zeros((4, 4))
        rhs = np.zeros((4, n_vars))
        for kk in range(k):
            nb = neighbor[p, kk]
            d0 = 1.0
            d1 = cell_centers[nb, 0] - nodes[p, 0]
            d2 = cell_centers[nb, 1] - nodes[p, 1]
            d3 = cell_centers[nb, 2] - nodes[p, 2]
            r = distance[p, kk] / bw
            w = np.exp(-(r * r))
            w0 = w * d0
            w1 = w * d1
            w2 = w * d2
            w3 = w * d3
            normal[0, 0] += w0 * d0
            normal[0, 1] += w0 * d1
            normal[0, 2] += w0 * d2
            normal[0, 3] += w0 * d3
            normal[1, 0] += w1 * d0
            normal[1, 1] += w1 * d1
            normal[1, 2] += w1 * d2
            normal[1, 3] += w1 * d3
            normal[2, 0] += w2 * d0
            normal[2, 1] += w2 * d1
            normal[2, 2] += w2 * d2
            normal[2, 3] += w2 * d3
            normal[3, 0] += w3 * d0
            normal[3, 1] += w3 * d1
            normal[3, 2] += w3 * d2
            normal[3, 3] += w3 * d3
            for v in range(n_vars):
                val = values[nb, v]
                rhs[0, v] += w0 * val
                rhs[1, v] += w1 * val
                rhs[2, v] += w2 * val
                rhs[3, v] += w3 * val
        # Relative ridge: bump every diagonal by a fraction of the largest diagonal entry.
        dmax = normal[0, 0]
        if normal[1, 1] > dmax:
            dmax = normal[1, 1]
        if normal[2, 2] > dmax:
            dmax = normal[2, 2]
        if normal[3, 3] > dmax:
            dmax = normal[3, 3]
        bump = ridge * dmax
        normal[0, 0] += bump
        normal[1, 1] += bump
        normal[2, 2] += bump
        normal[3, 3] += bump
        params = np.linalg.solve(normal, rhs)  # (4, n_vars); the constant term is the fit
        for v in range(n_vars):
            out[p, v] = params[0, v]
    return out
