"""Validation study: CPU-vs-GPU render parity for the CUDA LOS render kernel (report-only).

Renders one synthetic Q perp volume (smooth streamer-like structure, NaN holes for the coverage
machinery, a polarity channel, and a synthetic density for the Thomson weighting) through all
render backends and prints their agreement against the single-threaded NumPy reference:

    (a) large-fov eclipse: the angular Gaussian and radial-power weights;
    (b) small-fov opaque at a fine step: scale-height weights and the in-integral body mask;
    (c) polarity hue: the nearest-cell polarity channel;
    (d) Thomson K: the density gather and the radial coefficient table.

Report-only: prints max / p99 relative differences over finite pixels for the result arrays
(signal, coverage, grayscale, image, polarity) and asserts nothing; the float64 column shows the
FMA tail, the mixed column the float32 sampling noise floor (NaN-boundary pixels dominate its
max). Run on demand:

    python validation/render_parity.py
"""

from __future__ import annotations

from typing import Any

import numpy as np
from astropy import units as u

from qorona.field.density import DensityVolume
from qorona.geometry.camera import OrthographicCamera
from qorona.radiation.thomson import ThomsonWeight
from qorona.render.los import LARGE_FOV, SMALL_FOV, _render_numpy, render
from qorona.resample.grid import LogarithmicSpacing, SphericalGrid
from qorona.squashing.volume import QPerpVolume, _pack_volume

_GRID = SphericalGrid(LogarithmicSpacing(1.0, 3.0), n_r=48, n_theta=72, n_phi=144)
_CAMERA = OrthographicCamera.from_sub_observer(
    longitude=20.0, latitude=10.0, fov=6.5 * u.R_sun, pixels=(192, 192)
)
#: The fixed render knobs shared by every case (the defaults of :func:`qorona.render.los.render`).
_COMMON: dict[str, Any] = {
    "clamp": (float(np.log10(2.0)), 7.0),
    "floor": True,
    "r_occult": 1.0,
    "occult_softness": 0.03,
    "percentiles": (1.0, 99.5),
    "display": "balanced",
    "thomson": None,
    "polarity_mode": "none",
    "show_progress": False,
}


def _node_spherical() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flat per-node ``(r, theta, phi)`` for :data:`_GRID`."""
    nodes = _GRID.node_points().reshape(-1, 3)
    r = np.linalg.norm(nodes, axis=1)
    theta = np.arccos(np.clip(nodes[:, 2] / r, -1.0, 1.0))
    phi = np.mod(np.arctan2(nodes[:, 1], nodes[:, 0]), 2.0 * np.pi)
    return r, theta, phi


def _synthetic_volume() -> QPerpVolume:
    """A smooth streamer-like log10 Q perp field with NaN holes and a polarity channel."""
    r, theta, phi = _node_spherical()
    log_q = (
        0.5
        + 2.5 * np.exp(-(((theta - np.pi / 2) / 0.35) ** 2)) * (1.0 + 0.5 * np.sin(4.0 * phi))
        + 1.5 * np.exp(-(((r - 1.8) / 0.2) ** 2))
    )
    log_q[np.sin(7.0 * phi) + np.cos(5.0 * theta) > 1.6] = np.nan
    return _pack_volume(_GRID, log_q, np.sign(np.cos(theta)))


def _synthetic_thomson() -> ThomsonWeight:
    """A hydrostatic-like density shape wrapped as the Thomson K weight."""
    r, _, phi = _node_spherical()
    density = (np.exp(-(r - 1.0) / 0.3) * (1.0 + 0.2 * np.cos(2.0 * phi))).reshape(
        _GRID.n_r, _GRID.n_theta, _GRID.n_phi
    )
    return ThomsonWeight(density=DensityVolume.from_grid_values(_GRID, density), mode="K")


def _compare(name: str, reference: Any, candidate: Any, threshold: float) -> str:
    """One report line: max / p99 relative difference over finite reference pixels.

    Finite-mask mismatches (a pixel finite on one backend, NaN on the other) are counted into the
    above-threshold tally so a masking divergence cannot hide in the NaN-skipping comparison.
    """
    arrays = [
        (attr, getattr(reference, attr), getattr(candidate, attr))
        for attr in ("signal", "coverage", "grayscale", "image")
    ]
    if reference.polarity is not None:
        arrays.append(("polarity", reference.polarity, candidate.polarity))
    worst = 0.0
    p99 = 0.0
    above = 0
    total = 0
    for _, ref, cand in arrays:
        finite = np.isfinite(ref) & np.isfinite(cand)
        if finite.any():
            scale = max(1.0, float(np.abs(ref[finite]).max()))
            diff = np.abs(ref[finite] - cand[finite]) / scale
            worst = max(worst, float(diff.max()))
            p99 = max(p99, float(np.percentile(diff, 99.0)))
            above += int(np.count_nonzero(diff > threshold))
            total += int(diff.size)
        above += int(np.count_nonzero(np.isfinite(ref) != np.isfinite(cand)))
    return (
        f"  {name:14s} max rel {worst:.3e}   p99 {p99:.3e}   above {threshold:g}: {above}/{total}"
    )


def main() -> None:
    volume = _synthetic_volume()
    cases: list[tuple[str, dict[str, Any]]] = [
        (
            "(a) large-fov eclipse",
            {"preset": LARGE_FOV, "occult": "eclipse", "step": 0.02},
        ),
        (
            "(b) small-fov opaque",
            {"preset": SMALL_FOV, "occult": "opaque", "step": 0.005},
        ),
        (
            "(c) polarity hue",
            {"preset": LARGE_FOV, "occult": "eclipse", "step": 0.02, "polarity_mode": "hue"},
        ),
        (
            "(d) thomson K",
            {
                "preset": LARGE_FOV,
                "occult": "eclipse",
                "step": 0.02,
                "thomson": _synthetic_thomson(),
            },
        ),
    ]
    for label, case in cases:
        case_kw = {**_COMMON, **case}
        print(label)
        reference = _render_numpy(volume, _CAMERA, chunk_size=500_000, **case_kw)
        cpu = render(volume, _CAMERA, device="cpu", **case_kw)
        gpu64 = render(volume, _CAMERA, device="gpu", precision="float64", **case_kw)
        mixed = render(volume, _CAMERA, device="gpu", precision="mixed", **case_kw)
        print(_compare("numba-cpu", reference, cpu, 1e-9))
        print(_compare("cuda float64", reference, gpu64, 1e-6))
        print(_compare("cuda mixed", reference, mixed, 1e-4))
    print("report-only: no assertions; see thresholds in the module docstring")


if __name__ == "__main__":
    main()
