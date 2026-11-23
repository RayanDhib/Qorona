"""Synthetic EUV images: the line-of-sight integral of the collisional emission of the corona.

For channel ``b`` one native pixel of the instrument records

    I_b = Ω_pix ∫ nₑ² K_b(T, nₑ) e^(-τ_b) ds,        τ_b(s) = ∫_s^observer n_H κ_b(T) ds',

where ``K_b`` is the channel's response table (:mod:`.response`), ``Ω_pix`` the solid angle of
one native pixel, and ``τ_b`` the optical depth of the foreground plasma between the emitting
point and the observer (photoionization of H I, He I and He II). ``nₑ`` and ``T`` come from a
:class:`~qorona.field.plasma.PlasmaVolume`. The camera and the ``s``-march are those of the
white-light render. The march runs from the observer inward, so ``τ_b`` is known when each
sample is added. A ray that meets the body stops at its front surface, because the disk is
opaque. Its deepest sample weighs the path length down to that surface, so the integral does
not jump where the surface point crosses the sample grid. The result is in the archive unit of
each channel per native pixel (for example AIA DN s⁻¹ pix⁻¹), at any render size. For a
solution that includes an artificially broadened transition region, ``K_b`` is divided by the
broadening factor of the run
(:meth:`~qorona.radiation.response.ResponseTable.corrected_for_broadening`).

Implemented from Del Zanna & Mason (2018), "Solar UV and X-ray spectral diagnostics", Living
Rev. Sol. Phys. 15, 5 (the optically thin emission and the instrument response).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from astropy import units as u

from qorona.accel import HAVE_NUMBA, apply_workers
from qorona.console import print_success, progress_bar
from qorona.field.plasma import PlasmaVolume
from qorona.geometry.camera import Camera, PerspectiveCamera
from qorona.radiation.response import DEFAULT_ABUNDANCES, ResponseTable

__all__ = ["EuvResult", "render_euv"]

#: Progress updates per render (ray chunks).
_PROGRESS_CHUNKS = 64


@dataclass(frozen=True)
class EuvResult:
    """Synthetic EUV frames of one camera, one per channel.

    Attributes
    ----------
    frames
        Channel name → ``(H, W)`` frame in the channel unit per native pixel and second.
    tables
        Channel name → the :class:`ResponseTable` the frame was rendered with.
    impact
        ``(H, W)`` per-pixel impact parameter (R☉).
    x, y, axis_unit
        The camera's pixel axes and their unit.
    absorption
        Whether the foreground absorption was applied.
    step
        The line-of-sight step (R☉).
    observer_distance
        The observer distance (R☉) for a perspective camera, else ``None``.
    tr_temperature
        The transition-region broadening temperature (K) the response was corrected for, or
        ``None`` when no correction was applied.
    plasma
        Decode facts of the plasma volume (μ, temperature source, floored cells).
    """

    frames: dict[str, np.ndarray]
    tables: dict[str, ResponseTable]
    impact: np.ndarray
    x: np.ndarray
    y: np.ndarray
    axis_unit: str
    absorption: bool
    step: float
    observer_distance: float | None
    tr_temperature: float | None = None
    plasma: dict[str, Any] = field(default_factory=dict)

    def unit(self, channel: str) -> str:
        """The unit of one channel's frame, for example ``DN s-1 pix-1``."""
        return f"{self.tables[channel].channel.unit} s-1 pix-1"

    def summary(self) -> str:
        """One line per channel: the median and the 99th percentile of the frame."""
        parts = []
        for name, frame in self.frames.items():
            finite = frame[np.isfinite(frame) & (frame > 0.0)]
            if finite.size:
                median, high = np.percentile(finite, [50.0, 99.0])
                parts.append(f"{name} median {median:.3g}, p99 {high:.3g} {self.unit(name)}")
            else:
                parts.append(f"{name}: no emission")
        return "; ".join(parts)


def _s_grid(plasma: PlasmaVolume, step: float) -> tuple[np.ndarray, float]:
    """The shared line-of-sight grid across the plasma shell and its spacing (R☉)."""
    outer = float(plasma.grid.radii[-1])
    n_steps = int(np.ceil(2.0 * outer / step)) + 1
    s_grid = np.linspace(-outer, outer, n_steps)
    return s_grid, float(s_grid[1] - s_grid[0])


def _stacked_tables(tables: list[ResponseTable]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``log K``, ``κ`` and ``n_H/nₑ`` of every channel on the shared table axes."""
    first = tables[0]
    for table in tables[1:]:
        if not (
            np.array_equal(table.log_t, first.log_t) and np.array_equal(table.log_ne, first.log_ne)
        ):
            raise ValueError("EUV response tables rendered together must share their axes")
    log_k = np.ascontiguousarray(np.stack([t.log_k for t in tables]))
    kappa = np.ascontiguousarray(np.stack([t.kappa for t in tables]))
    hydrogen = np.array([float(t.provenance["n_h_over_n_e"]) for t in tables])
    return log_k, kappa, hydrogen


def _euv_numpy(
    plasma: PlasmaVolume,
    camera: Camera,
    tables: list[ResponseTable],
    *,
    step: float,
    absorption: bool,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-threaded NumPy EUV render: the no-numba fallback and the kernel's reference.

    Returns ``(intensity (n_rays, n_ch) per steradian, impact (n_rays,))``.
    """
    s_grid, ds = _s_grid(plasma, step)
    ds_cm = ds * (1.0 * plasma.length_unit).to_value(u.cm)
    descending = s_grid[::-1]
    r_body = float(plasma.grid.radii[0])
    outer = float(plasma.grid.radii[-1])
    _, _, hydrogen = _stacked_tables(tables)
    height, width = camera.pixels
    n_rays = height * width
    intensity = np.zeros((n_rays, len(tables)))
    impact = np.empty(n_rays)
    rays_per_chunk = max(1, 2_000_000 // descending.size)
    with progress_bar("Rendering EUV", n_rays, enabled=show_progress) as progress:
        for start in range(0, n_rays, rays_per_chunk):
            stop = min(start + rays_per_chunk, n_rays)
            rays = camera.ray_batch(start, stop)
            origins, directions, batch_impact, observer_limit = rays.packed()
            impact[start:stop] = batch_impact
            m = stop - start
            directions = np.broadcast_to(directions, (m, 3))
            limit = np.broadcast_to(observer_limit, (m,))
            s = descending[None, :]
            radius = np.sqrt(batch_impact[:, None] ** 2 + s**2)
            surface = np.where(
                batch_impact < r_body,
                np.sqrt(np.maximum(r_body**2 - batch_impact**2, 0.0)),
                -np.inf,
            )
            on_path = (
                (radius >= r_body)
                & (radius <= outer)
                & (s <= limit[:, None])
                & (s >= surface[:, None])
            )
            # Sample the plasma on the path only; elsewhere nₑ = 0 and T is below every table.
            ray, sample = np.nonzero(on_path)
            points = origins[ray] + descending[sample, None] * directions[ray]
            log_ne = np.full(radius.shape, -np.inf)
            log_t = np.zeros(radius.shape)
            log_ne[ray, sample], log_t[ray, sample] = plasma.sample(points)
            ne = 10.0**log_ne
            # The deepest sample of a ray that ends on the body covers the path down to it.
            length = np.full(radius.shape, ds_cm)
            deepest = on_path & ~np.roll(on_path, -1, axis=1)
            deepest[:, -1] = on_path[:, -1]
            s_full = np.broadcast_to(s, radius.shape)
            surface_full = np.broadcast_to(surface[:, None], radius.shape)
            ends = deepest & (s_full - ds < surface_full)
            length[ends] = ds_cm * (s_full[ends] + 0.5 * ds - surface_full[ends]) / ds
            for c, table in enumerate(tables):
                emission = ne**2 * table.evaluate(log_t, log_ne) * length
                if absorption:
                    d_tau = hydrogen[c] * ne * table.absorption(log_t) * length
                    tau = np.cumsum(d_tau, axis=1) - d_tau
                    emission = emission * np.exp(-tau)
                intensity[start:stop, c] = emission.sum(axis=1)
            progress(stop)
    return intensity, impact


def _euv_numba(
    plasma: PlasmaVolume,
    camera: Camera,
    tables: list[ResponseTable],
    *,
    step: float,
    absorption: bool,
    workers: int | None,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """numba ``prange``-over-rays EUV render; agrees with the NumPy reference to rounding."""
    from qorona.accel.kernels import euv_batch_jit

    apply_workers(workers)
    s_grid, ds = _s_grid(plasma, step)
    ds_cm = ds * (1.0 * plasma.length_unit).to_value(u.cm)
    log_k, kappa, hydrogen = _stacked_tables(tables)
    axes = tables[0]
    payload = np.ascontiguousarray(plasma.payload)
    grid = plasma.grid._jit_grid()
    height, width = camera.pixels
    n_rays = height * width
    intensity = np.zeros((n_rays, len(tables)))
    impact = np.empty(n_rays)
    rays_per_chunk = max(1, -(-n_rays // _PROGRESS_CHUNKS))
    with progress_bar("Rendering EUV (numba)", n_rays, enabled=show_progress) as progress:
        for start in range(0, n_rays, rays_per_chunk):
            stop = min(start + rays_per_chunk, n_rays)
            rays = camera.ray_batch(start, stop)
            origins, directions, batch_impact, observer_limit = rays.packed()
            impact[start:stop] = batch_impact
            intensity[start:stop] = euv_batch_jit(
                origins,
                directions,
                batch_impact,
                observer_limit,
                s_grid,
                payload,
                grid,
                float(plasma.grid.radii[0]),
                float(axes.log_t[0]),
                float(1.0 / (axes.log_t[1] - axes.log_t[0])),
                float(axes.log_ne[0]),
                float(1.0 / (axes.log_ne[1] - axes.log_ne[0])),
                log_k,
                kappa,
                hydrogen,
                absorption,
                ds_cm,
            )
            progress(stop)
    return intensity, impact


def render_euv(
    plasma: PlasmaVolume,
    camera: Camera,
    channels: tuple[str, ...],
    *,
    abundances: str = DEFAULT_ABUNDANCES,
    step: float = 0.001,
    absorption: bool = True,
    tr_temperature: float | None = None,
    backend: str = "auto",
    workers: int | None = None,
    show_progress: bool = True,
) -> EuvResult:
    """Render synthetic EUV frames of ``plasma`` through ``camera``, one per channel.

    Parameters
    ----------
    plasma
        Electron density and temperature.
    camera
        The camera. Its rays give the pixel geometry; the frame values are per native pixel of
        each channel's instrument, whatever the camera's pixel size.
    channels
        Channel names (keys of :data:`qorona.radiation.response.CHANNELS`).
    abundances
        Abundance set of the response tables.
    step
        Line-of-sight step (R☉).
    absorption
        Whether to apply the foreground absorption.
    tr_temperature
        The run's transition-region broadening temperature (K): the response is divided by the
        broadening factor of this temperature. ``None`` applies no correction.
    backend
        ``"auto"`` (numba when installed), ``"numba"`` or ``"numpy"``.
    workers
        numba thread count (``None``: all cores).
    show_progress
        Whether to show progress.

    Returns
    -------
    EuvResult
        The frames and their recipe.
    """
    if not channels:
        raise ValueError("render_euv needs at least one channel")
    tables = [ResponseTable.load(name, abundances) for name in channels]
    if tr_temperature is not None:
        tables = [table.corrected_for_broadening(tr_temperature) for table in tables]
    use_numba = backend == "numba" or (backend == "auto" and HAVE_NUMBA)
    if backend not in ("auto", "numba", "numpy"):
        raise ValueError(f"unknown EUV backend {backend!r}")
    if use_numba:
        intensity, impact = _euv_numba(
            plasma,
            camera,
            tables,
            step=step,
            absorption=absorption,
            workers=workers,
            show_progress=show_progress,
        )
    else:
        intensity, impact = _euv_numpy(
            plasma, camera, tables, step=step, absorption=absorption, show_progress=show_progress
        )
    height, width = camera.pixels
    frames = {
        name: (intensity[:, c] * tables[c].native_pixel_sr).reshape(height, width)
        for c, name in enumerate(channels)
    }
    x, y = camera.pixel_axes()
    observer = (
        float(camera.observer_distance.to_value(u.R_sun))
        if isinstance(camera, PerspectiveCamera)
        else None
    )
    result = EuvResult(
        frames=frames,
        tables=dict(zip(channels, tables, strict=True)),
        impact=impact.reshape(height, width),
        x=x,
        y=y,
        axis_unit=camera.axis_unit,
        absorption=absorption,
        step=step,
        observer_distance=observer,
        tr_temperature=tr_temperature,
        plasma={
            "mean_molecular_weight": plasma.mean_molecular_weight,
            "temperature_source": plasma.temperature_source,
            "floored_cells": plasma.floored,
            "transition_region": plasma.transition_region,
        },
    )
    if show_progress:
        print_success(f"Rendered {width}x{height} EUV image: {result.summary()}")
    return result
