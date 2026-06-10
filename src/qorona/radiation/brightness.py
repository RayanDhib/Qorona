"""The standalone white-light / polarized-brightness (pB) render: the secondary white-light product.

A Thomson-scattering line-of-sight integral over the electron density, on the *same* orthographic
camera and ``s``-march as the Q⊥ render but a different integrand: along each ray it accumulates
the tangential and polarized brightness

    K_tan(rho) = ∫ Nₑ · c_tan(r) ds,    K_pol(rho) = ∫ Nₑ · c_pol(r) · sin²χ̄ ds,

and forms the total brightness ``K_tot = 2 K_tan - K_pol`` (a cheap extra observable)
and the polarized brightness ``pB = K_pol``, the reference target finished by the display treatments
in :mod:`.display`. The radius-only coefficients ``c_tan``, ``c_pol`` come from :mod:`.thomson`; the
electron density is the only field input, and being dense the image is NaN-free.

The product is *relative*: the single-electron prefactor and the absolute electron-density
calibration are dropped (pB is conventionally shown in relative / log units), so pixel-to-pixel
structure and the polarization ratio ``P = pB / K_tot`` are exact while the overall scale is
arbitrary. The quadrature runs on a numba ``prange``-over-rays kernel when installed, with a
NumPy path as the fallback and reference implementation.

Implemented from Inhester (2015), "Thomson Scattering in the Solar Corona", arXiv:1512.00651
(Eqs. 4.3, the line-of-sight brightness integrals).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from qorona.accel import HAVE_NUMBA, apply_workers
from qorona.console import print_success, progress_bar
from qorona.field.density import DensityVolume
from qorona.geometry.camera import OrthographicCamera
from qorona.radiation.thomson import (
    ASYMPTOTIC_CROSSOVER,
    LIMB_DARKENING,
    RadialCoefficients,
    build_coefficient_table,
)
from qorona.render.los import _eclipse_alpha, _occultation_mask, _scale_intensity, _write_png

__all__ = ["BrightnessResult", "render_brightness"]

#: Number of ray-chunks the kernel brightness render splits the image into, for progress only (the
#: kernel touches one sample at a time and needs no memory bound); mirrors the Q⊥ render.
_PROGRESS_CHUNKS = 64

#: Which frame a bare :meth:`BrightnessResult.save_png` writes: the reference pB product.
_DEFAULT_FRAME: Literal["polarized", "total"] = "polarized"


@dataclass(frozen=True)
class BrightnessResult:
    """The line-of-sight brightness frames and the geometry needed to finish them.

    Attributes
    ----------
    polarized
        ``(H, W)`` polarized brightness ``pB = K_pol`` (relative units): the reference target for
        the display treatments.
    total
        ``(H, W)`` total / white-light brightness ``K_tot = 2 K_tan - K_pol`` (relative units).
    impact
        ``(H, W)`` per-pixel impact parameter rho (R☉): the radius the display treatments
        and the eclipse vignette use.
    u
        The limb-darkening ``u`` the coefficient table was built with (recorded for provenance).
    occult
        The occultation mode the frames were finished with (``"eclipse"`` / ``"opaque"`` /
        ``"none"``); recorded for provenance.
    r_occult, occult_softness
        The occulter radius and eclipse-edge feather (R☉) used; recorded for provenance.
    """

    polarized: np.ndarray
    total: np.ndarray
    impact: np.ndarray
    u: float
    occult: str
    r_occult: float
    occult_softness: float

    def polarization(self) -> np.ndarray:
        """Return the ``(H, W)`` polarization ``P = pB / K_tot`` (``0`` where ``K_tot ≤ 0``)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.total > 0.0, self.polarized / self.total, 0.0)

    def summary(self) -> str:
        """Return a one-line summary: the median polarization and the pB dynamic range."""
        polarized = self.polarized[self.polarized > 0.0]
        if not polarized.size:
            return "no positive pB samples"
        decades = np.log10(polarized.max() / polarized.min())
        median_p = float(np.median(self.polarization()[self.total > 0.0]))
        return f"median polarization {median_p:.2f} · pB spans {decades:.1f} decades"

    def save_png(
        self,
        path: str | Path,
        *,
        frame: Literal["polarized", "total"] = _DEFAULT_FRAME,
        percentiles: tuple[float, float] = (1.0, 99.5),
        scaling: Literal["linear", "log"] = "log",
    ) -> None:
        """Write a brightness frame to ``path`` as a stretched 8-bit grayscale PNG.

        The chosen frame (``"polarized"`` pB by default, or ``"total"``) is percentile-stretched,
        logarithmically by default, matching how pB is conventionally displayed. The disk is already
        occulted in the frame for ``"eclipse"`` mode. The stretch percentiles are anchored on the
        pixels carrying positive brightness, so the zeroed occulted disk and off-shell background do
        not collapse the corona's dynamic range (a log stretch would otherwise read those zeros as
        ``log10(0)`` and wash the corona to white).
        """
        image = self.polarized if frame == "polarized" else self.total
        flat = image.reshape(-1)
        stretched = _scale_intensity(flat[:, None], scaling, percentiles, anchor=flat > 0.0)[:, 0]
        gray = (np.clip(np.nan_to_num(stretched), 0.0, 1.0) * 255.0).round().astype(np.uint8)
        rgb = np.repeat(gray.reshape(*image.shape, 1), 3, axis=2)
        _write_png(Path(path), np.ascontiguousarray(rgb))


def _ray_geometry(
    camera: OrthographicCamera, density: DensityVolume, step: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return the flattened ``(origins, impact, look, s_grid, ds)`` shared by both render paths."""
    rays = camera.rays()
    origins = np.ascontiguousarray(rays.origins.reshape(-1, 3))
    impact = np.ascontiguousarray(rays.impact.reshape(-1))
    look = np.ascontiguousarray(rays.look, dtype=np.float64)
    outer_radius = float(density.grid.radii[-1])
    n_steps = int(np.ceil(2.0 * outer_radius / step)) + 1
    s_grid = np.linspace(-outer_radius, outer_radius, n_steps)
    ds = float(s_grid[1] - s_grid[0])
    return origins, impact, look, s_grid, ds


def _finalize_brightness(
    k_tan: np.ndarray,
    k_pol: np.ndarray,
    impact: np.ndarray,
    ds: float,
    *,
    height: int,
    width: int,
    u: float,
    occult: str,
    r_occult: float,
    occult_softness: float,
    show_progress: bool,
) -> BrightnessResult:
    """Scale by the step, occult the disk, form pB and K_tot, and assemble the result.

    In ``"eclipse"`` mode the occulter is opaque to all light at ``rho < r_occult``, so the frames
    are darkened image-side (the observable eclipse product: a dark disk, off-limb corona only) by
    the eclipse vignette, the same darkening the Q⊥ render applies, but baked into the frame here so
    the display treatments and the saved image all see the occulted pB. ``"opaque"`` (the 3-D view,
    near-side corona over the disk) and ``"none"`` leave the frames as integrated.
    """
    impact_grid = impact.reshape(height, width)
    polarized = (k_pol * ds).reshape(height, width)
    total = ((2.0 * k_tan - k_pol) * ds).reshape(height, width)
    if occult == "eclipse":
        alpha = _eclipse_alpha(impact_grid, r_occult, occult_softness)
        polarized = polarized * alpha
        total = total * alpha
    result = BrightnessResult(
        polarized=polarized,
        total=total,
        impact=impact_grid,
        u=u,
        occult=occult,
        r_occult=r_occult,
        occult_softness=occult_softness,
    )
    if show_progress:
        print_success(f"Rendered {width}x{height} brightness image: {result.summary()}")
    return result


def _brightness_numpy(
    density: DensityVolume,
    camera: OrthographicCamera,
    table: RadialCoefficients,
    *,
    step: float,
    occult: str,
    r_occult: float,
    occult_softness: float,
    u: float,
    chunk_size: int,
    show_progress: bool,
) -> BrightnessResult:
    """Single-threaded NumPy brightness render: the no-numba fallback and the kernel's reference."""
    height, width = camera.pixels
    origins, impact, look, s_grid, ds = _ray_geometry(camera, density, step)
    n_steps = s_grid.shape[0]
    n_rays = origins.shape[0]
    inner_radius = float(density.grid.radii[0])
    outer_radius = float(density.grid.radii[-1])
    occult_body = occult == "opaque"

    k_tan = np.zeros(n_rays)
    k_pol = np.zeros(n_rays)
    rays_per_chunk = max(1, chunk_size // n_steps)
    with progress_bar("Rendering brightness (pB)", n_rays, enabled=show_progress) as progress:
        for start in range(0, n_rays, rays_per_chunk):
            stop = min(start + rays_per_chunk, n_rays)
            batch_impact = impact[start:stop]
            points = origins[start:stop, None, :] + s_grid[None, :, None] * look
            radius = np.sqrt(batch_impact[:, None] ** 2 + s_grid[None, :] ** 2)

            on_path = (radius >= inner_radius) & (radius <= outer_radius)
            if occult_body:
                on_path = on_path & ~_occultation_mask(batch_impact, s_grid, r_occult)
            density_sample = density.sample(points.reshape(-1, 3)).reshape(radius.shape)
            with np.errstate(invalid="ignore", divide="ignore"):
                c_tan, c_pol = table.evaluate(radius)
                sin_sq_chi = batch_impact[:, None] ** 2 / radius**2
            weighted = np.where(on_path, density_sample, 0.0)
            k_tan[start:stop] = np.sum(weighted * c_tan, axis=1)
            k_pol[start:stop] = np.sum(weighted * c_pol * sin_sq_chi, axis=1)
            progress(stop)

    return _finalize_brightness(
        k_tan, k_pol, impact, ds, height=height, width=width, u=u, occult=occult,
        r_occult=r_occult, occult_softness=occult_softness, show_progress=show_progress,
    )


def _brightness_numba(
    density: DensityVolume,
    camera: OrthographicCamera,
    table: RadialCoefficients,
    *,
    step: float,
    occult: str,
    r_occult: float,
    occult_softness: float,
    u: float,
    workers: int | None,
    show_progress: bool,
) -> BrightnessResult:
    """numba ``prange``-over-rays brightness render; agrees with the NumPy reference to
    floating-point noise."""
    from qorona.accel.kernels import brightness_batch_jit

    apply_workers(workers)
    height, width = camera.pixels
    origins, impact, look, s_grid, ds = _ray_geometry(camera, density, step)
    n_rays = origins.shape[0]
    occult_body = occult == "opaque"
    density_vol = np.ascontiguousarray(density.density)
    density_grid = density.grid._jit_grid()
    c_tan_table = np.ascontiguousarray(table.c_tan)
    c_pol_table = np.ascontiguousarray(table.c_pol)

    k_tan = np.zeros(n_rays)
    k_pol = np.zeros(n_rays)
    rays_per_chunk = max(1, -(-n_rays // _PROGRESS_CHUNKS))
    with progress_bar("Rendering pB (numba)", n_rays, enabled=show_progress) as progress:
        for start in range(0, n_rays, rays_per_chunk):
            stop = min(start + rays_per_chunk, n_rays)
            chunk_tan, chunk_pol = brightness_batch_jit(
                origins[start:stop],
                look,
                impact[start:stop],
                s_grid,
                density_vol,
                density_grid,
                float(r_occult),
                occult_body,
                table.log_inner,
                table.inv_dlog,
                c_tan_table,
                c_pol_table,
            )
            k_tan[start:stop] = chunk_tan
            k_pol[start:stop] = chunk_pol
            progress(stop)

    return _finalize_brightness(
        k_tan, k_pol, impact, ds, height=height, width=width, u=u, occult=occult,
        r_occult=r_occult, occult_softness=occult_softness, show_progress=show_progress,
    )


def render_brightness(
    density: DensityVolume,
    camera: OrthographicCamera,
    *,
    u: float = LIMB_DARKENING,
    crossover: float = ASYMPTOTIC_CROSSOVER,
    step: float = 0.02,
    occult: Literal["eclipse", "opaque", "none"] = "eclipse",
    r_occult: float = 1.0,
    occult_softness: float = 0.03,
    chunk_size: int = 500_000,
    workers: int | None = None,
    show_progress: bool = True,
) -> BrightnessResult:
    """Render the polarized and total Thomson brightness over an electron-density volume.

    Integrates ``K_tan`` and ``K_pol`` along each orthographic line of sight and returns the
    polarized brightness ``pB = K_pol`` and the total ``K_tot = 2 K_tan - K_pol`` (relative units;
    the absolute calibration is dropped). Dispatches to a numba kernel when available, else the
    NumPy path (the reference implementation); the two agree to floating-point noise.

    Parameters
    ----------
    density
        The electron-density volume to integrate (the only field input; dense, so the image is
        NaN-free).
    camera
        The orthographic plane-of-sky camera (shared with the Q⊥ render).
    u
        Limb-darkening coefficient (default :data:`~qorona.radiation.thomson.LIMB_DARKENING`).
    crossover
        Closed-form to asymptotic coefficient crossover radius in R☉.
    step
        Line-of-sight sample spacing in R☉.
    occult
        Occultation mode. ``"eclipse"`` (default) integrates the full off-limb corona and darkens
        the disk image-side after integration; ``"opaque"`` drops the far side behind the body in
        the integral; ``"none"`` disables both.
    r_occult, occult_softness
        Occulter radius and eclipse-edge feather in R☉ (carried into the result for the display).
    chunk_size
        Rays x steps processed per batch on the NumPy path (bounds its peak memory; unused by the
        kernel, which chunks only for progress).
    workers
        numba thread count (``None`` = all cores). Ignored without numba.
    show_progress
        Whether to display progress.

    Returns
    -------
    BrightnessResult
        The polarized and total brightness frames, the impact-parameter grid, and a record of the
        limb-darkening and occultation settings the frames were built with.
    """
    if occult not in ("eclipse", "opaque", "none"):
        raise ValueError(f"occult must be 'eclipse', 'opaque', or 'none', not {occult!r}")
    table = build_coefficient_table(
        float(density.grid.radii[0]), float(density.grid.radii[-1]), u=u, crossover=crossover
    )
    if HAVE_NUMBA and density.grid._jit_grid() is not None:
        return _brightness_numba(
            density, camera, table, step=step, occult=occult, r_occult=r_occult,
            occult_softness=occult_softness, u=u, workers=workers, show_progress=show_progress,
        )
    return _brightness_numpy(
        density, camera, table, step=step, occult=occult, r_occult=r_occult,
        occult_softness=occult_softness, u=u, chunk_size=chunk_size, show_progress=show_progress,
    )
