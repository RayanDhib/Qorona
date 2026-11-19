"""The standalone white-light / polarized-brightness (pB) render: the secondary white-light product.

A Thomson-scattering line-of-sight integral over the electron density, on the *same* camera and
``s``-march as the Q⊥ render but a different integrand: along each ray it accumulates the
tangential and polarized brightness

    K_tan(rho) = ∫ Nₑ · c_tan(r) ds,    K_pol(rho) = ∫ Nₑ · c_pol(r) · sin²χ̄ ds,

and forms the total brightness ``K_tot = 2 K_tan - K_pol`` (a cheap extra observable)
and the polarized brightness ``pB = K_pol``, the reference target finished by the display treatments
in :mod:`.display`. The radius-only coefficients ``c_tan``, ``c_pol`` come from :mod:`.thomson`; the
electron density is the only field input, and being dense the image is NaN-free.

The sums carry no single-electron prefactor, so by themselves the frames are *relative*.
Pixel-to-pixel structure and the polarization ratio ``P = pB / K_tot`` are exact while the overall
scale is arbitrary. When the density volume carries its physical unit, the render restores the
prefactor and the density calibration at the end. It then returns the frames in units of the mean
solar brightness (MSB), the unit of PUNCH Level 3 images. The quadrature runs on a numba
``prange``-over-rays kernel when installed, with a NumPy path as the fallback and reference
implementation.

Implemented from Inhester (2015), "Thomson Scattering in the Solar Corona", arXiv:1512.00651
(Eqs. 4.3, the line-of-sight brightness integrals).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from astropy import units

from qorona.accel import HAVE_NUMBA, apply_workers
from qorona.console import print_success, progress_bar
from qorona.field.density import DensityVolume
from qorona.geometry.camera import Camera, PerspectiveCamera
from qorona.radiation.thomson import (
    ASYMPTOTIC_CROSSOVER,
    LIMB_DARKENING,
    RadialCoefficients,
    build_coefficient_table,
    msb_prefactor,
)
from qorona.render.image import eclipse_alpha, occulter_impact

__all__ = ["BrightnessResult", "render_brightness"]

#: Number of ray-chunks the kernel brightness render splits the image into, for progress only (the
#: kernel touches one sample at a time and needs no memory bound); mirrors the Q⊥ render.
_PROGRESS_CHUNKS = 64


@dataclass(frozen=True)
class BrightnessResult:
    """The line-of-sight brightness frames and the geometry needed to finish them.

    Attributes
    ----------
    polarized
        ``(H, W)`` polarized brightness ``pB = K_pol`` in :attr:`unit`: the reference target for
        the display treatments.
    total
        ``(H, W)`` total / white-light brightness ``K_tot = 2 K_tan - K_pol`` in :attr:`unit`.
    impact
        ``(H, W)`` per-pixel impact parameter rho (R☉): the radius the display treatments
        and the eclipse vignette use.
    x, y
        ``(W,)`` and ``(H,)`` pixel-centre coordinates in :attr:`axis_unit`: ``x`` increases to
        image-right, ``y`` to image-up with row 0 at the top. Carried so a data export is
        self-describing without re-deriving the camera geometry.
    axis_unit
        ``"R_sun"`` for the orthographic camera (plane-of-sky lengths, so ``rho = hypot(x, y)``)
        or ``"deg"`` for the perspective camera (projection-plane angular offsets from Sun
        centre). Away from the image axes, these offsets are not helioprojective longitude and
        latitude; use the FITS WCS for those.
    r_inner, r_outer
        The radial shell (R☉) the integral covered; the display vignette integrates its reference
        model over the same shell.
    u
        The limb-darkening ``u`` the coefficient table was built with (recorded for provenance).
    occult
        The occultation mode the frames were finished with (``"eclipse"`` /
        ``"none"``); recorded for provenance.
    r_occult, occult_softness
        The occulter radius and eclipse-edge feather (R☉) used; recorded for provenance.
    observer_distance
        The perspective observer's distance from Sun centre (R☉), or ``None`` for the
        orthographic camera.
    observer_limit
        ``(H, W)`` the observer's position ``s`` on each line of sight, or ``None`` for the
        orthographic camera. The integral stops there, and the newkirk vignette stops its model
        curve at the same place. ``s`` is in R☉, measured from the ray's closest approach to Sun
        centre, positive toward the observer.
    unit
        ``"MSB"`` when the frames are absolute, in units of the mean solar brightness ``B☉``
        (the density's physical unit was known). ``"relative"`` otherwise.
    """

    polarized: np.ndarray
    total: np.ndarray
    impact: np.ndarray
    x: np.ndarray
    y: np.ndarray
    axis_unit: str
    r_inner: float
    r_outer: float
    u: float
    occult: str
    r_occult: float
    occult_softness: float
    observer_distance: float | None
    observer_limit: np.ndarray | None
    unit: str = "relative"

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


def _s_grid(density: DensityVolume, step: float) -> tuple[np.ndarray, float]:
    """Return the shared line-of-sight grid ``s_grid`` and its spacing ``ds`` across the shell."""
    outer_radius = float(density.grid.radii[-1])
    n_steps = int(np.ceil(2.0 * outer_radius / step)) + 1
    s_grid = np.linspace(-outer_radius, outer_radius, n_steps)
    ds = float(s_grid[1] - s_grid[0])
    return s_grid, ds


def _finalize_brightness(
    k_tan: np.ndarray,
    k_pol: np.ndarray,
    impact: np.ndarray,
    observer_limit: np.ndarray,
    ds: float,
    *,
    camera: Camera,
    density: DensityVolume,
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
    the display treatments and the saved image all see the occulted pB. ``"none"`` leaves the
    frames as integrated. The pixel axes come from the camera and are carried on the result, so a
    data export is self-describing. A perspective camera also leaves its observer distance and the
    per-ray ``observer_limit`` on the result for the display treatments.
    """
    height, width = camera.pixels
    observer_distance: float | None = None
    observer_grid: np.ndarray | None = None
    if isinstance(camera, PerspectiveCamera):
        observer_distance = float(camera.observer_distance.to_value(units.R_sun))
        observer_grid = observer_limit.reshape(height, width)
    x, y = camera.pixel_axes()
    impact_grid = impact.reshape(height, width)
    polarized = (k_pol * ds).reshape(height, width)
    total = ((2.0 * k_tan - k_pol) * ds).reshape(height, width)
    if occult == "eclipse":
        occulter = occulter_impact(impact, observer_limit).reshape(height, width)
        alpha = eclipse_alpha(occulter, r_occult, occult_softness)
        polarized = polarized * alpha
        total = total * alpha
    result = BrightnessResult(
        polarized=polarized,
        total=total,
        impact=impact_grid,
        x=x,
        y=y,
        axis_unit=camera.axis_unit,
        r_inner=float(density.grid.radii[0]),
        r_outer=float(density.grid.radii[-1]),
        u=u,
        occult=occult,
        r_occult=r_occult,
        occult_softness=occult_softness,
        observer_distance=observer_distance,
        observer_limit=observer_grid,
    )
    if show_progress:
        print_success(f"Rendered {width}x{height} brightness image: {result.summary()}")
    return result


def _brightness_numpy(
    density: DensityVolume,
    camera: Camera,
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
    s_grid, ds = _s_grid(density, step)
    n_steps = s_grid.shape[0]
    height, width = camera.pixels
    n_rays = height * width
    impact = np.empty(n_rays)
    limit = np.full(n_rays, np.inf)
    inner_radius = float(density.grid.radii[0])
    outer_radius = float(density.grid.radii[-1])

    k_tan = np.zeros(n_rays)
    k_pol = np.zeros(n_rays)
    rays_per_chunk = max(1, chunk_size // n_steps)
    with progress_bar("Rendering brightness (pB)", n_rays, enabled=show_progress) as progress:
        for start in range(0, n_rays, rays_per_chunk):
            stop = min(start + rays_per_chunk, n_rays)
            rays = camera.ray_batch(start, stop)
            origins, directions, batch_impact, observer_limit = rays.packed()
            impact[start:stop] = batch_impact
            if not rays.parallel:
                limit[start:stop] = observer_limit
            points = origins[:, None, :] + s_grid[None, :, None] * directions[:, None, :]
            radius = np.sqrt(batch_impact[:, None] ** 2 + s_grid[None, :] ** 2)

            on_path = (
                (radius >= inner_radius)
                & (radius <= outer_radius)
                & (s_grid[None, :] <= observer_limit[:, None])
            )
            density_sample = density.sample(points.reshape(-1, 3)).reshape(radius.shape)
            with np.errstate(invalid="ignore", divide="ignore"):
                c_tan, c_pol = table.evaluate(radius)
            sin_sq_chi = np.divide(
                batch_impact[:, None] ** 2, radius**2, out=np.zeros_like(radius), where=radius > 0.0
            )
            weighted = np.where(on_path, density_sample, 0.0)
            k_tan[start:stop] = np.sum(weighted * c_tan, axis=1)
            k_pol[start:stop] = np.sum(weighted * c_pol * sin_sq_chi, axis=1)
            progress(stop)

    return _finalize_brightness(
        k_tan,
        k_pol,
        impact,
        limit,
        ds,
        camera=camera,
        density=density,
        u=u,
        occult=occult,
        r_occult=r_occult,
        occult_softness=occult_softness,
        show_progress=show_progress,
    )


def _brightness_numba(
    density: DensityVolume,
    camera: Camera,
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
    s_grid, ds = _s_grid(density, step)
    height, width = camera.pixels
    n_rays = height * width
    impact = np.empty(n_rays)
    limit = np.full(n_rays, np.inf)
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
            rays = camera.ray_batch(start, stop)
            origins, directions, chunk_impact, observer_limit = rays.packed()
            impact[start:stop] = chunk_impact
            if not rays.parallel:
                limit[start:stop] = observer_limit
            chunk_tan, chunk_pol = brightness_batch_jit(
                origins,
                directions,
                chunk_impact,
                observer_limit,
                s_grid,
                density_vol,
                density_grid,
                float(r_occult),
                False,
                table.log_inner,
                table.inv_dlog,
                c_tan_table,
                c_pol_table,
            )
            k_tan[start:stop] = chunk_tan
            k_pol[start:stop] = chunk_pol
            progress(stop)

    return _finalize_brightness(
        k_tan,
        k_pol,
        impact,
        limit,
        ds,
        camera=camera,
        density=density,
        u=u,
        occult=occult,
        r_occult=r_occult,
        occult_softness=occult_softness,
        show_progress=show_progress,
    )


def render_brightness(
    density: DensityVolume,
    camera: Camera,
    *,
    u: float = LIMB_DARKENING,
    crossover: float = ASYMPTOTIC_CROSSOVER,
    step: float = 0.02,
    occult: Literal["eclipse", "none"] = "eclipse",
    r_occult: float = 1.0,
    occult_softness: float = 0.03,
    scale: Literal["auto", "relative"] = "auto",
    chunk_size: int = 500_000,
    workers: int | None = None,
    show_progress: bool = True,
) -> BrightnessResult:
    """Render the polarized and total Thomson brightness over an electron-density volume.

    Integrates ``K_tan`` and ``K_pol`` along each line of sight and returns the
    polarized brightness ``pB = K_pol`` and the total ``K_tot = 2 K_tan - K_pol``. The frames are
    in mean solar brightness when the density carries its physical unit (see ``scale``), and in
    relative units otherwise. Dispatches to a numba kernel when available, else the NumPy path (the
    reference implementation); the two agree to floating-point noise.

    Parameters
    ----------
    density
        The electron-density volume to integrate (the only field input; dense, so the image is
        NaN-free).
    camera
        The camera, shared with the Q⊥ render: orthographic (parallel rays) or perspective (a
        finite observer, whose lines of sight stop at the observer).
    u
        Limb-darkening coefficient (default :data:`~qorona.radiation.thomson.LIMB_DARKENING`).
    crossover
        Closed-form to asymptotic coefficient crossover radius in R☉.
    step
        Line-of-sight sample spacing in R☉.
    occult
        Occultation mode. ``"eclipse"`` (default) integrates the full line of sight and darkens
        the disk image-side after integration; ``"none"`` leaves the frames as integrated. Both
        integrate the same full-line geometry as the display vignettes' model curves.
    r_occult, occult_softness
        Occulter radius and eclipse-edge feather in R☉ (carried into the result for the display).
    scale
        ``"auto"`` returns absolute frames in mean solar brightness (MSB) when the density carries
        a physical unit, and relative frames otherwise. ``"relative"`` always returns relative
        frames.
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
    if occult not in ("eclipse", "none"):
        raise ValueError(f"occult must be 'eclipse' or 'none', not {occult!r}")
    table = build_coefficient_table(
        float(density.grid.radii[0]), float(density.grid.radii[-1]), u=u, crossover=crossover
    )
    if HAVE_NUMBA and density.grid._jit_grid() is not None:
        result = _brightness_numba(
            density,
            camera,
            table,
            step=step,
            occult=occult,
            r_occult=r_occult,
            occult_softness=occult_softness,
            u=u,
            workers=workers,
            show_progress=show_progress,
        )
    else:
        result = _brightness_numpy(
            density,
            camera,
            table,
            step=step,
            occult=occult,
            r_occult=r_occult,
            occult_softness=occult_softness,
            u=u,
            chunk_size=chunk_size,
            show_progress=show_progress,
        )
    if scale == "auto" and density.unit is not None:
        return _absolute(result, density, u)
    return result


def _absolute(result: BrightnessResult, density: DensityVolume, u: float) -> BrightnessResult:
    """Return ``result`` in units of the mean solar brightness.

    The frames are sums of ``Nₑ c ds``, with ``Nₑ`` in stored density values and ``ds`` in model
    length units. The density's unit and length unit turn them into an electron column (m⁻²). The
    single-electron prefactor (:func:`~qorona.radiation.thomson.msb_prefactor`) then turns the
    column into ``B☉``.
    """
    assert density.unit is not None  # checked by the caller
    column = float(density.unit.to(units.m**-3)) * float(density.length_unit.to(units.m))
    factor = msb_prefactor(u) * column
    return replace(
        result, polarized=result.polarized * factor, total=result.total * factor, unit="MSB"
    )
