"""Thomson-scattering geometry: the Minnaert/Billings coefficients and the optional Q⊥ LOS weight.

The K-corona scattered intensity per electron depends only on the heliocentric radius ``r`` (through
the solar angular radius ``θ_max = asin(R☉/r)``) and the mean scattering angle ``χ̄`` (pure
line-of-sight geometry). This module evaluates the four Minnaert/Billings coefficients
``A, B, C, D(θ_max)`` and combines them with a limb-darkening ``u`` into the tangential, polarized,
and total single-electron intensities: the only radiometric physics behind both white-light products
(the optional Q⊥ weighting here and the standalone brightness render in :mod:`.brightness`).

Two numerical details are load-bearing:

- **Closed form near the Sun, asymptotic series far out.** The exact coefficients are evaluated in
  closed form, but at large ``r`` the closed forms cancel toward their small ``O(θ_max²)`` values
  (``B``, ``C``, ``D`` individually, and the combinations ``C - A`` / ``D - B`` most severely), so
  beyond a crossover radius (default ``10 R☉``) all four switch to their small-``θ_max`` asymptotic
  expansion. Double precision throughout.
- **The weighting is a relative shape.** Because the render forms a weight-*normalised* average, any
  constant prefactor on the weight cancels, so the single-electron prefactor (and any absolute
  electron-density calibration) is dropped; only the ``r``- and ``χ̄``-dependent shape is kept.

The per-sample weight a render applies is the product ``Nₑ(point) · I(r, χ̄)`` with ``I = I_tot``
for the ``"K"`` (total-brightness emphasis) mode and ``I = I_pol`` for the ``"pB"`` (polarized,
peaking at the Thomson sphere ``χ̄ = π/2``) mode. The electron density comes from a
:class:`~qorona.field.density.DensityVolume`; everything else here is field-free geometry.

The coefficients are evaluated on a 1-D radial grid once and interpolated, since they depend on
``r`` alone, forming the table the render kernel consumes (:class:`RadialCoefficients`).

Implemented from Inhester (2015), "Thomson Scattering in the Solar Corona", arXiv:1512.00651
(Sec. 3.3 single-electron intensities; Appendix A.1 closed-form and A.4 asymptotic coefficients).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from qorona.field.density import DensityVolume

__all__ = [
    "ASYMPTOTIC_CROSSOVER",
    "LIMB_DARKENING",
    "RadialCoefficients",
    "ThomsonWeight",
    "build_coefficient_table",
    "intensity_coefficients",
    "minnaert_coefficients",
]

#: Scattering mode of the Thomson weight / brightness product.
ThomsonMode = Literal["K", "pB"]

#: Optical limb-darkening coefficient ``u`` in ``L(cos ζ) = L_c (1 - u + u cos ζ)``: the
#: default; a wavelength-specific effective value can be set per passband.
LIMB_DARKENING = 0.6

#: Heliocentric radius (R☉) beyond which the closed-form coefficients are replaced by their
#: small-``θ_max`` asymptotic expansion, avoiding cancellation in ``C - A`` / ``D - B``.
ASYMPTOTIC_CROSSOVER = 10.0

#: Number of log-spaced radial nodes in the precomputed coefficient table: dense enough that the
#: linear interpolation error is far below the engine tolerance across the coefficient curves.
_TABLE_SIZE = 4096

#: ``cos θ_max`` below which the closed-form limb-darkening term ``F`` is taken at its surface limit
#: ``0`` (at ``r = R☉`` exactly, ``cos θ_max = 0`` makes ``F = cos²·ln((1+sin)/cos)`` a ``0·∞``
#: form whose limit is ``0``). Guards only the table builder; the kernel reads the finished table.
_COS_FLOOR = 1.0e-12


def _closed_form(sin_theta: np.ndarray, cos_theta: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return the closed-form ``(A, B, C, D)`` from ``sin θ_max`` and ``cos θ_max``."""
    sin_sq = sin_theta * sin_theta
    with np.errstate(divide="ignore", invalid="ignore"):
        safe_cos = np.where(cos_theta > _COS_FLOOR, cos_theta, 1.0)
        limb = (cos_theta * cos_theta / sin_theta) * np.log((1.0 + sin_theta) / safe_cos)
    limb = np.where(cos_theta > _COS_FLOOR, limb, 0.0)  # F → 0 at the surface (θ_max → π/2)
    a = cos_theta * sin_sq
    c = 4.0 / 3.0 - cos_theta - cos_theta**3 / 3.0
    d = (5.0 + sin_sq - (5.0 - sin_sq) * limb) / 8.0
    b = -(1.0 - 3.0 * sin_sq - (1.0 + 3.0 * sin_sq) * limb) / 8.0
    return a, b, c, d


def _asymptotic(theta_max: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return the small-``θ_max`` asymptotic ``(A, B, C, D)``, for large ``r``."""
    t2 = theta_max * theta_max
    t4 = t2 * t2
    a = t2 - (5.0 / 6.0) * t4
    b = (2.0 / 3.0) * t2 - (22.0 / 45.0) * t4
    c = t2 - t4 / 3.0
    d = (2.0 / 3.0) * t2 - (2.0 / 9.0) * t4
    return a, b, c, d


def minnaert_coefficients(
    radius: np.ndarray, *, crossover: float = ASYMPTOTIC_CROSSOVER
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the Minnaert/Billings ``(A, B, C, D)`` at heliocentric ``radius`` (R☉).

    Closed form for ``radius < crossover``; the asymptotic series beyond it (avoiding the
    ``C - A`` / ``D - B`` cancellation). ``θ_max = asin(R☉/r)`` with ``R☉ = 1`` in these units, so
    ``sin θ_max = 1/r`` (clamped at the surface). The coefficients depend on ``r`` only.

    Parameters
    ----------
    radius
        Heliocentric radius in R☉ (``≥ R☉``; values inside the Sun are clamped to the surface).
    crossover
        Radius (R☉) above which the asymptotic expansion is used.

    Returns
    -------
    tuple of numpy.ndarray
        ``(A, B, C, D)``, each the shape of ``radius``.
    """
    radius = np.asarray(radius, dtype=np.float64)
    sin_theta = np.clip(1.0 / radius, 0.0, 1.0)
    cos_theta = np.sqrt(np.clip(1.0 - sin_theta * sin_theta, 0.0, None))
    theta_max = np.arcsin(sin_theta)

    closed = _closed_form(sin_theta, cos_theta)
    far = _asymptotic(theta_max)
    use_far = radius >= crossover
    return tuple(np.where(use_far, far[k], closed[k]) for k in range(4))  # type: ignore[return-value]


def intensity_coefficients(
    radius: np.ndarray, *, u: float = LIMB_DARKENING, crossover: float = ASYMPTOTIC_CROSSOVER
) -> tuple[np.ndarray, np.ndarray]:
    """Return the radius-only intensity coefficients ``(c_tan, c_pol)`` (prefactor dropped).

    With ``A, B, C, D`` the Minnaert coefficients and ``u`` the limb darkening,
    ``c_tan = (1 - u)C + uD`` and ``c_pol = (1 - u)A + uB``. The single-electron intensities a
    render then uses are ``I_tan = c_tan``, ``I_pol = c_pol · sin²χ̄``, and
    ``I_tot = 2 I_tan - I_pol``, so ``c_tan`` and ``c_pol`` are everything that depends on ``r``
    (``sin²χ̄`` is pure ray geometry, applied at the sample).
    """
    a, b, c, d = minnaert_coefficients(radius, crossover=crossover)
    c_tan = (1.0 - u) * c + u * d
    c_pol = (1.0 - u) * a + u * b
    return c_tan, c_pol


@dataclass(frozen=True)
class RadialCoefficients:
    """The intensity coefficients ``c_tan(r)``, ``c_pol(r)`` tabulated on a log-spaced radial grid.

    Both the NumPy render path and the numba kernel read a sample's coefficients by linearly
    interpolating this table in ``ln r`` (a search-free index, since the nodes are uniform in
    ``ln r``), so both paths interpolate identically and stay parity-exact. The table depends only
    on the shell radii, the limb darkening ``u``, and the asymptotic crossover.

    Attributes
    ----------
    log_inner
        ``ln r`` at the inner table node.
    inv_dlog
        ``(size - 1) / (ln r_outer - ln r_inner)``: maps ``ln r`` to a fractional node index.
    c_tan, c_pol
        ``(size,)`` tabulated ``c_tan`` and ``c_pol`` at the node radii.
    """

    log_inner: float
    inv_dlog: float
    c_tan: np.ndarray
    c_pol: np.ndarray

    def evaluate(self, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(c_tan, c_pol)`` at ``radius`` by linear interpolation in ``ln r`` (NumPy path).

        Mirrors the kernel's inline interpolation exactly: clamp the fractional node index into
        ``[0, size - 1]`` and blend the bracketing nodes.
        """
        size = self.c_tan.shape[0]
        position = np.clip((np.log(radius) - self.log_inner) * self.inv_dlog, 0.0, size - 1.0)
        lower = np.floor(position).astype(np.intp)
        upper = np.minimum(lower + 1, size - 1)
        frac = position - lower
        c_tan = self.c_tan[lower] * (1.0 - frac) + self.c_tan[upper] * frac
        c_pol = self.c_pol[lower] * (1.0 - frac) + self.c_pol[upper] * frac
        return c_tan, c_pol


def build_coefficient_table(
    inner_radius: float,
    outer_radius: float,
    *,
    u: float = LIMB_DARKENING,
    crossover: float = ASYMPTOTIC_CROSSOVER,
    size: int = _TABLE_SIZE,
) -> RadialCoefficients:
    """Tabulate ``c_tan(r)``, ``c_pol(r)`` on ``size`` log-spaced nodes over the shell radii."""
    log_inner = float(np.log(inner_radius))
    log_outer = float(np.log(outer_radius))
    radius = np.exp(np.linspace(log_inner, log_outer, size))
    c_tan, c_pol = intensity_coefficients(radius, u=u, crossover=crossover)
    inv_dlog = (size - 1) / (log_outer - log_inner)
    return RadialCoefficients(
        log_inner=log_inner,
        inv_dlog=inv_dlog,
        c_tan=np.ascontiguousarray(c_tan),
        c_pol=np.ascontiguousarray(c_pol),
    )


@dataclass(frozen=True)
class ThomsonWeight:
    """The optional scalar radiometric LOS weight ``Nₑ(point) · I(r, χ̄)`` for the Q⊥ render.

    A composable, off-by-default factor on an axis orthogonal to the render's geometric depth
    weighting: passed to :func:`~qorona.render.los.render` as ``thomson=...``, it multiplies the
    per-sample value into the weighted-average numerator/denominator only (biasing the rendered Q⊥
    toward bright, dense low-corona plasma) while the depth-colour geometry and coverage stay
    unchanged. It bundles the electron-density volume with the scattering ``mode`` and the limb
    darkening ``u``; the radial coefficient table it hands the render is built from the density
    grid's shell radii.

    Attributes
    ----------
    density
        The electron-density volume sampled along each line of sight.
    mode
        ``"K"`` (total-brightness emphasis, ``I_tot``) or ``"pB"`` (polarized emphasis, ``I_pol``,
        peaking at the Thomson sphere).
    u
        Limb-darkening coefficient (default :data:`LIMB_DARKENING`).
    crossover
        Closed-form → asymptotic radius in R☉ (default :data:`ASYMPTOTIC_CROSSOVER`).
    """

    density: DensityVolume
    mode: ThomsonMode = "K"
    u: float = LIMB_DARKENING
    crossover: float = ASYMPTOTIC_CROSSOVER

    def __post_init__(self) -> None:
        if self.mode not in ("K", "pB"):
            raise ValueError(f"mode must be 'K' or 'pB', not {self.mode!r}")

    def coefficient_table(self) -> RadialCoefficients:
        """Return the coefficient table over the density grid's shell, for ``u``/crossover."""
        radii = self.density.grid.radii
        return build_coefficient_table(
            float(radii[0]), float(radii[-1]), u=self.u, crossover=self.crossover
        )
