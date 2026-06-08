"""The internal regular spherical grid and its radial spacing laws.

Qorona resamples every native solution onto one regular spherical ``(r, θ, φ)`` grid storing
Cartesian B components, so that all downstream interpolation, tracing, and squashing-factor
work is independent of the native model and mesh.

Grid conventions:

- **r** is node-centred (nodes at the inner and outer boundaries) and non-uniform via a
  templated :class:`RadialSpacing` law: a strategy mapping a uniform parameter ``u ∈ [0, 1]``
  to physical radius, with inverse and derivative, so the field is interpolated in the uniform
  ``u`` coordinate and the index-space gradient is chain-ruled back to physical radius. The
  default is logarithmic (constant ``dr/r``), matching coronal scaling and the native mesh's
  own coarsening.
- **θ** (colatitude) is uniform and **cell-centred**: no node lies on a pole, which bounds the
  ``1/sinθ`` amplification when the gradient is later rotated to Cartesian.
- **φ** (azimuth) is uniform and periodic.

The tricubic (Keys) interpolant is edge-agnostic and consumes a padded array; the ghost-layer
helpers here realize the boundary conventions (``φ`` wrap, ``θ`` reflect-through-pole, ``r``
quadratic extrapolation).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from qorona.accel import JitGrid
from qorona.geometry import cartesian_to_spherical, spherical_to_cartesian

#: Ghost layers required per side by the Keys 4-point stencil (offsets -1..+2).
GHOST = 2


class RadialSpacing(ABC):
    """A radial node distribution as an invertible map ``u ∈ [0, 1] → r``.

    Subclasses define the placement of the radial nodes between the inner and outer
    boundaries. The uniform parameter ``u`` is what the interpolant works in, so each law
    provides the radius, its inverse, and ``dr/du`` (for chain-ruling the radial gradient).
    :meth:`parameter` must be the exact inverse of :meth:`radius`: node ``i`` sits at
    ``u = i/(n_r-1)``, and the index lookup relies on ``parameter(radius(u)) == u``. Concrete
    laws are dataclasses carrying the ``inner`` and ``outer`` boundary radii (R☉).
    """

    @abstractmethod
    def radius(self, parameter: np.ndarray) -> np.ndarray:
        """Return the radius at uniform parameter ``u ∈ [0, 1]``."""

    @abstractmethod
    def parameter(self, radius: np.ndarray) -> np.ndarray:
        """Return the uniform parameter ``u`` at a given radius (the inverse of :meth:`radius`)."""

    @abstractmethod
    def radius_derivative(self, parameter: np.ndarray) -> np.ndarray:
        """Return ``dr/du`` at uniform parameter ``u``."""


@dataclass(frozen=True)
class LogarithmicSpacing(RadialSpacing):
    """Logarithmic radial spacing: nodes equally spaced in ``ln r`` (constant ``dr/r``).

    The default law: it resolves the low corona finely and coarsens outward in step with the
    physics and the native mesh, and its ``dr/du = r·ln(outer/inner)`` never vanishes, so the
    radial chain-rule factor stays finite everywhere, including at the inner seeding surface.
    """

    inner: float
    outer: float

    def radius(self, parameter: np.ndarray) -> np.ndarray:
        return self.inner * (self.outer / self.inner) ** parameter

    def parameter(self, radius: np.ndarray) -> np.ndarray:
        return np.log(radius / self.inner) / np.log(self.outer / self.inner)

    def radius_derivative(self, parameter: np.ndarray) -> np.ndarray:
        return self.radius(parameter) * np.log(self.outer / self.inner)


@dataclass(frozen=True)
class PowerLawSpacing(RadialSpacing):
    """Power-law radial spacing: nodes equally spaced in ``r ** (1/exponent)``.

    Writing ``s = r ** (1/exponent)``, the nodes are uniform in ``s`` and ``r = s ** exponent``.
    A tunable middle ground between uniform (``exponent = 1``) and the stronger inner
    clustering of the logarithmic law: larger ``exponent`` concentrates more nodes near the
    inner boundary.
    """

    inner: float
    outer: float
    exponent: float = 2.0

    @property
    def _inner_root(self) -> float:
        return self.inner ** (1.0 / self.exponent)

    @property
    def _outer_root(self) -> float:
        return self.outer ** (1.0 / self.exponent)

    def radius(self, parameter: np.ndarray) -> np.ndarray:
        root = self._inner_root + (self._outer_root - self._inner_root) * parameter
        return root**self.exponent

    def parameter(self, radius: np.ndarray) -> np.ndarray:
        root = radius ** (1.0 / self.exponent)
        return (root - self._inner_root) / (self._outer_root - self._inner_root)

    def radius_derivative(self, parameter: np.ndarray) -> np.ndarray:
        root = self._inner_root + (self._outer_root - self._inner_root) * parameter
        return self.exponent * root ** (self.exponent - 1.0) * (self._outer_root - self._inner_root)


@dataclass(frozen=True)
class UniformSpacing(RadialSpacing):
    """Uniform radial spacing: nodes equally spaced in ``r``."""

    inner: float
    outer: float

    def radius(self, parameter: np.ndarray) -> np.ndarray:
        return self.inner + (self.outer - self.inner) * parameter

    def parameter(self, radius: np.ndarray) -> np.ndarray:
        return (radius - self.inner) / (self.outer - self.inner)

    def radius_derivative(self, parameter: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(parameter, dtype=np.float64), self.outer - self.inner)


#: Radial-spacing law → kernel ``spacing_code`` (the numba index map's branch selector, read by
#: :meth:`SphericalGrid._jit_grid` and ``SampledField._jit_field``). A grid on an unlisted law
#: cannot be JIT-accelerated and falls back to the NumPy path.
_SPACING_CODES: dict[type, int] = {
    LogarithmicSpacing: 0,
    PowerLawSpacing: 1,
    UniformSpacing: 2,
}


@dataclass(frozen=True)
class SphericalGrid:
    """A regular spherical ``(r, θ, φ)`` grid for resampling and interpolation.

    Attributes
    ----------
    spacing
        The radial spacing law (carries the inner/outer radii).
    n_r
        Number of nodes along the radial axis.
    n_theta
        Number of nodes along the colatitude axis.
    n_phi
        Number of nodes along the azimuth axis (must be even for pole reflection).
    """

    spacing: RadialSpacing
    n_r: int
    n_theta: int
    n_phi: int

    def __post_init__(self) -> None:
        # n_phi must be even so the pole reflection (a φ → φ+π shift of n_phi/2 columns) lands
        # exactly on grid columns; the tricubic stencil plus ghost layers needs these minima.
        if self.n_phi % 2 != 0:
            raise ValueError(f"n_phi must be even for pole reflection, got {self.n_phi}")
        if self.n_r < 4 or self.n_theta < 2 * GHOST or self.n_phi < 2 * GHOST:
            raise ValueError(
                f"grid too small: need n_r >= 4, n_theta >= {2 * GHOST}, n_phi >= {2 * GHOST}; "
                f"got n_r={self.n_r}, n_theta={self.n_theta}, n_phi={self.n_phi}"
            )

    @property
    def radii(self) -> np.ndarray:
        """``(n_r,)`` node radii (node-centred; endpoints at the inner/outer boundary)."""
        return self.spacing.radius(np.linspace(0.0, 1.0, self.n_r))

    @property
    def colatitudes(self) -> np.ndarray:
        """``(n_theta,)`` node colatitudes (cell-centred in ``(0, π)``; none on a pole)."""
        return (np.arange(self.n_theta) + 0.5) * (np.pi / self.n_theta)

    @property
    def azimuths(self) -> np.ndarray:
        """``(n_phi,)`` node azimuths (uniform in ``[0, 2π)``; periodic)."""
        return np.arange(self.n_phi) * (2.0 * np.pi / self.n_phi)

    def node_points(self) -> np.ndarray:
        """Return the ``(n_r, n_theta, n_phi, 3)`` Cartesian coordinates of every node."""
        r, theta, phi = np.meshgrid(self.radii, self.colatitudes, self.azimuths, indexing="ij")
        return spherical_to_cartesian(np.stack([r, theta, phi], axis=-1))

    def cell_extent(self, points: np.ndarray) -> np.ndarray:
        """Return the smallest local cell extent ``(n,)`` at ``points`` (the CFL step metric).

        The node spacing in physical length along each axis at a point ``(r, θ, φ)`` is the
        radial spacing ``Δr`` (from the spacing law), the meridional arc ``r·Δθ``, and the
        azimuthal arc ``r·sinθ·Δφ``; the cell extent is their minimum. The tracer turns this
        into its step ceiling ``h_max = cfl · cell_extent`` so a step never skips a cell on the
        radially stretched grid. Essentially free: a spacing-law evaluation on the spherical
        coordinates :meth:`index_coordinates` already computes.

        The azimuthal arc ``r·sinθ·Δφ`` collapses to zero at the poles purely from meridian
        convergence (a coordinate artifact, not field structure): B is stored Cartesian and is C¹
        *through* the pole by the reflect-through-pole padding (and the θ grid is cell-centred to
        bound this same ``1/sinθ`` amplification), so it carries no independent azimuthal variation
        at that vanishing scale. The azimuthal arc is therefore bounded below by the meridional arc,
        so a near-pole line steps at the genuinely-resolved radial/meridional scale instead of
        stalling against ``max_steps`` on a sub-cell the field never resolves. This only relaxes the
        ceiling (the adaptive error control remains the accuracy guarantee).

        Parameters
        ----------
        points
            ``(n, 3)`` Cartesian coordinates in R☉, assumed inside the grid's radial range.

        Returns
        -------
        numpy.ndarray
            ``(n,)`` smallest local cell extent in R☉.
        """
        spherical = cartesian_to_spherical(points)
        radius = spherical[..., 0]
        colatitude = spherical[..., 1]
        radial_extent = self.spacing.radius_derivative(self.spacing.parameter(radius)) / (
            self.n_r - 1
        )
        meridional_extent = radius * (np.pi / self.n_theta)
        azimuthal_extent = np.maximum(
            radius * np.sin(colatitude) * (2.0 * np.pi / self.n_phi), meridional_extent
        )
        return np.minimum(radial_extent, np.minimum(meridional_extent, azimuthal_extent))

    def index_coordinates(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map Cartesian points to fractional grid-index coordinates (unpadded).

        The index map is diagonal (each index depends only on its own spherical coordinate),
        so the per-axis derivatives ``d(index)/d(spherical)`` are returned as three columns
        for the caller's gradient chain rule.

        Parameters
        ----------
        points
            ``(n, 3)`` Cartesian coordinates in R☉.

        Returns
        -------
        index : numpy.ndarray
            ``(n, 3)`` fractional indices ``(r_index, θ_index, φ_index)`` into the unpadded
            grid (add :data:`GHOST` to address the padded array).
        index_derivative : numpy.ndarray
            ``(n, 3)`` diagonal derivatives ``(dr_index/dr, dθ_index/dθ, dφ_index/dφ)``.
        """
        spherical = cartesian_to_spherical(points)
        radius = spherical[..., 0]
        colatitude = spherical[..., 1]
        azimuth = spherical[..., 2]

        parameter = self.spacing.parameter(radius)
        r_index = parameter * (self.n_r - 1)
        theta_step = np.pi / self.n_theta
        phi_step = 2.0 * np.pi / self.n_phi
        theta_index = colatitude / theta_step - 0.5
        phi_index = azimuth / phi_step

        index = np.stack([r_index, theta_index, phi_index], axis=-1)
        dr_index_dr = (self.n_r - 1) / self.spacing.radius_derivative(parameter)
        index_derivative = np.stack(
            [
                dr_index_dr,
                np.full_like(theta_index, 1.0 / theta_step),
                np.full_like(phi_index, 1.0 / phi_step),
            ],
            axis=-1,
        )
        return index, index_derivative

    def _jit_grid(self) -> JitGrid | None:
        """Return the raw geometry payload for the numba paint kernel, ``None`` if not accelerable.

        Hands the kernel the forward index map (node counts, radial spacing code and endpoints, and
        the power-law exponent) it bins swept-path points into, since the grid/spacing dataclasses
        do not survive nopython. A grid whose radial spacing is not one of the three supported laws
        (:data:`_SPACING_CODES`) returns ``None``, so the painter falls back to the NumPy path. It
        reads :data:`_SPACING_CODES`, the same dict ``SampledField._jit_field`` reads independently
        (neither delegates to the other).
        """
        spacing_code = _SPACING_CODES.get(type(self.spacing))
        if spacing_code is None:
            return None
        return JitGrid(
            n_r=self.n_r,
            n_theta=self.n_theta,
            n_phi=self.n_phi,
            spacing_code=spacing_code,
            r_inner=float(self.radii[0]),
            r_outer=float(self.radii[-1]),
            exponent=float(getattr(self.spacing, "exponent", 1.0)),
        )


def pad_phi(values: np.ndarray) -> np.ndarray:
    """Pad the φ axis (axis 2) periodically by :data:`GHOST` layers each side."""
    return np.concatenate([values[:, :, -GHOST:, :], values, values[:, :, :GHOST, :]], axis=2)


def pad_theta(values: np.ndarray) -> np.ndarray:
    """Pad the θ axis (axis 1) by reflection through the poles.

    A ghost row just past a pole is the real data one step inside the pole at azimuth φ + π
    (a shift of ``n_phi/2`` columns), exact for the smoothly-through-pole Cartesian B. Must be
    applied while φ still holds its real node count (before :func:`pad_phi`).
    """
    shift = values.shape[2] // 2
    top = np.roll(values[:, GHOST - 1 :: -1, :, :], shift, axis=2)
    bottom = np.roll(values[:, -1 : -GHOST - 1 : -1, :, :], shift, axis=2)
    return np.concatenate([top, values, bottom], axis=1)


def pad_radial(values: np.ndarray) -> np.ndarray:
    """Pad the r axis (axis 0) by quadratic extrapolation of :data:`GHOST` layers each side.

    The ghosts continue the unique quadratic through the three nearest node layers, evaluated
    in the uniform parameter space, so the Keys stencil reproduces a one-sided quadratic in the
    boundary cell and stays C¹ at the first interior seam. Applied on the signed Cartesian B
    components.
    """
    low = np.stack(
        [
            6.0 * values[0] - 8.0 * values[1] + 3.0 * values[2],
            3.0 * values[0] - 3.0 * values[1] + values[2],
        ]
    )
    high = np.stack(
        [
            values[-3] - 3.0 * values[-2] + 3.0 * values[-1],
            3.0 * values[-3] - 8.0 * values[-2] + 6.0 * values[-1],
        ]
    )
    return np.concatenate([low, values, high], axis=0)


def pad_field(values: np.ndarray) -> np.ndarray:
    """Apply the full ghost padding (θ reflect, then r extrapolate, then φ wrap).

    The order matters: θ reflection reads the real φ count, so it precedes the φ wrap; radial
    extrapolation runs over the θ ghosts so corner ghosts are filled consistently; the φ wrap
    runs last over all radial and θ ghosts.

    Parameters
    ----------
    values
        ``(n_r, n_theta, n_phi, C)`` field samples on the grid nodes.

    Returns
    -------
    numpy.ndarray
        ``(n_r + 2·GHOST, n_theta + 2·GHOST, n_phi + 2·GHOST, C)`` padded array, ready for the
        edge-agnostic tricubic with grid indices offset by :data:`GHOST`.
    """
    return pad_phi(pad_radial(pad_theta(values)))
