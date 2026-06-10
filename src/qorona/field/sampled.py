"""``SampledField``: a :class:`~qorona.field.base.Field` backed by resampled MHD data.

A native solution is resampled onto the internal regular spherical grid (Cartesian B
components), padded once with ghost layers, and queried by tricubic interpolation. The
construction step is the unit edge: native coordinates (astropy quantities in R☉) are
validated and stripped to float64, and the native normalization is recorded; everything
thereafter is unit-free.

The gradient returned by :meth:`sample` is the Cartesian Jacobian ``∂B_i/∂x_j``. It is formed
by the chain rule from the interpolant's index-space gradient: the tricubic gives
``∂B_i/∂(grid index)``, which is scaled by the grid's ``d(index)/d(spherical)`` and rotated by
the spherical-coordinate Jacobian ``∂(spherical)/∂x``, so the gradient is the exact analytic
derivative of the same interpolated field, with no separate finite-difference field. The
``∂φ/∂x`` factor diverges on the polar axis (the artificial coordinate singularity); it is
applied exactly, with no floor, because the true Cartesian ∇B is finite there and any floor
would compute the wrong chain-rule factor and bias the polar fine structure toward zero.
"""

from __future__ import annotations

import numpy as np

from qorona.accel import JitField
from qorona.field.base import Domain, Field, FieldSample
from qorona.field.density import DensityVolume
from qorona.field.interpolation import tricubic
from qorona.geometry import spherical_coordinate_jacobian
from qorona.io.native import NativeSolution
from qorona.resample.grid import _SPACING_CODES, GHOST, SphericalGrid, pad_field
from qorona.resample.resampler import KnnMlsResampler, Resampler

#: Default native variable names for the Cartesian magnetic-field components.
DEFAULT_B_COMPONENTS: tuple[str, str, str] = ("Bx", "By", "Bz")

#: Native variable name of the (mass) density carried into the Thomson / brightness branch.
DEFAULT_DENSITY_COMPONENT = "rho"


class SampledField(Field):
    """A magnetic field interpolated from MHD data on the internal spherical grid.

    Construct with :meth:`from_solution`, which resamples a :class:`NativeSolution` onto
    ``grid`` and pads it for interpolation.

    Attributes
    ----------
    grid
        The internal spherical grid the field is sampled on.
    normalization
        The native field normalization recorded from the source solution (e.g. ``"corona"``).
    """

    def __init__(
        self,
        grid: SphericalGrid,
        b_padded: np.ndarray,
        normalization: str,
        *,
        density: DensityVolume | None = None,
    ) -> None:
        self.grid = grid
        self.normalization = normalization
        self._b_padded = b_padded
        self._density = density
        self._reference_strength: float | None = None
        self._domain = Domain(
            inner_radius=float(grid.radii[0]),
            outer_radius=float(grid.radii[-1]),
            frame="solution",
        )

    @classmethod
    def from_solution(
        cls,
        solution: NativeSolution,
        grid: SphericalGrid,
        *,
        resampler: Resampler | None = None,
        b_components: tuple[str, str, str] = DEFAULT_B_COMPONENTS,
        density_component: str | None = DEFAULT_DENSITY_COMPONENT,
        show_progress: bool = True,
    ) -> SampledField:
        """Build a :class:`SampledField` by resampling a solution's B (and density) onto ``grid``.

        B and the (mass) density are resampled together in one pass (they share the node geometry,
        so the k-d tree and the moving-least-squares design are reused), and the density is carried
        as an optional :class:`~qorona.field.density.DensityVolume` for the Thomson / brightness
        branch; the magnetic core works the same whether or not it is present.

        Parameters
        ----------
        solution
            The native solution to resample from.
        grid
            The internal spherical grid to sample onto (its radii fix the domain).
        resampler
            Strategy mapping native cells to grid nodes; defaults to
            :class:`~qorona.resample.resampler.KnnMlsResampler` (smooth, low spurious ``∇·B``).
        b_components
            Native variable names of the Cartesian B components, in ``(x, y, z)`` order.
        density_component
            Native variable name of the (mass) density, or ``None`` to skip it. Skipped silently if
            the solution does not carry it, so the field still builds without a density volume.
        show_progress
            Whether to display resampling progress.

        Returns
        -------
        SampledField
            The interpolatable field, with the native normalization (and density, when available)
            recorded.
        """
        resampler = resampler if resampler is not None else KnnMlsResampler()
        density_name = (
            density_component
            if density_component is not None and density_component in solution.variables
            else None
        )
        requested = (*b_components, density_name) if density_name is not None else b_components
        components = resampler.resample(solution, grid, requested, show_progress=show_progress)
        b_padded = pad_field(np.stack([components[name] for name in b_components], axis=-1))
        density = (
            DensityVolume.from_grid_values(grid, components[density_name])
            if density_name is not None
            else None
        )
        return cls(grid, b_padded, solution.metadata.normalization, density=density)

    @property
    def domain(self) -> Domain:
        return self._domain

    @property
    def density(self) -> DensityVolume | None:
        """The electron-density volume resampled alongside B, or ``None`` if not carried.

        The optional plasma capability the Thomson / brightness branch consumes; absent on analytic
        fields and on a field built with ``density_component=None``.
        """
        return self._density

    def reference_strength(self) -> float:
        """Return the grid-max ``|B|``: the strong-field scale the sharp-turn guard's weak-field
        test is relative to (see :meth:`~qorona.field.base.Field.reference_strength`). Computed once
        from the padded grid and cached."""
        if self._reference_strength is None:
            magnitude = np.sqrt(np.sum(self._b_padded * self._b_padded, axis=-1))
            self._reference_strength = float(np.max(magnitude))
        return self._reference_strength

    def characteristic_length(self, points: np.ndarray) -> np.ndarray:
        """Return the smallest local grid-cell extent ``(n,)`` at ``points`` (the CFL metric).

        Delegates to :meth:`~qorona.resample.grid.SphericalGrid.cell_extent` (see
        :meth:`Field.characteristic_length` for why the ceiling must follow the local cell).
        """
        return self.grid.cell_extent(np.asarray(points, dtype=np.float64))

    def sample(
        self, points: np.ndarray, *, gradient: bool = True, validate: bool = False
    ) -> FieldSample:
        # Points are assumed in-domain (the Field precondition in base.py). Only the
        # radial axis can leave the domain: θ and φ are periodic / pole-reflected and always valid
        # through the padding. Explicit-RK stages overrun the shell by up to ~1 cell (past the two
        # radial ghost layers), so the radial coord is clamped into the Keys in-range band
        # (floor ∈ [1, N_r-3]); an out-of-shell probe then reads finite edge-extrapolation rather
        # than overrunning the array. In-domain radial coords already lie in the band (a boundary
        # node maps to its band edge), so the clamp is a no-op for them. validate=True instead
        # raises OutOfDomainError, for development.
        points = np.asarray(points, dtype=np.float64)
        if validate:
            self._domain.require_interior(points)
        index, index_derivative = self.grid.index_coordinates(points)
        coords = index + GHOST
        coords[:, 0] = np.clip(coords[:, 0], 1.0, self._b_padded.shape[0] - 3)
        value, index_gradient = tricubic(self._b_padded, coords, gradient=gradient)

        grad_b: np.ndarray | None = None
        if index_gradient is not None:
            # Chain-rule the index-space gradient to a Cartesian Jacobian ∂B_i/∂x_j:
            #   index_gradient[n, d, i] = ∂B_i/∂index_d
            #   index_derivative[n, d]  = ∂index_d/∂spherical_d   (diagonal index map)
            #   coordinate_jacobian[n, d, j] = ∂spherical_d/∂x_j
            # The ∂φ/∂x factor carries 1/(r sinθ): exactly on the polar axis this conversion
            # is a genuine coordinate singularity (the true Cartesian ∇B is finite, but this
            # path is not), so a point landing precisely on the axis yields a non-finite
            # Jacobian by design: no floor, since a floor would compute the wrong factor.
            coordinate_jacobian = spherical_coordinate_jacobian(points)
            scaled = index_gradient * index_derivative[:, :, None]
            grad_b = np.einsum("ndi,ndj->nij", scaled, coordinate_jacobian, optimize=True)

        b_magnitude = np.sqrt(np.sum(value * value, axis=-1))
        return FieldSample(b=value, b_magnitude=b_magnitude, grad_b=grad_b)

    def _jit_field(self) -> JitField | None:
        """Return the raw payload for the numba transport kernel, or ``None`` if not accelerable.

        Opts this field into the scalar-per-lane ``prange`` kernel (``qorona.accel``): it hands over
        the padded B array and the grid scalars the kernel inlines, since the grid/spacing
        dataclasses do not survive nopython. A grid whose radial spacing is not one of the three
        kernel-supported laws (:data:`_SPACING_CODES`) returns ``None``, so the tracer falls back to
        the NumPy integrator.
        """
        spacing_code = _SPACING_CODES.get(type(self.grid.spacing))
        if spacing_code is None:
            return None
        exponent = getattr(self.grid.spacing, "exponent", 1.0)
        return JitField(
            kind=0,
            b_padded=self._b_padded,
            n_r=self.grid.n_r,
            n_theta=self.grid.n_theta,
            n_phi=self.grid.n_phi,
            spacing_code=spacing_code,
            r_inner=self._domain.inner_radius,
            r_outer=self._domain.outer_radius,
            exponent=float(exponent),
            moment=0.0,
            background=0.0,
            char_const=0.0,
        )
