"""Closed-form analytic fields, used to validate the tracer and Q⊥ engine.

An :class:`AnalyticField` evaluates B and its Jacobian from a closed-form expression, with
no mesh, interpolation, or numerical gradient in the way, so a test built on one measures the
field-line integrator and the squashing-factor computation in isolation. Subclasses supply B
and ∇B directly in **Cartesian** components: the planned validation fields (the PFSS dipole
here; the linear null and X-line later) are all naturally Cartesian, which avoids the
spherical→Cartesian vector Jacobian and its singular behaviour on the polar axis, exactly
where the dipole's Q⊥ profile is checked.
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from qorona.accel import JitField
from qorona.field.base import Domain, Field, FieldSample

_Z_AXIS = np.array([0.0, 0.0, 1.0])
_IDENTITY = np.eye(3)

#: Unused gridded payload for the dipole's :class:`~qorona.accel.JitField`: the kernel's dipole
#: branch never reads ``b_padded``, but the field must carry a consistently-typed (C-contiguous
#: float64) placeholder so one compiled kernel signature serves both field kinds.
_DIPOLE_PLACEHOLDER = np.zeros((1, 1, 1, 3))


class AnalyticField(Field):
    """Base for fields given by a closed-form Cartesian expression.

    Subclasses supply the closed form by implementing :meth:`_evaluate` (B and, optionally,
    its Jacobian); this base computes ``|B|`` and assembles the :class:`FieldSample`. Per the
    :class:`Field` precondition, ``sample`` assumes in-domain points, so the closed forms are
    only ever evaluated at ``r ≥ inner_radius > 0`` and never see a singular radius.
    """

    #: CFL step ceiling as a fraction of the shell width (no grid; a loose ceiling the embedded
    #: controller resolves well below on the smooth analytic fields).
    _CHARACTERISTIC_FRACTION = 0.02

    def __init__(self, domain: Domain) -> None:
        """Initialise the analytic field.

        Parameters
        ----------
        domain
            The spherical shell the field is defined on.
        """
        self._domain = domain

    @property
    def domain(self) -> Domain:
        return self._domain

    def characteristic_length(self, points: np.ndarray) -> np.ndarray:
        """Return a fixed fraction of the shell width for every point ``(n,)``.

        A grid-free analytic field has no cells, so the CFL ceiling is a constant fraction of
        the domain width, loose by design (the embedded step controller resolves the smooth
        field far below it), and uniform because the closed-form fields carry no preferred
        local scale.
        """
        points = np.asarray(points, dtype=np.float64)
        width = self._domain.outer_radius - self._domain.inner_radius
        return np.full(points.shape[:-1], self._CHARACTERISTIC_FRACTION * width)

    @abstractmethod
    def _evaluate(
        self, points: np.ndarray, *, gradient: bool
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return the closed-form Cartesian ``(B, grad_b)`` at ``points``.

        ``grad_b`` is ``None`` when ``gradient`` is ``False``. ``points`` are assumed inside
        the domain (the :class:`Field` precondition; the dipole domain excludes ``r = 0``).
        """

    def sample(
        self, points: np.ndarray, *, gradient: bool = True, validate: bool = False
    ) -> FieldSample:
        points = np.asarray(points, dtype=np.float64)
        if validate:
            self._domain.require_interior(points)
        b, grad_b = self._evaluate(points, gradient=gradient)
        b_magnitude = np.sqrt(np.sum(b * b, axis=-1))
        return FieldSample(b=b, b_magnitude=b_magnitude, grad_b=grad_b)


class PfssDipoleField(AnalyticField):
    """Axisymmetric potential-field source-surface (PFSS) dipole, the engine accuracy test.

    A z-aligned point dipole plus the uniform field that enforces the source-surface
    condition ``B_θ(R_S, θ) = 0``, normalized so ``B_r(R_⊙, θ) = strength · cosθ``. In
    Cartesian components (``r = |x|``, ``m = strength·R_⊙³R_S³/N``, ``B₀ = strength·R_⊙³/N``,
    ``N = R_⊙³ + 2R_S³``):

        B = m (3 z x / r⁵ - ẑ / r³) + B₀ ẑ

    which is regular on the polar axis. The domain is capped at the source surface ``R_S`` so
    the C¹ kink there never falls inside an integration cell (the purely radial field above
    ``R_S`` is outside the domain). ``B_φ = 0``.

    The flux-function and separatrix diagnostics are exposed for the squashing-factor
    validation harness (they are specific to this field, not part of the generic
    :class:`Field` interface).
    """

    def __init__(self, *, r_sun: float = 1.0, r_source: float = 2.5, strength: float = 1.0) -> None:
        """Initialise the dipole.

        Parameters
        ----------
        r_sun
            Inner boundary radius ``R_⊙`` in R☉ (the photosphere; the seeding surface).
        r_source
            Source-surface radius ``R_S`` in R☉ (the outer boundary and open-line target).
        strength
            Overall field scale; sets ``B_r(R_⊙, θ) = strength · cosθ``. Q⊥ is
            scale-invariant, so this does not affect the squashing-factor validation.
        """
        if not 0.0 < r_sun < r_source:
            raise ValueError(
                f"require 0 < r_sun < r_source, got r_sun={r_sun}, r_source={r_source}"
            )
        super().__init__(Domain(inner_radius=r_sun, outer_radius=r_source, frame="dipole_aligned"))
        self.r_sun = r_sun
        self.r_source = r_source
        self.strength = strength

        normalization = r_sun**3 + 2.0 * r_source**3
        self._dipole_moment = strength * r_sun**3 * r_source**3 / normalization
        self._background_field = strength * r_sun**3 / normalization

    def _evaluate(
        self, points: np.ndarray, *, gradient: bool
    ) -> tuple[np.ndarray, np.ndarray | None]:
        moment = self._dipole_moment
        z = points[..., 2]
        r2 = np.sum(points * points, axis=-1)
        r3_inv = r2**-1.5
        r5_inv = r2**-2.5
        b = (
            moment * ((3.0 * z * r5_inv)[..., None] * points - r3_inv[..., None] * _Z_AXIS)
            + self._background_field * _Z_AXIS
        )

        grad_b = None
        if gradient:
            r7_inv = r2**-3.5
            x_z = points[..., :, None] * _Z_AXIS  # x_i ẑ_j
            z_x = _Z_AXIS[:, None] * points[..., None, :]  # ẑ_i x_j
            z_eye = z[..., None, None] * _IDENTITY  # z δ_ij
            x_x = points[..., :, None] * points[..., None, :]  # x_i x_j
            grad_b = 3.0 * moment * r5_inv[..., None, None] * (x_z + z_eye + z_x) - (
                15.0 * moment * (z * r7_inv)[..., None, None] * x_x
            )
        return b, grad_b

    def _jit_field(self) -> JitField:
        """Return the dipole's payload for the numba transport kernel (``kind = 1``).

        Opts the dipole into the scalar-per-lane ``prange`` kernel so the analytic validation gates
        run through it. The kernel inlines the same closed form as :meth:`_evaluate`; only the
        dipole scalars and the domain radii are needed (``b_padded`` is an unused placeholder, and
        ``char_const`` carries the constant CFL metric :meth:`characteristic_length` returns).
        """
        return JitField(
            kind=1,
            b_padded=_DIPOLE_PLACEHOLDER,
            n_r=2,
            n_theta=2,
            n_phi=2,
            spacing_code=0,
            r_inner=self.r_sun,
            r_outer=self.r_source,
            exponent=1.0,
            moment=self._dipole_moment,
            background=self._background_field,
            char_const=self._CHARACTERISTIC_FRACTION * (self.r_source - self.r_sun),
        )

    def flux_function(self, points: np.ndarray) -> np.ndarray:
        """Return the poloidal flux function ``Ψ ∝ (r² + 2R_S³/r) sin²θ`` at ``points``.

        Field lines are contours of ``Ψ`` (it is constant along each line), so the
        validation harness uses it to check traced field-line geometry. Returned up to the
        overall constant, in Cartesian form ``Ψ = (r² + 2R_S³/r)(x²+y²)/r²``.

        Parameters
        ----------
        points
            ``(..., 3)`` Cartesian coordinates in R☉.

        Returns
        -------
        numpy.ndarray
            ``(...)`` flux-function values.
        """
        points = np.asarray(points, dtype=np.float64)
        cylindrical2 = points[..., 0] ** 2 + points[..., 1] ** 2
        r2 = cylindrical2 + points[..., 2] ** 2
        r = np.sqrt(r2)
        return (r2 + 2.0 * self.r_source**3 / r) * cylindrical2 / r2

    def separatrix_colatitude(self, radius: float) -> float:
        """Return the separatrix (last-closed field line) colatitude ``θ_SL`` at ``radius``.

        From ``sin²θ_SL = 3 R_S² / (r² + 2R_S³/r)`` (the line through the cusp at
        ``(R_S, 90°)``). At ``r = 1.01``, ``R_S = 2.5`` this is 50.0°. The closed-field band
        is ``θ_SL < θ < 180° - θ_SL``.

        Parameters
        ----------
        radius
            Radius ``r`` in R☉ at which to evaluate the separatrix colatitude.

        Returns
        -------
        float
            ``θ_SL`` in radians.
        """
        sin2 = 3.0 * self.r_source**2 / (radius**2 + 2.0 * self.r_source**3 / radius)
        return float(np.arcsin(np.sqrt(sin2)))

    def q_perp_analytic(self, colatitude: np.ndarray, radius: float) -> np.ndarray:
        """Return the boundary-to-boundary Q⊥ of the field line through ``(radius, colatitude)``.

        Q⊥ is a property of the whole field line, constant along it (``B·∇Q⊥ = 0``) and anchored
        between the inner boundary ``R_⊙`` and the source surface ``R_S``. It is therefore
        evaluated by reducing the given point to its line's inner-boundary footpoint colatitude
        ``θ₀`` (field lines are flux contours ``Ψ = (r² + 2R_S³/r) sin²θ``) and applying the closed
        form there, so the value does not depend on where along the line the point sits. It is
        derived from this field and the perpendicular Q⊥ definition.
        The axisymmetric mapping factorizes into the azimuthal and meridional stretches, so
        ``Q⊥ = R + 1/R`` with ``Q⊥ ≥ 2``. Two regimes (in the footpoint colatitude ``θ₀``):

        - **Closed band** (``θ_SL < θ₀ < 180° - θ_SL``): equator-reflection symmetry gives
          ``R = 1``, so ``Q⊥ = 2`` exactly.
        - **Open polar caps** (``θ₀ < θ_SL`` or its mirror): the line maps to the source surface
          and ``R(θ₀) = (R_S/R_⊙)² e^{2C} (3R_⊙³/N) √(1 - e^{2C} sin²θ₀) / |B(R_⊙, θ₀)|`` with
          ``C = -ln sin θ_SL``, ``θ_SL = θ_SL(R_⊙)`` and ``N = R_⊙³ + 2R_S³``. ``R → 1`` at the pole
          (Q⊥ → 2) and ``R → 0`` at the separatrix (Q⊥ → ∞).

        Q⊥ is scale-invariant, so the overall field ``strength`` does not enter.

        Parameters
        ----------
        colatitude
            ``(...)`` colatitude(s) ``θ`` in radians of points on the field lines.
        radius
            Radius ``r`` in R☉ of those points (e.g. the seed radius ``R_seed`` the
            seeds sit at); only the line each point selects matters, not where along
            it the point falls.

        Returns
        -------
        numpy.ndarray
            ``(...)`` boundary-to-boundary Q⊥, matching the shape of ``colatitude``.
        """
        theta = np.asarray(colatitude, dtype=np.float64)
        r = float(radius)
        normalization = self.r_sun**3 + 2.0 * self.r_source**3

        # Reduce to the line's inner-boundary footpoint colatitude (flux conservation), staying in
        # the point's own hemisphere: Q⊥ is a footpoint-surface quantity, constant along the line.
        flux_ratio = (r**2 + 2.0 * self.r_source**3 / r) / (
            self.r_sun**2 + 2.0 * self.r_source**3 / self.r_sun
        )
        sin2_foot = np.clip(np.sin(theta) ** 2 * flux_ratio, 0.0, 1.0)
        theta_foot = np.arcsin(np.sqrt(sin2_foot))
        theta_foot = np.where(theta > 0.5 * np.pi, np.pi - theta_foot, theta_foot)

        theta_sl = self.separatrix_colatitude(self.r_sun)
        exp_2c = np.exp(-2.0 * np.log(np.sin(theta_sl)))  # e^{2C}, C = -ln sin θ_SL

        sin_foot = np.sin(theta_foot)
        cos_foot = np.cos(theta_foot)
        # |B(R_⊙, θ₀)| from the meridional components at the inner boundary (normalized field).
        b_r = (2.0 * self.r_source**3 + self.r_sun**3) / normalization * cos_foot
        b_theta = (self.r_source**3 - self.r_sun**3) / normalization * sin_foot
        b_inner = np.hypot(b_r, b_theta)

        with np.errstate(divide="ignore", invalid="ignore"):
            stretch = (
                (self.r_source**2 / self.r_sun**2)
                * exp_2c
                * (3.0 * self.r_sun**3 / normalization)
                * np.sqrt(np.clip(1.0 - exp_2c * sin_foot**2, 0.0, None))
                / b_inner
            )
            q_caps = stretch + 1.0 / stretch

        in_caps = (theta_foot < theta_sl) | (theta_foot > np.pi - theta_sl)
        return np.where(in_caps, q_caps, 2.0)
