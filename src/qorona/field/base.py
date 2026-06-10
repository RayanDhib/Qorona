"""The ``Field`` spine: the single interface every stage after the read stage consumes.

Both ``SampledField`` (real solutions on the internal grid) and ``AnalyticField`` (the PFSS
dipole and other closed-form validation fields) implement :class:`Field`, so the tracer,
squashing-factor, and render code is identical for real data and analytic test fields.

Contract (the numerical kernels speak plain ``float64``; units are handled once at the
construction edge, never in the hot loop):

- **Points** are Cartesian ``(x, y, z)`` in solar radii, in the field's named
  :class:`Domain` frame.
- **B** is returned in Cartesian components in the solution's native normalization (even
  though ``SampledField`` stores it on a spherical grid), which keeps the field and its
  gradient smooth through the poles.
- **∇B** is the *raw* Jacobian with convention ``grad_b[..., i, j] = ∂B_i/∂x_j``, so a
  directional derivative is ``(v·∇)B = grad_b @ v``. The unit-field gradient needed by the
  deviation transport, ``∇B̂ = (I - B̂ B̂ᵀ)·∇B / |B|``, is formed once by the deviation transport
  (``squashing/transport.py``), not here, so each field returns its natural Jacobian and the
  unit-field algebra lives in one place.
- **Domain.** ``sample`` assumes points are inside :attr:`Domain` and does not guard the
  boundary (see :class:`Field`'s precondition paragraph). A traced line leaving the domain is
  the normal terminating event, owned by the tracer (it stops crossed lines and triggers
  foot-landing), keeping the field branch-free and free of inf/nan arithmetic. This is unrelated
  to the genuine ``Q⊥ → ∞`` at a separatrix (a physical feature, not a boundary event).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class OutOfDomainError(ValueError):
    """Raised when a field is sampled outside its domain (a precondition violation).

    Surfaced only when :meth:`Field.sample` is called with ``validate=True`` (off by
    default); it turns a silent, meaningless out-of-domain read into a clear, immediate
    failure during development.
    """


@dataclass(frozen=True, slots=True)
class Domain:
    """The spherical shell a field is defined on, validated once at construction.

    Attributes
    ----------
    inner_radius
        Inner boundary radius in R☉ (the seeding surface, typically the photosphere).
    outer_radius
        Outer boundary radius in R☉ (the open-field-line target surface).
    frame
        Name of the physical coordinate frame the Cartesian axes are aligned with
        (descriptive metadata for downstream camera / WCS handling; not used by the field
        math).
    """

    inner_radius: float
    outer_radius: float
    frame: str

    def in_domain(self, points: np.ndarray) -> np.ndarray:
        """Return a boolean mask of which ``(..., 3)`` points lie within the shell.

        Parameters
        ----------
        points
            ``(..., 3)`` Cartesian coordinates in R☉.

        Returns
        -------
        numpy.ndarray
            ``(...)`` boolean mask, ``True`` where ``inner_radius ≤ |point| ≤ outer_radius``.
        """
        radius = np.sqrt(np.sum(points * points, axis=-1))
        return (radius >= self.inner_radius) & (radius <= self.outer_radius)

    def require_interior(self, points: np.ndarray) -> None:
        """Raise :class:`OutOfDomainError` if any of ``points`` lies outside the shell.

        The opt-in tripwire behind :meth:`Field.sample`'s ``validate`` flag.

        Parameters
        ----------
        points
            ``(..., 3)`` Cartesian coordinates in R☉.
        """
        inside = self.in_domain(points)
        if not inside.all():
            n_outside = int((~inside).sum())
            raise OutOfDomainError(
                f"{n_outside} of {inside.size} points lie outside the domain "
                f"[{self.inner_radius}, {self.outer_radius}] R_sun"
            )


@dataclass(frozen=True, eq=False, slots=True)
class FieldSample:
    """The result of a field evaluation: a struct-of-arrays bundle of plain ``float64``.

    ``b_magnitude`` is computed once, alongside ``b``, and reused everywhere ``|B|`` is
    needed (the unit field ``B̂ = B/|B|``, the unit-field gradient, and the Q⊥ prefactor at
    the seed and both footpoints) so no two code paths can disagree on it.

    Attributes
    ----------
    b
        ``(n, 3)`` Cartesian magnetic field, native normalization.
    b_magnitude
        ``(n,)`` field strength ``|B|``.
    grad_b
        ``(n, 3, 3)`` raw Jacobian ``∂B_i/∂x_j`` (``(v·∇)B = grad_b @ v``), or ``None`` when
        the evaluation was requested without the gradient.
    """

    b: np.ndarray
    b_magnitude: np.ndarray
    grad_b: np.ndarray | None


class Field(ABC):
    """A magnetic field queryable at arbitrary points (the spine of the pipeline).

    Subclasses implement the fused primitive :meth:`sample` and the :attr:`domain` property;
    the scalar/vector convenience accessors are derived from :meth:`sample` here. The hot loop
    binds :meth:`sample` directly; the convenience accessors are for out-of-loop and
    diagnostic use and must not be called per integration step.

    **Domain precondition.** :meth:`sample` evaluates the field assuming every point lies
    inside :attr:`domain`; it does *not* guard the boundary in the hot path. Callers test
    membership with :meth:`Domain.in_domain` and pass only in-domain points (the tracer
    already carries an active-line mask, and uses the same predicate to detect boundary
    crossings and trigger foot-landing). Sampling outside the domain returns a meaningless
    value (a wrong interpolation off the ghost padding, or a non-finite closed-form value),
    *not* a clean error, so out-of-domain policy stays with the caller that owns it and the
    hot path stays branch-free. Pass ``validate=True`` during development to turn that silent
    precondition into an explicit :class:`OutOfDomainError`.
    """

    @property
    @abstractmethod
    def domain(self) -> Domain:
        """The spherical shell the field is defined on."""

    @abstractmethod
    def characteristic_length(self, points: np.ndarray) -> np.ndarray:
        """Return the local cell metric ``(n,)`` at ``points``, a CFL step ceiling.

        The tracer caps each arc-length step at ``h_max = cfl · characteristic_length`` so no
        step skips sub-cell structure on a stretched mesh (the CFL condition specialised to
        unit-field tracing, where the "velocity" ``|B̂| = 1``). It must be a field method, not
        a scalar passed to the tracer, because a log-spaced grid varies cell size ~25x between
        the inner seeding surface and the outer corona, so a single ceiling cannot serve both.

        Parameters
        ----------
        points
            ``(n, 3)`` Cartesian coordinates in R☉, assumed inside :attr:`domain`.

        Returns
        -------
        numpy.ndarray
            ``(n,)`` local characteristic length in R☉ (the smallest local cell extent for a
            gridded field; a fixed fraction of the domain width for a grid-free analytic field).
        """

    @abstractmethod
    def sample(
        self, points: np.ndarray, *, gradient: bool = True, validate: bool = False
    ) -> FieldSample:
        """Evaluate B (and optionally ∇B) at in-domain ``points``, the fused primitive.

        B and ∇B share the expensive per-point setup, and ``|B|`` is computed once. Points
        are assumed inside :attr:`domain` (see the class precondition); ``gradient=False``
        skips the Jacobian (``grad_b`` is ``None``) for the B-only paths (the Q⊥ prefactor's
        footpoint ``|B|`` and the classical-Q channel).

        Parameters
        ----------
        points
            ``(n, 3)`` Cartesian coordinates in R☉, assumed inside :attr:`domain`.
        gradient
            Whether to also compute the Jacobian ``grad_b``.
        validate
            When ``True``, check the in-domain precondition and raise
            :class:`OutOfDomainError` if any point lies outside :attr:`domain`. Default
            ``False`` keeps the hot path free of the check; enable it during development.

        Returns
        -------
        FieldSample
            B, ``|B|``, and optionally ∇B.
        """

    def reference_strength(self) -> float:
        """Return a peak ``|B|`` scale for the tracer's sharp-turn guard, or ``0.0`` if undefined.

        The guard's weak-field test fires where ``|B|`` falls below a fraction of this reference (a
        whole-field strong-field scale, not a per-point value). The default ``0.0`` makes that test
        never pass, so the guard is inert on fields that do not define a strength scale (e.g. the
        analytic validation fields); :class:`~qorona.field.sampled.SampledField` overrides it with
        the grid-max ``|B|``, the peak-field scale ``weak_fraction`` is expressed against.
        """
        return 0.0

    def magnetic_field(self, points: np.ndarray) -> np.ndarray:
        """Return the ``(n, 3)`` Cartesian magnetic field at ``points`` (no gradient)."""
        return self.sample(points, gradient=False).b

    def field_magnitude(self, points: np.ndarray) -> np.ndarray:
        """Return the ``(n,)`` field strength ``|B|`` at ``points`` (no gradient)."""
        return self.sample(points, gradient=False).b_magnitude

    def field_gradient(self, points: np.ndarray) -> np.ndarray:
        """Return the ``(n, 3, 3)`` raw Jacobian ``∂B_i/∂x_j`` at ``points``."""
        grad_b = self.sample(points, gradient=True).grad_b
        assert grad_b is not None  # gradient=True always populates grad_b
        return grad_b
