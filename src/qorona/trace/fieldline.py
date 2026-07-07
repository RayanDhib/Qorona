"""The traced-field-line container: a struct-of-arrays over many lines.

A field line is a curve with **two ends**, and "inner" / "outer" is a *property* of an end,
not an identity for it: a closed line has **both** feet on the inner sphere, so keying storage
by sphere cannot represent it. The two ends are therefore labelled by **trace direction**
(``backward`` along ``-B̂``, ``forward`` along ``+B̂``) and each carries an :class:`Endpoint`
code; a line's classification (closed, open, or incomplete: the three mutually exclusive,
exhaustive classes ``{is_closed, is_open, is_incomplete}``) is *derived* from the two codes,
so no separate flag can disagree with the feet. This is the natural layout for a field-line
squashing computation (the two end positions plus a per-end boundary code), and it is what the
Q⊥ volume fill consumes: trace an interior point both ways, read the precomputed boundary Q⊥ at
each foot on the sphere its :class:`Endpoint` names, and average.

The lines are stored as a struct-of-arrays rather than a list of per-line objects. This matches
how the rest of the pipeline holds bulk geometry (:class:`~qorona.field.base.FieldSample`) and
keeps the per-line core (``feet`` + ``ends``) small enough to stay a cheap transient even when a
line-of-sight render streams ~10⁸ points through it.

The container holds geometry only; the squashing factor and the transverse deviation vectors are
the squashing-factor stage's own result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Endpoint(IntEnum):
    """How one end of a traced field line terminated.

    The terminations are stored as a plain ``int8`` array (so tests like ``ends == Endpoint.INNER``
    vectorize); this enum is the naming layer over that array. It gives the codes readable names
    for end-of-run diagnostics (``Endpoint(code).name``) and for the open/closed classification.
    """

    INNER = 0
    """Landed on the inner sphere (clean foot, by root-finding on the dense-output interpolant)."""
    OUTER = 1
    """Landed on the outer sphere (clean foot)."""
    NULL = 2
    """Stopped at an exact magnetic null (non-finite ``B̂``): no clean foot; foot is ``NaN``."""
    MAX_STEPS = 3
    """A resource guard (max steps, or step-size underflow) fired before landing: a flagged
    diagnostic; foot is ``NaN``."""
    STALLED = 4
    """The stall guard fired: the line reversed direction too many times, the signature of a line
    trapped and thrashing at a weak-field null (the current sheet), where the unit field ``B/|B|``
    turns meaningless. Stopped without a clean foot; foot is ``NaN``."""
    DEFLECTED = 5
    """The sharp-turn guard fired: the line made a single sharp turn in the outer corona where
    ``|B|`` is weak, a grid-locked staircase deflection at a null, distinct from the thrashing the
    stall guard catches. Stopped without a clean foot; foot is ``NaN``."""


@dataclass(frozen=True)
class TurnGuard:
    """Parameters of the outer-corona sharp-turn guard.

    A committed step is a *qualifying* sharp turn at a null when its direction turns by more than
    ``max_turn_angle``, the new point is above ``radius``, and the local ``|B|`` is below
    ``weak_fraction`` of the field's peak ``|B|``. Once a line has made ``min_turns`` such turns it
    is terminated as :attr:`Endpoint.DEFLECTED` (and excluded like any incomplete line): the
    signature of a grid-locked staircase, which the reversal-counting stall guard is blind to.
    Requiring more than one turn keeps a line that merely *grazes* a null once (legitimate high-Q
    near-separatrix structure) while catching the sustained grid-locking that nothing physical
    mimics.

    Attributes
    ----------
    max_turn_angle
        Single-step turn threshold in degrees; ``0`` disables the guard (zero overhead).
    radius
        Count turns only above this radius in R☉ (the outer corona, where the current sheet lives).
    weak_fraction
        Count turns only where ``|B|`` is below this fraction of the field's peak ``|B|``.
    min_turns
        Number of qualifying sharp turns that triggers termination (isolated null grazes are kept).
    """

    max_turn_angle: float = 45.0
    radius: float = 2.0
    weak_fraction: float = 1.0e-5
    min_turns: int = 3

    @property
    def enabled(self) -> bool:
        """Whether the guard fires at all (``max_turn_angle > 0``)."""
        return self.max_turn_angle > 0.0

    @property
    def cos_threshold(self) -> float:
        """Cosine of ``max_turn_angle``: a step turns past the threshold when ``B̂·B̂ <`` this."""
        return math.cos(math.radians(self.max_turn_angle))


#: The calibrated sharp-turn guard, shared as the default wherever a tracer entry point takes one
#: (a frozen, immutable singleton, safe as a default argument).
DEFAULT_TURN_GUARD = TurnGuard()


@dataclass(frozen=True, slots=True)
class FieldLines:
    """A batch of traced field lines, stored as a struct-of-arrays over ``n`` seeds.

    Each line was traced from its seed in both directions to the domain boundaries. The two
    ends are indexed along axis 1 in trace-direction order ``(backward -B̂, forward +B̂)``;
    a clean foot is a point on the inner or outer sphere, and an end that aborted at a null or
    a resource guard carries a ``NaN`` foot and the corresponding :class:`Endpoint` code.

    Attributes
    ----------
    seeds
        ``(n, 3)`` the interior point each line was traced from. Carried on the struct (not left to
        the caller) so each row is self-describing: the squashing-factor stage reads ``B₀ = |B|``
        at the seed for its Q⊥ prefactor, and rows stay aligned with their feet when the render
        streams the lines in chunks.
    feet
        ``(n, 2, 3)`` the two end positions; axis 1 is ``(backward, forward)``. A row is
        ``NaN`` for an end that aborted (``NULL`` / ``MAX_STEPS``).
    ends
        ``(n, 2)`` ``int8`` :class:`Endpoint` code per end.
    lengths
        ``(n, 2)`` arc length from the seed to each end; the total line length is
        ``lengths.sum(axis=1)``. The spread of these values measures how uneven the lines are in
        length, which is the signal for whether batched tracing would benefit from compaction.
    paths
        ragged ``list`` of ``(m_i, 3)`` arrays ordered ``backward-foot → seed → forward-foot``,
        present only when traced with ``store_path=True`` (otherwise ``None``).
    """

    seeds: np.ndarray
    feet: np.ndarray
    ends: np.ndarray
    lengths: np.ndarray
    paths: list[np.ndarray] | None

    @property
    def is_complete(self) -> np.ndarray:
        """``(n,)`` bool: both ends landed cleanly on a boundary sphere (two real feet).

        Only complete lines have a valid squashing factor and can be filled into the volume; an
        incomplete line (an end aborted at a null or a resource guard) carries a ``NaN`` foot.
        :attr:`is_closed` and :attr:`is_open` are the two disjoint kinds of complete line.
        """
        return ((self.ends == Endpoint.INNER) | (self.ends == Endpoint.OUTER)).all(axis=1)

    @property
    def is_closed(self) -> np.ndarray:
        """``(n,)`` bool: both feet on the inner sphere (a closed loop)."""
        return (self.ends == Endpoint.INNER).all(axis=1)

    @property
    def is_open(self) -> np.ndarray:
        """``(n,)`` bool: a complete line with at least one foot on the outer sphere."""
        return self.is_complete & (self.ends == Endpoint.OUTER).any(axis=1)

    @property
    def is_incomplete(self) -> np.ndarray:
        """``(n,)`` bool: an end aborted before landing (null or resource guard); no valid Q⊥.

        The exact complement of :attr:`is_complete`, so ``{is_closed, is_open, is_incomplete}``
        partition every line into mutually exclusive, exhaustive classes.
        """
        return ~self.is_complete

    @property
    def is_null(self) -> np.ndarray:
        """``(n,)`` bool: an end stopped at an exact null; a finer diagnostic within
        :attr:`is_incomplete` (``is_null`` implies ``is_incomplete``)."""
        return (self.ends == Endpoint.NULL).any(axis=1)

    @property
    def is_stalled(self) -> np.ndarray:
        """``(n,)`` bool: an end stopped by the stall guard (thrashing at a weak-field null); a
        finer diagnostic within :attr:`is_incomplete` (``is_stalled`` implies ``is_incomplete``)."""
        return (self.ends == Endpoint.STALLED).any(axis=1)

    @property
    def is_deflected(self) -> np.ndarray:
        """``(n,)`` bool: an end stopped by the sharp-turn guard (a staircase deflection at a
        weak-field outer null); a finer diagnostic within :attr:`is_incomplete` (implies it)."""
        return (self.ends == Endpoint.DEFLECTED).any(axis=1)

    def summary(self) -> str:
        """Return a one-line breakdown of line classifications for end-of-run reporting.

        Reads the derived predicates directly (the single source of truth): the two complete
        kinds, closed and open, then the incomplete lines split into those that hit an exact
        null, those stopped by the stall guard, those stopped by the sharp-turn guard, and
        those stopped by a resource guard (``max-steps``).
        """
        null = self.is_null
        stalled = self.is_stalled
        deflected = self.is_deflected
        return (
            f"{int(self.is_closed.sum())} closed · {int(self.is_open.sum())} open · "
            f"{int(null.sum())} null · {int(stalled.sum())} stalled · "
            f"{int(deflected.sum())} deflected · "
            f"{int((self.is_incomplete & ~null & ~stalled & ~deflected).sum())} max-steps"
        )
