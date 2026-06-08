"""The field abstraction: the ``Field`` spine and its implementations.

Everything after the read stage consumes only :class:`Field`. ``AnalyticField`` (closed-form
validation fields such as the PFSS dipole) and ``SampledField`` (real solutions on the
internal spherical grid) implement it, so downstream code is identical for both.
"""

from __future__ import annotations

from qorona.field.analytic import AnalyticField, PfssDipoleField
from qorona.field.base import Domain, Field, FieldSample, OutOfDomainError
from qorona.field.sampled import SampledField

__all__ = [
    "AnalyticField",
    "Domain",
    "Field",
    "FieldSample",
    "OutOfDomainError",
    "PfssDipoleField",
    "SampledField",
]
