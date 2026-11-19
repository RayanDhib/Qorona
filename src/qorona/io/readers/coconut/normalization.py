"""The COOLFluiD "corona" normalization of COCONUT solutions, as physical units.

COCONUT stores dimensionless state values, scaled by fixed reference values of the COOLFluiD
corona model. The reference density is ``rho0 = 1.67e-13 kg/m³`` and the reference length is
``l0 = 6.9e8 m``. The other reference values are ``B0 = 2.2e-4 T``, the Alfvén speed
``V0 = B0 / sqrt(mu0 rho0)``, and ``p0 = rho0 V0²``. One stored density unit is therefore
``rho0``, and one coordinate unit is ``l0``, which is 0.8 % below the IAU solar radius. The
equation of state is ``p = 2 rho k T / (μ m_H)`` with ``μ = 1.27`` (``mu_cor`` in COOLFluiD).
The readers declare these values on the solution metadata so the absolute white-light brightness
can restore them. The geometry keeps reading coordinates as solar radii, because the model's inner
boundary ``r = 1`` is its photosphere.
"""

from __future__ import annotations

from astropy import units as u

__all__ = ["DENSITY_UNIT", "LENGTH_UNIT", "MEAN_MOLECULAR_WEIGHT"]

#: The physical mass density of one stored density unit, ``rho0``.
DENSITY_UNIT = u.Unit(1.67e-13 * u.kg / u.m**3)
#: The physical length of one stored coordinate unit, ``l0``.
LENGTH_UNIT = u.Unit(6.9e8 * u.m)
#: The mean molecular weight of the equation of state.
MEAN_MOLECULAR_WEIGHT = 1.27
