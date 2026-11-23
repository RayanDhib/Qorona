"""EUV channel registry and response tables.

One table per channel and abundance set gives the response ``K_b(T, nₑ)`` of the channel, in
``cm⁵ s⁻¹ sr⁻¹`` times the channel unit. One pixel of the instrument records
``Ω_pix ∫ nₑ² K_b ds``. The same table gives the foreground absorption cross-section per hydrogen
nucleus ``κ_b(T)`` (cm²) at the channel's nominal wavelength.

The tables ship in ``radiation/tables/``. A maintainer script builds them once from the CHIANTI
atomic database and the instrument teams' wavelength responses, and records the recipe in each
table's provenance. Both axes are uniform in ``log₁₀``: temperature from 10⁴ to 10⁸ K and
electron density from 10⁶ to 10¹³ cm⁻³. The lookup is bilinear in ``log K``. Below the density
axis ``K`` keeps its low-density limit (the edge value); above it, the edge value too. Below
10⁴ K the plasma does not emit in these channels (``K = 0``); above 10⁸ K ``K`` keeps the edge
value.

Models that include the transition region broaden it artificially: below a temperature ``Tc`` of
the run, they raise the thermal conduction and lower the radiative loss by

    f(T) = [1 + (T / Tc)⁻¹⁰]^(1/4),

which tends to ``(Tc / T)^(5/2)`` well below ``Tc``. The broadened layer holds ``f`` times the
emission measure per unit temperature of a resolved one at the same pressure, so its emission is
divided by ``f`` (:meth:`ResponseTable.corrected_for_broadening`). Above about ``2 Tc``, ``f = 1``.

Implemented from Iijima & Imada (2021), ApJ 917, 65, Eq. 23 (the emission of a broadened
transition region).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from functools import cache
from importlib import resources
from typing import Any

import numpy as np

__all__ = [
    "ABUNDANCE_SETS",
    "CHANNELS",
    "DEFAULT_ABUNDANCES",
    "EuvChannel",
    "ResponseTable",
    "broadening_factor",
]

#: The abundance sets that ship with Qorona, by name.
ABUNDANCE_SETS = ("coronal-2021", "schmelz-2012")

#: The default abundance set: the CHIANTI 2021 coronal abundances.
DEFAULT_ABUNDANCES = "coronal-2021"

_ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)


def broadening_factor(temperature: np.ndarray, tr_temperature: float) -> np.ndarray:
    """The transition-region broadening factor ``f(T) = [1 + (T / Tc)⁻¹⁰]^(1/4)``.

    Parameters
    ----------
    temperature
        Temperatures (K).
    tr_temperature
        The run's broadening temperature ``Tc`` (K).
    """
    ratio = np.asarray(temperature, dtype=np.float64) / tr_temperature
    return (1.0 + ratio**-10.0) ** 0.25


@dataclass(frozen=True)
class EuvChannel:
    """One EUV instrument channel.

    Attributes
    ----------
    name
        Channel name used on the command line, for example ``aia-193`` or ``euvi-a-195``.
    instrument
        Instrument name as written in FITS headers (``AIA``, ``EUVI``, ``EUI``).
    band
        Nominal wavelength in Å.
    unit
        What one count is: ``DN`` (AIA, EUI) or ``ph`` (EUVI Level 1, detected photons).
    observer
        The spacecraft that carries the instrument, as a Qorona observer name.
    frame
        The instrument's camera frame (a key of
        :data:`qorona.config.schema.CAMERA_INSTRUMENTS`).
    colormap
        The sunpy colour map of the channel.
    display_range
        The default ``(low, high)`` of the ``log₁₀`` display, in the channel unit per second and
        native pixel.
    """

    name: str
    instrument: str
    band: int
    unit: str
    observer: str
    frame: str
    colormap: str
    display_range: tuple[float, float]


#: The default display ranges, in the channel unit per second and native pixel: the low end near
#: the level of the corona at 1.3 R☉, the high end near the brightest active regions.
_DISPLAY_RANGES: dict[str, tuple[float, float]] = {
    "aia-94": (0.03, 30.0),
    "aia-131": (0.2, 300.0),
    "aia-171": (10.0, 1.0e4),
    "aia-193": (10.0, 5000.0),
    "aia-211": (2.0, 2000.0),
    "aia-335": (0.1, 100.0),
    "euvi-171": (20.0, 5000.0),
    "euvi-195": (10.0, 2000.0),
    "euvi-284": (1.0, 300.0),
    "eui-fsi-174": (1.0, 1.0e4),
}


def _registry() -> dict[str, EuvChannel]:
    channels = [
        EuvChannel(
            f"aia-{band}",
            "AIA",
            band,
            "DN",
            "earth",
            "aia",
            f"sdoaia{band}",
            _DISPLAY_RANGES[f"aia-{band}"],
        )
        for band in (94, 131, 171, 193, 211, 335)
    ]
    channels += [
        EuvChannel(
            f"euvi-{craft}-{band}",
            "EUVI",
            band,
            "ph",
            f"stereo-{craft}",
            f"euvi-{craft}",
            f"euvi{band}",
            _DISPLAY_RANGES[f"euvi-{band}"],
        )
        for craft in ("a", "b")
        for band in (171, 195, 284)
    ]
    channels.append(
        EuvChannel(
            "eui-fsi-174",
            "EUI",
            174,
            "DN",
            "solo",
            "eui-fsi",
            "solar orbiterfsi174",
            _DISPLAY_RANGES["eui-fsi-174"],
        )
    )
    return {channel.name: channel for channel in channels}


#: Every supported EUV channel, by name.
CHANNELS: dict[str, EuvChannel] = _registry()


@dataclass(frozen=True, eq=False)
class ResponseTable:
    """The response table of one channel and abundance set.

    Attributes
    ----------
    channel
        The channel the table belongs to.
    abundances
        Name of the abundance set (one of :data:`ABUNDANCE_SETS`).
    log_t, log_ne
        The uniform ``log₁₀ T`` (K) and ``log₁₀ nₑ`` (cm⁻³) axes.
    log_k
        ``(n_t, n_ne)`` array of ``log₁₀ K`` in ``cm⁵ s⁻¹ sr⁻¹`` times the channel unit.
    kappa
        ``(n_t,)`` absorption cross-section per hydrogen nucleus (cm²).
    native_pixel_sr
        Solid angle of one native instrument pixel (sr).
    provenance
        The recipe the table was built with (atomic database, abundances, response file, ...).
    """

    channel: EuvChannel
    abundances: str
    log_t: np.ndarray
    log_ne: np.ndarray
    log_k: np.ndarray
    kappa: np.ndarray
    native_pixel_sr: float
    provenance: dict[str, Any] = field(repr=False)

    @classmethod
    def load(cls, channel: str, abundances: str = DEFAULT_ABUNDANCES) -> ResponseTable:
        """Load the shipped table of ``channel`` for the abundance set ``abundances``.

        Parameters
        ----------
        channel
            Channel name (a key of :data:`CHANNELS`).
        abundances
            Abundance set (one of :data:`ABUNDANCE_SETS`).

        Raises
        ------
        ValueError
            If the channel or the abundance set is unknown, or the table axes are not uniform.
        """
        return _load(channel, abundances)

    def evaluate(self, log_t: np.ndarray, log_ne: np.ndarray) -> np.ndarray:
        """Return ``K`` (not its logarithm) at the given ``log₁₀ T`` and ``log₁₀ nₑ``.

        Parameters
        ----------
        log_t, log_ne
            Broadcastable arrays of ``log₁₀ T`` (K) and ``log₁₀ nₑ`` (cm⁻³).

        Returns
        -------
        numpy.ndarray
            ``K`` in ``cm⁵ s⁻¹ sr⁻¹`` times the channel unit; zero below the temperature axis.
        """
        log_t, log_ne = np.broadcast_arrays(
            np.asarray(log_t, dtype=np.float64), np.asarray(log_ne, dtype=np.float64)
        )
        i, wi = _cell(log_t, self.log_t)
        j, wj = _cell(log_ne, self.log_ne)
        table = self.log_k
        log_k = (1.0 - wi) * ((1.0 - wj) * table[i, j] + wj * table[i, j + 1]) + wi * (
            (1.0 - wj) * table[i + 1, j] + wj * table[i + 1, j + 1]
        )
        return np.where(log_t < self.log_t[0], 0.0, 10.0**log_k)

    def corrected_for_broadening(self, tr_temperature: float) -> ResponseTable:
        """Return a copy whose ``K`` is divided by the broadening factor ``f(T)`` of ``Tc``.

        Parameters
        ----------
        tr_temperature
            The run's transition-region broadening temperature ``Tc`` (K).
        """
        if not tr_temperature > 0.0:
            raise ValueError(f"the broadening temperature must be > 0 K, got {tr_temperature}")
        log_f = np.log10(broadening_factor(10.0**self.log_t, tr_temperature))
        return replace(self, log_k=np.ascontiguousarray(self.log_k - log_f[:, None]))

    def absorption(self, log_t: np.ndarray) -> np.ndarray:
        """Return ``κ`` (cm² per hydrogen nucleus) at ``log₁₀ T``, linear in ``log₁₀ T``.

        Outside the temperature axis ``κ`` keeps the edge value.
        """
        log_t = np.asarray(log_t, dtype=np.float64)
        i, w = _cell(log_t, self.log_t)
        return (1.0 - w) * self.kappa[i] + w * self.kappa[i + 1]


def _cell(values: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lower cell index and weight on a uniform axis, with the values clamped to the axis."""
    step = axis[1] - axis[0]
    position = (np.clip(values, axis[0], axis[-1]) - axis[0]) / step
    index = np.minimum(np.floor(position).astype(np.int64), axis.size - 2)
    return index, position - index


def _uniform(axis: np.ndarray, name: str) -> np.ndarray:
    steps = np.diff(axis)
    if axis.size < 2 or not np.allclose(steps, steps[0], rtol=0.0, atol=1e-6):
        raise ValueError(f"the {name} axis of an EUV response table must be uniform")
    return axis


@cache
def _load(channel: str, abundances: str) -> ResponseTable:
    if channel not in CHANNELS:
        raise ValueError(f"unknown EUV channel {channel!r}; choose one of {sorted(CHANNELS)}")
    if abundances not in ABUNDANCE_SETS:
        raise ValueError(
            f"unknown abundance set {abundances!r}; choose one of {list(ABUNDANCE_SETS)}"
        )
    source = resources.files("qorona.radiation") / "tables" / f"{channel}_{abundances}.npz"
    with resources.as_file(source) as path, np.load(path, allow_pickle=False) as archive:
        provenance = json.loads(str(archive["meta"]))
        table = ResponseTable(
            channel=CHANNELS[channel],
            abundances=abundances,
            log_t=_uniform(np.asarray(archive["log_t"], dtype=np.float64), "temperature"),
            log_ne=_uniform(np.asarray(archive["log_ne"], dtype=np.float64), "density"),
            log_k=np.ascontiguousarray(archive["log_k"], dtype=np.float64),
            kappa=np.ascontiguousarray(archive["kappa"], dtype=np.float64),
            native_pixel_sr=float((provenance["native_pixel_arcsec"] * _ARCSEC_TO_RAD) ** 2),
            provenance=provenance,
        )
    return table
