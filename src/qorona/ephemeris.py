"""Ephemeris at the pipeline edge: UTC timestamps to Carrington rotation / Julian date, the
solution frame's longitude offset, and named observers' sub-observer angles.

The two solution frames Qorona accepts, Carrington (HGC) and Stonyhurst (HGS, the spherical form
of HEEQ), share the solar rotation axis and differ only by the longitude of the sub-Earth
meridian, L0(t) (Thompson 2006, A&A 449, 791). Every frame conversion here is therefore one
time-dependent longitude offset, with latitudes common to both frames. Earth's position comes
from sunpy's built-in ephemeris (offline); a spacecraft's from JPL Horizons over the network.

An Icarus snapshot is in a third frame. It shares the rotation axis, and it turns with a sidereal
period of 24.47 days. Earth is at ``phi = delta_phi`` at the magnetogram time and then drifts at a
constant rate, so the offset to Carrington is again one time-dependent longitude.
"""

from __future__ import annotations

import logging
from typing import Any

from qorona.config.schema import (
    DEFAULT_FRAME,
    ICARUS_FRAME,
    SOLUTION_FRAMES,
    IcarusRunConfig,
    SolutionFrame,
)

#: Named observers accepted by ``--observer``: Earth resolves offline, the spacecraft query JPL
#: Horizons by body id.
OBSERVERS = ("earth", "stereo-a", "solo", "psp")
_HORIZONS_IDS = {"stereo-a": "-234", "solo": "-144", "psp": "-96"}

#: Earth's drift through the Icarus frame, in degrees per hour. The frame turns with a sidereal
#: period of 24.47 days, and Icarus moves Earth at the mean rate of a 365.24-day year.
ICARUS_EARTH_DRIFT = 15.0 * (1.0 / 24.47 - 1.0 / 365.24)


def utc_time(timestamp: str) -> Any:
    """Parse a UTC ISO-8601 ``timestamp`` into an :class:`astropy.time.Time` (friendly on error)."""
    from astropy.time import Time

    try:
        return Time(timestamp, scale="utc")
    except Exception as exc:
        raise ValueError(
            f"could not parse timestamp {timestamp!r} as a UTC ISO-8601 datetime "
            f"(e.g. '2024-04-08T18:17:00'): {exc}"
        ) from exc


def derive_cr(timestamp: str) -> int:
    """Return the active Carrington-rotation number at a UTC ISO-8601 ``timestamp``.

    The active rotation is ``floor(carrington_rotation_number(t))``; sunpy's fractional part is the
    phase within the rotation.
    """
    from sunpy.coordinates.sun import carrington_rotation_number

    return int(carrington_rotation_number(utc_time(timestamp)))


def derive_jd(timestamp: str) -> float:
    """Return the Julian date for a UTC ISO-8601 ``timestamp``."""
    return float(utc_time(timestamp).jd)


def icarus_epoch(run: IcarusRunConfig, time_hours: float) -> str:
    """Return the UTC ISO-8601 time of the Icarus snapshot whose ``TIME`` is ``time_hours``."""
    import astropy.units as u

    epoch = utc_time(run.magnetogram_time) + (time_hours - run.magnetogram_hours) * u.hour
    epoch.precision = 0
    return str(epoch.isot)


def carrington_offset(frame: SolutionFrame, timestamp: str, epoch: str | None = None) -> float:
    """Return the Carrington longitude (deg) of ``frame``'s zero meridian at ``timestamp``.

    Zero for a Carrington-aligned solution; L0, the sub-Earth Carrington longitude, for a
    Stonyhurst one. For an Icarus run, ``L0 - delta_phi`` plus Earth's drift since the magnetogram
    time. Adding it takes a solution-frame longitude to Carrington, subtracting it takes a
    Carrington longitude into the solution frame. ``None`` (undeclared) counts as Carrington.

    A Stonyhurst solution holds at one time. ``epoch`` is that time when it is not ``timestamp``:
    the offset is then ``L0(epoch)``, and the solution turns with the Sun at the Carrington rate.
    """
    resolved = _resolved_frame(frame)
    if resolved == DEFAULT_FRAME:
        return 0.0

    from sunpy.coordinates.sun import L0

    if isinstance(frame, IcarusRunConfig):
        time = utc_time(timestamp)
        hours = float((time - utc_time(frame.magnetogram_time)).to_value("hour"))
        return (float(L0(time).deg) - frame.delta_phi + ICARUS_EARTH_DRIFT * hours) % 360.0
    return float(L0(utc_time(epoch or timestamp)).deg)


def sub_observer_point(
    observer: str, timestamp: str, frame: SolutionFrame, epoch: str | None = None
) -> tuple[float, float, float]:
    """Return ``observer``'s sub-observer ``(longitude, latitude)`` in degrees, expressed in the
    solution ``frame``, and its heliocentric distance in AU at ``timestamp``.

    Earth comes from sunpy's built-in ephemeris (no network); a spacecraft from JPL Horizons, whose
    Stonyhurst longitude is taken to Carrington by adding L0. The Carrington longitude then drops
    into the solution frame by subtracting the frame's offset (:func:`carrington_offset`, with
    ``epoch``), so Earth in a Stonyhurst frame sits at exactly zero.
    """
    import astropy.units as u
    from sunpy.coordinates.sun import B0, L0, earth_distance

    _resolved_frame(frame)  # validate before any network query
    time = utc_time(timestamp)
    sub_earth_longitude = float(L0(time).deg)
    if observer == "earth":
        carrington_longitude = sub_earth_longitude
        latitude = float(B0(time).deg)
        distance = float(earth_distance(time).to_value(u.au))
    elif observer in _HORIZONS_IDS:
        position = _horizons_position(observer, time)
        carrington_longitude = float(position.lon.deg) + sub_earth_longitude
        latitude = float(position.lat.deg)
        distance = float(position.radius.to_value(u.au))
    else:
        raise ValueError(f"unknown observer {observer!r}; choose from {', '.join(OBSERVERS)}")
    longitude = (carrington_longitude - carrington_offset(frame, timestamp, epoch)) % 360.0
    return longitude, latitude, distance


def _horizons_position(observer: str, time: Any) -> Any:
    """Query JPL Horizons for ``observer``'s Stonyhurst position at ``time``.

    Quiet on success (sunpy's per-query INFO line is silenced) and friendly on failure: the query
    needs network access, and the error names the manual pointing fallback.
    """
    from sunpy import log
    from sunpy.coordinates import get_horizons_coord

    level = log.level
    log.setLevel(logging.WARNING)
    try:
        return get_horizons_coord(_HORIZONS_IDS[observer], time)
    except Exception as exc:
        raise ValueError(
            f"could not resolve {observer} from JPL Horizons ({exc}); the query needs network "
            "access, or point manually with --longitude/--latitude"
        ) from exc
    finally:
        log.setLevel(level)


def _resolved_frame(frame: SolutionFrame) -> str:
    """Validate a solution ``frame``, mapping ``None`` (undeclared) to the default."""
    if isinstance(frame, IcarusRunConfig):
        return ICARUS_FRAME
    if frame == ICARUS_FRAME:
        raise ValueError(
            "the Icarus frame needs the run settings; pass --magnetogram-time and --relaxation "
            "(or --par)"
        )
    resolved = DEFAULT_FRAME if frame is None else frame
    if resolved not in SOLUTION_FRAMES:
        raise ValueError(f"frame must be one of {SOLUTION_FRAMES}, got {frame!r}")
    return resolved
