"""Ephemeris at the pipeline edge: UTC timestamps to Carrington rotation / Julian date, the
solution frame's longitude offset, and named observers' sub-observer angles.

The two solution frames Qorona accepts, Carrington (HGC) and Stonyhurst (HGS, the spherical form
of HEEQ), share the solar rotation axis and differ only by the longitude of the sub-Earth
meridian, L0(t) (Thompson 2006, A&A 449, 791). Every frame conversion here is therefore one
time-dependent longitude offset, with latitudes common to both frames. Earth's position comes
from sunpy's built-in ephemeris (offline); a spacecraft's from JPL Horizons over the network.
"""

from __future__ import annotations

import logging
from typing import Any

from qorona.config.schema import DEFAULT_FRAME, SOLUTION_FRAMES

#: Named observers accepted by ``--observer``: Earth resolves offline, the spacecraft query JPL
#: Horizons by body id.
OBSERVERS = ("earth", "stereo-a", "solo", "psp")
_HORIZONS_IDS = {"stereo-a": "-234", "solo": "-144", "psp": "-96"}


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


def carrington_offset(frame: str | None, timestamp: str) -> float:
    """Return the Carrington longitude (deg) of ``frame``'s zero meridian at ``timestamp``.

    Zero for a Carrington-aligned solution; L0(t), the sub-Earth Carrington longitude, for a
    Stonyhurst one. Adding it takes a solution-frame longitude to Carrington, subtracting it takes
    a Carrington longitude into the solution frame. ``None`` (undeclared) counts as Carrington.
    """
    if _resolved_frame(frame) == DEFAULT_FRAME:
        return 0.0

    from sunpy.coordinates.sun import L0

    return float(L0(utc_time(timestamp)).deg)


def sub_observer_point(
    observer: str, timestamp: str, frame: str | None
) -> tuple[float, float, float]:
    """Return ``observer``'s sub-observer ``(longitude, latitude)`` in degrees, expressed in the
    solution ``frame``, and its heliocentric distance in AU at ``timestamp``.

    Earth comes from sunpy's built-in ephemeris (no network); a spacecraft from JPL Horizons, whose
    Stonyhurst longitude is taken to Carrington by adding L0. The Carrington longitude then drops
    into the solution frame by subtracting the frame's offset, so Earth in a Stonyhurst frame sits
    at exactly zero.
    """
    import astropy.units as u
    from sunpy.coordinates.sun import B0, L0, earth_distance

    resolved_frame = _resolved_frame(frame)
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
    # The frame's offset, :func:`carrington_offset`'s value taken off the L0 already in hand:
    # zero for a Carrington-aligned solution, L0 itself for a Stonyhurst one.
    offset = 0.0 if resolved_frame == DEFAULT_FRAME else sub_earth_longitude
    longitude = (carrington_longitude - offset) % 360.0
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


def _resolved_frame(frame: str | None) -> str:
    """Validate a solution ``frame``, mapping ``None`` (undeclared) to the default."""
    resolved = DEFAULT_FRAME if frame is None else frame
    if resolved not in SOLUTION_FRAMES:
        raise ValueError(f"frame must be one of {SOLUTION_FRAMES}, got {frame!r}")
    return resolved
