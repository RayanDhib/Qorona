"""FITS + WCS export: registration headers and the per-product writers.

Maps the orthographic plane-of-sky camera to a helioprojective TAN WCS with a
camera-derived observer, so a Qorona render drops into JHelioviewer (or any
astropy/sunpy tool) registered in scale, orientation, position, and time.
Helioprojective coordinates and the solar FITS keyword conventions follow
Thompson (2006), A&A 449, 791.

The solution frame is Carrington-aligned (ingest contract: ``+z`` is the solar
rotation axis, the azimuth is Carrington longitude), so the camera's sub-observer
``(longitude, latitude)`` is the observer's Carrington position. The TAN
projection's intermediate coordinates are linear in plane-of-sky distance, so the
plate scale ``(pixel pitch)·R☉/D`` registers plane-of-sky features exactly.
"""

from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from qorona import __version__

if TYPE_CHECKING:
    from pathlib import Path

    from astropy.io import fits

    from qorona.radiation.brightness import BrightnessResult
    from qorona.render.los import RenderResult

#: Citation trail written to every primary header.
_DOI_URL = "https://doi.org/10.5281/zenodo.20630699"
_REPO_URL = "https://github.com/RayanDhib/Qorona"


def camera_wcs_header(
    camera: dict[str, Any], timestamp: str, shape: tuple[int, int]
) -> fits.Header:
    """Build the helioprojective TAN registration header for one image HDU.

    Parameters
    ----------
    camera
        The camera provenance block: ``longitude`` / ``latitude`` (sub-observer
        Carrington degrees), ``roll`` (degrees), ``fov`` (full width, R☉), and
        ``observer_distance`` (AU).
    timestamp
        UTC ISO-8601 epoch; becomes ``DATE-OBS`` / ``DATE-AVG`` and the observer
        ephemeris epoch.
    shape
        ``(H, W)`` image shape in pixels.

    Returns
    -------
    astropy.io.fits.Header
        CTYPE/CUNIT/CDELT/CRPIX/CRVAL/PC plus the observer block (``DSUN_OBS``,
        ``HGLN_OBS``/``HGLT_OBS``, ``CRLN_OBS``/``CRLT_OBS``, ``RSUN_REF``) and the
        time keywords.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.time import Time
    from sunpy.coordinates import HeliographicCarrington, Helioprojective
    from sunpy.map import make_fitswcs_header
    from sunpy.sun import constants as sun_constants

    time = Time(timestamp, scale="utc")
    height, width = int(shape[0]), int(shape[1])
    lon, lat = float(camera["longitude"]), float(camera["latitude"])
    roll = float(np.deg2rad(float(camera["roll"])))
    distance = float(camera["observer_distance"]) * u.au

    observer = SkyCoord(
        HeliographicCarrington(
            lon * u.deg, lat * u.deg, distance, observer="self", obstime=time
        )
    )
    reference = SkyCoord(
        0 * u.arcsec, 0 * u.arcsec, frame=Helioprojective(observer=observer, obstime=time)
    )
    # Plate scale: (pixel pitch in R☉)·R☉/D, radians converted to arcsec; the same solar
    # radius sunpy stamps as RSUN_REF, so the header stays self-consistent.
    pitch = float(camera["fov"]) / width * sun_constants.radius
    cdelt = float(np.rad2deg(float(pitch / distance.to(u.m)))) * 3600.0
    # PC maps FITS pixel axes (X image-right, Y image-up after the write-time flip) onto
    # the helioprojective axes: the camera roll rotates image content counter-clockwise,
    # so the matrix is the clockwise rotation by the same angle.
    cos_r, sin_r = float(np.cos(roll)), float(np.sin(roll))
    rotation = np.array([[cos_r, sin_r], [-sin_r, cos_r]])
    meta = make_fitswcs_header(
        (height, width),
        reference,
        scale=[cdelt, cdelt] * u.arcsec / u.pix,
        rotation_matrix=rotation,
        projection_code="TAN",
    )
    header = fits.Header()
    for key, value in meta.items():
        header[key] = value
    header["CRLN_OBS"] = (lon, "[deg] observer Carrington longitude")
    header["CRLT_OBS"] = (lat, "[deg] observer Carrington latitude")
    header["COMMENT"] = "WCS assumes the solution frame is Carrington-aligned"
    header["DATE-OBS"] = (time.isot, "epoch of the modelled observation (UTC)")
    header["DATE-AVG"] = (time.isot, "same as DATE-OBS (instantaneous snapshot)")
    return header


def _required_timestamp(provenance: dict[str, Any]) -> str:
    """Return the run timestamp or raise: FITS export is undefined without an epoch."""
    inp = provenance.get("input", {})
    timestamp = inp.get("timestamp") if isinstance(inp, dict) else None
    if not timestamp:
        raise ValueError(
            "FITS export needs a timestamp for DATE-OBS and the observer ephemeris; "
            "pass --timestamp (or bake the volume with one)"
        )
    return str(timestamp)


def _identity_cards(header: fits.Header, provenance: dict[str, Any], detector: str) -> None:
    """Stamp the identity keywords carried by every HDU onto ``header``."""
    inp = provenance.get("input", {}) if isinstance(provenance.get("input"), dict) else {}
    header["TELESCOP"] = ("Qorona", "synthetic coronal imagery")
    header["INSTRUME"] = (str(inp.get("model") or "unknown").upper(), "source MHD model")
    header["DETECTOR"] = (detector, "Qorona product")
    if inp.get("cr") is not None:
        header["CAR_ROT"] = (int(inp["cr"]), "Carrington rotation of DATE-OBS")


def _citation_cards(header: fits.Header) -> None:
    """Stamp the citation trail, carried by the primary header only, onto ``header``.

    ``ORIGIN`` carries the version of the qorona writing the file; the bake version
    rides in the ``HIERARCH QORONA VERSION`` provenance card.
    """
    header["ORIGIN"] = (f"Qorona {__version__}", _REPO_URL)
    header["REFERENC"] = (_DOI_URL, "concept DOI, all versions")


def _provenance_cards(header: fits.Header, provenance: dict[str, Any]) -> None:
    """Flatten the run provenance into ``HIERARCH QORONA <section> <field>`` cards.

    Values are ASCII-sanitized and keywords too long for an 80-char card are skipped.
    """

    def walk(prefix: str, node: Any) -> None:
        if len(prefix) > 65:  # cannot fit an 80-char card alongside "HIERARCH " and a value
            return
        if isinstance(node, dict):
            for key, value in node.items():
                walk(f"{prefix} {key.upper()}", value)
        elif node is not None:
            value = json.dumps(node) if isinstance(node, (list, tuple)) else node
            if isinstance(value, str):
                value = value.encode("ascii", "replace").decode()
            header[f"HIERARCH {prefix}"] = value

    walk("QORONA", provenance)


def _flipped(frame: np.ndarray) -> np.ndarray:
    """Return ``frame`` as float32 in FITS row order (row 1 at the image bottom)."""
    return np.ascontiguousarray(np.flipud(np.asarray(frame, dtype=np.float32)))


def write_render_fits(result: RenderResult, path: Path, provenance: dict[str, Any]) -> None:
    """Write the quantitative Q⊥ FITS: one image HDU of LOS-averaged log₁₀ Q⊥.

    The data is the channel-mean of :attr:`~qorona.render.los.RenderResult.signal` (the
    quantity the grayscale PNG stretches, before the stretch), float32, ``NaN`` where a
    pixel has no valid sample. Raises :class:`ValueError` when the provenance carries no
    timestamp (the CLI refuses earlier; this guards the library surface).
    """
    from astropy.io import fits

    timestamp = _required_timestamp(provenance)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN pixels
        data = np.nanmean(result.signal, axis=2)
    header = camera_wcs_header(provenance["camera"], timestamp, data.shape)
    _identity_cards(header, provenance, detector="QPERP")
    _citation_cards(header)
    header["BTYPE"] = ("log10(Qperp)", "LOS-averaged log10 squashing factor")
    header["BUNIT"] = ("1", "dimensionless (log10 of a ratio)")
    render_prov = provenance.get("render", {})
    clamp = render_prov.get("clamp") if isinstance(render_prov, dict) else None
    if clamp:
        header["HV_DMIN"] = (float(clamp[0]), "JHelioviewer display minimum")
        header["HV_DMAX"] = (float(clamp[1]), "JHelioviewer display maximum")
    _provenance_cards(header, provenance)
    fits.HDUList([fits.PrimaryHDU(_flipped(data), header)]).writeto(path, overwrite=True)


def write_brightness_fits(
    result: BrightnessResult, display: np.ndarray, path: Path, provenance: dict[str, Any]
) -> None:
    """Write the white-light FITS: the display frame plus raw ``PB``/``TOTAL`` extensions.

    The primary HDU carries ``display``, the finished frame exactly as the PNG shows it
    (post vignette/MGN/stretch, in ``[0, 1]``, before 8-bit quantization); the raw
    physical frames ride as extensions with the same WCS, so downstream tools get the
    unstyled data. Raises :class:`ValueError` when the provenance carries no timestamp.
    """
    from astropy.io import fits

    timestamp = _required_timestamp(provenance)
    camera = provenance["camera"]
    header = camera_wcs_header(camera, timestamp, display.shape)
    _identity_cards(header, provenance, detector="WL")
    _citation_cards(header)
    header["BTYPE"] = ("display brightness", "vignette/enhancement/stretch, as the PNG")
    header["HV_DMIN"] = (0.0, "JHelioviewer display minimum")
    header["HV_DMAX"] = (1.0, "JHelioviewer display maximum")
    _provenance_cards(header, provenance)
    hdus: list[Any] = [fits.PrimaryHDU(_flipped(display), header)]
    for name, frame, btype in (
        ("PB", result.polarized, "polarized brightness"),
        ("TOTAL", result.total, "total white-light brightness"),
    ):
        ext_header = camera_wcs_header(camera, timestamp, frame.shape)
        _identity_cards(ext_header, provenance, detector="WL")
        ext_header["BTYPE"] = (btype, "raw line-of-sight integral")
        ext_header["BUNIT"] = ("1", "relative brightness, arbitrary units")
        hdus.append(fits.ImageHDU(_flipped(frame), ext_header, name=name))
    fits.HDUList(hdus).writeto(path, overwrite=True)
