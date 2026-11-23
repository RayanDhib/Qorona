"""FITS + WCS export: registration headers and the per-product writers.

Maps the render camera to a helioprojective WCS with a camera-derived observer. A Qorona
render thus drops into JHelioviewer (or any astropy/sunpy tool), registered in scale,
orientation, position, and time. Helioprojective coordinates and the solar FITS keyword
conventions follow Thompson (2006), A&A 449, 791.

The camera's sub-observer ``(longitude, latitude)`` is in the solution's own heliographic
frame (ingest contract: ``+z`` is the solar rotation axis). The observer is placed in
Carrington coordinates by adding the frame's longitude offset
(:func:`qorona.ephemeris.carrington_offset`: zero for a Carrington-aligned solution, L0 for a
Stonyhurst one, the run-anchored offset for an Icarus one), so every observer card follows from
one correctly placed observer.

- **Orthographic camera:** a TAN header whose intermediate coordinates are linear in
  plane-of-sky distance, so the plate scale ``(pixel pitch)·R☉/D`` registers plane-of-sky
  features exactly.
- **Perspective camera:** the header names the camera's own zenithal projection, its angular
  plate scale in degrees, and its reference pixel, so the WCS is exact. The projection is ARC,
  or AZP with ``PV2_1 = 0`` for the camera's TAN. PUNCH (Polarimeter to Unify the Corona and
  Heliosphere) uses the same AZP label for its Narrow Field Imager images.
"""

from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from qorona import __version__
from qorona.config.schema import IcarusRunConfig, SolutionFrame, frame_from_provenance

if TYPE_CHECKING:
    from pathlib import Path

    from astropy.io import fits

    from qorona.radiation.brightness import BrightnessResult
    from qorona.radiation.euv import EuvResult
    from qorona.radiation.punch_l3 import PunchL3Result
    from qorona.render.los import RenderResult

#: ``BUNIT`` of absolute white-light frames: the mean solar brightness, written as the literal the
#: PUNCH Level 3 headers use so the units match exactly.
MSB_BUNIT = "2.009e+07 W/(m2 sr)"

#: Citation trail written to every primary header.
_DOI_URL = "https://doi.org/10.5281/zenodo.20630699"
_REPO_URL = "https://github.com/RayanDhib/Qorona"


def camera_wcs_header(
    camera: dict[str, Any], timestamp: str, shape: tuple[int, int], *, frame: SolutionFrame = None
) -> fits.Header:
    """Build the helioprojective registration header for one image HDU.

    Parameters
    ----------
    camera
        The camera provenance block: ``projection``, ``longitude`` / ``latitude``
        (sub-observer degrees in the solution frame), ``roll`` (degrees),
        ``observer_distance`` (AU), and the field of view. The orthographic camera
        gives the field of view as ``fov`` (full width, R☉). The perspective camera
        gives ``fov_deg`` (full width, degrees), ``projection_code`` and
        ``reference_pixel`` (0-based ``(col, row)``, row 0 at the top; ``None`` for
        the image centre).
    timestamp
        UTC ISO-8601 epoch; becomes ``DATE-OBS`` / ``DATE-AVG`` and the observer
        ephemeris epoch.
    shape
        ``(H, W)`` image shape in pixels.
    frame
        The declared solution frame (``"carrington"`` / ``"stonyhurst"``, or the settings of
        an Icarus run), or ``None`` when undeclared (taken as Carrington; the COMMENT card
        says so).

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

    from qorona.ephemeris import carrington_offset

    time = Time(timestamp, scale="utc")
    height, width = int(shape[0]), int(shape[1])
    lon = (float(camera["longitude"]) + carrington_offset(frame, timestamp)) % 360.0
    lat = float(camera["latitude"])
    roll = float(np.deg2rad(float(camera["roll"])))
    distance = float(camera["observer_distance"]) * u.au

    observer = SkyCoord(
        HeliographicCarrington(lon * u.deg, lat * u.deg, distance, observer="self", obstime=time)
    )
    # PC maps FITS pixel axes (X image-right, Y image-up after the write-time flip) onto
    # the helioprojective axes: the camera roll rotates image content counter-clockwise,
    # so the matrix is the clockwise rotation by the same angle.
    cos_r, sin_r = float(np.cos(roll)), float(np.sin(roll))
    rotation = np.array([[cos_r, sin_r], [-sin_r, cos_r]])
    frame_hp = Helioprojective(observer=observer, obstime=time)
    if camera.get("projection") == "perspective":
        reference = SkyCoord(0 * u.deg, 0 * u.deg, frame=frame_hp)
        code = "ARC" if camera["projection_code"] == "arc" else "AZP"
        cdelt_deg = float(camera["fov_deg"]) / width
        anchor = camera.get("reference_pixel")
        col0, row0 = anchor if anchor is not None else (0.5 * (width - 1), 0.5 * (height - 1))
        # FITS counts rows from the bottom (the image is flipped at write time).
        meta = make_fitswcs_header(
            (height, width),
            reference,
            reference_pixel=[float(col0), (height - 1) - float(row0)] * u.pix,
            scale=[cdelt_deg, cdelt_deg] * u.deg / u.pix,
            rotation_matrix=rotation,
            projection_code=code,
        )
        # sunpy writes helioprojective cards in arcsec. Restate them in degrees, as PUNCH does.
        for axis in ("1", "2"):
            meta["cunit" + axis] = "deg"
            meta["cdelt" + axis] = cdelt_deg
            meta["crval" + axis] = 0.0
        if code == "AZP":
            meta["PV2_1"] = 0.0  # AZP with no offset is the gnomonic (TAN) projection
    else:
        reference = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=frame_hp)
        # Plate scale: (pixel pitch in R☉)·R☉/D, radians converted to arcsec; the same solar
        # radius sunpy stamps as RSUN_REF, so the header stays self-consistent.
        pitch = float(camera["fov"]) / width * sun_constants.radius
        cdelt = float(np.rad2deg(float(pitch / distance.to(u.m)))) * 3600.0
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
    if isinstance(frame, IcarusRunConfig):
        header["COMMENT"] = (
            f"solution frame: icarus (magnetogram {frame.magnetogram_time}, "
            f"relaxation {frame.relaxation:g} d, delta_phi {frame.delta_phi:g} deg)"
        )
    else:
        header["COMMENT"] = (
            f"solution frame: {frame}"
            if frame
            else "solution frame undeclared; taken as Carrington-aligned"
        )
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


def _provenance_frame(provenance: dict[str, Any]) -> SolutionFrame:
    """Return the declared solution frame from the run provenance, ``None`` when undeclared."""
    inp = provenance.get("input", {})
    return frame_from_provenance(inp) if isinstance(inp, dict) else None


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


def _punch_cards(header: fits.Header, punch: PunchL3Result) -> None:
    """Record the PUNCH Level 3 treatment in ``header``."""
    block = punch.to_provenance()
    height, width = punch.background_pixels
    header["L3WINDOW"] = (block["window_days"], "[d] half-width of the background window")
    header["L3STEP"] = (block["step_hours"], "[h] time between background frames")
    header["L3PCTL"] = (block["percentile"], "background percentile, per polarizer channel")
    header["L3FRAMES"] = (block["frames"], "number of background frames")
    header["L3BKGPIX"] = (f"{width}x{height}", "size of the background frames")


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
    header = camera_wcs_header(
        provenance["camera"], timestamp, data.shape, frame=_provenance_frame(provenance)
    )
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
    result: BrightnessResult,
    display: np.ndarray,
    path: Path,
    provenance: dict[str, Any],
    *,
    punch: PunchL3Result | None = None,
) -> None:
    """Write the white-light FITS: the display frame plus raw ``PB``/``TOTAL`` extensions.

    The primary HDU carries ``display``, the finished frame exactly as the PNG shows it
    (post vignette/MGN/stretch, in ``[0, 1]``, before 8-bit quantization); the raw
    physical frames ride as extensions with the same WCS, so downstream tools get the
    unstyled data. If the frames are absolute, their ``BUNIT`` is the mean solar
    brightness; relative frames carry ``BUNIT = '1'``. Raises :class:`ValueError` when the
    provenance carries no timestamp.

    With ``punch`` (the PUNCH Level 3 treatment), ``PB`` and ``TOTAL`` are the treated frames,
    ``PB_FULL`` and ``TOTAL_FULL`` the untreated ones, and the primary header records the
    treatment.
    """
    from astropy.io import fits

    timestamp = _required_timestamp(provenance)
    camera = provenance["camera"]
    solution_frame = _provenance_frame(provenance)
    header = camera_wcs_header(camera, timestamp, display.shape, frame=solution_frame)
    _identity_cards(header, provenance, detector="WL")
    _citation_cards(header)
    header["BTYPE"] = ("display brightness", "vignette/enhancement/stretch, as the PNG")
    header["HV_DMIN"] = (0.0, "JHelioviewer display minimum")
    header["HV_DMAX"] = (1.0, "JHelioviewer display maximum")
    _provenance_cards(header, provenance)
    layers: list[tuple[str, np.ndarray, str, str]] = [
        ("PB", result.polarized, "polarized brightness", "raw line-of-sight integral"),
        ("TOTAL", result.total, "total white-light brightness", "raw line-of-sight integral"),
    ]
    if punch is not None:
        _punch_cards(header, punch)
        treated = "PUNCH Level 3 treatment"
        untreated = "line-of-sight integral, full K corona"
        layers = [
            ("PB", punch.treated.polarized, "polarized brightness", treated),
            ("TOTAL", punch.treated.total, "total white-light brightness", treated),
            ("PB_FULL", punch.full.polarized, "polarized brightness", untreated),
            ("TOTAL_FULL", punch.full.total, "total white-light brightness", untreated),
        ]
    hdus: list[Any] = [fits.PrimaryHDU(_flipped(display), header)]
    for name, image, btype, origin in layers:
        ext_header = camera_wcs_header(camera, timestamp, image.shape, frame=solution_frame)
        _identity_cards(ext_header, provenance, detector="WL")
        ext_header["BTYPE"] = (btype, origin)
        if result.unit == "MSB":
            ext_header["BUNIT"] = (MSB_BUNIT, "mean solar brightness (MSB)")
        else:
            ext_header["BUNIT"] = ("1", "relative brightness, arbitrary units")
        hdus.append(fits.ImageHDU(_flipped(image), ext_header, name=name))
    fits.HDUList(hdus).writeto(path, overwrite=True)


def write_euv_fits(
    result: EuvResult,
    channel: str,
    path: Path,
    provenance: dict[str, Any],
    *,
    display_range: tuple[float, float] | None = None,
) -> None:
    """Write one EUV channel's physical frame as a WCS-registered FITS image.

    The data is the frame in the channel unit per second and native instrument pixel (``BUNIT``
    ``DN/s`` or ``ph/s``), float32. The header names the channel (``DETECTOR``, ``WAVELNTH``), the
    native pixel the values refer to (``NATPIXEL``), the transition-region broadening temperature
    the emission was corrected for (``TRTEMP``), and the run provenance. Raises
    :class:`ValueError` when the provenance carries no timestamp.
    """
    from astropy.io import fits

    timestamp = _required_timestamp(provenance)
    frame = result.frames[channel]
    table = result.tables[channel]
    header = camera_wcs_header(
        provenance["camera"], timestamp, frame.shape, frame=_provenance_frame(provenance)
    )
    _identity_cards(header, provenance, detector=table.channel.name.upper())
    _citation_cards(header)
    header["WAVELNTH"] = (table.channel.band, "[Angstrom] nominal wavelength of the channel")
    header["WAVEUNIT"] = "angstrom"
    header["BTYPE"] = (
        f"{table.channel.instrument} {table.channel.band} intensity",
        "synthetic, optically thin",
    )
    header["BUNIT"] = (f"{table.channel.unit}/s", "per native instrument pixel")
    native = float(np.sqrt(table.native_pixel_sr) * 180.0 / np.pi * 3600.0)
    header["NATPIXEL"] = (round(native, 5), "[arcsec] native pixel the values refer to")
    header["TRTEMP"] = (
        result.tr_temperature if result.tr_temperature is not None else "none",
        "[K] transition-region broadening correction",
    )
    if display_range is not None:
        header["DISPMIN"] = (float(display_range[0]), "log display range of the PNG, low")
        header["DISPMAX"] = (float(display_range[1]), "log display range of the PNG, high")
    _provenance_cards(header, provenance)
    fits.HDUList([fits.PrimaryHDU(_flipped(frame), header)]).writeto(path, overwrite=True)
