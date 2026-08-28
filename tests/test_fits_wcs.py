"""FITS/WCS registration guard, plus the solution-frame identities the registration rests on.

The guard checks the exported header against the camera's own projection.

Known plane-of-sky points must land on the same pixels through astropy/sunpy's WCS
machinery as through ``OrthographicCamera.project``, pinning the scale, rotation-sign,
handedness, and row-flip conventions of the FITS export against independent code.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("astropy")
pytest.importorskip("sunpy")


def test_wcs_header_matches_camera_projection() -> None:
    import astropy.units as u
    from astropy.coordinates import CartesianRepresentation, SkyCoord
    from astropy.time import Time
    from astropy.wcs import WCS
    from sunpy.coordinates import HeliographicCarrington, Helioprojective

    from qorona.geometry.camera import OrthographicCamera
    from qorona.io.fits import camera_wcs_header

    lon, lat, roll_deg, fov = 30.0, 10.0, 15.0, 8.0
    height, width = 200, 256
    timestamp = "2019-07-02T19:23:00"
    camera_prov = {
        "longitude": lon,
        "latitude": lat,
        "roll": roll_deg,
        "fov": fov,
        "pixels": [height, width],
        "observer_distance": 1.0,
    }
    header = camera_wcs_header(camera_prov, timestamp, (height, width))

    # The registration keywords JHelioviewer requires, in the units it assumes.
    assert header["CTYPE1"] == "HPLN-TAN" and header["CTYPE2"] == "HPLT-TAN"
    assert header["CUNIT1"] == "arcsec" and header["CUNIT2"] == "arcsec"
    assert header["DSUN_OBS"] == pytest.approx(1.496e11, rel=0.01)  # metres, never AU
    assert str(header["DATE-OBS"]).startswith("2019-07-02T19:23:00")

    camera = OrthographicCamera.from_sub_observer(
        longitude=lon,
        latitude=lat,
        roll=float(np.deg2rad(roll_deg)),
        fov=fov * u.R_sun,
        pixels=(height, width),
    )
    # Plane-of-sky points (registrable exactly): the projected solar north at 1.5 R_sun
    # and the pre-roll image-right at 2.2 R_sun, both perpendicular to the look axis.
    look = np.asarray(camera.look, dtype=float)
    look /= np.linalg.norm(look)
    z_axis = np.array([0.0, 0.0, 1.0])
    north = z_axis - np.dot(z_axis, look) * look
    north /= np.linalg.norm(north)
    west = np.cross(north, look)
    points = np.array([1.5 * north, 2.2 * west])

    cols, rows, depth = camera.project(points)
    assert np.allclose(depth, 0.0, atol=1e-12)

    obstime = Time(timestamp, scale="utc")
    observer = SkyCoord(
        HeliographicCarrington(
            lon * u.deg, lat * u.deg, 1.0 * u.au, observer="self", obstime=obstime
        )
    )
    world = SkyCoord(
        CartesianRepresentation(
            points[:, 0] * u.R_sun, points[:, 1] * u.R_sun, points[:, 2] * u.R_sun
        ),
        frame=HeliographicCarrington(observer=observer, obstime=obstime),
    ).transform_to(Helioprojective(observer=observer, obstime=obstime))
    px, py = WCS(header).world_to_pixel(world)

    # FITS rows count from the bottom; the writers flip vertically at write time.
    assert np.allclose(px, cols, atol=1e-3)
    assert np.allclose((height - 1) - py, rows, atol=1e-3)


def test_stonyhurst_frame_registration() -> None:
    """The analytic identities the frame offset must satisfy: Earth in a Stonyhurst frame sits
    at longitude 0 (latitude B0) by definition, the offset is L0, and a Stonyhurst camera at
    longitude 0 registers at Carrington longitude L0 with a zero Stonyhurst observer longitude."""
    from astropy.time import Time
    from sunpy.coordinates.sun import B0, L0

    from qorona.ephemeris import carrington_offset, sub_observer_point
    from qorona.io.fits import camera_wcs_header

    timestamp = "2019-07-02T19:23:00"
    time = Time(timestamp, scale="utc")
    sub_earth_longitude, sub_earth_latitude = float(L0(time).deg), float(B0(time).deg)

    longitude, latitude, distance = sub_observer_point("earth", timestamp, "stonyhurst")
    assert longitude == pytest.approx(0.0, abs=1e-9)
    assert latitude == pytest.approx(sub_earth_latitude, abs=1e-9)
    assert 0.98 < distance < 1.02  # AU
    assert carrington_offset("stonyhurst", timestamp) == pytest.approx(sub_earth_longitude)
    assert carrington_offset(None, timestamp) == 0.0
    assert sub_observer_point("earth", timestamp, None)[0] == pytest.approx(sub_earth_longitude)

    camera = {
        "longitude": 0.0,
        "latitude": latitude,
        "roll": 0.0,
        "fov": 8.0,
        "pixels": [64, 64],
        "observer_distance": distance,
    }
    header = camera_wcs_header(camera, timestamp, (64, 64), frame="stonyhurst")
    assert header["CRLN_OBS"] == pytest.approx(sub_earth_longitude, abs=1e-6)
    assert header["HGLN_OBS"] == pytest.approx(0.0, abs=0.01)  # sunpy derives it from the observer
    assert "stonyhurst" in str(header["COMMENT"])
