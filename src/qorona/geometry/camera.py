"""The rendering cameras: an orthographic plane-of-sky view and a perspective view from an observer.

Both cameras emit straight lines of sight that one ray model describes. Every line-of-sight integral
(the Q⊥ render, the white-light brightness) thus runs unchanged on either camera.

Each ray is parameterised from its **closest-approach point** to Sun centre. ``rho`` is the impact
parameter (the closest-approach distance), and ``s`` is the signed distance from that point along
the ray, positive toward the observer. A sample then lies at heliocentric radius
``r² = rho² + s²`` and sees the Sun under the scattering angle ``χ``, with ``sin²χ = rho²/r²``.
Both relations hold for any straight ray, so the Thomson-scattering geometry needs no
camera-specific term.

- :class:`OrthographicCamera`: parallel rays from an observer at infinity, the eclipse /
  coronagraph geometry. The closest-approach points form the plane of sky, and image coordinates
  are lengths on it (R☉). No ray has an observer limit. This is the natural camera for a coronal
  field of view of a few solar radii seen from 1 AU, which subtends about one degree.
- :class:`PerspectiveCamera`: rays fanning out from an observer at a finite distance ``D``, always
  pointed at Sun centre. Image coordinates are angles (degrees). A zenithal projection, ARC
  (equidistant) or TAN (gnomonic), maps them to the elongation ``ε`` from Sun centre. The
  closest-approach points form the Thomson surface (the sphere on the Sun-observer diameter),
  with ``rho = D sin ε``. The observer sits at ``s = D cos ε`` on each ray, so samples with
  ``s > D cos ε`` lie behind the observer and are excluded. This camera is necessary when the
  observer is close to or inside the domain, as for heliospheric solutions seen from 1 AU.

Conventions shared by both cameras:

- ``look`` is the unit vector from Sun centre toward the observer. The rolled image basis is
  right-handed as the observer sees it. Image-up is ``up`` projected perpendicular to ``look``.
  Image-right is the observer's right ``up_proj x look``, so ``right x up = +look`` (out of the
  screen, with no left-right mirror). ``roll`` rotates the pair about ``look`` counter-clockwise
  in the observer's view. At the default ``roll = 0`` with ``up`` the solution's north, the image
  is solar-north-up.
- Images are indexed ``[row, col]`` with row 0 at the top. The solar-north roll defaults to 0. A
  real observation supplies its own roll, which the caller derives from the observer ephemeris.

The zenithal projections are implemented from Calabretta & Greisen (2002), "Representations of
celestial coordinates in FITS", A&A 395, 1077.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from astropy import units as u
from astropy.units import Quantity

__all__ = [
    "PROJECTION_CODES",
    "Camera",
    "OrthographicCamera",
    "PerspectiveCamera",
    "Rays",
    "check_arc_reach",
]

#: Below this ``|up x look|`` the default north ``+z`` is treated as parallel to the look axis (the
#: polar views at latitude +-90), and ``from_sub_observer`` substitutes the continuous-limit image
#: vertical rather than leaving the vertical undefined.
_POLE_PARALLEL_TOL = 1e-8

#: The zenithal projections a :class:`PerspectiveCamera` maps image angles through, named by their
#: FITS projection codes.
PROJECTION_CODES = ("arc", "tan")


@dataclass(frozen=True, slots=True)
class Rays:
    """A bundle of straight lines of sight in the solution frame, one per pixel.

    The per-ray arrays share a leading pixel shape: ``(H, W)`` for a whole image
    (:meth:`OrthographicCamera.rays`), or ``(m,)`` for a row-major run of pixels
    (``ray_batch``). A sample at signed line-of-sight distance ``s`` on ray ``k`` is
    ``origins[k] + s · directions[k]``.

    Attributes
    ----------
    look
        ``(3,)`` unit vector from Sun centre toward the observer (the camera axis).
    right, up
        ``(3,)`` the rolled image axes, orthonormal and perpendicular to :attr:`look`.
    origins
        ``(..., 3)`` each ray's closest-approach point to Sun centre (``s = 0``).
    directions
        ``(..., 3)`` each ray's unit direction, pointing toward the observer (``s`` increases
        along it).
    impact
        ``(...)`` impact parameter rho: each ray's closest-approach distance to Sun centre.
    observer_limit
        ``(...)`` the observer's position ``s`` on each ray. Samples beyond it lie behind the
        observer. ``+inf`` for parallel rays.
    parallel
        ``True`` when every ray runs along :attr:`look` with no observer limit (the orthographic
        camera). Then :attr:`directions` and :attr:`observer_limit` are read-only broadcast views.
    """

    look: np.ndarray
    right: np.ndarray
    up: np.ndarray
    origins: np.ndarray
    directions: np.ndarray
    impact: np.ndarray
    observer_limit: np.ndarray
    parallel: bool

    def points_at(self, s: np.ndarray) -> np.ndarray:
        """Return the sample points ``(..., k, 3)`` at the ``k`` line-of-sight distances ``s``."""
        steps = np.asarray(s)[:, None]
        return self.origins[..., None, :] + steps * self.directions[..., None, :]

    def packed(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return contiguous flat ``(origins, directions, impact, observer_limit)`` for the kernels.

        ``origins`` is ``(m, 3)`` and ``impact`` is ``(m,)``. A parallel bundle packs its one shared
        direction and unlimited observer distance as a single row (``(1, 3)`` and ``(1,)``). The
        line-of-sight kernels broadcast that row to every ray, so the orthographic render carries
        no per-ray direction arrays. For other bundles, ``directions`` is ``(m, 3)`` and
        ``observer_limit`` is ``(m,)``.
        """
        origins = np.ascontiguousarray(self.origins.reshape(-1, 3), dtype=np.float64)
        impact = np.ascontiguousarray(self.impact.reshape(-1), dtype=np.float64)
        if self.parallel:
            directions = np.ascontiguousarray(self.look[None, :], dtype=np.float64)
            observer_limit = np.full(1, np.inf)
        else:
            directions = np.ascontiguousarray(self.directions.reshape(-1, 3), dtype=np.float64)
            observer_limit = np.ascontiguousarray(self.observer_limit.reshape(-1), dtype=np.float64)
        return origins, directions, impact, observer_limit


def _sub_observer_axes(longitude: float, latitude: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(look, up)`` for a camera at heliographic ``(longitude, latitude)`` in degrees.

    ``look`` points from Sun centre toward the sub-observer point on the unit sphere, and ``up`` is
    the solution's north ``+z``. At the poles (``latitude = ±90``) north is parallel to the look
    axis and no image vertical is defined; ``up`` then takes the continuous limit of the image
    vertical as ``|latitude| → 90``, the camera-longitude meridian ``∓[cos lon, sin lon, 0]``
    (upper sign at ``+90``, lower at ``-90``), which points across the pole and keeps the polar
    view's orientation continuous with the nearby off-pole latitudes.
    """
    lon, lat = np.deg2rad(longitude), np.deg2rad(latitude)
    look = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    up = np.array([0.0, 0.0, 1.0])
    if np.linalg.norm(np.cross(up, look)) < _POLE_PARALLEL_TOL:
        up = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), 0.0])
    return look, up


def _image_basis(
    look_hint: np.ndarray, up_hint: np.ndarray, roll: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the orthonormal ``(look, right, up)`` image basis after applying ``roll``."""
    look = np.asarray(look_hint, dtype=np.float64)
    look = look / np.linalg.norm(look)
    up_vector = np.asarray(up_hint, dtype=np.float64)
    vertical = up_vector - np.dot(up_vector, look) * look
    norm = np.linalg.norm(vertical)
    if norm < 1e-12:
        raise ValueError("up vector is parallel to look; cannot define an image vertical")
    vertical /= norm
    right = np.cross(vertical, look)  # observer's right; right x up = +look (out of screen)
    # Roll counter-clockwise in the observer's view rotates the projected up-vector toward image
    # left, so solar north at roll alpha lands at image direction (-sin alpha, cos alpha).
    cos_roll, sin_roll = np.cos(roll), np.sin(roll)
    right_rolled = cos_roll * right - sin_roll * vertical
    up_rolled = sin_roll * right + cos_roll * vertical
    return look, right_rolled, up_rolled


def _batch_pixels(pixels: tuple[int, int], start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(rows, cols)`` of the row-major pixels ``[start, stop)`` of a ``pixels`` image."""
    height, width = pixels
    if not 0 <= start <= stop <= height * width:
        raise ValueError(f"pixel range [{start}, {stop}) is outside a {height}x{width} image")
    flat = np.arange(start, stop)
    return flat // width, flat % width


def _ring(right: np.ndarray, up: np.ndarray, radius: float, n_points: int) -> np.ndarray:
    """Return ``(n_points, 3)`` points evenly spaced on a circle of ``radius`` about the origin,
    in the plane spanned by the image axes ``right`` and ``up``."""
    angle = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    ring = np.cos(angle)[:, None] * right + np.sin(angle)[:, None] * up
    return ring * radius


@dataclass(frozen=True, kw_only=True)
class OrthographicCamera:
    """An orthographic plane-of-sky camera emitting parallel lines of sight.

    Parameters
    ----------
    look
        Line-of-sight direction, from Sun centre toward the observer (need not be normalised; it is
        treated as a direction).
    up
        Up-vector hint; its component perpendicular to ``look`` is the image's vertical before roll
        (typically the solution's north ``+z``). Must not be parallel to ``look``.
    roll
        Rotation about the look axis, in radians, counter-clockwise in the observer's view. ``0``
        (the default) is solar-north-up when ``up`` is the solution's north.
    fov
        Field of view as an :class:`~astropy.units.Quantity` length: the **full** physical width
        the image spans, in solar radii. Pixels are square, so the vertical extent is
        ``fov · H / W``.
    pixels
        ``(H, W)`` image shape in pixels (rows, columns).
    """

    look: np.ndarray
    up: np.ndarray
    fov: Quantity
    pixels: tuple[int, int]
    roll: float = 0.0

    #: Unit of the image coordinates (:meth:`pixel_axes`): lengths on the plane of sky.
    axis_unit = "R_sun"

    @classmethod
    def from_sub_observer(
        cls,
        *,
        longitude: float,
        latitude: float,
        fov: Quantity,
        pixels: tuple[int, int],
        roll: float = 0.0,
    ) -> OrthographicCamera:
        """Build a camera looking from the sub-observer point at heliographic ``(lon, lat)``.

        The look direction points from Sun centre toward the sub-observer point (degrees), with
        the image up-vector the solution's north ``+z``. ``(longitude, latitude) = (0, 0)`` is the
        equatorial front view down the ``+x`` axis (solar-north up). At the poles the up-vector
        takes the continuous limit of the image vertical, so the polar view's orientation stays
        continuous with the nearby off-pole latitudes.

        Parameters
        ----------
        longitude, latitude
            Sub-observer heliographic longitude and latitude in degrees (``latitude`` measured from
            the equator, so the north pole is ``+90``).
        fov, pixels, roll
            As on :class:`OrthographicCamera`.
        """
        look, up = _sub_observer_axes(longitude, latitude)
        return cls(look=look, up=up, fov=fov, pixels=pixels, roll=roll)

    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the orthonormal ``(look, right, up)`` image basis after applying the roll."""
        return _image_basis(self.look, self.up, self.roll)

    def _pixel_scale(self) -> float:
        """Return the pixel pitch in R☉."""
        return float(self.fov.to_value(u.R_sun)) / self.pixels[1]

    def pixel_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the pixel-centre coordinates ``(x (W,), y (H,))`` in R☉ on the plane of sky.

        ``x`` increases to image-right and ``y`` to image-up, with row 0 at the top, so a pixel's
        impact parameter is ``hypot(x, y)``.
        """
        height, width = self.pixels
        pixel_scale = self._pixel_scale()
        x = (np.arange(width) - 0.5 * (width - 1)) * pixel_scale
        y = (0.5 * (height - 1) - np.arange(height)) * pixel_scale
        return x, y

    def _bundle(self, x: np.ndarray, y: np.ndarray) -> Rays:
        """Build the parallel rays through the plane-of-sky points ``(x, y)`` (R☉)."""
        look, right, up = self._basis()
        origins = x[..., None] * right + y[..., None] * up
        impact = np.hypot(x, y)
        return Rays(
            look=look,
            right=right,
            up=up,
            origins=origins,
            directions=np.broadcast_to(look, origins.shape),
            impact=impact,
            observer_limit=np.broadcast_to(np.inf, impact.shape),
            parallel=True,
        )

    def rays(self) -> Rays:
        """Build the per-pixel parallel lines of sight for the whole image (``(H, W)`` arrays).

        The closest-approach points are the plane-of-sky points of the pixel centres.
        """
        x, y = self.pixel_axes()
        grid_x, grid_y = np.meshgrid(x, y)
        return self._bundle(grid_x, grid_y)

    def ray_batch(self, start: int, stop: int) -> Rays:
        """Build the lines of sight of the row-major pixels ``[start, stop)`` (``(m,)`` arrays).

        Each ray is identical to the same pixel's ray in :meth:`rays`. A render that walks the
        image in batches thus matches one that builds the whole bundle.
        """
        rows, cols = _batch_pixels(self.pixels, start, stop)
        height, width = self.pixels
        pixel_scale = self._pixel_scale()
        x = (cols - 0.5 * (width - 1)) * pixel_scale
        y = (0.5 * (height - 1) - rows) * pixel_scale
        return self._bundle(x, y)

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project world ``points`` onto the image plane: the inverse of :meth:`rays`.

        The rolled image basis ``(look, right, up)`` is orthonormal, so a point ``P`` decomposes
        into image-right ``x = P·right``, image-up ``y = P·up`` (both R☉ in the plane of sky), and
        the signed line-of-sight distance ``s = P·look`` (``> 0`` toward the observer). The pixel
        mapping is the exact algebraic inverse of the :meth:`rays` grid layout, so a point on pixel
        ``(row, col)``'s line of sight projects back to that ``(col, row)``.

        Parameters
        ----------
        points
            ``(..., 3)`` Cartesian world points in R☉.

        Returns
        -------
        tuple of numpy.ndarray
            ``(cols, rows, depth)``, each shaped ``(...)``: fractional pixel column (``+x`` to
            image-right, ``0`` at the left edge), fractional pixel row (``0`` at the top), and the
            signed line-of-sight distance ``s`` from the ray's closest-approach point.
        """
        look, right, up = self._basis()
        height, width = self.pixels
        pixel_scale = self._pixel_scale()
        pts = np.asarray(points, dtype=np.float64)
        depth = pts @ look
        cols = (pts @ right) / pixel_scale + 0.5 * (width - 1)
        rows = 0.5 * (height - 1) - (pts @ up) / pixel_scale
        return cols, rows, depth

    def faces_observer(self, points: np.ndarray) -> np.ndarray:
        """Return whether each point lies on the observer-facing side of its Sun-centred sphere.

        For parallel rays the facing side of every sphere is the hemisphere toward the observer,
        ``P · look > 0``.

        Parameters
        ----------
        points
            ``(..., 3)`` Cartesian world points in R☉.

        Returns
        -------
        numpy.ndarray
            ``(...)`` bool.
        """
        look = self._basis()[0]
        return np.asarray(points, dtype=np.float64) @ look > 0.0

    def limb(self, radius: float, n_points: int) -> np.ndarray:
        """Return ``(n_points, 3)`` points evenly spaced on the limb of a sphere of ``radius`` (R☉).

        For parallel rays the limb, where the lines of sight graze the sphere, is the great circle
        in the plane of sky.
        """
        _, right, up = self._basis()
        return _ring(right, up, radius, n_points)

    def limb_circle(self, radius: float) -> tuple[float, float, float]:
        """Return the image of the limb of a sphere of ``radius`` (R☉): a circle in pixels.

        Returns
        -------
        tuple of float
            ``(col, row, radius_px)``: the circle's centre, the projected Sun centre at the image
            centre, and its radius in pixels.
        """
        height, width = self.pixels
        return 0.5 * (width - 1), 0.5 * (height - 1), radius / self._pixel_scale()


def check_arc_reach(
    fov_deg: float, pixels: tuple[int, int], reference_pixel: tuple[float, float] | None
) -> None:
    """Raise :class:`ValueError` when an ARC frame reaches the anti-solar direction.

    ARC maps the angular radius on the projection plane to the elongation one to one, so the
    farthest pixel corner must stay below 180 degrees. ``reference_pixel`` is the 0-based
    ``(col, row)`` of Sun centre, ``None`` for the image centre. TAN needs no check: its arctan
    keeps every radius below 90 degrees.
    """
    height, width = pixels
    scale = fov_deg / width
    col, row = (
        reference_pixel if reference_pixel is not None else (0.5 * (width - 1), 0.5 * (height - 1))
    )
    reach_x = (max(col, width - 1 - col) + 0.5) * scale
    reach_y = (max(row, height - 1 - row) + 0.5) * scale
    reach = math.hypot(reach_x, reach_y)
    if reach >= 180.0:
        raise ValueError(
            f"the frame corners reach {reach:.1f} degrees from Sun centre, beyond the ARC "
            "projection's 180 degrees; narrow the field of view or move the reference pixel"
        )


@dataclass(frozen=True, kw_only=True)
class PerspectiveCamera:
    """A perspective camera at a finite observer distance, pointed at Sun centre.

    The image is Sun-centred: a pixel's offset from :attr:`reference_pixel` gives an angular radius
    ``R`` (degrees) and a position angle ``ψ`` (from image-right toward image-up). The projection
    maps ``R`` to the elongation ``ε``, the angle between the ray and the direction to Sun centre.
    ARC (zenithal equidistant) uses ``ε = R``. TAN (gnomonic, the zenithal perspective projection
    at infinite distance) uses ``tan ε = R`` (``R`` in radians). The ray leaving the observer is
    ``-cos ε · look + sin ε · (cos ψ · right + sin ψ · up)``.

    Parameters
    ----------
    look
        Direction from Sun centre toward the observer (need not be normalised).
    up
        Up-vector hint, as on :class:`OrthographicCamera`.
    observer_distance
        Heliocentric distance of the observer, an :class:`~astropy.units.Quantity` length.
    fov
        Field of view as an :class:`~astropy.units.Quantity` angle, below 180°: the **full** width
        of the image on the projection plane, the plate scale times the width in pixels (as the
        FITS ``CDELT``). Under ARC this is the angle the image spans; under TAN the angle is
        smaller, ``2 atan(tan(fov / 2))`` for a centred frame.
    pixels
        ``(H, W)`` image shape in pixels (rows, columns).
    roll
        Rotation about the look axis in radians, as on :class:`OrthographicCamera`.
    projection_code
        The zenithal projection by its FITS code: ``"arc"`` (zenithal equidistant, the default)
        or ``"tan"`` (gnomonic).
    reference_pixel
        ``(col, row)`` 0-based position of Sun centre in the image, row 0 at the top. It can be
        fractional. ``None`` (the default) is the image centre ``((W - 1)/2, (H - 1)/2)``.
    """

    look: np.ndarray
    up: np.ndarray
    observer_distance: Quantity
    fov: Quantity
    pixels: tuple[int, int]
    roll: float = 0.0
    projection_code: Literal["arc", "tan"] = "arc"
    reference_pixel: tuple[float, float] | None = None

    #: Unit of the image coordinates (:meth:`pixel_axes`): projection-plane angles from Sun
    #: centre.
    axis_unit = "deg"

    def __post_init__(self) -> None:
        if self.projection_code not in PROJECTION_CODES:
            raise ValueError(
                f"projection_code must be one of {PROJECTION_CODES}, got {self.projection_code!r}"
            )
        fov = float(self.fov.to_value(u.deg))
        if not 0.0 < fov < 180.0:
            raise ValueError(f"fov must be an angle between 0 and 180 degrees, got {self.fov}")
        if not float(self.observer_distance.to_value(u.R_sun)) > 0.0:
            raise ValueError(f"observer_distance must be positive, got {self.observer_distance}")
        if self.projection_code == "arc":
            check_arc_reach(fov, self.pixels, self.reference_pixel)

    @classmethod
    def from_sub_observer(
        cls,
        *,
        longitude: float,
        latitude: float,
        observer_distance: Quantity,
        fov: Quantity,
        pixels: tuple[int, int],
        roll: float = 0.0,
        projection_code: Literal["arc", "tan"] = "arc",
        reference_pixel: tuple[float, float] | None = None,
    ) -> PerspectiveCamera:
        """Build a camera at heliographic ``(longitude, latitude)`` and ``observer_distance``.

        The observer sits at the sub-observer point (degrees) scaled to ``observer_distance``. The
        image up-vector is the solution's north ``+z``, with the same polar handling as
        :meth:`OrthographicCamera.from_sub_observer`.

        Parameters
        ----------
        longitude, latitude
            Sub-observer heliographic longitude and latitude in degrees.
        observer_distance, fov, pixels, roll, projection_code, reference_pixel
            As on :class:`PerspectiveCamera`.
        """
        look, up = _sub_observer_axes(longitude, latitude)
        return cls(
            look=look,
            up=up,
            observer_distance=observer_distance,
            fov=fov,
            pixels=pixels,
            roll=roll,
            projection_code=projection_code,
            reference_pixel=reference_pixel,
        )

    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the orthonormal ``(look, right, up)`` image basis after applying the roll."""
        return _image_basis(self.look, self.up, self.roll)

    def _distance(self) -> float:
        """Return the observer distance in R☉."""
        return float(self.observer_distance.to_value(u.R_sun))

    def _pixel_scale(self) -> float:
        """Return the angular pixel pitch in degrees."""
        return float(self.fov.to_value(u.deg)) / self.pixels[1]

    def _reference(self) -> tuple[float, float]:
        """Return the 0-based ``(col, row)`` of Sun centre in the image."""
        if self.reference_pixel is not None:
            return float(self.reference_pixel[0]), float(self.reference_pixel[1])
        height, width = self.pixels
        return 0.5 * (width - 1), 0.5 * (height - 1)

    def pixel_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the pixel-centre angular offsets ``(x (W,), y (H,))`` from Sun centre, in degrees.

        ``x`` increases to image-right and ``y`` to image-up, with row 0 at the top. They are the
        projection-plane coordinates: a pixel's angular radius is ``hypot(x, y)``, which the
        projection maps to the elongation.
        """
        height, width = self.pixels
        pixel_scale = self._pixel_scale()
        col0, row0 = self._reference()
        x = (np.arange(width) - col0) * pixel_scale
        y = (row0 - np.arange(height)) * pixel_scale
        return x, y

    def _elongation(self, radius_deg: np.ndarray) -> np.ndarray:
        """Return the elongation (radians) of the projection-plane angular radius ``radius_deg``."""
        radius = np.deg2rad(radius_deg)
        return radius if self.projection_code == "arc" else np.arctan(radius)

    def _bundle(self, x: np.ndarray, y: np.ndarray) -> Rays:
        """Build the rays leaving the observer through the angular offsets ``(x, y)`` (degrees)."""
        look, right, up = self._basis()
        distance = self._distance()
        radius = np.hypot(x, y)
        elongation = self._elongation(radius)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos_psi = np.where(radius > 0.0, x / radius, 1.0)
            sin_psi = np.where(radius > 0.0, y / radius, 0.0)
        cos_e, sin_e = np.cos(elongation), np.sin(elongation)
        sky = -cos_e[..., None] * look + sin_e[..., None] * (
            cos_psi[..., None] * right + sin_psi[..., None] * up
        )
        observer_limit = distance * cos_e
        # The closest approach lies D cos ε from the observer along the outgoing ray.
        origins = distance * look + observer_limit[..., None] * sky
        return Rays(
            look=look,
            right=right,
            up=up,
            origins=origins,
            directions=-sky,
            impact=distance * sin_e,
            observer_limit=observer_limit,
            parallel=False,
        )

    def rays(self) -> Rays:
        """Build the per-pixel lines of sight for the whole image (``(H, W)`` arrays)."""
        x, y = self.pixel_axes()
        grid_x, grid_y = np.meshgrid(x, y)
        return self._bundle(grid_x, grid_y)

    def ray_batch(self, start: int, stop: int) -> Rays:
        """Build the lines of sight of the row-major pixels ``[start, stop)`` (``(m,)`` arrays).

        Each ray is identical to the same pixel's ray in :meth:`rays`. Walking the image in
        batches bounds the memory of a large frame.
        """
        rows, cols = _batch_pixels(self.pixels, start, stop)
        pixel_scale = self._pixel_scale()
        col0, row0 = self._reference()
        return self._bundle((cols - col0) * pixel_scale, (row0 - rows) * pixel_scale)

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project world ``points`` onto the image: the inverse of :meth:`rays`.

        A point is seen along the ray from the observer through it. Its elongation and position
        angle give the pixel through the projection. The returned ``depth`` is the point's signed
        line-of-sight distance ``s`` from that ray's closest-approach point (``> 0`` toward the
        observer). This is the same coordinate the rays use, so occlusion tests read the same on
        both cameras. A point at the observer, or (for TAN) at an elongation of 90° or more, has no
        image position and projects to ``NaN``.

        Parameters
        ----------
        points
            ``(..., 3)`` Cartesian world points in R☉.

        Returns
        -------
        tuple of numpy.ndarray
            ``(cols, rows, depth)``, each shaped ``(...)``: fractional pixel column, fractional
            pixel row (``0`` at the top), and the signed line-of-sight distance ``s``.
        """
        look, right, up = self._basis()
        distance = self._distance()
        pixel_scale = self._pixel_scale()
        col0, row0 = self._reference()
        pts = np.asarray(points, dtype=np.float64)
        offset = pts - distance * look
        reach = np.linalg.norm(offset, axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            sky = offset / reach[..., None]
        cos_e = np.clip(-(sky @ look), -1.0, 1.0)
        elongation = np.arccos(cos_e)
        psi = np.arctan2(sky @ up, sky @ right)
        if self.projection_code == "arc":
            radius = np.rad2deg(elongation)
            valid = reach > 0.0
        else:
            with np.errstate(invalid="ignore"):
                radius = np.rad2deg(np.tan(elongation))
            valid = (reach > 0.0) & (elongation < 0.5 * np.pi)
        cols = np.where(valid, col0 + radius * np.cos(psi) / pixel_scale, np.nan)
        rows = np.where(valid, row0 - radius * np.sin(psi) / pixel_scale, np.nan)
        depth = np.where(valid, distance * cos_e - reach, np.nan)
        return cols, rows, depth

    def faces_observer(self, points: np.ndarray) -> np.ndarray:
        """Return whether each point lies on the observer-facing side of its Sun-centred sphere.

        A point ``P`` on the sphere of radius ``|P|`` faces an observer at ``O = D · look`` when the
        observer is above its tangent plane. That condition is ``(O - P) · P > 0``, which is
        ``P · look > |P|² / D``. The facing points form the cap that the limb bounds. As ``D``
        increases, the cap grows toward the hemisphere ``P · look > 0``.

        Parameters
        ----------
        points
            ``(..., 3)`` Cartesian world points in R☉.

        Returns
        -------
        numpy.ndarray
            ``(...)`` bool.
        """
        look = self._basis()[0]
        pts = np.asarray(points, dtype=np.float64)
        return pts @ look > np.sum(pts * pts, axis=-1) / self._distance()

    def _check_outside(self, radius: float) -> float:
        """Return the observer distance in R☉, raising if the observer is inside the sphere."""
        distance = self._distance()
        if distance <= radius:
            raise ValueError(
                f"the observer at {distance:g} R_sun is inside the sphere of radius "
                f"{radius:g} R_sun, which then has no limb"
            )
        return distance

    def limb(self, radius: float, n_points: int) -> np.ndarray:
        """Return ``(n_points, 3)`` points evenly spaced on the limb of a sphere of ``radius`` (R☉).

        Seen from a finite distance ``D``, the limb is the circle where the observer's lines of
        sight touch the sphere. Its centre is ``(R²/D) · look`` and its radius is
        ``R · sqrt(1 - R²/D²)``.
        Raises if the observer is inside the sphere, which then has no limb.
        """
        distance = self._check_outside(radius)
        look, right, up = self._basis()
        centre = (radius * radius / distance) * look
        return centre + _ring(right, up, radius * np.sqrt(1.0 - (radius / distance) ** 2), n_points)

    def limb_circle(self, radius: float) -> tuple[float, float, float]:
        """Return the image of the limb of a sphere of ``radius`` (R☉): a circle in pixels.

        The limb lies at the elongation ``arcsin(R / D)`` in every direction, so both projections
        draw it as a circle about the reference pixel. Raises if the observer is inside the sphere.

        Returns
        -------
        tuple of float
            ``(col, row, radius_px)``: the circle's centre, the reference pixel, and its radius in
            pixels.
        """
        elongation = np.arcsin(radius / self._check_outside(radius))
        angle = elongation if self.projection_code == "arc" else np.tan(elongation)
        col0, row0 = self._reference()
        return col0, row0, float(np.rad2deg(angle)) / self._pixel_scale()


#: Either rendering camera: both expose ``pixels``, ``axis_unit``, ``pixel_axes``, ``rays``,
#: ``ray_batch``, ``project``, ``faces_observer``, ``limb``, and ``limb_circle``.
Camera = OrthographicCamera | PerspectiveCamera
