"""Orthographic plane-of-sky camera: parallel lines of sight through the solution frame.

The corona at 1 AU subtends only ~0.5°, so the projection is essentially parallel; the camera is an
orthographic plane-of-sky view, the standard eclipse / coronagraph geometry. Its parameters are a
look direction, an up vector, a roll about the look axis, a physical field of view, and a pixel
count; a parallel projection needs no observer position, only a direction. The solar-north roll
defaults to 0°, and a real observation supplies its own roll, derived from the observer ephemeris
by the caller.

Conventions:

- ``look`` points from Sun centre toward the observer. The plane of sky is the plane through Sun
  centre perpendicular to it; the line-of-sight coordinate ``s`` is the **signed distance from that
  plane** (``s = 0`` on the plane, at the limb, and ``s > 0`` toward the observer), so a sample at
  heliocentric radius ``r`` lies at ``r² = rho² + s²`` with ``rho`` the ray's impact parameter.
- The image basis is right-handed as the observer sees it: image-up is ``up`` projected into the
  plane of sky, image-right is the observer's right ``up_proj x look`` (so ``right x up = +look``,
  out of the screen toward the observer, with no left-right mirror), and ``roll`` rotates the pair
  about ``look`` counter-clockwise in the observer's view, so at the default ``roll = 0`` with
  ``up`` the solution's north, the image is solar-north-up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.units import Quantity


@dataclass(frozen=True, slots=True)
class Rays:
    """Per-pixel parallel lines of sight in the solution frame.

    The rays all run along :attr:`look`; a sample at signed line-of-sight distance ``s`` from the
    plane of sky is ``origins + s · look``. The image is indexed ``[row, col]`` with row 0 at the
    top, so :attr:`origins` ``[i, j]`` is the plane-of-sky point of pixel ``(i, j)``.

    Attributes
    ----------
    look
        ``(3,)`` unit line-of-sight direction (toward the observer).
    right, up
        ``(3,)`` the in-plane image axes after roll (image-right and image-up), orthonormal and
        perpendicular to :attr:`look`.
    origins
        ``(H, W, 3)`` the plane-of-sky point (``s = 0``) of each pixel.
    impact
        ``(H, W)`` impact parameter rho: the perpendicular distance from Sun centre to each ray.
    """

    look: np.ndarray
    right: np.ndarray
    up: np.ndarray
    origins: np.ndarray
    impact: np.ndarray

    def points_at(self, s: np.ndarray) -> np.ndarray:
        """Return the sample points ``(H, W, k, 3)`` at the ``k`` line-of-sight distances ``s``."""
        return self.origins[:, :, None, :] + np.asarray(s)[:, None] * self.look


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

        The look direction points from Sun centre toward the sub-observer point on the unit sphere
        (degrees), with the image up-vector the solution's north ``+z``. ``(longitude, latitude) =
        (0, 0)`` is the equatorial front view down the ``+x`` axis (solar-north up).

        Parameters
        ----------
        longitude, latitude
            Sub-observer heliographic longitude and latitude in degrees (``latitude`` measured from
            the equator, so the north pole is ``+90``).
        fov, pixels, roll
            As on :class:`OrthographicCamera`.
        """
        lon, lat = np.deg2rad(longitude), np.deg2rad(latitude)
        look = np.array(
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        )
        return cls(look=look, up=np.array([0.0, 0.0, 1.0]), fov=fov, pixels=pixels, roll=roll)

    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the orthonormal ``(look, right, up)`` image basis after applying the roll."""
        look = np.asarray(self.look, dtype=np.float64)
        look = look / np.linalg.norm(look)
        up_hint = np.asarray(self.up, dtype=np.float64)
        vertical = up_hint - np.dot(up_hint, look) * look
        norm = np.linalg.norm(vertical)
        if norm < 1e-12:
            raise ValueError("up vector is parallel to look; cannot define an image vertical")
        vertical /= norm
        right = np.cross(vertical, look)  # observer's right; right x up = +look (out of screen)
        # Roll counter-clockwise in the observer's view rotates the projected up-vector toward image
        # left, so solar north at roll alpha lands at image direction (-sin alpha, cos alpha).
        cos_roll, sin_roll = np.cos(self.roll), np.sin(self.roll)
        right_rolled = cos_roll * right - sin_roll * vertical
        up_rolled = sin_roll * right + cos_roll * vertical
        return look, right_rolled, up_rolled

    def rays(self) -> Rays:
        """Build the per-pixel parallel lines of sight.

        Returns
        -------
        Rays
            The look direction, the rolled image axes, the per-pixel plane-of-sky origins, and the
            per-pixel impact parameter.
        """
        look, right, up = self._basis()
        height, width = self.pixels
        pixel_scale = self.fov.to_value(u.R_sun) / width

        # Pixel centres in R☉: +x to image-right, +y to image-up, row 0 at the top of the image.
        x = (np.arange(width) - 0.5 * (width - 1)) * pixel_scale
        y = (0.5 * (height - 1) - np.arange(height)) * pixel_scale
        grid_x, grid_y = np.meshgrid(x, y)

        origins = grid_x[..., None] * right + grid_y[..., None] * up
        impact = np.hypot(grid_x, grid_y)
        return Rays(look=look, right=right, up=up, origins=origins, impact=impact)

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project world ``points`` onto the image plane: the inverse of :meth:`rays`.

        The rolled image basis ``(look, right, up)`` is orthonormal, so a point ``P`` decomposes
        into image-right ``x = P·right``, image-up ``y = P·up`` (both R☉ in the plane of sky), and
        the signed line-of-sight depth ``s = P·look`` (``> 0`` toward the observer). The pixel
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
            signed depth ``s`` along the look axis.
        """
        look, right, up = self._basis()
        height, width = self.pixels
        pixel_scale = self.fov.to_value(u.R_sun) / width
        pts = np.asarray(points, dtype=np.float64)
        depth = pts @ look
        cols = (pts @ right) / pixel_scale + 0.5 * (width - 1)
        rows = 0.5 * (height - 1) - (pts @ up) / pixel_scale
        return cols, rows, depth
