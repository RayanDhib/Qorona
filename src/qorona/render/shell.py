"""Q-map: signed log₁₀ Q⊥ on a spherical shell, sliced from the cached Q⊥ volume.

A Q-map is a constant-radius slice of the viewpoint-independent Q⊥ volume, signed by the local
radial field at the shell. Because Q⊥ is constant along a field line, a shell sample equals the
line's boundary-to-boundary value, so slicing the cached volume is the same quantity the
references obtain by tracing each shell point, and the same bake-once method PSI uses (Chitta
et al. 2023, Nat. Astron. 7, 133; Mikić et al. 2018, Nat. Astron. 2, 913). The displayed quantity is

    sign(B·r̂) · log₁₀ max(Q⊥, 2),

so the heliospheric current sheet is the warm↔cool boundary and the separatrix-web arcs are the
saturated ridges; the local-radial-field sign and the signed-log-Q idea follow the HMI QMap
(http://hmi.stanford.edu/QMap/). Colour: warm (red) outward, cool (blue) inward.

The orchestration (load the volume, sample the shell) lives in :mod:`qorona.pipeline`; this module
is the :class:`QMap` data structure and its figure / raster / data writers, with the dependency-free
PNG writer shared from the line-of-sight render (:mod:`qorona.render.los`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qorona.render.los import LOG_FLOOR, _write_png

__all__ = ["DEFAULT_SLOG_MAX", "QMap"]

#: Display ceiling for the map when the run sets none: it saturates at ±5. Override with
#: ``--slog-max``.
DEFAULT_SLOG_MAX = 5.0

#: Diverging palette for the map, from inward (negative) to outward (positive): dark navy → blue →
#: white at the floor / neutral line → red → dark maroon. Darker, more saturated ends than the
#: render's polarity tint, for a publication shell map.
_SLOG_STOPS = (
    (0.01, 0.08, 0.24),
    (0.13, 0.40, 0.92),
    (0.97, 0.97, 0.97),
    (0.82, 0.12, 0.12),
    (0.25, 0.00, 0.04),
)


def _slog_colour(signed: np.ndarray) -> np.ndarray:
    """Map signed values in ``[-1, 1]`` to the diverging :data:`_SLOG_STOPS` palette as RGB."""
    t = (np.clip(signed, -1.0, 1.0) + 1.0) / 2.0
    positions = np.linspace(0.0, 1.0, len(_SLOG_STOPS))
    stops = np.asarray(_SLOG_STOPS)
    return np.stack([np.interp(t, positions, stops[:, c]) for c in range(3)], axis=-1)


@dataclass(frozen=True)
class QMap:
    """Signed log₁₀ Q⊥ on a (θ, φ) shell at fixed radius (an S-web map).

    Attributes
    ----------
    radius
        Shell radius in R☉.
    theta, phi
        ``(n_theta,)`` colatitude and ``(n_phi,)`` longitude of the cell centres, in radians.
    log_q_perp
        ``(n_theta, n_phi)`` log₁₀ Q⊥ sampled from the volume at each cell; ``NaN`` where the
        volume carries no value (a gap or outside the shell).
    radial_sign
        ``(n_theta, n_phi)`` local radial polarity ``sign(B·r̂)`` at the shell, in ``{-1, 0, +1}``.
    """

    radius: float
    theta: np.ndarray
    phi: np.ndarray
    log_q_perp: np.ndarray
    radial_sign: np.ndarray

    @property
    def coverage(self) -> float:
        """Fraction of shell cells with a finite sampled Q⊥."""
        return float(np.mean(np.isfinite(self.log_q_perp)))

    @property
    def sub_floor_fraction(self) -> float:
        """Fraction of finite cells below the Q⊥ = 2 floor (a resampling ∇·B artifact; clamped to
        the floor for display)."""
        finite = np.isfinite(self.log_q_perp)
        n = int(np.count_nonzero(finite))
        return float(np.count_nonzero(self.log_q_perp[finite] < LOG_FLOOR) / n) if n else 0.0

    def slog_q(self) -> np.ndarray:
        """Return ``(n_theta, n_phi)`` ``sign(B·r̂) · log₁₀ max(Q⊥, 2)``.

        The sub-floor tail is clamped up to the Q⊥ = 2 floor exactly as the LOS render clamps it;
        ``NaN`` follows ``log_q_perp`` (a gap).
        """
        magnitude = np.clip(self.log_q_perp, LOG_FLOOR, None)
        return np.sign(np.nan_to_num(self.radial_sign)) * magnitude

    def resolved_slog_max(self, slog_max: float | None) -> float:
        """Return ``slog_max`` if set, else :data:`DEFAULT_SLOG_MAX`."""
        return slog_max if slog_max is not None else DEFAULT_SLOG_MAX

    def save_figure(
        self,
        path: str | Path,
        *,
        slog_max: float | None = None,
        title: str | None = None,
        dpi: int = 150,
    ) -> None:
        """Write a publication figure (lon/lat axes, diverging colour bar, title) to ``path``.

        A longitude/latitude map of ``sign(B_r)·log₁₀ Q⊥`` on the diverging palette: warm (red)
        outward, cool (blue) inward, white at the floor / neutral line, clamped to ``±slog_max``.
        Gap cells show grey. Requires matplotlib (raises :class:`ImportError` otherwise);
        :meth:`save_png` is the dependency-free fallback.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        from matplotlib.figure import Figure

        ceiling = self.resolved_slog_max(slog_max)
        cmap = LinearSegmentedColormap.from_list("qorona_slogq", _SLOG_STOPS)
        cmap.set_bad((0.85, 0.85, 0.85))

        figure = Figure(figsize=(11.0, 5.5))
        FigureCanvasAgg(figure)
        ax = figure.add_subplot(111)
        image = ax.imshow(
            np.ma.masked_invalid(self.slog_q()),
            extent=(0.0, 360.0, -90.0, 90.0),
            origin="upper",
            aspect="auto",
            cmap=cmap,
            norm=Normalize(vmin=-ceiling, vmax=ceiling),
            interpolation="nearest",
        )
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        ax.set_xticks(np.arange(0.0, 361.0, 60.0))
        ax.set_yticks(np.arange(-90.0, 91.0, 30.0))
        if title is not None:
            ax.set_title(title)
        bar = figure.colorbar(image, ax=ax, pad=0.02, extend="both")
        bar.set_label(r"sign($B_r$) $\cdot\ \log_{10} Q_\perp$")
        figure.savefig(path, dpi=dpi, bbox_inches="tight")

    def save_png(self, path: str | Path, *, slog_max: float | None = None) -> None:
        """Write the bare colour raster (matplotlib-free fallback; north up, longitude right)."""
        ceiling = self.resolved_slog_max(slog_max)
        signed = np.clip(self.slog_q() / ceiling, -1.0, 1.0)
        rgb = _slog_colour(np.nan_to_num(signed).ravel()).reshape(*signed.shape, 3)
        rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        _write_png(Path(path), np.ascontiguousarray(rgb))

    def save_npz(self, path: str | Path, *, meta: str | None = None) -> None:
        """Write the raw shell arrays as a dependency-free ``.npz`` for external plotting.

        Carries the raw ``log_q_perp`` and ``radial_sign``, the derived ``slog_q``, and the
        ``theta`` / ``phi`` / ``radius`` grid, so a downstream tool replots with its own axes and
        colour bar. ``meta`` (run provenance as a JSON string) rides along.
        """
        np.savez_compressed(
            path,
            radius=np.asarray(self.radius, dtype=np.float64),
            theta=self.theta.astype(np.float32),
            phi=self.phi.astype(np.float32),
            log_q_perp=self.log_q_perp.astype(np.float32),
            radial_sign=self.radial_sign.astype(np.float32),
            slog_q=self.slog_q().astype(np.float32),
            meta=np.array(meta if meta is not None else ""),
        )
