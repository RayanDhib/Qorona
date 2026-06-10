"""Validation study: PFSS dipole analytic Q⊥ vs Qorona's computed Q⊥.

End-to-end accuracy check of the tracer and Q⊥ engine against a field whose squashing factor is
known in closed form. It sweeps colatitude densely at the seed radius, runs
:func:`~qorona.squashing.compute_squashing` end-to-end on the analytic dipole, and overlays the
result on the closed-form theory curve
(:meth:`~qorona.field.analytic.PfssDipoleField.q_perp_analytic`):
the engine should return Q⊥ = 2 flat across the closed band, divergent spikes at the separatrices,
and the analytic profile over the open polar caps. It prints the quantitative comparison and a
seed-invariance check, and writes ``validation/figures/dipole_q_perp.png`` (regenerable).
This is **not** a unit test; the crisp regression gate lives in ``tests/test_dipole_squashing.py``;
run this on demand for the publication artifact:

    python validation/dipole_q_perp.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from qorona.field import PfssDipoleField
from qorona.geometry import spherical_to_cartesian
from qorona.squashing import SquashingResult, compute_squashing

#: Seed radius (just above the inner boundary) at which seeds sit; θ_SL = 50.0° here.
R_SEED = 1.01
_FIGURE_PATH = Path(__file__).parent / "figures" / "dipole_q_perp.png"


def _seeds(colatitude_deg: np.ndarray, *, azimuth: float = 0.7) -> np.ndarray:
    """Return Cartesian seeds at ``R_seed`` for the given colatitudes (degrees), at one azimuth."""
    colatitude = np.deg2rad(np.asarray(colatitude_deg, dtype=np.float64))
    spherical = np.stack(
        [np.full_like(colatitude, R_SEED), colatitude, np.full_like(colatitude, azimuth)], axis=-1
    )
    return spherical_to_cartesian(spherical)


def sweep(
    field: PfssDipoleField, colatitude_deg: np.ndarray, *, rtol: float = 1e-5
) -> SquashingResult:
    """Run :func:`compute_squashing` for seeds across ``colatitude_deg`` at ``R_seed``.

    Pinned to the CPU reference path: this study validates the engine against the closed-form
    theory, independent of any GPU rounding behaviour (that comparison lives in
    ``validation/cuda_parity.py``).
    """
    return compute_squashing(
        field, _seeds(colatitude_deg), rtol=rtol, device="cpu", show_progress=True
    )


def seed_invariance(field: PfssDipoleField, *, colatitude_deg: float = 30.0) -> tuple[float, float]:
    """Reseed one open-cap line at several radii along it; return ``(mean Q⊥, peak-to-peak)``.

    A direct test of the ``B₀²`` seed-position invariance the engine rests on: every seed on a
    given line must return that line's single boundary-to-boundary Q⊥.
    """
    line = compute_squashing(
        field, _seeds([colatitude_deg]), rtol=1e-6, store_path=True, device="cpu",
        show_progress=False,
    )
    path = line.lines.paths[0]
    radius = np.linalg.norm(path, axis=1)
    interior = path[(radius > field.r_sun + 0.02) & (radius < field.r_source - 0.02)]
    sample = interior[:: max(1, len(interior) // 6)][:6]
    reseeded = compute_squashing(field, sample, rtol=1e-6, device="cpu", show_progress=False)
    spread = float(np.nanmax(reseeded.q_perp) - np.nanmin(reseeded.q_perp))
    return float(np.nanmean(reseeded.q_perp)), spread


def _print_comparison(field: PfssDipoleField) -> None:
    """Print the engine vs the analytic theory at the reference colatitudes."""
    reference_deg = np.array([20.0, 30.0, 45.0, 49.0, 49.9, 60.0, 75.0, 90.0])
    result = compute_squashing(
        field, _seeds(reference_deg), rtol=1e-6, device="cpu", show_progress=False
    )
    theory = field.q_perp_analytic(np.deg2rad(reference_deg), R_SEED)
    theta_sl = np.degrees(field.separatrix_colatitude(R_SEED))
    print(f"\n  theta_SL(R_seed) = {theta_sl:.4f} deg  (closed band in between)")
    print(f"\n  {'theta':>8} {'engine':>12} {'theory':>12} {'rel err':>10}")
    for theta, q_engine, q_theory in zip(reference_deg, result.q_perp, theory, strict=True):
        rel = abs(q_engine / q_theory - 1.0)
        print(f"  {theta:8.1f} {q_engine:12.6f} {q_theory:12.6f} {rel:10.1e}")
    mean_q, spread = seed_invariance(field)
    print(f"\n  seed-invariance (theta=30 line): mean Q⊥ = {mean_q:.6f}, range {spread:.1e}")


def _make_figure(
    field: PfssDipoleField, result: SquashingResult, colatitude_deg: np.ndarray
) -> None:
    """Write the theory-vs-code figure: global view, a separatrix zoom, and the residual.

    Skips with a note if matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib not available - skipping figure (numerical comparison above stands).")
        return

    def theory(theta_deg: np.ndarray) -> np.ndarray:
        return field.q_perp_analytic(np.deg2rad(theta_deg), R_SEED)

    valid = result.valid
    theta = colatitude_deg[valid]
    q_engine = result.q_perp[valid]
    theta_sl = float(np.degrees(field.separatrix_colatitude(R_SEED)))
    theta_dense = np.unique(np.concatenate([
        np.linspace(1.0, 179.0, 1401),
        np.concatenate([
            center + sign * np.geomspace(2.0e-4, 6.0, 400)
            for center in (theta_sl, 180.0 - theta_sl) for sign in (-1.0, 1.0)
        ]),
    ]))

    # A window around the north separatrix for the zoom; drop a hair at θ_SL (Q⊥ → ∞ there).
    zoom_lo, zoom_hi = theta_sl - 6.0, theta_sl + 6.0
    theta_zoom = np.unique(np.concatenate([
        np.linspace(zoom_lo, zoom_hi, 1600),
        theta_sl + np.concatenate([-np.geomspace(2.0e-4, 6.0, 400), np.geomspace(0.02, 6.0, 200)]),
    ]))
    theta_zoom = theta_zoom[(theta_zoom >= zoom_lo) & (theta_zoom <= zoom_hi)]
    theta_zoom = theta_zoom[np.abs(theta_zoom - theta_sl) > 1.0e-4]
    in_zoom = (theta >= zoom_lo) & (theta <= zoom_hi)

    fig, axes = plt.subplot_mosaic([["global", "global"], ["zoom", "resid"]], figsize=(11, 8.5))
    title = (
        rf"$Q_\perp$ of the PFSS dipole: Qorona vs analytic theory"
        rf" ($R_\mathrm{{seed}} = {R_SEED}\,R_\odot$, $R_S = {field.r_source}\,R_\odot$)"
    )

    glob = axes["global"]
    glob.plot(theta_dense, theory(theta_dense), color="C0", lw=2.0, label="analytic theory")
    glob.scatter(theta, q_engine, s=12, color="k", zorder=5, label="Qorona")
    glob.axvspan(zoom_lo, zoom_hi, color="C1", alpha=0.10, label="zoom window")
    for separatrix in (theta_sl, 180.0 - theta_sl):
        glob.axvline(separatrix, color="0.6", ls=":", lw=1.0)
    glob.axhline(2.0, color="0.8", lw=0.8, zorder=0)
    glob.set_yscale("log")
    glob.set_xlabel("colatitude [deg]")
    glob.set_ylabel(r"$Q_\perp$")
    glob.set_title(title)
    glob.legend(loc="upper center", framealpha=0.9, ncol=3)

    zoom = axes["zoom"]
    zoom.plot(theta_zoom, theory(theta_zoom), color="C0", lw=2.0)
    zoom.scatter(theta[in_zoom], q_engine[in_zoom], s=22, color="k", zorder=5)
    zoom.axvline(theta_sl, color="0.6", ls=":", lw=1.0)
    zoom.axhline(2.0, color="0.8", lw=0.8)
    zoom.set_yscale("log")
    zoom.set_xlim(zoom_lo, zoom_hi)
    zoom.set_ylim(1.9, 4.0 * float(q_engine[in_zoom].max()))
    zoom.set_xlabel("colatitude [deg]")
    zoom.set_ylabel(r"$Q_\perp$")
    zoom.set_title(
        rf"zoom on the separatrix $\theta_\mathrm{{SL}} = {theta_sl:.2f}^\circ$"
    )

    resid = axes["resid"]
    resid.scatter(theta, np.abs(q_engine / theory(theta) - 1.0), s=10, color="C0")
    for separatrix in (theta_sl, 180.0 - theta_sl):
        resid.axvline(separatrix, color="0.6", ls=":", lw=1.0)
    resid.set_yscale("log")
    resid.set_xlabel("colatitude [deg]")
    resid.set_ylabel("relative error")
    resid.set_title(
        r"relative error"
        r" $|Q_\perp^\mathrm{engine} - Q_\perp^\mathrm{theory}|\,/\,Q_\perp^\mathrm{theory}$"
    )

    fig.tight_layout()
    _FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_FIGURE_PATH, dpi=300)
    plt.close(fig)
    print(f"\n  figure written to {_FIGURE_PATH}")


def main() -> None:
    """Run the dipole Q⊥ validation study: comparison table, seed-invariance check, and figure."""
    field = PfssDipoleField()
    _print_comparison(field)
    # Dense sweep for the figure: a uniform base grid plus a two-sided geometric refinement that
    # walks to within 0.03 deg of each separatrix, so the divergent spikes are sampled high up
    # their flanks. Seeds that run to the cusp null are unclassified and drop out via the valid
    # mask, exactly as in the tracer/squashing gates.
    theta_sl = np.degrees(field.separatrix_colatitude(R_SEED))
    base = np.linspace(1.0, 179.0, 713)
    offsets = np.geomspace(3.0e-4, 3.0, 45)
    refined = np.concatenate(
        [center + sign * offsets for center in (theta_sl, 180.0 - theta_sl) for sign in (-1, 1)]
    )
    colatitude_deg = np.unique(np.concatenate([base, refined]))
    clear_of_separatrix = (np.abs(colatitude_deg - theta_sl) > 2.5e-4) & (
        np.abs(colatitude_deg - (180.0 - theta_sl)) > 2.5e-4
    )
    colatitude_deg = colatitude_deg[clear_of_separatrix]
    result = sweep(field, colatitude_deg)
    _make_figure(field, result, colatitude_deg)


if __name__ == "__main__":
    main()
