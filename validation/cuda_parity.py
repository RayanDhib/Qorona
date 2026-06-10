"""Validation study: GPU-vs-CPU per-line Q⊥ parity for the CUDA backend (report-only).

A differential diagnostic that runs :func:`~qorona.squashing.compute_squashing` over the same seeds
on the float64 CPU reference (``device="cpu"``) and on **both** GPU precisions (``device="gpu"``
with the explicit ``precision`` argument set to ``float64`` then ``mixed``), printing each GPU
kernel's per-line Q⊥ agreement distribution against that reference over four cases.
The float64 column shows the FMA / decision-divergence tail; the **mixed** column shows the
mixed-mode tolerance (the float32 tricubic noise floor), which shows up only on the gridded
cases (b, c) since the dipole (a, d) is the float64 closed form in both GPU kernels. The four cases:

    (a) the analytic PFSS dipole, transport + position-only colatitude sweep;
    (b) a gridded :class:`~qorona.field.SampledField` (the axisymmetric dipole sampled on the
        internal mesh), the production interpolation path;
    (c) an asymmetric gridded field (a tilted dipole, genuinely φ-dependent), stressing the
        ∇B̂ contraction the transport relies on;
    (d) a near-separatrix / grazing colatitude band on the dipole, the decision-divergence
        regime where the FMA-rounding tail shows up.

This is **report-only**: it prints the distribution (n valid, max rel, p99 rel, count above
1e-6, and the three worst lines) and asserts **nothing**. The crisp correctness gate lives in
``tests/test_dipole_squashing.py``, and the analytic-dipole theory agreement in
``validation/dipole_q_perp.py``; this study merely characterizes how closely the two
backends agree line-by-line, including the small, expected ``>1e-6`` tail on the grazing band
that the GPU's FMA contraction produces.

Only Q⊥ is compared: the integration kernels do not expose per-line step counts.

Run on demand (writes ``validation/figures/cuda_parity.png`` when matplotlib is present):

    python validation/cuda_parity.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from qorona.field import PfssDipoleField, SampledField
from qorona.geometry import spherical_to_cartesian
from qorona.resample.grid import LogarithmicSpacing, SphericalGrid, pad_field
from qorona.squashing import compute_squashing

#: Seed radius (just above the inner boundary), matching the dipole study.
R_SEED = 1.01
#: Small internal mesh for the gridded cases (cheap, but resolves the closed/open structure).
_GRID = SphericalGrid(LogarithmicSpacing(1.0, 2.5), n_r=16, n_theta=48, n_phi=96)
_FIGURE_PATH = Path(__file__).parent / "figures" / "cuda_parity.png"


def _seeds(colatitude_deg: np.ndarray, *, azimuth: float = 0.7) -> np.ndarray:
    """Return Cartesian seeds at ``R_SEED`` for the given colatitudes (degrees), at one azimuth."""
    colatitude = np.deg2rad(np.asarray(colatitude_deg, dtype=np.float64))
    spherical = np.stack(
        [np.full_like(colatitude, R_SEED), colatitude, np.full_like(colatitude, azimuth)], axis=-1
    )
    return spherical_to_cartesian(spherical)


def _seeds_multi_azimuth(colatitude_deg: np.ndarray, azimuths: np.ndarray) -> np.ndarray:
    """Return seeds across the outer product of ``colatitude_deg`` and ``azimuths``."""
    return np.concatenate([_seeds(colatitude_deg, azimuth=float(az)) for az in azimuths], axis=0)


def _gridded_dipole() -> SampledField:
    """Sample the axisymmetric PFSS dipole onto :data:`_GRID` (the production path)."""
    nodes = _GRID.node_points()
    b_nodes = PfssDipoleField().sample(nodes.reshape(-1, 3), gradient=False).b
    b_nodes = b_nodes.reshape((_GRID.n_r, _GRID.n_theta, _GRID.n_phi, 3))
    return SampledField(_GRID, pad_field(b_nodes), normalization="test")


def _tilted_dipole(tilt_deg: float = 30.0) -> SampledField:
    """Sample a dipole tilted by ``tilt_deg`` about the y-axis onto :data:`_GRID`.

    Rotating the dipole off the polar axis makes the field genuinely φ-dependent (it is no longer
    axisymmetric), which exercises the full ∇B̂ contraction in the deviation-vector transport.
    Points are rotated into the dipole frame with ``P @ R`` (row-wise ``Rᵀ·p``), the dipole B is
    evaluated there, and the resulting vector is rotated back with ``R @ b``.
    """
    angle = np.deg2rad(tilt_deg)
    cos, sin = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]], dtype=np.float64)
    nodes = _GRID.node_points().reshape(-1, 3)
    b_dipole_frame = PfssDipoleField().sample(nodes @ rotation, gradient=False).b
    b_tilt = (rotation @ b_dipole_frame[..., None])[..., 0]
    b_tilt = b_tilt.reshape((_GRID.n_r, _GRID.n_theta, _GRID.n_phi, 3))
    return SampledField(_GRID, pad_field(b_tilt), normalization="test")


def _q_perp(
    field: object, seeds: np.ndarray, device: str, precision: str = "float64"
) -> np.ndarray:
    """Return the per-line Q⊥ for ``seeds`` on ``field`` using the given backend and precision.

    ``precision`` selects the CUDA kernel (``"float64"`` all-double or ``"mixed"`` float32
    tricubic) for the ``device="gpu"`` path; it is inert on the CPU tiers.
    """
    result = compute_squashing(
        field,  # type: ignore[arg-type]
        seeds, rtol=1e-6, device=device, precision=precision, show_progress=False,
    )
    return result.q_perp


def _report(name: str, q_gpu: np.ndarray, q_cpu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Print the GPU-vs-CPU per-line Q⊥ agreement distribution for one case.

    Returns the per-line ``(rel, valid)`` arrays; :func:`_make_figure` reuses ``rel``.
    ``rel = |q_gpu/q_cpu - 1|`` over lines where both backends are finite and the CPU
    value is non-zero.
    """
    valid = np.isfinite(q_gpu) & np.isfinite(q_cpu) & (q_cpu != 0.0)
    rel = np.abs(q_gpu[valid] / q_cpu[valid] - 1.0)
    if rel.size == 0:
        print(f"\n  [{name}]  n=0  (no lines valid on both backends)")
        return rel, valid
    over = int((rel > 1e-6).sum())
    order = np.argsort(rel)
    print(
        f"\n  [{name}]  n={int(valid.sum())}  max={rel.max():.2e}  "
        f"p99={np.percentile(rel, 99):.2e}  >1e-6: {over}"
    )
    q_gpu_valid = q_gpu[valid]
    q_cpu_valid = q_cpu[valid]
    for w in order[-3:][::-1]:
        print(
            f"           worst: Q⊥_gpu={q_gpu_valid[w]:.6g} "
            f"Q⊥_cpu={q_cpu_valid[w]:.6g} rel={rel[w]:.2e}"
        )
    return rel, valid


def _make_figure(distributions: dict[str, np.ndarray]) -> None:
    """Write a per-case rel-error histogram. Skips with a note if matplotlib is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib not available - skipping figure (numerical tables above stand).")
        return

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    floor = 1e-18
    for name, rel in distributions.items():
        finite = rel[np.isfinite(rel)]
        if finite.size == 0:
            continue
        ax.hist(
            np.log10(np.maximum(finite, floor)),
            bins=40,
            histtype="step",
            lw=1.8,
            label=f"{name} (n={finite.size})",
        )
    ax.axvline(-6.0, color="0.6", ls=":", lw=1.0, label="1e-6 threshold")
    ax.set_xlabel("log₁₀ |Q⊥_gpu / Q⊥_cpu - 1|")
    ax.set_ylabel("line count")
    ax.set_title("CUDA backend: GPU-vs-CPU per-line Q⊥ relative agreement")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    _FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_FIGURE_PATH, dpi=140)
    plt.close(fig)
    print(f"\n  figure written to {_FIGURE_PATH}")


def main() -> None:
    """Run the four parity cases in both GPU precisions vs the CPU reference; print distributions.

    Each case is reported twice against the float64 CPU reference: the **float64** GPU kernel (the
    FMA / decision-divergence tail) and the **mixed** GPU kernel (the float32 tricubic noise
    floor). The dipole cases (a, d) are ``kind == 1`` (closed-form, float64 in both
    GPU kernels), so their mixed tail matches their float64 tail; the gridded cases (b, c) ride the
    tricubic, so their mixed tail re-derives the mixed-mode Q⊥ agreement tolerance.
    """
    dipole = PfssDipoleField()
    theta_sl = float(np.degrees(dipole.separatrix_colatitude(R_SEED)))
    print(f"\n  theta_SL(R_seed) = {theta_sl:.4f} deg  (closed band in between)")
    distributions: dict[str, np.ndarray] = {}

    def run_case(key: str, name: str, field: object, seeds: np.ndarray) -> None:
        """Report float64-GPU and mixed-GPU agreement vs the CPU reference for one case."""
        q_cpu = _q_perp(field, seeds, "cpu")
        rel_f64, _ = _report(f"{name} [f64]", _q_perp(field, seeds, "gpu", "float64"), q_cpu)
        rel_mix, _ = _report(f"{name} [mixed]", _q_perp(field, seeds, "gpu", "mixed"), q_cpu)
        distributions[f"{key} [f64]"] = rel_f64
        distributions[f"{key} [mixed]"] = rel_mix

    # (a) analytic dipole, colatitude sweep clear of the separatrices.
    sweep = np.linspace(5.0, 175.0, 171)
    clear = (np.abs(sweep - theta_sl) > 1.0) & (np.abs(sweep - (180.0 - theta_sl)) > 1.0)
    sweep_clear = sweep[clear]
    run_case("a: dipole", "a: dipole (analytic)", dipole, _seeds(sweep_clear))

    # (b) gridded axisymmetric dipole (the production interpolation path).
    run_case("b: gridded", "b: gridded dipole", _gridded_dipole(), _seeds(sweep_clear))

    # (c) asymmetric tilted dipole on the grid, seeded across azimuths to sample the φ-dependence.
    azimuths = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    colat_c = np.linspace(20.0, 160.0, 21)
    run_case("c: tilted", "c: tilted dipole (asym)", _tilted_dipole(),
             _seeds_multi_azimuth(colat_c, azimuths))

    # (d) near-separatrix / grazing band on the analytic dipole (the decision-divergence regime),
    # dropping a hair either side of each separatrix where seeds run to the cusp null.
    north = np.linspace(theta_sl - 2.0, theta_sl + 2.0, 81)
    south = np.linspace(180.0 - theta_sl - 2.0, 180.0 - theta_sl + 2.0, 81)
    north = north[np.abs(north - theta_sl) > 0.05]
    south = south[np.abs(south - (180.0 - theta_sl)) > 0.05]
    run_case("d: near-sep", "d: near-separatrix", dipole, _seeds(np.concatenate([north, south])))

    _make_figure(distributions)


if __name__ == "__main__":
    main()
