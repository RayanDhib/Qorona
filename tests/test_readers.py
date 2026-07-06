"""Parser correctness for the solution readers (CFmesh, Tecplot, and the MAS HDF4 quartet).

Both COCONUT formats are checked on the same hand-written minimal mesh (one triangle extruded
into two stacked prisms, shells at r = 1, 2, 3) so the shared contract (state-to-cell mapping,
prism-centre averaging, the canonical variable layout, and inner/outer boundary tagging) is
verified without large data files, alongside each format's own provenance. The MAS reader is
checked on a synthetic staggered file set carrying an analytic dipole, covering sibling
discovery, mesh alignment, the spherical-to-Cartesian conversion, and the structured resample.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from qorona import read_solution

# Two prisms sharing the middle triangle: shells at r = 1, 2, 3.
MINIMAL_CFMESH = """\
!COOLFLUID_VERSION 2013.9
!NB_DIM 3
!NB_EQ 9
!NB_NODES 9 0
!NB_STATES 2 0
!NB_ELEM 2
!NB_ELEM_TYPES 1
!GEOM_POLYORDER 1
!SOL_POLYORDER 0
!ELEM_TYPES Prism
!NB_NODES_PER_TYPE 6
!NB_STATES_PER_TYPE 1
!LIST_ELEM
0 1 2 3 4 5 0
3 4 5 6 7 8 1
!NB_TRSs 2
!TRS_NAME Inlet
!NB_GEOM_ENTS 1
!LIST_GEOM_ENT
3 1 0 1 2 0
!TRS_NAME Outlet
!NB_GEOM_ENTS 1
!LIST_GEOM_ENT
3 1 6 7 8 1
!EXTRA_VARS
!LIST_NODE
1 0 0
0 1 0
0 0 1
2 0 0
0 2 0
0 0 2
3 0 0
0 3 0
0 0 3
!LIST_STATE 1
0.5 0.0 0.0 0.0 1.0 2.0 3.0 0.10 0.0
0.6 0.0 0.0 0.0 4.0 5.0 6.0 0.20 0.0
!END
"""

# The same mesh as a COOLFluiD Tecplot export: BLOCK packing (each variable a contiguous block,
# nodal coords 9 long and cell-centred fields 2 long), then the degenerate FEBRICK connectivity
# (prism nodes a,b,c repeated apex c, then d,e,f repeated apex f).
MINIMAL_TECPLOT = """\
TITLE = "minimal mesh"
VARIABLES = "x0" "x1" "x2" "rho" "u" "v" "w" "Bx" "By" "Bz" "p" "phi"
ZONE N=9, E=2, ZONETYPE=FEBRICK, DATAPACKING=BLOCK, VARLOCATION=([1-3]=NODAL,[4-12]=CELLCENTERED)
1 0 0 2 0 0 3 0 0
0 1 0 0 2 0 0 3 0
0 0 1 0 0 2 0 0 3
0.5 0.6
0 0
0 0
0 0
1 4
2 5
3 6
0.10 0.20
0 0
1 2 3 3 4 5 6 6
4 5 6 6 7 8 9 9
"""

CANONICAL_VARIABLES = ["rho", "vx", "vy", "vz", "Bx", "By", "Bz", "p", "psi"]


def test_readers_parse_minimal_mesh(tmp_path):
    cfmesh_path = tmp_path / "minimal.CFmesh"
    cfmesh_path.write_text(MINIMAL_CFMESH)
    tecplot_path = tmp_path / "minimal.plt"
    tecplot_path.write_text(MINIMAL_TECPLOT)

    cfmesh = read_solution(cfmesh_path, show_progress=False)
    tecplot = read_solution(tecplot_path, show_progress=False)

    # Prism centre = mean of its six nodes (shared by both formats).
    expected_center0 = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0], [0, 2, 0], [0, 0, 2]]
    ).mean(axis=0)

    # The shared NativeSolution contract: topology, canonical layout, fields, centres, shell radii.
    for solution in (cfmesh, tecplot):
        assert solution.n_cells == 2
        assert solution.n_nodes == 9
        assert solution.variable_names == CANONICAL_VARIABLES
        np.testing.assert_array_equal(solution.magnetic_field, [[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(solution.variables["rho"], [0.5, 0.6])
        np.testing.assert_allclose(solution.cell_centers[0].to_value(u.R_sun), expected_center0)
        inner, outer = solution.boundaries["inner"], solution.boundaries["outer"]
        np.testing.assert_allclose(inner.mean_radius.to_value(u.R_sun), 1.0)
        np.testing.assert_allclose(outer.mean_radius.to_value(u.R_sun), 3.0)

    # CFmesh carries tagged surfaces with their source names and adjacent cells.
    assert cfmesh.metadata.file_format == "cfmesh"
    assert cfmesh.boundaries["inner"].source_name == "Inlet"
    assert cfmesh.boundaries["outer"].source_name == "Outlet"
    np.testing.assert_array_equal(cfmesh.boundaries["inner"].adjacent_cells, [0])

    # Tecplot has no tagged surfaces; inner/outer are synthesised from the node radial extremes.
    assert tecplot.metadata.file_format == "tecplot"
    assert "synthesized" in tecplot.boundaries["inner"].source_name


def test_mas_reader_and_structured_resample(tmp_path):
    sd = pytest.importorskip("pyhdf.SD")

    # A dipole in spherical components on staggered per-variable axes (file order phi, theta, r),
    # with the real files' pole overhang; rho = r^-2. The radial axis is the finest: the
    # staggered alignment extrapolates half a cell at the shell edges, where 1/r^3 curves most.
    r_main = np.linspace(1.0, 2.0, 41)
    t_main = np.linspace(-0.05, np.pi + 0.05, 25)
    p_main = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)
    stagger = {
        "rho": (0.0, 0.0, 0.0),
        "br": (0.5, 0.0, 0.0),
        "bt": (0.0, 0.5, 0.0),
        "bp": (0.0, 0.0, 0.5),
    }

    def component(name, rr, tt):
        if name == "br":
            return 2.0 * np.cos(tt) / rr**3
        if name == "bt":
            return np.sin(tt) / rr**3
        if name == "bp":
            return np.zeros_like(rr)
        return 1.0 / rr**2

    for name, (dr, dt, dp) in stagger.items():
        r = r_main + dr * (r_main[1] - r_main[0])
        t = t_main + dt * (t_main[1] - t_main[0])
        p = p_main + dp * (p_main[1] - p_main[0])
        rr, tt, _ = np.meshgrid(r, t, p, indexing="ij")
        values = component(name, rr, tt).transpose(2, 1, 0)
        store = sd.SD(str(tmp_path / f"{name}002.hdf"), sd.SDC.WRITE | sd.SDC.CREATE)
        dataset = store.create("Data-Set-2", sd.SDC.FLOAT64, values.shape)
        for i, axis in enumerate((p, t, r)):
            dataset.dim(i).setscale(sd.SDC.FLOAT64, axis.tolist())
        dataset[:] = values
        store.end()

    solution = read_solution(tmp_path / "br002.hdf", show_progress=False)

    # Identity, sibling discovery from a B anchor, canonical variables, structured axes.
    assert solution.metadata.model == "mas"
    assert solution.variable_names == ["rho", "Bx", "By", "Bz"]
    assert solution.structured is not None
    assert solution.n_cells == 41 * 25 * 36

    # Spherical-to-Cartesian against the closed-form dipole B = (3 (z_hat . r_hat) r_hat - z_hat)
    # / r^3 (staggering alignment is linear, hence the loose tolerance).
    points = solution.cell_centers.to_value(u.R_sun)
    radius = np.linalg.norm(points, axis=1, keepdims=True)
    unit = points / radius
    expected = (3.0 * unit[:, 2:3] * unit - np.array([0.0, 0.0, 1.0])) / radius**3
    np.testing.assert_allclose(solution.magnetic_field, expected, atol=1.0e-2)
    np.testing.assert_allclose(solution.variables["rho"], 1.0 / radius[:, 0] ** 2, rtol=5.0e-3)

    # Boundaries synthesized from the radial extremes (Tecplot precedent).
    assert "synthesized" in solution.boundaries["inner"].source_name
    np.testing.assert_allclose(solution.boundaries["inner"].mean_radius.to_value(u.R_sun), 1.0)
    np.testing.assert_allclose(solution.boundaries["outer"].mean_radius.to_value(u.R_sun), 2.0)

    # The structured resampler reproduces the analytic field on the internal grid.
    from qorona.resample import LogarithmicSpacing, SphericalGrid, StructuredGridResampler

    grid = SphericalGrid(LogarithmicSpacing(1.05, 1.9), n_r=12, n_theta=18, n_phi=24)
    resampled = StructuredGridResampler().resample(
        solution, grid, ("Bx", "By", "Bz"), show_progress=False
    )
    nodes = grid.node_points().reshape(-1, 3)
    radius = np.linalg.norm(nodes, axis=1, keepdims=True)
    unit = nodes / radius
    expected = (3.0 * unit[:, 2:3] * unit - np.array([0.0, 0.0, 1.0])) / radius**3
    sampled = np.stack([resampled[name].ravel() for name in ("Bx", "By", "Bz")], axis=-1)
    np.testing.assert_allclose(sampled, expected, atol=2.0e-2)
