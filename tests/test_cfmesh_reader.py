"""Parser correctness for the CFmesh reader.

Uses a hand-written minimal mesh (one triangle extruded into two stacked prisms)
so the structural logic (state-to-cell mapping, prism-centre averaging, and
radius-based inner/outer boundary tagging) is checked without large data files.
"""

from __future__ import annotations

import numpy as np
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


def test_cfmesh_reader_parses_minimal_mesh(tmp_path):
    path = tmp_path / "minimal.CFmesh"
    path.write_text(MINIMAL_CFMESH)

    solution = read_solution(path, show_progress=False)

    # Topology and the COCONUT variable layout.
    assert solution.n_cells == 2
    assert solution.n_nodes == 9
    assert solution.variable_names == ["rho", "vx", "vy", "vz", "Bx", "By", "Bz", "p", "psi"]

    # State vectors split into the right cell-centred fields.
    np.testing.assert_array_equal(solution.magnetic_field, [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(solution.variables["rho"], [0.5, 0.6])

    # Prism centre = mean of its six nodes.
    expected_center0 = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0], [0, 2, 0], [0, 0, 2]]
    ).mean(axis=0)
    np.testing.assert_allclose(solution.cell_centers[0].to_value(u.R_sun), expected_center0)

    # Boundaries tagged by radius: Inlet (r=1) is inner, Outlet (r=3) is outer.
    inner, outer = solution.boundaries["inner"], solution.boundaries["outer"]
    assert inner.source_name == "Inlet"
    assert outer.source_name == "Outlet"
    np.testing.assert_allclose(inner.mean_radius.to_value(u.R_sun), 1.0)
    np.testing.assert_allclose(outer.mean_radius.to_value(u.R_sun), 3.0)
    np.testing.assert_array_equal(inner.adjacent_cells, [0])
    np.testing.assert_array_equal(outer.adjacent_cells, [1])
