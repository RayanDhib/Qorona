"""CLI smoke test: the build -> .qor -> render spine runs end to end, the two-level help wiring
holds, and a non-artifact is rejected with a clean error instead of a traceback.

The numerical kernels have their own analytic tests; this guards the command wiring (argument
parsing, help levels, config assembly, volume save/load, PNG output) they never touch. It runs on
the tiny hand-written mesh from :mod:`test_readers`, with the cheapest grid, the nearest-cell
resampler (the k-NN default needs more cells than the mesh has), the ``fast`` quality preset, and
a 64x64 image, pinned to ``--device cpu`` so it is deterministic and GPU-free.
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner
from test_readers import MINIMAL_CFMESH

from qorona.cli.main import main


def test_cli_build_render_roundtrip(tmp_path: Path) -> None:
    mesh = tmp_path / "minimal.CFmesh"
    mesh.write_text(MINIMAL_CFMESH)
    volume = tmp_path / "mini.qor"
    image = tmp_path / "mini.png"
    runner = CliRunner()

    # The two-level help: the default view keeps the engine knobs out and points to --help-all;
    # the full view shows them under their pipeline-stage section. Hand-rolled formatter wiring
    # that click no longer guarantees, so it is guarded here with the rest of the CLI plumbing.
    curated = runner.invoke(main, ["run", "-h"])
    assert curated.exit_code == 0, curated.output
    assert "--rtol" not in curated.output
    assert "--help-all" in curated.output
    full = runner.invoke(main, ["run", "--help-all"])
    assert full.exit_code == 0, full.output
    assert "--rtol" in full.output
    assert "Volume:" in full.output

    build = runner.invoke(
        main,
        [
            "build",
            str(mesh),
            "-o",
            str(volume),
            "--timestamp",
            "2024-01-01T00:00:00",
            "--resampler",
            "nearest-cell",
            "--inner-radius",
            "1",
            "--outer-radius",
            "2.5",
            "--n-r",
            "6",
            "--n-theta",
            "6",
            "--n-phi",
            "12",
            "--quality",
            "fast",
            "--device",
            "cpu",
            "--quiet",
        ],
    )
    assert build.exit_code == 0, build.output
    assert volume.exists()

    render = runner.invoke(
        main,
        [
            "render",
            str(volume),
            "-o",
            str(image),
            "--fov",
            "2.5",
            "--longitude",
            "0",
            "--width",
            "64",
            "--height",
            "64",
            "--quiet",
        ],
    )
    assert render.exit_code == 0, render.output
    assert image.exists()

    # The composite mode (disk filled with its near-side structure) shares the wiring but takes
    # the opaque integration path and the layered finalize; guard it end to end too.
    composite_image = tmp_path / "mini_composite.png"
    composite = runner.invoke(
        main,
        [
            "render",
            str(volume),
            "-o",
            str(composite_image),
            "--fov",
            "2.5",
            "--occult",
            "composite",
            "--width",
            "64",
            "--height",
            "64",
            "--quiet",
        ],
    )
    assert composite.exit_code == 0, composite.output
    assert composite_image.exists()

    # A non-artifact (the mesh fed where a .qor is expected) is rejected as a clean ClickException,
    # not a traceback: SystemExit from click, with the loader's message reaching the user.
    rejected = runner.invoke(
        main,
        [
            "render",
            str(mesh),
            "-o",
            str(image),
            "--fov",
            "2.5",
            "--longitude",
            "0",
            "--quiet",
        ],
    )
    assert rejected.exit_code != 0
    assert isinstance(rejected.exception, SystemExit)
    assert "not a" in rejected.output.lower()
