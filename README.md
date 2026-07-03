![Qorona: eclipse-like synthetic imagery from coronal MHD models](https://raw.githubusercontent.com/RayanDhib/Qorona/main/assets/banner.png)

<p align="center">
  <a href="https://github.com/RayanDhib/Qorona/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/RayanDhib/Qorona/ci.yml?branch=main&label=CI" alt="CI"></a>
  <a href="https://doi.org/10.5281/zenodo.20630699"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20630699-blue" alt="DOI"></a>
  <a href="https://pypi.org/project/qorona/"><img src="https://img.shields.io/pypi/v/qorona" alt="PyPI"></a>
</p>

Qorona turns a global coronal **MHD solution** into **eclipse-like synthetic imagery**. Its primary
product is a line-of-sight integral of the magnetic **squashing factor Q⊥**, the quantity that lights
up the thin loops, streamers, and current sheets seen at a total solar eclipse. The result is
rendered for morphological comparison against eclipse and coronagraph observations.

```
coronal MHD solution ──▶ read ──▶ resample ──▶ Q⊥ volume ──▶ LOS render ──▶ synthetic eclipse image
```

## Install

**With pip** you install the latest release; it is the quickest way to get the `qorona` command:

```bash
pip install qorona
```

**With conda** you clone the repository and build the provided environment from its root, giving you
the development version (`main`) as an editable checkout, e.g. to develop or add a model reader:

```bash
git clone https://github.com/RayanDhib/Qorona.git
cd Qorona
conda env create -f environment.yml
conda activate qorona
```

## On a cluster

Qorona installs from PyPI into a plain virtual environment; no conda needed:

```bash
module purge            # optional; stops other modules leaking packages via PYTHONPATH
module load python      # any Python ≥ 3.11
python -m venv ~/envs/qorona && source ~/envs/qorona/bin/activate
pip install qorona
```

On GPU nodes, load the site's CUDA module (`module load cuda`) so numba finds the toolkit;
`--device gpu` errors loudly if the GPU is unusable rather than silently falling back. Two flags
are made for batch jobs: `--workers` pins the kernel threads to your allocation (numba otherwise
takes every core it sees), and `--quiet` keeps job logs readable:

```bash
qorona build <solution> -o <solution>.qor --device gpu --workers $SLURM_CPUS_PER_TASK --quiet
```

## Example data

The quickstart uses `hmi_lmax50.CFmesh.xz` (~165 MB), an HMI-driven COCONUT corona MHD solution.
It is distributed as a [release asset](https://github.com/RayanDhib/Qorona/releases/tag/v0.1.0), not committed to the repo: download it
into `data/` before running the commands below. Qorona reads the compressed `.xz` directly, with no
manual decompression step.

## Quickstart

Qorona splits the pipeline at its natural cost seam: the **Q⊥ volume is expensive and
viewpoint-independent** (bake it once), while a **render off that volume is cheap** (do it for many
cameras). Three commands follow from that:

```bash
# 1. Inspect a solution (model, mesh, variables, boundaries), no rendering.
qorona info data/hmi_lmax50.CFmesh.xz --timestamp 2025-10-09T18:19:52

# 2. Bake the viewpoint-independent Q⊥ volume once (the minutes-scale stage).
qorona build data/hmi_lmax50.CFmesh.xz -o data/hmi_lmax50.qor \
    --timestamp 2025-10-09T18:19:52 --outer-radius 8

# 3. Render any number of viewpoints off that volume (seconds each).
qorona render data/hmi_lmax50.qor -o data/eclipse.png --fov 8 --longitude 317 --latitude 6.2
qorona render data/hmi_lmax50.qor -o data/polarity.png --fov 8 --longitude 317 --latitude 6.2 --polarity-mode hue
qorona render data/hmi_lmax50.qor -o data/sun.png --fov 3 --longitude 317 --latitude 6.2 --occult opaque --preset small-fov --step 0.002
qorona render data/hmi_lmax50.qor -o data/composite.png --fov 8 --longitude 317 --latitude 6.2 --occult composite
```

`--quality` picks the baked volume's resolution: `fast` for a quick preview, `standard` (the
default), or `high` for the finest structure (timings below).

Or do it all in one shot:

```bash
qorona run data/hmi_lmax50.CFmesh.xz -o data/eclipse.png \
    --timestamp 2025-10-09T18:19:52 --fov 8 --longitude 317 --save-volume data/hmi_lmax50.qor
```

For a field-line view:

```bash
qorona fieldlines data/hmi_lmax50.CFmesh.xz -o data/fieldlines.png --fov 8 --longitude 317
```

For a Q-map (a longitude/latitude shell of signed-log Q⊥ at a fixed radius, the
viewpoint-independent sibling of `render`), bake with the outer radius at the map radius:

```bash
qorona build data/hmi_lmax50.CFmesh.xz -o data/hmi_lmax50_r3.qor \
    --timestamp 2025-10-09T18:19:52 --outer-radius 3
qorona qmap data/hmi_lmax50_r3.qor -o data/qmap.png --radius 3
```

Every command prints a polished end-of-run summary of its parameters and metrics, and the rendered
PNG carries a corner stamp (CR · timestamp · sub-observer angles · roll · FOV) for reproducibility. Run
`qorona <command> --help` for the common options, or `--help-all` for the complete flag reference
(grid resolution, builder, engine tolerances, and more); the defaults reproduce the published
whole-corona Q⊥ render.

## GPU acceleration

Volume builds are CUDA-accelerated end to end, with no extra Qorona dependency: the kernels ride
the default-install numba and activate whenever it sees a CUDA-capable NVIDIA GPU (driver + CUDA
toolkit). Nothing to configure: `qorona build` already uses the GPU when one is present.

```bash
qorona build ... --device gpu                       # force the GPU (errors if none is usable)
qorona build ... --device cpu                       # multi-core CPU path (the reference)
qorona build ... --device gpu --precision float64   # all-double reference (~2× slower than mixed)
```

- `--device auto` (default) picks the GPU when present; the CPU kernels remain the reference
  implementation and produce the same images.
- `--precision mixed` (default) runs the field interpolation in float32 and everything else in
  float64, log-invisible against the `float64` reference. `float32` is an experimental fully-float32
  paint variant. GPU-only knob; the CPU path always computes in float64.
- Device memory adapts to free VRAM (the Q⊥ accumulation tiles itself), so the same command runs
  on small cards and at very high resolutions alike.
- The resolved backend and precision are stamped into the volume's provenance and the end-of-run
  summary.

Indicative volume-build timings (RTX 4080 vs 32-core CPU, mixed precision; not a benchmark):

| Q⊥ volume                                                | GPU    | CPU     |
|----------------------------------------------------------|--------|---------|
| `--quality standard` (default), 384×360×720 (100 M vox)  | ~85 s  | ~9 min  |
| `--quality high`, 576×540×1080 (336 M vox)               | ~3 min | ~22 min |

## How it works

The pipeline processes a single MHD snapshot through four stages, each isolated behind a clean
interface so a new input model is added by writing one reader and a new viewpoint costs only a
render:

1. **Read & resample** the native solution onto an internal regular spherical grid.
2. **Trace** magnetic field lines and **transport** deviation vectors along them.
3. **Squashing factor**: assemble Q⊥ boundary-to-boundary and bake it into a viewpoint-independent
   volume (cached to a dependency-free `.qor`).
4. **Render**: integrate log₁₀ Q⊥ along the line of sight on an orthographic plane-of-sky camera,
   with depth colouring and eclipse occultation.

The reference output is the line-of-sight squashing-factor render of the corona's fine structure.

## Supported models

Qorona is model-agnostic: each coronal model and file format sits behind a common reader interface,
so the whole pipeline runs on any solution once a reader exists.

**Currently supported:** COCONUT (COOLFluiD `.CFmesh`).

Support for other coronal MHD models can be added by writing a single reader against that interface.
A contributor guide is planned.

## License

GPL-3.0-or-later. See [`LICENSE`](https://github.com/RayanDhib/Qorona/blob/main/LICENSE).
