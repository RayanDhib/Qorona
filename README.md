![Qorona: eclipse-like synthetic imagery from coronal MHD models](https://raw.githubusercontent.com/RayanDhib/Qorona/main/assets/banner.png)

<p align="center">
  <a href="https://github.com/RayanDhib/Qorona/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/RayanDhib/Qorona/ci.yml?branch=main&label=CI" alt="CI"></a>
  <a href="https://doi.org/10.5281/zenodo.20630699"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20630699-blue" alt="DOI"></a>
  <a href="https://pypi.org/project/qorona/"><img src="https://img.shields.io/pypi/v/qorona" alt="PyPI"></a>
  <a href="https://rayandhib.github.io/Qorona/"><img src="https://img.shields.io/badge/docs-online-blue" alt="Documentation"></a>
</p>

<p align="center">
  <a href="https://rayandhib.github.io/Qorona/"><b>Documentation</b></a> ·
  <a href="https://rayandhib.github.io/Qorona/getting-started/first-eclipse-image/">Quickstart</a> ·
  <a href="https://rayandhib.github.io/Qorona/products/squashing-factor/">Products</a> ·
  <a href="https://rayandhib.github.io/Qorona/getting-started/hpc/">HPC guide</a>
</p>

Qorona turns a global coronal **MHD solution** into **eclipse-like synthetic imagery**. Its primary
product is a line-of-sight integral of the perpendicular magnetic **squashing factor Q⊥**, the
quantity that lights up the thin loops, streamers, and current sheets seen at a total solar
eclipse. The result is rendered for morphological comparison against eclipse and coronagraph
observations.

```
coronal MHD solution ──▶ read ──▶ resample ──▶ Q⊥ volume ──▶ LOS render ──▶ synthetic eclipse image
```

## Install

**With pip** (the latest release):

```bash
pip install qorona
```

**With conda** (the development version, from source):

```bash
git clone https://github.com/RayanDhib/Qorona.git
cd Qorona
conda env create -f environment.yml
conda activate qorona
```

On a cluster, install from PyPI into a plain virtual environment; the
[installation guide](https://rayandhib.github.io/Qorona/getting-started/installation/) has the
details.

## Quickstart

The quickstart data, `coconut_corona.CFmesh.xz` (~165 MB, an example COCONUT solution), is a
[release asset](https://github.com/RayanDhib/Qorona/releases/tag/v0.4.0): download it into `data/`.
The pipeline splits at its natural cost seam: build the viewpoint-independent Q⊥ volume once,
then render any number of viewpoints off it in seconds.

```bash
# 1. Inspect a solution (model, mesh, variables, boundaries), no rendering.
qorona info data/coconut_corona.CFmesh.xz --timestamp 2025-10-09T18:19:52

# 2. Build the viewpoint-independent Q⊥ volume once (the minutes-scale stage).
qorona build data/coconut_corona.CFmesh.xz -o data/coconut_corona.qor \
    --timestamp 2025-10-09T18:19:52 --outer-radius 8

# 3. Render any number of viewpoints off that volume (seconds each).
qorona render data/coconut_corona.qor -o data/eclipse.png --fov 8 --longitude 317 --latitude 6.2
```

<p align="center">
  <img src="https://raw.githubusercontent.com/RayanDhib/Qorona/main/docs/assets/eclipse.png" width="440" alt="Synthetic eclipse render of the COCONUT corona">
</p>

## How it works

Four stages behind clean interfaces: read and resample the native solution onto an internal
spherical grid; trace field lines and transport deviation vectors along them; assemble the
squashing factor Q⊥ into a viewpoint-independent volume (cached to a dependency-free `.qor`);
integrate log₁₀ Q⊥ along the line of sight on a plane-of-sky camera.

## Supported models

Qorona is model-agnostic: each coronal model and file format sits behind a common reader
interface, so the whole pipeline runs on any solution once a reader exists. Currently supported:
**COCONUT** ([COOLFluiD](https://github.com/andrealani/COOLFluiD) `.CFmesh`, Tecplot `.plt`)
and **[MAS](https://www.predsci.com/)** (HDF4). A contributor guide for adding a reader is
planned.

## Citing

If you use Qorona, please cite both the software and its accompanying paper.

**Software** (all versions), via the Zenodo concept DOI:
[10.5281/zenodo.20630699](https://doi.org/10.5281/zenodo.20630699).

**Paper** (under review at *The Astrophysical Journal*; the reference will be updated when it is
published):

> Dhib, R., Ben Ameur, F., Baratashvili, T., Jeong, H.-J., Wang, H., Noraz, Q., Schmieder, B.,
> Lani, A., & Poedts, S. *Qorona: Open, Model-Agnostic Line-of-Sight Rendering of the
> Perpendicular Squashing Factor for Eclipse-like Coronal Imaging*. Submitted to The
> Astrophysical Journal (2026).

```bibtex
@article{dhib2026qorona,
  author  = {Dhib, Rayan and Ben Ameur, Firas and Baratashvili, Tinatin and
             Jeong, Hyun-Jin and Wang, Haopeng and Noraz, Quentin and
             Schmieder, Brigitte and Lani, Andrea and Poedts, Stefaan},
  title   = {{Qorona: Open, Model-Agnostic Line-of-Sight Rendering of the
             Perpendicular Squashing Factor for Eclipse-like Coronal Imaging}},
  journal = {The Astrophysical Journal},
  year    = {2026},
  note    = {Submitted}
}
```

## License

GPL-3.0-or-later. See [`LICENSE`](https://github.com/RayanDhib/Qorona/blob/main/LICENSE).
