# Supported models

Qorona is model-agnostic: each model and file format sits behind a common reader
interface. The model is inferred from the file extension; `--model` overrides the inference when
needed.

| Model | Category | Input |
|-------|----------|-------|
| [COCONUT](#coconut) | corona | COOLFluiD `.CFmesh` (also `.xz`-compressed), Tecplot `.plt` |
| [MAS](#mas) | corona | HDF4, one file per variable |
| [Icarus](#icarus) | heliosphere | VTK `.vtu` |

The [category](../heliospheric.md#the-solution-category) selects the internal shell and the render
defaults.

## COCONUT

Two formats: [COOLFluiD](https://github.com/andrealani/COOLFluiD) `.CFmesh` files (including
`.xz`-compressed ones, read directly) and Tecplot `.plt` exports. Point any command at the
solution file:

```bash
qorona info data/coconut_corona.CFmesh.xz --timestamp 2025-10-09T18:19:52
```

## MAS

[MAS](https://www.predsci.com/) (Predictive Science) distributes one HDF4 file per variable
(`rho002.hdf`, `br002.hdf`, ...). Point the CLI at any file of the set; the companion variables
are located automatically. The file name must contain one of the MAS variable tokens (`rho`,
`br`, `bt`, `bp`, `vr`, `vt`, `vp`, `t`, `p`). Public MAS solutions are available from
Predictive Science's [MHDweb](https://www.predsci.com/mhdweb/data_access.php).

```bash
qorona info br002.hdf
```

HDF4 reading relies on pyhdf, which ships in the default install; if it is missing, MAS input
is unavailable and the CLI says so with an install hint.

## Icarus

Icarus is an inner-heliosphere MHD model built on MPI-AMRVAC. Qorona reads its VTK `.vtu`
snapshots. Point any command at the `.vtu`:

```bash
qorona info data/icarus_snapshot.vtu
```

Icarus is a *heliospheric* solution. Only single-level grids are read; an adaptively refined (AMR)
file is rejected. See [Heliospheric solutions](../heliospheric.md) for the derived defaults, the
two canonical views, the Icarus run settings and the grid support.

## Adding a model

A new model or file format is added by writing a single reader against the common interface.
A contributor guide is planned.
