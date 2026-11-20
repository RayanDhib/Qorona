# Supported models

Qorona finds the model from the file extension. `--model` overrides it.

| Model | Category | Input |
|-------|----------|-------|
| [COCONUT](#coconut) | corona | COOLFluiD `.CFmesh` (also `.xz`-compressed), Tecplot `.plt` |
| [MAS](#mas) | corona | HDF4, one file per variable |
| [Icarus](#icarus) | heliosphere | VTK `.vtu` |

The [category](../heliospheric.md#the-solution-category) sets the defaults that depend on scale.
The [References](#references) give the papers of each model.

## COCONUT

COCONUT writes [COOLFluiD](https://github.com/andrealani/COOLFluiD) `.CFmesh` files and Tecplot
`.plt` exports. Qorona reads `.xz`-compressed `.CFmesh` files directly. Give the solution file to
any command:

```bash
qorona info data/coconut_corona.CFmesh.xz --timestamp 2025-10-09T18:19:52
```

## MAS

[MAS](https://www.predsci.com/) (Predictive Science) writes one HDF4 file for each variable
(`rho002.hdf`, `br002.hdf` and more). Give any file of the set. Qorona finds the other variables.

```bash
qorona info br002.hdf
```

- The file name must contain a MAS variable token: `rho`, `br`, `bt`, `bp`, `vr`, `vt`, `vp`, `t`
  or `p`.
- Public MAS solutions are on [MHDweb](https://www.predsci.com/mhdweb/data_access.php).
- MAS input needs pyhdf, which is part of the default install. If pyhdf is missing, the CLI tells
  you how to install it.

## Icarus

Icarus is a heliospheric MHD model built on MPI-AMRVAC. Qorona reads its VTK `.vtu` snapshots:

```bash
qorona info data/icarus_snapshot.vtu
```

- Qorona reads single-level grids, with uniform or stretched radial spacing. It stops with an error
  on an adaptively refined (AMR) file.
- Qorona does not read the native AMRVAC `.dat` file.
- For the views, the run settings (`--par`, `--magnetogram-time`, `--relaxation`) and the defaults
  of a heliospheric solution, see [Heliospheric solutions](../heliospheric.md).

## Adding a model

A new model or file format needs one new reader for the common interface. A contributor guide is
planned.

## References

**COCONUT**

- Perri, B., et al. 2022, *The Astrophysical Journal*, 936, 19. The model description, with
  polytropic solutions. [doi:10.3847/1538-4357/ac7237](https://doi.org/10.3847/1538-4357/ac7237)
- Baratashvili, T., et al. 2024, *Astronomy & Astrophysics*, 690, A184. The full MHD model,
  coupled to Icarus from the Sun to Earth.
  [doi:10.1051/0004-6361/202449389](https://doi.org/10.1051/0004-6361/202449389)
- Wang, H., et al. 2025, *Astronomy & Astrophysics*, 694, A234. The time-evolving corona.
  [doi:10.1051/0004-6361/202452279](https://doi.org/10.1051/0004-6361/202452279)

**MAS**

- Mikić, Z., et al. 1999, *Physics of Plasmas*, 6, 2217. The global corona model.
  [doi:10.1063/1.873474](https://doi.org/10.1063/1.873474)
- Linker, J. A., et al. 1999, *Journal of Geophysical Research*, 104, 9809. The corona during Whole
  Sun Month. [doi:10.1029/1998JA900159](https://doi.org/10.1029/1998JA900159)

**Icarus**

- Verbeke, C., et al. 2022, *Astronomy & Astrophysics*, 662, A50. The model description.
  [doi:10.1051/0004-6361/202141981](https://doi.org/10.1051/0004-6361/202141981)
- Baratashvili, T., et al. 2025, *Astronomy & Astrophysics*, 694, A306. Icarus 3.0, the current
  version. [doi:10.1051/0004-6361/202452705](https://doi.org/10.1051/0004-6361/202452705)
