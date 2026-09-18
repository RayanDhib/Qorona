# Heliospheric solutions

Qorona was built on coronal solutions, whose domain is rooted at the solar surface. A
heliospheric solution is different in scale, not physics: its domain starts far above the
surface and reaches out toward 1 AU and beyond. The equations, the tracer, the Q⊥
mathematics, and the Thomson scattering are all unchanged; only the numbers baked into the
render defaults differ. [Icarus](getting-started/supported-models.md#icarus), an inner-heliosphere MHD model
built on MPI-AMRVAC, is Qorona's first heliospheric input (the model is described in Verbeke
et al. 2022, and its current version, Icarus 3.0, in Baratashvili et al. 2025).

Every command runs flag-free on a heliospheric `.vtu`:

```bash
qorona info       data/icarus_snapshot.vtu
qorona run        data/icarus_snapshot.vtu -o hcs_qperp.png
qorona build      data/icarus_snapshot.vtu -o icarus.qor
qorona wl         data/icarus_snapshot.vtu -o wl.png
qorona fieldlines data/icarus_snapshot.vtu -o lines.png
qorona qmap       icarus.qor -o hcs_map.png
```

## The solution category

Each reader declares the **category** of solution it ingests, `corona` or `heliosphere`. The
category rides the solution, is shown by `info`, and is recorded in every run's provenance.
COCONUT and MAS are coronal; Icarus is heliospheric. It is the category, not the model name,
that decides where the internal shell comes from and which set of render defaults applies, so
any future heliospheric model inherits the same handling for free.

## Defaults that follow the scale

The parameters coupled to physical scale default to the category's natural value. For a coronal
solution they resolve to the locked coronal constants; for a heliospheric solution they derive
from the solution's own radial shell (its native inner and outer boundaries). The numbers below
illustrate the heliospheric derivation on a sample whose domain runs from 21.5 to 231.5 solar
radii:

| Parameter | Coronal default | Heliospheric default | On the 21.5-231.5 sample |
|-----------|-----------------|----------------------|--------------------------|
| Internal shell (`--inner-radius` / `--outer-radius`) | 1.0 / 12.5 | the native radial extent | 21.5 / 231.5 |
| Camera field of view (`--fov`) | 12.5 | 2 x outer radius | 463 |
| Line-of-sight step (`--step`) | 0.02 | field of view / 1250 | ~0.37 |
| Render occulter (`--r-occult`, `render`/`run`) | 1.0 | inner radius | 21.5 |
| White-light occulter (`--r-occult`, `wl`) | 1.02 | 1.02 x inner radius | ~21.9 |
| Q-map shell (`--radius`, `qmap`) | 3.0 | 3 x inner radius | 64.5 |
| Camera latitude (`--latitude`) | 0 (edge-on) | 90 (equatorial plane from solar north) | 90 |
| White-light vignette (`--vignette`) | newkirk | adaptive | adaptive |

Radii are in solar radii. Explicit flags always override these, in both categories, and the
resolved number, never a placeholder, is what the run stamps on the image and writes to
provenance.

The white-light occulter sits just above the inner boundary so the empty region below the data
is covered, and the adaptive vignette self-calibrates its radial falloff to the image (the
Newkirk background is a low-corona density profile with no meaning out here). If adaptive misfits
a particularly steep falloff, `--vignette none` keeps the raw frame under a log stretch.

## The two canonical views

The heliospheric default camera looks down from solar north, so the equatorial plane fills the
frame. This is the conventional heliospheric presentation, the same top-down layout used by
solar-wind models such as ENLIL and EUHFORIA. The Parker-spiral heliospheric current sheet and
the streamer belt appear face-on, winding outward as spiral arms:

```bash
qorona run data/icarus_snapshot.vtu -o hcs_qperp.png
```

The edge-on view is one flag away, `--latitude 0`, the same projection a coronal render uses. It
compresses the current sheet into a thin band seen along the plane and is the closest analog of
an in-ecliptic imaging geometry:

```bash
qorona render icarus.qor -o hcs_edge.png --latitude 0
```

## Grid support and the AMR limitation

The reader handles a single-level separable spherical grid, whether the radial spacing is uniform
or stretched. An adaptively refined (AMR, multi-level) file is rejected at read time with a clear
error naming the limitation; reading AMR files is deferred until sample data is available. The
`.dat` native AMRVAC datfile is not read either, because its variables are stored in code units
whose normalization lives outside the file.
