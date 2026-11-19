# Heliospheric solutions

A heliospheric solution starts far above the solar surface and reaches out toward 1 astronomical
unit (AU) and beyond. Qorona runs the same pipeline on it, with defaults that follow its scale.
[Icarus](getting-started/supported-models.md#icarus) is a supported heliospheric model (Verbeke
et al. 2022; its current version, Icarus 3.0, in Baratashvili et al. 2025).

## Quick start

Every command runs flag-free on a heliospheric solution. The examples use an Icarus snapshot:

```bash
qorona info       data/icarus_snapshot.vtu
qorona run        data/icarus_snapshot.vtu -o hcs_qperp.png
qorona build      data/icarus_snapshot.vtu -o icarus.qor
qorona wl         data/icarus_snapshot.vtu -o wl.png
qorona fieldlines data/icarus_snapshot.vtu -o lines.png
qorona qmap       icarus.qor -o hcs_map.png
```

## The two canonical views

The default view is from 1 AU in the solar equatorial plane, as an Earth observer sees the Sun. It
uses the [perspective camera](camera.md#two-cameras) with a field of view of 92.16 degrees, the
width of a [PUNCH](punch.md) mosaic. Use this view to compare with wide-field heliospheric imagers:

```bash
qorona wl data/icarus_snapshot.vtu -o wl_1au.png
```

The top-down overview looks down from solar north, so the equatorial plane fills the frame. The
Parker-spiral heliospheric current sheet and the streamer belt appear face-on, winding outward as
spiral arms. This view uses the orthographic camera:

```bash
qorona run data/icarus_snapshot.vtu -o hcs_qperp.png --projection orthographic
```

`--fov` needs `--projection orthographic`. Set the width of the default view with `--fov-deg`.

## Icarus run settings

Some features need the run settings of an Icarus snapshot: `--observer`, `--export fits`, and the
SunJSON output of `fieldlines --export sunjson` and `export-lines`. Give `--par`, or
`--magnetogram-time` and `--relaxation`, instead of `--timestamp` and `--frame`. Qorona then reads
the snapshot time from the `TIME` field of the `.vtu` file. Add `--observer earth` for the real
position of Earth at the snapshot time.

| Icarus flag | Value |
|-------------|-------|
| `--par` | The `.par` file of the run. Qorona reads `magnetogram_time`, `relaxation`, `cme_insertion` and `delta_phi`. |
| `--magnetogram-time` | UTC time of the magnetogram of the run. Without `--par`, the CME insertion delay and `delta_phi` are 0. |
| `--relaxation` | Days of relaxation before the magnetogram time. Use 0 if the run starts at the magnetogram time. There is no default. |

- If you have the `.par` file, use `--par`. `--magnetogram-time` and `--relaxation` override its
  values.
- For a `.qor` volume, give the run settings to `build`. `render` reads them from the volume.
- The run summary shows the snapshot time. Check it against the time you expect.

The [PUNCH comparison](punch.md) uses the same run settings.

## The solution category

Each solution has a **category**, `corona` or `heliosphere`. COCONUT and MAS are coronal; Icarus
is heliospheric. The category selects the internal shell and the render defaults. `info` shows the
category, and every run records it in its provenance.

## Defaults that follow the scale

The parameters that depend on scale default to fixed values for a coronal solution. For a
heliospheric solution, they derive from the native inner and outer boundaries of the solution. The
last column gives the heliospheric values for a sample whose domain runs from 21.5 to 231.5 solar
radii:

| Parameter | Coronal default | Heliospheric default | On the 21.5-231.5 sample |
|-----------|-----------------|----------------------|--------------------------|
| Internal shell (`--inner-radius` / `--outer-radius`) | 1.0 / 12.5 | the native radial extent | 21.5 / 231.5 |
| Camera (`--projection`) | orthographic | perspective, observer at 1 AU | perspective |
| Camera field of view | `--fov 12.5` (`wl` 10) (solar radii) | `--fov-deg 92.16`, ARC | 92.16 degrees |
| Line-of-sight step (`--step`) | 0.02 | 2 x outer radius / 1250 | ~0.37 |
| Render occulter (`--r-occult`, `render`/`run`) | 1.0 | inner radius | 21.5 |
| White-light occulter (`--r-occult`, `wl`) | 1.02 | 1.02 x inner radius | ~21.9 |
| Q-map shell (`--radius`, `qmap`) | 3.0 | 3 x inner radius | 64.5 |
| Camera latitude (`--latitude`) | 0 (edge-on) | 0 (solar equatorial plane), or B0 with `--observer earth` | 0 |
| White-light vignette (`--vignette`) | newkirk (adaptive for COCONUT) | newkirk | newkirk |

Radii are in solar radii. With `--projection orthographic`, the heliospheric camera uses these
defaults: `--fov` of 2 x outer radius (463 on the sample), a step of `--fov` / 1250, and latitude
90. Explicit flags override these defaults in both categories. The run stamps the resolved values
on the image and writes them to the provenance.

The white-light occulter sits just above the inner boundary, so it covers the empty region below
the data. At heliospheric distances, the default `newkirk` vignette keeps a gentle outward fade in
the frame. `--vignette adaptive` amplifies the structure contrast. `--vignette none` keeps the raw
frame under a log stretch.

## Icarus grid support

The Icarus reader handles a single-level separable spherical grid, whether the radial spacing is
uniform or stretched. An adaptively refined (AMR, multi-level) file is rejected at read time with
an error. The native AMRVAC `.dat` file is not read.
