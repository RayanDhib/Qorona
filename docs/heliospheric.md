# Heliospheric solutions

A heliospheric solution starts far above the solar surface and reaches 1 astronomical unit (AU) or
more. Qorona runs the same pipeline on it, with defaults that follow its scale.
[Icarus](getting-started/supported-models.md#icarus) is a supported heliospheric model.

## Quick start

Each command runs with no flags on a heliospheric solution. The examples use an Icarus snapshot:

```bash
qorona info       data/icarus_snapshot.vtu
qorona run        data/icarus_snapshot.vtu -o qperp_1au.png
qorona build      data/icarus_snapshot.vtu -o icarus.qor
qorona wl         data/icarus_snapshot.vtu -o wl.png
qorona fieldlines data/icarus_snapshot.vtu -o lines.png
qorona qmap       icarus.qor -o hcs_map.png
```

## Views

The default view is from 1 AU in the solar equatorial plane, at longitude 0 of the solution frame.
It uses the [perspective camera](camera.md#two-cameras) with a field of view of 92.16 degrees, the
width of a [PUNCH](punch.md) mosaic:

```bash
qorona wl data/icarus_snapshot.vtu -o wl_1au.png
```

For the view from Earth at the snapshot time, add `--observer earth` and the
[run settings](#icarus-run-settings) of the run:

```bash
qorona wl data/icarus_snapshot.vtu -o wl_earth.png --observer earth \
    --magnetogram-time 2025-09-10T00:04:00 --relaxation 0
```

The top-down view looks down from solar north, so the equatorial plane fills the frame. The
heliospheric current sheet and the streamer belt show face-on, as spiral arms. This view uses the
orthographic camera:

```bash
qorona run data/icarus_snapshot.vtu -o hcs_qperp.png --projection orthographic
```

`--fov` needs `--projection orthographic`. Set the width of the default view with `--fov-deg`.

## Icarus run settings

An Icarus snapshot is in the Icarus frame: a frame that turns with the Sun, with a sidereal period
of 24.47 days. The run settings give the position of Earth in this frame at the snapshot time.

Give the run settings instead of `--timestamp` and `--frame`. These features need them:
`--observer`, `wl --punch-l3`, `--export fits`, `fieldlines --export sunjson` and `export-lines`.
Qorona reads the snapshot time from the `TIME` field of the `.vtu` file. The
[Earth view](#views) above is an example.

| Flag | Value |
|------|-------|
| `--par` | The `.par` file of the run. Qorona reads `magnetogram_time`, `relaxation`, `cme_insertion` and `delta_phi`. |
| `--magnetogram-time` | The UTC time of the magnetogram of the run. Without `--par`, the CME insertion delay and `delta_phi` are 0. |
| `--relaxation` | The days of relaxation before the magnetogram time. Use 0 if the run starts at the magnetogram time. There is no default. |

- If you have the `.par` file, use `--par`. `--magnetogram-time` and `--relaxation` override its
  values.
- For a `.qor` volume, give the run settings to `build`. `render` reads them from the volume.
- The run summary shows the snapshot time. Make sure that it is the time you expect.

## The solution category

Each solution has a **category**, `corona` or `heliosphere`. COCONUT and MAS are coronal; Icarus
is heliospheric. The category sets the internal shell and the render defaults. `info` shows the
category, and each run records it in its provenance.

## Defaults that follow the scale

For a coronal solution, the parameters that depend on scale have fixed defaults. For a heliospheric
solution, they come from the inner and outer boundaries of the solution. The last column gives the
values for a sample with a domain from 21.5 to 231.5 solar radii:

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

- Radii are in solar radii.
- With `--projection orthographic`, a heliospheric solution uses `--fov` 2 x outer radius (463 on
  the sample), a step of `--fov` / 1250, and latitude 90.
- Explicit flags override these defaults for both categories.
- The run writes the values in use on the image and in the provenance.

The white-light occulter is just above the inner boundary, so it covers the empty region below the
data. At heliospheric distances:

- the default `newkirk` vignette keeps a gentle outward fade;
- `--vignette adaptive` increases the contrast of the structures;
- `--vignette none` keeps the raw frame with a log stretch.
