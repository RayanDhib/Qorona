# Camera and viewpoint

`render`, `wl`, `fieldlines` and `run` use the same camera flags. Point the camera with angles, or
take the viewpoint of a real observer.

## Point the camera

```bash
qorona render data/coconut_corona.qor -o data/eclipse.png --fov 8 --longitude 317 --latitude 6.2
```

- `--longitude`, `--latitude`: the position of the camera, in degrees, in the
  [frame of the solution](time-and-frame.md). The camera always looks at Sun centre, so this is the
  sub-observer point.
- `--roll`: the rotation about the line of sight, in degrees counter-clockwise as the observer sees
  it. At 0 (the default), solar north is up.
- `--fov`: the full width of the image, in solar radii.

## Use a real observer

```bash
qorona render data/coconut_corona.qor -o data/earth.png --fov 8 --observer earth
qorona render data/coconut_corona.qor -o data/stereo.png --fov 8 --observer stereo-a
```

`--observer` puts the camera at the position of the observer at the solution time. It needs the
[solution time](time-and-frame.md). Without `--frame`, Qorona assumes a Carrington frame and warns.

| `--observer` | Position from |
|--------------|---------------|
| `earth` | Computed offline. |
| `stereo-a`, `stereo-b`, `solo` (Solar Orbiter), `psp` (Parker Solar Probe) | JPL Horizons. Needs network access; offline, the run stops with a message. |

`--observer` sets the direction and the distance of the camera. Do not use it with `--longitude`,
`--latitude` or `--observer-distance`.

## Two cameras

| Camera | Lines of sight | Image width | Use it for | Default for |
|--------|----------------|-------------|------------|-------------|
| Orthographic | Parallel, as for an observer at infinite distance | `--fov`, solar radii | Eclipse and coronagraph views | Coronal solutions |
| Perspective | From an observer at a finite distance; each stops at the observer | `--fov-deg`, degrees | Wide fields, and an observer near or inside the solution | [Heliospheric](heliospheric.md) solutions |

Select the camera with `--projection orthographic` or `--projection perspective`:

```bash
# Parker Solar Probe-like view of a coronal solution, from 10 solar radii
qorona render data/coconut_corona.qor -o data/psp.png --projection perspective \
    --fov-deg 60 --observer-distance 0.0465
```

The perspective camera uses the ARC projection (the distance from Sun centre in the image is
proportional to the angle), or the TAN (gnomonic) projection with `--instrument punch-nfi`, `aia`,
`euvi-a`, `euvi-b` or `eui-fsi` (Calabretta & Greisen 2002). `--instrument` sets the frame of a
PUNCH or EUV instrument. See [Comparing with PUNCH](punch.md) and [EUV imaging](products/euv.md).

## All camera flags

| Flag | Camera | Unit or values | Default (coronal solution) |
|------|--------|----------------|----------------------------|
| `--longitude` | both | degrees | 0 |
| `--latitude` | both | degrees | 0 |
| `--roll` | both | degrees | 0 |
| `--observer` | both | `earth`, `stereo-a`, `stereo-b`, `solo`, `psp` | none |
| `--projection` | both | `orthographic`, `perspective` | `orthographic` (`euv`: no flag, perspective only) |
| `--fov` | orthographic | solar radii | 12.5 (`wl` 10; `euv`: no flag) |
| `--fov-deg` | perspective | degrees, below 180 | none: set it (`euv`: the field of the instrument, see [EUV imaging](products/euv.md)) |
| `--observer-distance` | perspective; orthographic for the FITS header only | AU | 1, or from `--observer` |
| `--instrument` | perspective | `punch-mosaic`, `punch-nfi`, `aia`, `euvi-a`, `euvi-b`, `eui-fsi` | none |
| `--width`, `--height` | both | pixels | 1024 |

A heliospheric solution has other defaults. See
[Defaults that follow the scale](heliospheric.md#defaults-that-follow-the-scale).

These cases stop the run with a message:

- `--fov` with the perspective camera.
- `--fov-deg` with the orthographic camera.
- `--instrument` with `--projection orthographic`.
- A perspective observer inside the inner boundary of the solution.
