# Camera and viewpoint

`render`, `wl`, `fieldlines` and `run` share the same camera settings. The viewpoint is set in one
of two ways: prescribe it, or fetch it from a real observer's position at the observation time.
Qorona has [two cameras](#two-cameras), orthographic and perspective.

## Prescribing the viewpoint

```bash
qorona render data/coconut_corona.qor -o data/eclipse.png --fov 8 --longitude 317 --latitude 6.2
```

- `--longitude`, `--latitude`: where the camera sits, in degrees, as a heliographic longitude and
  latitude in the solution's frame. It always looks at Sun centre, so this pair is the
  sub-observer point.
- `--roll`: rotation about the line of sight, in degrees counter-clockwise as the observer sees it.
  At the default 0 the image is solar-north-up.
- `--fov` (solar radii) or `--fov-deg` (degrees): the field of view, the full width of the image
  (see [Two cameras](#two-cameras)).

The angles are read in the solution's own heliographic frame. In a **Carrington** frame (a run
driven by a Carrington synoptic map), `--longitude 317` is Carrington longitude 317. In a
**Stonyhurst** frame (HEEQ; the input magnetogram rotated to the Earth-facing meridian),
longitude 0 faces Earth at every time. To point at Earth by hand, print the two angles
with sunpy, which comes with the default Qorona install:

```python
from sunpy.coordinates import sun

t = "2025-10-09T18:19:52"  # an ISO-8601 UTC string, or a compact "20251009181952"
print(sun.L0(t).deg, sun.B0(t).deg)  # 316.29 6.24
```

`L0` is the Carrington longitude of the point facing Earth and `B0` its heliographic latitude: in
a Carrington frame the pair goes straight into `--longitude` and `--latitude`; in a Stonyhurst
frame the sub-Earth longitude is 0 by definition and only `--latitude` is needed.

## Fetching the viewpoint from an observer

```bash
qorona render data/coconut_corona.qor -o data/earth.png --fov 8 --observer earth
qorona render data/coconut_corona.qor -o data/stereo.png --fov 8 --observer stereo-a
```

When the volume carries its observation time and frame (`--timestamp` and `--frame` at `build`,
or the same flags on the render command), `--observer` resolves the viewpoint from the observer's
ephemeris and expresses it in the solution's frame, so no angles are typed.

- `earth` resolves offline (L0, B0).
- `stereo-a`, `solo` (Solar Orbiter) and `psp` (Parker Solar Probe) query JPL Horizons, which
  needs network access; offline, the run stops with a message naming the manual fallback.

`--observer` sets the pointing and the distance, so the run refuses `--longitude`, `--latitude` and
`--observer-distance` with it. The FITS export records this distance for both cameras. The same
declared frame places the FITS observer and the SunJSON field lines in Carrington coordinates for
[JHelioviewer](jhelioviewer.md).

## Declaring the frame

`--frame carrington|stonyhurst` (`heeq` is accepted for Stonyhurst) is declared at `build` and
baked into the volume with the timestamp; `render --frame` and `wl --frame` declare it for a
volume built without one, or override it. The run summary names the frame in use. Without a
declaration Qorona falls back to Carrington, reports `carrington frame (assumed)`, and prints one
warning whenever a frame-dependent feature runs (`--observer`, `--export fits`, `--export sunjson`,
or `export-lines`). If the frame of a solution is not known, prescribe the viewpoint by hand.

## Two cameras

The solution's category selects the default camera: the orthographic camera for coronal
solutions, the perspective camera for [heliospheric](heliospheric.md) ones.

**Orthographic** (the coronal default). All lines of sight are parallel, as for an observer at
infinite distance. This is the geometry of an eclipse photograph or a coronagraph image. The image
is a map of the plane of sky. Its field of view is a width in solar radii (`--fov`).

**Perspective** (the heliospheric default). The lines of sight fan out from an observer at a
finite distance. The image is a map of angles from Sun centre, in degrees. This is the geometry of
a wide-field heliospheric imager. Use it for a wide field of view, or for an observer close to or
inside the solution. Each line of sight stops at the observer.

```bash
# Parker Solar Probe-like view of a coronal solution, from 10 solar radii
qorona render data/coconut_corona.qor -o data/psp.png --projection perspective \
    --fov-deg 60 --observer-distance 0.0465
```

The perspective image maps angles to pixels through a zenithal projection (Calabretta & Greisen
2002):

| Projection | Mapping | Selected by |
|------------|---------|-------------|
| ARC | The pixel distance from Sun centre is proportional to the angle. | default |
| TAN | The gnomonic projection. | `--instrument punch-nfi` |

The FITS export registers the image in helioprojective coordinates (Thompson 2006).

| Flag | Camera | Unit | Default |
|------|--------|------|---------|
| `--projection` (`orthographic` or `perspective`) | both | | corona orthographic, heliosphere perspective |
| `--fov` | orthographic | solar radii | corona 12.5 (`wl` 10), heliosphere 2 x outer radius |
| `--fov-deg` | perspective | degrees, below 180 | heliosphere 92.16; a coronal perspective view must set it |
| `--observer-distance` | perspective | AU | 1, or from `--observer` |
| `--instrument` (`punch-mosaic` or `punch-nfi`) | perspective | | none; see [Comparing with PUNCH](punch.md) |

Each width flag belongs to one camera. These flag combinations stop the run with an error message:

- `--fov` with the perspective camera.
- `--fov-deg` with the orthographic camera.
- `--instrument` with `--projection orthographic`.

The run also stops if a perspective observer is inside the solution's inner boundary.
