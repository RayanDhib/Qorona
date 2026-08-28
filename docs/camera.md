# Camera and viewpoint

`render`, `wl`, `fieldlines` and `run` all use the same camera: a plane-of-sky view, the geometry
of an eclipse photograph or a coronagraph image. The viewpoint is set in one of two ways: prescribe
it, or fetch it from a real observer's position at the observation time.

## Prescribing the viewpoint

- `--longitude`, `--latitude`: where the camera sits, in degrees, as a heliographic longitude and
  latitude in the solution's frame. It always looks at Sun centre, so this pair is the
  sub-observer point.
- `--roll`: rotation about the line of sight, in degrees counter-clockwise as the observer sees it.
  At the default 0 the image is solar-north-up.
- `--fov`: the field of view, the full width of the image, in solar radii.

```bash
qorona render data/coconut_corona.qor -o data/eclipse.png --fov 8 --longitude 317 --latitude 6.2
```

The angles are read in the solution's own heliographic frame, exactly as given. Two frames are in
use; they share the rotation axis, so latitudes agree, but not the zero meridian: in a
**Carrington** frame (a run driven by a Carrington synoptic map) `--longitude 317` is Carrington
longitude 317; in a **Stonyhurst** frame (HEEQ; the input magnetogram rotated to the Earth-facing
meridian) longitude 0 faces Earth at every time. To point at Earth by hand, print the two angles
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

When the volume carries its observation time and frame (`--timestamp` and `--frame` at `build`,
or the same flags on the render command), `--observer` resolves the viewpoint from the observer's
ephemeris and expresses it in the solution's frame, so no angles are typed:

```bash
qorona render data/corona.qor -o data/eclipse.png --fov 8 --observer earth
qorona render data/corona.qor -o data/stereo.png --fov 8 --observer stereo-a
```

- `earth` resolves offline (L0, B0).
- `stereo-a`, `solo` (Solar Orbiter) and `psp` (Parker Solar Probe) query JPL Horizons, which
  needs network access; offline, the run stops with a message naming the manual fallback.

`--observer` sets the pointing itself, so `--longitude` and `--latitude` are refused alongside it.
The observer's true distance also goes into the FITS export, and the same declared frame places
the FITS observer and the SunJSON field lines in Carrington coordinates for
[JHelioviewer](jhelioviewer.md).

## Declaring the frame

`--frame carrington|stonyhurst` (`heeq` is accepted for Stonyhurst) is declared at `build` and
baked into the volume with the timestamp; `render --frame` and `wl --frame` declare it for a
volume built without one, or override it. The run summary names the frame in use. Without a
declaration Qorona falls back to Carrington, reports `carrington frame (assumed)`, and prints one
warning whenever a frame-dependent feature runs (`--observer`, `--export fits`, `--export sunjson`,
or `export-lines`). If the frame of a solution is not known, prescribe the viewpoint by hand.
