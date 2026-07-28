# Camera and viewpoint

`render`, `wl`, `fieldlines` and `run` all use the same camera: a plane-of-sky view, the geometry
of an eclipse photograph or a coronagraph image. Four flags set it, with the same meaning on every
one of those commands:

- `--longitude`, `--latitude`: where the camera sits, in degrees, as a heliographic longitude and
  latitude in the solution's frame. It always looks at Sun centre, so this pair is the
  sub-observer point.
- `--roll`: rotation about the line of sight, in degrees counter-clockwise as the observer sees it.
  At the default 0 the image is solar-north-up.
- `--fov`: the field of view, the full width of the image, in solar radii.

## Which frame the angles are in

Qorona reads the sub-observer angles in the solution's own heliographic frame, and it cannot tell
from the file which frame that is. A run driven by a Carrington synoptic map is Carrington-aligned,
so `--longitude 317` means Carrington longitude 317. A run driven by a Stonyhurst map is not, and
the same flag points the camera somewhere else. Choosing the angles starts with knowing which frame
the input was built in.

## Earth's viewpoint

For a Carrington-aligned solution, `--observer earth` points the camera the way Earth saw the Sun
at a given time, taking both angles from Earth's ephemeris:

```bash
qorona render data/coconut_corona.qor -o data/eclipse.png --fov 8 \
    --timestamp 2025-10-09T18:19:52 --observer earth
```

It needs an epoch, either `--timestamp` on the command or one baked into the volume, and it sets
the pointing itself, so `--longitude` and `--latitude` are refused alongside it.

To get the same two angles yourself, ahead of a run or to correct them for a frame that is not
Carrington-aligned, print them with sunpy, which comes with the default Qorona install:

```python
from sunpy.coordinates import sun

t = "2025-10-09T18:19:52"  # an ISO-8601 UTC string, or a compact "20251009181952"
print(sun.L0(t).deg, sun.B0(t).deg)  # 316.29 6.24
```

`L0` is the Carrington longitude of the point facing Earth and `B0` its heliographic latitude, so
the pair goes straight into `--longitude` and `--latitude`. In a Stonyhurst-aligned frame the
sub-Earth longitude is 0 by definition, so only the latitude is needed.

Registering a Qorona image against observed imagery is covered in
[Export to JHelioviewer](jhelioviewer.md).
