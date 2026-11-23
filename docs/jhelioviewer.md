# Export to JHelioviewer

Qorona writes two file types that [JHelioviewer](https://www.jhelioviewer.org) places at the
correct time and position:

| Export | Commands | JHelioviewer layer |
|--------|----------|--------------------|
| FITS image | `render --export fits`, `wl --export fits`, `euv --export fits` | Image layer |
| SunJSON field lines | `fieldlines --export sunjson`, `export-lines` | Connection Layer, 3-D lines |

Both need the [solution time](time-and-frame.md). Without `--frame`, Qorona assumes a Carrington
frame and warns.

## FITS images

```bash
qorona render data/coconut_corona.qor -o data/earth.png --fov 8 --observer earth --export fits
```

`--export fits` writes a FITS file beside the PNG:

- `render`: the line-of-sight average of log₁₀ Q⊥ (float32).
- `wl`: the display frame, and the raw pB and total frames as extensions.
- `euv`: the frame in the unit of the channel (float32). See
  [EUV imaging](products/euv.md#exported-values).

The FITS file registers the image in helioprojective coordinates (Thompson 2006), with the observer
distance of the camera. JHelioviewer shows it at the correct scale, orientation, position and time.
`sunpy.map.Map` reads the same registration. The FITS header holds the full provenance of the run.

- `--observer earth` puts the camera at Earth at the solution time. The image then overlays
  observations from Earth. All [named observers](camera.md#use-a-real-observer) work the same way.
- With the default `eclipse` occulter, the disk is NaN, which is transparent in JHelioviewer. Real
  disk images show through. `render --occult composite` fills the disk with the near-side surface
  Q⊥.

## Field lines as SunJSON

```bash
qorona export-lines data/coconut_corona.CFmesh.xz -o data/coconut_fieldlines.json \
    --timestamp 2025-10-09T18:19:52 --frame carrington --seeds 40 80
```

SunJSON is the geometry format of the JHelioviewer Connection Layer. Drop the `.json` file onto
JHelioviewer: the lines show in 3-D, at their time on the timeline.

- The lines use the colours of the [field-line view](products/fieldlines.md): open lines warm or
  cool by the sign of B_r at the inner foot, closed lines grey.
- Each line has an `open` or `closed` tag.

| Command | Lines |
|---------|-------|
| `fieldlines ... --export sunjson` | The lines of the image, with the same seeds, selection (`--show`) and colours. |
| `export-lines` | A full trace from a uniform longitude-latitude grid of seeds, with no camera. `--seeds N_THETA N_PHI` sets the grid (default 100 x 100). `--seed-radius` moves the seed sphere. |

## Positions in Carrington coordinates

Both exports write positions in Carrington coordinates, as JHelioviewer expects. Qorona converts them
from the declared frame of the solution. A wrong `--frame` moves the export by up to 360 degrees in
longitude. See [Solution time and frame](time-and-frame.md).
