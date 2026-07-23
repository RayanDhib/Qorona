# Export to JHelioviewer

Qorona output drops into [JHelioviewer](https://www.jhelioviewer.org) for viewing and overlay
on observed imagery, through two registered routes: WCS-registered FITS for the image
products, and SunJSON geometry for field lines. Both need a timestamp (`--timestamp`, or a
volume baked with one): it sets the observation time the file is registered at.

| Route | Commands | JHelioviewer layer |
|-------|----------|--------------------|
| FITS raster | `render --export fits`, `wl --export fits` | image layer, WCS-registered |
| SunJSON field lines | `fieldlines --export sunjson`, `export-lines` | Connection Layer, 3-D lines |

## FITS rasters

`--export fits` writes the quantitative frame beside the PNG: `render` stores the
LOS-averaged log₁₀ Q⊥ (float32), `wl` the display frame plus the raw pB/total frames as
extensions. Drop the file into JHelioviewer and it registers at the correct scale,
orientation, position, and time; `sunpy.map.Map` reads the same registration for scripted
analysis. The run's full provenance rides in the FITS header.

- `--observer earth` points the camera from the real Earth viewpoint at the timestamp, so
  the overlay matches Earth-based observations exactly.
- The disk: with the default eclipse occulter it stays NaN, transparent in JHelioviewer, so
  real disk imagery shows through; `render --occult composite` fills it with the near-side
  surface Q instead.

## Field lines as SunJSON

SunJSON is the geometry format of JHelioviewer's Connection Layer: drop the `.json` onto a
running JHelioviewer and the bundle appears in 3-D, at its timestamp on the timeline. Lines
carry the polarity palette of the [field-line view](products/fieldlines.md) (open warm/cool
by inner-foot B_r sign, closed neutral grey) and an `open`/`closed` topology tag.

- `qorona fieldlines ... --export sunjson`: exactly the drawn bundle beside the PNG, with
  the same seeding, selection (`--show`), and colours as the image, the eclipse look
  included.
- `qorona export-lines`: a self-contained full trace on a uniform longitude/latitude seed
  grid (`--seeds N_THETA N_PHI`, default 100x100; `--seed-radius` moves the seed sphere);
  no camera involved.

```bash
qorona export-lines data/coconut_corona.CFmesh.xz -o data/coconut_fieldlines.json \
    --timestamp 2025-10-09T18:19:52 --seeds 40 80
```

## The frame assumption

Both routes write positions in the solution's own frame and assume it is Carrington-aligned:
true for synoptic-driven runs (COCONUT, MAS), not guaranteed in general. It underlies the
`--observer earth` registration and the SunJSON longitudes alike.
