# Solution time and frame

A COCONUT or MAS file does not record its time or the orientation of its longitudes. Give them with
two flags:

| Flag | Value | Example |
|------|-------|---------|
| `--timestamp` | The solution time, UTC ISO-8601. | `2025-10-09T18:19:52` |
| `--frame` | The heliographic frame of the solution longitudes: `carrington` or `stonyhurst` (`heeq` is accepted for `stonyhurst`). | `carrington` |

Give them to each command that reads the solution file: `info`, `build`, `run`, `wl`, `fieldlines`
or `export-lines`.

```bash
qorona build data/coconut_corona.CFmesh.xz -o data/coconut_corona.qor \
    --timestamp 2025-10-09T18:19:52 --frame carrington --outer-radius 8
```

A `.qor` volume keeps them, and each render of the volume uses them. `render` and `wl` accept the
same flags for a volume: they set the values for a volume built without them, or replace the values
of the volume for this run only. The `.qor` file does not change. The run summary shows the frame in use.

## What needs them

| Feature | Without `--timestamp` | Without `--frame` |
|---------|-----------------------|-------------------|
| Carrington rotation and Julian date in the stamp and the summary | Left out. | No effect. |
| [`--observer`](camera.md#use-a-real-observer) (also `wl --punch-l3`) | The run stops with a message. | Carrington assumed, with a warning. |
| [`--export fits`](jhelioviewer.md#fits-images) | The run stops with a message. | Carrington assumed, with a warning. |
| [`fieldlines --export sunjson`, `export-lines`](jhelioviewer.md#field-lines-as-sunjson) | The run stops with a message. | Carrington assumed, with a warning. |

- The warning prints only when one of these features runs. The summary then shows
  `carrington frame (assumed)`.
- If you do not know the frame, do not use these features. Point the camera with `--longitude` and
  `--latitude`.

## Carrington or Stonyhurst

| `--frame` | Use it for a run driven by | Longitude 0 |
|-----------|----------------------------|-------------|
| `carrington` | a Carrington synoptic map | Carrington longitude 0 |
| `stonyhurst` | a magnetogram rotated to the Earth-facing meridian (Heliocentric Earth Equatorial, HEEQ) | faces Earth at every time |

A wrong frame moves the solution by up to 360 degrees in longitude. Nothing in the image or the
exported file shows the error.

## Point at Earth by hand

`--observer earth` does this for you. To get the angles yourself, print L0 and B0 with sunpy, which
comes with Qorona. L0 is the Carrington longitude of the point that faces Earth, and B0 is its
latitude.

```python
from sunpy.coordinates import sun

t = "2025-10-09T18:19:52"  # an ISO-8601 UTC string, or a compact "20251009181952"
print(sun.L0(t).deg, sun.B0(t).deg)  # 316.29 6.24
```


- In a Carrington frame, use L0 as `--longitude` and B0 as `--latitude`.
- In a Stonyhurst frame, Earth is at longitude 0. Use B0 as `--latitude`.

## Icarus snapshots

An Icarus snapshot is in its own frame. Give its run settings instead of `--timestamp` and
`--frame`. See [Icarus run settings](heliospheric.md#icarus-run-settings).
