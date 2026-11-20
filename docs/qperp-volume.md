# The Q⊥ volume

`build` traces the field lines of the solution and stores Q⊥ in a volume. The volume does not depend
on the viewpoint, so `render` and `qmap` use it as many times as you want. A build takes minutes; a
render takes seconds.

```bash
qorona build data/coconut_corona.CFmesh.xz -o data/coconut_corona.qor \
    --timestamp 2025-10-09T18:19:52 --frame carrington --outer-radius 8
```

## Quality

`--quality` sets the resolution of the volume:

| Preset | Volume grid | Voxels | File size |
|----------------------|-----------------|--------|-----------|
| `fast` | 192x180x360 | ~12 M | ~45 MB |
| `standard` (default) | 384x360x720 | ~100 M | ~0.4 GB |
| `high` | 576x540x1080 | ~336 M | ~1.1 GB |

- `fast` gives a quick preview. `high` resolves the finest structure.
- `--resolution-factor` and `--supersample` override the preset.
- [GPU acceleration](gpu.md) gives build times.

## Build flags

| Flag | Use | Default |
|------|-----|---------|
| `--quality` | The resolution preset. See [Quality](#quality). | `standard` |
| `--outer-radius` | How far out the volume goes, in solar radii. Use 8 for a whole-corona render at `--fov 8`. For a [Q-map](products/qmaps.md), use the map radius. | 12.5 |
| `--timestamp`, `--frame` | The solution time and frame. See [Solution time and frame](time-and-frame.md). | none |
| `--closed` | The polarity of closed loops, for the [polarity view](products/polarity.md). | `neutral` |
| `--device`, `--precision` | The compute backend. See [GPU acceleration](gpu.md). | `auto`, `mixed` |
| `--workers`, `--quiet` | The number of threads, and output without progress bars, for [batch jobs](getting-started/hpc.md). | all cores, off |

`qorona build --help-all` lists all the flags: field grid, resampler, tracer tolerances and more.

## The .qor file

`build` writes a `.qor` file: a float32 `.npz` that needs no other package. It holds the volume and
the build provenance:

- the hash of the input file;
- the Carrington rotation and the Julian date;
- each resolved parameter, with the compute backend and the precision.

The run summary shows the same values.
