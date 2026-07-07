# The Q⊥ volume

The pipeline splits at its natural cost seam. Building the Q⊥ volume means tracing field
lines everywhere: minutes-scale, and viewpoint-independent. A render is a line-of-sight
integral through that volume: seconds-scale, and different for every camera. So `build` runs
once, and `render` and `qmap` reuse the result any number of times.

## The .qor cache

`build` writes a dependency-free `.qor` (a float32 `.npz` under the hood) carrying the volume
together with its full build provenance: input hash, derived Carrington rotation and Julian
date, and every resolved parameter, including the compute backend and precision that produced
it. The same details are printed in the end-of-run summary.

## Quality

`--quality` picks the built volume's resolution:

| Preset | Volume grid | Voxels | File size |
|----------------------|-----------------|--------|-----------|
| `fast` | 192x180x360 | ~12 M | ~45 MB |
| `standard` (default) | 384x360x720 | ~100 M | ~0.4 GB |
| `high` | 576x540x1080 | ~336 M | ~1.1 GB |

`fast` is a quick preview, `high` resolves the finest structure. Explicit
`--resolution-factor` / `--supersample` override the preset; indicative build times are in
[GPU acceleration](gpu.md).

## Build flags

The build-time knobs, beyond `--quality`:

- `--outer-radius`: how far out the volume extends, in solar radii (default 12.5). Match it to
  the product: 8 is comfortable for a whole-corona render at `--fov 8`, while a
  [Q-map](products/qmaps.md) wants the outer radius equal to the map radius.
- `--closed neutral|dominant`: closed-loop polarity treatment (default `neutral`), read back
  by the [polarity view](products/polarity.md).
- `--device auto|gpu|cpu` and `--precision`: the compute backend, covered in
  [GPU acceleration](gpu.md).
- `--workers` and `--quiet`: thread count and quiet output, made for
  [batch jobs](getting-started/hpc.md).

`qorona build --help-all` is the complete reference (field grid, resampler, tracer tolerances,
and more).
