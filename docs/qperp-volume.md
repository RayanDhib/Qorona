# The Q⊥ volume

Building the Q⊥ volume means tracing field lines everywhere: minutes-scale, and
viewpoint-independent. A render is a line-of-sight integral through that volume: seconds-scale,
and different for every camera. So `build` runs once, and `render` and `qmap` reuse the result
any number of times.

```bash
qorona build data/coconut_corona.CFmesh.xz -o data/coconut_corona.qor \
    --timestamp 2025-10-09T18:19:52 --frame carrington --outer-radius 8
```

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
- `--timestamp`, `--frame`: the observation time (UTC ISO-8601) and the heliographic frame of
  the solution's longitudes (`carrington` or `stonyhurst`; `heeq` is accepted for Stonyhurst).
  Neither is in the file; both are baked into the volume and inherited by every product built
  from it, and together they let the camera take a named observer's viewpoint (see
  [Camera and viewpoint](camera.md)).
- `--closed neutral|dominant`: polarity assigned to closed loops (default `neutral`), for the
  [polarity view](products/polarity.md).
- `--device auto|gpu|cpu` and `--precision`: the compute backend, covered in
  [GPU acceleration](gpu.md).
- `--workers` and `--quiet`: thread count and quiet output, made for
  [batch jobs](getting-started/hpc.md).

`qorona build --help-all` is the complete reference (field grid, resampler, tracer tolerances,
and more).

## The .qor cache

`build` writes a dependency-free `.qor` (a float32 `.npz` under the hood) carrying the volume
together with its full build provenance: input hash, derived Carrington rotation and Julian
date, and every resolved parameter, including the compute backend and precision that produced
it. The same details are printed in the end-of-run summary.
