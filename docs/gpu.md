# GPU acceleration

`build` and `render` use an NVIDIA GPU when numba finds one (a driver and the CUDA toolkit). You do
not need other packages or settings.

```bash
qorona build ... --device gpu                       # use the GPU (error if none is usable)
qorona build ... --device cpu                       # use the CPU kernels (the reference)
qorona build ... --device gpu --precision float64   # all double precision (~2x slower than mixed)
qorona render ... --device cpu                      # render on the CPU
```

| Flag | Values | Default |
|------|--------|---------|
| `--device` | `auto` uses the GPU when there is one. `gpu` stops with an error if no GPU is usable. `cpu` uses the CPU kernels. | `auto` |
| `--precision` | GPU only. `mixed`: float32 for the field interpolation (`build`) or the volume samples (`render`), float64 for all else. `float64`: all double precision. `float32` (`build` only): experimental, all single precision. | `mixed` |

- The CPU kernels are the reference. The GPU gives the same images.
- `mixed` and `float64` give images that look the same on the log scale.
  [`validation/render_parity.py`](https://github.com/RayanDhib/Qorona/blob/main/validation/render_parity.py)
  measures the agreement of the render.
- The GPU memory use adapts to the free memory, so the same command runs on small cards and at high
  resolution.
- The volume provenance and the run summary record the backend and the precision.

Volume build times (RTX 4080 or 32-core CPU, mixed precision; examples only, not a benchmark):

| Q⊥ volume                                                |    GPU |     CPU |
|----------------------------------------------------------|-------:|--------:|
| `--quality standard` (default), 384x360x720 (100 M vox)  |  ~85 s |  ~9 min |
| `--quality high`, 576x540x1080 (336 M vox)               | ~3 min | ~22 min |

Render times on the high-quality volume (same hardware, line-of-sight integration only):

| Render (1024x1024)                   |   GPU |    CPU |
|--------------------------------------|------:|-------:|
| default (`--step 0.02`)              |  ~2 s |   ~9 s |
| near-limb fine step (`--step 0.002`) |  ~9 s |  ~78 s |

Use these times to set the `--time` of a [SLURM job](getting-started/hpc.md).
