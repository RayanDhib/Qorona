# GPU acceleration

Volume builds and renders are CUDA-accelerated end to end, with no extra Qorona dependency: the
kernels use the numba in the default install and activate whenever numba sees a CUDA-capable
NVIDIA GPU (driver plus CUDA toolkit). Nothing to configure: `qorona build` and `qorona render`
already use the GPU when one is present.

```bash
qorona build ... --device gpu                       # force the GPU (errors if none is usable)
qorona build ... --device cpu                       # multi-core CPU path (the reference)
qorona build ... --device gpu --precision float64   # all-double reference (~2x slower than mixed)
qorona render ... --device cpu                      # force the CPU render path
```

- `--device auto` (default) picks the GPU when present; the CPU kernels remain the reference
  implementation and produce the same images.
- `--precision mixed` (default) runs the field interpolation in float32 and everything else
  in float64, indistinguishable from the `float64` reference in the log-scaled image. `float32`
  is an experimental fully-float32 paint variant. GPU-only knob; the CPU path always computes
  in float64.
- The render has the same two flags: its `--precision mixed` samples the volume in float32 and
  accumulates in float64 (`float64` is the all-double reference;
  [`validation/render_parity.py`](https://github.com/RayanDhib/Qorona/blob/main/validation/render_parity.py)
  characterizes the agreement).
- Device memory adapts to free VRAM (the Q⊥ accumulation tiles itself), so the same command
  runs on small cards and at very high resolutions alike.
- The resolved backend and precision are stamped into the volume's provenance and the
  end-of-run summary.

Indicative volume-build timings (RTX 4080 vs 32-core CPU, mixed precision; not a benchmark):

| Q⊥ volume                                                |    GPU |     CPU |
|----------------------------------------------------------|-------:|--------:|
| `--quality standard` (default), 384x360x720 (100 M vox)  |  ~85 s |  ~9 min |
| `--quality high`, 576x540x1080 (336 M vox)               | ~3 min | ~22 min |

Indicative render timings off the high-quality volume (same hardware, LOS integration only):

| Render (1024x1024)                   |   GPU |    CPU |
|--------------------------------------|------:|-------:|
| default (`--step 0.02`)              |  ~2 s |   ~9 s |
| near-limb fine step (`--step 0.002`) |  ~9 s |  ~78 s |

On a cluster, these timings set your SLURM `--time`; see [HPC clusters](getting-started/hpc.md).
