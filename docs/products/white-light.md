# White-light imaging

Thomson-scattered brightness integrated over the solution's electron density. The default
frame is pB, the polarized brightness: the linearly polarized component of the white-light
K-corona, and the classic coronagraph observable. `--frame total` selects the total brightness
instead. This is a density product; no Q⊥ volume is built or used.

The input is either a raw solution (read and resampled, as `build` would) or a built `.qor`
volume, whose stored density is reused, skipping the resample:

![Polarized-brightness view of the COCONUT corona](../assets/white-light.png)

```bash
qorona wl data/coconut_corona.qor -o docs/assets/white-light.png \
    --longitude 317 --latitude 6.2 --width 1024 --height 1024
```

## The flags that matter

- `--frame polarized|total`: pB (the default) or the total white-light brightness.
- `--vignette newkirk|adaptive|none`: radial detrend of the frame. `newkirk` divides by the
  brightness of the smooth Newkirk background corona; `adaptive` self-calibrates the same
  curve family to the image's own falloff; `none` keeps the raw falloff. Default `newkirk`
  (`adaptive` for inputs whose radial falloff departs from it).
- `--percentiles LOW HIGH`: display stretch (default `1 99.5`; use `0 100` for the full
  untrimmed range).
- `--mgn`: optional fine-structure enhancement (multi-scale Gaussian normalization), applied
  last; needs sunkit-image, which is not part of the default install.
- `--occult eclipse|none`: the occulter (default `eclipse`).
- `--r-occult`: occulter radius in solar radii (default 1.02).
- `--export npz`: also write the raw frames (both pB and total, with plane-of-sky
  coordinates) beside the PNG.
- `--width`, `--height`: image size in pixels (default 512, smaller than `render`'s 1024).
- Camera flags are the same as the [squashing-factor render](squashing-factor.md); `wl` shares
  its `--fov 8` default.
